from __future__ import annotations

import asyncio
import math
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar, Token
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple, TypeVar, cast, overload

from fastapi import HTTPException
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import (
    DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    LAST_GOOD_AUTH_ENTRIES_IN_MEMORY,
    LAST_GOOD_ENTRY_TTL_SECONDS,
    LAST_GOOD_SERVED_COPY_TTL_SECONDS,
    LAST_GOOD_TOMBSTONE_TTL_SECONDS,
)
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.repositories.chunked_in import IN_LIST_CHUNK_SIZE

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.caching.redis_batch import BatchResult

T = TypeVar("T", bound=BaseModel)
_Written: Final = TypeVar("_Written")
_Row: Final = TypeVar("_Row")

AUTH_OBJECTS_TARGET: Final = "auth_objects"

_HASHED_TOKEN_CACHE_KEY: Final = re.compile(r"[0-9a-f]{64}")


def is_user_key_cache_key(key: str) -> bool:
    """Only user-key objects are cached under a bare ``hash_token`` digest; every other object uses a prefixed key."""
    return _HASHED_TOKEN_CACHE_KEY.fullmatch(key) is not None


_PIPELINED_SET_OPTIONS: Final = frozenset(("ttl",))


class _KeptEntry(NamedTuple):
    payload: object
    ends_at: float


class LastGoodMemory:
    """A pod's own copy of the last good entries, each until the end its store gave it, the moment its Redis entry
    ends too: bounded, and the least recently written entry goes first. Holds the JSON payload, so every read builds
    a fresh object. A tombstone, written when an entry is revoked, refuses every write of that entry until it lapses.
    Entries and tombstones lapse on the same clock."""

    __slots__ = ("_clock", "_entries", "_max_entries", "_tombstones")

    def __init__(self, max_entries: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._entries: Final[OrderedDict[str, _KeptEntry]] = OrderedDict()  # mutable-ok: bounded LRU, writes reorder
        self._tombstones: Final[OrderedDict[str, float]] = OrderedDict()  # mutable-ok: bounded, lapses by clock
        self._max_entries: Final = max_entries
        self._clock: Final = clock

    def set(self, key: str, payload: object, ends_at: float) -> bool:
        if self.tombstoned(key):
            return False
        _ = self._entries.pop(key, None)
        self._entries[key] = _KeptEntry(payload=payload, ends_at=ends_at)
        while len(self._entries) > self._max_entries:
            _ = self._entries.popitem(last=False)
        return True

    def get(self, key: str) -> object:
        kept: Final = self._entries.get(key)
        if kept is None:
            return None
        if kept.ends_at > self._clock():
            return kept.payload
        _ = self._entries.pop(key, None)
        return None

    def delete(self, key: str) -> None:
        _ = self._entries.pop(key, None)

    def clear(self) -> None:
        self._entries.clear()

    def tombstone(self, key: str, ttl_seconds: float) -> None:
        self.delete(key)
        _ = self._tombstones.pop(key, None)
        self._tombstones[key] = self._clock() + ttl_seconds
        while len(self._tombstones) > self._max_entries:
            _ = self._tombstones.popitem(last=False)

    def tombstoned(self, key: str) -> bool:
        until: Final = self._tombstones.get(key)
        if until is None:
            return False
        if until > self._clock():
            return True
        _ = self._tombstones.pop(key, None)
        return False


class UserApiKeyCache(DualCache):
    """
    DualCache wrapper for UserAPIKeyAuth-like payloads.

    Stores a Redis-safe JSON payload in BOTH in-memory and Redis to avoid
    "memory returns BaseModel, Redis returns dict" format drift.

    When ``model_type`` is provided:
    - writes are serialized via ``CacheCodec.serialize(..., model_type=...)``
    - reads are deserialized via ``CacheCodec.deserialize(..., model_type)``
      and return ``Optional[T]``: the model on success, ``None`` on cache miss
      **or** if the cached payload fails validation (schema drift). On
      validation failure after a cache hit, an error line is emitted via
      ``verbose_proxy_logger``.

    When ``model_type`` is omitted, the interface behaves like ``DualCache``:
    raw cached payload is returned (dict/str/etc.).

    ``async_set_cache_pipeline`` applies the same untyped Codec pass as omitting
    ``model_type`` on ``async_set_cache`` (so ``BaseModel`` rows are dumped before Redis).

    User-key objects (see ``is_user_key_cache_key``) live in their own in-memory partition,
    ``key_object_cache``, so churn in the other management objects cannot evict them. Both
    partitions share the same Redis backend and TTL settings.

    ``get_cache`` / ``async_get_cache`` overloads and implementations must be contiguous
    (no other methods in between) so mypy resolves ``@overload`` + implementation correctly.
    """

    def __init__(
        self,
        in_memory_cache: InMemoryCache | None = None,
        redis_cache: RedisCache | None = None,
        default_in_memory_ttl: float | None = None,
        default_redis_ttl: float | None = None,
        key_object_in_memory_cache: InMemoryCache | None = None,
        serve_last_good_entries: bool = False,
        last_good_clock: Callable[[], float] = time.monotonic,
        last_good_wall_clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(
            in_memory_cache=in_memory_cache,
            redis_cache=redis_cache,
            default_in_memory_ttl=default_in_memory_ttl,
            default_redis_ttl=default_redis_ttl,
        )
        self.key_object_cache: Final = DualCache(
            in_memory_cache=key_object_in_memory_cache or InMemoryCache(),
            redis_cache=redis_cache,
            default_in_memory_ttl=default_in_memory_ttl,
            default_redis_ttl=default_redis_ttl,
        )
        # One clock for the memory line's ends and tombstones and for the end each store takes, so a test drives all
        self.last_good_clock: Final = last_good_clock
        # Read together with last_good_clock once per store, to turn the entry's end into the Redis EXAT deadline
        self.last_good_wall_clock: Final = last_good_wall_clock
        self.last_good_memory: Final = LastGoodMemory(
            max_entries=LAST_GOOD_AUTH_ENTRIES_IN_MEMORY, clock=last_good_clock
        )
        # The end of each copy keep_last_good_served_copy keeps, on last_good_clock, since memory keeps its own expiry
        # on its own clock: a value derived from a copy is cached no longer than the copy lives
        self.last_good_copy_ends: Final[OrderedDict[str, float]] = OrderedDict()  # mutable-ok: bounded, writes reorder
        self.serve_last_good_entries = serve_last_good_entries

    def in_memory_cache_for(self, key: str) -> InMemoryCache:
        return self.key_object_cache.in_memory_cache if is_user_key_cache_key(key) else self.in_memory_cache

    def keep_served_copy_end(self, cache_key: str, ends_at: float) -> None:
        """Record when this pod's memory copy of a value served from a last good entry ends, keeping the newest
        ``LAST_GOOD_AUTH_ENTRIES_IN_MEMORY`` of them."""
        self.last_good_copy_ends[cache_key] = ends_at
        while len(self.last_good_copy_ends) > LAST_GOOD_AUTH_ENTRIES_IN_MEMORY:
            _ = self.last_good_copy_ends.popitem(last=False)

    def update_cache_ttl(self, default_in_memory_ttl: float | None, default_redis_ttl: float | None) -> None:
        super().update_cache_ttl(default_in_memory_ttl=default_in_memory_ttl, default_redis_ttl=default_redis_ttl)
        self.key_object_cache.update_cache_ttl(
            default_in_memory_ttl=default_in_memory_ttl, default_redis_ttl=default_redis_ttl
        )

    def update_in_memory_max_size(self, max_size: int | None) -> None:
        super().update_in_memory_max_size(max_size)
        self.key_object_cache.update_in_memory_max_size(max_size)

    def attach_redis_cache(
        self, redis_cache: RedisCache | None = None, *, default_redis_ttl: float | None = None
    ) -> None:
        super().attach_redis_cache(redis_cache, default_redis_ttl=default_redis_ttl)
        self.key_object_cache.attach_redis_cache(redis_cache, default_redis_ttl=default_redis_ttl)

    @overload
    def get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        *,
        model_type: type[T],
        **kwargs: object,
    ) -> T | None: ...

    @overload
    def get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        model_type: None = None,
        **kwargs: object,
    ) -> Any: ...

    def get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        model_type: type[BaseModel] | None = None,
        **kwargs: object,
    ) -> object:
        if model_type is None and "model_type" in kwargs:
            model_type = cast(type[BaseModel] | None, kwargs.pop("model_type", None))
        cached: Final = (
            self.key_object_cache.get_cache(key=key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs)
            if is_user_key_cache_key(key)
            else super().get_cache(key=key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs)
        )
        if model_type is None:
            return cached
        if cached is None:
            return None
        decoded: Final = CacheCodec.deserialize(cached, model_type=model_type)
        if decoded is None:
            verbose_proxy_logger.error(
                "UserApiKeyCache.get_cache failed to deserialize cached value for key=%r model_type=%s",
                key,
                getattr(model_type, "__name__", str(model_type)),
            )
            return None
        return decoded

    @overload
    async def async_get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        *,
        model_type: type[T],
        **kwargs: object,
    ) -> T | None: ...

    @overload
    async def async_get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        model_type: None = None,
        **kwargs: object,
    ) -> Any: ...

    async def async_get_cache(
        self,
        key: str,
        parent_otel_span: Span | None = None,
        local_only: bool = False,
        model_type: type[BaseModel] | None = None,
        **kwargs: object,
    ) -> object:
        if model_type is None and "model_type" in kwargs:
            model_type = cast(type[BaseModel] | None, kwargs.pop("model_type", None))
        cached: Final = (
            await self.key_object_cache.async_get_cache(
                key=key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs
            )
            if is_user_key_cache_key(key)
            else await super().async_get_cache(
                key=key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs
            )
        )
        if model_type is None:
            return cached
        if cached is None:
            return None
        decoded: Final = CacheCodec.deserialize(cached, model_type=model_type)
        if decoded is None:
            verbose_proxy_logger.error(
                "UserApiKeyCache.async_get_cache failed to deserialize cached value for key=%r model_type=%s",
                key,
                getattr(model_type, "__name__", str(model_type)),
            )
            return None
        return decoded

    def set_cache(self, key: str | None, value: object, local_only: bool = False, **kwargs: object):
        model_type: Final = cast(type[BaseModel] | None, kwargs.pop("model_type", None))
        payload: Final[object] = CacheCodec.serialize(value, model_type=model_type)
        if key is not None and is_user_key_cache_key(key):
            return self.key_object_cache.set_cache(key=key, value=payload, local_only=local_only, **kwargs)
        return super().set_cache(key=key, value=payload, local_only=local_only, **kwargs)

    async def async_set_cache(self, key: str | None, value: object, local_only: bool = False, **kwargs: object):
        """Inside a request the Redis SET rides the request's pipeline (memory is written at once); anywhere
        else, or with options the pipeline does not carry, it goes to Redis directly as before."""
        model_type: Final = cast(type[BaseModel] | None, kwargs.pop("model_type", None))
        payload: Final[object] = CacheCodec.serialize(value, model_type=model_type)
        ttl: Final = kwargs.get("ttl")
        pipelined: Final = (
            key is not None
            and not local_only
            and kwargs.keys() <= _PIPELINED_SET_OPTIONS
            and (ttl is None or isinstance(ttl, (int, float)))
        )
        if key is not None and is_user_key_cache_key(key):
            if pipelined and await self.key_object_cache.async_set_cache_pre_call(key, payload, ttl) is not None:
                return None
            return await self.key_object_cache.async_set_cache(key=key, value=payload, local_only=local_only, **kwargs)
        if pipelined and await super().async_set_cache_pre_call(key, payload, ttl) is not None:
            return None
        return await super().async_set_cache(key=key, value=payload, local_only=local_only, **kwargs)

    def delete_cache(self, key: str) -> None:
        if is_user_key_cache_key(key):
            self.key_object_cache.delete_cache(key)
            return
        super().delete_cache(key)

    async def async_delete_cache(self, key: str) -> None:
        if is_user_key_cache_key(key):
            await self.key_object_cache.async_delete_cache(key)
            return
        await super().async_delete_cache(key)

    async def async_delete_cache_pre_call(self, key: str) -> BatchResult[None] | None:
        if is_user_key_cache_key(key):
            return await self.key_object_cache.async_delete_cache_pre_call(key)
        return await super().async_delete_cache_pre_call(key)

    async def async_delete_cache_keys(self, keys: Sequence[str]) -> None:
        """Batch twin of ``async_delete_cache``, partitioned like
        ``async_set_cache_pipeline``.

        Both partitions are cleared even when one raises, because a caller
        batching these has already committed the rows they cache.
        """
        key_object_keys: Final = tuple(key for key in keys if is_user_key_cache_key(key))
        other_keys: Final = tuple(key for key in keys if not is_user_key_cache_key(key))
        outcomes: Final = await asyncio.gather(
            self.key_object_cache.async_delete_cache_keys(key_object_keys),
            super().async_delete_cache_keys(other_keys),
            return_exceptions=True,
        )
        failed: Final = tuple(outcome for outcome in outcomes if isinstance(outcome, BaseException))
        if failed:
            raise failed[0]

    def flush_cache(self) -> None:
        super().flush_cache()
        self.key_object_cache.in_memory_cache.flush_cache()

    async def async_set_cache_pipeline(
        self, cache_list: Sequence[tuple[str, object]], local_only: bool = False, **kwargs: object
    ) -> None:
        """
        Batch writes with the same Codec boundary as ``async_set_cache`` without
        ``model_type``: ``BaseModel`` values become JSON-safe dicts; dicts/scalars unchanged.
        """
        normalized: Final = tuple((key, CacheCodec.serialize(value, model_type=None)) for key, value in cache_list)
        key_object_entries: Final = tuple(entry for entry in normalized if is_user_key_cache_key(entry[0]))
        other_entries: Final = tuple(entry for entry in normalized if not is_user_key_cache_key(entry[0]))
        if key_object_entries:
            await self.key_object_cache.async_set_cache_pipeline(
                cache_list=key_object_entries, local_only=local_only, **kwargs
            )
        if other_entries:
            await super().async_set_cache_pipeline(cache_list=other_entries, local_only=local_only, **kwargs)


#: Value cached under ``user_object_permission_id_cache_key`` when the user links no permission row,
#: so a human without an entitlement costs no DB read per request. Lives beside the key builder
#: because it is part of the same cache protocol: a reader that knows the key must know this value.
USER_NO_MCP_PERMISSION_SENTINEL: Final = "__user_no_mcp_permission__"


def user_object_permission_id_cache_key(user_id: str) -> str:
    """Cache key for the ``user_id -> object_permission_id`` link.

    Lives here rather than next to either user because two modules own the two halves: the MCP auth
    resolver writes it on read, and ``/user/update`` deletes it after changing the link. A key format
    duplicated across those two drifts silently, and the failure is an entitlement change that never
    takes effect.
    """
    return f"user_object_permission_id:{user_id}"


def object_permission_cache_key(object_permission_id: str) -> str:
    """Cache key ``get_object_permission`` stores a permission row under."""
    return f"object_permission_id:{object_permission_id}"


#: Cached under ``tag_registry_cache_key`` when the table exceeds ``TAG_REGISTRY_MAX_SIZE``:
#: registry unusable, fall back to the per-tag lookup.
TAG_REGISTRY_OVERFLOW_SENTINEL: Final = "__tag_registry_overflow__"


def tag_cache_key(tag_name: str) -> str:
    """Cache key one tag row is stored under; shared so its five reader/writer modules cannot drift."""
    return f"tag:{tag_name}"


def tag_registry_cache_key() -> str:
    """Cache key for the set of tag names that exist in ``LiteLLM_TagTable``."""
    return "tag_registry"


#: Cached under ``model_access_group_registry_cache_key`` when the table exceeds
#: ``MODEL_ACCESS_GROUP_REGISTRY_MAX_SIZE``: registry unusable, fall back to the per-group lookup.
MODEL_ACCESS_GROUP_REGISTRY_OVERFLOW_SENTINEL: Final = "__model_access_group_registry_overflow__"


def model_access_group_cache_key(access_group_name: str) -> str:
    """Cache key one model access group budget row is stored under; shared so auth, spend tracking and the management endpoints cannot drift."""
    return f"model_access_group:{access_group_name}"


def model_access_group_registry_cache_key() -> str:
    """Cache key for the set of model access group names that have a budget row."""
    return "model_access_group_registry"


def model_access_group_spend_counter_key(access_group_name: str) -> str:
    """Spend counter key for one model access group; shared so its four owners cannot drift.

    The reservation path writes it up front, the cost callback writes it after the call, auth
    reads it to enforce ``max_budget``, and the reset job clears it on rollover. A copy that
    drifts in any one of them silently resets or reads a counter nobody else touches, which shows
    up as a budget that never trips or never resets.
    """
    return f"spend:model_access_group:{access_group_name}"


def project_cache_key(project_id: str) -> str:
    return f"project_id:{project_id}"


def project_spend_counter_key(project_id: str) -> str:
    return f"spend:project:{project_id}"


#: Cached under ``end_user_restricted_registry_cache_key`` when the restricted set exceeds
#: ``END_USER_RESTRICTED_REGISTRY_MAX_SIZE``: registry unusable, fall back to the per-id fetch.
END_USER_RESTRICTED_REGISTRY_OVERFLOW_SENTINEL: Final = "__end_user_restricted_registry_overflow__"


def end_user_cache_key(end_user_id: str) -> str:
    """Cache key one end-user row is stored under; shared so auth and spend tracking cannot drift."""
    return f"end_user_id:{end_user_id}"


def end_user_validation_cache_key(end_user_id: str) -> str:
    """Cache key of the verdict on whether an end-user id exists, which end-user id validation keeps; every customer
    mutation evicts it, so a verdict from before the customer was created cannot drop it until the verdict lapses."""
    return f"end_user_validation:{end_user_id}"


def end_user_restricted_registry_cache_key() -> str:
    """Cache key for the set of end-user ids whose row carries a restriction auth enforces."""
    return "end_user_restricted_registry"


def default_end_user_budget_cache_key(budget_id: str) -> str:
    return f"default_end_user_budget:{budget_id}"


def team_member_default_budget_cache_key(budget_id: str) -> str:
    return f"team_member_default_budget:{budget_id}"


def team_membership_auth_cache_key(team_id: str, user_id: str) -> str:
    """Cache key one team member's ``LiteLLM_TeamMembership`` row is stored under for the admission check."""
    return f"{team_id}_{user_id}"


def team_membership_reservation_cache_key(user_id: str, team_id: str) -> str:
    """Cache key the pre-call budget reservation stores the same ``LiteLLM_TeamMembership`` row under.

    Deliberately not unified with ``team_membership_auth_cache_key``: the two readers wrote independent
    keys before this file existed, so a fix that invalidates one must invalidate both explicitly rather
    than assume a single write is visible to both.
    """
    return f"team_membership:{user_id}:{team_id}"


#: Cached under ``team_membership_reservation_cache_key`` when a member has no ``LiteLLM_TeamMembership``
#: row, so a session-token member without a per-member budget costs no DB read per request. Lives beside
#: the key builder because it is part of the same cache protocol: every reader of the key must know that
#: a plain string here means "no row", distinct from a serialized membership. The two budget readers
#: already treat a non-model value as "no row", so they need no change to stay correct.
NO_TEAM_MEMBERSHIP_SENTINEL: Final = "__no_team_membership__"


def get_management_object_ttl(cache: DualCache) -> float:
    """
    In-memory TTL for management-object cache writes (keys, teams, users, budgets, ...).

    Honors ``general_settings.user_api_key_cache_ttl``, which ``proxy_server``
    propagates onto ``default_in_memory_ttl`` at startup, and falls back to
    ``DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`` when no default is configured.
    """
    configured: Final[float | None] = getattr(cache, "default_in_memory_ttl", None)
    if configured is not None:
        return configured
    return DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL


def _last_good_redis_key(cache_key: str) -> str:
    return f"last_good:{cache_key}"


def _last_good_tombstone_key(cache_key: str) -> str:
    return f"last_good_tombstone:{cache_key}"


def _last_good_store(cache: DualCache) -> UserApiKeyCache | None:
    """The last good entries are a ``UserApiKeyCache`` partition; any other cache keeps none."""
    return cache if isinstance(cache, UserApiKeyCache) else None


def serves_last_good_entries(cache: DualCache) -> bool:
    store: Final = _last_good_store(cache)
    return store is not None and store.serve_last_good_entries


#: When this context's auth reads began, on ``time.monotonic`` (the default ``last_good_clock``): set anew where a
#: request's or an MCP message's auth begins, else by the first bounded database read, and copied into every task the
#: context creates
_auth_reads_started_at: Final[ContextVar[float | None]] = ContextVar("last_good_auth_reads_started_at", default=None)

# Covers a drift or step between the Redis server's clock, on which tombstones expire, and the pod's, on which these
# checks run
LAST_GOOD_LATE_STORE_MARGIN_SECONDS: Final = 5


def mark_last_good_read_start() -> None:
    """Record that this context's auth reads begin now, unless a start is already recorded: inside a request, the
    request's own start wins, and an earlier start only ends an entry sooner and refuses more stores. Every last good
    entry a store keeps ends ``LAST_GOOD_ENTRY_TTL_SECONDS`` after the recorded start."""
    if _auth_reads_started_at.get() is None:
        _ = _auth_reads_started_at.set(time.monotonic())


def start_last_good_reads() -> None:
    """Record that this context's auth reads begin now, in place of any recorded start: called where a request's or an
    MCP message's auth begins, so a context that outlives one request cannot carry an earlier request's start into
    the reads of a later one. The record of the keys this task revoked starts empty for the same reason."""
    _ = _auth_reads_started_at.set(time.monotonic())
    _ = _revoked_by_this_request.set(None)


@contextmanager
def last_good_reads_started_at(start: float | None) -> Generator[None, None, None]:
    """Record ``start`` as the moment this context's auth reads began, in place of any recorded one, until the block
    ends; ``None`` records none. For a test that drives the cache's ``last_good_clock``."""
    token: Final = _auth_reads_started_at.set(start)
    try:
        yield
    finally:
        _auth_reads_started_at.reset(token)


def _late_store_seconds() -> float:
    return min(LAST_GOOD_TOMBSTONE_TTL_SECONDS - LAST_GOOD_LATE_STORE_MARGIN_SECONDS, LAST_GOOD_ENTRY_TTL_SECONDS)


def _store_is_late(read_started_at: float, clock: Callable[[], float]) -> bool:
    """A store this long after its read began comes after the entry's own expiry, or may follow a revocation whose
    tombstone has already lapsed."""
    return clock() - read_started_at >= _late_store_seconds()


def _log_late_store(cache_key: str, *, undone: bool) -> None:
    verbose_proxy_logger.warning(
        "%s the last good auth entry %s: its store came %s s or more after its auth reads began, past its expiry or "
        "the tombstone time of a revocation recorded since then%s",
        "Dropping" if undone else "Did not keep",
        cache_key,
        _late_store_seconds(),
        "" if undone else "; the entry kept before stays until its own expiry",
    )


async def _store_in_redis_unless_tombstoned(
    redis: RedisCache,
    cache_key: str,
    payload: object,
    *,
    read_started_at: float,
    expire_at: int,
    clock: Callable[[], float],
) -> Literal["kept", "late", "refused"]:
    """``kept`` when Redis keeps the entry until ``expire_at``, whole Unix seconds on the Redis server's clock (SET
    EXAT); ``late`` when the store turned late (``_store_is_late``) before its write, so it wrote and deleted nothing;
    ``refused`` when a tombstone refused it or its write was undone. Raises only when Redis cannot answer before the
    write, which leaves nothing in Redis.

    The tombstone is read again after the write, and the write is undone if one appeared or if the store turned late
    in between. A revocation writes its tombstone before it deletes the entry, so one recorded after the read began
    still has its tombstone when that last check runs: the check sees it, or the revocation's delete comes after this
    write. A write or a second read that fails is undone too, since the write may have landed after a tombstone this
    store could not see. The expiry is absolute, so however late Redis runs the SET, and whether or not an undo then
    reaches it, the entry ends at ``expire_at``. An undone write had replaced the entry an earlier read kept, so
    nothing of that is left.
    """
    entry: Final = _last_good_redis_key(cache_key)
    tombstone: Final = _last_good_tombstone_key(cache_key)
    if await redis.async_get_cache_or_raise(tombstone) is not None:
        return "refused"
    if _store_is_late(read_started_at, clock):
        return "late"
    try:
        await redis.async_set_cache_or_raise(entry, payload, expire_at=expire_at)
        if await redis.async_get_cache_or_raise(tombstone) is None:
            if not _store_is_late(read_started_at, clock):
                return "kept"
            _log_late_store(cache_key, undone=True)
    except Exception as e:  # noqa: BLE001  # the write may have landed after a tombstone, so it is undone below
        verbose_proxy_logger.warning(
            "Dropping the last good auth entry %s: Redis failed during its write or the tombstone check after it: %s",
            cache_key,
            e,
        )
    # Best effort: when this delete fails too, the entry's absolute expiry is the backstop
    try:
        _ = await redis.async_delete_cache(entry)
    except Exception as e:  # noqa: BLE001  # the entry ends at its expiry, and the read must not fail on Redis
        verbose_proxy_logger.warning(
            "Could not drop the last good auth entry %s from Redis, so it stays until its expiry at %s: %s",
            cache_key,
            expire_at,
            e,
        )
    return "refused"


async def remember_last_good(cache: DualCache, cache_key: str, value: BaseModel) -> None:
    """Keep ``value``, which the database just returned, as the last good entry for ``cache_key`` until
    ``LAST_GOOD_ENTRY_TTL_SECONDS`` after this context's auth reads began (``mark_last_good_read_start``): in Redis, so
    every pod can serve it, and in this pod's memory for when Redis cannot answer. The end is taken once per store, as
    whole Unix seconds rounded down, and both copies end at that instant: the memory line on the pod's clock, the Redis
    entry by an absolute expiry (SET EXAT, Redis 6.2 or later) on the server's clock, so the entry can live past its
    end only by as much as the server's clock is behind the pod's. An end already past stores nothing and sends no
    SET. Only a database read calls this, so only a read sets or restores an entry's time: serving the entry in an
    outage never extends it.

    A store that comes ``LAST_GOOD_TOMBSTONE_TTL_SECONDS`` less ``LAST_GOOD_LATE_STORE_MARGIN_SECONDS`` or more after
    its read began, or after the entry's expiry, is late: a revocation recorded after the read began may no longer have
    its tombstone. A store with no recorded start, or one that is late before its write, writes nothing and deletes
    nothing, in Redis or in this pod's memory: neither says the object was revoked, and an entry an earlier read kept
    ends at its own expiry. A tombstone refuses the entry, and a store whose Redis write fails, whose tombstone check
    after that write fails or finds one, or that turned late between the write and that check, undoes the write. Those
    refused stores also drop this pod's memory line, since another pod may have revoked the object; only a Redis that
    fails before the write leaves the entry to the memory line. Best effort: a failed write must not fail the read."""
    store: Final = _last_good_store(cache)
    if store is None or not store.serve_last_good_entries or store.last_good_memory.tombstoned(cache_key):
        return
    read_started_at: Final = _auth_reads_started_at.get()
    if read_started_at is None:
        verbose_proxy_logger.warning(
            "Did not keep the last good auth entry %s: no start of its auth reads was recorded, so its expiry is "
            "unknown; the entry kept before stays until its own expiry",
            cache_key,
        )
        return
    if _store_is_late(read_started_at, store.last_good_clock):
        _log_late_store(cache_key, undone=False)
        return
    # One reading of each clock, taken together, gives the one end both copies take
    now: Final = store.last_good_clock()
    wall_now: Final = store.last_good_wall_clock()
    # Rounded down, so the Redis entry never outlives the read's deadline; the memory line takes the same instant
    expire_at: Final = math.floor(wall_now + read_started_at + LAST_GOOD_ENTRY_TTL_SECONDS - now)
    if expire_at <= wall_now:
        verbose_proxy_logger.warning(
            "Did not keep the last good auth entry %s: its expiry passed before its store; the entry kept before stays "
            "until its own expiry",
            cache_key,
        )
        return
    ends_at: Final = now + (expire_at - wall_now)
    payload: Final = CacheCodec.serialize(value)
    if store.redis_cache is not None:
        try:
            stored: Final = await _store_in_redis_unless_tombstoned(
                store.redis_cache,
                cache_key,
                payload,
                read_started_at=read_started_at,
                expire_at=expire_at,
                clock=store.last_good_clock,
            )
        except Exception as e:  # noqa: BLE001  # Redis failed before the write: only the memory line can keep the entry
            verbose_proxy_logger.warning("Could not keep the last good auth entry %s in Redis: %s", cache_key, e)
        else:
            if stored == "late":
                _log_late_store(cache_key, undone=False)
                return
            if stored == "refused":
                # A refused store says this pod's earlier copy may be one a revocation removed elsewhere: it goes too
                store.last_good_memory.delete(cache_key)
                return
            _ = store.last_good_memory.set(cache_key, payload, ends_at)
            return
    # Redis did not check this store, and the time it took to fail counts toward the same bound
    if _store_is_late(read_started_at, store.last_good_clock):
        _log_late_store(cache_key, undone=False)
        return
    _ = store.last_good_memory.set(cache_key, payload, ends_at)


_active_auth_entries_scope: Final[ContextVar[last_good_auth_entries_scope | None]] = ContextVar(
    "last_good_auth_entries_scope", default=None
)


class last_good_auth_entries_scope:
    """Records the type of each last good entry ``recall_last_good`` serves inside the scope, and of each copy of one
    a cache read hands back, from ``asyncio.gather`` children too, since they share the scope object the parent
    opened. ``served_until``, on the cache's ``last_good_clock``, is the earliest end of what it served, so a value
    derived from it is cached no longer. A scope opened inside another one also records into it, so a lookup can tell
    whether it served a last good entry without hiding that from the request."""

    __slots__ = ("_outer", "_token", "served", "served_until")

    def __init__(self) -> None:
        self.served: tuple[str, ...] = ()
        self.served_until: float | None = None
        self._outer: last_good_auth_entries_scope | None = None
        self._token: Token[last_good_auth_entries_scope | None] | None = None

    def record(self, entry_type: str, until: float) -> None:
        self.served = (*self.served, entry_type)
        self.served_until = until if self.served_until is None else min(self.served_until, until)
        if self._outer is not None:
            self._outer.record(entry_type, until)

    def __enter__(self) -> last_good_auth_entries_scope:
        self._outer = _active_auth_entries_scope.get()
        self._token = _active_auth_entries_scope.set(self)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._token is not None:
            _active_auth_entries_scope.reset(self._token)


def _record_last_good_served(entry_type: str, until: float) -> None:
    scope: Final = _active_auth_entries_scope.get()
    if scope is not None:
        scope.record(entry_type, until)


def record_last_good_served_copy(cache: UserApiKeyCache, cache_key: str, entry_type: str) -> None:
    """Record ``entry_type`` in the open scope for a copy ``keep_last_good_served_copy`` kept under ``cache_key`` that a
    cache read handed back, so a request that rests on it reads last_good like the request that recalled it, and a
    value derived from it is cached no longer than the copy lives. A copy whose end this pod no longer holds counts as
    ending now, so nothing derived from it is cached."""
    _record_last_good_served(entry_type, cache.last_good_copy_ends.get(cache_key, cache.last_good_clock()))


def keep_last_good_served_copy(
    cache: UserApiKeyCache, cache_key: str, value: object, served_in: last_good_auth_entries_scope
) -> None:
    """Cache ``value``, served from a last good entry or decided with one inside ``served_in``, in this pod's memory
    only, for ``LAST_GOOD_SERVED_COPY_TTL_SECONDS`` or until the scope's ``served_until`` when that comes first: a
    value derived from a copy never outlives that copy, and with no time left nothing is cached. A Redis copy would be
    read back into memory with a fresh time, so the value could outlive its entry. A copy already in memory is dropped
    first, also when nothing replaces it: this value supersedes it, and memory keeps the time of a copy that has not
    lapsed, so writing over it would give this value that copy's longer time."""
    memory: Final = cache.in_memory_cache_for(cache_key)
    memory.delete_cache(cache_key)
    _ = cache.last_good_copy_ends.pop(cache_key, None)
    now: Final = cache.last_good_clock()
    # A scope that served nothing gives no end to derive from
    until: Final = now if served_in.served_until is None else served_in.served_until
    seconds: Final = min(LAST_GOOD_SERVED_COPY_TTL_SECONDS, until - now)
    if seconds <= 0:
        return
    memory.set_cache(key=cache_key, value=CacheCodec.serialize(value), ttl=seconds)
    cache.keep_served_copy_end(cache_key, now + seconds)


async def recall_last_good(cache: DualCache, cache_key: str, model_type: type[T]) -> T | None:
    """The last good entry for ``cache_key`` until its expiry, recorded in the open ``last_good_auth_entries_scope``
    when found. Serving it stores nothing, so the entry still ends where the database read that stored it put it."""
    store: Final = _last_good_store(cache)
    if store is None or not store.serve_last_good_entries:
        return None
    entry: Final = await _read_last_good(store, cache_key, model_type)
    if entry is not None:
        # A value derived from the entry is cached at most this long from now, however soon the entry itself ends
        _record_last_good_served(model_type.__name__, store.last_good_clock() + LAST_GOOD_SERVED_COPY_TTL_SECONDS)
    return entry


async def _read_last_good(store: UserApiKeyCache, cache_key: str, model_type: type[T]) -> T | None:
    """Redis answers first, and its miss is final, because another pod may have forgotten the entry before this pod
    heard of it; this pod's memory answers only when Redis cannot. Each drops an entry at its expiry."""
    if store.redis_cache is not None:
        try:
            return CacheCodec.deserialize(
                await store.redis_cache.async_get_cache_or_raise(_last_good_redis_key(cache_key)), model_type
            )
        except Exception as e:  # noqa: BLE001  # Redis away: the memory line answers
            verbose_proxy_logger.warning("Redis could not answer for the last good auth entry %s: %s", cache_key, e)
    return CacheCodec.deserialize(store.last_good_memory.get(cache_key), model_type)


async def forget_last_good(cache: DualCache, cache_key: str, *, tombstone: bool = False) -> None:
    """Drop the last good entry for ``cache_key`` from Redis and this pod's memory. Other pods drop their memory
    copy when the caller's auth cache invalidation broadcast for the same key reaches them.

    Without ``tombstone`` this is the eviction an update runs after its write. It is best effort: when it is lost, the
    entry as it was before the update stays until its expiry, ``LAST_GOOD_ENTRY_TTL_SECONDS`` after the database
    read that stored it, and a database outage until then can serve it.

    ``tombstone`` is for a delete or a block: for ``LAST_GOOD_TOMBSTONE_TTL_SECONDS`` no read may store the entry
    again, and a store that completes that time less ``LAST_GOOD_LATE_STORE_MARGIN_SECONDS`` or more after its read
    began keeps nothing, so a read that was in flight when the row changed cannot bring the old entry back."""
    store: Final = _last_good_store(cache)
    if store is None:
        return
    if tombstone:
        store.last_good_memory.tombstone(cache_key, LAST_GOOD_TOMBSTONE_TTL_SECONDS)
    else:
        store.last_good_memory.delete(cache_key)
    if store.redis_cache is None:
        return
    try:
        if tombstone:
            await store.redis_cache.async_set_cache_or_raise(
                _last_good_tombstone_key(cache_key), 1, ttl=LAST_GOOD_TOMBSTONE_TTL_SECONDS
            )
        _ = await store.redis_cache.async_delete_cache(_last_good_redis_key(cache_key))
    except Exception as e:  # noqa: BLE001  # best effort, like every auth cache eviction after a committed write
        verbose_proxy_logger.warning(
            "Could not drop the last good auth entry %s from Redis, so it stays until the database answers for it or "
            "the entry expires: %s",
            cache_key,
            e,
        )


async def forget_missing_last_good(cache: DualCache, cache_key: str) -> None:
    """Drop the entry of a key, team or membership a read just found missing. On the read path, so it costs nothing
    while the setting is off: every delete already forgets unconditionally."""
    store: Final = _last_good_store(cache)
    if store is None or not store.serve_last_good_entries:
        return
    await forget_last_good(cache, cache_key)


# The task of the request whose revocations Redis recorded, and those keys, each with when on the cache's clock. Only
# that task skips by it: a task the request starts, which copies the variable, keeps a record of its own; and the
# record starts empty where a request's auth begins (``start_last_good_reads``), so two requests one task handles
# never share it
_revoked_by_this_request: Final[  # mutable-ok: the record's dict grows as its task revokes
    ContextVar[tuple[asyncio.Task[object], dict[str, float]] | None]
] = ContextVar("last_good_revoked_by_this_request", default=None)

# The keys ``revoke_last_good`` revoked while a transaction that ends access is open
# (``last_good_tombstones_recorded_if_raised``), recorded again if it raises
_revoked_in_open_transaction: Final[  # mutable-ok: the list grows as the transaction revokes
    ContextVar[list[str] | None]
] = ContextVar("last_good_revoked_in_open_transaction", default=None)


# A revocation passes again over the keys whose tombstones may lapse before the write after it runs, at most this many
# times in all, and then refuses: a Redis that takes that long cannot give the write a standing tombstone
LAST_GOOD_REVOCATION_PASSES: Final = 3


def _tombstones_due(cache_keys: Sequence[str], revoked: Mapping[str, float], now: float) -> tuple[str, ...]:
    """The keys of ``cache_keys`` whose tombstone this task's record does not show standing for longer than
    ``LAST_GOOD_LATE_STORE_MARGIN_SECONDS`` from ``now``: none sent, or one sent that long ago or more. A tombstone
    is sent with a relative expiry, so it ends no earlier than its sending time plus
    ``LAST_GOOD_TOMBSTONE_TTL_SECONDS``."""
    fresh_for: Final = LAST_GOOD_TOMBSTONE_TTL_SECONDS - LAST_GOOD_LATE_STORE_MARGIN_SECONDS
    return tuple(dict.fromkeys(key for key in cache_keys if now - revoked.get(key, -math.inf) >= fresh_for))


def _refused_revocation(cache_keys: Sequence[str], why: str, retry: str) -> HTTPException:
    verbose_proxy_logger.error("Refused to revoke %s: %s", ", ".join(cache_keys), why)
    return HTTPException(
        status_code=503,
        detail={"error": f"The auth cache could not record this revocation, so nothing was changed. {retry}"},
    )


async def revoke_last_good(cache: DualCache, cache_keys: Sequence[str]) -> None:
    """Tombstone and drop the last good entries a request is about to revoke, before it writes anything. Only these
    revocations are strict: a delete of a key, team, user, organization, project or customer, a block of a key, team,
    project or customer, the revocations of the restricted-customer registry, a key's regenerate or rotate (for its
    old token), a key's ``expires`` moved to now or earlier, a team or organization membership removal, a SCIM
    deactivation or delete of a user, and a JWT key mapping's delete or an update of its ``token``, ``jwt_issuer`` or
    ``is_active``. With ``serve_last_good_auth_entries`` on, a Redis that cannot record the revocation refuses the
    request with 503: nothing has changed yet and the operator retries, rather than a revoked credential staying,
    until its expiry, where a database outage serves it. Every other update evicts after its write, best effort
    (``forget_last_good``).

    It returns only when every key's tombstone stands for more than ``LAST_GOOD_LATE_STORE_MARGIN_SECONDS``, so the
    write the caller makes right after it runs while they all stand. A pass records the tombstones of every key not
    already standing that long and drops their entries, in two Redis round trips however many keys
    (``RedisCache.async_set_then_delete_or_raise``); a key whose tombstone, by when this task sent it, may lapse within
    the margin is passed over again, so a pass that takes long is followed by another, and a key this task revoked
    moments ago, whose tombstone still stands, costs no Redis. After ``LAST_GOOD_REVOCATION_PASSES`` passes that each
    outlast that time the request is refused with 503, as when Redis cannot answer. Only the Redis writes are skipped
    for a standing key: this pod's memory tombstone, and every eviction and broadcast after the write, still run."""
    store: Final = _last_good_store(cache)
    if store is None or not store.serve_last_good_entries or not cache_keys:
        return
    redis: Final = store.redis_cache
    if redis is not None:
        task: Final = asyncio.current_task()
        kept: Final = _revoked_by_this_request.get()
        revoked: Final[dict[str, float]] = (  # mutable-ok: grows as the request revokes
            kept[1] if kept is not None and task is not None and kept[0] is task else {}
        )
        if task is not None and (kept is None or kept[1] is not revoked):
            _ = _revoked_by_this_request.set((task, revoked))
        for _ in range(LAST_GOOD_REVOCATION_PASSES):
            due = _tombstones_due(cache_keys, revoked, store.last_good_clock())
            if not due:
                break
            sent_at = store.last_good_clock()
            try:
                await redis.async_set_then_delete_or_raise(
                    tuple(_last_good_tombstone_key(key) for key in due),
                    1,
                    LAST_GOOD_TOMBSTONE_TTL_SECONDS,
                    tuple(_last_good_redis_key(key) for key in due),
                )
            except Exception as e:
                raise _refused_revocation(
                    cache_keys,
                    f"Redis could not record the revocation of their last good auth entries: {e}",
                    "Retry once Redis is reachable.",
                ) from e
            revoked.update(dict.fromkeys(due, sent_at))
        if _tombstones_due(cache_keys, revoked, store.last_good_clock()):
            raise _refused_revocation(
                cache_keys,
                f"each of {LAST_GOOD_REVOCATION_PASSES} passes recording it in Redis took longer than "
                f"LAST_GOOD_TOMBSTONE_TTL_SECONDS ({LAST_GOOD_TOMBSTONE_TTL_SECONDS} s) less "
                f"{LAST_GOOD_LATE_STORE_MARGIN_SECONDS} s, so its tombstones could lapse before the change",
                "Recording it took longer than its tombstones last; retry it.",
            )
    for cache_key in cache_keys:
        store.last_good_memory.tombstone(cache_key, LAST_GOOD_TOMBSTONE_TTL_SECONDS)
    in_transaction: Final = _revoked_in_open_transaction.get()
    if in_transaction is not None:
        in_transaction.extend(cache_keys)


async def record_last_good_tombstones_again(cache: DualCache, cache_keys: Sequence[str]) -> None:
    """Record again, once a strict write has returned or raised, the tombstones of the entries it ended, and drop the
    entries, in this pod's memory and in Redis (two round trips however many keys). Another worker's read that began
    before the write committed, while the first tombstones stood, can store the old row once they lapse; a tombstone
    recorded at time e refuses every store until e plus ``LAST_GOOD_TOMBSTONE_TTL_SECONDS``, by which time such a read
    is late, and its delete drops one stored before e. Nothing with the setting off or with no keys.

    Best effort, since the write may have committed: a Redis that fails here is logged, not raised, and leaves an old
    row another worker stored after the first tombstones lapsed until that entry expires."""
    store: Final = _last_good_store(cache)
    if store is None or not store.serve_last_good_entries or not cache_keys:
        return
    keys: Final = tuple(dict.fromkeys(cache_keys))
    for cache_key in keys:
        store.last_good_memory.tombstone(cache_key, LAST_GOOD_TOMBSTONE_TTL_SECONDS)
    if store.redis_cache is None:
        return
    try:
        await store.redis_cache.async_set_then_delete_or_raise(
            tuple(_last_good_tombstone_key(key) for key in keys),
            1,
            LAST_GOOD_TOMBSTONE_TTL_SECONDS,
            tuple(_last_good_redis_key(key) for key in keys),
        )
    except Exception as e:  # noqa: BLE001  # best effort: the write may have committed
        verbose_proxy_logger.warning(
            "Could not record the tombstones of %s again after their write, so an old row stored meanwhile stays "
            "until its entry expires: %s",
            ", ".join(keys),
            e,
        )


async def revoke_last_good_then(
    cache: DualCache, cache_keys: Sequence[str], write: Callable[[], Awaitable[_Written]]
) -> _Written:
    """``revoke_last_good`` of ``cache_keys``, then ``write``, with nothing awaited in between: a write that ends what
    those entries hold. A request that already revoked them, while their tombstones stand, needs no Redis here; once
    a slow step may have outlasted them, they are revoked again, or the 503 leaves ``write`` unmade. Once ``write``
    returns or raises, cancellation included, their tombstones are recorded again
    (``record_last_good_tombstones_again``), and then its result or its exception goes on unchanged: a write that
    needed no Redis here can start with only ``LAST_GOOD_LATE_STORE_MARGIN_SECONDS`` of them left, and a read another
    worker began before it committed could otherwise store the old row once they lapse. A write that raises may still
    have committed in part (several deletes run together, of which one failed; a driver error once the commit reached
    the database; an await cancelled after it), and its caller's evictions after it never run. A write inside a
    transaction commits later, so its caller records them again after the commit too. No keys: just the write."""
    await revoke_last_good(cache, cache_keys)
    try:
        return await write()
    finally:
        await record_last_good_tombstones_again(cache, cache_keys)


@asynccontextmanager
async def last_good_tombstones_recorded_if_raised(cache: DualCache) -> AsyncIterator[None]:
    """Around a database transaction whose writes end access: the keys ``revoke_last_good`` revokes while it is open are
    noted, and when the transaction raises, cancellation included, their tombstones are recorded again
    (``record_last_good_tombstones_again``) before the exception goes on. A transaction that raises may still have
    committed (a driver error once the commit reached the database), and then the evictions after its commit, which
    record them when it returns, never run. Not nested: an inner one would hide its keys from the outer one. Nothing is
    noted with the setting off, so nothing is sent then."""
    revoked: Final[list[str]] = []  # mutable-ok: grows as the transaction revokes
    token: Final = _revoked_in_open_transaction.set(revoked)
    try:
        yield
    except BaseException:
        await record_last_good_tombstones_again(cache, revoked)
        raise
    finally:
        _revoked_in_open_transaction.reset(token)


# Rows that keep appearing between a read and its delete for this many rounds are left, and the request refused
LAST_GOOD_EXACT_DELETE_ROUNDS: Final = 3


async def revoke_and_delete_read_rows(
    cache: DualCache,
    read: Callable[[], Awaitable[Sequence[_Row]]],
    cache_keys_of: Callable[[Sequence[_Row]], Sequence[str]],
    delete: Callable[[Sequence[_Row]], Awaitable[object]],
    *,
    prepare: Callable[[Sequence[_Row]], Awaitable[object]] | None = None,
    chunk_size: int = IN_LIST_CHUNK_SIZE,
) -> tuple[_Row, ...]:
    """Delete the rows ``read`` finds and only those, each round paired with the revocation of their entries
    (``revoke_last_good_then``: ``delete`` runs right after it), then read again, until a read finds none. It starts
    with a read of its own, never an earlier one, so a row added before it began is found too, and a row added between
    a read and its delete is not deleted unrevoked: the next round reads, revokes and deletes it. ``read`` must see
    what ``delete`` changes, so a caller with a read replica reads from the writer. ``prepare`` gets the rows before
    their revocation, for the reads and writes that ready their delete (the cache keys of rows the delete cascades to,
    an archive), so ``delete`` is only the write that ends them. A round of more than ``chunk_size`` rows goes in
    chunks, each its own prepare, revocation and delete, so ``prepare`` and ``delete`` never get more than
    ``chunk_size`` rows: an ``in`` filter built from them stays under Postgres' limit on a statement's parameters. When
    the rows of ``LAST_GOOD_EXACT_DELETE_ROUNDS`` rounds are each followed by more, answers 409 with the rows of the
    last read kept, their entries not revoked by this call. Returns every row it deleted."""
    deleted: Final[list[_Row]] = []  # mutable-ok: grows by the rows of each chunk
    rows = await read()  # rebind-ok: each round reads again
    for _ in range(LAST_GOOD_EXACT_DELETE_ROUNDS):
        if not rows:
            return tuple(deleted)
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            if prepare is not None:
                await prepare(chunk)
            await revoke_last_good_then(cache, cache_keys_of(chunk), partial(delete, chunk))
            deleted.extend(chunk)
        rows = await read()
    if not rows:
        return tuple(deleted)
    raise HTTPException(
        status_code=409,
        detail={
            "error": f"Rows kept being added while they were deleted, {LAST_GOOD_EXACT_DELETE_ROUNDS} times, so the "
            "request stopped before deleting the last ones it found. Retry it."
        },
    )
