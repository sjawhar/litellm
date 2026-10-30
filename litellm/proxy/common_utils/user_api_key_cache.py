from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Final, TypeVar, cast, overload

from fastapi import HTTPException
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import (
    DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    LAST_GOOD_AUTH_ENTRIES_IN_MEMORY,
    LAST_GOOD_TOMBSTONE_TTL_SECONDS,
)
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.caching.redis_batch import BatchResult

T = TypeVar("T", bound=BaseModel)

AUTH_OBJECTS_TARGET: Final = "auth_objects"

_HASHED_TOKEN_CACHE_KEY: Final = re.compile(r"[0-9a-f]{64}")


def is_user_key_cache_key(key: str) -> bool:
    """Only user-key objects are cached under a bare ``hash_token`` digest; every other object uses a prefixed key."""
    return _HASHED_TOKEN_CACHE_KEY.fullmatch(key) is not None


_PIPELINED_SET_OPTIONS: Final = frozenset(("ttl",))


class LastGoodMemory:
    """A pod's own copy of the last good team and key entries: no expiry, bounded, and the least recently
    written entry goes first. Holds the JSON payload, so every read builds a fresh object. A tombstone, written
    when a key or team is deleted or blocked, refuses every write of that entry until it lapses."""

    __slots__ = ("_clock", "_entries", "_max_entries", "_tombstones")

    def __init__(self, max_entries: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._entries: Final[OrderedDict[str, object]] = OrderedDict()  # mutable-ok: bounded LRU every read updates
        self._tombstones: Final[OrderedDict[str, float]] = OrderedDict()  # mutable-ok: bounded, lapses by clock
        self._max_entries: Final = max_entries
        self._clock: Final = clock

    def set(self, key: str, payload: object) -> bool:
        if self.tombstoned(key):
            return False
        _ = self._entries.pop(key, None)
        self._entries[key] = payload
        while len(self._entries) > self._max_entries:
            _ = self._entries.popitem(last=False)
        return True

    def get(self, key: str) -> object:
        return self._entries.get(key)

    def delete(self, key: str) -> None:
        _ = self._entries.pop(key, None)

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
        self.last_good_memory: Final = LastGoodMemory(max_entries=LAST_GOOD_AUTH_ENTRIES_IN_MEMORY)
        self.serve_last_good_entries = serve_last_good_entries

    def in_memory_cache_for(self, key: str) -> InMemoryCache:
        return self.key_object_cache.in_memory_cache if is_user_key_cache_key(key) else self.in_memory_cache

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


def end_user_restricted_registry_cache_key() -> str:
    """Cache key for the set of end-user ids whose row carries a restriction auth enforces."""
    return "end_user_restricted_registry"


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


async def _store_in_redis_unless_tombstoned(redis: RedisCache, cache_key: str, payload: object) -> bool:
    """True when Redis keeps the entry, False when a tombstone refuses it; raises when Redis cannot answer.

    The tombstone is read again after the write and the write is undone if one appeared. A forget writes its
    tombstone before it deletes the entry, so either this second read sees the tombstone or the forget's
    delete comes after this write: a read that was in flight when the key was deleted never leaves it behind.
    """
    entry: Final = _last_good_redis_key(cache_key)
    tombstone: Final = _last_good_tombstone_key(cache_key)
    if await redis.async_get_cache_or_raise(tombstone) is not None:
        return False
    await redis.async_set_cache_or_raise(entry, payload)
    if await redis.async_get_cache_or_raise(tombstone) is None:
        return True
    _ = await redis.async_delete_cache(entry)
    return False


async def remember_last_good(cache: DualCache, cache_key: str, value: BaseModel) -> None:
    """Keep ``value`` as the last good entry for ``cache_key``: in Redis with no expiry, so every pod can serve it,
    and in this pod's memory for when Redis cannot answer. Only an entry the database just returned belongs here,
    and a tombstone left by a delete or a block refuses it. Best effort: a failed write must not fail the read
    that produced it."""
    store: Final = _last_good_store(cache)
    if store is None or not store.serve_last_good_entries or store.last_good_memory.tombstoned(cache_key):
        return
    payload: Final = CacheCodec.serialize(value)
    if store.redis_cache is not None:
        try:
            stored: Final = await _store_in_redis_unless_tombstoned(store.redis_cache, cache_key, payload)
        except Exception as e:  # noqa: BLE001  # the memory line still holds the entry; the read must not fail on Redis
            verbose_proxy_logger.warning("Could not keep the last good auth entry %s in Redis: %s", cache_key, e)
        else:
            if not stored:
                return
    _ = store.last_good_memory.set(cache_key, payload)


async def recall_last_good(cache: DualCache, cache_key: str, model_type: type[T]) -> T | None:
    """The last good entry for ``cache_key``. Redis answers first, and its miss is final, because another pod may
    have forgotten the entry before this pod heard of it; this pod's memory answers only when Redis cannot."""
    store: Final = _last_good_store(cache)
    if store is None or not store.serve_last_good_entries:
        return None
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

    ``tombstone`` is for a delete or a block: for ``LAST_GOOD_TOMBSTONE_TTL_SECONDS``, longer than any database
    read takes, no read may store the entry again, so a read that was in flight when the row changed cannot bring
    the old entry back."""
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
            "Could not drop the last good auth entry %s from Redis, so it stays until the database answers for it: %s",
            cache_key,
            e,
        )


async def revoke_last_good(cache: DualCache, cache_keys: Sequence[str]) -> None:
    """Tombstone and drop the last good entries of keys or teams a request is about to delete or block, before it
    writes anything. A Redis entry never expires, so with ``serve_last_good_auth_entries`` on, a Redis that cannot
    record the revocation refuses the request with 503: nothing has changed yet and the operator retries, rather
    than a revoked credential staying where a later database outage serves it."""
    store: Final = _last_good_store(cache)
    if store is None or not store.serve_last_good_entries or not cache_keys:
        return
    if store.redis_cache is not None:
        try:
            for cache_key in cache_keys:
                await store.redis_cache.async_set_cache_or_raise(
                    _last_good_tombstone_key(cache_key), 1, ttl=LAST_GOOD_TOMBSTONE_TTL_SECONDS
                )
                _ = await store.redis_cache.async_delete_cache(_last_good_redis_key(cache_key))
        except Exception as e:
            verbose_proxy_logger.error(
                "Refused to revoke %s: Redis could not record the revocation of their last good auth entries: %s",
                ", ".join(cache_keys),
                e,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "The auth cache could not record this revocation, so nothing was changed. "
                    "Retry once Redis is reachable."
                },
            ) from e
    for cache_key in cache_keys:
        store.last_good_memory.tombstone(cache_key, LAST_GOOD_TOMBSTONE_TTL_SECONDS)
