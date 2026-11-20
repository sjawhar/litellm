import asyncio
import hashlib
import json
import logging
import math
from collections.abc import Sequence
from typing import Any, Final, Literal

import pytest
from fakeredis import FakeServer

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import (
    DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    LAST_GOOD_ENTRY_TTL_SECONDS,
    LAST_GOOD_TOMBSTONE_TTL_SECONDS,
)
from litellm.models.team import LiteLLM_TeamTableCachedObj
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec
from litellm.proxy.common_utils.user_api_key_cache import (
    LAST_GOOD_LATE_STORE_MARGIN_SECONDS,
    LastGoodMemory,
    UserApiKeyCache,
    end_user_cache_key,
    forget_last_good,
    get_management_object_ttl,
    is_user_key_cache_key,
    last_good_auth_entries_scope,
    last_good_reads_started_at,
    recall_last_good,
    remember_last_good,
    revoke_and_delete_read_rows,
    revoke_last_good,
    revoke_last_good_then,
    start_last_good_reads,
)
from litellm.proxy.proxy_server import UserAPIKeyCacheTTLEnum
from tests.unit.caching.fake_redis_cache import drive_fake_redis_time, fake_redis_cache

HASHED_TOKEN = hashlib.sha256(b"sk-lit7563-hot-key").hexdigest()


class CapturingInMemoryCache(InMemoryCache):
    """Records ``ttl`` passed into ``set_cache`` (what DualCache injects)."""

    def __init__(self) -> None:
        super().__init__()
        self.last_ttl: Any = None

    def set_cache(self, key, value, **kwargs):  # type: ignore[override]
        self.last_ttl = kwargs.get("ttl")
        super().set_cache(key, value, **kwargs)


class _DeleteLandsDuringWrite(RedisCache):
    """The reading pod's Redis: another pod's delete lands just before this pod's entry write reaches Redis."""

    def __init__(self, redis: RedisCache, deleting_pod: UserApiKeyCache) -> None:
        self._redis = redis
        self._deleting_pod = deleting_pod

    async def async_get_cache_or_raise(self, key: str) -> object:
        return await self._redis.async_get_cache_or_raise(key)

    async def async_set_cache_or_raise(
        self, key: str, value: object, ttl: int | None = None, *, expire_at: int | None = None
    ) -> None:
        if key.startswith("last_good:"):
            await forget_last_good(self._deleting_pod, key.removeprefix("last_good:"), tombstone=True)
        await self._redis.async_set_cache_or_raise(key, value, ttl=ttl, expire_at=expire_at)

    async def async_delete_cache(self, key: str) -> None:
        await self._redis.async_delete_cache(key)


class _UndoFailsAfterADeleteLandsDuringWrite(_DeleteLandsDuringWrite):
    """As ``_DeleteLandsDuringWrite``, and Redis goes away before the reading pod can take its write back."""

    async def async_delete_cache(self, key: str) -> None:
        raise ConnectionError("Redis went away before the write was taken back")


class _FailsAfterTheWrite(RedisCache):
    """The reading pod's Redis: the entry write lands, then its answer or the tombstone check after it fails, as when
    Redis goes away in the middle of a store."""

    def __init__(self, redis: RedisCache, failing: Literal["write", "tombstone check"]) -> None:
        self._redis = redis
        self._failing = failing
        self.wrote = False

    async def async_get_cache_or_raise(self, key: str) -> object:
        if self.wrote and self._failing == "tombstone check":
            raise ConnectionError("Redis went away after the write")
        return await self._redis.async_get_cache_or_raise(key)

    async def async_set_cache_or_raise(
        self, key: str, value: object, ttl: int | None = None, *, expire_at: int | None = None
    ) -> None:
        await self._redis.async_set_cache_or_raise(key, value, ttl=ttl, expire_at=expire_at)
        self.wrote = True  # rebind-ok: records that the write reached Redis
        if self._failing == "write":
            raise TimeoutError("the write reached Redis but its answer was lost")

    async def async_delete_cache(self, key: str) -> None:
        await self._redis.async_delete_cache(key)


# How far the wall clock a test drives runs ahead of _Clock: a quarter second into a Unix second, so that a store's
# rounding of the end down to a whole second shows
_WALL_CLOCK_AHEAD_SECONDS: Final = 1_700_000_000.25


class _Clock:
    """The last good entries' clock, which a test moves instead of sleeping. ``wall`` is the wall clock that moves with
    it, for ``last_good_wall_clock`` and for fake Redis (``drive_fake_redis_time``)."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def wall(self) -> float:
        return self.now + _WALL_CLOCK_AHEAD_SECONDS

    def move_to_wall(self, wall: float) -> None:
        self.now = wall - _WALL_CLOCK_AHEAD_SECONDS


class _SlowTombstoneCheck(RedisCache):
    """The reading pod's Redis: its first read, the tombstone check before the entry write, takes ``seconds``."""

    def __init__(self, redis: RedisCache, clock: _Clock, seconds: float) -> None:
        self._redis = redis
        self._clock = clock
        self._seconds = seconds

    async def async_get_cache_or_raise(self, key: str) -> object:
        answer: Final = await self._redis.async_get_cache_or_raise(key)
        self._clock.now += self._seconds
        self._seconds = 0.0  # rebind-ok: only the first read is slow
        return answer

    async def async_set_cache_or_raise(
        self, key: str, value: object, ttl: int | None = None, *, expire_at: int | None = None
    ) -> None:
        await self._redis.async_set_cache_or_raise(key, value, ttl=ttl, expire_at=expire_at)

    async def async_delete_cache(self, key: str) -> None:
        await self._redis.async_delete_cache(key)


class _FailsSlowlyBeforeTheWrite(RedisCache):
    """The reading pod's Redis: the tombstone check before the entry write fails after ``seconds``, as a Redis that
    times out does."""

    def __init__(self, clock: _Clock, seconds: float) -> None:
        self._clock = clock
        self._seconds = seconds

    async def async_get_cache_or_raise(self, key: str) -> object:
        self._clock.now += self._seconds
        raise TimeoutError("Redis did not answer the tombstone check")


class _SlowWrite(RedisCache):
    """The pod's Redis: each SET takes ``seconds`` before it lands, and the keys it SETs are recorded in order."""

    def __init__(self, redis: RedisCache, clock: _Clock, seconds: float) -> None:
        self._redis = redis
        self._clock = clock
        self.seconds = seconds
        self.set_keys: Final[list[str]] = []  # mutable-ok: records every SET in order

    async def async_get_cache_or_raise(self, key: str) -> object:
        return await self._redis.async_get_cache_or_raise(key)

    async def async_set_cache_or_raise(
        self, key: str, value: object, ttl: int | None = None, *, expire_at: int | None = None
    ) -> None:
        self._clock.now += self.seconds
        self.set_keys.append(key)
        await self._redis.async_set_cache_or_raise(key, value, ttl=ttl, expire_at=expire_at)

    async def async_delete_cache(self, key: str) -> None:
        await self._redis.async_delete_cache(key)


class _LostRightAfterASlowWrite(_SlowWrite):
    """As ``_SlowWrite``, and the Redis server is lost right after each SET lands, until the test restores it: the
    tombstone check after the entry's write and the undo both fail."""

    def __init__(self, redis: RedisCache, server: FakeServer, clock: _Clock, seconds: float) -> None:
        super().__init__(redis, clock, seconds)
        self._server = server

    async def async_set_cache_or_raise(
        self, key: str, value: object, ttl: int | None = None, *, expire_at: int | None = None
    ) -> None:
        await super().async_set_cache_or_raise(key, value, ttl, expire_at=expire_at)
        self._server.connected = False


async def _held_in(redis: RedisCache, cache: UserApiKeyCache, cache_key: str) -> tuple[bool, bool]:
    """Whether the Redis server and this pod's memory each hold the last good entry for ``cache_key``."""
    return (
        await redis.init_async_client().get(f"last_good:{cache_key}") is not None,
        cache.last_good_memory.get(cache_key) is not None,
    )


def _cache_on_driven_clocks(monkeypatch: pytest.MonkeyPatch, clock: _Clock, redis: RedisCache) -> UserApiKeyCache:
    """A cache serving last good entries whose stores, and every fake Redis server, run on ``clock`` and its wall."""
    drive_fake_redis_time(monkeypatch, clock.wall)
    return UserApiKeyCache(
        redis_cache=redis, serve_last_good_entries=True, last_good_clock=clock, last_good_wall_clock=clock.wall
    )


class FakeRedisCache(RedisCache):
    """
    In-memory fake that enforces the UserApiKeyCache Redis payload contract.

    For user_api_key_cache entries we expect Redis to store a JSON object (dict)
    produced by `CacheCodec.serialize(..., model_type=...)`.

    This fake:
    - raises TypeError if the value is not a dict
    - raises TypeError if the dict is not JSON-serializable

    Records the ``ttl`` kwarg DualCache forwards on each Redis write for tests.
    """

    def __init__(self):
        self._store: dict[str, str] = {}
        self.last_ttl: Any = None

    def set_cache(self, key: str, value: Any, **kwargs):  # type: ignore[override]
        if not isinstance(value, dict):
            raise TypeError("FakeRedisCache only accepts dict payloads")
        self.last_ttl = kwargs.get("ttl")
        self._store[key] = json.dumps(value)
        return True

    def get_cache(self, key: str, **kwargs):  # type: ignore[override]
        raw = self._store.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def async_set_cache(self, key: str, value: Any, **kwargs):  # type: ignore[override]
        if not isinstance(value, dict):
            raise TypeError("FakeRedisCache only accepts dict payloads")
        self.last_ttl = kwargs.get("ttl")
        self._store[key] = json.dumps(value)
        return True

    async def async_get_cache(self, key: str, **kwargs):  # type: ignore[override]
        raw = self._store.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def delete_cache(self, key: str):  # type: ignore[override]
        self._store.pop(key, None)

    async def async_delete_cache(self, key: str):  # type: ignore[override]
        self._store.pop(key, None)

    async def delete_cache_keys(self, keys):  # type: ignore[override]
        for key in keys:
            self._store.pop(key, None)


class PartitionFailingRedisCache(FakeRedisCache):
    """Fails the batch delete for the key-object partition and no other."""

    async def delete_cache_keys(self, keys):  # type: ignore[override]
        if any(is_user_key_cache_key(key) for key in keys):
            raise ConnectionError("redis unavailable")
        await super().delete_cache_keys(keys)


def _make_key_obj(token: str = "tok") -> UserAPIKeyAuth:
    # Minimal object (UserAPIKeyAuth inherits token from base view).
    return UserAPIKeyAuth(token=token)


class TestUserApiKeyCache:
    @pytest.mark.asyncio
    async def test_async_set_in_memory_gets_enum_default_when_user_api_key_cache_ttl_omitted(
        self,
    ):
        """
        If ``general_settings.user_api_key_cache_ttl`` is absent, the proxy never
        calls ``update_cache_ttl``; ``user_api_key_cache`` keeps
        ``default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl``.
        DualCache must forward that as the in-memory ``ttl`` kwarg on each set.
        """
        mem = CapturingInMemoryCache()
        cache = UserApiKeyCache(
            in_memory_cache=mem,
            redis_cache=FakeRedisCache(),
            default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value,
        )
        await cache.async_set_cache(
            "k",
            _make_key_obj("t"),
            model_type=UserAPIKeyAuth,
        )
        expected = UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value
        assert mem.last_ttl == expected

    def test_sync_set_in_memory_gets_enum_default_when_user_api_key_cache_ttl_omitted(
        self,
    ):
        mem = CapturingInMemoryCache()
        cache = UserApiKeyCache(
            in_memory_cache=mem,
            redis_cache=FakeRedisCache(),
            default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value,
        )
        cache.set_cache("sk", _make_key_obj("s"), model_type=UserAPIKeyAuth)
        assert mem.last_ttl == UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value

    @pytest.mark.asyncio
    async def test_async_set_forwards_default_in_memory_ttl_to_redis_layer(self):
        """
        DualCache injects missing ``ttl`` from ``default_in_memory_ttl`` into kwargs
        before calling ``redis_cache.async_set_cache`` — Redis should receive the same
        TTL as memory (matches proxy defaults: enum 60s).
        """
        fake = FakeRedisCache()
        cache = UserApiKeyCache(
            redis_cache=fake,
            default_in_memory_ttl=60,
        )

        await cache.async_set_cache(
            key="ttl-key",
            value=_make_key_obj("ttl-tok"),
            model_type=UserAPIKeyAuth,
        )

        assert fake.last_ttl == 60

    @pytest.mark.asyncio
    async def test_async_set_explicit_ttl_override_reaches_redis(self):
        fake = FakeRedisCache()
        cache = UserApiKeyCache(
            redis_cache=fake,
            default_in_memory_ttl=60,
        )

        await cache.async_set_cache(
            key="k",
            value=_make_key_obj("x"),
            model_type=UserAPIKeyAuth,
            ttl=900,
        )

        assert fake.last_ttl == 900

    def test_sync_set_forwards_default_in_memory_ttl_to_redis_layer(self):
        fake = FakeRedisCache()
        cache = UserApiKeyCache(
            redis_cache=fake,
            default_in_memory_ttl=45,
        )
        cache.set_cache(
            "sk",
            _make_key_obj("sync"),
            model_type=UserAPIKeyAuth,
        )
        assert fake.last_ttl == 45

    @pytest.mark.asyncio
    async def test_async_set_typed_stores_serialized_payload_in_memory_and_redis(self):
        cache = UserApiKeyCache(redis_cache=FakeRedisCache())
        obj = _make_key_obj("abc")

        await cache.async_set_cache("k", obj, model_type=UserAPIKeyAuth)

        # In-memory hit should still be raw dict (not BaseModel) because wrapper
        # stores the serialized payload into both layers.
        raw = await cache.in_memory_cache.async_get_cache("k")  # type: ignore[union-attr]
        assert isinstance(raw, dict)
        assert raw["token"] == "abc"

        # Redis should also hold the same serialized dict
        redis_raw = await cache.redis_cache.async_get_cache("k")  # type: ignore[union-attr]
        assert redis_raw == raw

    @pytest.mark.asyncio
    async def test_async_get_typed_returns_model_on_valid_hit(self):
        cache = UserApiKeyCache(redis_cache=FakeRedisCache())
        await cache.async_set_cache("k", {"token": "abc"}, model_type=UserAPIKeyAuth)

        value = await cache.async_get_cache("k", model_type=UserAPIKeyAuth)
        assert value is not None
        assert isinstance(value, UserAPIKeyAuth)
        assert value.token == "abc"

    @pytest.mark.asyncio
    async def test_async_get_typed_returns_none_on_validation_failure_after_hit(self):
        cache = UserApiKeyCache(redis_cache=FakeRedisCache())

        # Bypass UserApiKeyCache.serialize: CacheCodec rejects non-dict cached values
        # for dict-based models (deserialize returns None).
        await cache.in_memory_cache.async_set_cache(key="k", value="invalid-payload-not-a-dict")

        value = await cache.async_get_cache("k", model_type=UserAPIKeyAuth)
        assert value is None

    def test_fake_redis_cache_rejects_non_json_serializable_values(self):
        fake = FakeRedisCache()

        class NotSerializable:
            pass

        with pytest.raises(TypeError):
            fake.set_cache("k", NotSerializable())

        with pytest.raises(TypeError):
            fake.set_cache("k2", {"ok": NotSerializable()})


class TestUserKeyObjectPartition:
    """
    Regression for LIT-7563: user-key objects share one 200-entry ``InMemoryCache`` with
    every other management object, so end-user / team / tag churn evicts hot keys and
    forces a ``LiteLLM_VerificationToken`` lookup on the next request.
    """

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            (HASHED_TOKEN, True),
            (HASHED_TOKEN.upper(), False),
            (f"team_id:{HASHED_TOKEN}", False),
            (end_user_cache_key("u1"), False),
            ("sk-lit7563-hot-key", False),
        ],
    )
    def test_is_user_key_cache_key(self, key: str, expected: bool):
        assert is_user_key_cache_key(key) is expected

    @pytest.mark.asyncio
    async def test_management_object_churn_does_not_evict_key_object(self):
        cache = UserApiKeyCache(in_memory_cache=InMemoryCache(max_size_in_memory=2))
        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth, ttl=100)
        for i in range(2):
            await cache.async_set_cache(end_user_cache_key(f"u{i}"), {"user_id": f"u{i}"}, ttl=200)

        key_obj = await cache.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth)
        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN
        assert cache.get_cache(end_user_cache_key("u1")) == {"user_id": "u1"}
        assert HASHED_TOKEN not in cache.in_memory_cache.cache_dict

    def test_sync_write_and_read_route_to_key_object_partition(self):
        cache = UserApiKeyCache(in_memory_cache=InMemoryCache(max_size_in_memory=2))
        cache.set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth, ttl=100)
        for i in range(2):
            cache.set_cache(end_user_cache_key(f"u{i}"), {"user_id": f"u{i}"}, ttl=200)

        key_obj = cache.get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth)
        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN

    @pytest.mark.asyncio
    async def test_redis_hit_backfills_key_object_partition_with_configured_ttl(self):
        redis = FakeRedisCache()
        writer = UserApiKeyCache(redis_cache=redis, default_in_memory_ttl=30)
        await writer.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)

        key_partition = CapturingInMemoryCache()
        reader = UserApiKeyCache(redis_cache=redis, default_in_memory_ttl=30, key_object_in_memory_cache=key_partition)
        key_obj = await reader.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth)

        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN
        assert key_partition.last_ttl == 30
        assert HASHED_TOKEN not in reader.in_memory_cache.cache_dict

    @pytest.mark.asyncio
    async def test_update_cache_ttl_applies_to_key_object_partition(self):
        key_partition = CapturingInMemoryCache()
        cache = UserApiKeyCache(default_in_memory_ttl=60, key_object_in_memory_cache=key_partition)
        cache.update_cache_ttl(default_in_memory_ttl=7, default_redis_ttl=7)

        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)

        assert key_partition.last_ttl == 7

    @pytest.mark.asyncio
    async def test_attach_redis_cache_applies_to_key_object_partition(self):
        redis = FakeRedisCache()
        cache = UserApiKeyCache()
        cache.attach_redis_cache(redis)

        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)

        other_worker = UserApiKeyCache(redis_cache=redis)
        key_obj = await other_worker.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth)
        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN

    @pytest.mark.asyncio
    async def test_delete_removes_key_object_from_partition_and_redis(self):
        redis = FakeRedisCache()
        cache = UserApiKeyCache(redis_cache=redis)
        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)
        assert await cache.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth) is not None

        cache.delete_cache(HASHED_TOKEN)

        assert await cache.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth) is None
        assert await redis.async_get_cache(HASHED_TOKEN) is None

    @pytest.mark.asyncio
    async def test_async_delete_removes_key_object_from_partition_and_redis(self):
        redis = FakeRedisCache()
        cache = UserApiKeyCache(redis_cache=redis)
        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)

        await cache.async_delete_cache(HASHED_TOKEN)

        assert await cache.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth) is None
        assert await redis.async_get_cache(HASHED_TOKEN) is None

    @pytest.mark.asyncio
    async def test_batch_delete_routes_each_key_to_its_partition(self):
        """A batch delete has to clear the same partition the single delete does.

        ``DualCache``'s batch delete only knows about the main in-memory cache, so
        inheriting it unchanged leaves a key object sitting in ``key_object_cache``
        with its pre-reset spend, and the next request is authorized against that
        stale copy until the local entry expires.
        """
        redis = FakeRedisCache()
        cache = UserApiKeyCache(redis_cache=redis)
        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)
        await cache.async_set_cache(end_user_cache_key("u1"), {"user_id": "u1"})

        await cache.async_delete_cache_keys([HASHED_TOKEN, end_user_cache_key("u1")])

        assert await cache.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth) is None
        assert await cache.async_get_cache(end_user_cache_key("u1")) is None
        assert await redis.async_get_cache(HASHED_TOKEN) is None
        assert await redis.async_get_cache(end_user_cache_key("u1")) is None

    @pytest.mark.asyncio
    async def test_batch_delete_clears_the_other_partition_when_one_fails(self):
        """One partition failing must not cost the other its deletions.

        A caller batching these has already committed the rows they cache, so a
        partition that is skipped keeps authorizing against pre-reset spend until
        the entry expires. The failure is still raised for the caller to report.
        """
        redis = PartitionFailingRedisCache()
        cache = UserApiKeyCache(redis_cache=redis)
        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)
        await cache.async_set_cache(end_user_cache_key("u1"), {"user_id": "u1"})

        with pytest.raises(ConnectionError):
            await cache.async_delete_cache_keys([HASHED_TOKEN, end_user_cache_key("u1")])

        assert await cache.async_get_cache(end_user_cache_key("u1")) is None
        assert await redis.async_get_cache(end_user_cache_key("u1")) is None

    @pytest.mark.asyncio
    async def test_pipeline_write_routes_each_entry_to_its_partition(self):
        cache = UserApiKeyCache(in_memory_cache=InMemoryCache(max_size_in_memory=2))
        await cache.async_set_cache_pipeline(
            [(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN))]
            + [(end_user_cache_key(f"u{i}"), {"user_id": f"u{i}"}) for i in range(2)],
            ttl=100,
        )

        key_obj = await cache.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth)
        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN
        assert HASHED_TOKEN not in cache.in_memory_cache.cache_dict
        assert cache.get_cache(end_user_cache_key("u1")) == {"user_id": "u1"}

    def test_flush_clears_key_object_partition(self):
        cache = UserApiKeyCache()
        cache.set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)
        cache.set_cache(end_user_cache_key("u1"), {"user_id": "u1"})

        cache.flush_cache()

        assert cache.get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth) is None
        assert cache.get_cache(end_user_cache_key("u1")) is None

    def test_in_memory_cache_for_routes_by_key(self):
        cache = UserApiKeyCache()
        assert cache.in_memory_cache_for(HASHED_TOKEN) is cache.key_object_cache.in_memory_cache
        assert cache.in_memory_cache_for(end_user_cache_key("u1")) is cache.in_memory_cache

    def test_update_in_memory_max_size_applies_to_key_object_partition(self):
        cache = UserApiKeyCache(
            in_memory_cache=InMemoryCache(max_size_in_memory=2),
            key_object_in_memory_cache=InMemoryCache(max_size_in_memory=2),
        )
        cache.update_in_memory_max_size(3)

        tokens = tuple(hashlib.sha256(f"sk-key-{i}".encode()).hexdigest() for i in range(3))
        for token in tokens:
            cache.set_cache(token, _make_key_obj(token), model_type=UserAPIKeyAuth, ttl=100)
        for i in range(3):
            cache.set_cache(end_user_cache_key(f"u{i}"), {"user_id": f"u{i}"}, ttl=100)

        first_key = cache.get_cache(tokens[0], model_type=UserAPIKeyAuth)
        assert first_key is not None, "key partition still evicts at its old capacity"
        assert first_key.token == tokens[0]
        assert cache.get_cache(end_user_cache_key("u0")) == {"user_id": "u0"}

    def test_update_in_memory_max_size_none_resets_key_object_partition_to_default(self):
        cache = UserApiKeyCache(key_object_in_memory_cache=InMemoryCache(max_size_in_memory=1))
        cache.update_in_memory_max_size(None)

        tokens = tuple(hashlib.sha256(f"sk-key-{i}".encode()).hexdigest() for i in range(2))
        for token in tokens:
            cache.set_cache(token, _make_key_obj(token), model_type=UserAPIKeyAuth, ttl=100)

        first_key = cache.get_cache(tokens[0], model_type=UserAPIKeyAuth)
        assert first_key is not None
        assert first_key.token == tokens[0]


class TestManagementObjectTTL:
    """
    Regression for LIT-3338: ``general_settings.user_api_key_cache_ttl`` (which the
    proxy propagates onto ``default_in_memory_ttl``) must win over the hardcoded
    ``DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`` for management-object writes.
    """

    def test_returns_configured_default_in_memory_ttl(self):
        cache = UserApiKeyCache(default_in_memory_ttl=300)
        assert get_management_object_ttl(cache) == 300

    def test_falls_back_to_constant_when_no_default_configured(self):
        cache = UserApiKeyCache()
        assert cache.default_in_memory_ttl is None
        assert get_management_object_ttl(cache) == DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL

    def test_resolves_on_a_plain_dual_cache(self):
        # Many call sites are typed UserApiKeyCache but exercised in tests with a
        # bare DualCache; the resolver must work on the base type, not just the subclass.
        assert get_management_object_ttl(DualCache(default_in_memory_ttl=300)) == 300
        assert get_management_object_ttl(DualCache()) == DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL

    @pytest.mark.asyncio
    async def test_management_write_uses_configured_ttl_over_constant(self):
        mem = CapturingInMemoryCache()
        cache = UserApiKeyCache(
            in_memory_cache=mem,
            redis_cache=FakeRedisCache(),
            default_in_memory_ttl=300,
        )
        assert get_management_object_ttl(cache) != (DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL)

        await cache.async_set_cache(
            "team_id:abc",
            _make_key_obj("t"),
            model_type=UserAPIKeyAuth,
            ttl=get_management_object_ttl(cache),
        )

        assert mem.last_ttl == 300


class TestLastGoodEntries:
    """The last good team or key entry: Redis first, this pod's memory when Redis cannot answer, both until the entry
    expires ``LAST_GOOD_ENTRY_TTL_SECONDS`` after the database read that stored it."""

    @staticmethod
    def _team(team_id: str, alias: str = "alias") -> LiteLLM_TeamTableCachedObj:
        return LiteLLM_TeamTableCachedObj(team_id=team_id, team_alias=alias, models=["claude-opus-5-5"])

    @pytest.mark.asyncio
    async def test_the_redis_entry_ends_with_the_memory_line(self, monkeypatch: pytest.MonkeyPatch):
        """The end is taken as the store begins, from the moment the read began, rounded down to a whole Unix second:
        Redis gets it as an absolute expiry, so the 30.5 s the tombstone check before the write takes do not move it,
        and the memory line ends at that same instant."""
        clock: Final = _Clock()
        redis: Final = fake_redis_cache()
        cache: Final = _cache_on_driven_clocks(
            monkeypatch, clock, _SlowTombstoneCheck(redis.cache, clock, seconds=30.5)
        )
        expire_at: Final = math.floor(clock.wall() + LAST_GOOD_ENTRY_TTL_SECONDS)

        with last_good_reads_started_at(clock()):
            await remember_last_good(cache, "team_id:t1", self._team("t1"))

        assert await redis.cache.init_async_client().expiretime("last_good:team_id:t1") == expire_at
        redis.server.connected = False
        clock.move_to_wall(expire_at - 0.001)
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")
        clock.move_to_wall(expire_at)
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_an_outage_is_served_an_entry_inside_its_expiry_and_not_one_older(self):
        clock = _Clock()
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=clock)
        with last_good_reads_started_at(clock()):
            await remember_last_good(cache, "team_id:t1", self._team("t1"))
        redis.server.connected = False

        clock.now += LAST_GOOD_ENTRY_TTL_SECONDS - 1
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")
        clock.now += 1
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_serving_an_entry_does_not_extend_its_expiry(self):
        """Only a database read sets an entry's time: an entry served from Redis and then from memory during an
        outage still ends where its read put it."""
        clock = _Clock()
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=clock)
        stored_at: Final = clock.now
        with last_good_reads_started_at(clock()):
            await remember_last_good(cache, "team_id:t1", self._team("t1"))

        clock.now = stored_at + LAST_GOOD_ENTRY_TTL_SECONDS / 2
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")
        redis.server.connected = False
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")
        clock.now = stored_at + LAST_GOOD_ENTRY_TTL_SECONDS

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_a_thirty_day_expiry_still_serves_an_entry_far_older_than_the_default(self, monkeypatch):
        thirty_days: Final = 2592000
        monkeypatch.setattr(
            "litellm.proxy.common_utils.user_api_key_cache.LAST_GOOD_ENTRY_TTL_SECONDS", thirty_days
        )
        clock = _Clock()
        redis = fake_redis_cache()
        cache = _cache_on_driven_clocks(monkeypatch, clock, redis.cache)
        with last_good_reads_started_at(clock()):
            await remember_last_good(cache, "team_id:t1", self._team("t1"))

        assert await redis.cache.init_async_client().expiretime("last_good:team_id:t1") == math.floor(
            clock.wall() + thirty_days
        )
        redis.server.connected = False
        clock.now += thirty_days - 1
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")
        clock.now += 1
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_an_entry_ends_its_expiry_after_its_read_began_not_after_its_store(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The read began at 0 and its store completed at 100: the entry ends at the expiry after 0, in memory and in
        Redis, not 100 s later."""
        clock: Final = _Clock(0.0)
        redis: Final = fake_redis_cache()
        cache: Final = _cache_on_driven_clocks(monkeypatch, clock, redis.cache)
        read_began_on_the_wall: Final = clock.wall()
        with last_good_reads_started_at(0.0):
            clock.now = 100.0
            await remember_last_good(cache, "team_id:t1", self._team("t1"))

        assert await redis.cache.init_async_client().expiretime("last_good:team_id:t1") == math.floor(
            read_began_on_the_wall + LAST_GOOD_ENTRY_TTL_SECONDS
        )
        redis.server.connected = False
        clock.now = LAST_GOOD_ENTRY_TTL_SECONDS - 1
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")
        clock.now = LAST_GOOD_ENTRY_TTL_SECONDS
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entry_ttl_seconds", [LAST_GOOD_ENTRY_TTL_SECONDS, 2592000])
    async def test_a_store_that_completes_after_the_tombstone_of_a_delete_lapsed_keeps_nothing(
        self, entry_ttl_seconds: int, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """A read begins, another pod deletes the team and leaves its tombstone, the tombstone lapses, and only then
        does the read's store complete: whatever the entry's expiry, the deleted team is kept nowhere."""
        monkeypatch.setattr(
            "litellm.proxy.common_utils.user_api_key_cache.LAST_GOOD_ENTRY_TTL_SECONDS", entry_ttl_seconds
        )
        clock: Final = _Clock()
        redis: Final = fake_redis_cache()
        deleting_pod: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        reading_pod: Final = UserApiKeyCache(
            redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=clock
        )
        read_started_at: Final = clock()
        await forget_last_good(deleting_pod, "team_id:t1", tombstone=True)
        # The tombstone lapses: Redis expires it, and the reading pod's clock passes its whole life
        await redis.cache.init_async_client().delete("last_good_tombstone:team_id:t1")
        clock.now += LAST_GOOD_TOMBSTONE_TTL_SECONDS

        with last_good_reads_started_at(read_started_at), caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
            await remember_last_good(reading_pod, "team_id:t1", self._team("t1"))

        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is None
        assert reading_pod.last_good_memory.get("team_id:t1") is None
        assert any(r.levelname == "WARNING" and "team_id:t1" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "lateness", ["no start", "an old start", "late after the check before the write", "late while Redis fails"]
    )
    async def test_a_store_late_before_its_write_keeps_the_entry_an_earlier_read_kept(
        self, lateness: str, caplog: pytest.LogCaptureFixture
    ):
        """A store with no recorded start, or one already late before its write, says nothing about a revocation: it
        writes and deletes nothing, so the entry an earlier, timely read kept stays in Redis and in this pod's memory
        until its own expiry."""
        clock: Final = _Clock()
        redis: Final = fake_redis_cache()
        late_after: Final = LAST_GOOD_TOMBSTONE_TTL_SECONDS - LAST_GOOD_LATE_STORE_MARGIN_SECONDS
        reading_pod_redis: Final[dict[str, RedisCache]] = {
            "no start": redis.cache,
            "an old start": redis.cache,
            "late after the check before the write": _SlowTombstoneCheck(redis.cache, clock, seconds=late_after),
            "late while Redis fails": _FailsSlowlyBeforeTheWrite(clock, seconds=late_after),
        }
        reading_pod: Final = UserApiKeyCache(
            redis_cache=reading_pod_redis[lateness], serve_last_good_entries=True, last_good_clock=clock
        )
        earlier_pod: Final = UserApiKeyCache(
            redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=clock
        )
        before: Final = self._team("t1", alias="before")
        with last_good_reads_started_at(clock()):
            await remember_last_good(earlier_pod, "team_id:t1", before)
        assert reading_pod.last_good_memory.set(
            "team_id:t1", CacheCodec.serialize(before), ends_at=clock() + LAST_GOOD_ENTRY_TTL_SECONDS
        )
        starts: Final = {"no start": None, "an old start": clock() - late_after}

        with last_good_reads_started_at(starts.get(lateness, clock())):
            with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
                await remember_last_good(reading_pod, "team_id:t1", self._team("t1", alias="after"))

        # Redis answers for the earlier pod, so its recall shows what Redis holds
        assert await recall_last_good(earlier_pod, "team_id:t1", LiteLLM_TeamTableCachedObj) == before
        assert reading_pod.last_good_memory.get("team_id:t1") == CacheCodec.serialize(before)
        assert any(r.levelname == "WARNING" and "team_id:t1" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_a_store_that_turns_late_during_its_write_is_undone_and_drops_the_memory_line(self):
        """The write replaced the entry an earlier read kept, so taking it back leaves no entry in Redis, and this
        pod's memory copy goes too, as for every undone store."""
        clock: Final = _Clock()
        redis: Final = fake_redis_cache()
        late_after: Final = LAST_GOOD_TOMBSTONE_TTL_SECONDS - LAST_GOOD_LATE_STORE_MARGIN_SECONDS
        earlier_pod: Final = UserApiKeyCache(
            redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=clock
        )
        reading_pod: Final = UserApiKeyCache(
            redis_cache=_SlowWrite(redis.cache, clock, seconds=late_after),
            serve_last_good_entries=True,
            last_good_clock=clock,
        )
        before: Final = self._team("t1", alias="before")

        with last_good_reads_started_at(clock()):
            await remember_last_good(earlier_pod, "team_id:t1", before)
            assert reading_pod.last_good_memory.set(
                "team_id:t1", CacheCodec.serialize(before), ends_at=clock() + LAST_GOOD_ENTRY_TTL_SECONDS
            )
            await remember_last_good(reading_pod, "team_id:t1", self._team("t1", alias="after"))

        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is None
        assert reading_pod.last_good_memory.get("team_id:t1") is None

    @pytest.mark.asyncio
    async def test_a_set_redis_runs_100_s_late_is_kept_and_both_copies_end_at_the_reads_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The SET carries the end the store took before sending it, as an absolute expiry, so a SET that Redis runs
        100 s later, far inside the time a tombstone stands, is kept: Redis still drops the entry at the read's
        deadline, the instant the memory line ends, and not 100 s past it."""
        clock: Final = _Clock()
        redis: Final = fake_redis_cache()
        cache: Final = _cache_on_driven_clocks(monkeypatch, clock, _SlowWrite(redis.cache, clock, seconds=100))
        expire_at: Final = math.floor(clock.wall() + LAST_GOOD_ENTRY_TTL_SECONDS)

        with last_good_reads_started_at(clock()):
            await remember_last_good(cache, "team_id:t1", self._team("t1"))

        assert await redis.cache.init_async_client().expiretime("last_good:team_id:t1") == expire_at
        clock.move_to_wall(expire_at - 0.01)
        assert await _held_in(redis.cache, cache, "team_id:t1") == (True, True)
        clock.move_to_wall(expire_at + 0.01)
        assert await _held_in(redis.cache, cache, "team_id:t1") == (False, False)

    @pytest.mark.asyncio
    async def test_a_late_set_whose_check_and_undo_both_fail_leaves_an_entry_that_ends_at_the_reads_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Redis runs the SET 100 s late and is lost right after it, so the tombstone check after the write and the
        undo both fail, and Redis comes back later: the entry left in Redis ends at the read's deadline, as its
        absolute expiry says, not 100 s past it, and this pod keeps no memory line."""
        clock: Final = _Clock()
        redis: Final = fake_redis_cache()
        cache: Final = _cache_on_driven_clocks(
            monkeypatch, clock, _LostRightAfterASlowWrite(redis.cache, redis.server, clock, seconds=100)
        )
        expire_at: Final = math.floor(clock.wall() + LAST_GOOD_ENTRY_TTL_SECONDS)

        with last_good_reads_started_at(clock()):
            await remember_last_good(cache, "team_id:t1", self._team("t1"))
        redis.server.connected = True

        clock.move_to_wall(expire_at - 0.01)
        assert await _held_in(redis.cache, cache, "team_id:t1") == (True, False)
        clock.move_to_wall(expire_at + 0.01)
        assert await _held_in(redis.cache, cache, "team_id:t1") == (False, False)

    @pytest.mark.asyncio
    async def test_a_store_whose_end_passed_before_it_began_sends_no_set_and_keeps_the_earlier_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A 10 s expiry, and a read that began 9.5 s ago a quarter second into a Unix second: rounded down to a whole
        second, its end has passed when the store begins. No SET is sent and nothing goes into memory, so the entry an
        earlier read kept stays in Redis and in memory until its own expiry."""
        monkeypatch.setattr(  # test-quality-ok: the store reads the entry's expiry from this module's constant
            "litellm.proxy.common_utils.user_api_key_cache.LAST_GOOD_ENTRY_TTL_SECONDS", 10
        )
        clock: Final = _Clock()
        redis: Final = _SlowWrite(fake_redis_cache().cache, clock, seconds=0.0)
        cache: Final = _cache_on_driven_clocks(monkeypatch, clock, redis)
        before: Final = self._team("t1", alias="before")
        with last_good_reads_started_at(clock()):
            await remember_last_good(cache, "team_id:t1", before)
        redis.set_keys.clear()

        with last_good_reads_started_at(clock() - 9.5):
            await remember_last_good(cache, "team_id:t1", self._team("t1", alias="after"))

        assert redis.set_keys == []
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == before
        assert cache.last_good_memory.get("team_id:t1") == CacheCodec.serialize(before)

    @pytest.mark.asyncio
    async def test_the_memory_line_and_the_redis_entry_end_at_the_same_instant(self, monkeypatch: pytest.MonkeyPatch):
        """The wall clock is a quarter second into a Unix second, so the end, rounded down to a whole second, comes a
        quarter second before the read's full expiry: both copies are there just before that instant and both are gone
        just after it."""
        clock: Final = _Clock()
        redis: Final = fake_redis_cache()
        cache: Final = _cache_on_driven_clocks(monkeypatch, clock, redis.cache)
        expire_at: Final = math.floor(clock.wall() + LAST_GOOD_ENTRY_TTL_SECONDS)

        with last_good_reads_started_at(clock()):
            await remember_last_good(cache, "team_id:t1", self._team("t1"))

        clock.move_to_wall(expire_at - 0.01)
        assert await _held_in(redis.cache, cache, "team_id:t1") == (True, True)
        clock.move_to_wall(expire_at + 0.01)
        assert await _held_in(redis.cache, cache, "team_id:t1") == (False, False)

    @pytest.mark.asyncio
    async def test_memory_line_answers_while_redis_is_away(self):
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"))

        redis.server.connected = False

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")

    @pytest.mark.asyncio
    async def test_a_redis_that_answers_without_the_entry_wins_over_this_pods_memory(self):
        """Another pod forgot the entry (a delete or a block): Redis answers that it is gone, so the memory copy
        this pod still holds, because the broadcast has not reached it yet, is not served."""
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"))

        await redis.cache.init_async_client().delete("last_good:team_id:t1")

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_forget_removes_the_entry_from_redis_and_memory(self):
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"))

        await forget_last_good(cache, "team_id:t1")

        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is None
        redis.server.connected = False
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_forget_with_redis_away_still_clears_memory_and_does_not_raise(self):
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"))
        redis.server.connected = False

        await forget_last_good(cache, "team_id:t1")

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_remember_with_redis_away_keeps_the_memory_line(self):
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        redis.server.connected = False

        await remember_last_good(cache, "team_id:t1", self._team("t1"))

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")

    @pytest.mark.asyncio
    async def test_without_redis_the_memory_line_is_the_only_line(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"))

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")
        await forget_last_good(cache, "team_id:t1")
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_recalled_entry_is_a_fresh_object_each_time(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"))

        first = await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj)
        assert first is not None
        first.models.append("leaked")

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")

    @pytest.mark.asyncio
    async def test_a_recalled_entry_is_recorded_in_the_open_scope(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"))

        with last_good_auth_entries_scope() as scope:
            await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj)

        assert scope.served == ("LiteLLM_TeamTableCachedObj",)

    @pytest.mark.asyncio
    async def test_a_recall_that_finds_no_entry_records_nothing(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)

        with last_good_auth_entries_scope() as scope:
            assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

        assert scope.served == ()

    @pytest.mark.asyncio
    async def test_a_recall_in_a_gather_child_is_recorded_in_the_parents_scope(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"))
        await remember_last_good(cache, HASHED_TOKEN, UserAPIKeyAuth(token=HASHED_TOKEN))

        with last_good_auth_entries_scope() as scope:
            await asyncio.gather(
                recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj),
                recall_last_good(cache, HASHED_TOKEN, UserAPIKeyAuth),
            )

        assert sorted(scope.served) == ["LiteLLM_TeamTableCachedObj", "UserAPIKeyAuth"]

    @pytest.mark.asyncio
    async def test_a_recall_with_no_open_scope_records_nowhere(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"))
        with last_good_auth_entries_scope() as closed:
            pass

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")
        assert closed.served == ()

    @pytest.mark.asyncio
    async def test_concurrent_requests_record_into_their_own_scopes(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"))

        async def request(recalls: bool) -> tuple[str, ...]:
            with last_good_auth_entries_scope() as scope:
                await asyncio.sleep(0)
                if recalls:
                    await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj)
                await asyncio.sleep(0)
            return scope.served

        assert await asyncio.gather(request(True), request(False)) == [("LiteLLM_TeamTableCachedObj",), ()]

    @pytest.mark.asyncio
    async def test_an_auth_object_read_back_from_the_cache_carries_no_mark(self):
        from litellm.proxy.auth.auth_checks import copy_user_api_key_auth_for_cache

        cache = UserApiKeyCache(redis_cache=fake_redis_cache().cache)
        marked = UserAPIKeyAuth(token=HASHED_TOKEN)
        marked.auth_entries = "last_good"
        await cache.async_set_cache(HASHED_TOKEN, marked, model_type=UserAPIKeyAuth)
        cache.in_memory_cache_for(HASHED_TOKEN).delete_cache(HASHED_TOKEN)

        from_redis = await cache.async_get_cache(key=HASHED_TOKEN, model_type=UserAPIKeyAuth)

        assert from_redis is not None
        assert (from_redis.token, from_redis.auth_entries) == (HASHED_TOKEN, None)
        assert copy_user_api_key_auth_for_cache(marked).auth_entries is None

    def test_memory_line_drops_the_least_recently_written_entry_first(self):
        memory = LastGoodMemory(max_entries=2, clock=lambda: 0.0)
        memory.set("a", {"v": 1}, ends_at=600.0)
        memory.set("b", {"v": 2}, ends_at=600.0)
        memory.set("a", {"v": 3}, ends_at=600.0)
        memory.set("c", {"v": 4}, ends_at=600.0)

        assert (memory.get("a"), memory.get("b"), memory.get("c")) == ({"v": 3}, None, {"v": 4})

    @pytest.mark.asyncio
    async def test_a_tombstone_refuses_to_store_the_entry_again(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await forget_last_good(cache, "team_id:t1", tombstone=True)

        await remember_last_good(cache, "team_id:t1", self._team("t1"))

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_another_pods_tombstone_refuses_the_entry_in_redis_and_in_this_pods_memory(self):
        redis = fake_redis_cache()
        deleting_pod = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        reading_pod = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await forget_last_good(deleting_pod, "team_id:t1", tombstone=True)

        await remember_last_good(reading_pod, "team_id:t1", self._team("t1"))

        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is None
        redis.server.connected = False
        assert await recall_last_good(reading_pod, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_a_write_that_lands_as_the_tombstone_appears_is_undone(self):
        """The delete runs between the read's tombstone check and its write: the read checks again after
        writing and takes its write back."""
        redis = fake_redis_cache()
        deleting_pod = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        reading_pod = UserApiKeyCache(
            redis_cache=_DeleteLandsDuringWrite(redis.cache, deleting_pod), serve_last_good_entries=True
        )

        await remember_last_good(reading_pod, "team_id:t1", self._team("t1"))

        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is None
        assert reading_pod.last_good_memory.get("team_id:t1") is None
        assert await recall_last_good(reading_pod, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_a_write_that_finds_a_tombstone_keeps_no_memory_line_when_it_cannot_be_taken_back(self):
        """Taking the write back is best effort: the entry left in Redis ends at its expiry, and the memory line
        does not keep the value the tombstone refused."""
        redis = fake_redis_cache()
        deleting_pod = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        reading_pod = UserApiKeyCache(
            redis_cache=_UndoFailsAfterADeleteLandsDuringWrite(redis.cache, deleting_pod), serve_last_good_entries=True
        )

        await remember_last_good(reading_pod, "team_id:t1", self._team("t1"))

        assert reading_pod.last_good_memory.get("team_id:t1") is None
        assert 0 < await redis.cache.init_async_client().ttl("last_good:team_id:t1") <= LAST_GOOD_ENTRY_TTL_SECONDS

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing", ["write", "tombstone check"])
    async def test_a_store_that_fails_once_its_write_began_takes_it_back_and_keeps_no_memory_line(
        self, failing: Literal["write", "tombstone check"]
    ):
        """The write may have landed after a tombstone the store could not see, so neither Redis nor the memory
        line may keep the value it was writing."""
        redis = fake_redis_cache()
        reading_pod_redis: Final = _FailsAfterTheWrite(redis.cache, failing=failing)
        cache = UserApiKeyCache(redis_cache=reading_pod_redis, serve_last_good_entries=True)

        await remember_last_good(cache, "team_id:t1", self._team("t1"))

        assert reading_pod_redis.wrote
        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is None
        assert cache.last_good_memory.get("team_id:t1") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "refusal", ["tombstone before the write", "tombstone after the write", "write fails", "tombstone check fails"]
    )
    async def test_a_refused_store_drops_the_copy_this_pod_already_kept(self, refusal: str):
        """This pod kept the team from an earlier read. A store that is then refused, whatever the reason, may be
        refused because another pod revoked the team, so the earlier copy must not stay to be served by this pod
        when Redis and the database are both unreachable."""
        redis = fake_redis_cache()
        deleting_pod = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        reading_pod_redis: Final[dict[str, RedisCache]] = {
            "tombstone before the write": redis.cache,
            "tombstone after the write": _DeleteLandsDuringWrite(redis.cache, deleting_pod),
            "write fails": _FailsAfterTheWrite(redis.cache, failing="write"),
            "tombstone check fails": _FailsAfterTheWrite(redis.cache, failing="tombstone check"),
        }
        reading_pod = UserApiKeyCache(redis_cache=reading_pod_redis[refusal], serve_last_good_entries=True)
        # The copy an earlier store left in this pod's memory, as it does when Redis cannot answer that store
        earlier_copy: Final = self._team("t1", alias="before")
        assert reading_pod.last_good_memory.set(
            "team_id:t1",
            CacheCodec.serialize(earlier_copy),
            ends_at=reading_pod.last_good_clock() + LAST_GOOD_ENTRY_TTL_SECONDS,
        )
        if refusal == "tombstone before the write":
            await forget_last_good(deleting_pod, "team_id:t1", tombstone=True)

        await remember_last_good(reading_pod, "team_id:t1", self._team("t1", alias="after"))

        assert reading_pod.last_good_memory.get("team_id:t1") is None

    def test_a_tombstone_lapses(self):
        now = [100.0]
        memory = LastGoodMemory(max_entries=10, clock=lambda: now[0])
        memory.tombstone("a", ttl_seconds=600)
        assert memory.set("a", {"v": 1}, ends_at=now[0] + 600) is False

        now[0] += 601

        assert memory.set("a", {"v": 1}, ends_at=now[0] + 600) is True
        assert memory.get("a") == {"v": 1}

    @pytest.mark.asyncio
    async def test_revoke_tombstones_and_drops_the_entry_so_no_read_can_store_it_again(self):
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        other_pod = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"))

        await revoke_last_good(cache, ("team_id:t1",))
        await remember_last_good(other_pod, "team_id:t1", self._team("t1"))

        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is None
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_revoke_refuses_with_503_and_logs_an_error_when_redis_cannot_take_it(self, caplog):
        from fastapi import HTTPException

        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await remember_last_good(cache, "hashed-token-1", UserAPIKeyAuth(token="hashed-token-1"))
        redis.server.connected = False

        with pytest.raises(HTTPException) as refused:
            await revoke_last_good(cache, ("hashed-token-1",))

        assert refused.value.status_code == 503
        assert any(r.levelname == "ERROR" and "hashed-token-1" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_revoke_does_nothing_while_the_setting_is_off(self):
        redis = fake_redis_cache()
        redis.server.connected = False

        cache = UserApiKeyCache(redis_cache=redis.cache)
        await revoke_last_good(cache, ("hashed-token-1",))

        assert cache.last_good_memory.tombstoned("hashed-token-1") is False

    @pytest.mark.asyncio
    async def test_revoke_skips_a_key_this_request_revoked_while_its_tombstone_stands(self):
        """A request revokes its whole set before its first write, so a revocation nested after that write skips Redis
        for a key the request revoked less than the tombstone's time minus 5 s ago; any other key, or one revoked
        longer ago, goes to Redis again and is refused while Redis is away."""
        from fastapi import HTTPException

        from litellm.constants import LAST_GOOD_TOMBSTONE_TTL_SECONDS

        revoked_token: Final = "hashed-token-revoked-before-the-write"
        other_token: Final = "hashed-token-never-revoked"
        clock: Final = _Clock()
        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=clock)
        await revoke_last_good(cache, (revoked_token,))
        redis.server.connected = False

        clock.now += LAST_GOOD_TOMBSTONE_TTL_SECONDS - 6
        await revoke_last_good(cache, (revoked_token,))
        with pytest.raises(HTTPException) as never_revoked:
            await revoke_last_good(cache, (other_token,))
        clock.now += 1
        with pytest.raises(HTTPException) as revoked_long_ago:
            await revoke_last_good(cache, (revoked_token,))

        assert never_revoked.value.status_code == 503
        assert revoked_long_ago.value.status_code == 503

    @pytest.mark.asyncio
    async def test_a_later_request_revokes_a_key_an_earlier_request_revoked_again(self):
        """The record of revoked keys belongs to one request: the next request, in a task of its own, sends the same
        key to Redis again and is refused while Redis is away."""
        from fastapi import HTTPException

        token: Final = "hashed-token-revoked-by-the-earlier-request"
        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=_Clock())
        await asyncio.create_task(revoke_last_good(cache, (token,)))
        redis.server.connected = False

        with pytest.raises(HTTPException) as refused:
            await asyncio.create_task(revoke_last_good(cache, (token,)))

        assert refused.value.status_code == 503

    @pytest.mark.asyncio
    async def test_a_task_a_request_starts_after_revoking_does_not_skip_by_its_record(self):
        """A task the request starts copies the request's context, record included, yet only the request's own task
        skips by it: the started task sends the key to Redis again and is refused while Redis is away."""
        from fastapi import HTTPException

        token: Final = "hashed-token-revoked-before-the-task-started"
        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=_Clock())

        async def request_starting_a_task() -> asyncio.Task[None]:
            await revoke_last_good(cache, (token,))
            redis.server.connected = False
            return asyncio.create_task(revoke_last_good(cache, (token,)))

        started_task: Final = await asyncio.create_task(request_starting_a_task())
        with pytest.raises(HTTPException) as refused:
            await started_task

        assert refused.value.status_code == 503

    @pytest.mark.asyncio
    async def test_a_revocation_checks_its_record_when_each_key_comes_up_not_when_the_call_began(self):
        """K was revoked earlier. A later call that revokes J then K begins 1 s inside the time its record allows, but
        J's Redis commands take 2 s, so K's tombstone may lapse before the request's write: K goes to Redis again."""
        clock: Final = _Clock()
        redis: Final = _SlowWrite(fake_redis_cache().cache, clock, seconds=0.0)
        cache: Final = UserApiKeyCache(redis_cache=redis, serve_last_good_entries=True, last_good_clock=clock)
        await revoke_last_good(cache, ("hashed-k",))
        clock.now += LAST_GOOD_TOMBSTONE_TTL_SECONDS - LAST_GOOD_LATE_STORE_MARGIN_SECONDS - 1
        redis.seconds = 2.0
        redis.set_keys.clear()

        await revoke_last_good(cache, ("hashed-j", "hashed-k"))

        assert redis.set_keys == ["last_good_tombstone:hashed-j", "last_good_tombstone:hashed-k"]

    @pytest.mark.asyncio
    async def test_a_second_auth_run_in_the_same_task_starts_with_an_empty_record_of_revoked_keys(self):
        """One task can handle two requests: where the second one's auth begins (``start_last_good_reads``, the first
        statement of user_api_key_auth and of every MCP request), the first one's record is dropped, so the second
        sends the same key to Redis again and is refused while Redis is away."""
        from fastapi import HTTPException

        token: Final = "hashed-token-revoked-by-the-earlier-request"
        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=_Clock())
        await revoke_last_good(cache, (token,))
        redis.server.connected = False

        start_last_good_reads()
        with pytest.raises(HTTPException) as refused:
            await revoke_last_good(cache, (token,))

        assert refused.value.status_code == 503

    @pytest.mark.asyncio
    async def test_a_write_paired_with_its_revocation_runs_only_once_the_revocation_is_recorded(self):
        """``revoke_last_good_then`` revokes, then writes: with Redis away and the key not revoked by this task, the 503
        leaves the write unmade; a key this task revoked moments ago needs no Redis, so its write runs; no keys, just
        the write."""
        from fastapi import HTTPException

        written: Final[list[str]] = []  # mutable-ok: records each write that ran

        async def write(name: str) -> str:
            written.append(name)
            return name

        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=_Clock())
        await revoke_last_good(cache, ("hashed-revoked",))
        redis.server.connected = False

        with pytest.raises(HTTPException) as refused:
            await revoke_last_good_then(cache, ("hashed-never-revoked",), lambda: write("refused"))
        kept: Final = await revoke_last_good_then(cache, ("hashed-revoked",), lambda: write("revoked before"))
        bare: Final = await revoke_last_good_then(cache, (), lambda: write("no keys"))

        assert refused.value.status_code == 503
        assert (kept, bare, written) == ("revoked before", "no keys", ["revoked before", "no keys"])

    @pytest.mark.asyncio
    async def test_an_exact_delete_revokes_and_deletes_a_row_added_after_its_read_in_a_next_round(self):
        """The first read finds a, the second b, the third nothing: a is deleted alone, then b, each right after its own
        revocation, so b, added after the first read, is never deleted unrevoked."""
        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=_Clock())
        answers: Final = [("a",), ("b",)]  # mutable-ok: each read takes the next answer
        deletes: Final[list[tuple[tuple[str, ...], bool]]] = []  # mutable-ok: each delete and its tombstones

        async def read() -> tuple[str, ...]:
            return answers.pop(0) if answers else ()

        async def delete(rows: Sequence[str]) -> None:
            stood: Final = all([await redis.cache.async_get_cache(f"last_good_tombstone:{row}") for row in rows])
            deletes.append((tuple(rows), stood))

        deleted: Final = await revoke_and_delete_read_rows(cache, read, tuple, delete)

        assert deleted == ("a", "b")
        assert deletes == [(("a",), True), (("b",), True)]

    @pytest.mark.asyncio
    async def test_an_exact_delete_whose_late_row_cannot_be_revoked_answers_503_and_leaves_it(self):
        """The first round's row is deleted; Redis is lost before the row added since can be revoked, so the 503
        leaves it undeleted."""
        from fastapi import HTTPException

        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=_Clock())
        answers: Final = [("read first",), ("late",)]  # mutable-ok: each read takes the next answer
        deletes: Final[list[tuple[str, ...]]] = []  # mutable-ok: records each delete in order

        async def read() -> tuple[str, ...]:
            return answers.pop(0)

        async def delete(rows: Sequence[str]) -> None:
            deletes.append(tuple(rows))
            redis.server.connected = False

        with pytest.raises(HTTPException) as refused:
            await revoke_and_delete_read_rows(cache, read, tuple, delete)

        assert refused.value.status_code == 503
        assert deletes == [("read first",)]

    @pytest.mark.asyncio
    async def test_rows_that_keep_appearing_refuse_with_409_after_the_last_round_and_stay(self):
        """Every read finds a new row: after LAST_GOOD_EXACT_DELETE_ROUNDS deletes the request answers 409 and the row
        of the last read is neither revoked nor deleted."""
        from fastapi import HTTPException

        from litellm.proxy.common_utils.user_api_key_cache import LAST_GOOD_EXACT_DELETE_ROUNDS

        cache: Final = UserApiKeyCache(serve_last_good_entries=True, last_good_clock=_Clock())
        reads: Final = [0]  # mutable-ok: counts the reads
        deletes: Final[list[tuple[str, ...]]] = []  # mutable-ok: records each delete in order

        async def read() -> tuple[str, ...]:
            reads[0] += 1
            return (f"row-{reads[0]}",)

        async def delete(rows: Sequence[str]) -> None:
            deletes.append(tuple(rows))

        with pytest.raises(HTTPException) as refused:
            await revoke_and_delete_read_rows(cache, read, tuple, delete)

        last_read: Final = f"row-{LAST_GOOD_EXACT_DELETE_ROUNDS + 1}"
        assert refused.value.status_code == 409
        assert deletes == [(f"row-{n}",) for n in range(1, LAST_GOOD_EXACT_DELETE_ROUNDS + 1)]
        assert cache.last_good_memory.tombstoned(last_read) is False
