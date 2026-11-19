import asyncio
import hashlib
import json
import logging
from typing import Any, Final

import pytest
from fakeredis import FakeServer
from fastapi import HTTPException
from redis.asyncio.client import Pipeline

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import (
    DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    LAST_GOOD_STORE_MARGIN_SECONDS,
    LAST_GOOD_TOMBSTONE_TTL_SECONDS,
    PROXY_DB_LOOKUP_DEADLINE_SECONDS,
)
from litellm.models.team import LiteLLM_TeamTableCachedObj
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import (
    LastGoodMemory,
    UserApiKeyCache,
    end_user_cache_key,
    forget_last_good,
    get_management_object_ttl,
    is_user_key_cache_key,
    last_good_auth_entries_scope,
    last_good_read_start,
    rearm_last_good,
    recall_last_good,
    remember_last_good,
    revoke_last_good,
)
from litellm.proxy.proxy_server import UserAPIKeyCacheTTLEnum
from tests.unit.caching.fake_redis_cache import fake_redis_cache

HASHED_TOKEN = hashlib.sha256(b"sk-lit7563-hot-key").hexdigest()

# A store whose last tombstone read ends later than this after its database read began is refused.
_STORE_BOUND: Final = PROXY_DB_LOOKUP_DEADLINE_SECONDS + LAST_GOOD_STORE_MARGIN_SECONDS


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

    async def async_set_cache_or_raise(self, key: str, value: object, ttl: int | None = None) -> None:
        if key.startswith("last_good:"):
            await forget_last_good(self._deleting_pod, key.removeprefix("last_good:"), tombstone=True)
        await self._redis.async_set_cache_or_raise(key, value, ttl=ttl)

    async def async_delete_cache(self, key: str) -> None:
        await self._redis.async_delete_cache(key)


class _Clock:
    """A monotonic clock the test moves by hand, handed to ``UserApiKeyCache(last_good_clock=...)``."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _SlowRedis(RedisCache):
    """The storing pod's Redis: the clock moves on by ``seconds`` while it answers the ``slow_at``-th command of a
    store (1 the first tombstone read, 2 the entry write, 3 the last tombstone read); with ``lost`` that command then
    fails, the way a Redis that times out does."""

    def __init__(self, redis: RedisCache, clock: _Clock, slow_at: int, seconds: float, lost: bool = False) -> None:
        self._redis = redis
        self._clock = clock
        self._slow_at = slow_at
        self._seconds = seconds
        self._lost = lost
        self._commands = 0

    def _answered(self) -> None:
        self._commands += 1
        if self._commands != self._slow_at:
            return
        self._clock.now += self._seconds
        if self._lost:
            raise ConnectionError("Timeout reading from redis")

    async def async_get_cache_or_raise(self, key: str) -> object:
        value = await self._redis.async_get_cache_or_raise(key)
        self._answered()
        return value

    async def async_set_cache_or_raise(self, key: str, value: object, ttl: int | None = None) -> None:
        await self._redis.async_set_cache_or_raise(key, value, ttl=ttl)
        self._answered()

    async def async_delete_cache(self, key: str) -> None:
        await self._redis.async_delete_cache(key)


def _late_store_warned(caplog: pytest.LogCaptureFixture, cache_key: str) -> bool:
    return any(
        r.levelname == "WARNING" and cache_key in r.getMessage() and "slower than the bound" in r.getMessage()
        for r in caplog.records
    )


class _CountedRoundTrips:
    """The client a ``fake_redis_cache`` hands its ``RedisCache``, counting round trips: a command sent alone is one, a
    pipeline is one however many commands it carries. ``lose_redis_at`` ("pipeline" or "delete") takes Redis away as
    the first such round trip is sent, the way a connection lost part-way through a revocation does."""

    def __init__(self, cache: RedisCache, server: FakeServer, lose_redis_at: str | None = None) -> None:
        self.uncounted: Final = cache.init_async_client()
        self._server = server
        self._lose_redis_at = lose_redis_at
        self.round_trips = 0
        litellm.in_memory_llm_clients_cache.set_cache(
            key=cache._get_async_client_cache_key(),  # pyright: ignore[reportPrivateUsage]  # the cache's client key
            value=self,
        )

    def send(self, round_trip: str) -> None:
        self.round_trips += 1
        if round_trip == self._lose_redis_at:
            self._lose_redis_at = None
            self._server.connected = False

    async def get(self, name: str) -> object:
        self.send("get")
        return await self.uncounted.get(name)

    async def set(self, name: str, value: str, ex: int | None = None) -> object:
        self.send("set")
        return await self.uncounted.set(name, value, ex=ex)

    async def delete(self, *names: str) -> int:
        self.send("delete")
        return await self.uncounted.delete(*names)

    def pipeline(self, transaction: bool = True) -> "_CountedPipeline":
        return _CountedPipeline(self, self.uncounted.pipeline(transaction=transaction))


class _CountedPipeline:
    """A pipeline of ``_CountedRoundTrips``: queues like the real one and counts one round trip when sent."""

    def __init__(self, counter: _CountedRoundTrips, pipe: Pipeline) -> None:
        self._counter = counter
        self._pipe = pipe

    async def __aenter__(self) -> "_CountedPipeline":
        await self._pipe.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._pipe.__aexit__(*exc_info)

    def set(self, name: str, value: str, ex: int | None = None) -> None:
        self._pipe.set(name, value, ex=ex)

    async def execute(self, raise_on_error: bool = True) -> list[object]:
        self._counter.send("pipeline")
        return await self._pipe.execute(raise_on_error=raise_on_error)


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
    """The last good team or key entry: Redis with no expiry first, this pod's memory when Redis cannot answer."""

    @staticmethod
    def _team(team_id: str, alias: str = "alias") -> LiteLLM_TeamTableCachedObj:
        return LiteLLM_TeamTableCachedObj(team_id=team_id, team_alias=alias, models=["claude-opus-5-5"])

    @pytest.mark.asyncio
    async def test_remembered_entry_lives_in_redis_without_expiry(self):
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)

        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))

        client = redis.cache.init_async_client()
        assert await client.ttl("last_good:team_id:t1") == -1
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")

    @pytest.mark.asyncio
    async def test_memory_line_answers_while_redis_is_away(self):
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))

        redis.server.connected = False

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")

    @pytest.mark.asyncio
    async def test_a_redis_that_answers_without_the_entry_wins_over_this_pods_memory(self):
        """Another pod forgot the entry (a delete or a block): Redis answers that it is gone, so the memory copy
        this pod still holds, because the broadcast has not reached it yet, is not served."""
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))

        await redis.cache.init_async_client().delete("last_good:team_id:t1")

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_forget_removes_the_entry_from_redis_and_memory(self):
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))

        await forget_last_good(cache, "team_id:t1")

        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is None
        redis.server.connected = False
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_forget_with_redis_away_still_clears_memory_and_does_not_raise(self):
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))
        redis.server.connected = False

        await forget_last_good(cache, "team_id:t1")

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_remember_with_redis_away_keeps_the_memory_line(self):
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        redis.server.connected = False

        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")

    @pytest.mark.asyncio
    async def test_without_redis_the_memory_line_is_the_only_line(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")
        await forget_last_good(cache, "team_id:t1")
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_recalled_entry_is_a_fresh_object_each_time(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))

        first = await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj)
        assert first is not None
        first.models.append("leaked")

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")

    @pytest.mark.asyncio
    async def test_a_recalled_entry_is_recorded_in_the_open_scope(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))

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
        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))
        await remember_last_good(
            cache, HASHED_TOKEN, UserAPIKeyAuth(token=HASHED_TOKEN), read_started=last_good_read_start(cache)
        )

        with last_good_auth_entries_scope() as scope:
            await asyncio.gather(
                recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj),
                recall_last_good(cache, HASHED_TOKEN, UserAPIKeyAuth),
            )

        assert sorted(scope.served) == ["LiteLLM_TeamTableCachedObj", "UserAPIKeyAuth"]

    @pytest.mark.asyncio
    async def test_a_recall_with_no_open_scope_records_nowhere(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))
        with last_good_auth_entries_scope() as closed:
            pass

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")
        assert closed.served == ()

    @pytest.mark.asyncio
    async def test_concurrent_requests_record_into_their_own_scopes(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))

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
        memory = LastGoodMemory(max_entries=2)
        memory.set("a", {"v": 1})
        memory.set("b", {"v": 2})
        memory.set("a", {"v": 3})
        memory.set("c", {"v": 4})

        assert (memory.get("a"), memory.get("b"), memory.get("c")) == ({"v": 3}, None, {"v": 4})

    @pytest.mark.asyncio
    async def test_a_tombstone_refuses_to_store_the_entry_again(self):
        cache = UserApiKeyCache(serve_last_good_entries=True)
        await forget_last_good(cache, "team_id:t1", tombstone=True)

        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))

        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_another_pods_tombstone_refuses_the_entry_in_redis_and_in_this_pods_memory(self):
        redis = fake_redis_cache()
        deleting_pod = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        reading_pod = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await forget_last_good(deleting_pod, "team_id:t1", tombstone=True)

        await remember_last_good(
            reading_pod, "team_id:t1", self._team("t1"), read_started=last_good_read_start(reading_pod)
        )

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

        await remember_last_good(
            reading_pod, "team_id:t1", self._team("t1"), read_started=last_good_read_start(reading_pod)
        )

        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is None
        assert await recall_last_good(reading_pod, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    def test_a_tombstone_lapses(self):
        now = [100.0]
        memory = LastGoodMemory(max_entries=10, clock=lambda: now[0])
        memory.tombstone("a", ttl_seconds=600)
        assert memory.set("a", {"v": 1}) is False

        now[0] += 601

        assert memory.set("a", {"v": 1}) is True
        assert memory.get("a") == {"v": 1}

    @pytest.mark.asyncio
    async def test_revoke_tombstones_and_drops_the_entry_so_no_read_can_store_it_again(self):
        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        other_pod = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))

        await revoke_last_good(cache, ("team_id:t1",))
        await remember_last_good(
            other_pod, "team_id:t1", self._team("t1"), read_started=last_good_read_start(other_pod)
        )

        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is None
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_revoke_refuses_with_503_and_logs_an_error_when_redis_cannot_take_it(self, caplog):
        from fastapi import HTTPException

        redis = fake_redis_cache()
        cache = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        hashed_token: Final = "hashed-token-1"
        await remember_last_good(
            cache, hashed_token, UserAPIKeyAuth(token=hashed_token), read_started=last_good_read_start(cache)
        )
        redis.server.connected = False

        with pytest.raises(HTTPException) as refused:
            await revoke_last_good(cache, (hashed_token,))

        assert refused.value.status_code == 503
        assert any(r.levelname == "ERROR" and hashed_token in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_revoke_does_nothing_while_the_setting_is_off(self):
        redis = fake_redis_cache()
        redis.server.connected = False

        cache = UserApiKeyCache(redis_cache=redis.cache)
        await revoke_last_good(cache, ("hashed-token-1",))

        assert cache.last_good_memory.tombstoned("hashed-token-1") is False

    async def _remember_fifty_teams(self, cache: UserApiKeyCache) -> tuple[str, ...]:
        """Fifty team entries, the cache keys a revocation of a large team's keys hands ``revoke_last_good`` at once."""
        team_ids: Final = tuple(f"t{i}" for i in range(50))
        for team_id in team_ids:
            await remember_last_good(
                cache, f"team_id:{team_id}", self._team(team_id), read_started=last_good_read_start(cache)
            )
        return team_ids

    @pytest.mark.asyncio
    async def test_revoking_fifty_entries_takes_two_round_trips_and_leaves_every_tombstone_and_no_entry(self):
        """One pipeline writes every tombstone, then one DEL drops every entry, where a SET and a DEL per entry made the
        revocation a hundred serial round trips."""
        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        team_ids: Final = await self._remember_fifty_teams(cache)
        counted: Final = _CountedRoundTrips(redis.cache, redis.server)

        await revoke_last_good(cache, tuple(f"team_id:{team_id}" for team_id in team_ids))

        assert counted.round_trips == 2
        assert await counted.uncounted.exists(*(f"last_good_tombstone:team_id:{t}" for t in team_ids)) == 50
        assert 0 < await counted.uncounted.ttl("last_good_tombstone:team_id:t49") <= LAST_GOOD_TOMBSTONE_TTL_SECONDS
        assert await counted.uncounted.exists(*(f"last_good:team_id:{t}" for t in team_ids)) == 0
        assert all(cache.last_good_memory.tombstoned(f"team_id:{t}") for t in team_ids)

    @pytest.mark.asyncio
    async def test_revoking_fifty_entries_with_redis_down_answers_503_and_leaves_every_entry(self):
        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        team_ids: Final = await self._remember_fifty_teams(cache)
        redis.server.connected = False

        with pytest.raises(HTTPException) as refused:
            await revoke_last_good(cache, tuple(f"team_id:{team_id}" for team_id in team_ids))

        assert refused.value.status_code == 503
        redis.server.connected = True
        assert await redis.cache.init_async_client().exists(*(f"last_good:team_id:{t}" for t in team_ids)) == 50
        assert not any(cache.last_good_memory.tombstoned(f"team_id:{t}") for t in team_ids)

    @pytest.mark.asyncio
    async def test_a_tombstone_pipeline_redis_cannot_take_answers_503_before_any_entry_is_deleted(self):
        """The pipeline's failure reaches the caller, so no DEL runs: no entry is ever left without its tombstone."""
        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        team_ids: Final = await self._remember_fifty_teams(cache)
        counted: Final = _CountedRoundTrips(redis.cache, redis.server, lose_redis_at="pipeline")

        with pytest.raises(HTTPException) as refused:
            await revoke_last_good(cache, tuple(f"team_id:{team_id}" for team_id in team_ids))

        assert refused.value.status_code == 503
        redis.server.connected = True
        assert await counted.uncounted.exists(*(f"last_good:team_id:{t}" for t in team_ids)) == 50

    @pytest.mark.asyncio
    async def test_redis_lost_between_the_tombstones_and_the_deletes_answers_503_and_the_tombstones_refuse_entries(
        self,
    ):
        """Every tombstone is written before any entry is deleted, so a Redis lost in between leaves the entries, still
        those of the unchanged rows since the caller writes nothing after the 503, under tombstones that refuse a newer
        entry until they lapse."""
        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        other_pod: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        team_ids: Final = await self._remember_fifty_teams(cache)
        counted: Final = _CountedRoundTrips(redis.cache, redis.server, lose_redis_at="delete")

        with pytest.raises(HTTPException) as refused:
            await revoke_last_good(cache, tuple(f"team_id:{team_id}" for team_id in team_ids))

        assert refused.value.status_code == 503
        redis.server.connected = True
        assert await counted.uncounted.exists(*(f"last_good_tombstone:team_id:{t}" for t in team_ids)) == 50
        assert await counted.uncounted.exists(*(f"last_good:team_id:{t}" for t in team_ids)) == 50
        for team_id in team_ids:
            await remember_last_good(
                other_pod,
                f"team_id:{team_id}",
                self._team(team_id, alias="read after"),
                read_started=last_good_read_start(other_pod),
            )
        assert [
            await recall_last_good(other_pod, f"team_id:{team_id}", LiteLLM_TeamTableCachedObj) for team_id in team_ids
        ] == [self._team(team_id) for team_id in team_ids]
        assert all(other_pod.last_good_memory.get(f"team_id:{t}") is None for t in team_ids)

    @pytest.mark.asyncio
    async def test_a_store_already_beyond_the_bound_sends_nothing_to_redis_and_keeps_nothing(self, caplog):
        """The database answered later than PROXY_DB_LOOKUP_DEADLINE_SECONDS + LAST_GOOD_STORE_MARGIN_SECONDS after the
        read began: a write's tombstone may have lapsed meanwhile, so the store is refused before it writes anything."""
        caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")
        redis: Final = fake_redis_cache()
        clock: Final = _Clock()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=clock)
        read_started: Final = clock()
        clock.now += _STORE_BOUND + 1
        counted: Final = _CountedRoundTrips(redis.cache, redis.server)

        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=read_started)

        assert counted.round_trips == 0
        assert await counted.uncounted.get("last_good:team_id:t1") is None
        assert cache.last_good_memory.get("team_id:t1") is None
        assert _late_store_warned(caplog, "team_id:t1")

    @pytest.mark.asyncio
    async def test_a_store_that_ends_inside_the_bound_is_kept_in_redis_and_memory(self):
        redis: Final = fake_redis_cache()
        clock: Final = _Clock()
        read_started: Final = clock()
        cache: Final = UserApiKeyCache(
            redis_cache=_SlowRedis(redis.cache, clock, slow_at=3, seconds=_STORE_BOUND - 1),
            serve_last_good_entries=True,
            last_good_clock=clock,
        )

        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=read_started)

        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is not None
        assert cache.last_good_memory.get("team_id:t1") is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("slow_at", [1, 2, 3], ids=["first_tombstone_read", "entry_write", "last_tombstone_read"])
    async def test_a_store_redis_carries_beyond_the_bound_is_undone_and_not_kept_in_memory(self, caplog, slow_at: int):
        """Redis answers one command of the store so late that its last tombstone read ends beyond the bound: the
        write is taken back as for a tombstone, and the memory line does not keep the entry either."""
        caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")
        redis: Final = fake_redis_cache()
        clock: Final = _Clock()
        read_started: Final = clock()
        cache: Final = UserApiKeyCache(
            redis_cache=_SlowRedis(redis.cache, clock, slow_at=slow_at, seconds=_STORE_BOUND + 1),
            serve_last_good_entries=True,
            last_good_clock=clock,
        )

        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=read_started)

        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is None
        assert cache.last_good_memory.get("team_id:t1") is None
        assert _late_store_warned(caplog, "team_id:t1")

    @pytest.mark.asyncio
    async def test_a_store_that_falls_back_to_memory_beyond_the_bound_keeps_nothing(self, caplog):
        """Redis times out on the first tombstone read after the bound has passed: the memory line, which keeps the
        entry when Redis cannot answer, refuses it as well."""
        caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")
        redis: Final = fake_redis_cache()
        clock: Final = _Clock()
        read_started: Final = clock()
        cache: Final = UserApiKeyCache(
            redis_cache=_SlowRedis(redis.cache, clock, slow_at=1, seconds=_STORE_BOUND + 1, lost=True),
            serve_last_good_entries=True,
            last_good_clock=clock,
        )

        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=read_started)

        assert cache.last_good_memory.get("team_id:t1") is None
        assert await redis.cache.init_async_client().get("last_good:team_id:t1") is None
        assert _late_store_warned(caplog, "team_id:t1")

    @pytest.mark.asyncio
    async def test_rearm_tombstones_and_drops_each_entry_so_no_pod_can_store_it_again(self):
        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        other_pod: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        cache_keys: Final = ("team_id:t1", "team_id:t2")
        for team_id in ("t1", "t2"):
            await remember_last_good(
                cache, f"team_id:{team_id}", self._team(team_id), read_started=last_good_read_start(cache)
            )

        await rearm_last_good(cache, cache_keys)

        client: Final = redis.cache.init_async_client()
        assert await client.exists(*(f"last_good:{k}" for k in cache_keys)) == 0
        assert await client.exists(*(f"last_good_tombstone:{k}" for k in cache_keys)) == 2
        assert 0 < await client.ttl("last_good_tombstone:team_id:t2") <= LAST_GOOD_TOMBSTONE_TTL_SECONDS
        assert all(cache.last_good_memory.tombstoned(k) for k in cache_keys)
        assert all(cache.last_good_memory.get(k) is None for k in cache_keys)
        await remember_last_good(
            other_pod, "team_id:t1", self._team("t1"), read_started=last_good_read_start(other_pod)
        )
        assert await recall_last_good(other_pod, "team_id:t1", LiteLLM_TeamTableCachedObj) is None

    @pytest.mark.asyncio
    async def test_rearm_after_a_slow_write_gives_the_revocations_tombstones_their_full_time_again(self):
        """The write took nearly all the tombstone time the revocation before it wrote: after the re-arm, the tombstone
        outlives that first one, in Redis and in this pod's memory."""
        redis: Final = fake_redis_cache()
        clock: Final = _Clock()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=clock)
        client: Final = redis.cache.init_async_client()
        await revoke_last_good(cache, ("team_id:t1",))
        clock.now += LAST_GOOD_TOMBSTONE_TTL_SECONDS - 1
        await client.expire("last_good_tombstone:team_id:t1", 1)

        await rearm_last_good(cache, ("team_id:t1",))

        clock.now += 2
        assert await client.ttl("last_good_tombstone:team_id:t1") > 1
        assert cache.last_good_memory.tombstoned("team_id:t1")

    @pytest.mark.asyncio
    async def test_rearm_with_redis_down_does_not_raise_and_leaves_the_memory_tombstones(self, caplog):
        """The write is already committed, so a Redis that cannot take the re-arm is logged, not raised."""
        caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")
        redis: Final = fake_redis_cache()
        cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
        await remember_last_good(cache, "team_id:t1", self._team("t1"), read_started=last_good_read_start(cache))
        redis.server.connected = False

        await rearm_last_good(cache, ("team_id:t1",))

        assert cache.last_good_memory.tombstoned("team_id:t1")
        assert cache.last_good_memory.get("team_id:t1") is None
        assert any(
            r.levelname == "WARNING" and "re-arm" in r.getMessage() and "team_id:t1" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_rearm_does_nothing_while_the_setting_is_off(self):
        redis: Final = fake_redis_cache()
        redis.server.connected = False
        cache: Final = UserApiKeyCache(redis_cache=redis.cache)

        await rearm_last_good(cache, ("team_id:t1",))

        assert cache.last_good_memory.tombstoned("team_id:t1") is False
