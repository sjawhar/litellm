import hashlib
import json
from typing import Any

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.models.team import LiteLLM_TeamTableCachedObj
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import (
    LastGoodMemory,
    UserApiKeyCache,
    end_user_cache_key,
    forget_last_good,
    get_management_object_ttl,
    is_user_key_cache_key,
    recall_last_good,
    remember_last_good,
    revoke_last_good,
)
from litellm.proxy.proxy_server import UserAPIKeyCacheTTLEnum
from tests.unit.caching.fake_redis_cache import fake_redis_cache

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

    def __init__(self, redis: RedisCache, deleting_pod: UserApiKeyCache) -> None:  # noqa: super().__init__ skipped intentionally
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

    def __init__(self):  # noqa: super().__init__ skipped intentionally
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

        await remember_last_good(cache, "team_id:t1", self._team("t1"))

        client = redis.cache.init_async_client()
        assert await client.ttl("last_good:team_id:t1") == -1
        assert await recall_last_good(cache, "team_id:t1", LiteLLM_TeamTableCachedObj) == self._team("t1")

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

        await revoke_last_good(UserApiKeyCache(redis_cache=redis.cache), ("hashed-token-1",))
