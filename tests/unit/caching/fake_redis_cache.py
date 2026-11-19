import uuid
from dataclasses import dataclass

from fakeredis import FakeRedis as FakeSyncRedis
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

import litellm
from litellm.caching.redis_cache import RedisCache


@dataclass(frozen=True, slots=True)
class FakeRedisCache:
    """A real ``RedisCache`` over a fakeredis server; ``server.connected = False`` makes Redis unreachable."""

    cache: RedisCache
    server: FakeServer


def fake_redis_cache(namespace: str | None = None) -> FakeRedisCache:
    server = FakeServer()
    cache = RedisCache(
        host=f"fake-redis-{uuid.uuid4().hex}",
        namespace=namespace,
        connection_pool=FakeSyncRedis(server=server).connection_pool,
    )
    litellm.in_memory_llm_clients_cache.set_cache(
        key=cache._get_async_client_cache_key(), value=FakeRedis(server=server)
    )
    return FakeRedisCache(cache=cache, server=server)
