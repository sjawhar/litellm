import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Final, TypeAlias

import pytest
from fakeredis import FakeRedis as FakeSyncRedis
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from redis.asyncio.client import Pipeline

import litellm
from litellm.caching.redis_cache import RedisCache

# One command of a pipeline as redis-py stacks it for ``execute()``: its arguments, then its options
_PipelineCommand: TypeAlias = tuple[tuple[str | bytes, ...], Mapping[str, Any]]


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


def drive_fake_redis_time(monkeypatch: pytest.MonkeyPatch, clock: Callable[[], float]) -> None:
    """Fake Redis runs each command, its expiries and TTLs, at ``clock``'s Unix time: tests move it, never sleep."""
    monkeypatch.setattr(  # test-quality-ok: fakeredis reads each command's time through this module's time.time
        "fakeredis._basefakesocket.time", SimpleNamespace(time=clock)
    )


class ScriptedPipelines:
    """Scripts the ``execute()`` of every pipeline the async client of ``cache``, a ``RedisCache`` over fake Redis,
    opens: the one call that sends a pipeline's commands to the server and returns once every answer is in.

    A SET of a key in ``refused_sets`` reaches the server with an expiry of 0 seconds, which Redis refuses with an error
    reply. The server still runs every other command of the pipeline, and ``execute()``, with redis-py's default
    ``raise_on_error=True``, raises that error once every answer is in, as it does for any command the server refuses in
    a pipeline that is not a transaction. Rebind ``refused_sets`` to change it between calls.

    Once the server has answered an execution with no error, ``executions`` counts it, from 1, and ``after_execution``
    is awaited with that count: a test moves its clock there, or does what another client does meanwhile. A pipeline
    executed inside ``after_execution`` is sent as it is and not counted. Plain commands are never scripted."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cache: RedisCache,
        *,
        refused_sets: frozenset[str] = frozenset(),
        after_execution: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        client: Final = cache.init_async_client()
        if not isinstance(client, FakeRedis):
            raise TypeError(f"ScriptedPipelines needs a RedisCache over fake Redis, not over {type(client).__name__}")
        self._client: Final = client
        self._after_execution: Final = after_execution
        self.in_after_execution = False
        self.refused_sets = refused_sets
        self.executions = 0
        monkeypatch.setattr(client, "pipeline", self._pipeline)

    def _pipeline(self, transaction: bool = True, shard_hint: str | None = None) -> Pipeline:
        return _ScriptedPipeline(self, self._client, transaction, shard_hint)

    def as_sent(self, command: _PipelineCommand) -> _PipelineCommand:
        """``command`` as the server gets it: a SET of a refused key with an expiry of 0 seconds."""
        args: Final = command[0]
        if args[0] != "SET" or args[1] not in self.refused_sets:
            return command
        return ("SET", args[1], args[2], "EX", "0"), command[1]

    async def answered(self) -> None:
        """Count an execution the server answered with no error, then run ``after_execution`` for it."""
        self.executions += 1
        if self._after_execution is None:
            return
        self.in_after_execution = True
        try:
            await self._after_execution(self.executions)
        finally:
            self.in_after_execution = False


class _ScriptedPipeline(Pipeline):
    """redis-py's pipeline over the fake client's connections, built as ``Redis.pipeline`` builds it, whose
    ``execute()`` goes through ``script``."""

    def __init__(self, script: ScriptedPipelines, client: FakeRedis, transaction: bool, shard_hint: str | None) -> None:
        super().__init__(client.connection_pool, client.response_callbacks, transaction, shard_hint)
        self._script: Final = script

    async def execute(self, raise_on_error: bool = True) -> list[Any]:  # mutable-ok: Pipeline.execute returns a list
        if self._script.in_after_execution:
            return await super().execute(raise_on_error)
        self.command_stack = [self._script.as_sent(command) for command in self.command_stack]
        answers: Final = await super().execute(raise_on_error)
        await self._script.answered()
        return answers
