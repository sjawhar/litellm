"""The race of a strict write whose revocation needed no Redis, for the endpoint tests of each strict path.

With a 30 s tombstone, the request revokes an entry at t=0 (``revoked_before_the_write``); its paired write runs at
t=19, when the request's record shows that tombstone standing for 11 s more, so the pair sends no Redis command.
Another pod's read began at t=19, before the write committed, and got the old row; it stores it at t=31, once the
first tombstone lapsed at t=30, and read 12 s before, far inside the time a store may take
(``served_after_another_pod_stores``). Only a tombstone recorded again after the write refuses that store; without
one, an outage serves the old row. The endpoint must run in the test's own task, so the request's record is the one
``revoked_before_the_write`` filled.
"""

from dataclasses import dataclass
from typing import Final, TypeVar

import pytest
from pydantic import BaseModel

from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    last_good_reads_started_at,
    recall_last_good,
    remember_last_good,
    revoke_last_good,
)
from tests.unit.caching.fake_redis_cache import FakeRedisCache, drive_fake_redis_time, fake_redis_cache

RACE_TOMBSTONE_SECONDS: Final = 30
# When the request's paired write runs and another pod's read of the old row begins, seconds after its first revocation
RACE_WRITE_AT: Final = 19.0
# When that read's store completes: after the first tombstone lapsed, 12 s after the read began
RACE_STORE_AT: Final = 31.0
# How far the wall clock, which fake Redis runs on, is ahead of the race's clock
_WALL_CLOCK_AHEAD_SECONDS: Final = 1_700_000_000.25

_Row = TypeVar("_Row", bound=BaseModel)


class RaceClock:
    """The last good entries' clock, which the race moves instead of waiting, and the wall clock moving with it."""

    def __init__(self) -> None:
        self.now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def wall(self) -> float:
        return self.now + _WALL_CLOCK_AHEAD_SECONDS


@dataclass(frozen=True, slots=True)
class TombstoneRace:
    clock: RaceClock
    redis: FakeRedisCache
    # The auth cache of the pod that serves the request
    cache: UserApiKeyCache

    def pod(self) -> UserApiKeyCache:
        """Another pod's auth cache over the same Redis and clocks."""
        return UserApiKeyCache(
            redis_cache=self.redis.cache,
            serve_last_good_entries=True,
            last_good_clock=self.clock,
            last_good_wall_clock=self.clock.wall,
        )


def tombstone_race(monkeypatch: pytest.MonkeyPatch) -> TombstoneRace:
    """A 30 s tombstone, fake Redis on the race's wall clock, and the serving pod's auth cache at t=0."""
    monkeypatch.setattr(  # test-quality-ok: the race takes a 30 s tombstone; no setting seam
        "litellm.proxy.common_utils.user_api_key_cache.LAST_GOOD_TOMBSTONE_TTL_SECONDS", RACE_TOMBSTONE_SECONDS
    )
    clock: Final = RaceClock()
    redis: Final = fake_redis_cache()
    drive_fake_redis_time(monkeypatch, clock.wall)
    cache: Final = UserApiKeyCache(
        redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=clock, last_good_wall_clock=clock.wall
    )
    return TombstoneRace(clock=clock, redis=redis, cache=cache)


async def revoked_before_the_write(race: TombstoneRace, cache_keys: tuple[str, ...]) -> None:
    """The request's first revocation of ``cache_keys`` at t=0, recorded in the test's task, then the clock to the
    write at t=19."""
    await revoke_last_good(race.cache, cache_keys)
    race.clock.now = RACE_WRITE_AT


async def served_after_another_pod_stores(
    race: TombstoneRace,
    cache_key: str,
    old_row: _Row,
    *,
    read_began_at: float = RACE_WRITE_AT,
    stored_at: float = RACE_STORE_AT,
) -> _Row | None:
    """Another pod stores ``old_row``, which its read begun at ``read_began_at`` (t=19) got, at ``stored_at`` (t=31);
    returns what a third pod serves for ``cache_key`` in an outage that follows."""
    race.clock.now = stored_at
    with last_good_reads_started_at(read_began_at):
        await remember_last_good(race.pod(), cache_key, old_row)
    return await recall_last_good(race.pod(), cache_key, type(old_row))
