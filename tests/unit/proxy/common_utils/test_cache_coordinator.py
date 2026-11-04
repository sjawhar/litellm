import asyncio
from typing import Final

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.proxy.common_utils.cache_coordinator import EventDrivenCacheCoordinator


async def _let_waiters_queue() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_waiters_raise_the_error_the_load_raised():
    """A waiter that read the empty cache after a failed load would answer as if the resource held nothing."""
    coordinator: Final = EventDrivenCacheCoordinator(log_prefix="")
    cache: Final = DualCache()
    release: Final = asyncio.Event()
    loads: Final = asyncio.Queue[None]()

    async def fails() -> float:
        loads.put_nowait(None)
        await release.wait()
        raise ConnectionError("database unreachable")

    requests: Final = [asyncio.create_task(coordinator.get_or_load("spend", cache, fails)) for _ in range(3)]
    await _let_waiters_queue()
    release.set()

    outcomes: Final = await asyncio.gather(*requests, return_exceptions=True)

    assert [repr(outcome) for outcome in outcomes] == [repr(ConnectionError("database unreachable"))] * 3
    assert loads.qsize() == 1


@pytest.mark.asyncio
async def test_waiters_of_a_cancelled_load_load_again():
    coordinator: Final = EventDrivenCacheCoordinator(log_prefix="")
    cache: Final = DualCache()
    started: Final = asyncio.Event()
    never: Final = asyncio.Event()

    async def hangs() -> float:
        started.set()
        await never.wait()
        return 1.0

    async def answers() -> float:
        return 2.0

    loader: Final = asyncio.create_task(coordinator.get_or_load("spend", cache, hangs))
    await started.wait()
    waiter: Final = asyncio.create_task(coordinator.get_or_load("spend", cache, answers))
    await _let_waiters_queue()
    loader.cancel()

    assert await asyncio.wait_for(waiter, timeout=5) == 2.0
    with pytest.raises(asyncio.CancelledError):
        await loader
