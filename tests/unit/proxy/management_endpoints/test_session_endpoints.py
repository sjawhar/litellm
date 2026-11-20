"""
Tests for POST /session/logout and revoke_ui_session_keys
(litellm/proxy/management_endpoints/session_endpoints.py).
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Final
from unittest.mock import DEFAULT, AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Response

from litellm.constants import LAST_GOOD_TOMBSTONE_TTL_SECONDS, UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import LiteLLM_VerificationToken, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import LAST_GOOD_LATE_STORE_MARGIN_SECONDS, UserApiKeyCache
from litellm.proxy.management_endpoints.session_endpoints import (
    revoke_ui_session_keys,
    session_logout,
)
from tests.unit.caching.fake_redis_cache import FakeRedisCache, fake_redis_cache

HASHED_TOKEN = "hashed-session-token"
USER_ID = "user-123"


def _session_row(token: str = HASHED_TOKEN, user_id: str = USER_ID) -> LiteLLM_VerificationToken:
    return LiteLLM_VerificationToken(token=token, team_id=UI_SESSION_TOKEN_TEAM_ID, user_id=user_id)


def _make_prisma(
    find_unique_row: LiteLLM_VerificationToken | None = None,
    find_many_rows: list[LiteLLM_VerificationToken] | None = None,
) -> MagicMock:
    prisma = MagicMock()
    table = prisma.db.litellm_verificationtoken
    table.find_unique = AsyncMock(return_value=find_unique_row)
    table.find_many = AsyncMock(return_value=find_many_rows or [])
    table.delete_many = AsyncMock(return_value=1)
    return prisma


def _ui_session_caller(token: str | None = HASHED_TOKEN) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(token=token, team_id=UI_SESSION_TOKEN_TEAM_ID, user_id=USER_ID)


def _patched_globals(prisma, cache: UserApiKeyCache | None = None):
    return (
        patch(  # test-quality-ok: endpoint reads proxy_server module globals; no injection seam
            "litellm.proxy.proxy_server.prisma_client", prisma
        ),
        patch(  # test-quality-ok: endpoint reads proxy_server module globals; no injection seam
            "litellm.proxy.proxy_server.proxy_logging_obj", None
        ),
        patch(  # test-quality-ok: endpoint reads proxy_server module globals; no injection seam
            "litellm.proxy.proxy_server.user_api_key_cache", cache if cache is not None else MagicMock()
        ),
    )


@pytest.mark.asyncio
async def test_session_logout_revokes_presented_session():
    prisma = _make_prisma(find_unique_row=_session_row())
    persist_mock = AsyncMock()
    evict_mock = AsyncMock()
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            persist_mock,
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            evict_mock,
        ),
    ):
        response = await session_logout(
            response=Response(),
            user_api_key_dict=_ui_session_caller(),
        )

    assert response.message == "Session revoked."
    delete_kwargs = prisma.db.litellm_verificationtoken.delete_many.call_args.kwargs
    assert delete_kwargs["where"] == {"token": HASHED_TOKEN}
    # Audit record persisted before the row is gone.
    persist_mock.assert_awaited_once()
    assert persist_mock.await_args.kwargs["keys"][0].token == HASHED_TOKEN
    # Cache evicted + broadcast even on the delete path.
    evict_mock.assert_awaited_once()
    assert tuple(evict_mock.await_args.kwargs["hashed_tokens"]) == (HASHED_TOKEN,)


@pytest.mark.asyncio
async def test_session_logout_clears_token_cookie():
    prisma = _make_prisma(find_unique_row=_session_row())
    fastapi_response = Response()
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            AsyncMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            AsyncMock(),
        ),
    ):
        await session_logout(
            response=fastapi_response,
            user_api_key_dict=_ui_session_caller(),
        )

    set_cookie_headers = [v.decode() for k, v in fastapi_response.raw_headers if k == b"set-cookie"]
    assert any(h.startswith('token="";') or h.startswith("token=;") for h in set_cookie_headers)


@pytest.mark.asyncio
async def test_session_logout_refuses_non_ui_session_key():
    """The endpoint must not become a generic key-deletion oracle: a normal
    virtual key (no UI team id) is refused outright."""
    prisma = _make_prisma()
    p1, p2, p3 = _patched_globals(prisma)

    with p1, p2, p3:
        with pytest.raises(HTTPException) as exc_info:
            await session_logout(
                response=Response(),
                user_api_key_dict=UserAPIKeyAuth(token=HASHED_TOKEN, team_id="some-real-team", user_id=USER_ID),
            )

    assert exc_info.value.status_code == 403
    prisma.db.litellm_verificationtoken.delete_many.assert_not_called()


@pytest.mark.asyncio
async def test_session_logout_is_idempotent_when_row_already_gone():
    prisma = _make_prisma(find_unique_row=None)
    evict_mock = AsyncMock()
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            evict_mock,
        ),
    ):
        response = await session_logout(
            response=Response(),
            user_api_key_dict=_ui_session_caller(),
        )

    assert response.message == "Session already revoked."
    prisma.db.litellm_verificationtoken.delete_many.assert_not_called()
    # The cache entry may outlive the row; evict regardless.
    evict_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_logout_requires_db():
    p2 = patch("litellm.proxy.proxy_server.proxy_logging_obj", None)
    p3 = patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock())
    with (
        patch(  # test-quality-ok: endpoint reads proxy_server module globals; no injection seam
            "litellm.proxy.proxy_server.prisma_client", None
        ),
        p2,
        p3,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await session_logout(
                response=Response(),
                user_api_key_dict=_ui_session_caller(),
            )

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_revokes_all_and_broadcasts():
    rows = [_session_row(token="t1"), _session_row(token="t2"), _session_row(token="t3")]
    prisma = _make_prisma(find_many_rows=rows)
    persist_mock = AsyncMock()
    evict_mock = AsyncMock()
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            persist_mock,
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            evict_mock,
        ),
    ):
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
        )

    assert revoked == 3
    find_kwargs = prisma.db.litellm_verificationtoken.find_many.call_args.kwargs
    assert find_kwargs["where"] == {"user_id": USER_ID, "team_id": UI_SESSION_TOKEN_TEAM_ID}
    delete_kwargs = prisma.db.litellm_verificationtoken.delete_many.call_args.kwargs
    assert delete_kwargs["where"] == {"token": {"in": ["t1", "t2", "t3"]}}
    persist_mock.assert_awaited_once()
    evict_mock.assert_awaited_once()
    assert evict_mock.await_args.kwargs["hashed_tokens"] == ["t1", "t2", "t3"]


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_keeps_callers_session():
    rows = [_session_row(token="t1"), _session_row(token=HASHED_TOKEN), _session_row(token="t3")]
    prisma = _make_prisma(find_many_rows=rows)
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            AsyncMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            AsyncMock(),
        ),
    ):
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
            keep_hashed_token=HASHED_TOKEN,
        )

    assert revoked == 2
    delete_kwargs = prisma.db.litellm_verificationtoken.delete_many.call_args.kwargs
    assert delete_kwargs["where"] == {"token": {"in": ["t1", "t3"]}}


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_noop_when_no_sessions():
    prisma = _make_prisma(find_many_rows=[])
    p1, p2, p3 = _patched_globals(prisma)

    with p1, p2, p3:
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
        )

    assert revoked == 0
    prisma.db.litellm_verificationtoken.delete_many.assert_not_called()


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_failure_is_swallowed():
    """The password write has already committed when this runs; a revocation
    failure must not fail the caller's request."""
    prisma = _make_prisma(find_many_rows=[_session_row(token="t1")])
    prisma.db.litellm_verificationtoken.delete_many = AsyncMock(side_effect=RuntimeError("db down"))
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            AsyncMock(),
        ),
    ):
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
        )

    assert revoked == 0


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_noop_without_db():
    with patch(  # test-quality-ok: helper reads proxy_server module globals; no injection seam
        "litellm.proxy.proxy_server.prisma_client", None
    ):
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
        )

    assert revoked == 0


class _DrivenClock:
    """The last good entries' clock, which a test moves instead of waiting."""

    def __init__(self) -> None:
        self.now: float = 1000.0

    def __call__(self) -> float:
        return self.now


# Past this since a revocation, the request's record no longer skips Redis for the key: its tombstone may lapse first
_PAST_THE_RECORD: Final = LAST_GOOD_TOMBSTONE_TTL_SECONDS - LAST_GOOD_LATE_STORE_MARGIN_SECONDS + 1


def _driven_cache(redis: FakeRedisCache, clock: _DrivenClock, *, serves_last_good: bool = True) -> UserApiKeyCache:
    return UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=serves_last_good, last_good_clock=clock)


def _outlasts_the_tombstones(
    redis: FakeRedisCache, clock: _DrivenClock, *, loses_redis: bool
) -> Callable[..., Awaitable[object]]:
    """A write mock's side effect: the write takes past the time the request's record of its revocations allows, the
    tombstones in Redis lapse meanwhile, and Redis is lost once it lands when ``loses_redis``. The write answers as
    configured."""

    async def outlast(*_args: object, **_kwargs: object) -> object:
        clock.now += _PAST_THE_RECORD
        client: Final = redis.cache.init_async_client()
        lapsed: Final = await client.keys("last_good_tombstone:*")
        if lapsed:
            await client.delete(*lapsed)
        if loses_redis:
            redis.server.connected = False
        return DEFAULT

    return outlast


def _tombstones_standing_at(
    redis: FakeRedisCache, cache_keys: Sequence[str], standing: list[tuple[str, ...]]
) -> Callable[..., Awaitable[object]]:
    """A write mock's side effect: records which of ``cache_keys`` have a tombstone in Redis when the write runs, none
    while Redis is lost. The write then answers as configured."""

    async def record(*_args: object, **_kwargs: object) -> object:
        client: Final = redis.cache.init_async_client()
        standing.append(
            tuple(
                [
                    cache_key
                    for cache_key in cache_keys
                    if redis.server.connected and await client.exists(f"last_good_tombstone:{cache_key}") == 1
                ]
            )
        )
        return DEFAULT

    return record


@pytest.mark.asyncio
async def test_session_logout_revokes_the_session_again_when_its_archive_outlasts_the_tombstone():
    """The logout revokes the session before it archives it; the archive takes past the time the request's record of
    that revocation allows and the tombstone lapses meanwhile, so another pod may store the session's entry again.
    The delete revokes the session again right before it: its tombstone stands in Redis when the row is deleted."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    prisma: Final = _make_prisma(find_unique_row=_session_row())
    standing: Final[list[tuple[str, ...]]] = []  # mutable-ok: one entry for each delete
    prisma.db.litellm_verificationtoken.delete_many.side_effect = _tombstones_standing_at(
        redis, (HASHED_TOKEN,), standing
    )
    p1, p2, p3 = _patched_globals(prisma, _driven_cache(redis, clock))

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            AsyncMock(side_effect=_outlasts_the_tombstones(redis, clock, loses_redis=False)),
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            AsyncMock(),
        ),
    ):
        response: Final = await session_logout(response=Response(), user_api_key_dict=_ui_session_caller())

    assert response.message == "Session revoked."
    assert standing == [(HASHED_TOKEN,)]


@pytest.mark.asyncio
async def test_session_logout_answers_503_and_keeps_the_session_when_redis_is_lost_after_its_archive_outlasts_it():
    """The same slow archive, with Redis lost once it lands: the session's revocation has to reach Redis again and
    cannot, so the logout answers 503 with the session row kept."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    prisma: Final = _make_prisma(find_unique_row=_session_row())
    persist_mock: Final = AsyncMock(side_effect=_outlasts_the_tombstones(redis, clock, loses_redis=True))
    evict_mock: Final = AsyncMock()
    p1, p2, p3 = _patched_globals(prisma, _driven_cache(redis, clock))

    with (
        p1,
        p2,
        p3,
        patch("litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens", persist_mock),
        patch("litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects", evict_mock),
    ):
        with pytest.raises(HTTPException) as refused:
            await session_logout(response=Response(), user_api_key_dict=_ui_session_caller())

    assert refused.value.status_code == 503
    persist_mock.assert_awaited_once()
    prisma.db.litellm_verificationtoken.delete_many.assert_not_awaited()
    evict_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_logout_with_the_setting_off_deletes_the_session_after_a_slow_archive_with_redis_lost():
    """With last good entries not served the revocations return at once, so the logout archives and deletes the
    session as it always did, whatever Redis does meanwhile."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    prisma: Final = _make_prisma(find_unique_row=_session_row())
    p1, p2, p3 = _patched_globals(prisma, _driven_cache(redis, clock, serves_last_good=False))

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            AsyncMock(side_effect=_outlasts_the_tombstones(redis, clock, loses_redis=True)),
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            AsyncMock(),
        ),
    ):
        response: Final = await session_logout(response=Response(), user_api_key_dict=_ui_session_caller())

    assert response.message == "Session revoked."
    prisma.db.litellm_verificationtoken.delete_many.assert_awaited_once_with(where={"token": HASHED_TOKEN})


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_revokes_the_sessions_again_when_their_archive_outlasts_the_tombstones():
    """The password write's session revocation revokes the sessions before it archives them; the archive takes past
    the time the request's record allows and the tombstones lapse meanwhile. The delete revokes the sessions again
    right before it: their tombstones stand in Redis when the rows are deleted."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    prisma: Final = _make_prisma(find_many_rows=[_session_row(token="t1"), _session_row(token="t2")])
    standing: Final[list[tuple[str, ...]]] = []  # mutable-ok: one entry for each delete
    prisma.db.litellm_verificationtoken.delete_many.side_effect = _tombstones_standing_at(
        redis, ("t1", "t2"), standing
    )
    p1, p2, p3 = _patched_globals(prisma, _driven_cache(redis, clock))

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            AsyncMock(side_effect=_outlasts_the_tombstones(redis, clock, loses_redis=False)),
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            AsyncMock(),
        ),
    ):
        revoked: Final = await revoke_ui_session_keys(user_id=USER_ID, user_api_key_dict=_ui_session_caller())

    assert revoked == 2
    assert standing == [("t1", "t2")]


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_keeps_every_session_when_redis_is_lost_after_their_archive_outlasts_them():
    """The same slow archive, with Redis lost once it lands: the sessions' revocation has to reach Redis again and
    cannot, so no session row is deleted. The password write before it has committed, so the refusal is logged and
    the call reports no session revoked rather than failing the request."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    prisma: Final = _make_prisma(find_many_rows=[_session_row(token="t1"), _session_row(token="t2")])
    persist_mock: Final = AsyncMock(side_effect=_outlasts_the_tombstones(redis, clock, loses_redis=True))
    evict_mock: Final = AsyncMock()
    p1, p2, p3 = _patched_globals(prisma, _driven_cache(redis, clock))

    with (
        p1,
        p2,
        p3,
        patch("litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens", persist_mock),
        patch("litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects", evict_mock),
    ):
        revoked: Final = await revoke_ui_session_keys(user_id=USER_ID, user_api_key_dict=_ui_session_caller())

    assert revoked == 0
    persist_mock.assert_awaited_once()
    prisma.db.litellm_verificationtoken.delete_many.assert_not_awaited()
    evict_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_with_the_setting_off_deletes_the_sessions_after_a_slow_archive_with_redis_lost():
    """With last good entries not served the revocations return at once, so the sessions are archived and deleted as
    they always were, whatever Redis does meanwhile."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    prisma: Final = _make_prisma(find_many_rows=[_session_row(token="t1"), _session_row(token="t2")])
    p1, p2, p3 = _patched_globals(prisma, _driven_cache(redis, clock, serves_last_good=False))

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            AsyncMock(side_effect=_outlasts_the_tombstones(redis, clock, loses_redis=True)),
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            AsyncMock(),
        ),
    ):
        revoked: Final = await revoke_ui_session_keys(user_id=USER_ID, user_api_key_dict=_ui_session_caller())

    assert revoked == 2
    prisma.db.litellm_verificationtoken.delete_many.assert_awaited_once_with(where={"token": {"in": ["t1", "t2"]}})
