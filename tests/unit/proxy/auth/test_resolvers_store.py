from __future__ import annotations

from typing import Dict, Optional

import httpx
import pytest

from litellm.proxy._types import UserAPIKeyAuth, hash_token
from litellm.proxy.auth.auth_method import AuthMethod
from litellm.proxy.auth.resolvers.exceptions import (
    KeyNotFoundError,
    NoDatabaseConnectionError,
    PrincipalMissingSourceKeyError,
)
from litellm.proxy.auth.resolvers.models import Principal, PrincipalType
from litellm.proxy.auth.resolvers.store import IdentityStore
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


class _FakeCache:
    """Stands in for the DualCache get_key_object reads. It returns a cache hit
    before the DB is touched, so seeding it exercises resolve without a database
    (a non-None prisma client is still required; it is never reached on a hit)."""

    def __init__(self, entries: Optional[Dict[str, object]] = None) -> None:
        self._entries = entries or {}

    async def async_get_cache(self, key, *args, **kwargs):
        return self._entries.get(key)

    async def async_set_cache(self, *args, **kwargs):
        return None


async def test_resolve_returns_a_principal_projected_from_the_looked_up_key():
    raw = "sk-live-abc"
    key = UserAPIKeyAuth(token=hash_token(raw), user_id="u-1", team_id="t-1")
    store = IdentityStore(object(), _FakeCache({hash_token(raw): key}))

    principal = await store.resolve(hashed_token=hash_token(raw))

    assert isinstance(principal, Principal)
    assert principal.principal_type == PrincipalType.HUMAN
    assert principal.user is not None and principal.user.id == "u-1"
    assert [t.id for t in principal.teams] == ["t-1"]


async def test_resolve_carries_the_key_for_key_from_principal():
    raw = "sk-live-abc"
    key = UserAPIKeyAuth(token=hash_token(raw), user_id="u-1", team_id="t-1")
    store = IdentityStore(object(), _FakeCache({hash_token(raw): key}))

    principal = await store.resolve(hashed_token=hash_token(raw))
    recovered = IdentityStore.key_from_principal(principal)

    assert recovered.user_id == "u-1"
    assert recovered.team_id == "t-1"


def test_key_from_principal_raises_when_no_source_key_is_carried():
    bare = Principal(
        principal_type=PrincipalType.SERVICE_ACCOUNT,
        subject="svc",
        auth_method=AuthMethod.API_KEY,
    )
    with pytest.raises(PrincipalMissingSourceKeyError):
        IdentityStore.key_from_principal(bare)


async def test_resolve_raises_without_a_db_connection():
    store = IdentityStore(None, _FakeCache())
    with pytest.raises(NoDatabaseConnectionError):
        await store.resolve(hashed_token="missing")


class _KeyRows:
    """The key reads ``IdentityStore`` makes, answered from rows or failed the way a stopped database fails."""

    def __init__(self, rows: Dict[str, UserAPIKeyAuth]) -> None:
        self.rows = rows
        self.stopped = False

    async def get_data(self, token: str, table_name: str, parent_otel_span: object, proxy_logging_obj: object):
        if self.stopped:
            raise httpx.ConnectError("All connection attempts failed")
        return self.rows.get(token)

    async def attempt_db_reconnect(self, *, reason: str, timeout_seconds: float, lock_timeout_seconds: float) -> bool:
        return False


async def test_resolve_serves_the_last_good_key_while_the_database_is_stopped():
    hashed = hash_token("sk-lane")
    database = _KeyRows({hashed: UserAPIKeyAuth(token=hashed, team_id="t-1", models=["claude-opus-5-5"])})
    cache = UserApiKeyCache(serve_last_good_entries=True)
    await IdentityStore(database, cache).resolve(hashed_token=hashed)
    await cache.async_delete_cache(hashed)
    database.stopped = True

    served = IdentityStore.key_from_principal(await IdentityStore(database, cache).resolve(hashed_token=hashed))

    assert (served.token, served.team_id, served.models) == (hashed, "t-1", ["claude-opus-5-5"])


async def test_resolve_refuses_a_deleted_key_as_soon_as_the_database_stops():
    from litellm.proxy.auth.auth_checks import delete_cache_key_object

    hashed = hash_token("sk-lane")
    database = _KeyRows({hashed: UserAPIKeyAuth(token=hashed, models=["claude-opus-5-5"])})
    cache = UserApiKeyCache(serve_last_good_entries=True)
    await IdentityStore(database, cache).resolve(hashed_token=hashed)
    del database.rows[hashed]
    await delete_cache_key_object(hashed_token=hashed, user_api_key_cache=cache, proxy_logging_obj=None)
    database.stopped = True

    with pytest.raises(httpx.ConnectError):
        await IdentityStore(database, cache).resolve(hashed_token=hashed)


async def test_resolve_keeps_refusing_a_key_the_database_reported_missing():
    hashed = hash_token("sk-lane")
    database = _KeyRows({hashed: UserAPIKeyAuth(token=hashed, models=["claude-opus-5-5"])})
    cache = UserApiKeyCache(serve_last_good_entries=True)
    await IdentityStore(database, cache).resolve(hashed_token=hashed)
    del database.rows[hashed]
    await cache.async_delete_cache(hashed)
    with pytest.raises(KeyNotFoundError):
        await IdentityStore(database, cache).resolve(hashed_token=hashed)
    database.stopped = True

    with pytest.raises(httpx.ConnectError):
        await IdentityStore(database, cache).resolve(hashed_token=hashed)


async def test_resolve_does_not_keep_a_rotated_out_key_under_its_old_hash():
    old_hash = hash_token("sk-rotated-out")
    new_hash = hash_token("sk-rotated-in")
    database = _KeyRows({old_hash: UserAPIKeyAuth(token=new_hash, models=["claude-opus-5-5"])})
    cache = UserApiKeyCache(serve_last_good_entries=True)
    await IdentityStore(database, cache).resolve(hashed_token=old_hash)
    await cache.async_delete_cache(old_hash)
    database.stopped = True

    with pytest.raises(httpx.ConnectError):
        await IdentityStore(database, cache).resolve(hashed_token=old_hash)
