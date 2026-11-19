# tests/test_budget_endpoints.py

import asyncio
import json
import time
import types
from datetime import datetime, timedelta, timezone
from typing import Final
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy.proxy_server import app
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles, CommonProxyErrors


class _KeyRows:
    """The key table, answering a column filter, or the AND or OR of column filters a budget change reads keys by."""

    def __init__(self, *rows: types.SimpleNamespace) -> None:
        self._rows = rows

    async def find_many(self, where: dict) -> list[types.SimpleNamespace]:
        def matches(row: types.SimpleNamespace, clause: dict) -> bool:
            if "AND" in clause:
                return all(matches(row, part) for part in clause["AND"])
            return all(
                getattr(row, column) in wanted["in"] if isinstance(wanted, dict) else getattr(row, column) == wanted
                for column, wanted in clause.items()
            )

        clauses = where["OR"] if "OR" in where else [where]
        return [row for row in self._rows if any(matches(row, clause) for clause in clauses)]


@pytest.fixture
def client_and_mocks(monkeypatch):
    # Setup MagicMock Prisma
    mock_prisma = MagicMock()
    mock_table = MagicMock()
    mock_table.create = AsyncMock(side_effect=lambda *, data: data)
    mock_table.update = AsyncMock(side_effect=lambda *, where, data: {**where, **data})

    mock_prisma.db = types.SimpleNamespace(
        litellm_budgettable=mock_table,
        litellm_dailyspend=mock_table,
        litellm_teammembership=types.SimpleNamespace(find_many=AsyncMock(return_value=[])),
        litellm_organizationtable=types.SimpleNamespace(find_many=AsyncMock(return_value=[])),
        litellm_projecttable=types.SimpleNamespace(find_many=AsyncMock(return_value=[])),
        litellm_endusertable=types.SimpleNamespace(find_many=AsyncMock(return_value=[])),
        litellm_tagtable=types.SimpleNamespace(find_many=AsyncMock(return_value=[])),
        litellm_modelaccessgroupbudgettable=types.SimpleNamespace(find_many=AsyncMock(return_value=[])),
        litellm_verificationtoken=_KeyRows(),
    )

    # Monkeypatch Mocked Prisma client into the server module
    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    # override returned auth user
    fake_user = UserAPIKeyAuth(
        user_id="test_user",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: fake_user

    client = TestClient(app)

    yield client, mock_prisma, mock_table

    # teardown
    app.dependency_overrides.clear()
    monkeypatch.setattr(ps, "prisma_client", ps.prisma_client)


@pytest.mark.asyncio
async def test_new_budget_success(client_and_mocks):
    client, _, mock_table = client_and_mocks

    # Call /budget/new endpoint
    payload = {
        "budget_id": "budget_123",
        "max_budget": 42.0,
        "budget_duration": "30d",
    }
    resp = client.post("/budget/new", json=payload)
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["budget_id"] == payload["budget_id"]
    assert body["max_budget"] == payload["max_budget"]
    assert body["budget_duration"] == payload["budget_duration"]
    assert body["created_by"] == "test_user"
    assert body["updated_by"] == "test_user"

    mock_table.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_budget_drops_the_last_good_entries_of_everything_on_it(client_and_mocks, monkeypatch):
    """Keys, memberships, organizations, projects, customers, tags and model access groups carry their budget row, a
    key also carries its organization's and its team membership's, and the default customer and team-member budgets
    are cached by id. A last good entry never expires, so a changed budget must evict all of them, or a database
    outage enforces the old limits."""
    from litellm.proxy._types import LiteLLM_BudgetTable
    from litellm.proxy.common_utils.user_api_key_cache import (
        UserApiKeyCache,
        default_end_user_budget_cache_key,
        end_user_cache_key,
        model_access_group_cache_key,
        project_cache_key,
        recall_last_good,
        remember_last_good,
        tag_cache_key,
        team_member_default_budget_cache_key,
        team_membership_auth_cache_key,
    )

    client, mock_prisma, _ = client_and_mocks
    mock_prisma.db.litellm_teammembership.find_many = AsyncMock(
        return_value=[types.SimpleNamespace(user_id="u1", team_id="t1")]
    )
    mock_prisma.db.litellm_organizationtable.find_many = AsyncMock(
        return_value=[types.SimpleNamespace(organization_id="o1")]
    )
    mock_prisma.db.litellm_projecttable.find_many = AsyncMock(return_value=[types.SimpleNamespace(project_id="p1")])
    mock_prisma.db.litellm_endusertable.find_many = AsyncMock(return_value=[types.SimpleNamespace(user_id="c1")])
    mock_prisma.db.litellm_tagtable.find_many = AsyncMock(return_value=[types.SimpleNamespace(tag_name="tag-1")])
    mock_prisma.db.litellm_modelaccessgroupbudgettable.find_many = AsyncMock(
        return_value=[types.SimpleNamespace(access_group_name="group-1")]
    )

    def key(token: str, budget_id: str | None, organization_id: str | None, team_id: str | None, user_id: str | None):
        return types.SimpleNamespace(
            token=token, budget_id=budget_id, organization_id=organization_id, team_id=team_id, user_id=user_id
        )

    mock_prisma.db.litellm_verificationtoken = _KeyRows(
        key("hashed-own-budget", "budget_shared", None, None, None),
        key("hashed-org-budget", None, "o1", None, "u9"),
        key("hashed-member-budget", None, None, "t1", "u1"),
        key("hashed-unrelated", "budget_other", "o2", "t1", "u2"),
    )
    cache: Final = UserApiKeyCache(serve_last_good_entries=True)
    keys: Final = (
        team_membership_auth_cache_key(team_id="t1", user_id="u1"),
        "org_id:o1:with_budget",
        project_cache_key("p1"),
        end_user_cache_key("c1"),
        tag_cache_key("tag-1"),
        model_access_group_cache_key("group-1"),
        default_end_user_budget_cache_key("budget_shared"),
        team_member_default_budget_cache_key("budget_shared"),
        "hashed-own-budget",
        "hashed-org-budget",
        "hashed-member-budget",
    )
    for key_name in (*keys, "hashed-unrelated"):
        await remember_last_good(
            cache,
            key_name,
            LiteLLM_BudgetTable(budget_id="budget_shared", max_budget=50.0),
            read_started=time.monotonic(),
        )
    monkeypatch.setattr(ps, "user_api_key_cache", cache)

    resp = client.post("/budget/update", json={"budget_id": "budget_shared", "max_budget": 5.0})

    assert resp.status_code == 200, resp.text
    assert {
        key_name: await recall_last_good(cache, key_name, LiteLLM_BudgetTable) for key_name in keys
    } == dict.fromkeys(keys)
    assert await recall_last_good(cache, "hashed-unrelated", LiteLLM_BudgetTable) is not None
    for table in (
        "litellm_teammembership",
        "litellm_organizationtable",
        "litellm_projecttable",
        "litellm_endusertable",
        "litellm_tagtable",
        "litellm_modelaccessgroupbudgettable",
    ):
        getattr(mock_prisma.db, table).find_many.assert_awaited_once_with(where={"budget_id": "budget_shared"})


@pytest.mark.asyncio
async def test_update_budget_refuses_and_writes_nothing_when_a_holder_table_cannot_be_read(
    client_and_mocks, monkeypatch
):
    """While last good entries are served, a holder the update cannot list is a holder it cannot revoke, so the budget
    is left as it was and the operator retries."""
    from litellm.proxy._types import LiteLLM_BudgetTable
    from litellm.proxy.common_utils.user_api_key_cache import (
        UserApiKeyCache,
        recall_last_good,
        remember_last_good,
    )

    client, mock_prisma, mock_table = client_and_mocks
    own_budget_key: Final = "hashed-own-budget"
    mock_prisma.db.litellm_tagtable.find_many = AsyncMock(side_effect=Exception("tag table is unreadable"))
    mock_prisma.db.litellm_verificationtoken = _KeyRows(
        types.SimpleNamespace(
            token=own_budget_key, budget_id="budget_shared", organization_id=None, team_id=None, user_id=None
        )
    )
    cache: Final = UserApiKeyCache(serve_last_good_entries=True)
    budget: Final = LiteLLM_BudgetTable(budget_id="budget_shared", max_budget=50.0)
    await remember_last_good(cache, own_budget_key, budget, read_started=time.monotonic())
    monkeypatch.setattr(ps, "user_api_key_cache", cache)

    resp = client.post("/budget/update", json={"budget_id": "budget_shared", "max_budget": 5.0})

    assert resp.status_code == 503, resp.text
    mock_table.update.assert_not_awaited()
    assert await recall_last_good(cache, own_budget_key, LiteLLM_BudgetTable) is not None


@pytest.mark.asyncio
async def test_update_budget_evicts_the_other_holders_when_one_table_cannot_be_read(client_and_mocks, monkeypatch):
    """With last good entries off, a budget update reads seven tables best effort, as before: one of them failing still
    writes the budget and evicts the keys on it."""
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    client, mock_prisma, mock_table = client_and_mocks
    own_budget_key: Final = "hashed-own-budget"
    mock_prisma.db.litellm_tagtable.find_many = AsyncMock(side_effect=Exception("tag table is unreadable"))
    mock_prisma.db.litellm_verificationtoken = _KeyRows(
        types.SimpleNamespace(
            token=own_budget_key, budget_id="budget_shared", organization_id=None, team_id=None, user_id=None
        )
    )
    cache: Final = UserApiKeyCache(serve_last_good_entries=False)
    cache.set_cache(key=own_budget_key, value="the key object cached before the update")
    monkeypatch.setattr(ps, "user_api_key_cache", cache)

    resp = client.post("/budget/update", json={"budget_id": "budget_shared", "max_budget": 5.0})

    assert resp.status_code == 200, resp.text
    mock_table.update.assert_awaited_once()
    assert cache.get_cache(key=own_budget_key) is None


@pytest.mark.parametrize("bad_duration", ["0s", "-5m"])
@pytest.mark.asyncio
async def test_new_budget_rejects_a_duration_that_never_advances(
    client_and_mocks, bad_duration
):
    """A zero-length window resets to "now", so the row is due again the moment
    it is written and the reset job re-reads it on every tick forever."""
    client, _, mock_table = client_and_mocks

    resp = client.post(
        "/budget/new",
        json={"budget_id": "budget_bad", "max_budget": 10.0, "budget_duration": bad_duration},
    )

    assert resp.status_code == 400, resp.text
    assert "Invalid budget_duration" in resp.json()["detail"]["error"]
    mock_table.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_budget_rejects_a_duration_that_never_advances(client_and_mocks):
    client, _, mock_table = client_and_mocks

    resp = client.post(
        "/budget/update",
        json={"budget_id": "budget_456", "budget_duration": "0s"},
    )

    assert resp.status_code == 400, resp.text
    assert "Invalid budget_duration" in resp.json()["detail"]["error"]
    mock_table.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_budget_db_not_connected(client_and_mocks, monkeypatch):
    client, mock_prisma, mock_table = client_and_mocks

    # override the prisma_client that the handler imports at runtime
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", None)

    # Call /budget/new endpoint
    resp = client.post("/budget/new", json={"budget_id": "no_db", "max_budget": 1.0})
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["error"] == CommonProxyErrors.db_not_connected_error.value


@pytest.mark.asyncio
async def test_update_budget_success(client_and_mocks, monkeypatch):
    client, mock_prisma, mock_table = client_and_mocks

    payload = {
        "budget_id": "budget_456",
        "max_budget": 99.0,
        "soft_budget": 50.0,
    }
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["budget_id"] == payload["budget_id"]
    assert body["max_budget"] == payload["max_budget"]
    assert body["soft_budget"] == payload["soft_budget"]
    assert body["updated_by"] == "test_user"


@pytest.mark.asyncio
async def test_new_and_update_budget_persist_tpd_limit(client_and_mocks):
    client, _, mock_table = client_and_mocks

    resp = client.post("/budget/new", json={"budget_id": "budget_tpd", "tpd_limit": 250000})
    assert resp.status_code == 200, resp.text
    assert resp.json()["tpd_limit"] == 250000
    assert mock_table.create.await_args.kwargs["data"]["tpd_limit"] == 250000

    resp = client.post("/budget/update", json={"budget_id": "budget_tpd", "tpd_limit": 500000})
    assert resp.status_code == 200, resp.text
    assert resp.json()["tpd_limit"] == 500000
    assert mock_table.update.await_args.kwargs["data"]["tpd_limit"] == 500000


@pytest.mark.asyncio
async def test_update_budget_missing_id(client_and_mocks, monkeypatch):
    client, mock_prisma, mock_table = client_and_mocks

    payload = {"max_budget": 10.0}
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "budget_id is required"


@pytest.mark.asyncio
async def test_update_budget_db_not_connected(client_and_mocks, monkeypatch):
    client, mock_prisma, mock_table = client_and_mocks

    # override the prisma_client that the handler imports at runtime
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", None)

    payload = {"budget_id": "any", "max_budget": 1.0}
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["error"] == CommonProxyErrors.db_not_connected_error.value


@pytest.mark.asyncio
async def test_update_budget_allows_null_max_budget(client_and_mocks):
    """
    Test that /budget/update allows setting max_budget to null.

    Previously, using exclude_none=True would drop null values,
    making it impossible to remove a budget limit. With exclude_unset=True,
    explicitly setting max_budget to null should include it in the update.
    """
    client, _, mock_table = client_and_mocks

    captured_data = {}

    async def capture_update(*, where, data):
        captured_data.update(data)
        return {**where, **data}

    mock_table.update = AsyncMock(side_effect=capture_update)

    payload = {
        "budget_id": "budget_789",
        "max_budget": None,  # Explicitly setting to null to remove budget limit
    }
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 200, resp.text

    # Verify that max_budget=None was included in the update data
    assert (
        "max_budget" in captured_data
    ), "max_budget should be included when explicitly set to null"
    assert captured_data["max_budget"] is None, "max_budget should be None"

    mock_table.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_new_budget_negative_max_budget(client_and_mocks):
    """
    Test that /budget/new rejects negative max_budget values.

    This prevents the issue where negative budgets would always trigger
    budget exceeded errors.
    """
    client, _, _ = client_and_mocks

    payload = {
        "budget_id": "budget_negative",
        "max_budget": -7.0,
    }
    resp = client.post("/budget/new", json=payload)
    assert resp.status_code == 400, resp.text

    detail = resp.json()["detail"]
    assert "max_budget must be a non-negative finite number" in str(detail)


@pytest.mark.asyncio
async def test_new_budget_negative_soft_budget(client_and_mocks):
    """
    Test that /budget/new rejects negative soft_budget values.
    """
    client, _, _ = client_and_mocks

    payload = {
        "budget_id": "budget_negative_soft",
        "soft_budget": -10.0,
    }
    resp = client.post("/budget/new", json=payload)
    assert resp.status_code == 400, resp.text

    detail = resp.json()["detail"]
    assert "soft_budget must be a non-negative finite number" in str(detail)


@pytest.mark.asyncio
async def test_update_budget_negative_max_budget(client_and_mocks):
    """
    Test that /budget/update rejects negative max_budget values.
    """
    client, _, _ = client_and_mocks

    payload = {
        "budget_id": "budget_update_negative",
        "max_budget": -5.0,
    }
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 400, resp.text

    detail = resp.json()["detail"]
    assert "max_budget must be a non-negative finite number" in str(detail)


@pytest.mark.asyncio
async def test_update_budget_negative_soft_budget(client_and_mocks):
    """
    Test that /budget/update rejects negative soft_budget values.
    """
    client, _, _ = client_and_mocks

    payload = {
        "budget_id": "budget_update_negative_soft",
        "soft_budget": -15.0,
    }
    resp = client.post("/budget/update", json=payload)
    assert resp.status_code == 400, resp.text

    detail = resp.json()["detail"]
    assert "soft_budget must be a non-negative finite number" in str(detail)


@pytest.mark.asyncio
async def test_new_budget_invalid_model_max_budget(client_and_mocks, monkeypatch):
    """
    Test that /budget/new validates model_max_budget and returns 400 for invalid structure.
    Per-model budget implementation: validate_model_max_budget is called in new_budget.
    """
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "premium_user", True)

    client, _, _ = client_and_mocks

    payload = {
        "budget_id": "budget_invalid_mmb",
        "max_budget": 10.0,
        "model_max_budget": {"gpt-4": "not-a-dict"},
    }
    resp = client.post("/budget/new", json=payload)
    # Pydantic may reject invalid structure with 422 before our validator runs
    assert resp.status_code in (400, 422), resp.text
    detail = resp.json()["detail"]
    assert "model_max_budget" in str(detail) or "dictionary" in str(detail).lower()


def _capture_update_data(mock_table):
    captured = {}

    async def capture(*, where, data):
        captured.update(data)
        return {**where, **data}

    mock_table.update = AsyncMock(side_effect=capture)
    return captured


@pytest.mark.asyncio
async def test_update_budget_recomputes_reset_at_when_duration_changes(
    client_and_mocks,
):
    """
    Regression for LIT-3362: shortening budget_duration without an explicit
    budget_reset_at must bring the reset forward instead of leaving it pinned
    to the previous (longer) schedule.
    """
    client, _, mock_table = client_and_mocks
    captured = _capture_update_data(mock_table)

    before = datetime.now(timezone.utc)
    resp = client.post(
        "/budget/update",
        json={"budget_id": "budget_reset_recompute", "budget_duration": "1d"},
    )
    assert resp.status_code == 200, resp.text

    assert (
        "budget_reset_at" in captured
    ), "duration change must recompute budget_reset_at"
    reset_at = captured["budget_reset_at"]
    assert isinstance(reset_at, datetime)
    assert reset_at > before, "recomputed reset must be in the future"
    # "1d" resets at the next standardized day boundary, always within ~24h
    assert reset_at <= before + timedelta(days=1, hours=1), reset_at
    # and it must be far closer than a stale 30d schedule would have left it
    assert reset_at < before + timedelta(days=29)


@pytest.mark.asyncio
@pytest.mark.parametrize("budget_duration", ["1d", None])
async def test_update_budget_preserves_explicit_reset_at(client_and_mocks, budget_duration):
    """An explicit budget_reset_at from the caller always wins over recompute."""
    client, _, mock_table = client_and_mocks
    captured = _capture_update_data(mock_table)

    explicit = datetime(2027, 1, 1, tzinfo=timezone.utc)
    resp = client.post(
        "/budget/update",
        json={
            "budget_id": "budget_explicit_reset",
            "budget_duration": budget_duration,
            "budget_reset_at": explicit.isoformat(),
        },
    )
    assert resp.status_code == 200, resp.text

    assert captured["budget_reset_at"] == explicit


@pytest.mark.asyncio
async def test_update_budget_without_duration_leaves_reset_at_untouched(
    client_and_mocks,
):
    """Updates that do not touch budget_duration must not introduce budget_reset_at."""
    client, _, mock_table = client_and_mocks
    captured = _capture_update_data(mock_table)

    resp = client.post(
        "/budget/update",
        json={"budget_id": "budget_other_field", "max_budget": 200.0},
    )
    assert resp.status_code == 200, resp.text

    assert "budget_reset_at" not in captured


@pytest.mark.asyncio
async def test_update_budget_duration_none_clears_obsolete_reset(client_and_mocks):
    client, _, mock_table = client_and_mocks
    captured = _capture_update_data(mock_table)

    resp = client.post(
        "/budget/update",
        json={"budget_id": "budget_clear_duration", "budget_duration": None},
    )
    assert resp.status_code == 200, resp.text

    assert "budget_duration" in captured and captured["budget_duration"] is None
    assert captured["budget_reset_at"] is None


@pytest.mark.asyncio
async def test_update_budget_serializes_model_max_budget_for_prisma(
    client_and_mocks, monkeypatch
):
    monkeypatch.setattr(ps, "premium_user", True)

    client, _, mock_table = client_and_mocks
    captured: Final = _capture_update_data(mock_table)

    resp: Final = client.post(
        "/budget/update",
        json={
            "budget_id": "budget_per_model",
            "model_max_budget": {
                "gpt4o": {"budget_limit": 5.0, "time_period": "1d"},
                "glm-5.2": {"budget_limit": 7.5, "time_period": "30d"},
            },
        },
    )
    assert resp.status_code == 200, resp.text

    stored: Final = captured["model_max_budget"]
    assert isinstance(stored, str), (
        f"model_max_budget must reach prisma as a JSON string, got {type(stored).__name__}"
    )
    assert json.loads(stored) == {
        "gpt4o": {"max_budget": 5.0, "budget_duration": "1d"},
        "glm-5.2": {"max_budget": 7.5, "budget_duration": "30d"},
    }


async def _update_budget_directly(budget_id: str, max_budget: float):
    """``/budget/update`` called in the test's own event loop, so a read held in that loop can straddle it."""
    from litellm.proxy._types import BudgetNewRequest
    from litellm.proxy.management_endpoints.budget_management_endpoints import update_budget

    return await update_budget(
        budget_obj=BudgetNewRequest(budget_id=budget_id, max_budget=max_budget),
        user_api_key_dict=UserAPIKeyAuth(user_id="test_user", user_role=LitellmUserRoles.PROXY_ADMIN),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("serves_last_good", [True, False], ids=["serving", "not_serving"])
async def test_update_budget_answers_503_and_changes_nothing_when_redis_cannot_record_the_revocation(
    client_and_mocks, monkeypatch, serves_last_good: bool
):
    """A customer on the budget keeps a last good entry that never expires. With Redis unable to record its revocation,
    the budget stays as it was rather than the old limit staying in that entry; with last good entries off, nothing is
    revoked and the budget is written as before."""
    from fastapi import HTTPException
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from tests.unit.caching.fake_redis_cache import fake_redis_cache

    _, mock_prisma, mock_table = client_and_mocks
    mock_prisma.db.litellm_endusertable.find_many = AsyncMock(return_value=[types.SimpleNamespace(user_id="c1")])
    redis: Final = fake_redis_cache()
    redis.server.connected = False
    monkeypatch.setattr(
        ps, "user_api_key_cache", UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=serves_last_good)
    )

    if not serves_last_good:
        await _update_budget_directly("budget_shared", 5.0)
        mock_table.update.assert_awaited_once()
        return
    with pytest.raises(HTTPException) as refused:
        await _update_budget_directly("budget_shared", 5.0)

    assert refused.value.status_code == 503
    mock_table.update.assert_not_awaited()


class _HeldAuthReads:
    """The database of budget ``b1``, held by customer ``c1``, key ``k1`` and the membership of ``u1`` in ``t1``. Each
    auth read takes its row when it starts and answers only once ``release`` is set; the holder reads of a budget update
    and its write answer at once."""

    def __init__(self, hashed_key: str) -> None:
        from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_EndUserTable, LiteLLM_TeamMembership

        self.all_started = asyncio.Event()
        self.release = asyncio.Event()
        self._started = 0
        old_budget: Final = LiteLLM_BudgetTable(budget_id="b1", max_budget=50.0, rpm_limit=100)
        self.customer: Final = LiteLLM_EndUserTable(
            user_id="c1", blocked=False, budget_id="b1", litellm_budget_table=old_budget
        )
        self.membership: Final = LiteLLM_TeamMembership(
            user_id="u1", team_id="t1", budget_id="b1", litellm_budget_table=old_budget
        )
        self.key: Final = UserAPIKeyAuth(token=hashed_key, budget_id="b1", max_budget=50.0)
        self.budget_writes: Final = AsyncMock(side_effect=lambda *, where, data: {**where, **data})
        nothing_on_it: Final = types.SimpleNamespace(find_many=AsyncMock(return_value=[]))
        self.db: Final = types.SimpleNamespace(
            litellm_budgettable=types.SimpleNamespace(update=self.budget_writes),
            litellm_endusertable=types.SimpleNamespace(find_unique=self._held_customer, find_many=self._customers),
            litellm_teammembership=types.SimpleNamespace(
                find_first=self._held_membership, find_many=AsyncMock(return_value=[self.membership])
            ),
            litellm_verificationtoken=_KeyRows(
                types.SimpleNamespace(
                    token=hashed_key, budget_id="b1", organization_id=None, team_id=None, user_id=None
                )
            ),
            litellm_organizationtable=nothing_on_it,
            litellm_projecttable=nothing_on_it,
            litellm_tagtable=nothing_on_it,
            litellm_modelaccessgroupbudgettable=nothing_on_it,
        )

    async def _held(self, row: object) -> object:
        self._started += 1
        if self._started == 3:
            self.all_started.set()
        await self.release.wait()
        return row

    async def _held_customer(self, where: object, include: object = None) -> object:
        return await self._held(self.customer)

    async def _held_membership(self, where: object, include: object = None) -> object:
        return await self._held(self.membership)

    async def _customers(self, **_: object) -> list[object]:
        # The holder read of the budget and the restricted-customer registry both find c1
        return [self.customer]

    async def get_data(self, token: str, table_name: str, **_: object) -> object:
        return await self._held(self.key)

    async def attempt_db_reconnect(self, **_: object) -> bool:
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize("pods", ["one_pod", "two_pods"])
async def test_a_read_in_flight_across_a_budget_update_leaves_no_holder_the_old_limit(monkeypatch, pods: str):
    """A customer, a key and a team membership on the budget are each being read when the budget is lowered. Each read
    took its row before the write and stores it after, so without a revocation before the write each would keep the
    old limit as a last good entry, served in the next outage."""
    from typing import cast

    from litellm.proxy._types import LiteLLM_EndUserTable, LiteLLM_TeamMembership
    from litellm.proxy.auth.auth_checks import get_end_user_object, get_key_object
    from litellm.proxy.auth.user_api_key_auth import _read_team_membership  # pyright: ignore[reportPrivateUsage]  # the admission check's read
    from litellm.proxy.common_utils.user_api_key_cache import (
        UserApiKeyCache,
        end_user_cache_key,
        recall_last_good,
        team_membership_auth_cache_key,
    )
    from litellm.proxy.utils import PrismaClient
    from tests.unit.caching.fake_redis_cache import fake_redis_cache

    hashed_key: Final = "hashed-k1"
    database: Final = _HeldAuthReads(hashed_key)
    prisma: Final = cast(PrismaClient, database)
    redis: Final = fake_redis_cache()
    reading_pod: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
    updating_pod: Final = (
        reading_pod if pods == "one_pod" else UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
    )
    monkeypatch.setattr(ps, "prisma_client", database)
    monkeypatch.setattr(ps, "user_api_key_cache", updating_pod)
    in_flight: Final = asyncio.gather(
        get_end_user_object(end_user_id="c1", prisma_client=prisma, user_api_key_cache=reading_pod),
        get_key_object(hashed_token=hashed_key, prisma_client=prisma, user_api_key_cache=reading_pod),
        _read_team_membership(prisma, reading_pod, team_id="t1", user_id="u1"),
    )
    await database.all_started.wait()

    await _update_budget_directly("b1", 5.0)
    database.release.set()
    await in_flight

    database.budget_writes.assert_awaited_once()
    assert {
        "customer": await recall_last_good(reading_pod, end_user_cache_key("c1"), LiteLLM_EndUserTable),
        "key": await recall_last_good(reading_pod, hashed_key, UserAPIKeyAuth),
        "team membership": await recall_last_good(
            reading_pod, team_membership_auth_cache_key(team_id="t1", user_id="u1"), LiteLLM_TeamMembership
        ),
    } == {"customer": None, "key": None, "team membership": None}


class _HandedBackKeyRows(_KeyRows):
    """The key table, remembering every row it hands back."""

    def __init__(self, *rows: types.SimpleNamespace) -> None:
        super().__init__(*rows)
        self.handed_back: list[types.SimpleNamespace] = []

    async def find_many(self, where: dict) -> list[types.SimpleNamespace]:
        rows: Final = await super().find_many(where)
        self.handed_back.extend(rows)
        return rows


@pytest.mark.asyncio
async def test_a_budget_on_memberships_in_two_teams_reads_each_members_keys_in_that_team_only(
    client_and_mocks, monkeypatch
):
    """A key carries the member budget of its own team's membership, so the revocation reads each member's keys in that
    team only: a member's keys in other teams are never loaded, however many teams the member is in."""
    from litellm.proxy._types import LiteLLM_TeamMembership
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.management_endpoints.budget_management_endpoints import read_budget_holders

    _, mock_prisma, _ = client_and_mocks
    mock_prisma.db.litellm_teammembership.find_many = AsyncMock(
        return_value=[
            LiteLLM_TeamMembership(user_id="u1", team_id="t1", budget_id="b1"),
            LiteLLM_TeamMembership(user_id="u2", team_id="t2", budget_id="b1"),
        ]
    )

    def key(token: str, user_id: str, team_id: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            token=token, budget_id=None, organization_id=None, team_id=team_id, user_id=user_id
        )

    keys: Final = _HandedBackKeyRows(
        key("u1-in-t1", "u1", "t1"),
        key("u1-in-t2", "u1", "t2"),
        key("u2-in-t2", "u2", "t2"),
        key("u2-in-t1", "u2", "t1"),
    )
    mock_prisma.db.litellm_verificationtoken = keys
    monkeypatch.setattr(ps, "user_api_key_cache", UserApiKeyCache(serve_last_good_entries=True))

    holders: Final = await read_budget_holders("b1")

    assert sorted(row.token for row in keys.handed_back) == ["u1-in-t1", "u2-in-t2"]
    assert sorted(holders.hashed_tokens) == ["u1-in-t1", "u2-in-t2"]
