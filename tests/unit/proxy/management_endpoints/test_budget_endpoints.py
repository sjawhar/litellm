# tests/test_budget_endpoints.py

import json
import types
from datetime import datetime, timedelta, timezone
from typing import Final
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy.proxy_server import app
from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles, CommonProxyErrors



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
    """Memberships, organizations, projects, customers, tags and model access groups carry their budget row, and the
    default customer and team-member budgets are cached by id. A last good entry never expires, so a changed budget
    must evict all of them, or a database outage enforces the old limits."""
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
    )
    for key in keys:
        await remember_last_good(cache, key, LiteLLM_BudgetTable(budget_id="budget_shared", max_budget=50.0))
    monkeypatch.setattr(ps, "user_api_key_cache", cache)

    resp = client.post("/budget/update", json={"budget_id": "budget_shared", "max_budget": 5.0})

    assert resp.status_code == 200, resp.text
    assert {key: await recall_last_good(cache, key, LiteLLM_BudgetTable) for key in keys} == dict.fromkeys(keys)
    for table in (
        "litellm_teammembership",
        "litellm_organizationtable",
        "litellm_projecttable",
        "litellm_endusertable",
        "litellm_tagtable",
        "litellm_modelaccessgroupbudgettable",
    ):
        getattr(mock_prisma.db, table).find_many.assert_awaited_once_with(where={"budget_id": "budget_shared"})


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
