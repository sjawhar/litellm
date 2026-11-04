import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from litellm.constants import LAST_GOOD_TOMBSTONE_TTL_SECONDS
from litellm.proxy._types import (
    BlockUsers,
    LiteLLM_BudgetTable,
    LiteLLM_BudgetTableFull,
    LiteLLM_EndUserTable,
    LitellmUserRoles,
    NewCustomerRequest,
    ProxyException,
    UpdateCustomerRequest,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import LAST_GOOD_LATE_STORE_MARGIN_SECONDS, UserApiKeyCache
from litellm.proxy.management_endpoints.customer_endpoints import _should_update_field, router
from tests.unit.caching.fake_redis_cache import FakeRedisCache, fake_redis_cache
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
)
from litellm.types.proxy.management_endpoints.customer_endpoints import (
    BlockUsersResponse,
    CustomerResponse,
    DeleteCustomersResponse,
    UnblockUsersResponse,
)

app = FastAPI()


@app.exception_handler(ProxyException)
async def openai_exception_handler(request: Request, exc: ProxyException):
    headers = exc.headers
    error_dict = exc.to_dict()
    return JSONResponse(
        status_code=(int(exc.code) if exc.code else status.HTTP_500_INTERNAL_SERVER_ERROR),
        content={"error": error_dict},
        headers=headers,
    )


app.include_router(router)
client = TestClient(app)


@pytest.mark.parametrize(
    ("field", "value", "sent_fields", "expected"),
    (
        ("models", None, frozenset({"models"}), False),
        ("blocked", False, frozenset({"blocked"}), True),
        ("blocked", False, frozenset(), False),
        ("models", [], frozenset({"models"}), True),
        ("models", [], frozenset(), False),
        ("metadata", {}, frozenset({"metadata"}), False),
        ("object_permission", {"mcp_servers": ["s1"]}, frozenset(), True),
        ("object_permission", {"mcp_servers": ["s1"]}, frozenset({"object_permission"}), True),
        ("metadata", ["m1"], frozenset(), True),
        ("models", ["m1"], frozenset(), True),
        ("max_budget", 0, frozenset({"max_budget"}), False),
        ("max_budget", 5.0, frozenset(), True),
        ("alias", "a", frozenset(), True),
    ),
    ids=(
        "null-models",
        "explicit-false",
        "omitted-false",
        "clear-models",
        "omitted-empty-models",
        "empty-metadata",
        "nonempty-object-permission-omitted",
        "nonempty-object-permission-sent",
        "nonempty-list-other-field",
        "nonempty-models",
        "zero-budget",
        "nonzero-budget",
        "alias",
    ),
)
def test_should_update_field(
    field: str,
    value: object,
    sent_fields: frozenset[str],
    expected: bool,
) -> None:
    assert _should_update_field(field, value, sent_fields) is expected


@pytest.fixture
def mock_prisma_client():
    with patch("litellm.proxy.proxy_server.prisma_client") as mock:
        yield mock


@pytest.fixture
def mock_user_api_key_auth():
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    try:
        yield
    finally:
        app.dependency_overrides = original_overrides


def test_update_customer_success(mock_prisma_client, mock_user_api_key_auth):
    # Mock the database responses
    mock_end_user = LiteLLM_EndUserTable(user_id="test-user-1", alias="Test User", blocked=False)
    updated_mock_end_user = LiteLLM_EndUserTable(user_id="test-user-1", alias="Updated Test User", blocked=False)

    # Mock the find_first response
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=mock_end_user)

    # Mock the update response
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=updated_mock_end_user)

    # Test data
    test_data = {"user_id": "test-user-1", "alias": "Updated Test User"}

    # Make the request
    response = client.post("/customer/update", json=test_data, headers={"Authorization": "Bearer test-key"})

    # Assert response
    assert response.status_code == 200
    assert response.json()["user_id"] == "test-user-1"
    assert response.json()["alias"] == "Updated Test User"


def test_update_customer_unblock(mock_prisma_client, mock_user_api_key_auth):
    mock_end_user = LiteLLM_EndUserTable(user_id="test-user-1", blocked=True)
    updated_mock_end_user = LiteLLM_EndUserTable(user_id="test-user-1", blocked=False)

    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=mock_end_user)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=updated_mock_end_user)

    response = client.post(
        "/customer/update",
        json={"user_id": "test-user-1", "blocked": False},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    assert response.json()["blocked"] is False
    update_mock = mock_prisma_client.db.litellm_endusertable.update
    update_mock.assert_called_once()
    assert update_mock.call_args.kwargs["data"]["blocked"] is False


def test_update_customer_keeps_blocked_when_omitted(mock_prisma_client, mock_user_api_key_auth):
    """
    Regression test: updating a blocked customer without supplying `blocked`
    must NOT reset it to unblocked. `blocked=False` is the model default and
    should only be applied when explicitly provided by the caller.
    """
    mock_end_user = LiteLLM_EndUserTable(user_id="test-user-1", blocked=True)
    updated_mock_end_user = LiteLLM_EndUserTable(user_id="test-user-1", blocked=True)

    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=mock_end_user)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=updated_mock_end_user)

    response = client.post(
        "/customer/update",
        json={"user_id": "test-user-1", "alias": "Updated Test User"},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    update_mock = mock_prisma_client.db.litellm_endusertable.update
    update_mock.assert_called_once()
    assert "blocked" not in update_mock.call_args.kwargs["data"]


def test_update_customer_not_found(mock_prisma_client, mock_user_api_key_auth):
    """
    Test that update_end_user raises a 404 ProxyException when user_id does not exist.
    """
    # Mock the database response to return None (user not found)
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=None)

    # Test data
    test_data = {"user_id": "non-existent-user", "alias": "Test User"}

    # Make the request
    response = client.post(
        "/customer/update",
        json=test_data,
        headers={"Authorization": "Bearer test-key"},
    )

    # Assert response
    assert response.status_code == 404
    response_json = response.json()
    assert "error" in response_json
    assert response_json["error"]["message"] == "End User Id=non-existent-user does not exist in db"
    assert response_json["error"]["type"] == "not_found"
    assert response_json["error"]["param"] == "user_id"
    assert response_json["error"]["code"] == "404"


def test_info_customer_not_found(mock_prisma_client, mock_user_api_key_auth):
    """
    Test that end_user_info raises a 404 ProxyException when end_user_id does not exist.
    """
    # Mock the database response to return None (user not found)
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=None)

    # Make the request
    response = client.get(
        "/customer/info?end_user_id=non-existent-user",
        headers={"Authorization": "Bearer test-key"},
    )

    # Assert response
    assert response.status_code == 404
    response_json = response.json()
    assert "error" in response_json
    assert response_json["error"]["message"] == "End User Id=non-existent-user does not exist in db"
    assert response_json["error"]["type"] == "not_found"
    assert response_json["error"]["param"] == "end_user_id"
    assert response_json["error"]["code"] == "404"


def test_delete_customer_not_found(mock_prisma_client, mock_user_api_key_auth):
    """
    Test that delete_end_user raises a 404 ProxyException when user_ids do not exist.
    """
    # Mock the database response to return empty list (no users found)
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(return_value=[])

    # Test data
    test_data = {"user_ids": ["non-existent-user-1", "non-existent-user-2"]}

    # Make the request
    response = client.post(
        "/customer/delete",
        json=test_data,
        headers={"Authorization": "Bearer test-key"},
    )

    # Assert response
    assert response.status_code == 404
    response_json = response.json()
    assert "error" in response_json
    assert "do not exist in db" in response_json["error"]["message"]
    assert "non-existent-user-1" in response_json["error"]["message"]
    assert response_json["error"]["type"] == "not_found"
    assert response_json["error"]["param"] == "user_ids"
    assert response_json["error"]["code"] == "404"


def test_error_schema_consistency(mock_prisma_client, mock_user_api_key_auth):
    """
    Test that all customer endpoints return the same error schema format.
    All ProxyException errors should have: message, type, param, and code fields.
    """

    def validate_error_schema(response_json):
        assert "error" in response_json, "Response should have 'error' key"
        error = response_json["error"]
        assert "message" in error, "Error should have 'message' field"
        assert "type" in error, "Error should have 'type' field"
        assert "param" in error, "Error should have 'param' field"
        assert "code" in error, "Error should have 'code' field"
        assert isinstance(error["message"], str), "message should be a string"
        assert isinstance(error["type"], str), "type should be a string"
        assert isinstance(error["code"], str), "code should be a string"
        return error

    # Test /customer/info - not found error
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=None)
    response = client.get(
        "/customer/info?end_user_id=non-existent",
        headers={"Authorization": "Bearer test-key"},
    )
    error = validate_error_schema(response.json())
    assert error["type"] == "not_found"
    assert error["code"] == "404"

    # Test /customer/update - not found error
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=None)
    response = client.post(
        "/customer/update",
        json={"user_id": "non-existent", "alias": "Test"},
        headers={"Authorization": "Bearer test-key"},
    )
    error = validate_error_schema(response.json())
    assert error["type"] == "not_found"
    assert error["code"] == "404"

    # Test /customer/delete - not found error
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(return_value=[])
    response = client.post(
        "/customer/delete",
        json={"user_ids": ["non-existent"]},
        headers={"Authorization": "Bearer test-key"},
    )
    error = validate_error_schema(response.json())
    assert error["type"] == "not_found"
    assert error["code"] == "404"

    # Test /customer/new - duplicate user error
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(
        side_effect=Exception("Unique constraint failed on the fields: (`user_id`)")
    )
    response = client.post(
        "/customer/new",
        json={"user_id": "existing-user"},
        headers={"Authorization": "Bearer test-key"},
    )
    error = validate_error_schema(response.json())
    assert error["type"] == "bad_request"
    assert error["code"] == "400"


def test_customer_endpoints_error_schema_consistency(mock_prisma_client, mock_user_api_key_auth):
    """
    Test the exact scenarios from the curl examples provided.

    Scenario 1: GET /end_user/info with non-existent user
    OLD (incorrect): {"detail":{"error":"End User Id=... does not exist in db"}}
    NEW (correct):   {"error":{"message":"...","type":"not_found","param":"end_user_id","code":"404"}}

    Scenario 2: POST /end_user/new with existing user
    Expected:        {"error":{"message":"...","type":"bad_request","param":"user_id","code":"400"}}

    Both should use the same error format structure.
    """

    # Scenario 1: GET /end_user/info with non-existent user
    # Should return 404 with proper error schema
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=None)

    response1 = client.get(
        "/end_user/info?end_user_id=fake-test-end-user-michaels-local-testng",
        headers={"Authorization": "Bearer test-key"},
    )

    assert response1.status_code == 404, "Should return 404 for non-existent user"
    response1_json = response1.json()

    # Should have the correct format with {"error": {...}}
    assert "error" in response1_json, "Should have top-level 'error' key"
    error1 = response1_json["error"]
    assert "message" in error1, "Error should have 'message' field"
    assert "type" in error1, "Error should have 'type' field"
    assert "param" in error1, "Error should have 'param' field"
    assert "code" in error1, "Error should have 'code' field"
    assert error1["type"] == "not_found"
    assert error1["code"] == "404"
    assert "does not exist in db" in error1["message"]

    # Scenario 2: POST /end_user/new with existing user
    # Should return 400 with proper error schema
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(
        side_effect=Exception("Unique constraint failed on the fields: (`user_id`)")
    )

    response2 = client.post(
        "/end_user/new",
        json={
            "user_id": "fake-test-end-user-michaels-local-testing",
            "budget_id": "Tier0",
        },
        headers={"Authorization": "Bearer test-key"},
    )

    assert response2.status_code == 400, "Should return 400 for duplicate user"
    response2_json = response2.json()

    # Should have the same error structure as Scenario 1
    assert "error" in response2_json, "Should have top-level 'error' key"
    error2 = response2_json["error"]
    assert "message" in error2, "Error should have 'message' field"
    assert "type" in error2, "Error should have 'type' field"
    assert "param" in error2, "Error should have 'param' field"
    assert "code" in error2, "Error should have 'code' field"
    assert error2["type"] == "bad_request"
    assert error2["code"] == "400"
    assert "Customer already exists" in error2["message"]

    # Verify both errors have the same schema structure
    assert set(error1.keys()) == set(error2.keys()), "Both errors should have the same top-level keys"

    # Both should have string values for all fields
    for key in ["message", "type", "code"]:
        assert isinstance(error1[key], str), f"error1[{key}] should be a string"
        assert isinstance(error2[key], str), f"error2[{key}] should be a string"


EXPECTED_RESPONSE_MODELS = {
    "/customer/block": BlockUsersResponse,
    "/customer/unblock": UnblockUsersResponse,
    "/customer/new": CustomerResponse,
    "/customer/update": CustomerResponse,
    "/customer/delete": DeleteCustomersResponse,
    "/customer/info": CustomerResponse,
    "/customer/list": list[CustomerResponse],
    "/customer/daily/activity": SpendAnalyticsPaginatedResponse,
}


@pytest.mark.parametrize("path, expected_model", EXPECTED_RESPONSE_MODELS.items())
def test_customer_routes_declare_response_model(path, expected_model):
    """
    Every public /customer/* operation must declare a typed response_model so
    the generated OpenAPI schema documents the response body. Regression for the
    OpenAPI response-type coverage goal: drop a response_model and this fails.
    """
    route = next(r for r in router.routes if isinstance(r, APIRoute) and r.path == path)
    assert route.response_model == expected_model


def test_customer_new_documented_in_openapi_schema():
    """
    The response_model must surface in the OpenAPI schema as a concrete ref, not
    an empty/default response. This is what the coverage metric measures.
    """
    schema = app.openapi()["paths"]["/customer/new"]["post"]
    json_schema = schema["responses"]["200"]["content"]["application/json"]["schema"]
    assert json_schema["$ref"].endswith("/CustomerResponse")


def test_update_customer_response_preserves_budget_id(mock_prisma_client, mock_user_api_key_auth):
    """
    Regression for the response_model field-stripping concern: budget_id is a real
    column on the end-user table that /customer/update echoes. response_model=
    LiteLLM_EndUserTable must NOT drop it, so budget_id stays in LiteLLM_EndUserTable.
    """
    existing = LiteLLM_EndUserTable(user_id="cust-1", blocked=False)
    updated = LiteLLM_EndUserTable(user_id="cust-1", blocked=False, budget_id="budget-123")
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=existing)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=updated)

    response = client.post(
        "/customer/update",
        json={"user_id": "cust-1", "budget_id": "budget-123"},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    assert response.json()["budget_id"] == "budget-123"


@pytest.mark.parametrize(
    "budget_payload",
    [{"max_budget": None}, {}],
    ids=["explicit-null", "omitted"],
)
def test_update_customer_budget_omission_and_null_preserve_existing_budget(
    mock_prisma_client, mock_user_api_key_auth, budget_payload
):
    from litellm.proxy._types import LiteLLM_BudgetTable

    class BudgetState:
        def __init__(self) -> None:
            self.max_budget: float | None = 100.0

        def store(self, data) -> None:
            self.max_budget = data.get("max_budget", self.max_budget)

    budget_state = BudgetState()

    def end_user_row():
        return LiteLLM_EndUserTable(
            user_id="cust-1",
            blocked=False,
            budget_id="budget-1",
            litellm_budget_table=LiteLLM_BudgetTable(budget_id="budget-1", max_budget=budget_state.max_budget),
        )

    def response_row():
        row = MagicMock()
        row.model_dump.return_value = {
            "user_id": "cust-1",
            "blocked": False,
            "budget_id": "budget-1",
            "litellm_budget_table": {
                "budget_id": "budget-1",
                "max_budget": budget_state.max_budget,
                "created_at": "2024-01-01T00:00:00",
            },
        }
        return row

    async def update_budget(*, where, data):
        budget_state.store(data)
        return LiteLLM_BudgetTable(budget_id="budget-1", max_budget=budget_state.max_budget)

    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=end_user_row())
    mock_prisma_client.db.litellm_budgettable.update = AsyncMock(side_effect=update_budget)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(side_effect=lambda **_: response_row())

    response = client.post(
        "/customer/update",
        json={"user_id": "cust-1", **budget_payload},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["litellm_budget_table"]["max_budget"] == 100.0


def test_update_customer_response_keeps_nested_budget_server_fields(mock_prisma_client, mock_user_api_key_auth):
    """
    Faithfulness regression: /customer/update embeds the full budget row. The
    response_model must keep the server-managed budget fields the endpoint used
    to return (budget_reset_at, created_at) instead of the narrow write-allowlist
    shape. The intentionally-internal audit fields (created_by/updated_by) stay out.
    """
    existing = LiteLLM_EndUserTable(user_id="cust-1", blocked=False)
    raw_row = MagicMock()
    raw_row.model_dump.return_value = {
        "user_id": "cust-1",
        "blocked": False,
        "alias": "renamed",
        "spend": 0.0,
        "allowed_model_region": None,
        "default_model": None,
        "budget_id": "b-1",
        "object_permission_id": None,
        "object_permission": None,
        "litellm_budget_table": {
            "budget_id": "b-1",
            "max_budget": 10.0,
            "budget_duration": "30d",
            "budget_reset_at": "2024-02-01T00:00:00",
            "created_at": "2024-01-01T00:00:00",
            "created_by": "admin",
            "updated_at": "2024-01-02T00:00:00",
            "updated_by": "admin",
        },
    }
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=existing)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=raw_row)

    response = client.post(
        "/customer/update",
        json={"user_id": "cust-1", "alias": "renamed"},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    budget = response.json()["litellm_budget_table"]
    assert budget["budget_reset_at"] == "2024-02-01T00:00:00"
    assert budget["created_at"] == "2024-01-01T00:00:00"
    assert "created_by" not in budget
    assert "updated_by" not in budget


def test_block_customer_success_serializes_through_response_model(mock_prisma_client, mock_user_api_key_auth):
    """
    /customer/block returns {"blocked_users": [<end user rows>]}. With
    response_model=BlockUsersResponse, a shape mismatch would raise a 500
    ResponseValidationError, so a clean 200 proves the model matches runtime output.
    """
    blocked_row = LiteLLM_EndUserTable(user_id="blocked-1", blocked=True)
    mock_prisma_client.db.litellm_endusertable.upsert = AsyncMock(return_value=blocked_row)

    response = client.post(
        "/customer/block",
        json={"user_ids": ["blocked-1"]},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["blocked_users"][0]["user_id"] == "blocked-1"
    assert body["blocked_users"][0]["blocked"] is True


def test_delete_customer_success_serializes_through_response_model(mock_prisma_client, mock_user_api_key_auth):
    """
    /customer/delete returns {"deleted_customers": <int>, "message": <str>}.
    response_model=DeleteCustomersResponse enforces that exact shape.
    """
    existing = [
        LiteLLM_EndUserTable(user_id="u1", blocked=False),
        LiteLLM_EndUserTable(user_id="u2", blocked=False),
    ]
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(return_value=existing)
    mock_prisma_client.db.litellm_endusertable.delete_many = AsyncMock(return_value=2)

    response = client.post(
        "/customer/delete",
        json={"user_ids": ["u1", "u2"]},
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "deleted_customers": 2,
        "message": "Successfully deleted customers with ids: ['u1', 'u2']",
    }


@pytest.mark.asyncio
async def test_get_customer_daily_activity_admin_param_passing(monkeypatch):
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import customer_endpoints
    from litellm.proxy.management_endpoints.customer_endpoints import (
        get_customer_daily_activity,
    )

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(customer_endpoints, "get_daily_activity", get_daily_activity_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin1")
    result = await get_customer_daily_activity(
        end_user_ids="end-user-1,end-user-2",
        start_date="2024-01-01",
        end_date="2024-01-31",
        model="gpt-4",
        api_key="test-key",
        page=2,
        page_size=5,
        exclude_end_user_ids="end-user-3",
        user_api_key_dict=auth,
    )

    get_daily_activity_mock.assert_awaited_once()
    kwargs = get_daily_activity_mock.call_args.kwargs
    assert kwargs["table_name"] == "litellm_dailyenduserspend"
    assert kwargs["entity_id_field"] == "end_user_id"
    assert kwargs["entity_id"] == ["end-user-1", "end-user-2"]
    assert kwargs["exclude_entity_ids"] == ["end-user-3"]
    assert kwargs["start_date"] == "2024-01-01"
    assert kwargs["end_date"] == "2024-01-31"
    assert kwargs["model"] == "gpt-4"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["page"] == 2
    assert kwargs["page_size"] == 5

    assert result is mocked_response


@pytest.mark.asyncio
async def test_get_customer_daily_activity_with_end_user_aliases(monkeypatch):
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import customer_endpoints
    from litellm.proxy.management_endpoints.customer_endpoints import (
        get_customer_daily_activity,
    )

    mock_prisma_client = AsyncMock()
    mock_end_user1 = MagicMock()
    mock_end_user1.user_id = "end-user-1"
    mock_end_user1.alias = "Customer One"
    mock_end_user2 = MagicMock()
    mock_end_user2.user_id = "end-user-2"
    mock_end_user2.alias = "Customer Two"

    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(return_value=[mock_end_user1, mock_end_user2])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(customer_endpoints, "get_daily_activity", get_daily_activity_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin1")
    await get_customer_daily_activity(
        end_user_ids="end-user-1,end-user-2",
        start_date="2024-01-01",
        end_date="2024-01-31",
        model=None,
        api_key=None,
        page=1,
        page_size=10,
        exclude_end_user_ids=None,
        user_api_key_dict=auth,
    )

    kwargs = get_daily_activity_mock.call_args.kwargs
    assert kwargs["entity_metadata_field"] == {
        "end-user-1": {"alias": "Customer One"},
        "end-user-2": {"alias": "Customer Two"},
    }


@pytest.mark.asyncio
async def test_get_customer_daily_activity_non_admin_is_rejected(monkeypatch):
    """
    Security regression: any non-admin caller must receive 401 from
    /customer/daily/activity and /end_user/daily/activity.

    Before this fix, the endpoint performed no role check. A caller with
    user_role=INTERNAL_USER could omit end_user_ids, causing entity_id=None
    to flow into get_daily_activity where the SQL builder treats it as no
    filter — returning every tenant's spend across the full
    LiteLLM_DailyEndUserSpend table.

    LiteLLM_EndUserTable has no per-tenant ownership column, so non-admin
    scoping is not possible. The correct fix is admin-only, matching the
    existing /customer/list gate.
    """
    from litellm.proxy.management_endpoints import customer_endpoints
    from litellm.proxy.management_endpoints.customer_endpoints import (
        get_customer_daily_activity,
    )

    mock_prisma_client = MagicMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    get_daily_activity_mock = AsyncMock()
    monkeypatch.setattr(customer_endpoints, "get_daily_activity", get_daily_activity_mock)

    non_admin_key = UserAPIKeyAuth(
        user_id="regular-user-abc",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_customer_daily_activity(
            end_user_ids=None,
            start_date="2025-01-01",
            end_date="2025-01-31",
            model=None,
            api_key=None,
            page=1,
            page_size=10,
            exclude_end_user_ids=None,
            user_api_key_dict=non_admin_key,
        )

    assert exc_info.value.status_code == 401
    assert "Admin-only endpoint" in str(exc_info.value.detail)
    get_daily_activity_mock.assert_not_called()


@pytest.mark.asyncio
async def test_get_customer_daily_activity_service_account_key_is_rejected(monkeypatch):
    """
    Security regression: service-account keys (user_id=None, role=INTERNAL_USER)
    must be rejected at the admin gate before reaching get_daily_activity.

    A service-account key with end_user_ids omitted is the worst-case caller:
    entity_id=None and no user identity to scope by — the SQL builder would
    return the full LiteLLM_DailyEndUserSpend table with no WHERE clause.
    """
    from litellm.proxy.management_endpoints import customer_endpoints
    from litellm.proxy.management_endpoints.customer_endpoints import (
        get_customer_daily_activity,
    )

    mock_prisma_client = MagicMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    get_daily_activity_mock = AsyncMock()
    monkeypatch.setattr(customer_endpoints, "get_daily_activity", get_daily_activity_mock)

    service_account_key = UserAPIKeyAuth(
        user_id=None,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_customer_daily_activity(
            end_user_ids=None,
            start_date="2025-01-01",
            end_date="2025-01-31",
            model=None,
            api_key=None,
            page=1,
            page_size=10,
            exclude_end_user_ids=None,
            user_api_key_dict=service_account_key,
        )

    assert exc_info.value.status_code == 401
    assert "Admin-only endpoint" in str(exc_info.value.detail)
    get_daily_activity_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Characterization (golden-master) tests.
#
# These lock the EXACT JSON body every customer-object endpoint returns today,
# so a type-safety refactor of the handlers is only allowed to land if it
# reproduces these byte for byte. The input below is what a Prisma row's
# .model_dump() yields (full nested budget incl. audit fields + object_permission
# incl. reverse relations); the expected output is what the live endpoint emits.
# ---------------------------------------------------------------------------

_FULL_DB_ROW = {
    "user_id": "c1",
    "blocked": False,
    "alias": "Acme",
    "spend": 1.5,
    "allowed_model_region": None,
    "default_model": None,
    "models": ["allowed-model"],
    "budget_id": "b1",
    "object_permission_id": "p1",
    "litellm_budget_table": {
        "budget_id": "b1",
        "max_budget": 10.0,
        "soft_budget": None,
        "max_parallel_requests": None,
        "tpm_limit": None,
        "rpm_limit": None,
        "model_max_budget": None,
        "budget_duration": "30d",
        "allowed_models": [],
        "budget_reset_at": "2024-02-01T00:00:00",
        "created_at": "2024-01-01T00:00:00",
        "created_by": "admin",
        "updated_at": "2024-01-02T00:00:00",
        "updated_by": "admin",
    },
    "object_permission": {
        "object_permission_id": "p1",
        "mcp_servers": ["s1"],
        "mcp_access_groups": [],
        "mcp_tool_permissions": None,
        "vector_stores": [],
        "agents": [],
        "agent_access_groups": [],
        "models": [],
        "mcp_toolsets": None,
        "blocked_tools": [],
        "search_tools": [],
        "teams": [{"team_id": "t1"}],
        "users": [{"user_id": "x"}],
        "end_users": [],
        "organizations": [],
        "verification_tokens": [],
    },
}

_EXPECTED_CUSTOMER = {
    "user_id": "c1",
    "blocked": False,
    "alias": "Acme",
    "spend": 1.5,
    "allowed_model_region": None,
    "default_model": None,
    "models": ["allowed-model"],
    "budget_id": "b1",
    "litellm_budget_table": {
        "budget_id": "b1",
        "soft_budget": None,
        "max_budget": 10.0,
        "max_parallel_requests": None,
        "tpm_limit": None,
        "rpm_limit": None,
        "tpd_limit": None,
        "model_max_budget": None,
        "budget_duration": "30d",
        "allowed_models": [],
        "temp_budget_increase": None,
        "temp_budget_expiry": None,
        "budget_reset_at": "2024-02-01T00:00:00",
        "created_at": "2024-01-01T00:00:00",
    },
    "object_permission_id": "p1",
    "object_permission": {
        "object_permission_id": "p1",
        "mcp_servers": ["s1"],
        "mcp_access_groups": [],
        "mcp_tool_permissions": None,
        "vector_stores": [],
        "agents": [],
        "agent_access_groups": [],
        "models": [],
        "mcp_toolsets": None,
        "blocked_tools": [],
        "search_tools": [],
        "mcp_tool_search_enabled": None,
        "skills": None,
    },
}


def _row(dump: dict) -> MagicMock:
    row = MagicMock()
    row.model_dump.return_value = dump
    return row


def test_char_info_body(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(return_value=_row(_FULL_DB_ROW))
    response = client.get("/customer/info?end_user_id=c1", headers={"Authorization": "Bearer k"})
    assert response.status_code == 200
    assert response.json() == _EXPECTED_CUSTOMER


def test_char_list_body(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(return_value=[_row(_FULL_DB_ROW)])
    response = client.get("/customer/list", headers={"Authorization": "Bearer k"})
    assert response.status_code == 200
    assert response.json() == [_EXPECTED_CUSTOMER]


def test_char_new_body(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(return_value=_row(_FULL_DB_ROW))
    response = client.post("/customer/new", json={"user_id": "c1"}, headers={"Authorization": "Bearer k"})
    assert response.status_code == 200
    assert response.json() == _EXPECTED_CUSTOMER


def test_customer_new_forwards_models_to_db(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(return_value=_row(_FULL_DB_ROW))

    response = client.post(
        "/customer/new",
        json={"user_id": "c1", "models": ["allowed-model"]},
        headers={"Authorization": "Bearer k"},
    )

    assert response.status_code == 200, response.text
    assert mock_prisma_client.db.litellm_endusertable.create.call_args.kwargs["data"]["models"] == ["allowed-model"]


@pytest.mark.parametrize("bad_duration", ["0s", "-5m"])
def test_customer_new_rejects_a_duration_that_never_advances(
    mock_prisma_client, mock_user_api_key_auth, bad_duration
):
    """A zero-length window resets to "now", leaving the customer's budget row
    permanently due for the reset job to re-read every tick."""
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(return_value=_row(_FULL_DB_ROW))

    response = client.post(
        "/customer/new",
        json={"user_id": "c1", "max_budget": 10.0, "budget_duration": bad_duration},
        headers={"Authorization": "Bearer k"},
    )

    assert response.status_code == 400, response.text
    assert "Invalid budget_duration" in response.text
    mock_prisma_client.db.litellm_endusertable.create.assert_not_awaited()


def test_customer_new_accepts_a_normal_duration(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(return_value=_row(_FULL_DB_ROW))
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=_row({"budget_id": "b1", "max_budget": 10.0})
    )

    response = client.post(
        "/customer/new",
        json={"user_id": "c1", "max_budget": 10.0, "budget_duration": "30d"},
        headers={"Authorization": "Bearer k"},
    )

    assert response.status_code == 200, response.text


def test_char_update_body(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=_row({"user_id": "c1", "blocked": False})
    )
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=_row(_FULL_DB_ROW))
    response = client.post(
        "/customer/update",
        json={"user_id": "c1", "alias": "Acme"},
        headers={"Authorization": "Bearer k"},
    )
    assert response.status_code == 200
    assert response.json() == _EXPECTED_CUSTOMER
    assert "models" not in mock_prisma_client.db.litellm_endusertable.update.call_args.kwargs["data"]


def test_customer_update_clears_models_allowlist(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=_row({"user_id": "c1", "blocked": False})
    )
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=_row(_FULL_DB_ROW))

    response = client.post(
        "/customer/update",
        json={"user_id": "c1", "models": []},
        headers={"Authorization": "Bearer k"},
    )

    assert response.status_code == 200, response.text
    assert mock_prisma_client.db.litellm_endusertable.update.call_args.kwargs["data"]["models"] == []


def test_customer_update_applies_nonempty_object_permission(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=_row({"user_id": "c1", "blocked": False, "object_permission_id": None})
    )
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(return_value=None)
    updated_permission = MagicMock()
    updated_permission.object_permission_id = "permission-1"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(return_value=updated_permission)
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(
        return_value=LiteLLM_EndUserTable(user_id="c1", blocked=False, object_permission_id="permission-1")
    )

    response = client.post(
        "/customer/update",
        json={"user_id": "c1", "object_permission": {"mcp_servers": ["s1"]}},
        headers={"Authorization": "Bearer k"},
    )

    assert response.status_code == 200, response.text
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_awaited_once()
    permission_upsert = mock_prisma_client.db.litellm_objectpermissiontable.upsert.call_args.kwargs
    assert permission_upsert["data"]["create"]["mcp_servers"] == ["s1"]
    assert mock_prisma_client.db.litellm_endusertable.update.call_args.kwargs["data"]["object_permission_id"] == (
        "permission-1"
    )


def test_char_delete_body(mock_prisma_client, mock_user_api_key_auth):
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(
        return_value=[
            LiteLLM_EndUserTable(user_id="c1", blocked=False),
            LiteLLM_EndUserTable(user_id="c2", blocked=False),
        ]
    )
    mock_prisma_client.db.litellm_endusertable.delete_many = AsyncMock(return_value=2)
    response = client.post(
        "/customer/delete",
        json={"user_ids": ["c1", "c2"]},
        headers={"Authorization": "Bearer k"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "deleted_customers": 2,
        "message": "Successfully deleted customers with ids: ['c1', 'c2']",
    }


class _RecordingAuthCache:
    """Captures the keys an endpoint evicts, so tests assert on cache keys not mock plumbing."""

    def __init__(self):
        self.deleted: list[str] = []

    async def async_delete_cache(self, key: str) -> None:
        self.deleted.append(key)


@contextmanager
def _end_user_cache_doubles():
    """Swaps in the auth cache and the cross-worker publisher a customer mutation is expected to hit."""
    recording_cache = _RecordingAuthCache()
    mock_publish = AsyncMock()
    with (
        patch("litellm.proxy.proxy_server.user_api_key_cache", recording_cache),
        patch(
            "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.publish_auth_cache_invalidation",
            mock_publish,
        ),
    ):
        yield recording_cache, mock_publish


def _published_keys(mock_publish) -> list[str]:
    return [call.kwargs["cache_key"] for call in mock_publish.call_args_list]


def test_customer_new_invalidates_end_user_and_registry_caches(mock_prisma_client, mock_user_api_key_auth):
    """
    A customer created on one worker must be visible to every worker's auth path immediately.

    Auth serves end users cache-first, and the cached restricted-id registry is what decides whether
    the row is read at all, so a create that leaves both entries stale means the new customer's
    budget or block goes unenforced until the TTL expires.
    """
    mock_prisma_client.db.litellm_endusertable.create = AsyncMock(return_value=_row(_FULL_DB_ROW))

    with _end_user_cache_doubles() as (recording_cache, mock_publish):
        response = client.post(
            "/customer/new",
            json={"user_id": "c1", "blocked": True},
            headers={"Authorization": "Bearer k"},
        )

    assert response.status_code == 200, response.text
    assert recording_cache.deleted == ["end_user_id:c1", "end_user_validation:c1", "end_user_restricted_registry"]
    assert _published_keys(mock_publish) == ["end_user_id:c1", "end_user_validation:c1", "end_user_restricted_registry"]


def test_customer_update_invalidates_end_user_and_registry_caches(mock_prisma_client, mock_user_api_key_auth):
    """An update can add or drop a budget, block, region or permission, moving the id in the registry."""
    mock_prisma_client.db.litellm_endusertable.find_first = AsyncMock(
        return_value=_row({"user_id": "c1", "blocked": False})
    )
    mock_prisma_client.db.litellm_endusertable.update = AsyncMock(return_value=_row(_FULL_DB_ROW))

    with _end_user_cache_doubles() as (recording_cache, mock_publish):
        response = client.post(
            "/customer/update",
            json={"user_id": "c1", "budget_id": "b1"},
            headers={"Authorization": "Bearer k"},
        )

    assert response.status_code == 200, response.text
    assert recording_cache.deleted == ["end_user_id:c1", "end_user_validation:c1", "end_user_restricted_registry"]
    assert _published_keys(mock_publish) == ["end_user_id:c1", "end_user_validation:c1", "end_user_restricted_registry"]


def test_customer_block_invalidates_end_user_and_registry_caches(mock_prisma_client, mock_user_api_key_auth):
    """Blocking is the one mutation that must take effect instantly; a stale registry keeps serving it."""
    mock_prisma_client.db.litellm_endusertable.upsert = AsyncMock(
        return_value=LiteLLM_EndUserTable(user_id="c1", blocked=True)
    )

    with _end_user_cache_doubles() as (recording_cache, mock_publish):
        response = client.post(
            "/customer/block",
            json={"user_ids": ["c1", "c2"]},
            headers={"Authorization": "Bearer k"},
        )

    assert response.status_code == 200, response.text
    assert recording_cache.deleted == [
        "end_user_id:c1",
        "end_user_validation:c1",
        "end_user_id:c2",
        "end_user_validation:c2",
        "end_user_restricted_registry",
    ]
    assert _published_keys(mock_publish) == [
        "end_user_id:c1",
        "end_user_validation:c1",
        "end_user_id:c2",
        "end_user_validation:c2",
        "end_user_restricted_registry",
    ]


def test_customer_delete_invalidates_end_user_and_registry_caches(mock_prisma_client, mock_user_api_key_auth):
    """Without this a deleted customer keeps its cached budget and block enforced until the TTL expires."""
    mock_prisma_client.db.litellm_endusertable.find_many = AsyncMock(
        return_value=[
            LiteLLM_EndUserTable(user_id="c1", blocked=False),
            LiteLLM_EndUserTable(user_id="c2", blocked=False),
        ]
    )
    mock_prisma_client.db.litellm_endusertable.delete_many = AsyncMock(return_value=2)

    with _end_user_cache_doubles() as (recording_cache, mock_publish):
        response = client.post(
            "/customer/delete",
            json={"user_ids": ["c1", "c2"]},
            headers={"Authorization": "Bearer k"},
        )

    assert response.status_code == 200, response.text
    assert recording_cache.deleted == [
        "end_user_id:c1",
        "end_user_validation:c1",
        "end_user_id:c2",
        "end_user_validation:c2",
        "end_user_restricted_registry",
    ]
    assert _published_keys(mock_publish) == [
        "end_user_id:c1",
        "end_user_validation:c1",
        "end_user_id:c2",
        "end_user_validation:c2",
        "end_user_restricted_registry",
    ]


class _EndUserRows:
    """The end-user table a block, a create or an update writes and auth reads. While ``stopped`` every read fails the
    way an unreachable Postgres does; ``pause_next_registry_read`` holds the next restricted-registry read once it has
    its rows."""

    def __init__(self, *rows: LiteLLM_EndUserTable) -> None:
        self.rows = {row.user_id: row for row in rows}
        self.stopped = False
        self.registry_read_held = asyncio.Event()
        self._release: asyncio.Event | None = None

    def _answer(self) -> None:
        if self.stopped:
            raise httpx.ConnectError("All connection attempts failed")

    def pause_next_registry_read(self) -> asyncio.Event:
        self._release = asyncio.Event()
        return self._release

    async def upsert(self, where: dict[str, str], data: dict[str, dict[str, object]]) -> LiteLLM_EndUserTable:
        self._answer()
        row = LiteLLM_EndUserTable(user_id=where["user_id"], blocked=True)
        self.rows[row.user_id] = row
        return row

    async def create(self, data: dict[str, object], include: object = None) -> LiteLLM_EndUserTable:
        self._answer()
        row = LiteLLM_EndUserTable.model_validate(data)
        self.rows[row.user_id] = row
        return row

    async def update(
        self, where: dict[str, str], data: dict[str, object], include: object = None
    ) -> LiteLLM_EndUserTable:
        self._answer()
        row = self.rows[where["user_id"]].model_copy(update=data)
        self.rows[row.user_id] = row
        return row

    async def find_first(self, where: dict[str, str], include: object = None) -> LiteLLM_EndUserTable | None:
        self._answer()
        return self.rows.get(where["user_id"])

    async def find_unique(self, where: dict[str, str], include: object = None) -> LiteLLM_EndUserTable | None:
        return await self.find_first(where)

    async def find_many(self, where: object = None, take: int | None = None) -> list[LiteLLM_EndUserTable]:
        self._answer()
        restricted = [row for row in self.rows.values() if row.blocked or row.models or row.budget_id]
        release = self._release
        if release is not None:
            self._release = None
            self.registry_read_held.set()
            await release.wait()
        return restricted

    async def delete_many(self, where: dict[str, dict[str, list[str]]]) -> int:
        self._answer()
        doomed: Final = [user_id for user_id in self.rows if user_id in where["user_id"]["in"]]
        for user_id in doomed:
            del self.rows[user_id]
        return len(doomed)


class _BudgetRows:
    """The budget table a customer update writes."""

    def __init__(self, *rows: LiteLLM_BudgetTable) -> None:
        self.rows = {row.budget_id: row for row in rows}

    async def update(self, where: dict[str, str], data: dict[str, object]) -> LiteLLM_BudgetTable:
        row = self.rows[where["budget_id"]].model_copy(update=data)
        self.rows[where["budget_id"]] = row
        return row


class _EmptyTable:
    """A table with no row for any lookup: the user table, which end-user id validation also reads."""

    async def find_unique(self, where: object = None, include: object = None) -> None:
        return None


def _serve_end_user_rows(
    monkeypatch: pytest.MonkeyPatch, rows: _EndUserRows, cache: UserApiKeyCache, budgets: _BudgetRows | None = None
) -> None:
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        SimpleNamespace(db=SimpleNamespace(litellm_endusertable=rows, litellm_budgettable=budgets)),
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", cache)


@pytest.mark.asyncio
async def test_customer_block_answers_503_and_writes_nothing_when_redis_cannot_record_it(monkeypatch):
    from litellm.proxy.management_endpoints.customer_endpoints import block_user

    redis = fake_redis_cache()
    redis.server.connected = False
    rows = _EndUserRows(LiteLLM_EndUserTable(user_id="c1", blocked=False))
    _serve_end_user_rows(monkeypatch, rows, UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True))

    with pytest.raises(HTTPException) as refused:
        await block_user(data=BlockUsers(user_ids=["c1"]))

    assert (refused.value.status_code, rows.rows["c1"].blocked) == (503, False)


@pytest.mark.asyncio
async def test_customer_delete_answers_503_and_deletes_nothing_when_redis_cannot_record_it(monkeypatch):
    """A deleted customer falls back to the default end-user budget and leaves the restricted registry, so its entries
    are revoked before the delete: a Redis that cannot record that refuses the delete and the customer stays."""
    from litellm.proxy.management_endpoints.customer_endpoints import DeleteCustomerRequest, delete_end_user

    redis: Final = fake_redis_cache()
    redis.server.connected = False
    rows: Final = _EndUserRows(LiteLLM_EndUserTable(user_id="c1", blocked=True))
    _serve_end_user_rows(monkeypatch, rows, UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True))

    with pytest.raises(ProxyException) as refused:
        await delete_end_user(data=DeleteCustomerRequest(user_ids=["c1"]), user_api_key_dict=_ADMIN)

    assert (str(refused.value.code), sorted(rows.rows)) == ("503", ["c1"])


@pytest.mark.asyncio
async def test_customer_update_that_blocks_answers_503_and_writes_nothing_when_redis_cannot_record_it(monkeypatch):
    from litellm.proxy.management_endpoints.customer_endpoints import update_end_user

    redis = fake_redis_cache()
    redis.server.connected = False
    rows = _EndUserRows(LiteLLM_EndUserTable(user_id="c1", blocked=False))
    _serve_end_user_rows(monkeypatch, rows, UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True))

    with pytest.raises(ProxyException) as refused:
        await update_end_user(
            data=UpdateCustomerRequest(user_id="c1", blocked=True),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert (str(refused.value.code), rows.rows["c1"].blocked) == ("503", False)


@pytest.mark.asyncio
async def test_after_a_customer_block_a_registry_read_in_flight_cannot_vouch_for_the_customer_in_an_outage(monkeypatch):
    """The restricted registry the database last returned left the customer out, which lets an outage serve it
    unchecked. A registry read that took its rows before the block and finishes after it must not keep that answer:
    the outage refuses the customer instead."""
    from litellm.proxy.auth.auth_checks import get_end_user_object
    from litellm.proxy.db.exception_handler import LastGoodEntryMissing
    from litellm.proxy.management_endpoints.customer_endpoints import block_user

    rows = _EndUserRows(LiteLLM_EndUserTable(user_id="c1", blocked=False))
    cache = UserApiKeyCache(serve_last_good_entries=True)
    _serve_end_user_rows(monkeypatch, rows, cache)

    database = SimpleNamespace(db=SimpleNamespace(litellm_endusertable=rows))

    async def customer() -> LiteLLM_EndUserTable | None:
        return await get_end_user_object(
            end_user_id="c1",
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
        )

    assert await customer() is None
    await cache.async_delete_cache("end_user_restricted_registry")
    release = rows.pause_next_registry_read()
    in_flight = asyncio.create_task(customer())
    await rows.registry_read_held.wait()
    await block_user(data=BlockUsers(user_ids=["c1"]))
    release.set()
    assert await in_flight is None
    for key in ("end_user_id:c1", "end_user_restricted_registry"):
        await cache.async_delete_cache(key)
    rows.stopped = True

    with pytest.raises(LastGoodEntryMissing):
        await customer()


_ADMIN: Final = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "restriction", [{"models": ["m1"]}, {"budget_id": "b-paid"}, {"blocked": True}], ids=["models", "budget", "blocked"]
)
async def test_customer_created_with_a_restriction_answers_503_and_writes_nothing_when_redis_cannot_record_it(
    monkeypatch, restriction: dict[str, object]
):
    """A restricted customer belongs in the restricted registry, whose copy would still leave it out: the create is
    refused, as a block is, while Redis cannot record the revocation of that copy."""
    from litellm.proxy.management_endpoints.customer_endpoints import new_end_user

    redis = fake_redis_cache()
    redis.server.connected = False
    rows = _EndUserRows()
    _serve_end_user_rows(monkeypatch, rows, UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True))

    with pytest.raises(ProxyException) as refused:
        await new_end_user(data=NewCustomerRequest(user_id="c-new", **restriction), user_api_key_dict=_ADMIN)

    assert (str(refused.value.code), rows.rows) == ("503", {})


@pytest.mark.asyncio
async def test_customer_created_without_a_restriction_is_written_while_redis_cannot_record_a_revocation(monkeypatch):
    """A customer that restricts nothing is not in the restricted registry, so its create revokes nothing."""
    from litellm.proxy.management_endpoints.customer_endpoints import new_end_user

    redis = fake_redis_cache()
    redis.server.connected = False
    rows = _EndUserRows()
    _serve_end_user_rows(monkeypatch, rows, UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True))

    await new_end_user(data=NewCustomerRequest(user_id="c-free", alias="free"), user_api_key_dict=_ADMIN)

    assert list(rows.rows) == ["c-free"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "restriction",
    [{"models": ["m1"]}, {"budget_id": "b-paid"}, {"allowed_model_region": "eu"}],
    ids=["models", "budget", "region"],
)
async def test_customer_update_that_adds_a_restriction_answers_503_and_writes_nothing_when_redis_cannot_record_it(
    monkeypatch, restriction: dict[str, object]
):
    from litellm.proxy.management_endpoints.customer_endpoints import update_end_user

    redis = fake_redis_cache()
    redis.server.connected = False
    unrestricted: Final = LiteLLM_EndUserTable(user_id="c1", blocked=False)
    rows = _EndUserRows(unrestricted)
    _serve_end_user_rows(monkeypatch, rows, UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True))

    with pytest.raises(ProxyException) as refused:
        await update_end_user(data=UpdateCustomerRequest(user_id="c1", **restriction), user_api_key_dict=_ADMIN)

    assert (str(refused.value.code), rows.rows["c1"]) == ("503", unrestricted)


@pytest.mark.asyncio
@pytest.mark.parametrize("write", ["create", "update"])
async def test_after_a_customer_gains_a_restriction_a_registry_read_in_flight_cannot_vouch_for_it_in_an_outage(
    monkeypatch, write: str
):
    """A customer created with a restriction, or given one by an update, joins the restricted registry. A registry read
    that took its rows before the write and finishes after it must not keep its copy without the customer: the outage
    refuses the customer instead of serving it as one that restricts nothing."""
    from litellm.proxy.auth.auth_checks import get_end_user_object
    from litellm.proxy.db.exception_handler import LastGoodEntryMissing
    from litellm.proxy.management_endpoints.customer_endpoints import new_end_user, update_end_user

    rows = _EndUserRows(*([LiteLLM_EndUserTable(user_id="c1", blocked=False)] if write == "update" else []))
    cache = UserApiKeyCache(serve_last_good_entries=True)
    _serve_end_user_rows(monkeypatch, rows, cache)

    database = SimpleNamespace(db=SimpleNamespace(litellm_endusertable=rows))

    async def customer() -> LiteLLM_EndUserTable | None:
        return await get_end_user_object(
            end_user_id="c1",
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
        )

    assert await customer() is None
    await cache.async_delete_cache("end_user_restricted_registry")
    release = rows.pause_next_registry_read()
    in_flight = asyncio.create_task(customer())
    await rows.registry_read_held.wait()
    if write == "create":
        await new_end_user(data=NewCustomerRequest(user_id="c1", models=["m1"]), user_api_key_dict=_ADMIN)
    else:
        await update_end_user(data=UpdateCustomerRequest(user_id="c1", models=["m1"]), user_api_key_dict=_ADMIN)
    release.set()
    assert await in_flight is None
    for key in ("end_user_id:c1", "end_user_restricted_registry"):
        await cache.async_delete_cache(key)
    rows.stopped = True

    with pytest.raises(LastGoodEntryMissing):
        await customer()


class _DbBudgetRow(LiteLLM_BudgetTableFull):
    """A budget row with every column the database returns: each row has its audit columns, and the customer response
    requires ``created_at``."""

    created_by: str
    updated_at: datetime
    updated_by: str


class _CustomerRowWithBudget(LiteLLM_EndUserTable):
    """A customer row as a read or write with ``include={"litellm_budget_table": True}`` returns it: with its whole
    budget row, which a dump of ``LiteLLM_EndUserTable`` would cut down to the writable budget fields."""

    litellm_budget_table: _DbBudgetRow | None = None  # pyright: ignore[reportIncompatibleVariableOverride]  # whole row


def _budgeted_customer() -> tuple[_EndUserRows, _BudgetRows]:
    """Customer c1 on its own budget b1 with max_budget 10, as the database returns them."""
    created: Final = datetime(2024, 1, 1, tzinfo=timezone.utc)
    budget: Final = _DbBudgetRow(
        budget_id="b1",
        max_budget=10.0,
        created_at=created,
        created_by="admin",
        updated_at=created,
        updated_by="admin",
    )
    return (
        _EndUserRows(_CustomerRowWithBudget(user_id="c1", blocked=False, budget_id="b1", litellm_budget_table=budget)),
        _BudgetRows(budget),
    )


@pytest.mark.asyncio
async def test_customer_update_of_its_existing_budget_is_written_while_redis_is_down(monkeypatch):
    """Lowering a customer's existing budget moves it into or out of no restricted registry, so it revokes nothing
    before the write: it writes, then evicts the customer's entries best effort, and a Redis that is down refuses
    nothing."""
    from litellm.proxy.management_endpoints.customer_endpoints import update_end_user

    redis: Final = fake_redis_cache()
    redis.server.connected = False
    rows, budgets = _budgeted_customer()
    _serve_end_user_rows(
        monkeypatch, rows, UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True), budgets
    )

    updated: Final = await update_end_user(
        data=UpdateCustomerRequest(user_id="c1", max_budget=5.0), user_api_key_dict=_ADMIN
    )

    assert (updated.user_id, budgets.rows["b1"].max_budget) == ("c1", 5.0)


@pytest.mark.asyncio
async def test_a_customer_budget_lowered_while_redis_is_down_drops_the_entry_this_pod_kept_with_the_higher_limit(
    monkeypatch,
):
    """The eviction after the budget write is best effort: Redis failing there does not fail the update, and this pod
    still drops the customer's cached row and last good entry, so the next outage refuses the customer instead of
    serving the higher limit."""
    from litellm.proxy.auth.auth_checks import get_end_user_object
    from litellm.proxy.db.exception_handler import LastGoodEntryMissing
    from litellm.proxy.management_endpoints.customer_endpoints import update_end_user

    redis: Final = fake_redis_cache()
    redis.server.connected = False
    rows, budgets = _budgeted_customer()
    cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
    _serve_end_user_rows(monkeypatch, rows, cache, budgets)
    database: Final = SimpleNamespace(db=SimpleNamespace(litellm_endusertable=rows))

    async def customer() -> LiteLLM_EndUserTable | None:
        return await get_end_user_object(
            end_user_id="c1",
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
        )

    before: Final = await customer()
    assert before is not None and before.litellm_budget_table is not None
    assert before.litellm_budget_table.max_budget == 10.0
    await update_end_user(data=UpdateCustomerRequest(user_id="c1", max_budget=5.0), user_api_key_dict=_ADMIN)
    rows.stopped = True

    with pytest.raises(LastGoodEntryMissing):
        await customer()


@pytest.mark.asyncio
async def test_a_customer_created_after_a_request_found_its_id_invalid_is_kept_by_the_next_request(monkeypatch):
    """With end-user ids validated and no default budget, a request naming an id no row holds caches the verdict
    "invalid" and drops the id. Creating that customer with a model restriction evicts the verdict, so the next request
    keeps the id and the restriction applies, rather than the id being dropped until the verdict lapses."""
    from litellm.proxy.auth.auth_checks import resolve_and_validate_end_user_id
    from litellm.proxy.management_endpoints.customer_endpoints import new_end_user

    monkeypatch.setattr("litellm.validate_end_user_id_in_db", True)
    monkeypatch.setattr("litellm.max_end_user_budget_id", None)
    rows: Final = _EndUserRows()
    cache: Final = UserApiKeyCache()
    _serve_end_user_rows(monkeypatch, rows, cache)
    database: Final = SimpleNamespace(db=SimpleNamespace(litellm_endusertable=rows, litellm_usertable=_EmptyTable()))

    async def resolve() -> str | None:
        return await resolve_and_validate_end_user_id(
            raw_end_user_id="c1",
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
        )

    assert (await resolve(), await cache.async_get_cache("end_user_validation:c1")) == (None, "invalid")
    await new_end_user(data=NewCustomerRequest(user_id="c1", models=["m1"]), user_api_key_dict=_ADMIN)

    assert await resolve() == "c1"


@pytest.mark.asyncio
async def test_a_customer_block_drops_the_verdict_and_registry_copies_last_good_entries_answered(monkeypatch):
    """In an outage the customer's last good entry vouches for its id and the restricted registry's entry leaves it
    out. This pod keeps both answers in its memory only, never in Redis; a block once the database answers again drops
    both, so the next request reads the blocked customer instead of either copy."""
    from litellm.proxy.auth.auth_checks import _load_end_user_restricted_registry, resolve_and_validate_end_user_id
    from litellm.proxy.management_endpoints.customer_endpoints import block_user

    monkeypatch.setattr("litellm.validate_end_user_id_in_db", True)
    monkeypatch.setattr("litellm.max_end_user_budget_id", None)
    rows: Final = _EndUserRows(LiteLLM_EndUserTable(user_id="c1", blocked=False))
    redis: Final = fake_redis_cache()
    cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
    _serve_end_user_rows(monkeypatch, rows, cache)
    database: Final = SimpleNamespace(db=SimpleNamespace(litellm_endusertable=rows, litellm_usertable=_EmptyTable()))
    copy_keys: Final = ("end_user_validation:c1", "end_user_restricted_registry")

    async def registry() -> frozenset[str] | None:
        return await _load_end_user_restricted_registry(
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
        )

    async def resolve() -> str | None:
        return await resolve_and_validate_end_user_id(
            raw_end_user_id="c1",
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
        )

    assert (await registry(), await resolve()) == (frozenset(), "c1")
    for key in ("end_user_id:c1", *copy_keys):
        await cache.async_delete_cache(key)
    rows.stopped = True
    assert (await registry(), await resolve()) == (frozenset(), "c1")
    assert [cache.in_memory_cache.get_cache(key=key) for key in copy_keys] == ["valid_from_last_good", {"ids": []}]
    assert [await redis.cache.async_get_cache(key) for key in copy_keys] == [None, None]
    rows.stopped = False

    await block_user(data=BlockUsers(user_ids=["c1"]))

    assert [cache.in_memory_cache.get_cache(key=key) for key in copy_keys] == [None, None]


class _DrivenClock:
    """The last good entries' clock, which a test moves instead of waiting."""

    def __init__(self) -> None:
        self.now: float = 1000.0

    def __call__(self) -> float:
        return self.now


# Past this since a revocation, the request's record no longer skips Redis for the key: its tombstone may lapse first
_PAST_THE_RECORD: Final = LAST_GOOD_TOMBSTONE_TTL_SECONDS - LAST_GOOD_LATE_STORE_MARGIN_SECONDS + 1


async def _outlast_the_tombstones(redis: FakeRedisCache, clock: _DrivenClock, *, loses_redis: bool) -> None:
    """A step that takes past the time the request's record of its revocations allows: the tombstones in Redis lapse
    meanwhile, and Redis is lost once the step ends when ``loses_redis``."""
    clock.now += _PAST_THE_RECORD
    client: Final = redis.cache.init_async_client()
    lapsed: Final = await client.keys("last_good_tombstone:*")
    if lapsed:
        await client.delete(*lapsed)
    if loses_redis:
        redis.server.connected = False


def _driven_cache(redis: FakeRedisCache, clock: _DrivenClock, *, serves_last_good: bool) -> UserApiKeyCache:
    return UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=serves_last_good, last_good_clock=clock)


def _customer_cache_keys(user_id: str) -> tuple[str, ...]:
    """What a write that blocks or restricts customer ``user_id`` ends: its entry, its id's validation verdict, and the
    restricted registry."""
    return (f"end_user_id:{user_id}", f"end_user_validation:{user_id}", "end_user_restricted_registry")


class _WitnessedEndUserRows(_EndUserRows):
    """End-user rows that record, for each upsert, create and update in turn, the customer's id and which of the cache
    keys its write ends have a tombstone in Redis as the write runs, none while Redis is lost. The upsert of
    ``outlasting`` then takes past the time the request's record of its revocations allows."""

    def __init__(
        self,
        redis: FakeRedisCache,
        clock: _DrivenClock,
        *rows: LiteLLM_EndUserTable,
        outlasting: str | None = None,
        loses_redis: bool = False,
    ) -> None:
        super().__init__(*rows)
        self.standing: Final[list[tuple[str, tuple[str, ...]]]] = []  # mutable-ok: one entry per end-user row write
        self._redis: Final = redis
        self._clock: Final = clock
        self._outlasting: Final = outlasting
        self._loses_redis: Final = loses_redis

    async def _witness(self, user_id: str) -> None:
        connected: Final = self._redis.server.connected
        client: Final = self._redis.cache.init_async_client()
        self.standing.append(
            (
                user_id,
                tuple(
                    [
                        cache_key
                        for cache_key in _customer_cache_keys(user_id)
                        if connected and await client.exists(f"last_good_tombstone:{cache_key}") == 1
                    ]
                ),
            )
        )

    async def upsert(self, where: dict[str, str], data: dict[str, dict[str, object]]) -> LiteLLM_EndUserTable:
        await self._witness(where["user_id"])
        row: Final = await super().upsert(where, data)
        if where["user_id"] == self._outlasting:
            await _outlast_the_tombstones(self._redis, self._clock, loses_redis=self._loses_redis)
        return row

    async def create(self, data: dict[str, object], include: object = None) -> LiteLLM_EndUserTable:
        await self._witness(str(data["user_id"]))
        return await super().create(data, include)

    async def update(
        self, where: dict[str, str], data: dict[str, object], include: object = None
    ) -> LiteLLM_EndUserTable:
        await self._witness(where["user_id"])
        return await super().update(where, data, include)


class _OutlastingBudgetRows(_BudgetRows):
    """The budget table a customer create or update adds a budget to: each create takes past the time the request's
    record of its revocations allows, then answers with the new row as the database returns it."""

    def __init__(self, redis: FakeRedisCache, clock: _DrivenClock, *, loses_redis: bool) -> None:
        super().__init__()
        self._redis: Final = redis
        self._clock: Final = clock
        self._loses_redis: Final = loses_redis

    async def create(self, data: dict[str, object], include: object = None) -> _DbBudgetRow:
        await _outlast_the_tombstones(self._redis, self._clock, loses_redis=self._loses_redis)
        created: Final = datetime(2024, 1, 1, tzinfo=timezone.utc)
        row: Final = _DbBudgetRow.model_validate(
            {**data, "budget_id": f"budget-{len(self.rows) + 1}", "created_at": created, "updated_at": created}
        )
        self.rows[row.budget_id] = row
        return row


def _two_unblocked_customers(
    redis: FakeRedisCache, clock: _DrivenClock, *, loses_redis: bool
) -> tuple[_WitnessedEndUserRows, _OutlastingBudgetRows]:
    """Customers c1 and c2, neither restricted, whose block's upsert of c1 and whose budget creates each outlast the
    tombstones."""
    return (
        _WitnessedEndUserRows(
            redis,
            clock,
            LiteLLM_EndUserTable(user_id="c1", blocked=False),
            LiteLLM_EndUserTable(user_id="c2", blocked=False),
            outlasting="c1",
            loses_redis=loses_redis,
        ),
        _OutlastingBudgetRows(redis, clock, loses_redis=loses_redis),
    )


@pytest.mark.asyncio
async def test_customer_block_revokes_a_later_customer_again_when_an_earlier_upsert_outlasts_the_tombstones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The block revokes every customer before its first upsert, then upserts them one by one. The first upsert takes
    past the time the request's record of that revocation allows and the tombstones lapse meanwhile, so another pod may
    store the second customer's entry, its verdict or the registry again: the second upsert revokes them again right
    before it writes, so their tombstones stand in Redis as it runs."""
    from litellm.proxy.management_endpoints.customer_endpoints import block_user

    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    rows, budgets = _two_unblocked_customers(redis, clock, loses_redis=False)
    _serve_end_user_rows(monkeypatch, rows, _driven_cache(redis, clock, serves_last_good=True), budgets)

    await block_user(data=BlockUsers(user_ids=["c1", "c2"]))

    assert rows.standing == [("c1", _customer_cache_keys("c1")), ("c2", _customer_cache_keys("c2"))]


@pytest.mark.asyncio
async def test_customer_block_answers_503_before_a_later_upsert_when_redis_is_lost_after_an_earlier_one_outlasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis is lost once the first upsert has outlasted the tombstones: the second customer's revocation cannot be
    recorded again, so the block answers 503 before that customer's upsert and it stays unblocked, rather than its
    entry outliving the lapsed tombstone into an outage."""
    from litellm.proxy.management_endpoints.customer_endpoints import block_user

    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    rows, budgets = _two_unblocked_customers(redis, clock, loses_redis=True)
    _serve_end_user_rows(monkeypatch, rows, _driven_cache(redis, clock, serves_last_good=True), budgets)

    with pytest.raises(HTTPException) as refused:
        await block_user(data=BlockUsers(user_ids=["c1", "c2"]))

    assert (refused.value.status_code, rows.standing, rows.rows["c2"].blocked) == (
        503,
        [("c1", _customer_cache_keys("c1"))],
        False,
    )


@pytest.mark.asyncio
async def test_restricted_customer_create_revokes_it_again_when_its_budget_create_outlasts_the_tombstones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A customer created with a budget joins the restricted registry, so it is revoked before the budget's create.
    That create takes past the time the request's record allows and the tombstones lapse meanwhile: the customer's
    create revokes its entry, its verdict and the registry again right before it writes, so their tombstones stand."""
    from litellm.proxy.management_endpoints.customer_endpoints import new_end_user

    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    rows, budgets = _two_unblocked_customers(redis, clock, loses_redis=False)
    _serve_end_user_rows(monkeypatch, rows, _driven_cache(redis, clock, serves_last_good=True), budgets)

    created: Final = await new_end_user(
        data=NewCustomerRequest(user_id="c-new", max_budget=5.0), user_api_key_dict=_ADMIN
    )

    assert (created.budget_id, rows.standing) == ("budget-1", [("c-new", _customer_cache_keys("c-new"))])


@pytest.mark.asyncio
async def test_restricted_customer_create_answers_503_and_writes_no_customer_when_redis_is_lost_after_its_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis is lost once the budget's create has outlasted the tombstones: the customer's revocation cannot be
    recorded again, so the create answers 503 with no customer written, and the new budget stays linked to none."""
    from litellm.proxy.management_endpoints.customer_endpoints import new_end_user

    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    rows, budgets = _two_unblocked_customers(redis, clock, loses_redis=True)
    _serve_end_user_rows(monkeypatch, rows, _driven_cache(redis, clock, serves_last_good=True), budgets)

    with pytest.raises(ProxyException) as refused:
        await new_end_user(data=NewCustomerRequest(user_id="c-new", max_budget=5.0), user_api_key_dict=_ADMIN)

    assert (str(refused.value.code), rows.standing, sorted(rows.rows), sorted(budgets.rows)) == (
        "503",
        [],
        ["c1", "c2"],
        ["budget-1"],
    )


@pytest.mark.asyncio
async def test_customer_update_giving_a_budget_revokes_it_again_when_the_budget_create_outlasts_the_tombstones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An update that gives an unrestricted customer a new budget moves it into the restricted registry, so it is
    revoked before the budget's create. That create takes past the time the request's record allows and the tombstones
    lapse meanwhile: the customer's update revokes its entry, its verdict and the registry again right before it
    writes, so their tombstones stand."""
    from litellm.proxy.management_endpoints.customer_endpoints import update_end_user

    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    rows, budgets = _two_unblocked_customers(redis, clock, loses_redis=False)
    _serve_end_user_rows(monkeypatch, rows, _driven_cache(redis, clock, serves_last_good=True), budgets)

    updated: Final = await update_end_user(
        data=UpdateCustomerRequest(user_id="c1", max_budget=5.0), user_api_key_dict=_ADMIN
    )

    assert (updated.budget_id, rows.standing) == ("budget-1", [("c1", _customer_cache_keys("c1"))])


@pytest.mark.asyncio
async def test_customer_update_giving_a_budget_answers_503_and_leaves_the_customer_when_redis_is_lost_after_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis is lost once the budget's create has outlasted the tombstones: the customer's revocation cannot be
    recorded again, so the update answers 503 and the customer's row is left as it was."""
    from litellm.proxy.management_endpoints.customer_endpoints import update_end_user

    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    rows, budgets = _two_unblocked_customers(redis, clock, loses_redis=True)
    _serve_end_user_rows(monkeypatch, rows, _driven_cache(redis, clock, serves_last_good=True), budgets)

    with pytest.raises(ProxyException) as refused:
        await update_end_user(data=UpdateCustomerRequest(user_id="c1", max_budget=5.0), user_api_key_dict=_ADMIN)

    assert (str(refused.value.code), rows.standing, rows.rows["c1"]) == (
        "503",
        [],
        LiteLLM_EndUserTable(user_id="c1", blocked=False),
    )


async def _customer_write(write: str) -> object:
    """The customer write a test names: a block of c1 then c2, the create of c-new with a budget, or an update that
    gives c1 a budget."""
    from litellm.proxy.management_endpoints.customer_endpoints import block_user, new_end_user, update_end_user

    if write == "block":
        return await block_user(data=BlockUsers(user_ids=["c1", "c2"]))
    if write == "create":
        return await new_end_user(data=NewCustomerRequest(user_id="c-new", max_budget=5.0), user_api_key_dict=_ADMIN)
    return await update_end_user(data=UpdateCustomerRequest(user_id="c1", max_budget=5.0), user_api_key_dict=_ADMIN)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("write", "written"),
    [
        ("block", [("c1", True, None), ("c2", True, None)]),
        ("create", [("c-new", False, "budget-1")]),
        ("update", [("c1", False, "budget-1")]),
    ],
)
async def test_with_last_good_entries_off_each_customer_write_runs_in_its_order_while_redis_is_lost(
    monkeypatch: pytest.MonkeyPatch, write: str, written: list[tuple[str, bool, str | None]]
) -> None:
    """With ``serve_last_good_auth_entries`` off nothing is revoked: a block, a restricted create and an update that
    gives a budget write every row in their order, as they always have, though Redis is lost after a step that outlasts
    the tombstone time."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    rows, budgets = _two_unblocked_customers(redis, clock, loses_redis=True)
    _serve_end_user_rows(monkeypatch, rows, _driven_cache(redis, clock, serves_last_good=False), budgets)

    await _customer_write(write)

    assert [
        (user_id, rows.rows[user_id].blocked, rows.rows[user_id].budget_id) for user_id, _ in rows.standing
    ] == written


@pytest.mark.asyncio
async def test_customer_block_records_its_tombstone_again_after_the_upsert_so_no_outage_serves_the_unblocked_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request first revokes customer c1's entry, its id's verdict and the restricted registry at t=0, with a 30 s
    tombstone. The block of c1 runs at t=19, when the request's record shows that tombstone standing, so neither its
    revocation nor the one paired with c1's upsert sends Redis anything. Another pod's read began at t=19, before the
    upsert committed, and got c1 unblocked; it stores that row at t=31, once the first tombstone lapsed at t=30. The
    block records c1's tombstone again once its upsert returns, standing until t=49, so that store keeps nothing and an
    outage that follows serves no entry for c1, rather than the unblocked row."""
    from litellm.proxy.common_utils.user_api_key_cache import end_user_cache_key
    from litellm.proxy.management_endpoints.customer_endpoints import block_user
    from tests.unit.proxy.last_good_tombstone_race import (
        revoked_before_the_write,
        served_after_another_pod_stores,
        tombstone_race,
    )

    race: Final = tombstone_race(monkeypatch)
    unblocked: Final = LiteLLM_EndUserTable(user_id="c1", blocked=False)
    rows: Final = _EndUserRows(unblocked)
    _serve_end_user_rows(monkeypatch, rows, race.cache)
    await revoked_before_the_write(race, _customer_cache_keys("c1"))

    blocked: Final = await block_user(data=BlockUsers(user_ids=["c1"]))

    assert (blocked, rows.rows["c1"].blocked) == (
        {"blocked_users": [LiteLLM_EndUserTable(user_id="c1", blocked=True)]},
        True,
    )
    served: Final = await served_after_another_pod_stores(race, end_user_cache_key("c1"), unblocked)
    assert served is None


@pytest.mark.asyncio
async def test_restricting_customer_update_records_its_tombstone_again_so_no_outage_serves_the_unrestricted_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An update that gives customer c1 a model allowlist moves it into the restricted registry, so it revokes c1's
    entry, its id's verdict and the registry, then writes the row paired with a second revocation of them. The request
    first revokes them at t=0, with a 30 s tombstone. The update runs at t=19, when the request's record shows that
    tombstone standing, so neither of its revocations sends Redis anything. Another pod's read began at t=19, before
    the update committed, and got c1 with no allowlist; it stores that row at t=31, once the first tombstone lapsed at
    t=30. The update records c1's tombstone again once its write returns, standing until t=49, so that store keeps
    nothing and an outage that follows serves no entry for c1, rather than the row that restricts no model."""
    from litellm.proxy.common_utils.user_api_key_cache import end_user_cache_key
    from litellm.proxy.management_endpoints.customer_endpoints import update_end_user
    from tests.unit.proxy.last_good_tombstone_race import (
        revoked_before_the_write,
        served_after_another_pod_stores,
        tombstone_race,
    )

    race: Final = tombstone_race(monkeypatch)
    unrestricted: Final = LiteLLM_EndUserTable(user_id="c1", blocked=False)
    rows: Final = _EndUserRows(unrestricted)
    _serve_end_user_rows(monkeypatch, rows, race.cache)
    await revoked_before_the_write(race, _customer_cache_keys("c1"))

    updated: Final = await update_end_user(
        data=UpdateCustomerRequest(user_id="c1", models=["m1"]), user_api_key_dict=_ADMIN
    )

    assert (updated.user_id, updated.models, rows.rows["c1"].models) == ("c1", ["m1"], ["m1"])
    served: Final = await served_after_another_pod_stores(race, end_user_cache_key("c1"), unrestricted)
    assert served is None
