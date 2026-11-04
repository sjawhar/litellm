import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Final, Mapping, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from litellm._uuid import uuid
from tests.unit.proxy.management_endpoints.jwt_key_mapping_doubles import (
    CascadingJWTMappingTable,
    JWTMappingRow,
)

if TYPE_CHECKING:
    from litellm.proxy._types import LiteLLM_OrganizationTable
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from tests.unit.caching.fake_redis_cache import FakeRedisCache


@pytest.mark.asyncio
async def test_organization_update_object_permissions_existing_permission(monkeypatch):
    """
    Test updating object permissions when an organization already has an existing object_permission_id.

    This test verifies that when updating vector stores for an organization that already has an
    object_permission_id, the existing LiteLLM_ObjectPermissionTable record is updated
    with the new permissions and the object_permission_id remains the same.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        LiteLLM_OrganizationTable,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import (
        handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Mock existing organization with object_permission_id
    existing_organization_row = LiteLLM_OrganizationTable(
        organization_id="test_org_id",
        object_permission_id="existing_perm_id_123",
        organization_alias="test_org",
        budget_id="test_budget_id",
        models=["test_model_1", "test_model_2"],
        created_by="test_created_by",
        updated_by="test_updated_by",
    )

    # Mock existing object permission record
    existing_object_permission = MagicMock()
    existing_object_permission.model_dump.return_value = {
        "object_permission_id": "existing_perm_id_123",
        "vector_stores": ["old_store_1", "old_store_2"],
    }

    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(return_value=existing_object_permission)

    # Mock upsert operation
    updated_permission = MagicMock()
    updated_permission.object_permission_id = "existing_perm_id_123"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(return_value=updated_permission)

    # Test data with new object permission
    data_json = {
        "object_permission": LiteLLM_ObjectPermissionBase(
            vector_stores=["new_store_1", "new_store_2", "new_store_3"]
        ).model_dump(exclude_unset=True, exclude_none=True),
        "organization_alias": "updated_org",
    }

    # Call the function
    result = await handle_update_object_permission(
        data_json=data_json,
        existing_organization_row=existing_organization_row,
    )

    # Verify the object_permission was removed from data_json and object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "existing_perm_id_123"

    # Verify database operations were called correctly
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": "existing_perm_id_123"}
    )
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_get_organization_daily_activity_admin_param_passing(monkeypatch):
    """
    As admin, ensure parsed params are forwarded to get_daily_activity with correct values.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.management_endpoints.organization_endpoints import (
        get_organization_daily_activity,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_organizationtable.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Admin view -> skip membership restriction
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.organization_endpoints.user_api_key_has_admin_view",
        lambda _: True,
    )

    # Patch downstream common function and verify call args
    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(organization_endpoints, "get_daily_activity", get_daily_activity_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin1")
    result = await get_organization_daily_activity(
        organization_ids="org1,org2",
        start_date="2024-01-01",
        end_date="2024-01-31",
        model="gpt-4",
        api_key="test-key",
        page=2,
        page_size=5,
        exclude_organization_ids="org3",
        user_api_key_dict=auth,
    )

    # Ensure passthrough to common method with correct args
    get_daily_activity_mock.assert_awaited_once()
    kwargs = get_daily_activity_mock.call_args.kwargs
    assert kwargs["table_name"] == "litellm_dailyorganizationspend"
    assert kwargs["entity_id_field"] == "organization_id"
    assert kwargs["entity_id"] == ["org1", "org2"]
    assert kwargs["exclude_entity_ids"] == ["org3"]
    assert kwargs["start_date"] == "2024-01-01"
    assert kwargs["end_date"] == "2024-01-31"
    assert kwargs["model"] == "gpt-4"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["page"] == 2
    assert kwargs["page_size"] == 5

    assert result is mocked_response


@pytest.mark.asyncio
async def test_get_organization_daily_activity_non_admin_defaults_to_admin_orgs(
    monkeypatch,
):
    """
    Non-admin with no explicit organization_ids should default to orgs they are ORG_ADMIN of.
    """
    from types import SimpleNamespace

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.management_endpoints.organization_endpoints import (
        get_organization_daily_activity,
    )

    # Mock prisma client and memberships
    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_organizationtable.find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_organizationmembership.find_many = AsyncMock(
        return_value=[
            SimpleNamespace(organization_id="orgA", user_role=LitellmUserRoles.ORG_ADMIN.value),
            SimpleNamespace(organization_id="orgB", user_role=LitellmUserRoles.ORG_ADMIN.value),
        ]
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Non-admin view
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.organization_endpoints.user_api_key_has_admin_view",
        lambda _: False,
    )

    # Patch downstream aggregator
    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(organization_endpoints, "get_daily_activity", get_daily_activity_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="regular-user")
    await get_organization_daily_activity(
        organization_ids=None,
        start_date="2024-02-01",
        end_date="2024-02-28",
        model=None,
        api_key=None,
        page=1,
        page_size=10,
        exclude_organization_ids=None,
        user_api_key_dict=auth,
    )

    kwargs = get_daily_activity_mock.call_args.kwargs
    assert kwargs["entity_id"] == ["orgA", "orgB"]
    assert kwargs["start_date"] == "2024-02-01"
    assert kwargs["end_date"] == "2024-02-28"


@pytest.mark.asyncio
async def test_get_organization_daily_activity_non_admin_unauthorized_org_raises(
    monkeypatch,
):
    """
    Non-admin requesting an org they aren't ORG_ADMIN for should raise 403.
    """
    from types import SimpleNamespace

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import (
        get_organization_daily_activity,
    )

    # Mock prisma client and memberships (only orgA is admin)
    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_organizationmembership.find_many = AsyncMock(
        return_value=[SimpleNamespace(organization_id="orgA", user_role=LitellmUserRoles.ORG_ADMIN.value)]
    )
    mock_prisma_client.db.litellm_organizationtable.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Non-admin view
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.organization_endpoints.user_api_key_has_admin_view",
        lambda _: False,
    )

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="regular-user")

    with pytest.raises(HTTPException) as exc:
        await get_organization_daily_activity(
            organization_ids="orgA,orgX",  # orgX is unauthorized
            start_date="2024-03-01",
            end_date="2024-03-31",
            model=None,
            api_key=None,
            page=1,
            page_size=10,
            exclude_organization_ids=None,
            user_api_key_dict=auth,
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_organization_update_object_permissions_no_existing_permission(
    monkeypatch,
):
    """
    Test creating object permissions when an organization has no existing object_permission_id.

    This test verifies that when updating object permissions for an organization that has
    object_permission_id set to None, a new entry is created in the
    LiteLLM_ObjectPermissionTable and the organization is updated with the new object_permission_id.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        LiteLLM_OrganizationTable,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import (
        handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    existing_organization_row_no_perm = LiteLLM_OrganizationTable(
        organization_id="test_org_id_2",
        object_permission_id=None,
        organization_alias="test_org_2",
        budget_id="test_budget_id_2",
        models=["test_model_1", "test_model_2"],
        created_by="test_created_by_2",
        updated_by="test_updated_by_2",
    )

    # Mock find_unique to return None (no existing permission)
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(return_value=None)

    # Mock upsert to create new record
    new_permission = MagicMock()
    new_permission.object_permission_id = "new_perm_id_456"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(return_value=new_permission)

    data_json = {
        "object_permission": LiteLLM_ObjectPermissionBase(vector_stores=["brand_new_store"]).model_dump(
            exclude_unset=True, exclude_none=True
        ),
        "organization_alias": "updated_org_2",
    }

    result = await handle_update_object_permission(
        data_json=data_json,
        existing_organization_row=existing_organization_row_no_perm,
    )

    # Verify new object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "new_perm_id_456"

    # Verify upsert was called to create new record
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_organization_update_object_permissions_missing_permission_record(
    monkeypatch,
):
    """
    Test creating object permissions when existing object_permission_id record is not found.

    This test verifies that when updating object permissions for an organization that has an
    object_permission_id but the corresponding record cannot be found in the database,
    a new entry is created in the LiteLLM_ObjectPermissionTable with the new permissions.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        LiteLLM_OrganizationTable,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import (
        handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    existing_organization_row_missing_perm = LiteLLM_OrganizationTable(
        organization_id="test_org_id_3",
        object_permission_id="missing_perm_id_789",
        organization_alias="test_org_3",
        budget_id="test_budget_id_3",
        models=["test_model_1", "test_model_2"],
        created_by="test_created_by_3",
        updated_by="test_updated_by_3",
    )

    # Mock find_unique to return None (permission record not found)
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(return_value=None)

    # Mock upsert to create new record
    new_permission = MagicMock()
    new_permission.object_permission_id = "recreated_perm_id_789"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(return_value=new_permission)

    data_json = {
        "object_permission": LiteLLM_ObjectPermissionBase(vector_stores=["recreated_store"]).model_dump(
            exclude_unset=True, exclude_none=True
        ),
        "organization_alias": "updated_org_3",
    }

    result = await handle_update_object_permission(
        data_json=data_json,
        existing_organization_row=existing_organization_row_missing_perm,
    )

    # Verify new object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "recreated_perm_id_789"

    # Verify find_unique was called with the missing permission ID
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": "missing_perm_id_789"}
    )

    # Verify upsert was called to create new record
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_list_organization_filter_by_org_id(monkeypatch):
    """
    Test filtering organizations by org_id query parameter.

    This test verifies that when org_id is provided, only the organization
    with that exact organization_id is returned.
    """
    from types import SimpleNamespace

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import (
        list_organization,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()

    # Mock organization data
    mock_org1 = SimpleNamespace(
        organization_id="org-123",
        organization_alias="Test Org 1",
        model_dump=lambda: {
            "organization_id": "org-123",
            "organization_alias": "Test Org 1",
        },
    )

    # Mock find_many to return filtered results
    mock_prisma_client.db.litellm_organizationtable.find_many = AsyncMock(return_value=[mock_org1])

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Test as proxy admin
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-user")

    result = await list_organization(org_id="org-123", org_alias=None, user_api_key_dict=auth)

    # Verify the correct organization was returned
    assert len(result) == 1
    assert result[0].organization_id == "org-123"
    assert result[0].organization_alias == "Test Org 1"

    # Verify find_many was called with correct where conditions
    mock_prisma_client.db.litellm_organizationtable.find_many.assert_called_once()
    call_args = mock_prisma_client.db.litellm_organizationtable.find_many.call_args
    assert call_args.kwargs["where"] == {"organization_id": "org-123"}
    assert call_args.kwargs["include"] == {
        "litellm_budget_table": True,
        "members": True,
        "teams": True,
    }


@pytest.mark.asyncio
async def test_list_organization_filter_by_org_alias(monkeypatch):
    """
    Test filtering organizations by org_alias query parameter with case-insensitive partial matching.

    This test verifies that when org_alias is provided, organizations with matching
    organization_alias (case-insensitive partial match) are returned.
    """
    from types import SimpleNamespace

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import (
        list_organization,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()

    # Mock organization data
    mock_org1 = SimpleNamespace(
        organization_id="org-123",
        organization_alias="My Test Organization",
        model_dump=lambda: {
            "organization_id": "org-123",
            "organization_alias": "My Test Organization",
        },
    )
    mock_org2 = SimpleNamespace(
        organization_id="org-456",
        organization_alias="Another Test Org",
        model_dump=lambda: {
            "organization_id": "org-456",
            "organization_alias": "Another Test Org",
        },
    )

    # Mock find_many to return filtered results
    mock_prisma_client.db.litellm_organizationtable.find_many = AsyncMock(return_value=[mock_org1, mock_org2])

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Test as proxy admin with org_alias filter
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-user")

    result = await list_organization(org_id=None, org_alias="test", user_api_key_dict=auth)

    # Verify organizations with "test" in alias were returned
    assert len(result) == 2
    assert all("test" in org.organization_alias.lower() for org in result)

    # Verify find_many was called with correct where conditions (case-insensitive contains)
    mock_prisma_client.db.litellm_organizationtable.find_many.assert_called_once()
    call_args = mock_prisma_client.db.litellm_organizationtable.find_many.call_args
    assert call_args.kwargs["where"] == {"organization_alias": {"contains": "test", "mode": "insensitive"}}
    assert call_args.kwargs["include"] == {
        "litellm_budget_table": True,
        "members": True,
        "teams": True,
    }


@pytest.mark.asyncio
async def test_organization_info_includes_user_email(monkeypatch):
    """
    Test that GET /organization/info returns user_email in members list.
    """
    from datetime import datetime

    from litellm.proxy._types import LiteLLM_OrganizationMembershipTable

    # Simulate a membership row with a nested user object that has user_email
    raw_membership = {
        "user_id": "user_abc",
        "organization_id": "org_xyz",
        "user_role": "org_admin",
        "spend": 0.0,
        "budget_id": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "user": {"user_email": "alice@example.com"},
        "litellm_budget_table": None,
    }

    membership = LiteLLM_OrganizationMembershipTable(**raw_membership)
    assert membership.user_email == "alice@example.com"


# Regression tests for IDOR fixes on org-scoped endpoints. Sibling cluster
# to GHSA-xxv2-fprq-9x93 (team callback IDOR): the same shape of "any
# authenticated key holder reaches an endpoint that takes an
# organization_id from the request body without an access guard." The
# fix routes ``update_organization``, ``organization_member_add``,
# ``organization_member_update``, and ``organization_member_delete``
# through the existing ``_verify_org_access`` helper.


@pytest.fixture
def unauthorized_caller():
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="random_authenticated_user",
        api_key="sk-random",
    )


@pytest.fixture
def patched_org_prisma():
    """Mock prisma so that find_unique returns a victim org and
    get_user_object reports the caller has no org membership — so
    _verify_org_access raises 403."""
    victim_row = MagicMock()
    victim_row.organization_id = "org-victim"
    victim_row.metadata = {}
    victim_row.model_dump.return_value = {"organization_id": "org-victim"}

    caller_user = MagicMock()
    caller_user.organization_memberships = []  # no admin role anywhere

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch(
            "litellm.proxy.management_endpoints.organization_endpoints.get_user_object",
            new_callable=AsyncMock,
            return_value=caller_user,
        ),
        patch(
            "litellm.proxy.proxy_server.user_api_key_cache",
        ),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
    ):
        mock_prisma.db.litellm_organizationtable.find_unique = AsyncMock(return_value=victim_row)
        yield mock_prisma


@pytest.mark.asyncio
async def test_organization_member_add_rejects_unauthorized_caller(patched_org_prisma, unauthorized_caller):
    # ``organization_member_add`` catches HTTPException in its
    # catch-all and re-wraps as ProxyException with the original status
    # code preserved.
    from unittest.mock import Mock

    from fastapi import Request

    from litellm.proxy._types import (
        OrganizationMemberAddRequest,
        OrgMember,
        ProxyException,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import (
        organization_member_add,
    )

    data = OrganizationMemberAddRequest(
        organization_id="org-victim",
        member=OrgMember(role="internal_user", user_id="attacker-user"),
    )

    with pytest.raises((HTTPException, ProxyException)) as exc:
        await organization_member_add(
            data=data,
            http_request=Mock(spec=Request),
            user_api_key_dict=unauthorized_caller,
        )
    code = getattr(exc.value, "status_code", None) or getattr(exc.value, "code", None)
    assert int(code) == 403


@pytest.mark.asyncio
async def test_organization_member_update_rejects_unauthorized_caller(patched_org_prisma, unauthorized_caller):
    from litellm.proxy._types import OrganizationMemberUpdateRequest
    from litellm.proxy.management_endpoints.organization_endpoints import (
        organization_member_update,
    )

    data = OrganizationMemberUpdateRequest(
        organization_id="org-victim",
        user_id="some-other-user",
        role="org_admin",
    )

    with pytest.raises(HTTPException) as exc:
        await organization_member_update(
            data=data,
            user_api_key_dict=unauthorized_caller,
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget_payload",
    [{"max_budget_in_organization": None}, {}],
    ids=["explicit-null", "omitted"],
)
async def test_organization_member_add_budget_omission_and_null_leave_budget_unset(budget_payload, monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace

    from litellm.proxy._types import (
        LiteLLM_OrganizationMembershipTable,
        LiteLLM_UserTable,
        LitellmUserRoles,
        OrganizationMemberAddRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import organization_member_add

    user = LiteLLM_UserTable(user_id="user-1", user_role="internal_user")
    async def create_membership(data):
        return LiteLLM_OrganizationMembershipTable(
            user_id="user-1",
            organization_id="org-1",
            user_role="internal_user",
            budget_id=data.get("budget_id"),
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
        )

    mock_db = SimpleNamespace(
        litellm_organizationtable=SimpleNamespace(find_unique=AsyncMock(return_value=SimpleNamespace())),
        litellm_usertable=SimpleNamespace(find_unique=AsyncMock(return_value=user)),
        litellm_organizationmembership=SimpleNamespace(create=create_membership),
    )
    mock_prisma = SimpleNamespace(db=mock_db)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.organization_endpoints._verify_org_access",
        AsyncMock(),
    )

    response = await organization_member_add(
        data=OrganizationMemberAddRequest(
            organization_id="org-1",
            member={"role": "internal_user", "user_id": "user-1"},
            **budget_payload,
        ),
        http_request=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert response.updated_organization_memberships[0].budget_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget_payload",
    [{"max_budget_in_organization": None}, {}],
    ids=["explicit-null", "omitted"],
)
async def test_organization_member_update_budget_omission_and_null_preserve_existing_budget(
    budget_payload, monkeypatch
):
    from datetime import datetime
    from types import SimpleNamespace

    from litellm.proxy._types import LitellmUserRoles, OrganizationMemberUpdateRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import organization_endpoints

    class BudgetState:
        def __init__(self) -> None:
            self.max_budget: float | None = 100.0

        def store(self, max_budget: float | None) -> None:
            self.max_budget = max_budget

    budget_state = BudgetState()

    def membership_row():
        row = MagicMock()
        row.budget_id = "budget-1"

        def dump(**_):
            return {
                "user_id": "user-1",
                "organization_id": "org-1",
                "user_role": "internal_user",
                "budget_id": "budget-1",
                "created_at": datetime(2024, 1, 1),
                "updated_at": datetime(2024, 1, 1),
                "litellm_budget_table": {"budget_id": "budget-1", "max_budget": budget_state.max_budget},
            }

        row.model_dump.side_effect = dump
        return row

    async def update_budget(*, budget_obj, user_api_key_dict):
        budget_state.store(budget_obj.max_budget)

    mock_db = SimpleNamespace(
        litellm_organizationtable=SimpleNamespace(find_unique=AsyncMock(return_value=SimpleNamespace())),
        litellm_organizationmembership=SimpleNamespace(
            find_unique=AsyncMock(side_effect=[membership_row(), membership_row()]),
            update=AsyncMock(),
        ),
        litellm_usertable=SimpleNamespace(
            find_unique=AsyncMock(return_value=SimpleNamespace(user_role="internal_user"))
        ),
    )
    mock_prisma = SimpleNamespace(db=mock_db)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    monkeypatch.setattr(organization_endpoints, "update_budget", update_budget)
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.organization_endpoints._verify_org_access",
        AsyncMock(),
    )

    response = await organization_endpoints.organization_member_update(
        data=OrganizationMemberUpdateRequest(
            organization_id="org-1",
            user_id="user-1",
            **budget_payload,
        ),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert response.litellm_budget_table is not None
    assert response.litellm_budget_table.max_budget == 100.0


@pytest.mark.asyncio
async def test_organization_member_delete_rejects_unauthorized_caller(patched_org_prisma, unauthorized_caller):
    from litellm.proxy._types import OrganizationMemberDeleteRequest
    from litellm.proxy.management_endpoints.organization_endpoints import (
        organization_member_delete,
    )

    data = OrganizationMemberDeleteRequest(
        organization_id="org-victim",
        user_id="some-other-user",
    )

    with pytest.raises(HTTPException) as exc:
        await organization_member_delete(
            data=data,
            user_api_key_dict=unauthorized_caller,
        )
    assert exc.value.status_code == 403


@pytest.mark.parametrize(
    "body",
    [{"tpm_limit": ""}, {"tmp_limit": None}],
    ids=["non-numeric-limit", "unknown-key"],
)
def test_v2_model_rejects_invalid_body(body):
    """A non-numeric limit and an unknown/misspelled key are both rejected at model validation (422 at the route)."""
    from pydantic import ValidationError

    from litellm.proxy._types import OrganizationUpdateRequestV2

    with pytest.raises(ValidationError):
        OrganizationUpdateRequestV2.model_validate(body)


class _FakeTxContext:
    def __init__(self, tx):
        self._tx = tx

    async def __aenter__(self):
        return self._tx

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def _run_update_organization_v2(
    monkeypatch,
    *,
    body: dict,
    existing_budget_id,
    existing_metadata,
    existing_object_permission_id=None,
    existing_object_permission_row=None,
    key_rows=(),
):
    from litellm.proxy._types import (
        LitellmUserRoles,
        OrganizationUpdateRequestV2,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.management_endpoints.organization_endpoints import (
        update_organization_v2,
    )
    from litellm.proxy.utils import jsonify_object

    mock_prisma_client = AsyncMock()
    mock_prisma_client.jsonify_object = jsonify_object

    existing_org = MagicMock()
    existing_org.budget_id = existing_budget_id
    existing_org.object_permission_id = existing_object_permission_id
    existing_org.metadata = existing_metadata

    mock_prisma_client.db.litellm_organizationtable.find_unique = AsyncMock(return_value=existing_org)
    mock_prisma_client.db.litellm_organizationtable.update = AsyncMock(return_value=MagicMock())
    mock_prisma_client.db.litellm_budgettable.update = AsyncMock()
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=existing_object_permission_row
    )
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=list(key_rows))

    tx = MagicMock()
    tx.litellm_organizationtable = mock_prisma_client.db.litellm_organizationtable
    tx.litellm_budgettable = mock_prisma_client.db.litellm_budgettable
    tx.litellm_objectpermissiontable.upsert = AsyncMock()
    mock_prisma_client.db.tx = MagicMock(return_value=_FakeTxContext(tx))
    mock_prisma_client.tx = tx

    call_order = MagicMock()
    call_order.attach_mock(tx.litellm_objectpermissiontable.upsert, "permission_upsert")
    call_order.attach_mock(mock_prisma_client.db.litellm_organizationtable.update, "org_update")
    mock_prisma_client.call_order = call_order

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr(organization_endpoints, "verify_org_access", AsyncMock())

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")
    await update_organization_v2(
        organization_id="org-1",
        data=OrganizationUpdateRequestV2.model_validate(body),
        user_api_key_dict=auth,
    )
    return mock_prisma_client


def test_v2_update_route_is_public_in_openapi():
    """PATCH /v2/organization/{organization_id} is a public route: hiding it again (include_in_schema=False)
    would drop it from openapi.json, /docs, and the generated UI API types."""
    from litellm.proxy.proxy_server import get_openapi_schema

    v2_path = get_openapi_schema()["paths"].get("/v2/organization/{organization_id}")
    assert v2_path is not None
    assert "patch" in v2_path


@pytest.mark.asyncio
async def test_v2_update_clears_tpm_limit_and_metadata(monkeypatch):
    """A cleared tpm_limit is written to the budget row as None; a cleared metadata is written as {}."""
    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"tpm_limit": None, "metadata": None},
        existing_budget_id="budget-1",
        existing_metadata={"stale": "value"},
    )

    budget_write = prisma.db.litellm_budgettable.update.await_args
    assert budget_write.kwargs["where"] == {"budget_id": "budget-1"}
    assert budget_write.kwargs["data"]["tpm_limit"] is None
    assert "soft_budget" not in budget_write.kwargs["data"]

    write_data = prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]
    assert json.loads(write_data["metadata"]) == {}
    assert "budget_id" not in write_data


@pytest.mark.asyncio
async def test_v2_update_untouched_fields_not_written(monkeypatch):
    """Omitted fields are left untouched: only organization_alias is written, no budget-row write."""
    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"organization_alias": "renamed"},
        existing_budget_id="budget-1",
        existing_metadata={"keep": "me"},
    )

    prisma.db.litellm_budgettable.update.assert_not_awaited()
    write_data = prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]
    assert write_data["organization_alias"] == "renamed"
    assert "metadata" not in write_data
    assert "tpm_limit" not in write_data


@pytest.mark.asyncio
async def test_v2_update_metadata_replaces_not_merges(monkeypatch):
    """Sending metadata replaces the stored blob wholesale; a previously-present key is gone."""
    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"metadata": {"a": 1}},
        existing_budget_id="budget-1",
        existing_metadata={"stale": "value"},
    )
    write_data = prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]
    assert json.loads(write_data["metadata"]) == {"a": 1}


@pytest.mark.asyncio
async def test_v2_rejects_null_clear_of_non_nullable_fields(monkeypatch):
    """organization_alias and models are non-nullable columns, so a null clear is a 422, not a 500."""
    from litellm.proxy._types import LitellmUserRoles, OrganizationUpdateRequestV2, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import update_organization_v2

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", AsyncMock())
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")

    for body in ({"organization_alias": None}, {"models": None}):
        with pytest.raises(HTTPException) as exc:
            await update_organization_v2(
                organization_id="org-1",
                data=OrganizationUpdateRequestV2.model_validate(body),
                user_api_key_dict=auth,
            )
        assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_v2_rejects_negative_max_budget(monkeypatch):
    """v2 rejects a negative max_budget with a 422 before touching the DB."""
    from litellm.proxy._types import LitellmUserRoles, OrganizationUpdateRequestV2, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import update_organization_v2

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", AsyncMock())

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")
    with pytest.raises(HTTPException) as exc:
        await update_organization_v2(
            organization_id="org-1",
            data=OrganizationUpdateRequestV2.model_validate({"max_budget": -5}),
            user_api_key_dict=auth,
        )
    assert exc.value.status_code == 422
    assert "max_budget" in str(exc.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["tpm_limit", "rpm_limit", "max_parallel_requests"])
async def test_v2_rejects_negative_integer_limits(monkeypatch: pytest.MonkeyPatch, field: str):
    """v2 rejects negative tpm/rpm/parallel-request limits with a 422 instead of persisting them to the budget row."""
    from litellm.proxy._types import LitellmUserRoles, OrganizationUpdateRequestV2, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import update_organization_v2

    prisma_mock = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")
    with pytest.raises(HTTPException) as exc:
        await update_organization_v2(
            organization_id="org-1",
            data=OrganizationUpdateRequestV2.model_validate({field: -1}),
            user_api_key_dict=auth,
        )
    assert exc.value.status_code == 422
    assert field in str(exc.value.detail)
    prisma_mock.db.tx.assert_not_called()
    prisma_mock.db.litellm_budgettable.update.assert_not_awaited()
    prisma_mock.db.litellm_organizationtable.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_v2_rejects_unparseable_budget_duration(monkeypatch: pytest.MonkeyPatch):
    """v2 rejects a budget_duration the parser can't read with a 422 instead of persisting it alongside a silent
    next-midnight fallback reset."""
    from litellm.proxy._types import LitellmUserRoles, OrganizationUpdateRequestV2, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import update_organization_v2

    prisma_mock = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")
    with pytest.raises(HTTPException) as exc:
        await update_organization_v2(
            organization_id="org-1",
            data=OrganizationUpdateRequestV2.model_validate({"budget_duration": "bogus"}),
            user_api_key_dict=auth,
        )
    assert exc.value.status_code == 422
    assert "budget_duration" in str(exc.value.detail)
    prisma_mock.db.tx.assert_not_called()
    prisma_mock.db.litellm_budgettable.update.assert_not_awaited()
    prisma_mock.db.litellm_organizationtable.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_v2_rejects_caller_without_org_access(monkeypatch):
    """v2 runs the real _verify_org_access guard: a non-admin without ORG_ADMIN on the org gets 403 and no write."""
    from litellm.proxy._types import LitellmUserRoles, OrganizationUpdateRequestV2, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.management_endpoints.organization_endpoints import update_organization_v2

    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr(organization_endpoints, "user_api_key_has_admin_view", lambda _: False)

    caller = MagicMock()
    caller.organization_memberships = []
    monkeypatch.setattr(organization_endpoints, "get_user_object", AsyncMock(return_value=caller))

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user-1")
    with pytest.raises(HTTPException) as exc:
        await update_organization_v2(
            organization_id="org-1",
            data=OrganizationUpdateRequestV2.model_validate({"tpm_limit": 5}),
            user_api_key_dict=auth,
        )
    assert exc.value.status_code == 403
    mock_prisma_client.db.litellm_organizationtable.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_v2_wires_object_permission_onto_org_write(monkeypatch):
    """A sent object_permission merges over the existing permission row and its id is linked onto the org write."""
    existing_row = MagicMock()
    existing_row.model_dump.return_value = {
        "object_permission_id": "op-123",
        "mcp_servers": ["server-1"],
    }

    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"object_permission": {"vector_stores": ["vs-1"]}},
        existing_budget_id="budget-1",
        existing_metadata={},
        existing_object_permission_id="op-123",
        existing_object_permission_row=existing_row,
    )

    upsert = prisma.tx.litellm_objectpermissiontable.upsert.await_args.kwargs
    assert upsert["where"] == {"object_permission_id": "op-123"}
    assert upsert["data"]["update"]["mcp_servers"] == ["server-1"]
    assert upsert["data"]["update"]["vector_stores"] == ["vs-1"]
    write_data = prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]
    assert write_data["object_permission_id"] == "op-123"


@pytest.mark.asyncio
async def test_v2_object_permission_upsert_runs_inside_transaction(monkeypatch):
    """The permission upsert runs on the tx client, before the org write that links it, so a rollback cannot
    leave merged grants live on a row the org still points at."""
    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"object_permission": {"vector_stores": ["vs-1"]}},
        existing_budget_id="budget-1",
        existing_metadata={},
    )

    prisma.tx.litellm_objectpermissiontable.upsert.assert_awaited_once()
    prisma.db.litellm_objectpermissiontable.upsert.assert_not_awaited()

    upsert = prisma.tx.litellm_objectpermissiontable.upsert.await_args.kwargs
    linked_id = prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]["object_permission_id"]
    assert upsert["where"] == {"object_permission_id": linked_id}
    assert upsert["data"]["create"]["object_permission_id"] == linked_id

    ordered = [name for name, _, _ in prisma.call_order.mock_calls if name in ("permission_upsert", "org_update")]
    assert ordered == ["permission_upsert", "org_update"]


@pytest.mark.asyncio
async def test_v2_clears_object_permission_when_sent_null(monkeypatch):
    """object_permission: null detaches the org's permission row (object_permission_id -> None), no merge."""
    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"object_permission": None},
        existing_budget_id="budget-1",
        existing_metadata={},
    )

    prisma.tx.litellm_objectpermissiontable.upsert.assert_not_awaited()
    prisma.db.litellm_objectpermissiontable.find_unique.assert_not_awaited()
    write_data = prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]
    assert write_data["object_permission_id"] is None


@pytest.mark.asyncio
async def test_v2_rejects_empty_object_permission(monkeypatch):
    """object_permission: {} merges nothing, so it is rejected (send null to clear) rather than silently leaving grants."""
    from litellm.proxy._types import LitellmUserRoles, OrganizationUpdateRequestV2, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.management_endpoints.organization_endpoints import update_organization_v2

    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr(organization_endpoints, "verify_org_access", AsyncMock())

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")
    with pytest.raises(HTTPException) as exc:
        await update_organization_v2(
            organization_id="org-1",
            data=OrganizationUpdateRequestV2.model_validate({"object_permission": {}}),
            user_api_key_dict=auth,
        )
    assert exc.value.status_code == 422
    assert "object_permission" in str(exc.value.detail)
    mock_prisma_client.db.litellm_organizationtable.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_v2_writes_budget_and_org_in_one_transaction(monkeypatch):
    """A change touching both the budget row and the org row runs both writes inside one prisma transaction."""
    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"tpm_limit": 500, "metadata": {"a": 1}},
        existing_budget_id="budget-1",
        existing_metadata={},
    )

    prisma.db.tx.assert_called_once()
    prisma.db.litellm_budgettable.update.assert_awaited_once()
    prisma.db.litellm_organizationtable.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_v2_serializes_model_max_budget_on_budget_write(monkeypatch):
    """model_max_budget is a Json column, so it is JSON-serialized on the budget-row write like new_budget/metadata."""
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.validate_model_max_budget",
        lambda _: None,
    )

    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"model_max_budget": {"gpt-4o": {"max_budget": 10}}},
        existing_budget_id="budget-1",
        existing_metadata={},
    )

    written = prisma.db.litellm_budgettable.update.await_args.kwargs["data"]["model_max_budget"]
    assert isinstance(written, str)
    assert json.loads(written) == {"gpt-4o": {"max_budget": 10}}


async def _run_legacy_update_organization(
    monkeypatch: pytest.MonkeyPatch, *, body: dict[str, object], existing_budget_id: str
) -> AsyncMock:
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.management_endpoints.organization_endpoints import update_organization
    from litellm.proxy.utils import jsonify_object

    mock_prisma_client = AsyncMock()
    mock_prisma_client.jsonify_object = jsonify_object

    existing_org = MagicMock()
    existing_org.budget_id = existing_budget_id
    existing_org.metadata = {}
    mock_prisma_client.db.litellm_organizationtable.find_unique = AsyncMock(return_value=existing_org)
    mock_prisma_client.db.litellm_organizationtable.update = AsyncMock(return_value=MagicMock())
    mock_prisma_client.db.litellm_budgettable.update = AsyncMock()

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr(organization_endpoints, "verify_org_access", AsyncMock())

    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")
    await update_organization(request=request, user_api_key_dict=auth)
    return mock_prisma_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"organization_id": "org-1", "tpm_limit": None},
        {"organization_id": "org-1", "litellm_budget_table": {"tpm_limit": None}},
    ],
)
async def test_legacy_update_clears_tpm_limit_when_sent_null(monkeypatch, body):
    """PATCH /organization/update with tpm_limit: null writes None to the budget row instead of dropping it."""
    prisma = await _run_legacy_update_organization(monkeypatch, body=body, existing_budget_id="budget-1")

    budget_write = prisma.db.litellm_budgettable.update.await_args
    assert budget_write.kwargs["where"] == {"budget_id": "budget-1"}
    assert budget_write.kwargs["data"]["tpm_limit"] is None
    assert "rpm_limit" not in budget_write.kwargs["data"]
    assert "tpm_limit" not in prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]


@pytest.mark.asyncio
async def test_legacy_update_without_budget_fields_skips_budget_write(monkeypatch):
    """Omitted budget fields are left untouched: renaming the org must not write the budget row."""
    prisma = await _run_legacy_update_organization(
        monkeypatch,
        body={"organization_id": "org-1", "organization_alias": "renamed"},
        existing_budget_id="budget-1",
    )

    prisma.db.litellm_budgettable.update.assert_not_awaited()
    assert prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]["organization_alias"] == "renamed"


@pytest.mark.asyncio
async def test_legacy_update_writes_sent_metadata_to_the_organization_row(monkeypatch):
    prisma = await _run_legacy_update_organization(
        monkeypatch,
        body={"organization_id": "org-1", "metadata": {"team": "search", "limits": {"rpm": 5}}},
        existing_budget_id="budget-1",
    )

    organization_write = prisma.db.litellm_organizationtable.update.await_args
    assert organization_write.kwargs["where"] == {"organization_id": "org-1"}
    assert json.loads(organization_write.kwargs["data"]["metadata"]) == {"team": "search", "limits": {"rpm": 5}}
    prisma.db.litellm_budgettable.update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [[], ["litellm_budget_table"], "organization_id", 5, 1.5, True, None])
async def test_legacy_update_rejects_json_body_that_is_not_an_object_before_reading_the_database(monkeypatch, body):
    from pydantic import ValidationError

    from litellm.proxy import proxy_server

    with pytest.raises(ValidationError):
        await _run_legacy_update_organization(monkeypatch, body=body, existing_budget_id="budget-1")

    proxy_server.prisma_client.db.litellm_organizationtable.find_unique.assert_not_awaited()
    proxy_server.prisma_client.db.litellm_organizationtable.update.assert_not_awaited()


def test_build_budget_write_data_recomputes_reset_at_on_duration():
    """A sent budget_duration recomputes budget_reset_at so the reset window follows the new duration."""
    from litellm.proxy.management_endpoints.organization_endpoints import build_budget_write_data

    data = build_budget_write_data({"budget_duration": "30d"}, "admin-1")
    assert data["budget_duration"] == "30d"
    assert "budget_reset_at" in data
    assert data["updated_by"] == "admin-1"


def test_build_budget_write_data_no_reset_at_without_duration():
    """Clearing a limit writes it through untouched and does not recompute budget_reset_at."""
    from litellm.proxy.management_endpoints.organization_endpoints import build_budget_write_data

    data = build_budget_write_data({"tpm_limit": None}, "admin-1")
    assert data["tpm_limit"] is None
    assert "budget_reset_at" not in data


def test_build_budget_write_data_clears_reset_at_with_null_duration():
    """Clearing budget_duration also nulls budget_reset_at so no stale reset timestamp survives."""
    from litellm.proxy.management_endpoints.organization_endpoints import build_budget_write_data

    data = build_budget_write_data({"budget_duration": None}, "admin-1")
    assert data["budget_duration"] is None
    assert data["budget_reset_at"] is None


@pytest.mark.asyncio
async def test_get_organization_daily_activity_non_admin_without_org_admin_role_sees_nothing(
    monkeypatch,
):
    """A caller who is ORG_ADMIN of no organization must resolve to an EMPTY id
    list, never to None. None means "no entity filter" downstream, i.e. every
    organization's spend, so the natural simplification of falling back to None
    on an empty membership set turns a scoping rule into a proxy-wide leak. The
    organization-alias lookup must be scoped by that same empty list rather than
    reading the whole table.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.management_endpoints.organization_endpoints import (
        get_organization_daily_activity,
    )

    mock_prisma_client = AsyncMock()
    org_table_find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_organizationtable.find_many = org_table_find_many
    mock_prisma_client.db.litellm_organizationmembership.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.organization_endpoints.user_api_key_has_admin_view",
        lambda _: False,
    )

    get_daily_activity_mock = AsyncMock(return_value=MagicMock(name="SpendAnalyticsPaginatedResponse"))
    monkeypatch.setattr(organization_endpoints, "get_daily_activity", get_daily_activity_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="no-orgs-user")
    await get_organization_daily_activity(
        organization_ids=None,
        start_date="2024-04-01",
        end_date="2024-04-30",
        model=None,
        api_key=None,
        page=1,
        page_size=10,
        exclude_organization_ids=None,
        user_api_key_dict=auth,
    )

    assert get_daily_activity_mock.call_args.kwargs["entity_id"] == []
    org_table_find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_find_member_if_email_missing_row_raises_documented_400():
    """A user_email lookup that matches nothing returns None instead of raising, so the
    only failure the surrounding try/except models is never entered. Without an explicit
    None guard the next line dereferences None and /organization/member_add answers with
    an AttributeError-driven 500 rather than the documented 400.
    """
    from litellm.proxy.management_endpoints.organization_endpoints import (
        find_member_if_email,
    )

    prisma_client = AsyncMock()
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        await find_member_if_email("missing@example.com", prisma_client)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "error": (
            "Unique user not found for user_email=missing@example.com. Potential duplicate OR "
            "non-existent user_email in LiteLLM_UserTable. Use 'user_id' instead."
        )
    }


@pytest.mark.asyncio
async def test_new_organization_rejects_shared_alias_tool_permission_key():
    """/organization/new creates its permission row through its own helper, so the
    ambiguous mcp_tool_permissions key check (LIT-4982) has to run there too."""
    from litellm.proxy._types import LiteLLM_ObjectPermissionBase, NewOrganizationRequest
    from litellm.proxy.management_endpoints.organization_endpoints import (
        _set_object_permission,
    )

    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.find_many = AsyncMock(
        return_value=[
            MagicMock(server_id="wiki-a-id", alias="wiki", server_name="wiki_a"),
            MagicMock(server_id="wiki-b-id", alias="wiki", server_name="wiki_b"),
        ]
    )
    prisma_client.db.litellm_objectpermissiontable.create = AsyncMock()
    data = NewOrganizationRequest(
        organization_alias="org",
        object_permission=LiteLLM_ObjectPermissionBase(mcp_tool_permissions={"wiki": ["ask_question"]}),
    )

    with pytest.raises(HTTPException) as exc_info:
        await _set_object_permission(data=data, prisma_client=prisma_client)

    assert exc_info.value.status_code == 400
    assert "wiki-a-id" in str(exc_info.value.detail)
    assert "wiki-b-id" in str(exc_info.value.detail)
    prisma_client.db.litellm_objectpermissiontable.create.assert_not_called()


@pytest.mark.asyncio
async def test_new_organization_temp_budget_fields_go_to_budget_row_not_metadata(monkeypatch):
    """temp_budget_increase/expiry are budget columns and also key-metadata field names, so
    /organization/new must write them to the budget row and keep the datetime out of the org
    metadata JSON (a datetime there broke JSON serialization and 500'd the request)."""
    from datetime import datetime, timezone

    from litellm.proxy._types import LitellmUserRoles, NewOrganizationRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import new_organization
    from litellm.proxy.utils import PrismaClient

    expiry = datetime(2099, 1, 1, tzinfo=timezone.utc)
    prisma_client = MagicMock()
    prisma_client.jsonify_object = MagicMock(side_effect=lambda data: PrismaClient.jsonify_object(prisma_client, data))
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_budgettable.create = AsyncMock(return_value=MagicMock(budget_id="budget-1"))
    prisma_client.db.litellm_organizationtable.create = AsyncMock(return_value={"organization_id": "org-1"})
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True, raising=False)

    response = await new_organization(
        data=NewOrganizationRequest(
            organization_alias="org",
            max_budget=10,
            temp_budget_increase=5,
            temp_budget_expiry=expiry,
        ),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert response == {"organization_id": "org-1"}
    budget_write = prisma_client.db.litellm_budgettable.create.await_args.kwargs["data"]
    assert (budget_write["max_budget"], budget_write["temp_budget_increase"], budget_write["temp_budget_expiry"]) == (
        10,
        5,
        expiry,
    )
    org_write = prisma_client.db.litellm_organizationtable.create.await_args.kwargs["data"]
    assert org_write["budget_id"] == "budget-1"
    assert json.loads(org_write.get("metadata", "{}")) == {}


def test_v2_update_organization_is_in_openapi_schema():
    """PATCH /v2/organization/{organization_id} is documented in the generated OpenAPI spec."""
    from fastapi import FastAPI

    from litellm.proxy.management_endpoints.organization_endpoints import router

    app = FastAPI()
    app.include_router(router)

    v2_path = app.openapi()["paths"]["/v2/organization/{organization_id}"]
    assert v2_path["patch"]["tags"] == ["organization management"]
    assert "OrganizationUpdateRequestV2" in json.dumps(v2_path["patch"]["requestBody"])


def _organization_route_targets() -> list[tuple[str, str]]:
    from fastapi.routing import APIRoute

    from litellm.proxy.management_endpoints.organization_endpoints import router

    return [
        (method, route.path.replace("{organization_id}", "org-under-test"))
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in sorted(route.methods - {"HEAD", "OPTIONS"})
    ]


_ORGANIZATION_ROUTE_REQUESTS: Final[Mapping[tuple[str, str], Mapping[str, object]]] = MappingProxyType(
    {
        ("POST", "/organization/new"): {"json": {"organization_alias": "org-under-test"}},
        ("DELETE", "/organization/delete"): {"json": {"organization_ids": ["org-under-test"]}},
        ("GET", "/organization/info"): {"params": {"organization_id": "org-under-test"}},
        ("POST", "/organization/info"): {"json": {"organizations": ["org-under-test"]}},
        ("POST", "/organization/member_add"): {
            "json": {"organization_id": "org-under-test", "member": {"user_id": "user-1", "role": "internal_user"}}
        },
        ("PATCH", "/organization/member_update"): {"json": {"organization_id": "org-under-test", "user_id": "user-1"}},
        ("DELETE", "/organization/member_delete"): {"json": {"organization_id": "org-under-test", "user_id": "user-1"}},
    }
)


def _organization_request(method: str, path: str) -> Mapping[str, object]:
    return _ORGANIZATION_ROUTE_REQUESTS.get((method, path), {"json": {}})


def _organization_test_client() -> TestClient:
    from fastapi import FastAPI

    from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.management_endpoints.organization_endpoints import router
    from litellm.proxy.proxy_server import openai_exception_handler

    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ProxyException, openai_exception_handler)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="sk-test", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(("method", "path"), _organization_route_targets())
def test_organization_routes_are_blocked_without_enterprise_license(monkeypatch, method, path):
    """Every /organization route is enterprise-only, even for a proxy admin sending a valid request."""
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", False, raising=False)
    monkeypatch.setattr(proxy_server, "prisma_client", None, raising=False)

    response = _organization_test_client().request(method, path, **_organization_request(method, path))

    assert response.status_code == 403
    assert "Organizations" in response.json()["detail"]["error"]


@pytest.mark.parametrize(("method", "path"), _organization_route_targets())
def test_organization_routes_reach_their_handler_with_enterprise_license(monkeypatch, method, path):
    """The same request a license refuses above now reaches the handler, which is the code reporting the missing database."""
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy._types import CommonProxyErrors

    monkeypatch.setattr(proxy_server, "premium_user", True, raising=False)
    monkeypatch.setattr(proxy_server, "prisma_client", None, raising=False)

    response = _organization_test_client().request(method, path, **_organization_request(method, path))

    assert response.status_code == 500
    assert any(
        message in response.text for message in (CommonProxyErrors.db_not_connected_error.value, "No db connected")
    )


def _column_matches(value: object, condition: object) -> bool:
    if isinstance(condition, Mapping):
        allowed: Final = condition.get("in")
        return isinstance(allowed, (list, tuple)) and value in allowed
    return value == condition


def _row_matches(row: object, where: Mapping[str, object]) -> bool:
    """Prisma's equality and ``in`` filters, the only ones these endpoints send to the tables below."""
    return all(_column_matches(getattr(row, column), condition) for column, condition in where.items())


class _Rows:
    """One table as one database holds it: the writer, or a read replica lagging behind it. Each delete notes the rows
    it takes whose last good entry ``served`` still finds, and the first one can be preceded by a row arriving, as one a
    concurrent request writes between a read and that delete does."""

    def __init__(
        self,
        rows: Sequence[object],
        *,
        served: Callable[[object], Awaitable[bool]] | None = None,
        arrives_before_first_delete: object | None = None,
    ) -> None:
        self.rows: tuple[object, ...] = tuple(rows)
        self.deleted_while_served: list[object] = []
        self._served = served
        self._arriving = arrives_before_first_delete

    async def find_many(self, where: Mapping[str, object]) -> list[object]:
        return [row for row in self.rows if _row_matches(row, where)]

    async def find_unique(self, where: Mapping[str, object]) -> object | None:
        return next(iter(await self.find_many(where)), None)

    async def delete_many(self, where: Mapping[str, object]) -> int:
        if self._arriving is not None:
            self.rows = (*self.rows, self._arriving)
            self._arriving = None
        doomed: Final = await self.find_many(where)
        if self._served is not None:
            self.deleted_while_served.extend([row for row in doomed if await self._served(row)])
        self.rows = tuple(row for row in self.rows if row not in doomed)
        return len(doomed)

    async def delete(self, where: Mapping[str, Mapping[str, object]]) -> object | None:
        """A delete by one compound unique key, such as a membership's ``user_id_organization_id``."""
        unique: Final = next(iter(where.values()))
        doomed: Final = await self.find_unique(unique)
        await self.delete_many(unique)
        return doomed

    async def create(self, data: Mapping[str, object]) -> object:
        raise AssertionError(f"these tests create no row, yet got {data}")


def _served_by(
    cache: "UserApiKeyCache", cache_key_of: Callable[[object], str], model: type[BaseModel]
) -> Callable[[object], Awaitable[bool]]:
    """Whether a row's last good entry is still served: not yet revoked."""
    from litellm.proxy.common_utils.user_api_key_cache import recall_last_good

    async def served(row: object) -> bool:
        return await recall_last_good(cache, cache_key_of(row), model) is not None

    return served


def _routed_database(writer: Mapping[str, object], replica: Mapping[str, object]) -> SimpleNamespace:
    """A ``PrismaClient`` with ``DATABASE_URL_READ_REPLICA`` set: ``db`` sends writes to the writer, which ``writer_db``
    is, and reads to the replica unless the context's reads are pinned to the writer."""
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer_db: Final = PrismaWrapper(original_prisma=SimpleNamespace(**writer), iam_token_db_auth=False)
    replica_db: Final = PrismaWrapper(original_prisma=SimpleNamespace(**replica), iam_token_db_auth=False)
    return SimpleNamespace(db=RoutingPrismaWrapper(writer=writer_db, reader=replica_db), writer_db=writer_db)


def _org_endpoints_use(monkeypatch: pytest.MonkeyPatch, prisma_client: object, cache: "UserApiKeyCache") -> None:
    """The proxy globals the organization endpoints read their license, database and auth cache from."""
    for name, value in (
        ("premium_user", True),
        ("prisma_client", prisma_client),
        ("user_api_key_cache", cache),
        ("proxy_logging_obj", None),
    ):
        # test-quality-ok: the endpoints import these from proxy_server when called; no parameter carries them
        monkeypatch.setattr(f"litellm.proxy.proxy_server.{name}", value, raising=False)


def _deleted_organization_table() -> SimpleNamespace:
    return SimpleNamespace(delete=AsyncMock(return_value=SimpleNamespace(organization_id="org-doomed")))


@pytest.mark.asyncio
async def test_delete_organization_evicts_the_cache_of_the_keys_it_deletes(monkeypatch):
    """/organization/delete bulk-deletes the org's keys without going through /key/delete, so the
    key objects and the jwt_key_mapping entries (issuer-scoped ones included) pointing at them
    must be evicted here, or a deleted key keeps authenticating and a JWT identity keeps resolving
    a token hash that no longer exists until the TTLs expire. The FK cascade drops the mapping
    rows with the key rows, so the cache keys have to be read before the delete (LIT-5387)."""
    from litellm.proxy._types import DeleteOrganizationRequest, LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.auth_checks import jwt_key_mapping_cache_key
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.management_endpoints.organization_endpoints import delete_organization

    doomed_cache_keys: Final = (
        "hashed-org-key",
        jwt_key_mapping_cache_key("sub", "svc-account", None),
        jwt_key_mapping_cache_key("sub", "svc-account", "https://issuer.example"),
    )
    kept_cache_keys: Final = ("hashed-other-key", jwt_key_mapping_cache_key("sub", "other-account", None))
    kept_row: Final = JWTMappingRow("hashed-other-key", "sub", "other-account")
    jwt_table: Final = CascadingJWTMappingTable(
        [
            JWTMappingRow("hashed-org-key", "sub", "svc-account"),
            JWTMappingRow("hashed-org-key", "sub", "svc-account", "https://issuer.example"),
            kept_row,
        ]
    )
    cache: Final = UserApiKeyCache()
    for cache_key in (*doomed_cache_keys, *kept_cache_keys):
        cache.set_cache(key=cache_key, value={"retained": True})

    prisma_client: Final = AsyncMock()
    prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[SimpleNamespace(token="hashed-org-key")]
    )

    async def cascading_delete_many(where):
        jwt_table.cascade(("hashed-org-key",))
        return 1

    prisma_client.db.litellm_verificationtoken.delete_many = AsyncMock(side_effect=cascading_delete_many)
    prisma_client.db.litellm_jwtkeymapping = jwt_table
    prisma_client.db.litellm_organizationtable.delete = AsyncMock(return_value=MagicMock())

    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True, raising=False)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", cache)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", None)

    await delete_organization(
        data=DeleteOrganizationRequest(organization_ids=["org-doomed"]),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-admin", user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert all(cache.get_cache(key=cache_key) is None for cache_key in doomed_cache_keys)
    assert all(cache.get_cache(key=cache_key) == {"retained": True} for cache_key in kept_cache_keys)
    assert jwt_table.rows == (kept_row,)


@pytest.mark.asyncio
async def test_delete_organization_drops_the_last_good_entries_of_the_teams_it_deletes(monkeypatch):
    """A deleted organization's teams must not survive as last good entries a later database outage serves."""
    from litellm.models.team import LiteLLM_TeamTableCachedObj
    from litellm.proxy._types import DeleteOrganizationRequest, LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, recall_last_good, remember_last_good
    from litellm.proxy.management_endpoints.organization_endpoints import delete_organization

    cache: Final = UserApiKeyCache(serve_last_good_entries=True)
    await remember_last_good(cache, "team_id:t-org", LiteLLM_TeamTableCachedObj(team_id="t-org", team_alias="org-lane"))
    await remember_last_good(cache, "team_alias:org-lane", LiteLLM_TeamTableCachedObj(team_id="t-org"))
    prisma_client: Final = AsyncMock()
    prisma_client.db.litellm_teamtable = _Rows(
        (SimpleNamespace(team_id="t-org", team_alias="org-lane", organization_id="org-doomed"),)
    )
    prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    prisma_client.db.litellm_organizationmembership.find_many = AsyncMock(return_value=[])
    prisma_client.db.litellm_jwtkeymapping = CascadingJWTMappingTable([])
    prisma_client.db.litellm_organizationtable.delete = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True, raising=False)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", cache)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", None)

    await delete_organization(
        data=DeleteOrganizationRequest(organization_ids=["org-doomed"]),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-admin", user_role=LitellmUserRoles.PROXY_ADMIN),
    )
    await remember_last_good(cache, "team_id:t-org", LiteLLM_TeamTableCachedObj(team_id="t-org"))

    assert await recall_last_good(cache, "team_id:t-org", LiteLLM_TeamTableCachedObj) is None
    assert await recall_last_good(cache, "team_alias:org-lane", LiteLLM_TeamTableCachedObj) is None


class _OrganizationReads:
    """Answers the organization lookups auth runs, or fails each the way an unreachable Postgres does once stopped."""

    def __init__(self, org: object) -> None:
        self.stopped = False
        self._org = org
        self.db = SimpleNamespace(litellm_organizationtable=self)

    async def find_unique(self, where: Mapping[str, object], include: object = None) -> object:
        import httpx

        if self.stopped:
            raise httpx.ConnectError("All connection attempts failed")
        return self._org


async def _org_read_by_every_lookup(monkeypatch, org_id: str, *, serving: bool):
    """An auth cache holding the organization as every auth lookup caches it: with and without its budget, as the
    last known organization, and, while last good entries are served, as their last good entries."""
    from litellm.proxy._types import LiteLLM_OrganizationTable
    from litellm.proxy.auth.auth_checks import get_org_object, get_org_object_for_request
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    cache: Final = UserApiKeyCache(serve_last_good_entries=serving)
    reads: Final = _OrganizationReads(
        LiteLLM_OrganizationTable(organization_id=org_id, budget_id="b-org", created_by="admin", updated_by="admin")
    )
    await get_org_object(org_id=org_id, prisma_client=reads, user_api_key_cache=cache, include_budget_table=False)  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
    await get_org_object_for_request(
        org_id=org_id,
        prisma_client=reads,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
        user_api_key_cache=cache,
        parent_otel_span=None,
        proxy_logging_obj=None,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", cache)
    return cache, reads


async def _org_served_while_the_database_is_stopped(
    cache, reads: _OrganizationReads, org_id: str
) -> Mapping[str, bool]:
    """Which auth lookups still answer with the organization once the database stops. A lookup that does not answer
    raises the unreachable database's error, or refuses for lack of a last good entry."""
    import httpx

    from litellm.proxy.auth.auth_checks import get_org_object, get_org_object_for_request
    from litellm.proxy.db.exception_handler import LastGoodEntryMissing

    reads.stopped = True
    lookups: Final = {
        "org": lambda: get_org_object(
            org_id=org_id,
            prisma_client=reads,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
            include_budget_table=False,
        ),
        "org_for_request": lambda: get_org_object_for_request(
            org_id=org_id,
            prisma_client=reads,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
            parent_otel_span=None,
            proxy_logging_obj=None,
        ),
    }

    async def served(lookup) -> bool:
        try:
            return await lookup() is not None
        except (httpx.ConnectError, LastGoodEntryMissing):
            return False

    return {name: await served(lookup) for name, lookup in lookups.items()}


_SERVING: Final = pytest.mark.parametrize("serving", [False, True], ids=["cache_only", "last_good_entries"])


@pytest.mark.asyncio
@_SERVING
async def test_an_organization_nothing_evicted_is_served_while_the_database_is_stopped(monkeypatch, serving: bool):
    cache, reads = await _org_read_by_every_lookup(monkeypatch, "org-1", serving=serving)

    assert await _org_served_while_the_database_is_stopped(cache, reads, "org-1") == {
        "org": True,
        "org_for_request": True,
    }


@pytest.mark.asyncio
@_SERVING
async def test_v1_update_evicts_the_organization_and_its_last_good_entries(monkeypatch, serving: bool):
    """Auth reads the organization cache-first, and its last good entry lasts until it expires: an update that leaves
    either in place keeps enforcing the old limits, during a database outage until the entry expires."""
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.utils import jsonify_object

    cache, reads = await _org_read_by_every_lookup(monkeypatch, "org-1", serving=serving)
    prisma_client: Final = AsyncMock()
    prisma_client.jsonify_object = jsonify_object
    prisma_client.db.litellm_organizationtable.find_unique = AsyncMock(
        return_value=SimpleNamespace(metadata=None, budget_id="b-org", object_permission_id=None)
    )
    prisma_client.db.litellm_organizationtable.update = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr(organization_endpoints, "verify_org_access", AsyncMock())
    request: Final = MagicMock()
    request.json = AsyncMock(return_value={"organization_id": "org-1", "organization_alias": "renamed"})

    await organization_endpoints.update_organization(
        request=request,
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1"),
    )

    assert not any((await _org_served_while_the_database_is_stopped(cache, reads, "org-1")).values())


@pytest.mark.asyncio
@_SERVING
async def test_v2_update_evicts_the_organization_and_its_last_good_entries(monkeypatch, serving: bool):
    cache, reads = await _org_read_by_every_lookup(monkeypatch, "org-1", serving=serving)

    await _run_update_organization_v2(
        monkeypatch, body={"max_budget": 1.0}, existing_budget_id="b-org", existing_metadata=None
    )

    assert not any((await _org_served_while_the_database_is_stopped(cache, reads, "org-1")).values())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, evicted",
    [({"max_budget": 1.0}, True), ({"metadata": {"lane": "a"}}, False)],
    ids=["budget_changed", "budget_untouched"],
)
async def test_v2_budget_change_drops_the_last_good_entries_of_the_organizations_keys(
    monkeypatch, body: dict, evicted: bool
):
    """A key's own entry carries its organization's budget, which a v2 update writes straight to the budget row."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, recall_last_good, remember_last_good

    cache: Final = UserApiKeyCache(serve_last_good_entries=True)
    await remember_last_good(
        cache, "hashed-org-key", UserAPIKeyAuth(token="hashed-org-key", organization_max_budget=50.0)
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", cache)

    await _run_update_organization_v2(
        monkeypatch,
        body=body,
        existing_budget_id="b-org",
        existing_metadata=None,
        key_rows=(SimpleNamespace(token="hashed-org-key"),),
    )

    assert (await recall_last_good(cache, "hashed-org-key", UserAPIKeyAuth) is None) is evicted


@pytest.mark.asyncio
@_SERVING
async def test_delete_organization_evicts_the_organization_and_its_last_good_entries(monkeypatch, serving: bool):
    from litellm.proxy._types import DeleteOrganizationRequest, LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import delete_organization

    cache, reads = await _org_read_by_every_lookup(monkeypatch, "org-doomed", serving=serving)
    prisma_client: Final = AsyncMock()
    prisma_client.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    prisma_client.db.litellm_organizationmembership.find_many = AsyncMock(return_value=[])
    prisma_client.db.litellm_jwtkeymapping = CascadingJWTMappingTable([])
    prisma_client.db.litellm_organizationtable.delete = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", None)

    await delete_organization(
        data=DeleteOrganizationRequest(organization_ids=["org-doomed"]),
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-admin", user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert not any((await _org_served_while_the_database_is_stopped(cache, reads, "org-doomed")).values())


@pytest.mark.asyncio
async def test_delete_organization_answers_503_and_deletes_nothing_when_redis_cannot_record_it(monkeypatch):
    """The organization's own entries are revoked with its teams' and keys' before the first delete, so a Redis that
    cannot record the revocation refuses the delete with nothing deleted, even for an organization with no team and
    no key."""
    from litellm.proxy._types import DeleteOrganizationRequest, LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.management_endpoints.organization_endpoints import delete_organization
    from tests.unit.caching.fake_redis_cache import fake_redis_cache

    redis: Final = fake_redis_cache()
    redis.server.connected = False
    prisma_client: Final = AsyncMock()
    prisma_client.db.litellm_teamtable.find_many = AsyncMock(return_value=[])
    prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    prisma_client.db.litellm_jwtkeymapping = CascadingJWTMappingTable([])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_api_key_cache",
        UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True),
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", None)
    admin_key: Final = "sk-admin"

    with pytest.raises(HTTPException) as refused:
        await delete_organization(
            data=DeleteOrganizationRequest(organization_ids=["org-doomed"]),
            user_api_key_dict=UserAPIKeyAuth(api_key=admin_key, user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert refused.value.status_code == 503
    prisma_client.db.litellm_teamtable.delete_many.assert_not_awaited()
    prisma_client.db.litellm_organizationmembership.delete_many.assert_not_awaited()
    prisma_client.db.litellm_verificationtoken.delete_many.assert_not_awaited()
    prisma_client.db.litellm_organizationtable.delete.assert_not_awaited()


def _org_member_removal(monkeypatch, *, serves_last_good: bool) -> AsyncMock:
    """/organization/member_delete's database, behind an unreachable Redis with last good entries served or not."""
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from tests.unit.caching.fake_redis_cache import fake_redis_cache

    redis: Final = fake_redis_cache()
    redis.server.connected = False
    prisma_client: Final = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_api_key_cache",
        UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=serves_last_good),
    )
    return prisma_client


@pytest.mark.asyncio
async def test_organization_member_delete_answers_503_and_deletes_nothing_when_redis_cannot_record_it(monkeypatch):
    """The user's entry carries its organization memberships, which the org admin route checks read, so a removal
    Redis cannot revoke that entry for is refused with 503 before the delete."""
    from litellm.proxy._types import LitellmUserRoles, OrganizationMemberDeleteRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import organization_member_delete

    prisma_client: Final = _org_member_removal(monkeypatch, serves_last_good=True)

    with pytest.raises(HTTPException) as refused:
        await organization_member_delete(
            data=OrganizationMemberDeleteRequest(organization_id="org-1", user_id="member-1"),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert refused.value.status_code == 503
    prisma_client.db.litellm_organizationmembership.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_organization_member_delete_deletes_while_redis_is_down_with_last_good_entries_off(monkeypatch):
    """With last good entries off nothing is revoked, so Redis being down does not stop the removal."""
    from litellm.proxy._types import LitellmUserRoles, OrganizationMemberDeleteRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import organization_member_delete

    prisma_client: Final = _org_member_removal(monkeypatch, serves_last_good=False)

    await organization_member_delete(
        data=OrganizationMemberDeleteRequest(organization_id="org-1", user_id="member-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    prisma_client.db.litellm_organizationmembership.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_organization_member_role_change_drops_the_users_last_good_entry(monkeypatch):
    """The user's entry carries its organization memberships, which the org admin route checks read, so a role change
    drops it: a database outage after it cannot serve the old role."""
    from datetime import datetime, timezone

    from litellm.proxy._types import (
        LiteLLM_OrganizationMembershipTable,
        LiteLLM_UserTable,
        LitellmUserRoles,
        OrganizationMemberUpdateRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, recall_last_good, remember_last_good
    from litellm.proxy.management_endpoints.organization_endpoints import organization_member_update

    joined: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    admin_membership: Final = LiteLLM_OrganizationMembershipTable(
        user_id="member-1",
        organization_id="org-1",
        user_role=LitellmUserRoles.ORG_ADMIN.value,
        created_at=joined,
        updated_at=joined,
    )
    cache: Final = UserApiKeyCache(serve_last_good_entries=True)
    await remember_last_good(
        cache, "member-1", LiteLLM_UserTable(user_id="member-1", organization_memberships=[admin_membership])
    )
    prisma_client: Final = AsyncMock()
    prisma_client.db.litellm_organizationtable.find_unique = AsyncMock(return_value=MagicMock())
    prisma_client.db.litellm_organizationmembership.find_unique = AsyncMock(return_value=admin_membership)
    prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=LiteLLM_UserTable(user_id="member-1"))
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", cache)

    await organization_member_update(
        data=OrganizationMemberUpdateRequest(
            organization_id="org-1", user_id="member-1", role=LitellmUserRoles.INTERNAL_USER
        ),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    prisma_client.db.litellm_organizationmembership.update.assert_awaited_once()
    assert await recall_last_good(cache, "member-1", LiteLLM_UserTable) is None


@pytest.mark.asyncio
async def test_delete_organization_pinned_to_the_writer_revokes_the_children_a_lagging_replica_lacks(monkeypatch):
    """With a read replica, auth pins the delete's reads to the writer, so a team, key or member that the writer has
    and a lagging replica lacks is revoked before the delete that takes it, like the ones both have."""
    from litellm.models.team import LiteLLM_TeamTableCachedObj
    from litellm.proxy._types import DeleteOrganizationRequest, LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, remember_last_good
    from litellm.proxy.db.routing_prisma_wrapper import reads_pinned_to_writer
    from litellm.proxy.management_endpoints.organization_endpoints import delete_organization

    cache: Final = UserApiKeyCache(serve_last_good_entries=True)
    await remember_last_good(cache, "team_id:t-new", LiteLLM_TeamTableCachedObj(team_id="t-new"))
    await remember_last_good(cache, "hashed-new-key", UserAPIKeyAuth(token="hashed-new-key"))
    await remember_last_good(cache, "new-member", LiteLLM_UserTable(user_id="new-member"))
    old_team: Final = SimpleNamespace(team_id="t-old", team_alias=None, organization_id="org-doomed")
    old_key: Final = SimpleNamespace(token="hashed-old-key", organization_id="org-doomed")
    old_member: Final = SimpleNamespace(user_id="old-member", organization_id="org-doomed")
    teams: Final = _Rows(
        (old_team, SimpleNamespace(team_id="t-new", team_alias=None, organization_id="org-doomed")),
        served=_served_by(cache, lambda team: f"team_id:{team.team_id}", LiteLLM_TeamTableCachedObj),
    )
    keys: Final = _Rows(
        (old_key, SimpleNamespace(token="hashed-new-key", organization_id="org-doomed")),
        served=_served_by(cache, lambda key: key.token, UserAPIKeyAuth),
    )
    members: Final = _Rows(
        (old_member, SimpleNamespace(user_id="new-member", organization_id="org-doomed")),
        served=_served_by(cache, lambda member: member.user_id, LiteLLM_UserTable),
    )
    unrouted: Final = {
        "litellm_jwtkeymapping": CascadingJWTMappingTable([]),
        "litellm_organizationtable": _deleted_organization_table(),
    }
    database: Final = _routed_database(
        {
            "litellm_teamtable": teams,
            "litellm_verificationtoken": keys,
            "litellm_organizationmembership": members,
            **unrouted,
        },
        {
            "litellm_teamtable": _Rows((old_team,)),
            "litellm_verificationtoken": _Rows((old_key,)),
            "litellm_organizationmembership": _Rows((old_member,)),
            **unrouted,
        },
    )
    _org_endpoints_use(monkeypatch, database, cache)

    with reads_pinned_to_writer():
        await delete_organization(
            data=DeleteOrganizationRequest(organization_ids=["org-doomed"]),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert (teams.rows, keys.rows, members.rows) == ((), (), ())
    assert teams.deleted_while_served == []
    assert keys.deleted_while_served == []
    assert members.deleted_while_served == []


@pytest.mark.asyncio
async def test_delete_organization_revokes_a_child_added_between_a_read_and_its_delete(monkeypatch):
    """A team, key or membership added to the organization between a read and the delete that would take it is not
    deleted unrevoked: each delete takes only the rows read and revoked, and the next read finds the new one."""
    from litellm.models.team import LiteLLM_TeamTableCachedObj
    from litellm.proxy._types import DeleteOrganizationRequest, LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, remember_last_good
    from litellm.proxy.management_endpoints.organization_endpoints import delete_organization

    cache: Final = UserApiKeyCache(serve_last_good_entries=True)
    await remember_last_good(cache, "team_id:t-late", LiteLLM_TeamTableCachedObj(team_id="t-late"))
    await remember_last_good(cache, "hashed-late-key", UserAPIKeyAuth(token="hashed-late-key"))
    await remember_last_good(cache, "late-member", LiteLLM_UserTable(user_id="late-member"))
    teams: Final = _Rows(
        (SimpleNamespace(team_id="t-org", team_alias=None, organization_id="org-doomed"),),
        served=_served_by(cache, lambda team: f"team_id:{team.team_id}", LiteLLM_TeamTableCachedObj),
        arrives_before_first_delete=SimpleNamespace(team_id="t-late", team_alias=None, organization_id="org-doomed"),
    )
    keys: Final = _Rows(
        (SimpleNamespace(token="hashed-org-key", organization_id="org-doomed"),),
        served=_served_by(cache, lambda key: key.token, UserAPIKeyAuth),
        arrives_before_first_delete=SimpleNamespace(token="hashed-late-key", organization_id="org-doomed"),
    )
    members: Final = _Rows(
        (SimpleNamespace(user_id="org-member", organization_id="org-doomed"),),
        served=_served_by(cache, lambda member: member.user_id, LiteLLM_UserTable),
        arrives_before_first_delete=SimpleNamespace(user_id="late-member", organization_id="org-doomed"),
    )
    database: Final = SimpleNamespace(
        db=SimpleNamespace(
            litellm_teamtable=teams,
            litellm_verificationtoken=keys,
            litellm_organizationmembership=members,
            litellm_jwtkeymapping=CascadingJWTMappingTable([]),
            litellm_organizationtable=_deleted_organization_table(),
        )
    )
    _org_endpoints_use(monkeypatch, database, cache)

    await delete_organization(
        data=DeleteOrganizationRequest(organization_ids=["org-doomed"]),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert (teams.rows, keys.rows, members.rows) == ((), (), ())
    assert teams.deleted_while_served == []
    assert keys.deleted_while_served == []
    assert members.deleted_while_served == []


class _JWTMappingTableNotingReads(CascadingJWTMappingTable):
    """The mapping table, with the tokens each read asks for noted in ``reads``, in order."""

    def __init__(self, rows: Sequence[JWTMappingRow]) -> None:
        super().__init__(rows)
        self.reads: Final[list[tuple[str, ...]]] = []  # mutable-ok: one entry per read, in order

    async def find_many(self, where: Mapping[str, Mapping[str, Sequence[str]]]) -> list[JWTMappingRow]:
        self.reads.append(tuple(where["token"]["in"]))
        return await super().find_many(where)


@pytest.mark.asyncio
async def test_delete_organization_reads_the_jwt_mappings_of_its_keys_one_chunk_at_a_time(monkeypatch):
    """While last good entries are served, the keys' JWT mappings are read chunk by chunk, right before each chunk's
    revocation, so no mapping read's ``in`` list holds more than one chunk's tokens; every mapping is still evicted
    once the keys are gone."""
    from functools import partial

    from litellm.proxy._types import (
        DeleteOrganizationRequest,
        LitellmUserRoles,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.auth_checks import jwt_key_mapping_cache_key
    from litellm.proxy.common_utils.user_api_key_cache import (
        UserApiKeyCache,
        remember_last_good,
        revoke_and_delete_read_rows,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import delete_organization

    tokens: Final = tuple(f"hashed-org-key-{n}" for n in range(5))
    doomed_mapping_keys: Final = tuple(jwt_key_mapping_cache_key("sub", f"svc-{token}", None) for token in tokens)
    kept_mapping_key: Final = jwt_key_mapping_cache_key("sub", "other-account", None)
    cache: Final = UserApiKeyCache(serve_last_good_entries=True)
    for token in tokens:
        await remember_last_good(cache, token, UserAPIKeyAuth(token=token))
    for cache_key in (*doomed_mapping_keys, kept_mapping_key):
        cache.set_cache(key=cache_key, value={"retained": True})
    keys: Final = _Rows(
        # The key rows the endpoint reads are the database's, with their organization_id column, which the proxy's
        # LiteLLM_VerificationToken does not carry (it has org_id)
        tuple(SimpleNamespace(token=token, organization_id="org-doomed") for token in tokens),
        served=_served_by(cache, lambda key: key.token, UserAPIKeyAuth),
    )
    jwt_table: Final = _JWTMappingTableNotingReads(
        (
            *(JWTMappingRow(token, "sub", f"svc-{token}") for token in tokens),
            JWTMappingRow("hashed-other-key", "sub", "other-account"),
        )
    )
    database: Final = SimpleNamespace(
        db=SimpleNamespace(
            litellm_teamtable=_Rows(()),
            litellm_verificationtoken=keys,
            litellm_organizationmembership=_Rows(()),
            litellm_jwtkeymapping=jwt_table,
            litellm_organizationtable=_deleted_organization_table(),
        )
    )
    _org_endpoints_use(monkeypatch, database, cache)
    # Chunks of two, so five keys make three chunks without writing 5,001 rows
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.organization_endpoints.revoke_and_delete_read_rows",
        partial(revoke_and_delete_read_rows, chunk_size=2),
    )

    await delete_organization(
        data=DeleteOrganizationRequest(organization_ids=["org-doomed"]),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert jwt_table.reads == [tokens[:2], tokens[2:4], tokens[4:]]
    assert all(cache.get_cache(key=cache_key) is None for cache_key in doomed_mapping_keys)
    assert cache.get_cache(key=kept_mapping_key) == {"retained": True}
    assert keys.rows == ()
    assert keys.deleted_while_served == []


@pytest.mark.asyncio
async def test_organization_member_delete_by_email_pinned_to_the_writer_removes_the_user_the_writer_maps_it_to(
    monkeypatch,
):
    """The email is resolved where the removal writes: a lagging replica that still maps it to the user who held it
    before would revoke and remove that user instead, leaving the current holder's membership and entry in place."""
    from litellm.proxy._types import (
        LiteLLM_UserTable,
        LitellmUserRoles,
        OrganizationMemberDeleteRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, recall_last_good, remember_last_good
    from litellm.proxy.db.routing_prisma_wrapper import reads_pinned_to_writer
    from litellm.proxy.management_endpoints.organization_endpoints import organization_member_delete

    cache: Final = UserApiKeyCache(serve_last_good_entries=True)
    await remember_last_good(cache, "u-current", LiteLLM_UserTable(user_id="u-current"))
    await remember_last_good(cache, "u-former", LiteLLM_UserTable(user_id="u-former"))
    memberships: Final = _Rows(
        (
            SimpleNamespace(user_id="u-current", organization_id="org-1"),
            SimpleNamespace(user_id="u-former", organization_id="org-1"),
        )
    )
    database: Final = _routed_database(
        {
            "litellm_usertable": _Rows(
                (
                    LiteLLM_UserTable(user_id="u-current", user_email="member@example.com"),
                    LiteLLM_UserTable(user_id="u-former", user_email="former@example.com"),
                )
            ),
            "litellm_organizationmembership": memberships,
        },
        {
            "litellm_usertable": _Rows((LiteLLM_UserTable(user_id="u-former", user_email="member@example.com"),)),
            "litellm_organizationmembership": memberships,
        },
    )
    _org_endpoints_use(monkeypatch, database, cache)

    with reads_pinned_to_writer():
        await organization_member_delete(
            data=OrganizationMemberDeleteRequest(organization_id="org-1", user_email="member@example.com"),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert [membership.user_id for membership in memberships.rows] == ["u-former"]
    assert await recall_last_good(cache, "u-current", LiteLLM_UserTable) is None
    assert await recall_last_good(cache, "u-former", LiteLLM_UserTable) is not None


_FIRST_ORG: Final = "org-first"
_SECOND_ORG: Final = "org-second"


class _DrivenClock:
    """The last good entries' clock, which a test moves instead of waiting."""

    def __init__(self) -> None:
        self.now: float = 1000.0

    def __call__(self) -> float:
        return self.now


async def _outlast_the_tombstones(redis: "FakeRedisCache", clock: _DrivenClock, *, loses_redis: bool) -> None:
    """A write that takes past the time the request's record of its revocations allows, so the request revokes those
    keys through Redis again: the tombstones in Redis lapse meanwhile, and Redis is lost once the write lands when
    ``loses_redis``."""
    from litellm.constants import LAST_GOOD_TOMBSTONE_TTL_SECONDS
    from litellm.proxy.common_utils.user_api_key_cache import LAST_GOOD_LATE_STORE_MARGIN_SECONDS

    clock.now += LAST_GOOD_TOMBSTONE_TTL_SECONDS - LAST_GOOD_LATE_STORE_MARGIN_SECONDS + 1
    client: Final = redis.cache.init_async_client()
    lapsed: Final = await client.keys("last_good_tombstone:*")
    if lapsed:
        await client.delete(*lapsed)
    if loses_redis:
        redis.server.connected = False


async def _tombstones_standing(redis: "FakeRedisCache", cache_keys: Sequence[str]) -> tuple[str, ...]:
    """Which of ``cache_keys`` have a tombstone in Redis now; none while Redis is lost."""
    if not redis.server.connected:
        return ()
    client: Final = redis.cache.init_async_client()
    return tuple(
        [cache_key for cache_key in cache_keys if await client.exists(f"last_good_tombstone:{cache_key}") == 1]
    )


class _LoggedTable:
    """``table`` with each read and delete the endpoint awaits noted in ``log`` as ``<name>.<method>``, in order."""

    def __init__(self, name: str, table: _Rows, log: list[str]) -> None:
        self._name: Final = name
        self._table: Final = table
        self._log: Final = log

    async def find_many(self, where: Mapping[str, object]) -> list[object]:
        self._log.append(f"{self._name}.find_many")
        return await self._table.find_many(where)

    async def delete_many(self, where: Mapping[str, object]) -> int:
        self._log.append(f"{self._name}.delete_many")
        return await self._table.delete_many(where)


class _OrganizationTable:
    """The organization table holding ``org_ids``. Each delete is noted in ``log``, and in ``deletes`` with which of
    the organization's cache keys had a tombstone in Redis as it ran; deleting ``slow_org`` then outlasts the
    tombstones (``_outlast_the_tombstones``)."""

    def __init__(
        self,
        org_ids: Sequence[str],
        redis: "FakeRedisCache",
        clock: _DrivenClock,
        log: list[str],
        *,
        slow_org: str,
        loses_redis: bool,
    ) -> None:
        self.org_ids: Final[list[str]] = list(org_ids)  # mutable-ok: a delete takes its row
        self.deletes: Final[list[tuple[str, tuple[str, ...]]]] = []  # mutable-ok: one entry per delete, in order
        self._redis: Final = redis
        self._clock: Final = clock
        self._log: Final = log
        self._slow_org: Final = slow_org
        self._loses_redis: Final = loses_redis

    async def delete(
        self, where: Mapping[str, str], include: Mapping[str, bool] | None = None
    ) -> "LiteLLM_OrganizationTable | None":
        from litellm.proxy._types import LiteLLM_OrganizationTable
        from litellm.proxy.auth.auth_checks import organization_cache_keys

        org_id: Final = where["organization_id"]
        self._log.append("organizations.delete")
        if org_id not in self.org_ids:
            return None
        self.deletes.append((org_id, await _tombstones_standing(self._redis, organization_cache_keys(org_id))))
        self.org_ids.remove(org_id)
        if org_id == self._slow_org:
            await _outlast_the_tombstones(self._redis, self._clock, loses_redis=self._loses_redis)
        return LiteLLM_OrganizationTable(
            organization_id=org_id, budget_id=f"budget-{org_id}", created_by="admin", updated_by="admin"
        )


def _two_organizations_the_first_outlasting(
    monkeypatch: pytest.MonkeyPatch, *, serving: bool, loses_redis: bool, log: list[str]
) -> tuple[_OrganizationTable, tuple[_Rows, _Rows, _Rows]]:
    """Two organizations behind Redis, on a driven clock, with last good entries served or not: the first holds a
    team, a key and a member, the second nothing, and the first's row delete outlasts the tombstones, Redis lost once
    it lands when ``loses_redis``. Every read and delete is noted in ``log``. Returns the organization table and the
    team, key and member tables."""
    from datetime import datetime, timezone

    from litellm.proxy._types import (
        LiteLLM_OrganizationMembershipTable,
        LiteLLM_TeamTable,
    )
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from tests.unit.caching.fake_redis_cache import fake_redis_cache

    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    joined: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    teams: Final = _Rows((LiteLLM_TeamTable(team_id="t-first", organization_id=_FIRST_ORG),))
    # The key row the endpoint reads is the database's, with its organization_id column, which the proxy's
    # LiteLLM_VerificationToken does not carry (it has org_id)
    keys: Final = _Rows((SimpleNamespace(token="hashed-first-key", organization_id=_FIRST_ORG),))
    members: Final = _Rows(
        (
            LiteLLM_OrganizationMembershipTable(
                user_id="first-member", organization_id=_FIRST_ORG, created_at=joined, updated_at=joined
            ),
        )
    )
    organizations: Final = _OrganizationTable(
        (_FIRST_ORG, _SECOND_ORG), redis, clock, log, slow_org=_FIRST_ORG, loses_redis=loses_redis
    )
    database: Final = SimpleNamespace(
        db=SimpleNamespace(
            litellm_teamtable=_LoggedTable("teams", teams, log),
            litellm_verificationtoken=_LoggedTable("keys", keys, log),
            litellm_organizationmembership=_LoggedTable("members", members, log),
            litellm_jwtkeymapping=CascadingJWTMappingTable([]),
            litellm_organizationtable=organizations,
        )
    )
    cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=serving, last_good_clock=clock)
    _org_endpoints_use(monkeypatch, database, cache)
    return organizations, (teams, keys, members)


@pytest.mark.asyncio
async def test_delete_organization_revokes_the_second_organization_again_after_the_first_outlasts_the_tombstones(
    monkeypatch,
):
    """The request revokes both organizations before its first delete, then the first organization's deletes take past
    the time the request's record of that revocation allows and the tombstones lapse meanwhile, so another pod may
    store the second organization's entries again. The second organization's delete revokes them again right before
    it: its tombstones stand in Redis when its row is deleted."""
    from litellm.proxy._types import DeleteOrganizationRequest, LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.auth_checks import organization_cache_keys
    from litellm.proxy.management_endpoints.organization_endpoints import delete_organization

    organizations, _children = _two_organizations_the_first_outlasting(
        monkeypatch, serving=True, loses_redis=False, log=[]
    )

    deleted: Final = await delete_organization(
        data=DeleteOrganizationRequest(organization_ids=[_FIRST_ORG, _SECOND_ORG]),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert [org.organization_id for org in deleted] == [_FIRST_ORG, _SECOND_ORG]
    assert organizations.deletes == [
        (_FIRST_ORG, organization_cache_keys(_FIRST_ORG)),
        (_SECOND_ORG, organization_cache_keys(_SECOND_ORG)),
    ]


@pytest.mark.asyncio
async def test_delete_organization_answers_503_with_the_second_organization_kept_when_redis_is_lost_after_the_first(
    monkeypatch,
):
    """The same slow first organization, with Redis lost once its row is deleted: the second organization's revocation
    has to reach Redis again and cannot, so the request answers 503 with the second organization's row not deleted."""
    from litellm.proxy._types import DeleteOrganizationRequest, LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import delete_organization

    organizations, _children = _two_organizations_the_first_outlasting(
        monkeypatch, serving=True, loses_redis=True, log=[]
    )

    with pytest.raises(HTTPException) as refused:
        await delete_organization(
            data=DeleteOrganizationRequest(organization_ids=[_FIRST_ORG, _SECOND_ORG]),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert refused.value.status_code == 503
    assert [org_id for org_id, _standing in organizations.deletes] == [_FIRST_ORG]
    assert organizations.org_ids == [_SECOND_ORG]


@pytest.mark.asyncio
async def test_delete_organization_with_last_good_entries_off_reads_nothing_before_its_first_delete(monkeypatch):
    """With the setting off nothing is revoked, so no read runs before the first organization's tables, which are read
    and deleted in the order they were before last good entries existed: the team and member reads, which find what
    the cache evictions after their deletes drop, are the only additions. Redis lost after the first organization
    stops nothing."""
    from litellm.proxy._types import DeleteOrganizationRequest, LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import delete_organization

    log: Final[list[str]] = []  # mutable-ok: every read and delete, in order
    organizations, children = _two_organizations_the_first_outlasting(
        monkeypatch, serving=False, loses_redis=True, log=log
    )

    await delete_organization(
        data=DeleteOrganizationRequest(organization_ids=[_FIRST_ORG, _SECOND_ORG]),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    each_organization: Final = (
        "teams.find_many",
        "teams.delete_many",
        "members.find_many",
        "members.delete_many",
        "keys.find_many",
        "keys.delete_many",
        "organizations.delete",
    )
    assert log == [*each_organization, *each_organization]
    assert organizations.org_ids == []
    assert [table.rows for table in children] == [(), (), ()]


@pytest.mark.asyncio
async def test_delete_organization_records_its_tombstones_again_after_the_row_delete_so_no_outage_serves_the_old_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request first revokes organization org-doomed's entries at t=0, with a 30 s tombstone. Its delete runs at
    t=19, when the request's record shows those tombstones standing, so neither the revocation before its first delete
    nor the one paired with the organization's row delete sends Redis anything. Another pod's read began at t=19,
    before the row delete committed, and got the organization; it stores that row at t=31, once the first tombstones
    lapsed at t=30. The delete records the organization's tombstones again once its row delete returns, standing until
    t=49, so that store keeps nothing and an outage that follows serves no entry for the deleted organization, rather
    than its old row."""
    from litellm.proxy._types import (
        DeleteOrganizationRequest,
        LiteLLM_OrganizationTable,
        LitellmUserRoles,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.auth_checks import organization_cache_keys
    from litellm.proxy.management_endpoints.organization_endpoints import delete_organization
    from tests.unit.proxy.last_good_tombstone_race import (
        revoked_before_the_write,
        served_after_another_pod_stores,
        tombstone_race,
    )

    race: Final = tombstone_race(monkeypatch)
    old_row: Final = LiteLLM_OrganizationTable(
        organization_id="org-doomed", budget_id="budget-org-doomed", created_by="admin", updated_by="admin"
    )
    organizations: Final = _deleted_organization_table()
    database: Final = SimpleNamespace(
        db=SimpleNamespace(
            litellm_teamtable=_Rows(()),
            litellm_verificationtoken=_Rows(()),
            litellm_organizationmembership=_Rows(()),
            litellm_jwtkeymapping=CascadingJWTMappingTable([]),
            litellm_organizationtable=organizations,
        )
    )
    _org_endpoints_use(monkeypatch, database, race.cache)
    await revoked_before_the_write(race, organization_cache_keys("org-doomed"))

    deleted: Final = await delete_organization(
        data=DeleteOrganizationRequest(organization_ids=["org-doomed"]),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert [org.organization_id for org in deleted] == ["org-doomed"]
    organizations.delete.assert_awaited_once()
    served: Final = await served_after_another_pod_stores(race, organization_cache_keys("org-doomed")[0], old_row)
    assert served is None


@pytest.mark.asyncio
async def test_organization_member_delete_records_the_tombstone_again_so_no_outage_serves_the_removed_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request first revokes member-1's entry at t=0, with a 30 s tombstone. The removal of member-1 from org-1
    runs at t=19, when the request's record shows that tombstone standing, so the revocation paired with the
    membership's delete sends Redis nothing. Another pod's read began at t=19, before the delete committed, and got
    member-1 as an admin of org-1; it stores that row at t=31, once the first tombstone lapsed at t=30. The removal
    records member-1's tombstone again once its delete returns, standing until t=49, so that store keeps nothing and an
    outage that follows serves no entry for member-1, rather than the row that still makes it an admin of org-1."""
    from datetime import datetime, timezone

    from litellm.proxy._types import (
        LiteLLM_OrganizationMembershipTable,
        LiteLLM_UserTable,
        LitellmUserRoles,
        OrganizationMemberDeleteRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import organization_member_delete
    from tests.unit.proxy.last_good_tombstone_race import (
        revoked_before_the_write,
        served_after_another_pod_stores,
        tombstone_race,
    )

    race: Final = tombstone_race(monkeypatch)
    joined: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    admin_membership: Final = LiteLLM_OrganizationMembershipTable(
        user_id="member-1",
        organization_id="org-1",
        user_role=LitellmUserRoles.ORG_ADMIN.value,
        created_at=joined,
        updated_at=joined,
    )
    memberships: Final = _Rows((admin_membership,))
    database: Final = SimpleNamespace(db=SimpleNamespace(litellm_organizationmembership=memberships))
    _org_endpoints_use(monkeypatch, database, race.cache)
    await revoked_before_the_write(race, ("member-1",))

    removed: Final = await organization_member_delete(
        data=OrganizationMemberDeleteRequest(organization_id="org-1", user_id="member-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
    )

    assert (removed, memberships.rows) == (admin_membership, ())
    served: Final = await served_after_another_pod_stores(
        race, "member-1", LiteLLM_UserTable(user_id="member-1", organization_memberships=[admin_membership])
    )
    assert served is None
