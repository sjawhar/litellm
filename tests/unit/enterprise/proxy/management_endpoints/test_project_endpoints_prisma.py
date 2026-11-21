import asyncio
import os
import traceback
import httpx
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final
from litellm._uuid import uuid
from unittest import mock

from dotenv import load_dotenv
from fastapi import HTTPException, Request

load_dotenv()
import time


import pytest

import litellm
from litellm.proxy.management_endpoints.team_endpoints import (
    new_team,
)
from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
    new_project,
    update_project,
    delete_project,
    project_info,
)
from litellm.proxy.proxy_server import (
    LitellmUserRoles,
)
from litellm.proxy.common_utils.user_api_key_cache import (
    LAST_GOOD_LATE_STORE_MARGIN_SECONDS,
    UserApiKeyCache,
    project_cache_key,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging



from litellm.caching.caching import DualCache
from litellm.constants import LAST_GOOD_TOMBSTONE_TTL_SECONDS
from litellm.proxy._types import (
    NewProjectRequest,
    UpdateProjectRequest,
    DeleteProjectRequest,
    NewTeamRequest,
    UserAPIKeyAuth,
    LiteLLM_BudgetTable,
    LiteLLM_ObjectPermissionBase,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_ProjectTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LiteLLM_VerificationToken,
    Member,
    ProxyException,
)
from litellm.proxy.management.teams.authz import TeamAccess
from litellm.proxy.management.users.service import PrismaOrgRoles
from tests._master_key import MASTER_KEY
from tests.unit.caching.fake_redis_cache import FakeRedisCache, fake_redis_cache

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


@pytest.fixture
def prisma_client():
    from litellm.proxy.proxy_cli import append_query_params

    ### add connection pool + pool timeout args
    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    # Assuming PrismaClient is a class that needs to be instantiated
    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    # Reset litellm.proxy.proxy_server.prisma_client to None
    litellm.proxy.proxy_server.litellm_proxy_budget_name = (
        f"litellm-proxy-budget-{time.time()}"
    )
    litellm.proxy.proxy_server.user_custom_key_generate = None

    # Enable premium_user for project management tests
    setattr(litellm.proxy.proxy_server, "premium_user", True)

    return prisma_client


@pytest.mark.skip(reason="Requires reliable external DB connection (prisma).")
@pytest.mark.asyncio
async def test_new_project(prisma_client):
    """
    Test creating a new project with budget, models, and metadata.
    """
    try:
        print("prisma client=", prisma_client)

        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)

        await litellm.proxy.proxy_server.prisma_client.connect()

        # Create a team first
        _team_id = f"project-test-team_{uuid.uuid4()}"
        await new_team(
            NewTeamRequest(
                team_id=_team_id,
            ),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="1234",
            ),
        )

        # Create a project
        project_data = NewProjectRequest(
            project_alias="test-project",
            description="Test project for unit testing",
            team_id=_team_id,
            metadata={"use_case_id": "TEST-001", "responsible_ai_id": "RAI-001"},
            models=["gpt-5.5", "gpt-5-mini"],
            max_budget=100.0,
            model_rpm_limit={"gpt-5.5": 100},
            model_tpm_limit={"gpt-5.5": 1000},
        )

        response = await new_project(
            data=project_data,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="1234",
            ),
        )

        print("New project response:", response)

        # Assertions
        assert response.project_id is not None
        assert response.project_alias == "test-project"
        assert response.description == "Test project for unit testing"
        assert response.team_id == _team_id
        assert response.models == ["gpt-5.5", "gpt-5-mini"]
        # model_rpm_limit and model_tpm_limit are stored in metadata
        assert response.metadata["use_case_id"] == "TEST-001"
        assert response.metadata["responsible_ai_id"] == "RAI-001"
        assert response.metadata["model_rpm_limit"] == {"gpt-5.5": 100}
        assert response.metadata["model_tpm_limit"] == {"gpt-5.5": 1000}
        assert response.litellm_budget_table is not None
        assert response.litellm_budget_table.max_budget == 100.0

    except Exception as e:
        print("Got Exception", e)
        traceback.print_exc()
        pytest.fail(f"Got exception {e}")


@pytest.mark.skip(reason="Requires reliable external DB connection (prisma).")
@pytest.mark.asyncio
async def test_update_project(prisma_client):
    """
    Test updating an existing project's budget, models, and metadata.
    """
    try:
        print("prisma client=", prisma_client)

        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)

        await litellm.proxy.proxy_server.prisma_client.connect()

        # Create a team first
        _team_id = f"project-test-team_{uuid.uuid4()}"
        await new_team(
            NewTeamRequest(
                team_id=_team_id,
            ),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="1234",
            ),
        )

        # Create a project
        project_data = NewProjectRequest(
            project_alias="test-project-update",
            description="Original description",
            team_id=_team_id,
            metadata={
                "use_case_id": "TEST-002",
            },
            models=["gpt-5.5"],
            max_budget=50.0,
        )

        create_response = await new_project(
            data=project_data,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="1234",
            ),
        )

        print("Created project:", create_response)
        project_id = create_response.project_id

        # Update the project
        update_data = UpdateProjectRequest(
            project_id=project_id,
            project_alias="test-project-updated",
            description="Updated description",
            metadata={
                "use_case_id": "TEST-002-UPDATED",
                "additional_field": "new_value",
            },
            models=["gpt-5.5", "gpt-5-mini", "claude-3"],
            max_budget=200.0,
            model_rpm_limit={"gpt-5.5": 200, "claude-3": 50},
            model_tpm_limit={"gpt-5.5": 2000, "claude-3": 500},
        )

        update_response = await update_project(
            data=update_data,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="1234",
            ),
        )

        print("Updated project response:", update_response)

        # Assertions
        assert update_response.project_id == project_id
        assert update_response.project_alias == "test-project-updated"
        assert update_response.description == "Updated description"
        assert update_response.models == ["gpt-5.5", "gpt-5-mini", "claude-3"]
        # model_rpm_limit and model_tpm_limit are stored in metadata
        assert update_response.metadata["use_case_id"] == "TEST-002-UPDATED"
        assert update_response.metadata["additional_field"] == "new_value"
        assert update_response.metadata["model_rpm_limit"] == {
            "gpt-5.5": 200,
            "claude-3": 50,
        }
        assert update_response.metadata["model_tpm_limit"] == {
            "gpt-5.5": 2000,
            "claude-3": 500,
        }
        assert update_response.litellm_budget_table is not None
        assert update_response.litellm_budget_table.max_budget == 200.0

    except Exception as e:
        print("Got Exception", e)
        traceback.print_exc()
        pytest.fail(f"Got exception {e}")


@pytest.mark.skip(reason="Requires reliable external DB connection (prisma).")
@pytest.mark.asyncio
async def test_delete_project(prisma_client):
    """
    Test deleting a project.
    """
    try:
        print("prisma client=", prisma_client)

        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)

        await litellm.proxy.proxy_server.prisma_client.connect()

        # Create a team first
        _team_id = f"project-test-team_{uuid.uuid4()}"
        await new_team(
            NewTeamRequest(
                team_id=_team_id,
            ),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="1234",
            ),
        )

        # Create a project
        project_data = NewProjectRequest(
            project_alias="test-project-delete",
            team_id=_team_id,
            models=["gpt-5.5"],
            max_budget=50.0,
        )

        create_response = await new_project(
            data=project_data,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="1234",
            ),
        )

        print("Created project:", create_response)
        project_id = create_response.project_id

        # Delete the project
        delete_data = DeleteProjectRequest(project_ids=[project_id])

        delete_response = await delete_project(
            data=delete_data,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="1234",
            ),
        )

        print("Delete project response:", delete_response)

        # Assertions - delete_project returns a list of deleted project objects
        assert isinstance(delete_response, list)
        assert len(delete_response) == 1
        assert delete_response[0].project_id == project_id

        # Try to get info on the deleted project - should fail or return None
        try:
            await project_info(
                project_id=project_id,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN,
                    api_key=MASTER_KEY,
                    user_id="1234",
                ),
            )
            pytest.fail("Expected to fail when fetching deleted project")
        except Exception as e:
            print("Expected error when fetching deleted project:", e)
            # This is expected behavior

    except Exception as e:
        print("Got Exception", e)
        traceback.print_exc()
        pytest.fail(f"Got exception {e}")


@pytest.mark.skip(reason="Requires reliable external DB connection (prisma).")
@pytest.mark.asyncio
async def test_project_info(prisma_client):
    """
    Test getting project info.
    """
    try:
        print("prisma client=", prisma_client)

        setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
        setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)

        await litellm.proxy.proxy_server.prisma_client.connect()

        # Create a team first
        _team_id = f"project-test-team_{uuid.uuid4()}"
        await new_team(
            NewTeamRequest(
                team_id=_team_id,
            ),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="1234",
            ),
        )

        # Create a project
        project_data = NewProjectRequest(
            project_alias="test-project-info",
            description="Test project info endpoint",
            team_id=_team_id,
            metadata={"use_case_id": "TEST-003", "cost_center": "engineering"},
            models=["gpt-5.5", "claude-3"],
            max_budget=150.0,
            model_rpm_limit={"gpt-5.5": 150},
            model_tpm_limit={"gpt-5.5": 1500},
        )

        create_response = await new_project(
            data=project_data,
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="1234",
            ),
        )

        print("Created project:", create_response)
        project_id = create_response.project_id

        # Get project info
        info_response = await project_info(
            project_id=project_id,
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="1234",
            ),
        )

        print("Project info response:", info_response)

        # Assertions - project_info returns the project object directly
        assert info_response.project_id == project_id
        assert info_response.project_alias == "test-project-info"
        assert info_response.description == "Test project info endpoint"
        assert info_response.team_id == _team_id
        assert info_response.models == ["gpt-5.5", "claude-3"]
        # model_rpm_limit and model_tpm_limit are stored in metadata
        assert info_response.metadata["use_case_id"] == "TEST-003"
        assert info_response.metadata["cost_center"] == "engineering"
        assert info_response.metadata["model_rpm_limit"] == {"gpt-5.5": 150}
        assert info_response.metadata["model_tpm_limit"] == {"gpt-5.5": 1500}
        assert info_response.litellm_budget_table is not None
        assert info_response.litellm_budget_table.max_budget == 150.0

    except Exception as e:
        print("Got Exception", e)
        traceback.print_exc()
        pytest.fail(f"Got exception {e}")


### VALIDATION TESTS ###


def test_check_team_project_limits_models_not_in_team():
    """
    Test that creating a project with models not in the team raises an error.
    """
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-5.5", "gpt-5-mini"],
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-5.5", "claude-3"],  # claude-3 not in team
    )

    with pytest.raises(Exception, match="not in team's allowed models\\. Team allowed models") as exc_info:
        _check_team_project_limits(team_object=team, data=data)

    assert "claude-3" in str(exc_info.value.detail)
    assert "not in team's allowed models" in str(exc_info.value.detail)


def test_check_team_project_limits_budget_exceeds_team():
    """
    Test that creating a project with budget > team budget raises an error.
    """
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-5.5"],
        max_budget=100.0,
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-5.5"],
        max_budget=150.0,  # exceeds team's 100.0
    )

    with pytest.raises(Exception, match='Project max_budget') as exc_info:
        _check_team_project_limits(team_object=team, data=data)

    assert "exceeds team's max_budget" in str(exc_info.value.detail)


def test_check_team_project_limits_valid_subset():
    """
    Test that a valid project (models subset, budget within limit) passes.
    """
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-5.5", "gpt-5-mini", "claude-3"],
        max_budget=1000.0,
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-5.5", "gpt-5-mini"],
        max_budget=500.0,
    )

    # Should not raise
    _check_team_project_limits(team_object=team, data=data)


def test_check_team_project_limits_all_proxy_models():
    """
    Test that team with 'all-proxy-models' allows any project models.
    """
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["all-proxy-models"],
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-5.5", "claude-3", "anything-goes"],
    )

    # Should not raise - team allows all models
    _check_team_project_limits(team_object=team, data=data)


def test_check_team_project_limits_tpm_exceeds_team():
    """
    Test that project tpm_limit exceeding team tpm_limit raises an error.
    """
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-5.5"],
        tpm_limit=10000,
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-5.5"],
        tpm_limit=20000,  # exceeds team's 10000
    )

    with pytest.raises(Exception, match='Project tpm_limit') as exc_info:
        _check_team_project_limits(team_object=team, data=data)

    assert "exceeds team's tpm_limit" in str(exc_info.value.detail)


def test_check_team_project_limits_negative_budget():
    """
    Test that negative budget values raise an error.
    """
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-5.5"],
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-5.5"],
        max_budget=-10.0,
    )

    with pytest.raises(Exception, match='max_budget cannot be negative\\. Received') as exc_info:
        _check_team_project_limits(team_object=team, data=data)

    assert "cannot be negative" in str(exc_info.value.detail)


def test_check_team_project_limits_soft_budget_gte_max():
    """
    Test that soft_budget >= max_budget raises an error.
    """
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _check_team_project_limits,
    )
    from litellm.proxy._types import LiteLLM_TeamTable

    team = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-5.5"],
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-5.5"],
        max_budget=100.0,
        soft_budget=100.0,  # equal to max, should fail
    )

    with pytest.raises(Exception, match='must be strictly lower than max_budget') as exc_info:
        _check_team_project_limits(team_object=team, data=data)

    assert "must be strictly lower" in str(exc_info.value.detail)


def test_premium_user_gate():
    """
    Test that project endpoints require premium_user=True.
    """

    # This test just validates the premium_user check exists
    # The actual endpoint test would need prisma, but we can verify
    # the import path works
    setattr(litellm.proxy.proxy_server, "premium_user", False)

    # Verify that CommonProxyErrors.not_premium_user exists
    from litellm.proxy._types import CommonProxyErrors

    assert hasattr(CommonProxyErrors, "not_premium_user")

    # Reset
    setattr(litellm.proxy.proxy_server, "premium_user", True)


def test_project_model_access_denied_error_type():
    """
    Test that ProxyErrorTypes.project_model_access_denied exists.
    """
    from litellm.proxy._types import ProxyErrorTypes

    assert hasattr(ProxyErrorTypes, "project_model_access_denied")
    assert (
        ProxyErrorTypes.project_model_access_denied.value
        == "project_model_access_denied"
    )

    # Test the classmethod resolves correctly
    result = ProxyErrorTypes.get_model_access_error_type_for_object("project")
    assert result == ProxyErrorTypes.project_model_access_denied


def test_project_cached_obj_has_last_refreshed_at():
    """
    Test that LiteLLM_ProjectTableCachedObj has last_refreshed_at field
    matching LiteLLM_TeamTableCachedObj pattern.
    """
    from litellm.proxy._types import (
        LiteLLM_ProjectTableCachedObj,
        LiteLLM_ProjectTable,
    )

    # Verify inheritance
    assert issubclass(LiteLLM_ProjectTableCachedObj, LiteLLM_ProjectTable)

    # Verify last_refreshed_at field exists and defaults to None
    obj = LiteLLM_ProjectTableCachedObj(
        project_id="test",
        created_by="admin",
        updated_by="admin",
    )
    assert obj.last_refreshed_at is None

    # Verify it can be set
    obj.last_refreshed_at = 1234567890.0
    assert obj.last_refreshed_at == 1234567890.0


@pytest.mark.asyncio
async def test_project_max_budget_check_fires_alert():
    """
    Test that _project_max_budget_check fires a budget alert
    when project exceeds its max budget (matches _team_max_budget_check pattern).
    """
    from litellm.proxy.auth.auth_checks import _project_max_budget_check
    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_ProjectTableCachedObj,
    )

    project = LiteLLM_ProjectTableCachedObj(
        project_id="test-project",
        spend=150.0,
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=100.0),
    )

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="user-1",
        team_id="team-1",
    )

    mock_proxy_logging = mock.AsyncMock(spec=ProxyLogging)
    mock_proxy_logging.budget_alerts = mock.AsyncMock()

    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await _project_max_budget_check(
            project_object=project,
            valid_token=valid_token,
            proxy_logging_obj=mock_proxy_logging,
        )

    assert "Project=test-project" in str(exc_info.value)
    assert "150.0" in str(exc_info.value)


@pytest.mark.asyncio
async def test_project_soft_budget_check():
    """
    Test that _project_soft_budget_check triggers alert when soft budget is exceeded.
    """
    from litellm.proxy.auth.auth_checks import _project_soft_budget_check
    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_ProjectTableCachedObj,
    )

    project = LiteLLM_ProjectTableCachedObj(
        project_id="test-project",
        spend=80.0,
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=LiteLLM_BudgetTable(soft_budget=75.0),
    )

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="user-1",
        team_id="team-1",
    )

    mock_proxy_logging = mock.AsyncMock(spec=ProxyLogging)
    mock_proxy_logging.budget_alerts = mock.AsyncMock()

    # Should not raise (soft budget only alerts, doesn't block)
    await _project_soft_budget_check(
        project_object=project,
        valid_token=valid_token,
        proxy_logging_obj=mock_proxy_logging,
    )


@pytest.mark.asyncio
async def test_project_soft_budget_check_no_alert_under_budget():
    """
    Test that _project_soft_budget_check does NOT trigger alert when under soft budget.
    """
    from litellm.proxy.auth.auth_checks import _project_soft_budget_check
    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_ProjectTableCachedObj,
    )

    project = LiteLLM_ProjectTableCachedObj(
        project_id="test-project",
        spend=50.0,
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=LiteLLM_BudgetTable(soft_budget=75.0),
    )

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="user-1",
        team_id="team-1",
    )

    mock_proxy_logging = mock.AsyncMock(spec=ProxyLogging)
    mock_proxy_logging.budget_alerts = mock.AsyncMock()

    # Should not raise and should not alert
    await _project_soft_budget_check(
        project_object=project,
        valid_token=valid_token,
        proxy_logging_obj=mock_proxy_logging,
    )


def test_litellm_entity_type_has_project():
    """
    Test that Litellm_EntityType has PROJECT member for budget alerts.
    """
    from litellm.proxy._types import Litellm_EntityType

    assert hasattr(Litellm_EntityType, "PROJECT")
    assert Litellm_EntityType.PROJECT.value == "project"


@pytest.mark.asyncio
async def test_list_projects_returns_timestamps():
    """
    Test that /project/list returns created_at and updated_at for each project.
    """
    from datetime import datetime, timezone
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm_enterprise.proxy.management_endpoints.project_endpoints import list_projects
    from litellm.proxy._types import LiteLLM_ProjectTable

    now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    # Build a fake DB row that includes created_at and updated_at
    fake_project = MagicMock()
    fake_project.model_dump.return_value = {
        "project_id": "proj-1",
        "project_alias": "test-project",
        "team_id": "team-1",
        "created_by": "admin",
        "updated_by": "admin",
        "created_at": now,
        "updated_at": now,
        "models": [],
        "spend": 0.0,
        "blocked": False,
        "budget_id": None,
        "description": None,
        "metadata": None,
        "model_spend": None,
        "model_rpm_limit": None,
        "model_tpm_limit": None,
        "object_permission_id": None,
        "litellm_budget_table": None,
        "object_permission": None,
    }
    # Make the fake row behave like a Pydantic model for FastAPI serialization
    fake_project.project_id = "proj-1"
    fake_project.created_at = now
    fake_project.updated_at = now

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_projecttable.find_many = AsyncMock(
        return_value=[fake_project]
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        response = await list_projects(
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key=MASTER_KEY,
                user_id="1234",
            ),
        )

    assert len(response) == 1
    project = response[0]
    assert project.created_at == now
    assert project.updated_at == now


def test_litellm_project_table_has_timestamp_fields():
    """
    Test that LiteLLM_ProjectTable model includes created_at and updated_at fields,
    so the /project/list response_model exposes them.
    """
    from litellm.proxy._types import LiteLLM_ProjectTable

    fields = LiteLLM_ProjectTable.model_fields
    assert "created_at" in fields, "LiteLLM_ProjectTable must have created_at field"
    assert "updated_at" in fields, "LiteLLM_ProjectTable must have updated_at field"


@pytest.mark.asyncio
async def test_update_project_invalidates_cached_project_object(monkeypatch):
    """
    LIT-3803 regression: auth reads projects cache-first with no freshness check,
    so /project/update must evict the cached project. Before the fix, a project
    cached with models=[] kept bypassing the new allowlist until the TTL expired.
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.auth.auth_checks import get_project_object
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    project_id = f"project-{uuid.uuid4()}"
    cache = UserApiKeyCache()

    stale_row = MagicMock()
    stale_row.model_dump = lambda: {"project_id": project_id, "team_id": None, "models": []}

    mock_prisma = MagicMock()
    mock_prisma.jsonify_object = lambda data: data
    mock_prisma.db.litellm_projecttable.find_unique = AsyncMock(return_value=stale_row)

    seeded = await get_project_object(
        project_id=project_id, prisma_client=mock_prisma, user_api_key_cache=cache
    )
    assert seeded is not None and seeded.models == []

    updated_models = ["gemini-2.5-flash-image", "gemini-3.1-flash-lite-preview"]
    existing_row = MagicMock(team_id=None, budget_id=None, object_permission_id=None)
    updated_row = MagicMock()
    updated_row.model_dump = lambda: {
        "project_id": project_id,
        "team_id": None,
        "models": updated_models,
    }

    mock_prisma.db.litellm_projecttable.find_unique = AsyncMock(return_value=existing_row)
    mock_prisma.db.litellm_projecttable.update = AsyncMock(return_value=updated_row)

    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", mock_prisma)
    monkeypatch.setattr(litellm.proxy.proxy_server, "user_api_key_cache", cache)

    await update_project(
        data=UpdateProjectRequest(project_id=project_id, models=updated_models),
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key=MASTER_KEY,
            user_id="1234",
        ),
    )

    mock_prisma.db.litellm_projecttable.find_unique = AsyncMock(return_value=updated_row)
    refreshed = await get_project_object(
        project_id=project_id, prisma_client=mock_prisma, user_api_key_cache=cache
    )
    assert refreshed is not None
    assert refreshed.models == updated_models


@pytest.mark.asyncio
async def test_delete_project_invalidates_cached_project_object(monkeypatch):
    """
    LIT-3803 regression: /project/delete must evict the cached project so auth
    stops enforcing (or trusting) a project that no longer exists.
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.auth.auth_checks import get_project_object
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    project_id = f"project-{uuid.uuid4()}"
    cache = UserApiKeyCache()

    row = MagicMock()
    row.model_dump = lambda: {"project_id": project_id, "team_id": None, "models": ["gpt-5.5"]}

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_projecttable.find_unique = AsyncMock(return_value=row)

    seeded = await get_project_object(
        project_id=project_id, prisma_client=mock_prisma, user_api_key_cache=cache
    )
    assert seeded is not None

    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_projecttable.delete = AsyncMock(return_value=row)

    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", mock_prisma)
    monkeypatch.setattr(litellm.proxy.proxy_server, "user_api_key_cache", cache)

    await delete_project(
        data=DeleteProjectRequest(project_ids=[project_id]),
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key=MASTER_KEY,
            user_id="1234",
        ),
    )

    mock_prisma.db.litellm_projecttable.find_unique = AsyncMock(return_value=None)
    assert (
        await get_project_object(
            project_id=project_id, prisma_client=mock_prisma, user_api_key_cache=cache
        )
        is None
    )


@pytest.mark.asyncio
async def test_update_project_succeeds_when_cache_eviction_fails(monkeypatch):
    """
    The DB write has already committed when eviction runs, so a cache backend
    error must not turn a successful update into a 500; the stale entry is
    bounded by the TTL instead.
    """
    from unittest.mock import AsyncMock, MagicMock

    project_id = f"project-{uuid.uuid4()}"
    failing_cache = MagicMock()
    failing_cache.async_delete_cache = AsyncMock(side_effect=ConnectionError("redis down"))

    existing_row = MagicMock(team_id=None, budget_id=None, object_permission_id=None)
    updated_row = MagicMock()

    mock_prisma = MagicMock()
    mock_prisma.jsonify_object = lambda data: data
    mock_prisma.db.litellm_projecttable.find_unique = AsyncMock(return_value=existing_row)
    mock_prisma.db.litellm_projecttable.update = AsyncMock(return_value=updated_row)

    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", mock_prisma)
    monkeypatch.setattr(litellm.proxy.proxy_server, "user_api_key_cache", failing_cache)

    response = await update_project(
        data=UpdateProjectRequest(project_id=project_id, models=["gpt-oss-120b"]),
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key=MASTER_KEY,
            user_id="1234",
        ),
    )

    assert response is updated_row
    failing_cache.async_delete_cache.assert_awaited_once()


class _ProjectRows:
    """One project row the update and delete endpoints write and auth reads. While ``stopped`` every read fails the
    way an unreachable Postgres does; ``pause_next_read`` holds the next read once it has the row. Behind ``redis``,
    each update and delete records in ``tombstone_stood`` whether the project's tombstone stood in Redis as it ran
    (never while Redis is lost)."""

    def __init__(
        self,
        project_id: str,
        *,
        budget_id: str | None = None,
        object_permission_id: str | None = None,
        redis: FakeRedisCache | None = None,
    ) -> None:
        self.project_id = project_id
        self.budget_id = budget_id
        self.object_permission_id = object_permission_id
        self.blocked = False
        self.deleted = False
        self.stopped = False
        self.read_held = asyncio.Event()
        self.tombstone_stood: Final[list[bool]] = []  # mutable-ok: one entry for each write behind redis
        self._redis = redis
        self._release: asyncio.Event | None = None

    def pause_next_read(self) -> asyncio.Event:
        self._release = asyncio.Event()
        return self._release

    def _row(self) -> mock.MagicMock:
        row = mock.MagicMock(team_id=None, budget_id=self.budget_id, object_permission_id=self.object_permission_id)
        columns = {"project_id": self.project_id, "team_id": None, "models": [], "blocked": self.blocked}
        row.model_dump = lambda: columns
        return row

    async def find_unique(self, where: Mapping[str, object], include: object = None) -> mock.MagicMock:
        if self.stopped:
            raise httpx.ConnectError("All connection attempts failed")
        row = self._row()
        release = self._release
        if release is not None:
            self._release = None
            self.read_held.set()
            await release.wait()
        return row

    async def update(self, where: Mapping[str, object], data: Mapping[str, object], include: object = None):
        await self._record_tombstone()
        self.blocked = bool(data.get("blocked", self.blocked))
        return self._row()

    async def delete(self, where: Mapping[str, object]) -> mock.MagicMock:
        await self._record_tombstone()
        self.deleted = True
        return self._row()

    async def _record_tombstone(self) -> None:
        if self._redis is not None:
            self.tombstone_stood.append(await _tombstone_stands(self._redis, project_cache_key(self.project_id)))


def _serve_project_rows(monkeypatch, rows: _ProjectRows, cache: UserApiKeyCache) -> mock.MagicMock:
    prisma = mock.MagicMock()
    prisma.jsonify_object = lambda data: data
    prisma.db.litellm_projecttable = rows
    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(litellm.proxy.proxy_server, "user_api_key_cache", cache)
    return prisma


async def _block_project(project_id: str) -> None:
    await update_project(
        data=UpdateProjectRequest(project_id=project_id, blocked=True),
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key=MASTER_KEY, user_id="1234"),
    )


@pytest.mark.asyncio
async def test_project_block_answers_503_and_writes_nothing_when_redis_cannot_record_it(monkeypatch):
    redis = fake_redis_cache()
    redis.server.connected = False
    rows = _ProjectRows(f"project-{uuid.uuid4()}")
    _serve_project_rows(monkeypatch, rows, UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True))

    with pytest.raises(ProxyException) as refused:
        await _block_project(rows.project_id)

    assert (str(refused.value.code), rows.blocked) == ("503", False)


@pytest.mark.asyncio
async def test_project_delete_answers_503_and_deletes_nothing_when_redis_cannot_record_it(monkeypatch):
    """A deleted project's entry is revoked before the delete, so a Redis that cannot record the revocation refuses
    the delete and the project stays."""
    redis: Final = fake_redis_cache()
    redis.server.connected = False
    rows: Final = _ProjectRows(f"project-{uuid.uuid4()}")
    prisma: Final = _serve_project_rows(
        monkeypatch, rows, UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=True)
    )
    prisma.db.litellm_verificationtoken.find_many = mock.AsyncMock(return_value=[])

    with pytest.raises(ProxyException) as refused:
        await delete_project(
            data=DeleteProjectRequest(project_ids=[rows.project_id]),
            http_request=Request(scope={"type": "http"}),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN, api_key=MASTER_KEY, user_id="1234"
            ),
        )

    assert (str(refused.value.code), rows.deleted) == ("503", False)


@pytest.mark.asyncio
async def test_after_a_project_block_a_read_in_flight_cannot_bring_the_unblocked_project_back_in_an_outage(monkeypatch):
    from litellm.proxy.auth.auth_checks import get_project_object
    from litellm.proxy.db.exception_handler import LastGoodEntryMissing

    rows = _ProjectRows(f"project-{uuid.uuid4()}")
    cache = UserApiKeyCache(serve_last_good_entries=True)
    prisma = _serve_project_rows(monkeypatch, rows, cache)

    async def project():
        return await get_project_object(project_id=rows.project_id, prisma_client=prisma, user_api_key_cache=cache)

    release = rows.pause_next_read()
    in_flight = asyncio.create_task(project())
    await rows.read_held.wait()
    await _block_project(rows.project_id)
    release.set()
    assert (await in_flight).blocked is False
    await cache.async_delete_cache(f"project_id:{rows.project_id}")
    rows.stopped = True

    with pytest.raises(LastGoodEntryMissing):
        await project()


class _DrivenClock:
    """The last good entries' clock, which a test moves instead of waiting."""

    def __init__(self) -> None:
        self.now: float = 1000.0

    def __call__(self) -> float:
        return self.now


# Past this since a revocation, the request's record no longer skips Redis for the key: its tombstone may lapse first
_PAST_THE_RECORD: Final = LAST_GOOD_TOMBSTONE_TTL_SECONDS - LAST_GOOD_LATE_STORE_MARGIN_SECONDS + 1

_PROXY_ADMIN: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key=MASTER_KEY, user_id="1234")


def _outlasts_the_tombstones(
    redis: FakeRedisCache, clock: _DrivenClock, *, loses_redis: bool
) -> Callable[..., Awaitable[object]]:
    """A write's side effect: the write takes past the time the request's record of its revocations allows, the
    tombstones in Redis lapse meanwhile, and Redis is lost once it lands when ``loses_redis``. A mock's write then
    answers as configured."""

    async def outlast(*_args: object, **_kwargs: object) -> object:
        clock.now += _PAST_THE_RECORD
        client: Final = redis.cache.init_async_client()
        lapsed: Final = await client.keys("last_good_tombstone:*")
        if lapsed:
            await client.delete(*lapsed)
        if loses_redis:
            redis.server.connected = False
        return mock.DEFAULT

    return outlast


async def _tombstone_stands(redis: FakeRedisCache, cache_key: str) -> bool:
    """Whether ``cache_key`` has a tombstone in Redis now; never while Redis is lost."""
    tombstone: Final = f"last_good_tombstone:{cache_key}"
    return redis.server.connected and await redis.cache.init_async_client().exists(tombstone) == 1


def _driven_cache(redis: FakeRedisCache, clock: _DrivenClock, *, serves_last_good: bool = True) -> UserApiKeyCache:
    return UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=serves_last_good, last_good_clock=clock)


# The writes /project/update makes before the project row's own update
_EARLIER_PROJECT_WRITES: Final = ("budget update", "permission update", "permission create")


def _serve_a_project_block_after(
    monkeypatch: pytest.MonkeyPatch,
    earlier_write: str,
    redis: FakeRedisCache,
    clock: _DrivenClock,
    *,
    loses_redis: bool,
    serves_last_good: bool = True,
) -> tuple[_ProjectRows, mock.AsyncMock, dict[str, object]]:
    """A project whose block, a /project/update with ``blocked``, first makes ``earlier_write``, which outlasts the
    tombstones (``_outlasts_the_tombstones``). Returns the project, the earlier write's mock and the update's fields."""
    rows: Final = _ProjectRows(
        f"project-{uuid.uuid4()}",
        budget_id="budget-1" if earlier_write == "budget update" else None,
        object_permission_id="op-1" if earlier_write == "permission update" else None,
        redis=redis,
    )
    cache: Final = _driven_cache(redis, clock, serves_last_good=serves_last_good)
    prisma: Final = _serve_project_rows(monkeypatch, rows, cache)
    outlast: Final = _outlasts_the_tombstones(redis, clock, loses_redis=loses_redis)
    if earlier_write == "budget update":
        budget_write: Final = mock.AsyncMock(
            side_effect=outlast, return_value=LiteLLM_BudgetTable(budget_id="budget-1", max_budget=10.0)
        )
        prisma.db.litellm_budgettable.update = budget_write
        return rows, budget_write, {"blocked": True, "max_budget": 10.0}
    permission_write: Final = mock.AsyncMock(
        side_effect=outlast,
        return_value=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-1" if earlier_write == "permission update" else "op-new", vector_stores=["vs-1"]
        ),
    )
    if earlier_write == "permission update":
        prisma.db.litellm_objectpermissiontable.update = permission_write
    else:
        prisma.db.litellm_objectpermissiontable.create = permission_write
    permissions: Final = LiteLLM_ObjectPermissionBase(vector_stores=["vs-1"])
    return rows, permission_write, {"blocked": True, "object_permission": permissions}


@pytest.mark.asyncio
@pytest.mark.parametrize("earlier_write", _EARLIER_PROJECT_WRITES)
async def test_project_block_revokes_the_project_again_when_an_earlier_write_outlasts_the_tombstone(
    monkeypatch, earlier_write
):
    """The block revokes the project before its first write; that write, to the project's budget or permissions,
    takes past the time the request's record of the revocation allows and the tombstone lapses meanwhile, so another
    pod may store the unblocked project's entry again. The project row's update revokes the project again right
    before it: its tombstone stands in Redis when the project is blocked."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    rows, earlier, fields = _serve_a_project_block_after(monkeypatch, earlier_write, redis, clock, loses_redis=False)

    await _run_project_update(rows.project_id, **fields)

    earlier.assert_awaited_once()
    assert (rows.blocked, rows.tombstone_stood) == (True, [True])


@pytest.mark.asyncio
@pytest.mark.parametrize("earlier_write", _EARLIER_PROJECT_WRITES)
async def test_project_block_answers_503_with_the_project_unblocked_when_redis_is_lost_after_an_earlier_write(
    monkeypatch, earlier_write
):
    """The same slow earlier write, with Redis lost once it lands: the project's revocation has to reach Redis again
    and cannot, so the block answers 503 with the project row not updated."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    rows, earlier, fields = _serve_a_project_block_after(monkeypatch, earlier_write, redis, clock, loses_redis=True)

    with pytest.raises(ProxyException) as refused:
        await _run_project_update(rows.project_id, **fields)

    earlier.assert_awaited_once()
    assert (str(refused.value.code), rows.blocked, rows.tombstone_stood) == ("503", False, [])


@pytest.mark.asyncio
@pytest.mark.parametrize("earlier_write", _EARLIER_PROJECT_WRITES)
async def test_project_block_with_the_setting_off_blocks_the_project_after_a_slow_earlier_write_with_redis_lost(
    monkeypatch, earlier_write
):
    """With last good entries not served the revocations return at once, so the block makes its writes and blocks
    the project as it always did, whatever Redis does meanwhile."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    rows, earlier, fields = _serve_a_project_block_after(
        monkeypatch, earlier_write, redis, clock, loses_redis=True, serves_last_good=False
    )

    await _run_project_update(rows.project_id, **fields)

    earlier.assert_awaited_once()
    assert (rows.blocked, rows.tombstone_stood) == (True, [False])


class _ProjectTable:
    """Projects by id, as /project/delete reads and deletes them. ``calls`` records each read of a project, of its
    keys (one key for each project in ``keyed``), and each delete, in order; each delete records in ``deletes``
    whether its project's tombstone stood in ``redis`` as it ran (never while Redis is lost). The first delete then
    runs ``after_first_delete`` when given."""

    def __init__(
        self,
        project_ids: Sequence[str],
        redis: FakeRedisCache,
        *,
        keyed: Sequence[str] = (),
        after_first_delete: Callable[..., Awaitable[object]] | None = None,
    ) -> None:
        # mutable-ok: each delete removes its row
        self.rows: Final[dict[str, LiteLLM_ProjectTable]] = {
            project_id: LiteLLM_ProjectTable(project_id=project_id) for project_id in project_ids
        }
        self.calls: Final[list[tuple[str, str]]] = []  # mutable-ok: one entry for each read and delete
        self.deletes: Final[list[tuple[str, bool]]] = []  # mutable-ok: one entry for each delete
        self._keyed: Final = frozenset(keyed)
        self._redis: Final = redis
        self._after_first_delete: Final = after_first_delete

    async def find_unique(self, where: Mapping[str, object], include: object = None) -> LiteLLM_ProjectTable | None:
        project_id: Final = str(where["project_id"])
        self.calls.append(("read", project_id))
        return self.rows.get(project_id)

    async def find_keys(self, where: Mapping[str, object]) -> list[LiteLLM_VerificationToken]:
        project_id: Final = str(where["project_id"])
        self.calls.append(("keys", project_id))
        return (
            [LiteLLM_VerificationToken(token=f"hashed-key-of-{project_id}", project_id=project_id)]
            if project_id in self._keyed
            else []
        )

    async def delete(self, where: Mapping[str, object]) -> LiteLLM_ProjectTable:
        project_id: Final = str(where["project_id"])
        self.calls.append(("delete", project_id))
        self.deletes.append((project_id, await _tombstone_stands(self._redis, project_cache_key(project_id))))
        deleted: Final = self.rows.pop(project_id)
        if len(self.deletes) == 1 and self._after_first_delete is not None:
            await self._after_first_delete()
        return deleted


def _serve_project_table(monkeypatch: pytest.MonkeyPatch, projects: _ProjectTable, cache: UserApiKeyCache) -> None:
    prisma: Final = mock.MagicMock()
    prisma.db.litellm_projecttable = projects
    prisma.db.litellm_verificationtoken.find_many = mock.AsyncMock(side_effect=projects.find_keys)
    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(litellm.proxy.proxy_server, "user_api_key_cache", cache)


async def _delete_projects(*project_ids: str) -> list[LiteLLM_ProjectTable]:
    return await delete_project(
        data=DeleteProjectRequest(project_ids=list(project_ids)),
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=_PROXY_ADMIN,
    )


@pytest.mark.asyncio
async def test_project_delete_revokes_a_later_project_again_when_an_earlier_delete_outlasts_the_tombstones(monkeypatch):
    """The delete revokes both projects before its first read; the first project's delete takes past the time the
    request's record of that revocation allows and the tombstones lapse meanwhile, so another pod may store the
    second project's entry again. The second project's delete revokes it again right before it: its tombstone stands
    in Redis when that project is deleted."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    first, second = f"project-{uuid.uuid4()}", f"project-{uuid.uuid4()}"
    projects: Final = _ProjectTable(
        (first, second), redis, after_first_delete=_outlasts_the_tombstones(redis, clock, loses_redis=False)
    )
    _serve_project_table(monkeypatch, projects, _driven_cache(redis, clock))

    deleted: Final = await _delete_projects(first, second)

    assert [project.project_id for project in deleted] == [first, second]
    assert projects.deletes == [(first, True), (second, True)]


@pytest.mark.asyncio
async def test_project_delete_answers_503_with_the_later_project_kept_when_redis_is_lost_after_an_earlier_delete(
    monkeypatch,
):
    """The same slow first delete, with Redis lost once it lands: the second project's revocation has to reach Redis
    again and cannot, so the request answers 503 with the second project kept."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    first, second = f"project-{uuid.uuid4()}", f"project-{uuid.uuid4()}"
    projects: Final = _ProjectTable(
        (first, second), redis, after_first_delete=_outlasts_the_tombstones(redis, clock, loses_redis=True)
    )
    _serve_project_table(monkeypatch, projects, _driven_cache(redis, clock))

    with pytest.raises(ProxyException) as refused:
        await _delete_projects(first, second)

    assert str(refused.value.code) == "503"
    assert projects.deletes == [(first, True)]
    assert tuple(projects.rows) == (second,)


@pytest.mark.asyncio
async def test_project_delete_with_the_setting_off_reads_and_deletes_each_project_in_turn(monkeypatch):
    """With last good entries not served, the delete checks and deletes each project before it reads the next, as it
    always did: a later project that still holds a key is refused with 400 after the earlier one is deleted."""
    clock: Final = _DrivenClock()
    redis: Final = fake_redis_cache()
    first, second = f"project-{uuid.uuid4()}", f"project-{uuid.uuid4()}"
    projects: Final = _ProjectTable((first, second), redis, keyed=(second,))
    _serve_project_table(monkeypatch, projects, _driven_cache(redis, clock, serves_last_good=False))

    with pytest.raises(ProxyException) as refused:
        await _delete_projects(first, second)

    assert str(refused.value.code) == "400"
    assert projects.calls == [
        ("read", first),
        ("keys", first),
        ("delete", first),
        ("read", second),
        ("keys", second),
    ]
    assert tuple(projects.rows) == (second,)


@pytest.mark.asyncio
async def test_project_delete_records_the_tombstone_again_after_the_delete_so_an_outage_cannot_serve_the_old_project(
    monkeypatch,
):
    """With a 30 s tombstone, the request revokes the project's entry at t=0 and deletes the project at t=19, when its
    record shows that tombstone standing for 11 s more, so the delete sends no Redis command before it. Another pod's
    read began at t=19, before the delete committed, and got the old project; it stores it at t=31, after the first
    tombstone lapsed at t=30. Once the delete returns, the request records the project's tombstone again for 30 s
    from t=19, so that store is refused and an outage that follows serves no project."""
    from litellm.proxy._types import LiteLLM_ProjectTableCachedObj
    from tests.unit.proxy.last_good_tombstone_race import (
        revoked_before_the_write,
        served_after_another_pod_stores,
        tombstone_race,
    )

    race: Final = tombstone_race(monkeypatch)
    project_id: Final = f"project-{uuid.uuid4()}"
    cache_key: Final = project_cache_key(project_id)
    projects: Final = _ProjectTable((project_id,), race.redis)
    _serve_project_table(monkeypatch, projects, race.cache)
    await revoked_before_the_write(race, (cache_key,))

    deleted: Final = await _delete_projects(project_id)

    assert [project.project_id for project in deleted] == [project_id]
    assert projects.deletes == [(project_id, True)]
    served: Final = await served_after_another_pod_stores(
        race, cache_key, LiteLLM_ProjectTableCachedObj(project_id=project_id)
    )
    assert served is None


@pytest.mark.asyncio
async def test_project_eviction_publishes_cross_worker_invalidation(monkeypatch):
    """
    Single-worker eviction only fixes the handling worker; the broadcast is what
    lets every other worker drop its in-memory copy instead of serving the stale
    project until the TTL expires.
    """
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import delete_cached_project_object
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    project_id = f"project-{uuid.uuid4()}"
    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.publish_auth_cache_invalidation",
        new=AsyncMock(),
    ) as mock_publish:
        await delete_cached_project_object(
            project_id=project_id, user_api_key_cache=UserApiKeyCache()
        )

    mock_publish.assert_awaited_once_with(cache_key=f"project_id:{project_id}")


def test_enforce_project_model_quota_missing_both_raises():
    """A model added to a project without rpm/tpm is rejected."""
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota,
    )

    data = NewProjectRequest(team_id="test-team", models=["gpt-5.5"])
    with pytest.raises(HTTPException) as exc_info:
        _raise_on_missing_project_model_quota(data)
    assert "gpt-5.5" in str(exc_info.value.detail)
    assert "rpm/tpm quota" in str(exc_info.value.detail)


def test_enforce_project_model_quota_missing_tpm_raises():
    """A model with rpm but no tpm is rejected."""
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota,
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-5.5"],
        model_rpm_limit={"gpt-5.5": 100},
    )
    with pytest.raises(HTTPException):
        _raise_on_missing_project_model_quota(data)


def test_enforce_project_model_quota_all_present_passes():
    """A model with both rpm and tpm set passes."""
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota,
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-5.5"],
        model_rpm_limit={"gpt-5.5": 100},
        model_tpm_limit={"gpt-5.5": 1000},
    )
    assert _raise_on_missing_project_model_quota(data) is None


def test_enforce_project_model_quota_no_models_passes():
    """A project with no models has nothing to enforce."""
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota,
    )

    data = NewProjectRequest(team_id="test-team")
    assert _raise_on_missing_project_model_quota(data) is None


def test_enforce_project_model_quota_zero_rejected():
    """A zero quota is non-positive -> rejected (downstream treats it as exhausted)."""
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota,
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-5.5"],
        model_rpm_limit={"gpt-5.5": 0},
        model_tpm_limit={"gpt-5.5": 1000},
    )
    with pytest.raises(HTTPException):
        _raise_on_missing_project_model_quota(data)


def test_enforce_project_model_quota_negative_rejected():
    """A negative quota is non-positive -> rejected."""
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota,
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["gpt-5.5"],
        model_rpm_limit={"gpt-5.5": 100},
        model_tpm_limit={"gpt-5.5": -1},
    )
    with pytest.raises(HTTPException):
        _raise_on_missing_project_model_quota(data)


def test_update_quota_adds_model_without_quota_rejected():
    """Adding a model via /project/update without quota is rejected (the bypass)."""
    import types

    from litellm.proxy._types import UpdateProjectRequest
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota_on_update,
    )

    existing = types.SimpleNamespace(models=[], metadata={})
    data = UpdateProjectRequest(project_id="p", models=["gpt-5.5"])  # adds model, no quota
    with pytest.raises(HTTPException):
        _raise_on_missing_project_model_quota_on_update(data, existing)


def test_update_quota_adds_model_with_quota_passes():
    """Adding a model with a positive quota via update passes."""
    import types

    from litellm.proxy._types import UpdateProjectRequest
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota_on_update,
    )

    existing = types.SimpleNamespace(models=[], metadata={})
    data = UpdateProjectRequest(
        project_id="p",
        models=["gpt-5.5"],
        model_rpm_limit={"gpt-5.5": 100},
        model_tpm_limit={"gpt-5.5": 1000},
    )
    assert _raise_on_missing_project_model_quota_on_update(data, existing) is None


def test_update_quota_partial_update_keeps_existing_valid_passes():
    """A partial update that doesn't touch models/quota keeps existing valid quota -> passes."""
    import types

    from litellm.proxy._types import UpdateProjectRequest
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota_on_update,
    )

    existing = types.SimpleNamespace(
        models=["gpt-5.5"],
        metadata={"model_rpm_limit": {"gpt-5.5": 100}, "model_tpm_limit": {"gpt-5.5": 1000}},
    )
    data = UpdateProjectRequest(project_id="p", description="unrelated change")
    assert _raise_on_missing_project_model_quota_on_update(data, existing) is None


def test_update_quota_existing_quotaless_model_rejected():
    """A project already holding a quota-less model is rejected on any update (fail-closed)."""
    import types

    from litellm.proxy._types import UpdateProjectRequest
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota_on_update,
    )

    existing = types.SimpleNamespace(models=["gpt-5.5"], metadata={})
    data = UpdateProjectRequest(project_id="p", description="unrelated change")
    with pytest.raises(HTTPException):
        _raise_on_missing_project_model_quota_on_update(data, existing)


def _enforced_new_project_mocks(monkeypatch, team_models: list[str], llm_router: mock.MagicMock | None) -> None:
    from litellm.proxy._types import LiteLLM_TeamTable
    from litellm_enterprise.proxy.management_endpoints import project_endpoints as pe

    team = LiteLLM_TeamTable(team_id="test-team", models=team_models)
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", mock.MagicMock())
    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    monkeypatch.setattr(litellm.proxy.proxy_server, "llm_router", llm_router)
    monkeypatch.setattr(litellm.proxy.proxy_server, "general_settings", {"enforce_project_model_quota": True})
    monkeypatch.setattr(pe, "_validate_team_exists", mock.AsyncMock(return_value=team))
    monkeypatch.setattr(pe, "_check_user_permission_for_project", mock.AsyncMock(return_value=True))


async def _run_new_project(data: NewProjectRequest) -> None:
    await new_project(
        data=data,
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key=MASTER_KEY, user_id="1234"),
    )


@pytest.mark.asyncio
async def test_new_project_flag_on_missing_rpm_tpm_returns_400(monkeypatch):
    """End-to-end: with the flag on, POST /project/new rejects a model added without rpm/tpm."""
    _enforced_new_project_mocks(monkeypatch, team_models=["gpt-5.5"], llm_router=None)

    with pytest.raises(ProxyException, match="rpm/tpm quota") as exc_info:
        await _run_new_project(NewProjectRequest(team_id="test-team", models=["gpt-5.5"]))

    # new_project re-wraps the HTTPException, so assert on the string form.
    assert "rpm/tpm quota" in str(exc_info.value)


def _project_update_mocks(monkeypatch, stored_metadata: dict) -> mock.MagicMock:
    existing_row = mock.MagicMock(
        team_id=None, budget_id=None, object_permission_id=None, metadata=stored_metadata
    )
    mock_prisma = mock.MagicMock()
    mock_prisma.jsonify_object = lambda data: data
    mock_prisma.db.litellm_projecttable.find_unique = mock.AsyncMock(return_value=existing_row)
    mock_prisma.db.litellm_projecttable.update = mock.AsyncMock(return_value=mock.MagicMock())

    monkeypatch.setattr(litellm.proxy.proxy_server, "premium_user", True)
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", mock_prisma)
    monkeypatch.setattr(litellm.proxy.proxy_server, "user_api_key_cache", UserApiKeyCache())
    return mock_prisma


async def _run_project_update(project_id: str, **fields) -> None:
    await update_project(
        data=UpdateProjectRequest(project_id=project_id, **fields),
        http_request=Request(scope={"type": "http"}),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key=MASTER_KEY,
            user_id="1234",
        ),
    )


def _written_project_data(mock_prisma: mock.MagicMock) -> dict:
    return mock_prisma.db.litellm_projecttable.update.await_args.kwargs["data"]


@pytest.mark.asyncio
async def test_update_project_clears_model_itpm_limit_sent_as_an_empty_map(monkeypatch):
    """
    LIT-4693 regression: an omitted key means "leave this alone", so the only way to drop a
    per-model input/output TPM quota is to send it as an explicitly empty map. The written
    metadata must stop carrying the quota, otherwise the proxy keeps enforcing a limit the
    operator has already removed in the UI.
    """
    project_id = f"project-{uuid.uuid4()}"
    mock_prisma = _project_update_mocks(
        monkeypatch,
        {"owner": "platform", "model_itpm_limit": {"gpt-4": 60}, "model_otpm_limit": {"gpt-4": 40}},
    )

    await _run_project_update(project_id, model_itpm_limit={}, model_otpm_limit={})

    written_metadata = _written_project_data(mock_prisma)["metadata"]
    assert written_metadata["model_itpm_limit"] == {}
    assert written_metadata["model_otpm_limit"] == {}


@pytest.mark.asyncio
async def test_update_project_leaves_metadata_untouched_when_no_limit_is_sent(monkeypatch):
    """
    The other half of the same contract: an update that says nothing about the limits must not
    write metadata at all. That is what makes a dropped key silently preserve the old quota, so
    the UI has to send the empty map instead of omitting it.
    """
    project_id = f"project-{uuid.uuid4()}"
    mock_prisma = _project_update_mocks(monkeypatch, {"model_itpm_limit": {"gpt-4": 60}})

    await _run_project_update(project_id, description="renamed only")

    assert "metadata" not in _written_project_data(mock_prisma)


@pytest.mark.asyncio
async def test_update_project_clears_only_the_explicit_budget_cap(monkeypatch):
    mock_prisma = _project_update_mocks(monkeypatch, {})
    mock_prisma.db.litellm_projecttable.find_unique.return_value.budget_id = "budget-clear-test"
    mock_prisma.db.litellm_budgettable.update = mock.AsyncMock()

    await _run_project_update("project-clear-test", max_budget=None)

    mock_prisma.db.litellm_budgettable.update.assert_awaited_once_with(
        where={"budget_id": "budget-clear-test"},
        data={"max_budget": None, "updated_by": "1234"},
    )
    assert "max_budget" not in _written_project_data(mock_prisma)


@pytest.mark.parametrize("entry", ["all-proxy-models", "*", "azure/*"])
def test_enforce_project_model_quota_rejects_entries_that_expand_at_request_time(entry):
    """A quota keyed on a wildcard entry is never applied by the limiter, so it fails loudly."""
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota,
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=[entry],
        model_rpm_limit={entry: 10},
        model_tpm_limit={entry: 1000},
    )
    with pytest.raises(HTTPException) as exc_info:
        _raise_on_missing_project_model_quota(data)
    assert exc_info.value.status_code == 400
    assert entry in str(exc_info.value.detail)
    assert "expand to multiple models at request time" in str(exc_info.value.detail)


def test_enforce_project_model_quota_rejects_access_group_only_when_router_defines_it():
    """A plain model name passes; the same name is rejected once the router reports it as an access group."""
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota,
    )

    data = NewProjectRequest(
        team_id="test-team",
        models=["prod-models"],
        model_rpm_limit={"prod-models": 10},
        model_tpm_limit={"prod-models": 1000},
    )
    assert _raise_on_missing_project_model_quota(data, access_group_names=frozenset()) is None
    with pytest.raises(HTTPException) as exc_info:
        _raise_on_missing_project_model_quota(data, access_group_names=frozenset({"prod-models"}))
    assert "prod-models" in str(exc_info.value.detail)
    assert "expand to multiple models at request time" in str(exc_info.value.detail)


def test_update_quota_rejects_wildcard_left_on_project():
    """An update that leaves a wildcard entry on the project is rejected even when it carries a quota."""
    import types

    from litellm_enterprise.proxy.management_endpoints.project_endpoints import (
        _raise_on_missing_project_model_quota_on_update,
    )

    existing = types.SimpleNamespace(
        models=["all-proxy-models"],
        metadata={"model_rpm_limit": {"all-proxy-models": 10}, "model_tpm_limit": {"all-proxy-models": 1000}},
    )
    data = UpdateProjectRequest(project_id="p", description="unrelated change")
    with pytest.raises(HTTPException) as exc_info:
        _raise_on_missing_project_model_quota_on_update(data, existing)
    assert "all-proxy-models" in str(exc_info.value.detail)
    assert "expand to multiple models at request time" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_new_project_flag_on_access_group_model_returns_400(monkeypatch):
    """End-to-end: the router's access groups reach the check, so an access-group entry is rejected."""
    llm_router = mock.MagicMock()
    llm_router.get_model_access_groups.return_value = {"prod-models": ["gpt-5.5"]}
    _enforced_new_project_mocks(monkeypatch, team_models=["prod-models"], llm_router=llm_router)
    data = NewProjectRequest(
        team_id="test-team",
        models=["prod-models"],
        model_rpm_limit={"prod-models": 10},
        model_tpm_limit={"prod-models": 1000},
    )

    with pytest.raises(ProxyException, match="expand to multiple models at request time") as exc_info:
        await _run_new_project(data)

    assert "prod-models" in str(exc_info.value)
    assert "expand to multiple models at request time" in str(exc_info.value)


_ACCESS_NOW: Final = datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class _ProjectRow:
    project_id: str
    team_id: str | None


def _membership(user_id: str, org_id: str, role: LitellmUserRoles) -> LiteLLM_OrganizationMembershipTable:
    return LiteLLM_OrganizationMembershipTable(
        user_id=user_id, organization_id=org_id, user_role=role.value, created_at=_ACCESS_NOW, updated_at=_ACCESS_NOW
    )


def _user(user_id: str, teams: tuple[str, ...] = (), admin_of: tuple[str, ...] = ()) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(
        user_id=user_id,
        teams=list(teams),
        organization_memberships=[_membership(user_id, "org-a", LitellmUserRoles.INTERNAL_USER)]
        + [_membership(user_id, org_id, LitellmUserRoles.ORG_ADMIN) for org_id in admin_of],
    )


_TEAMS: Final = {
    team.team_id: team
    for team in (
        LiteLLM_TeamTable(
            team_id="team-a1",
            organization_id="org-a",
            members_with_roles=[Member(user_id="member", role="user"), Member(user_id="team-admin", role="admin")],
        ),
        LiteLLM_TeamTable(team_id="team-a2", organization_id="org-a"),
        LiteLLM_TeamTable(
            team_id="team-b1",
            organization_id="org-b",
            members_with_roles=[Member(user_id="org-admin-a-member-b1", role="user")],
        ),
        LiteLLM_TeamTable(team_id="team-orgless"),
    )
}
_PROJECTS: Final = (
    _ProjectRow("p-a1", "team-a1"),
    _ProjectRow("p-a2", "team-a2"),
    _ProjectRow("p-b1", "team-b1"),
    _ProjectRow("p-orgless", "team-orgless"),
    _ProjectRow("p-teamless", None),
    _ProjectRow("p-deleted-team", "team-deleted"),
)
_USERS: Final = {
    user.user_id: user
    for user in (
        _user("member", teams=("team-a1",)),
        _user("team-admin", teams=("team-a1",)),
        _user("org-admin-a", admin_of=("org-a",)),
        _user("org-admin-b", admin_of=("org-b",)),
        _user("org-admin-a-member-b1", teams=("team-b1",), admin_of=("org-a",)),
    )
}


def _row_matches(row: object, where: Mapping[str, object]) -> bool:
    return all(_condition_holds(row, key, condition) for key, condition in where.items())


def _condition_holds(row: object, key: str, condition) -> bool:
    if key == "OR":
        return any(_row_matches(row, branch) for branch in condition)
    if key == "litellm_team_table":
        team = _TEAMS.get(getattr(row, "team_id"))
        return team is not None and _row_matches(team, condition["is"])
    value = getattr(row, key)
    return value in condition["in"] if isinstance(condition, dict) else value == condition


class _FakeTeamTable:
    async def find_unique(self, where: Mapping[str, str], include: object = None) -> LiteLLM_TeamTable | None:
        return _TEAMS.get(where["team_id"])


class _FakeProjectTable:
    async def find_unique(self, where: Mapping[str, str], include: object = None) -> _ProjectRow | None:
        return next((p for p in _PROJECTS if p.project_id == where["project_id"]), None)

    async def find_many(self, where: Mapping[str, object] | None = None, include: object = None) -> list[_ProjectRow]:
        return [p for p in _PROJECTS if where is None or _row_matches(p, where)]


class _FakeUserTable:
    async def find_unique(
        self, where: Mapping[str, str], include: Mapping[str, bool] | None = None
    ) -> LiteLLM_UserTable | None:
        user = _USERS.get(where["user_id"])
        if user is None or (include or {}).get("organization_memberships"):
            return user
        return user.model_copy(update={"organization_memberships": None})


@pytest.fixture
def project_access_db(monkeypatch):
    db = mock.MagicMock(
        litellm_teamtable=_FakeTeamTable(), litellm_projecttable=_FakeProjectTable(), litellm_usertable=_FakeUserTable()
    )
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", mock.MagicMock(db=db))


async def _team_access_over_cached_users() -> TeamAccess:
    cache = UserApiKeyCache()
    for user in _USERS.values():
        await cache.async_set_cache(key=user.user_id, value=user)
    return TeamAccess(org_roles=PrismaOrgRoles(None, cache, proxy_logging_obj))


def _caller(user_id: str, role: LitellmUserRoles = LitellmUserRoles.INTERNAL_USER) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=role, api_key="sk-caller", user_id=user_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id", "expected"),
    [
        ("member", {"p-a1"}),
        ("team-admin", {"p-a1"}),
        ("org-admin-a", {"p-a1", "p-a2"}),
        ("org-admin-b", {"p-b1"}),
        ("org-admin-a-member-b1", {"p-a1", "p-a2", "p-b1"}),
        ("unknown-user", set()),
    ],
)
async def test_list_projects_scopes_to_teams_the_caller_can_view(project_access_db, user_id, expected):
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import list_projects

    projects = await list_projects(user_api_key_dict=_caller(user_id))

    assert {p.project_id for p in projects} == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
async def test_list_projects_admin_view_sees_every_project(project_access_db, role):
    from litellm_enterprise.proxy.management_endpoints.project_endpoints import list_projects

    projects = await list_projects(user_api_key_dict=_caller("someone", role))

    assert {p.project_id for p in projects} == {p.project_id for p in _PROJECTS}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id", "project_id"),
    [
        ("member", "p-a1"),
        ("team-admin", "p-a1"),
        ("org-admin-a", "p-a1"),
        ("org-admin-a", "p-a2"),
        ("org-admin-a-member-b1", "p-b1"),
    ],
)
async def test_project_info_allows_team_members_and_org_admins_of_the_team_org(project_access_db, user_id, project_id):
    project = await project_info(
        project_id=project_id,
        user_api_key_dict=_caller(user_id),
        team_access=await _team_access_over_cached_users(),
    )

    assert project.project_id == project_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id", "project_id"),
    [
        ("member", "p-a2"),
        ("team-admin", "p-b1"),
        ("org-admin-b", "p-a2"),
        ("org-admin-a", "p-b1"),
        ("org-admin-a", "p-orgless"),
        ("org-admin-a", "p-teamless"),
        ("member", "p-deleted-team"),
    ],
)
async def test_project_info_denies_callers_who_cannot_view_the_team(project_access_db, user_id, project_id):
    with pytest.raises(ProxyException) as exc_info:
        await project_info(
            project_id=project_id,
            user_api_key_dict=_caller(user_id),
            team_access=await _team_access_over_cached_users(),
        )

    assert str(exc_info.value.code) == "403"
    assert "You don't have access to this project" in exc_info.value.message


@pytest.mark.asyncio
async def test_project_info_missing_project_is_404_even_for_org_admins(project_access_db):
    with pytest.raises(ProxyException) as exc_info:
        await project_info(
            project_id="p-missing",
            user_api_key_dict=_caller("org-admin-a"),
            team_access=await _team_access_over_cached_users(),
        )

    assert str(exc_info.value.code) == "404"
