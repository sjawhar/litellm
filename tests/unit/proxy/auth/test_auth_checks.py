# What is this?
## Tests if 'get_end_user_object' works as expected

import sys, os, asyncio, time, random, uuid
import traceback
from dotenv import load_dotenv

load_dotenv()

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest, litellm
import httpx
from prisma import Prisma
from pydantic import BaseModel
from litellm._service_logger import ServiceTypes
from litellm.proxy._types import LiteLLM_OrganizationTable, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import get_org_object, get_user_object
from litellm.proxy.db.exception_handler import LastGoodEntryMissing
from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.auth.auth_checks import get_end_user_object
from litellm.caching.caching import DualCache
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy._types import (
    LiteLLM_EndUserTable,
    LiteLLM_BudgetTable,
    LiteLLM_UserTable,
    LiteLLM_TeamTable,
    Litellm_EntityType,
)
from litellm.proxy.utils import PrismaClient
from litellm.proxy.auth.auth_checks import (
    can_team_access_model,
    is_model_cost_zero,
    virtual_key_soft_budget_check,
    _team_soft_budget_check,
)
from litellm.proxy.utils import ProxyLogging
from litellm.proxy.utils import CallInfo
from tests.unit.caching.fake_redis_cache import fake_redis_cache
from tests.unit.proxy.db.cancelled_statements import CancelReason, cancelled_statement
from litellm.proxy.common_utils.user_api_key_cache import remember_last_good


@pytest.mark.parametrize("customer_spend, customer_budget", [(0, 10), (10, 0)])
@pytest.mark.asyncio
async def test_get_end_user_object(customer_spend, customer_budget):
    """
    Scenario 1: normal - get_end_user_object returns the cached user
    Scenario 2: user over budget - NOTE: budget enforcement now happens in 
                common_checks() via _check_end_user_budget(), not in get_end_user_object()
    
    This test verifies that get_end_user_object correctly retrieves the end user
    from cache. Budget enforcement is tested separately in test_check_end_user_budget().
    """
    end_user_id = "my-test-customer"
    _budget = LiteLLM_BudgetTable(max_budget=customer_budget)
    end_user_obj = LiteLLM_EndUserTable(
        user_id=end_user_id,
        spend=customer_spend,
        litellm_budget_table=_budget,
        blocked=False,
    )
    # UserApiKeyCache applies model_type on get/set; plain DualCache returns raw dicts
    # and breaks get_end_user_object's typed async_get_cache path.
    _cache = UserApiKeyCache()
    _key = "end_user_id:{}".format(end_user_id)
    await _cache.async_set_cache(
        key=_key,
        value=end_user_obj,
        model_type=LiteLLM_EndUserTable,
    )
    # get_end_user_object only fetches data - it no longer enforces budget
    # Budget enforcement happens in common_checks() via _check_end_user_budget()
    result = await get_end_user_object(
        end_user_id=end_user_id,
        prisma_client="RANDOM VALUE",  # type: ignore
        user_api_key_cache=_cache,
        route="/v1/chat/completions",
    )
    assert result is not None
    assert result.user_id == end_user_id


@pytest.mark.parametrize("customer_spend, customer_budget", [(0, 10), (10, 0)])
@pytest.mark.asyncio
async def test_check_end_user_budget(customer_spend, customer_budget):
    """
    Test _check_end_user_budget enforcement:
    - Scenario 1: customer_spend=0, customer_budget=10 - should pass (under budget)
    - Scenario 2: customer_spend=10, customer_budget=0 - should fail (over budget)
    
    Note: Budget enforcement for end users happens in common_checks() via 
    _check_end_user_budget(), not in get_end_user_object().
    """
    from litellm.proxy.auth.auth_checks import check_end_user_budget
    
    _budget = LiteLLM_BudgetTable(max_budget=customer_budget)
    end_user_obj = LiteLLM_EndUserTable(
        user_id="my-test-customer",
        spend=customer_spend,
        litellm_budget_table=_budget,
        blocked=False,
    )
    
    should_exceed = customer_spend > customer_budget
    
    if not should_exceed:
        await check_end_user_budget(
            end_user_obj=end_user_obj,
            route="/v1/chat/completions",
        )
        return

    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await check_end_user_budget(
            end_user_obj=end_user_obj,
            route="/v1/chat/completions",
        )
    # Verify the error has correct info
    assert exc_info.value.current_cost == customer_spend
    assert exc_info.value.max_budget == customer_budget


@pytest.mark.parametrize(
    "model, expect_to_work",
    [
        ("openai/gpt-4o-mini", True),
        ("openai/gpt-4o", False),
    ],
)
@pytest.mark.asyncio
async def test_can_key_call_model(model, expect_to_work):
    """
    If wildcard model + specific model is used, choose the specific model settings
    """
    from litellm.proxy.auth.auth_checks import can_key_call_model
    from fastapi import HTTPException

    llm_model_list = [
        {
            "model_name": "openai/*",
            "litellm_params": {
                "model": "openai/*",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "e6e7006f83029df40ebc02ddd068890253f4cd3092bcb203d3d8e6f6f606f30f",
                "db_model": False,
                "access_groups": ["public-openai-models"],
            },
        },
        {
            "model_name": "openai/gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "0cfcd87f2cb12a783a466888d05c6c89df66db23e01cecd75ec0b83aed73c9ad",
                "db_model": False,
                "access_groups": ["private-openai-models"],
            },
        },
    ]
    router = litellm.Router(model_list=llm_model_list)
    args = {
        "model": model,
        "llm_model_list": llm_model_list,
        "valid_token": UserAPIKeyAuth(
            models=["public-openai-models"],
        ),
        "llm_router": router,
    }
    if expect_to_work:
        await can_key_call_model(**args)
    else:
        with pytest.raises(Exception, match='is not available for this API key') as e:
            await can_key_call_model(**args)

        print(e)


@pytest.mark.parametrize(
    "model, expect_to_work",
    [("openai/gpt-4o", False), ("openai/gpt-4o-mini", True)],
)
@pytest.mark.asyncio
async def test_can_team_call_model(model, expect_to_work):
    from litellm.proxy.auth.auth_checks import model_in_access_group
    from fastapi import HTTPException

    llm_model_list = [
        {
            "model_name": "openai/*",
            "litellm_params": {
                "model": "openai/*",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "e6e7006f83029df40ebc02ddd068890253f4cd3092bcb203d3d8e6f6f606f30f",
                "db_model": False,
                "access_groups": ["public-openai-models"],
            },
        },
        {
            "model_name": "openai/gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "0cfcd87f2cb12a783a466888d05c6c89df66db23e01cecd75ec0b83aed73c9ad",
                "db_model": False,
                "access_groups": ["private-openai-models"],
            },
        },
    ]
    router = litellm.Router(model_list=llm_model_list)

    args = {
        "model": model,
        "team_models": ["public-openai-models"],
        "llm_router": router,
    }
    if expect_to_work:
        assert model_in_access_group(**args)
    else:
        assert not model_in_access_group(**args)


@pytest.mark.parametrize(
    "key_models, model, expect_to_work",
    [
        (["openai/*"], "openai/gpt-4o", True),
        (["openai/*"], "openai/gpt-4o-mini", True),
        (["openai/*"], "openaiz/gpt-4o-mini", False),
        (["bedrock/*"], "bedrock/anthropic.claude-3-5-sonnet-20240620", True),
        (["bedrock/*"], "bedrockz/anthropic.claude-3-5-sonnet-20240620", False),
        (["bedrock/us.*"], "bedrock/us.amazon.nova-micro-v1:0", True),
        (["openai/*"], "ft:gpt-4-0613", True),
        (["openai/*"], "bedrockz/ft:gpt-4-0613", False),
    ],
)
@pytest.mark.asyncio
async def test_can_key_call_model_wildcard_access(key_models, model, expect_to_work):
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.auth_checks import can_key_call_model

    llm_model_list = [
        {
            "model_name": "openai/*",
            "litellm_params": {
                "model": "openai/*",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "e6e7006f83029df40ebc02ddd068890253f4cd3092bcb203d3d8e6f6f606f30f",
                "db_model": False,
            },
        },
        {
            "model_name": "bedrock/*",
            "litellm_params": {
                "model": "bedrock/*",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "e6e7006f83029df40ebc02ddd068890253f4cd3092bcb203d3d8e6f6f606f30f",
                "db_model": False,
            },
        },
        {
            "model_name": "openai/gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "test-api-key",
            },
            "model_info": {
                "id": "0cfcd87f2cb12a783a466888d05c6c89df66db23e01cecd75ec0b83aed73c9ad",
                "db_model": False,
            },
        },
    ]
    router = litellm.Router(model_list=llm_model_list)

    user_api_key_object = UserAPIKeyAuth(
        models=key_models,
    )

    if expect_to_work:
        await can_key_call_model(
            model=model,
            llm_model_list=llm_model_list,
            valid_token=user_api_key_object,
            llm_router=router,
        )
    else:
        with pytest.raises(ProxyException):
            await can_key_call_model(
                model=model,
                llm_model_list=llm_model_list,
                valid_token=user_api_key_object,
                llm_router=router,
            )


@pytest.mark.parametrize(
    "key_models, model, expect_to_work",
    [
        # After a cost-map reload, add_known_models() updates anthropic_models so
        # the anthropic/* wildcard can match a newly-added Anthropic model.
        (["anthropic/*"], "claude-brand-new-model-reload-test", True),
        # Wrong provider wildcard must still be denied even after reload.
        (["openai/*"], "claude-brand-new-model-reload-test", False),
    ],
)
@pytest.mark.asyncio
async def test_wildcard_access_after_cost_map_reload(key_models, model, expect_to_work):
    """
    Regression test: after a cost-map hot-reload, calling
    add_known_models(model_cost_map=new_map) must update litellm.anthropic_models
    so that the anthropic/* wildcard correctly grants (or denies) access to
    newly-added models.

    Root cause: both reload paths in proxy_server.py only updated
    litellm.model_cost but never re-ran add_known_models(), so the provider sets
    stayed stale and wildcard matching failed for new models.

    Fix: each reload now calls litellm.add_known_models(model_cost_map=new_map)
    with the fetched map passed explicitly to avoid any reference ambiguity.
    """
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.auth_checks import can_key_call_model

    # Build a new cost map that includes the brand-new model — exactly what
    # proxy_server.py receives from get_model_cost_map() during a reload.
    new_cost_map = dict(litellm.model_cost)
    new_cost_map[model] = {
        "litellm_provider": "anthropic",
        "max_tokens": 8192,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    }

    original_model_cost = litellm.model_cost
    litellm.model_cost = new_cost_map

    # Confirm the model is NOT yet in the provider set before reload propagation.
    assert model not in litellm.anthropic_models

    # Simulate what proxy_server.py now does after every reload.
    litellm.add_known_models(model_cost_map=new_cost_map)

    # After add_known_models(), the model must be in the set.
    assert model in litellm.anthropic_models

    llm_model_list = [
        {
            "model_name": "anthropic/*",
            "litellm_params": {"model": "anthropic/*", "api_key": "test-api-key"},
            "model_info": {"id": "test-id-anthropic-wildcard", "db_model": False},
        },
        {
            "model_name": "openai/*",
            "litellm_params": {"model": "openai/*", "api_key": "test-api-key"},
            "model_info": {"id": "test-id-openai-wildcard", "db_model": False},
        },
    ]
    router = litellm.Router(model_list=llm_model_list)
    user_api_key_object = UserAPIKeyAuth(models=key_models)

    try:
        if expect_to_work:
            await can_key_call_model(
                model=model,
                llm_model_list=llm_model_list,
                valid_token=user_api_key_object,
                llm_router=router,
            )
        else:
            with pytest.raises(ProxyException):
                await can_key_call_model(
                    model=model,
                    llm_model_list=llm_model_list,
                    valid_token=user_api_key_object,
                    llm_router=router,
                )
    finally:
        litellm.model_cost = original_model_cost
        litellm.anthropic_models.discard(model)


@pytest.mark.asyncio
async def test_add_known_models_explicit_map_updates_provider_sets():
    """
    Regression test: after a cost-map hot-reload, calling
    add_known_models(model_cost_map=new_map) with the new map passed explicitly
    must add any new provider models to the correct provider sets so that
    wildcard access checks (anthropic/*, openai/*, …) work immediately.

    This covers the proxy_server.py fix where both reload paths now call
    litellm.add_known_models(model_cost_map=new_model_cost_map) instead of
    relying on the module-level model_cost being up to date.
    """
    fake_new_model = "claude-brand-new-explicit-map-test"

    # Baseline: the model must not be in the sets before we do anything.
    assert fake_new_model not in litellm.anthropic_models

    new_cost_map = dict(litellm.model_cost)
    new_cost_map[fake_new_model] = {
        "litellm_provider": "anthropic",
        "max_tokens": 8192,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    }

    # Simulate what proxy_server.py does on reload.
    original_model_cost = litellm.model_cost
    litellm.model_cost = new_cost_map
    litellm.add_known_models(model_cost_map=new_cost_map)

    try:
        assert fake_new_model in litellm.anthropic_models, (
            "add_known_models(model_cost_map=...) did not add the new model to "
            "litellm.anthropic_models — wildcard access checks would fail."
        )
    finally:
        # Clean up: restore original state.
        litellm.model_cost = original_model_cost
        litellm.anthropic_models.discard(fake_new_model)


@pytest.mark.asyncio
async def test_is_valid_fallback_model():
    from litellm.proxy.auth.auth_checks import is_valid_fallback_model
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "openai/gpt-3.5-turbo"},
            }
        ]
    )

    try:
        await is_valid_fallback_model(
            model="gpt-3.5-turbo", llm_router=router, user_model=None
        )
    except Exception as e:
        pytest.fail(f"Expected is_valid_fallback_model to work, got exception: {e}")

    with pytest.raises(Exception, match="Invalid") as exc_info:
        await is_valid_fallback_model(
            model="gpt-4o", llm_router=router, user_model=None
        )
    e = exc_info.value
    assert "Invalid" in str(e)


@pytest.mark.parametrize(
    "token_spend, max_budget, expect_budget_error",
    [
        (5.0, 10.0, False),  # Under budget
        (10.0, 10.0, True),  # At budget limit
        (15.0, 10.0, True),  # Over budget
    ],
)
@pytest.mark.asyncio
async def test_virtual_key_max_budget_check(
    token_spend, max_budget, expect_budget_error
):
    """
    Test if virtual key budget checks work as expected:
    1. Triggers budget alert for all cases
    2. Raises BudgetExceededError when spend >= max_budget
    """
    from litellm.proxy.auth.auth_checks import virtual_key_max_budget_check

    # Setup test data
    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=token_spend,
        max_budget=max_budget,
        user_id="test-user",
        key_alias="test-key",
    )

    user_obj = LiteLLM_UserTable(
        user_id="test-user",
        user_email="test@email.com",
        max_budget=None,
    )

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=None,
    )

    # Track if budget alert was called
    alert_called = False

    async def mock_budget_alert(*args, **kwargs):
        nonlocal alert_called
        alert_called = True

    proxy_logging_obj.budget_alerts = mock_budget_alert

    if expect_budget_error:
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await virtual_key_max_budget_check(
                valid_token=valid_token,
                proxy_logging_obj=proxy_logging_obj,
                user_obj=user_obj,
            )
        assert exc_info.value.current_cost == token_spend
        assert exc_info.value.max_budget == max_budget
    else:
        await virtual_key_max_budget_check(
            valid_token=valid_token,
            proxy_logging_obj=proxy_logging_obj,
            user_obj=user_obj,
        )

    await asyncio.sleep(1)

    # Verify budget alert was triggered
    assert alert_called, "Budget alert should be triggered"


@pytest.mark.parametrize(
    "model, team_models, expect_to_work",
    [
        ("gpt-4", ["gpt-4"], True),  # exact match
        ("gpt-4", ["all-proxy-models"], True),  # all-proxy-models access
        ("gpt-4", ["*"], True),  # wildcard access
        ("gpt-4", ["openai/*"], True),  # openai wildcard access
        (
            "bedrock/anthropic.claude-3-5-sonnet-20240620",
            ["bedrock/*"],
            True,
        ),  # wildcard access
        (
            "bedrockz/anthropic.claude-3-5-sonnet-20240620",
            ["bedrock/*"],
            False,
        ),  # non-match wildcard access
        ("bedrock/very_new_model", ["bedrock/*"], True),  # bedrock wildcard access
        (
            "bedrock/claude-3-5-sonnet-20240620",
            ["bedrock/claude-*"],
            True,
        ),  # match on pattern
        (
            "bedrock/claude-3-6-sonnet-20240620",
            ["bedrock/claude-3-5-*"],
            False,
        ),  # don't match on pattern
        ("openai/gpt-4o", ["openai/*"], True),  # openai wildcard access
        ("gpt-4", ["gpt-3.5-turbo"], False),  # model not in allowed list
        ("claude-3", [], True),  # empty model list (allows all)
    ],
)
@pytest.mark.asyncio
async def test_can_team_access_model(model, team_models, expect_to_work):
    """
    Test cases for can_team_access_model:
    1. Exact model match
    2. all-proxy-models access
    3. Wildcard (*) access
    4. OpenAI wildcard access
    5. Model not in allowed list
    6. Empty model list
    7. None model list
    """
    try:
        team_object = LiteLLM_TeamTable(
            team_id="test-team",
            models=team_models,
        )
        result = await can_team_access_model(
            model=model,
            team_object=team_object,
            llm_router=None,
            team_model_aliases=None,
        )
        if not expect_to_work:
            pytest.fail(
                f"Expected model access check to fail for model={model}, team_models={team_models}"
            )
    except Exception as e:
        if expect_to_work:
            pytest.fail(
                f"Expected model access check to work for model={model}, team_models={team_models}. Got error: {str(e)}"
            )


@pytest.mark.parametrize(
    "spend, soft_budget, expect_alert",
    [
        (100, 50, True),  # Over soft budget
        (50, 50, True),  # At soft budget
        (25, 50, False),  # Under soft budget
        (100, None, False),  # No soft budget set
    ],
)
@pytest.mark.asyncio
async def test_virtual_key_soft_budget_check(spend, soft_budget, expect_alert):
    """
    Test cases for _virtual_key_soft_budget_check:
    1. Spend over soft budget
    2. Spend at soft budget
    3. Spend under soft budget
    4. No soft budget set
    """
    alert_triggered = False

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered
            alert_triggered = True
            assert type == "soft_budget"
            assert isinstance(user_info, CallInfo)

    valid_token = UserAPIKeyAuth(
        token="test-token",
        spend=spend,
        soft_budget=soft_budget,
        user_id="test-user",
        team_id="test-team",
        key_alias="test-key",
    )

    proxy_logging_obj = MockProxyLogging()

    await virtual_key_soft_budget_check(
        valid_token=valid_token,
        proxy_logging_obj=proxy_logging_obj,
    )

    await asyncio.sleep(0.1)  # Allow time for the alert task to complete

    assert (
        alert_triggered == expect_alert
    ), f"Expected alert_triggered to be {expect_alert} for spend={spend}, soft_budget={soft_budget}"


@pytest.mark.parametrize(
    "spend, soft_budget, expect_alert, metadata, expected_alert_emails",
    [
        (
            100,
            50,
            False,
            None,
            None,
        ),  # Over soft budget, no metadata - no alert_emails configured, so no alert
        (
            50,
            50,
            False,
            None,
            None,
        ),  # At soft budget, no metadata - no alert_emails configured, so no alert
        (25, 50, False, None, None),  # Under soft budget
        (100, None, False, None, None),  # No soft budget set
        (
            100,
            50,
            True,
            {"soft_budget_alerting_emails": ["team1@example.com", "team2@example.com"]},
            ["team1@example.com", "team2@example.com"],
        ),  # Over soft budget with list of emails
        (
            100,
            50,
            True,
            {"soft_budget_alerting_emails": "team1@example.com,team2@example.com"},
            ["team1@example.com", "team2@example.com"],
        ),  # Over soft budget with comma-separated emails
        (
            100,
            50,
            True,
            {
                "soft_budget_alerting_emails": [
                    "team1@example.com",
                    "",
                    "  ",
                    "team2@example.com",
                ]
            },
            ["team1@example.com", "team2@example.com"],
        ),  # Over soft budget with empty strings filtered
    ],
)
@pytest.mark.asyncio
async def test_team_soft_budget_check(
    spend, soft_budget, expect_alert, metadata, expected_alert_emails
):
    """
    Test cases for _team_soft_budget_check:
    1. Spend over soft budget, no alert_emails configured - should NOT trigger alert (alerts only sent when alert_emails configured)
    2. Spend at soft budget, no alert_emails configured - should NOT trigger alert (alerts only sent when alert_emails configured)
    3. Spend under soft budget - should not trigger alert
    4. No soft budget set - should not trigger alert
    5. Team with alert emails in metadata (list) - should include alert_emails in CallInfo
    6. Team with alert emails in metadata (comma-separated string) - should parse and include alert_emails
    7. Team with alert emails containing empty strings - should filter them out
    """
    alert_triggered = False
    captured_call_info = None

    class MockProxyLogging:
        async def budget_alerts(self, type, user_info):
            nonlocal alert_triggered, captured_call_info
            alert_triggered = True
            captured_call_info = user_info
            assert type == "soft_budget"
            assert isinstance(user_info, CallInfo)

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
        team_alias="test-team-alias",
        key_alias="test-key",
    )

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        spend=spend,
        soft_budget=soft_budget,
        max_budget=100.0,
        metadata=metadata,
    )

    proxy_logging_obj = MockProxyLogging()

    await _team_soft_budget_check(
        team_object=team_object,
        valid_token=valid_token,
        proxy_logging_obj=proxy_logging_obj,
    )

    await asyncio.sleep(0.1)  # Allow time for the alert task to complete

    assert (
        alert_triggered == expect_alert
    ), f"Expected alert_triggered to be {expect_alert} for spend={spend}, soft_budget={soft_budget}"

    if expect_alert:
        assert captured_call_info is not None
        assert captured_call_info.team_id == "test-team"
        assert captured_call_info.spend == spend
        assert captured_call_info.soft_budget == soft_budget
        assert captured_call_info.event_group == Litellm_EntityType.TEAM
        # Verify alert_emails if expected
        if expected_alert_emails is not None:
            assert captured_call_info.alert_emails == expected_alert_emails
        else:
            assert (
                captured_call_info.alert_emails is None
                or captured_call_info.alert_emails == []
            )


@pytest.mark.asyncio
async def test_can_user_call_model():
    from litellm.proxy.auth.auth_checks import can_user_call_model
    from litellm.proxy._types import ProxyException
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "anthropic-claude",
                "litellm_params": {"model": "anthropic/anthropic-claude"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "test-api-key"},
            },
        ]
    )

    args = {
        "model": "anthropic-claude",
        "llm_router": router,
        "user_object": LiteLLM_UserTable(
            user_id="testuser21@mycompany.com",
            max_budget=None,
            spend=0.0042295,
            model_max_budget={},
            model_spend={},
            user_email="testuser@mycompany.com",
            models=["gpt-3.5-turbo"],
        ),
    }

    with pytest.raises(ProxyException) as e:
        await can_user_call_model(**args)

    args["model"] = "gpt-3.5-turbo"
    await can_user_call_model(**args)


@pytest.mark.asyncio
async def test_can_user_call_model_with_no_default_models():
    from litellm.proxy.auth.auth_checks import can_user_call_model
    from litellm.proxy._types import ProxyException, SpecialModelNames
    from unittest.mock import MagicMock

    args = {
        "model": "anthropic-claude",
        "llm_router": MagicMock(),
        "user_object": LiteLLM_UserTable(
            user_id="testuser21@mycompany.com",
            max_budget=None,
            spend=0.0042295,
            model_max_budget={},
            model_spend={},
            user_email="testuser@mycompany.com",
            models=[SpecialModelNames.no_default_models.value],
        ),
    }

    with pytest.raises(ProxyException) as e:
        await can_user_call_model(**args)


@pytest.mark.asyncio
async def test_get_fuzzy_user_object():
    from litellm.proxy.auth.auth_checks import _get_fuzzy_user_object
    from unittest.mock import AsyncMock, MagicMock

    # Setup mock Prisma client
    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_usertable = MagicMock()

    # Mock user data
    test_user = LiteLLM_UserTable(
        user_id="test_123",
        sso_user_id="sso_123",
        user_email="test@example.com",
        organization_memberships=[],
        max_budget=None,
    )

    # Test 1: Find user by SSO ID
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=test_user)
    result = await _get_fuzzy_user_object(
        prisma_client=mock_prisma, sso_user_id="sso_123", user_email="test@example.com"
    )
    assert result == test_user
    mock_prisma.db.litellm_usertable.find_unique.assert_called_with(
        where={"sso_user_id": "sso_123"}, include={"organization_memberships": True}
    )

    # Test 2: SSO ID not found, find by email
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=test_user)
    mock_prisma.db.litellm_usertable.update = AsyncMock()

    result = await _get_fuzzy_user_object(
        prisma_client=mock_prisma,
        sso_user_id="new_sso_456",
        user_email="test@example.com",
    )
    assert result == test_user
    mock_prisma.db.litellm_usertable.find_first.assert_called_with(
        where={"user_email": {"equals": "test@example.com", "mode": "insensitive"}},
        include={"organization_memberships": True},
    )

    # Test 3: Verify background SSO update task when user found by email
    await asyncio.sleep(0.1)  # Allow time for background task
    mock_prisma.db.litellm_usertable.update.assert_called_with(
        where={"user_id": "test_123"}, data={"sso_user_id": "new_sso_456"}
    )

    # Test 4: User not found by either method
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=None)

    result = await _get_fuzzy_user_object(
        prisma_client=mock_prisma,
        sso_user_id="unknown_sso",
        user_email="unknown@example.com",
    )
    assert result is None

    # Test 5: Only email provided (no SSO ID)
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=test_user)
    result = await _get_fuzzy_user_object(
        prisma_client=mock_prisma, user_email="test@example.com"
    )
    assert result == test_user
    mock_prisma.db.litellm_usertable.find_first.assert_called_with(
        where={"user_email": {"equals": "test@example.com", "mode": "insensitive"}},
        include={"organization_memberships": True},
    )

    # Test 6: Only SSO ID provided (no email)
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=test_user)
    result = await _get_fuzzy_user_object(
        prisma_client=mock_prisma, sso_user_id="sso_123"
    )
    assert result == test_user
    mock_prisma.db.litellm_usertable.find_unique.assert_called_with(
        where={"sso_user_id": "sso_123"}, include={"organization_memberships": True}
    )


@pytest.mark.parametrize(
    "model, alias_map, expect_to_work",
    [
        ("gpt-4", {"gpt-4": "gpt-4-team1"}, True),  # model matches alias value
        ("gpt-5", {"gpt-4": "gpt-4-team1"}, False),
    ],
)
@pytest.mark.asyncio
async def test_can_key_call_model_with_aliases(model, alias_map, expect_to_work):
    """
    Test if can_key_call_model correctly handles model aliases in the token
    """
    from litellm.proxy.auth.auth_checks import can_key_call_model

    llm_model_list = [
        {
            "model_name": "gpt-4-team1",
            "litellm_params": {
                "model": "gpt-4",
                "api_key": "test-api-key",
            },
        }
    ]
    router = litellm.Router(model_list=llm_model_list)

    user_api_key_object = UserAPIKeyAuth(
        models=[
            "gpt-4-team1",
        ],
        team_model_aliases=alias_map,
    )

    if expect_to_work:
        await can_key_call_model(
            model=model,
            llm_model_list=llm_model_list,
            valid_token=user_api_key_object,
            llm_router=router,
        )
    else:
        with pytest.raises(Exception, match='is not available for this API key') as e:
            await can_key_call_model(
                model=model,
                llm_model_list=llm_model_list,
                valid_token=user_api_key_object,
                llm_router=router,
            )


# ---------------------------------------------------------------------------
# Access group cache helpers (_cache_access_object, _delete_cache_access_object)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_access_object():
    """Test _cache_access_object stores access group in cache with correct key."""
    from litellm.proxy.auth.auth_checks import cache_access_object
    from litellm.proxy._types import LiteLLM_AccessGroupTable

    cache = DualCache()
    ag_id = "ag-test-123"
    ag_table = LiteLLM_AccessGroupTable(
        access_group_id=ag_id,
        access_group_name="test-group",
        access_model_names=["gpt-4"],
    )
    await cache_access_object(
        access_group_id=ag_id,
        access_group_table=ag_table,
        user_api_key_cache=cache,
    )
    cached = await cache.async_get_cache(key=f"access_group_id:{ag_id}")
    assert cached is not None
    if isinstance(cached, dict):
        assert cached.get("access_group_id") == ag_id
        assert cached.get("access_group_name") == "test-group"
    else:
        assert cached.access_group_id == ag_id
        assert cached.access_group_name == "test-group"


@pytest.mark.asyncio
async def test_delete_cache_access_object():
    """Test _delete_cache_access_object removes access group from in-memory cache."""
    from litellm.proxy.auth.auth_checks import delete_cache_access_object
    from litellm.proxy._types import LiteLLM_AccessGroupTable

    cache = DualCache()
    ag_id = "ag-delete-test"
    ag_table = LiteLLM_AccessGroupTable(
        access_group_id=ag_id,
        access_group_name="to-delete",
    )
    await cache.async_set_cache(key=f"access_group_id:{ag_id}", value=ag_table, ttl=60)
    await delete_cache_access_object(access_group_id=ag_id, user_api_key_cache=cache)
    cached = await cache.async_get_cache(key=f"access_group_id:{ag_id}")
    assert cached is None


# ---------------------------------------------------------------------------
# Access group resource fetchers (_get_models_from_access_groups, _get_agent_ids_from_access_groups)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resource_field, access_group_data, expected",
    [
        (
            "access_model_names",
            {"access_group_id": "ag-1", "access_model_names": ["gpt-4", "claude-3"]},
            ["gpt-4", "claude-3"],
        ),
        (
            "access_agent_ids",
            {"access_group_id": "ag-2", "access_agent_ids": ["agent-a", "agent-b"]},
            ["agent-a", "agent-b"],
        ),
        (
            "access_model_names",
            {"access_group_id": "ag-3", "access_model_names": []},
            [],
        ),
    ],
)
@pytest.mark.asyncio
async def test_get_resources_from_access_groups(
    resource_field, access_group_data, expected
):
    """Test _get_resources_from_access_groups returns correct resource list from access groups."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy._types import LiteLLM_AccessGroupTable
    from litellm.proxy.auth.auth_checks import (
        get_agent_ids_from_access_groups,
        get_models_from_access_groups,
    )

    ag_table = LiteLLM_AccessGroupTable(
        access_group_id=access_group_data["access_group_id"],
        access_group_name="test",
        access_model_names=access_group_data.get("access_model_names", []),
        access_agent_ids=access_group_data.get("access_agent_ids", []),
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_access_object",
        new_callable=AsyncMock,
        return_value=ag_table,
    ):
        if resource_field == "access_model_names":
            result = await get_models_from_access_groups(
                access_group_ids=[access_group_data["access_group_id"]],
                prisma_client=MagicMock(),
                user_api_key_cache=DualCache(),
            )
        else:
            result = await get_agent_ids_from_access_groups(
                access_group_ids=[access_group_data["access_group_id"]],
                prisma_client=MagicMock(),
                user_api_key_cache=DualCache(),
            )
        assert sorted(result) == sorted(expected)


@pytest.mark.asyncio
async def test_get_models_from_access_groups_empty_ids():
    """Test _get_models_from_access_groups returns empty list when access_group_ids is empty."""
    from litellm.proxy.auth.auth_checks import get_models_from_access_groups

    result = await get_models_from_access_groups(access_group_ids=[])
    assert result == []


# ---------------------------------------------------------------------------
# can_team_access_model with access_group_ids fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_can_team_access_model_via_access_group_ids():
    """Test can_team_access_model allows access when team has access_group_ids granting model access."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import can_team_access_model

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        models=[],
        access_group_ids=["ag-with-gpt4"],
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_models_from_access_groups",
        new_callable=AsyncMock,
        return_value=["gpt-4"],
    ):
        result = await can_team_access_model(
            model="gpt-4",
            team_object=team_object,
            llm_router=None,
            team_model_aliases=None,
        )
        assert result is True


@pytest.mark.asyncio
async def test_can_team_access_model_access_group_ids_denied():
    """Test can_team_access_model denies when neither team models nor access_group_ids grant access."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import can_team_access_model
    from litellm.proxy._types import ProxyException

    team_object = LiteLLM_TeamTable(
        team_id="test-team",
        models=["gpt-3.5-turbo"],
        access_group_ids=["ag-other"],
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_models_from_access_groups",
        new_callable=AsyncMock,
        return_value=["claude-3"],
    ):
        with pytest.raises(ProxyException):
            await can_team_access_model(
                model="gpt-4",
                team_object=team_object,
                llm_router=None,
                team_model_aliases=None,
            )


# ---------------------------------------------------------------------------
# can_key_call_model with access_group_ids fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_can_key_call_model_via_access_group_ids():
    """Test can_key_call_model allows access when key has access_group_ids granting model access."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import can_key_call_model

    user_api_key_object = UserAPIKeyAuth(
        token="test-token",
        models=[],
        access_group_ids=["ag-with-gpt4"],
    )
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4", "api_key": "test"},
            }
        ]
    )

    with patch(
        "litellm.proxy.auth.auth_checks.get_models_from_access_groups",
        new_callable=AsyncMock,
        return_value=["gpt-4"],
    ):
        await can_key_call_model(
            model="gpt-4",
            llm_model_list=[],
            valid_token=user_api_key_object,
            llm_router=router,
        )


# ---------------------------------------------------------------------------
# _key_access_group_grants_model (key access group overriding team restriction)
# ---------------------------------------------------------------------------


def _patch_proxy_server_globals():
    """Patch proxy_server's prisma_client and user_api_key_cache to non-None mocks
    so the helper's None-guard doesn't short-circuit. The actual values don't
    matter because get_access_object is patched separately to return fixtures."""
    from unittest.mock import MagicMock, patch

    return [
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
    ]


def _fake_access_group(
    access_group_id: str,
    access_model_names=None,
    assigned_team_ids=None,
    assigned_key_ids=None,
):
    from litellm.proxy._types import LiteLLM_AccessGroupTable

    return LiteLLM_AccessGroupTable(
        access_group_id=access_group_id,
        access_group_name=access_group_id,
        access_model_names=access_model_names or [],
        assigned_team_ids=assigned_team_ids or [],
        assigned_key_ids=assigned_key_ids or [],
    )


@pytest.mark.asyncio
async def test_key_access_group_grants_model_when_team_authorized():
    """Group's assigned_team_ids includes the key's team and grants the model → True.

    This is the happy path equivalent of Andres's report: admin creates an
    access group with assigned_team_ids=[team-a], grants claude-haiku-4-5,
    attaches it to a key on team-a. Override fires.
    """
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import key_access_group_grants_model

    valid_token = UserAPIKeyAuth(
        token="test-token",
        models=[],
        access_group_ids=["premium-group"],
        team_id="team-a",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-a",
        models=["mock-success"],
        access_group_ids=[],  # deliberately not synced — the access group itself authorizes
    )

    fake_ag = _fake_access_group(
        access_group_id="premium-group",
        access_model_names=["claude-haiku-4-5"],
        assigned_team_ids=["team-a"],
    )

    patches = _patch_proxy_server_globals() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            return_value=fake_ag,
        ),
    ]
    for p in patches:
        p.start()
    try:
        assert (
            await key_access_group_grants_model(
                model="claude-haiku-4-5",
                valid_token=valid_token,
                team_object=team_object,
                llm_router=None,
            )
            is True
        )
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_key_access_group_grants_model_when_key_directly_authorized():
    """Group's assigned_key_ids includes the key's token and grants the model → True.

    Per-key authorization path: an admin scopes a group directly to a key
    (assigned_key_ids) without listing the team.
    """
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import key_access_group_grants_model

    valid_token = UserAPIKeyAuth(
        token="test-token-hashed",
        models=[],
        access_group_ids=["per-key-group"],
        team_id="team-a",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-a",
        models=["mock-success"],
        access_group_ids=[],
    )

    fake_ag = _fake_access_group(
        access_group_id="per-key-group",
        access_model_names=["claude-haiku-4-5"],
        assigned_team_ids=[],
        assigned_key_ids=["test-token-hashed"],
    )

    patches = _patch_proxy_server_globals() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            return_value=fake_ag,
        ),
    ]
    for p in patches:
        p.start()
    try:
        assert (
            await key_access_group_grants_model(
                model="claude-haiku-4-5",
                valid_token=valid_token,
                team_object=team_object,
                llm_router=None,
            )
            is True
        )
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_key_access_group_grants_model_when_key_has_no_groups():
    """Key with no access_group_ids → False (early return, no DB read)."""
    from litellm.proxy.auth.auth_checks import key_access_group_grants_model

    valid_token = UserAPIKeyAuth(
        token="test-token",
        models=[],
        access_group_ids=[],
        team_id="team-a",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-a",
        models=["mock-success"],
        access_group_ids=["any-group"],
    )
    assert (
        await key_access_group_grants_model(
            model="claude-haiku-4-5",
            valid_token=valid_token,
            team_object=team_object,
            llm_router=None,
        )
        is False
    )


@pytest.mark.asyncio
async def test_key_access_group_grants_model_when_group_does_not_cover_model():
    """Group authorizes the team but does not grant the requested model → False."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import key_access_group_grants_model

    valid_token = UserAPIKeyAuth(
        token="test-token",
        models=[],
        access_group_ids=["basic-group"],
        team_id="team-a",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-a",
        models=["mock-success"],
        access_group_ids=[],
    )

    fake_ag = _fake_access_group(
        access_group_id="basic-group",
        access_model_names=["gpt-4o-mini"],
        assigned_team_ids=["team-a"],
    )

    patches = _patch_proxy_server_globals() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            return_value=fake_ag,
        ),
    ]
    for p in patches:
        p.start()
    try:
        assert (
            await key_access_group_grants_model(
                model="claude-haiku-4-5",
                valid_token=valid_token,
                team_object=team_object,
                llm_router=None,
            )
            is False
        )
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_key_access_group_grants_model_when_group_authorizes_neither():
    """
    Bypass regression test: a team member sets a foreign access group on their
    key. The group grants the requested model but its assigned_team_ids /
    assigned_key_ids do not include this caller's team or token. Override is
    denied — the team's 401 propagates.
    """
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import key_access_group_grants_model

    valid_token = UserAPIKeyAuth(
        token="team-a-token",
        models=[],
        access_group_ids=["team-b-premium"],
        team_id="team-a",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-a",
        models=["mock-success"],
        access_group_ids=[],
    )

    fake_ag = _fake_access_group(
        access_group_id="team-b-premium",
        access_model_names=["claude-opus-4-5"],
        assigned_team_ids=["team-b"],
        assigned_key_ids=["team-b-token"],
    )

    patches = _patch_proxy_server_globals() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            return_value=fake_ag,
        ),
    ]
    for p in patches:
        p.start()
    try:
        assert (
            await key_access_group_grants_model(
                model="claude-opus-4-5",
                valid_token=valid_token,
                team_object=team_object,
                llm_router=None,
            )
            is False
        )
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_key_access_group_grants_model_when_get_access_object_raises():
    """Group lookup failure (404, network, etc.) is treated as no authorization."""
    from unittest.mock import AsyncMock, patch

    from litellm.proxy.auth.auth_checks import key_access_group_grants_model

    valid_token = UserAPIKeyAuth(
        token="test-token",
        models=[],
        access_group_ids=["missing-group"],
        team_id="team-a",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-a",
        models=["mock-success"],
        access_group_ids=[],
    )

    patches = _patch_proxy_server_globals() + [
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            side_effect=Exception("not found"),
        ),
    ]
    for p in patches:
        p.start()
    try:
        assert (
            await key_access_group_grants_model(
                model="claude-haiku-4-5",
                valid_token=valid_token,
                team_object=team_object,
                llm_router=None,
            )
            is False
        )
    finally:
        for p in patches:
            p.stop()


@pytest.fixture
def db_success_hook() -> Iterator[AsyncMock]:
    hook: Final = AsyncMock()
    with patch(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        MagicMock(service_logging_obj=MagicMock(async_service_success_hook=hook)),
    ):
        yield hook


async def _db_service_call_types(hook: AsyncMock) -> tuple[str, ...]:
    await asyncio.sleep(0)
    return tuple(call.kwargs["call_type"] for call in hook.await_args_list if call.kwargs["service"] == ServiceTypes.DB)


def _prisma_client_serving(user_id: str) -> SimpleNamespace:
    row: Final = {
        "user_id": user_id,
        "user_role": "internal_user",
        "teams": [],
        "spend": 0.0,
        "models": [],
        "metadata": "{}",
        "allowed_cache_controls": [],
        "policies": [],
        "model_spend": "{}",
        "model_max_budget": "{}",
        "organization_memberships": [],
    }
    engine: Final = SimpleNamespace(query=AsyncMock(return_value={"data": {"result": row}}), stop=lambda: None)
    generated_client: Final = Prisma()
    generated_client._engine = engine
    return SimpleNamespace(db=PrismaWrapper(original_prisma=generated_client, iam_token_db_auth=False))


@pytest.mark.asyncio
async def test_get_user_object_cache_hit_emits_no_postgres_service_event(db_success_hook: AsyncMock) -> None:
    user_id: Final = f"cached-user-{uuid.uuid4()}"
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(key=user_id, value=LiteLLM_UserTable(user_id=user_id, user_role="internal_user"))

    result: Final = await get_user_object(
        user_id=user_id,
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
        user_id_upsert=False,
        parent_otel_span="auth-span",
    )

    assert result is not None and result.user_id == user_id
    assert await _db_service_call_types(db_success_hook) == ()


@pytest.mark.asyncio
async def test_get_org_object_cache_hit_emits_no_postgres_service_event(db_success_hook: AsyncMock) -> None:
    org_id: Final = f"cached-org-{uuid.uuid4()}"
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=f"org_id:{org_id}",
        value=LiteLLM_OrganizationTable(
            organization_id=org_id, budget_id="b", models=[], created_by="t", updated_by="t"
        ),
    )

    result: Final = await get_org_object(
        org_id=org_id,
        prisma_client=MagicMock(),
        user_api_key_cache=cache,
        parent_otel_span="auth-span",
    )

    assert result is not None and result.organization_id == org_id
    assert await _db_service_call_types(db_success_hook) == ()


@pytest.mark.asyncio
async def test_get_user_object_cache_miss_emits_exactly_one_postgres_get_user_object_event(
    db_success_hook: AsyncMock,
) -> None:
    user_id: Final = f"db-user-{uuid.uuid4()}"
    prisma_client: Final = _prisma_client_serving(user_id)

    result: Final = await get_user_object(
        user_id=user_id,
        prisma_client=prisma_client,
        user_api_key_cache=UserApiKeyCache(),
        user_id_upsert=False,
        parent_otel_span="auth-span",
    )

    assert result is not None and result.user_id == user_id
    assert await _db_service_call_types(db_success_hook) == ("get_user_object",)
    assert db_success_hook.await_args_list[0].kwargs["parent_otel_span"] == "auth-span"


@pytest.mark.parametrize("entry_first", [True, False])
def test_is_model_cost_zero_judges_an_alias_chain_by_the_deployment_its_entry_routes_to(
    monkeypatch: pytest.MonkeyPatch, entry_first: bool
) -> None:
    """chain-entry resolves one hop to local-free and is served by local-free's own free
    deployment, so an over-budget key is waived for it; local-free by name resolves to paid-gpt
    and stays enforced. local-free's alias is a hop the router never takes for chain-entry, and
    the per-name verdict cache must not let either name's verdict leak into the other's."""
    monkeypatch.setattr(litellm, "model_cost", dict(litellm.model_cost))
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "local-free",
                "litellm_params": {
                    "model": "ollama/qwen3:0.6b",
                    "api_base": "http://localhost:11434",
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                },
            },
            {
                "model_name": "paid-gpt",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "fake",
                    "input_cost_per_token": 3e-06,
                    "output_cost_per_token": 1.5e-05,
                },
            },
        ],
        model_group_alias={"chain-entry": "local-free", "local-free": "paid-gpt"},
    )
    expected: Final = {"chain-entry": True, "local-free": False, "paid-gpt": False}
    order: Final = ("chain-entry", "local-free", "paid-gpt") if entry_first else ("local-free", "paid-gpt", "chain-entry")

    verdicts: Final = {name: is_model_cost_zero(model=name, llm_router=router) for name in order}

    assert verdicts == expected
    assert {name: is_model_cost_zero(model=name, llm_router=router) for name in order} == expected


class _TeamRows:
    def __init__(self, database: "_Database", *, writer: bool = False) -> None:
        self._database = database
        self._writer = writer

    async def find_unique(self, where: dict[str, str], include: object = None) -> LiteLLM_TeamTable | None:
        self._database.answer(writer=self._writer)
        return self._database.teams.get(where["team_id"])

    async def find_many(self, where: dict[str, str], include: object = None) -> list[LiteLLM_TeamTable]:
        self._database.answer(writer=self._writer)
        return [team for team in self._database.teams.values() if team.team_alias == where["team_alias"]]


class _Database:
    """Answers team and key reads from its rows, on the reader or the writer, or fails every read with
    ``stopped_with`` once stopped."""

    def __init__(self) -> None:
        self.teams: dict[str, LiteLLM_TeamTable] = {}
        self.keys: dict[str, UserAPIKeyAuth] = {}
        self.stopped_with: Exception | None = None
        self.reads = 0
        self.writer_reads = 0
        self.db = SimpleNamespace(litellm_teamtable=_TeamRows(self))
        self.writer_db = SimpleNamespace(litellm_teamtable=_TeamRows(self, writer=True))

    def answer(self, *, writer: bool = False) -> None:
        self.reads += 1
        self.writer_reads += writer
        if self.stopped_with is not None:
            raise self.stopped_with

    async def get_data(
        self, token: str, table_name: str, parent_otel_span: object, proxy_logging_obj: object, use_writer: bool = False
    ):
        self.answer(writer=use_writer)
        return self.keys.get(token)

    async def attempt_db_reconnect(self, *, reason: str, timeout_seconds: float, lock_timeout_seconds: float) -> bool:
        return False


_STOPPED: Final = httpx.ConnectError("All connection attempts failed")


def _p1001() -> Exception:
    from prisma.errors import DataError

    return DataError(data={"user_facing_error": {"message": "Can't reach database server at `127.0.0.1`:`5432`"}})


def _serving_cache(**kwargs: object) -> UserApiKeyCache:
    return UserApiKeyCache(serve_last_good_entries=True, **kwargs)  # pyright: ignore[reportArgumentType]  # forwards the cache's own keyword arguments


class _HeldDatabase(_Database):
    """A database whose first read takes its rows when it starts and answers only when ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.read_started = asyncio.Event()
        self.release = asyncio.Event()
        self.db = SimpleNamespace(litellm_teamtable=_HeldTeamRows(self))

    async def held(self, row: object) -> object:
        self.read_started.set()
        await self.release.wait()
        return row

    async def get_data(
        self, token: str, table_name: str, parent_otel_span: object, proxy_logging_obj: object, use_writer: bool = False
    ):
        self.answer(writer=use_writer)
        return await self.held(self.keys.get(token))


class _HeldTeamRows(_TeamRows):
    def __init__(self, database: _HeldDatabase) -> None:
        super().__init__(database)
        self._held = database

    async def find_unique(self, where: dict[str, str], include: object = None) -> LiteLLM_TeamTable | None:
        self._held.answer()
        team = self._held.teams.get(where["team_id"])
        return await self._held.held(team)  # pyright: ignore[reportReturnType]  # held returns the row it was given


class TestLastGoodEntryWhileTheDatabaseIsUnreachable:
    """R69: a team or key the database last answered for is served while the database is unreachable,
    whatever the age of its cached entry; an unknown or missing one is refused."""

    @staticmethod
    async def _expire(cache: UserApiKeyCache, *keys: str) -> None:
        for key in keys:
            await cache.async_delete_cache(key)

    @staticmethod
    async def _team(database: _Database, cache: UserApiKeyCache, team_id: str):
        from litellm.proxy.auth.auth_checks import get_team_object

        return await get_team_object(
            team_id=team_id,
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
        )

    @staticmethod
    async def _team_by_alias(database: _Database, cache: UserApiKeyCache, alias: str):
        from litellm.proxy.auth.auth_checks import get_team_object_by_alias

        return await get_team_object_by_alias(
            team_alias=alias,
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
        )

    @staticmethod
    async def _key(database: _Database, cache: UserApiKeyCache, hashed_token: str) -> UserAPIKeyAuth:
        from litellm.proxy.auth.auth_checks import get_key_object

        return await get_key_object(
            hashed_token=hashed_token,
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stopped_with", [_STOPPED, _p1001()], ids=["engine_unreachable", "postgres_unreachable"])
    async def test_team_found_by_id_is_served_past_its_cache_life(self, stopped_with: Exception):
        database = _Database()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", team_alias="lanes", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._team(database, cache, "t1")
        await self._expire(cache, "team_id:t1")
        database.stopped_with = stopped_with

        served = await self._team(database, cache, "t1")

        assert (served.team_id, served.models) == ("t1", ["claude-opus-5-5"])

    @pytest.mark.asyncio
    async def test_team_the_gateway_never_read_is_refused(self):
        from fastapi import HTTPException

        database = _Database()
        database.stopped_with = _STOPPED

        with pytest.raises(HTTPException) as refused:
            await self._team(database, _serving_cache(), "t1")
        assert refused.value.status_code == 404

    @pytest.mark.asyncio
    async def test_nothing_is_kept_or_served_until_the_setting_turns_it_on(self):
        from fastapi import HTTPException

        redis = fake_redis_cache()
        database = _Database()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", models=["claude-opus-5-5"])
        cache = UserApiKeyCache(redis_cache=redis.cache)
        await self._team(database, cache, "t1")
        await self._expire(cache, "team_id:t1")
        database.stopped_with = _STOPPED

        with pytest.raises(HTTPException):
            await self._team(database, cache, "t1")
        assert await redis.cache.init_async_client().keys("last_good*") == []

    @pytest.mark.asyncio
    async def test_team_the_database_reported_missing_stays_refused(self):
        from fastapi import HTTPException
        from litellm.proxy.auth.auth_checks import TeamNotFoundError

        database = _Database()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._team(database, cache, "t1")
        del database.teams["t1"]
        await self._expire(cache, "team_id:t1")
        with pytest.raises(TeamNotFoundError):
            await self._team(database, cache, "t1")
        database.stopped_with = _STOPPED

        with pytest.raises(HTTPException) as refused:
            await self._team(database, cache, "t1")
        assert refused.value.status_code == 404

    @pytest.mark.asyncio
    async def test_team_is_read_afresh_when_the_database_answers_again(self):
        database = _Database()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._team(database, cache, "t1")
        await self._expire(cache, "team_id:t1")
        database.stopped_with = _STOPPED
        assert (await self._team(database, cache, "t1")).models == ["claude-opus-5-5"]

        database.stopped_with = None
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", models=["claude-opus-5"])
        assert (await self._team(database, cache, "t1")).models == ["claude-opus-5"]

        await self._expire(cache, "team_id:t1")
        database.stopped_with = _STOPPED
        assert (await self._team(database, cache, "t1")).models == ["claude-opus-5"]

    @pytest.mark.asyncio
    async def test_authoritative_team_read_is_never_served_a_last_good_entry(self):
        """``check_db_only`` is the authoritative read behind team admin checks: it reads the writer, and a writer
        that cannot be reached refuses rather than answering from the last good entry the cached read serves."""
        from fastapi import HTTPException
        from litellm.proxy.auth.auth_checks import get_team_object

        database = _Database()
        database.teams["t-authoritative"] = LiteLLM_TeamTable(team_id="t-authoritative", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._team(database, cache, "t-authoritative")
        await self._expire(cache, "team_id:t-authoritative")
        database.stopped_with = _STOPPED
        assert (await self._team(database, cache, "t-authoritative")).models == ["claude-opus-5-5"]

        with pytest.raises(HTTPException) as refused:
            await get_team_object(
                team_id="t-authoritative",
                prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
                user_api_key_cache=cache,
                check_db_only=True,
            )

        assert refused.value.status_code == 404
        assert database.writer_reads == 1

    @pytest.mark.asyncio
    async def test_authoritative_key_read_is_never_served_a_last_good_entry(self):
        from litellm.proxy.auth.auth_checks import get_key_object

        database = _Database()
        database.keys["k-authoritative"] = UserAPIKeyAuth(token="k-authoritative", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._key(database, cache, "k-authoritative")
        await self._expire(cache, "k-authoritative")
        database.stopped_with = _STOPPED
        assert (await self._key(database, cache, "k-authoritative")).token == "k-authoritative"

        with pytest.raises(httpx.ConnectError):
            await get_key_object(
                hashed_token="k-authoritative",
                prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
                user_api_key_cache=cache,
                check_db_only=True,
            )

        assert database.writer_reads == 1

    @pytest.mark.asyncio
    async def test_memory_line_serves_the_team_while_redis_is_away_too(self):
        redis = fake_redis_cache()
        database = _Database()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", models=["claude-opus-5-5"])
        cache = _serving_cache(redis_cache=redis.cache)
        await self._team(database, cache, "t1")
        await self._expire(cache, "team_id:t1")
        redis.server.connected = False
        database.stopped_with = _STOPPED

        assert (await self._team(database, cache, "t1")).models == ["claude-opus-5-5"]

    @pytest.mark.asyncio
    async def test_another_pod_serves_the_team_from_the_shared_redis_line(self):
        redis = fake_redis_cache()
        database = _Database()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", models=["claude-opus-5-5"])
        await self._team(database, _serving_cache(redis_cache=redis.cache), "t1")
        other_pod = _serving_cache(redis_cache=redis.cache)
        await self._expire(other_pod, "team_id:t1")
        database.stopped_with = _STOPPED

        assert (await self._team(database, other_pod, "t1")).models == ["claude-opus-5-5"]

    @pytest.mark.asyncio
    async def test_team_found_by_alias_is_served_past_its_cache_life(self):
        database = _Database()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", team_alias="lanes", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._team_by_alias(database, cache, "lanes")
        await self._expire(cache, "team_alias:lanes", "team_id:t1")
        database.stopped_with = _STOPPED

        served = await self._team_by_alias(database, cache, "lanes")

        assert (served.team_id, served.models) == ("t1", ["claude-opus-5-5"])

    @pytest.mark.asyncio
    async def test_alias_lookup_keeps_its_entry_for_the_management_ttl(self):
        from litellm.caching.in_memory_cache import InMemoryCache
        from litellm.constants import DEFAULT_IN_MEMORY_TTL
        from litellm.proxy.common_utils.user_api_key_cache import get_management_object_ttl

        now = [1000.0]
        cache = _serving_cache(in_memory_cache=InMemoryCache(max_size_in_memory=100, clock=lambda: now[0]))
        database = _Database()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", team_alias="lanes", models=["claude-opus-5-5"])
        await self._team_by_alias(database, cache, "lanes")
        reads_after_first_lookup = database.reads

        now[0] += DEFAULT_IN_MEMORY_TTL + 1
        await self._team_by_alias(database, cache, "lanes")
        assert database.reads == reads_after_first_lookup

        now[0] += get_management_object_ttl(cache)
        await self._team_by_alias(database, cache, "lanes")
        assert database.reads == reads_after_first_lookup + 1

    @pytest.mark.asyncio
    async def test_renamed_team_is_not_served_under_its_old_alias(self):
        from fastapi import HTTPException

        database = _Database()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", team_alias="lanes", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._team_by_alias(database, cache, "lanes")
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", team_alias="cuts", models=["claude-opus-5-5"])
        await self._expire(cache, "team_id:t1")
        await self._team(database, cache, "t1")
        await self._expire(cache, "team_alias:lanes", "team_id:t1")
        database.stopped_with = _STOPPED

        with pytest.raises(HTTPException):
            await self._team_by_alias(database, cache, "lanes")

    @pytest.mark.asyncio
    async def test_alias_the_database_reported_missing_stays_refused(self):
        from fastapi import HTTPException

        database = _Database()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", team_alias="lanes", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._team_by_alias(database, cache, "lanes")
        database.teams["t2"] = LiteLLM_TeamTable(team_id="t2", team_alias="lanes", models=["claude-opus-5"])
        await self._expire(cache, "team_alias:lanes")
        with pytest.raises(HTTPException) as ambiguous:
            await self._team_by_alias(database, cache, "lanes")
        assert ambiguous.value.status_code == 400
        await self._expire(cache, "team_id:t1")
        database.stopped_with = _STOPPED

        with pytest.raises(HTTPException):
            await self._team_by_alias(database, cache, "lanes")

    @pytest.mark.asyncio
    async def test_key_is_served_past_its_cache_life(self):
        database = _Database()
        database.keys["k1"] = UserAPIKeyAuth(token="k1", team_id="t1", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._key(database, cache, "k1")
        await self._expire(cache, "k1")
        database.stopped_with = _STOPPED

        served = await self._key(database, cache, "k1")

        assert (served.token, served.team_id, served.models) == ("k1", "t1", ["claude-opus-5-5"])

    @pytest.mark.asyncio
    async def test_key_the_gateway_never_read_is_refused(self):
        database = _Database()
        database.stopped_with = _STOPPED

        with pytest.raises(httpx.ConnectError):
            await self._key(database, _serving_cache(), "k1")

    @pytest.mark.asyncio
    async def test_deleted_key_is_refused_as_soon_as_the_database_stops(self):
        from litellm.proxy.auth.auth_checks import delete_cache_key_object

        database = _Database()
        database.keys["k1"] = UserAPIKeyAuth(token="k1", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._key(database, cache, "k1")
        del database.keys["k1"]
        await delete_cache_key_object(hashed_token="k1", user_api_key_cache=cache, proxy_logging_obj=None)
        database.stopped_with = _STOPPED

        with pytest.raises(httpx.ConnectError):
            await self._key(database, cache, "k1")

    @pytest.mark.asyncio
    async def test_rotated_out_key_read_in_its_grace_period_is_not_kept_under_the_old_hash(self):
        """During a regenerate grace period the old hash reads back the new key's row. Keeping that under the old
        hash, with no expiry, would let a later outage admit the old key after its grace period ended."""
        database = _Database()
        database.keys["old-hash"] = UserAPIKeyAuth(token="new-hash", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._key(database, cache, "old-hash")
        await self._expire(cache, "old-hash")
        database.stopped_with = _STOPPED

        with pytest.raises(httpx.ConnectError):
            await self._key(database, cache, "old-hash")

    @pytest.mark.asyncio
    async def test_key_the_database_reported_missing_stays_refused(self):
        from litellm.proxy._types import ProxyException

        database = _Database()
        database.keys["k1"] = UserAPIKeyAuth(token="k1", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._key(database, cache, "k1")
        del database.keys["k1"]
        await self._expire(cache, "k1")
        with pytest.raises(ProxyException):
            await self._key(database, cache, "k1")
        database.stopped_with = _STOPPED

        with pytest.raises(httpx.ConnectError):
            await self._key(database, cache, "k1")

    @pytest.mark.asyncio
    async def test_deleted_team_is_refused_as_soon_as_the_database_stops(self):
        from fastapi import HTTPException
        from litellm.proxy.auth.auth_checks import delete_cache_team_object

        database = _Database()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", team_alias="lanes", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._team_by_alias(database, cache, "lanes")
        del database.teams["t1"]
        await delete_cache_team_object(
            team_id="t1", team_alias="lanes", user_api_key_cache=cache, proxy_logging_obj=None
        )
        database.stopped_with = _STOPPED

        with pytest.raises(HTTPException):
            await self._team(database, cache, "t1")
        with pytest.raises(HTTPException):
            await self._team_by_alias(database, cache, "lanes")

    @pytest.mark.asyncio
    async def test_member_budget_or_role_change_drops_the_memberships_last_good_entry(self):
        """Member removal and budget or role changes run through invalidate_team_member_spend_state; the
        membership's last good entry must go with the caches, or an outage serves the old budget."""
        from litellm.proxy._types import LiteLLM_TeamMembership
        from litellm.proxy.auth.auth_checks import invalidate_team_member_spend_state
        from litellm.proxy.common_utils.user_api_key_cache import recall_last_good, team_membership_auth_cache_key

        cache = _serving_cache()
        key = team_membership_auth_cache_key(team_id="t1", user_id="u1")
        await remember_last_good(cache, key, LiteLLM_TeamMembership(user_id="u1", team_id="t1"))

        await invalidate_team_member_spend_state(user_id="u1", team_id="t1", user_api_key_cache=cache)

        assert await recall_last_good(cache, key, LiteLLM_TeamMembership) is None

    @pytest.mark.asyncio
    async def test_member_budget_change_drops_the_last_good_entries_of_that_members_keys(self, monkeypatch):
        """A key's own entry carries its team member's spend and rate limits as of its last read, so a member budget
        edit, which runs through invalidate_team_member_spend_state, drops the entries of that member's keys in the
        team, and of no other key."""
        from litellm.proxy import proxy_server
        from litellm.proxy.auth.auth_checks import invalidate_team_member_spend_state
        from litellm.proxy.common_utils.user_api_key_cache import recall_last_good

        rows: Final = (
            SimpleNamespace(token="hashed-member", user_id="u1", team_id="t1"),
            SimpleNamespace(token="hashed-member-other-team", user_id="u1", team_id="t2"),
            SimpleNamespace(token="hashed-teammate", user_id="u2", team_id="t1"),
        )

        async def find_many(where: dict[str, str]) -> list[SimpleNamespace]:
            return [row for row in rows if all(getattr(row, column) == value for column, value in where.items())]

        monkeypatch.setattr(
            proxy_server,
            "prisma_client",
            SimpleNamespace(db=SimpleNamespace(litellm_verificationtoken=SimpleNamespace(find_many=find_many))),
        )
        cache = _serving_cache()
        for row in rows:
            await remember_last_good(cache, row.token, UserAPIKeyAuth(token=row.token, team_member_spend=1.0))

        await invalidate_team_member_spend_state(user_id="u1", team_id="t1", user_api_key_cache=cache)

        assert {row.token: await recall_last_good(cache, row.token, UserAPIKeyAuth) is not None for row in rows} == {
            "hashed-member": False,
            "hashed-member-other-team": True,
            "hashed-teammate": True,
        }

    @pytest.mark.asyncio
    async def test_budget_reset_drops_the_reset_entries_last_good_copies(self, monkeypatch):
        from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
        from litellm.proxy.common_utils.user_api_key_cache import recall_last_good

        cache = _serving_cache()
        await remember_last_good(cache, "hashed-reset", UserAPIKeyAuth(token="hashed-reset", spend=99.0))
        monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", cache)

        await ResetBudgetJob._invalidate_cache("user_api_key_cache", ("hashed-reset",))

        assert await recall_last_good(cache, "hashed-reset", UserAPIKeyAuth) is None

    @pytest.mark.asyncio
    async def test_a_miss_with_the_setting_off_makes_no_redis_call(self):
        """The read path's forget runs only while entries are kept: with the setting off, an unknown key costs no
        Redis round trip."""
        from litellm.proxy._types import ProxyException

        redis = fake_redis_cache()
        client = redis.cache.init_async_client()
        await client.set("last_good:k-missing", '{"token": "k-missing"}')

        with pytest.raises(ProxyException):
            await self._key(_Database(), UserApiKeyCache(redis_cache=redis.cache), "k-missing")

        assert await client.exists("last_good:k-missing") == 1

    @pytest.mark.asyncio
    async def test_an_answer_that_is_not_an_outage_is_not_covered(self):
        from fastapi import HTTPException
        from prisma.errors import DataError

        database = _Database()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._team(database, cache, "t1")
        await self._expire(cache, "team_id:t1")
        database.stopped_with = DataError(data={"user_facing_error": {"message": "invalid input syntax"}})

        with pytest.raises(HTTPException):
            await self._team(database, cache, "t1")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", [False, True], ids=["model_query", "raw_query"])
    @pytest.mark.parametrize("reason", ["statement timeout", "user request", "lock timeout"])
    async def test_a_statement_the_database_cancelled_is_not_covered(self, reason: CancelReason, raw: bool):
        """A healthy database under load cancels statements. Only a database the gateway cannot reach is covered:
        the same team and key are served when the connection fails, and refused when the statement is cancelled."""
        from fastapi import HTTPException

        database = _Database()
        database.teams["t-cancelled"] = LiteLLM_TeamTable(team_id="t-cancelled", models=["claude-opus-5-5"])
        database.keys["k-cancelled"] = UserAPIKeyAuth(token="k-cancelled", models=["claude-opus-5-5"])
        cache = _serving_cache()
        await self._team(database, cache, "t-cancelled")
        await self._key(database, cache, "k-cancelled")
        await self._expire(cache, "team_id:t-cancelled", "k-cancelled")
        database.stopped_with = _STOPPED
        assert (await self._team(database, cache, "t-cancelled")).team_id == "t-cancelled"
        assert (await self._key(database, cache, "k-cancelled")).token == "k-cancelled"

        cancelled: Final = cancelled_statement(reason, raw=raw)
        database.stopped_with = cancelled

        with pytest.raises(HTTPException) as team_refused:
            await self._team(database, cache, "t-cancelled")
        with pytest.raises(type(cancelled)) as key_refused:
            await self._key(database, cache, "k-cancelled")
        assert team_refused.value.status_code == 404
        assert key_refused.value is cancelled

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pods", ["one_pod", "two_pods"])
    async def test_a_read_that_finishes_after_the_delete_does_not_bring_the_key_back(self, pods: str):
        from litellm.proxy.auth.auth_checks import delete_cache_key_object

        redis = fake_redis_cache()
        database = _HeldDatabase()
        database.keys["k1"] = UserAPIKeyAuth(token="k1", models=["claude-opus-5-5"])
        reading_pod = _serving_cache(redis_cache=redis.cache) if pods == "two_pods" else _serving_cache()
        deleting_pod = _serving_cache(redis_cache=redis.cache) if pods == "two_pods" else reading_pod
        in_flight = asyncio.create_task(self._key(database, reading_pod, "k1"))
        await database.read_started.wait()

        del database.keys["k1"]
        await delete_cache_key_object(
            hashed_token="k1", user_api_key_cache=deleting_pod, proxy_logging_obj=None, revoked=True
        )
        database.release.set()
        await in_flight
        await self._expire(reading_pod, "k1")
        database.stopped_with = _STOPPED

        with pytest.raises(httpx.ConnectError):
            await self._key(database, reading_pod, "k1")

    @pytest.mark.asyncio
    async def test_a_read_that_finishes_after_the_block_does_not_bring_the_team_back(self):
        from fastapi import HTTPException
        from litellm.proxy.auth.auth_checks import delete_cache_team_object

        database = _HeldDatabase()
        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", models=["claude-opus-5-5"])
        cache = _serving_cache()
        in_flight = asyncio.create_task(self._team(database, cache, "t1"))
        await database.read_started.wait()

        database.teams["t1"] = LiteLLM_TeamTable(team_id="t1", models=["claude-opus-5-5"], blocked=True)
        await delete_cache_team_object(
            team_id="t1", team_alias=None, user_api_key_cache=cache, proxy_logging_obj=None, revoked=True
        )
        database.release.set()
        assert (await in_flight).blocked is False
        await self._expire(cache, "team_id:t1")
        database.stopped_with = _STOPPED

        with pytest.raises(HTTPException):
            await self._team(database, cache, "t1")


class _LimitRows:
    """One table of ``_LimitsDatabase``, answering by its key column. ``find_many`` answers a batch read by that column,
    a registry read with no filter with every row, and the restricted-customer registry filter with the blocked rows."""

    def __init__(self, database: "_LimitsDatabase", key: str) -> None:
        self._database = database
        self._key = key
        self.rows: dict[str, BaseModel] = {}

    async def find_unique(self, where: dict[str, str], include: object = None) -> BaseModel | None:
        self._database.answer()
        return self.rows.get(where[self._key])

    async def find_many(
        self, where: dict[str, object] | None = None, take: int | None = None, include: object = None
    ) -> list[BaseModel]:
        self._database.answer()
        if where is None:
            return list(self.rows.values())
        batch = where.get(self._key)
        if isinstance(batch, dict):
            return [row for name, row in self.rows.items() if name in batch["in"]]
        return [row for row in self.rows.values() if getattr(row, "blocked", False)]


class _LimitsDatabase:
    """Answers project, organization, user, customer and budget reads, or fails every read with ``stopped_with``."""

    def __init__(self) -> None:
        self.stopped_with: Exception | None = None
        self.projects = _LimitRows(self, "project_id")
        self.organizations = _LimitRows(self, "organization_id")
        self.users = _LimitRows(self, "user_id")
        self.customers = _LimitRows(self, "user_id")
        self.budgets = _LimitRows(self, "budget_id")
        self.tags = _LimitRows(self, "tag_name")
        self.access_groups = _LimitRows(self, "access_group_name")
        self.db = SimpleNamespace(
            litellm_projecttable=self.projects,
            litellm_organizationtable=self.organizations,
            litellm_usertable=self.users,
            litellm_endusertable=self.customers,
            litellm_budgettable=self.budgets,
            litellm_tagtable=self.tags,
            litellm_modelaccessgroupbudgettable=self.access_groups,
        )
        self.writer_db = self.db

    def answer(self) -> None:
        if self.stopped_with is not None:
            raise self.stopped_with


def _limits_proxy_logging() -> MagicMock:
    proxy_logging = MagicMock()
    proxy_logging.budget_alerts = AsyncMock()
    return proxy_logging


class TestLastGoodLimitsWhileTheDatabaseIsUnreachable:
    """While the database is unreachable, the project, organization, user and customer a request is checked against
    come from the last good entry the database returned, and a request whose limits have no entry is refused: an
    absent object reads as one that limits nothing."""

    @staticmethod
    async def _expire(cache: UserApiKeyCache, *keys: str) -> None:
        for key in keys:
            await cache.async_delete_cache(key)

    @staticmethod
    async def _project(database: _LimitsDatabase, cache: UserApiKeyCache, project_id: str):
        from litellm.proxy.auth.auth_checks import get_project_object

        return await get_project_object(
            project_id=project_id,
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
        )

    @staticmethod
    async def _user(database: _LimitsDatabase, cache: UserApiKeyCache, user_id: str, check_db_only: bool = False):
        return await get_user_object(
            user_id=user_id,
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
            user_id_upsert=False,
            check_db_only=check_db_only,
        )

    @staticmethod
    async def _customer(database: _LimitsDatabase, cache: UserApiKeyCache, end_user_id: str):
        return await get_end_user_object(
            end_user_id=end_user_id,
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
        )

    @staticmethod
    async def _org_budget_check(database: _LimitsDatabase, cache: UserApiKeyCache, org_id: str) -> None:
        from litellm.proxy.auth.auth_checks import organization_max_budget_check

        await organization_max_budget_check(
            valid_token=UserAPIKeyAuth(token="k-org", org_id=org_id),
            team_object=None,
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
            proxy_logging_obj=_limits_proxy_logging(),
        )

    @pytest.mark.asyncio
    async def test_blocked_project_stays_blocked(self):
        from litellm.proxy._types import LiteLLM_ProjectTableCachedObj
        from litellm.proxy.auth.auth_checks import _run_project_checks
        from litellm.proxy.common_utils.user_api_key_cache import project_cache_key

        database = _LimitsDatabase()
        database.projects.rows["p-blocked"] = LiteLLM_ProjectTableCachedObj(project_id="p-blocked", blocked=True)
        cache = _serving_cache()
        await self._project(database, cache, "p-blocked")
        await self._expire(cache, project_cache_key("p-blocked"))
        database.stopped_with = _STOPPED

        served = await self._project(database, cache, "p-blocked")

        with pytest.raises(Exception, match="Project=p-blocked is blocked"):
            await _run_project_checks(
                project_object=served,
                _model=None,
                llm_router=None,
                skip_budget_checks=True,
                valid_token=None,
                proxy_logging_obj=_limits_proxy_logging(),
            )

    @pytest.mark.asyncio
    async def test_project_never_read_refuses_the_request(self):
        from litellm.proxy.auth.user_api_key_auth import _safe_fetch

        database = _LimitsDatabase()
        database.stopped_with = _STOPPED

        with pytest.raises(LastGoodEntryMissing) as refused:
            await _safe_fetch("project", self._project(database, _serving_cache(), "p-unknown"))
        assert refused.value.code == "503"

    @pytest.mark.asyncio
    async def test_project_the_database_reported_missing_refuses_the_request(self):
        from litellm.proxy._types import LiteLLM_ProjectTableCachedObj
        from litellm.proxy.common_utils.user_api_key_cache import project_cache_key

        database = _LimitsDatabase()
        database.projects.rows["p-gone"] = LiteLLM_ProjectTableCachedObj(project_id="p-gone", blocked=True)
        cache = _serving_cache()
        await self._project(database, cache, "p-gone")
        del database.projects.rows["p-gone"]
        await self._expire(cache, project_cache_key("p-gone"))
        assert await self._project(database, cache, "p-gone") is None
        database.stopped_with = _STOPPED

        with pytest.raises(LastGoodEntryMissing):
            await self._project(database, cache, "p-gone")

    @pytest.mark.asyncio
    async def test_nothing_changes_while_the_setting_is_off(self):
        """With the setting off, an unreachable database still surfaces as the error it was, which the auth path
        treats exactly as before."""
        from litellm.proxy._types import LiteLLM_ProjectTableCachedObj
        from litellm.proxy.auth.user_api_key_auth import _safe_fetch
        from litellm.proxy.common_utils.user_api_key_cache import project_cache_key

        database = _LimitsDatabase()
        database.projects.rows["p-off"] = LiteLLM_ProjectTableCachedObj(project_id="p-off", blocked=True)
        cache = UserApiKeyCache()
        await self._project(database, cache, "p-off")
        await self._expire(cache, project_cache_key("p-off"))
        database.stopped_with = _STOPPED

        assert await _safe_fetch("project", self._project(database, cache, "p-off")) is None
        assert await _safe_fetch("user", self._user(database, cache, "u-off")) is None
        await self._org_budget_check(database, cache, "o-off")

    @pytest.mark.asyncio
    async def test_over_budget_organization_stays_over_budget(self):
        from litellm.proxy._types import LiteLLM_BudgetTable

        database = _LimitsDatabase()
        database.organizations.rows["o-spent"] = LiteLLM_OrganizationTable(
            organization_id="o-spent",
            budget_id="b-spent",
            created_by="admin",
            updated_by="admin",
            spend=50.0,
            litellm_budget_table=LiteLLM_BudgetTable(budget_id="b-spent", max_budget=10.0),
        )
        cache = _serving_cache()
        with pytest.raises(litellm.BudgetExceededError):
            await self._org_budget_check(database, cache, "o-spent")
        await self._expire(cache, "org_id:o-spent:with_budget")
        database.stopped_with = _STOPPED

        with pytest.raises(litellm.BudgetExceededError):
            await self._org_budget_check(database, cache, "o-spent")

    @pytest.mark.asyncio
    async def test_organization_never_read_refuses_the_request(self, monkeypatch):
        """``organization_max_budget_check`` used to skip the organization's budget on any lookup error, and the
        organization lookup of a request let ``allow_requests_on_db_unavailable`` drop its limits."""
        from litellm.proxy import proxy_server
        from litellm.proxy.auth.auth_checks import get_org_object_for_request

        database = _LimitsDatabase()
        database.stopped_with = _STOPPED
        cache = _serving_cache()
        monkeypatch.setitem(proxy_server.general_settings, "allow_requests_on_db_unavailable", True)

        with pytest.raises(LastGoodEntryMissing):
            await self._org_budget_check(database, cache, "o-unknown")
        with pytest.raises(LastGoodEntryMissing):
            await get_org_object_for_request(
                org_id="o-unknown",
                prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
                user_api_key_cache=cache,
                parent_otel_span=None,
                proxy_logging_obj=None,
            )

    @pytest.mark.asyncio
    async def test_user_budget_comes_from_the_last_good_entry(self):
        from litellm.proxy._types import LiteLLM_UserTable
        from litellm.proxy.auth.user_api_key_auth import _safe_fetch

        database = _LimitsDatabase()
        database.users.rows["u-budget"] = LiteLLM_UserTable(user_id="u-budget", max_budget=10.0, spend=50.0)
        cache = _serving_cache()
        await self._user(database, cache, "u-budget")
        await self._expire(cache, "u-budget")
        database.stopped_with = _STOPPED

        served = await _safe_fetch("user", self._user(database, cache, "u-budget"))

        assert served is not None
        assert (served.user_id, served.max_budget, served.spend) == ("u-budget", 10.0, 50.0)

    @pytest.mark.asyncio
    async def test_user_never_read_refuses_the_request(self):
        from litellm.proxy.auth.user_api_key_auth import _safe_fetch

        database = _LimitsDatabase()
        database.stopped_with = _STOPPED

        with pytest.raises(LastGoodEntryMissing):
            await _safe_fetch("user", self._user(database, _serving_cache(), "u-unknown"))

    @pytest.mark.asyncio
    async def test_authoritative_user_read_is_never_served_a_last_good_entry(self):
        from litellm.proxy._types import LiteLLM_UserTable

        database = _LimitsDatabase()
        database.users.rows["u-authoritative"] = LiteLLM_UserTable(user_id="u-authoritative", user_role="proxy_admin")
        cache = _serving_cache()
        await self._user(database, cache, "u-authoritative")
        await self._expire(cache, "u-authoritative")
        database.stopped_with = _STOPPED
        assert (await self._user(database, cache, "u-authoritative")).user_role == "proxy_admin"

        with pytest.raises(httpx.ConnectError):
            await self._user(database, cache, "u-authoritative", check_db_only=True)

    @pytest.mark.asyncio
    async def test_customer_budget_comes_from_the_last_good_entry(self):
        from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_EndUserTable
        from litellm.proxy.common_utils.user_api_key_cache import end_user_cache_key

        database = _LimitsDatabase()
        database.customers.rows["c-budget"] = LiteLLM_EndUserTable(
            user_id="c-budget",
            blocked=True,
            budget_id="b-customer",
            spend=50.0,
            litellm_budget_table=LiteLLM_BudgetTable(budget_id="b-customer", max_budget=10.0),
        )
        cache = _serving_cache()
        await self._customer(database, cache, "c-budget")
        await self._expire(cache, end_user_cache_key("c-budget"), "end_user_restricted_registry")
        database.stopped_with = _STOPPED

        served = await self._customer(database, cache, "c-budget")

        assert served is not None
        assert (served.blocked, served.litellm_budget_table.max_budget) == (True, 10.0)

    @pytest.mark.asyncio
    async def test_customer_the_last_good_registry_proves_unrestricted_is_not_refused(self):
        """A customer outside the last good restricted registry restricted nothing when the database last answered,
        so it is not refused for having no entry of its own."""
        from litellm.proxy._types import LiteLLM_EndUserTable

        database = _LimitsDatabase()
        database.customers.rows["c-restricted"] = LiteLLM_EndUserTable(user_id="c-restricted", blocked=True)
        cache = _serving_cache()
        assert await self._customer(database, cache, "c-free") is None
        await self._expire(cache, "end_user_restricted_registry")
        database.stopped_with = _STOPPED

        assert await self._customer(database, cache, "c-free") is None
        await self._expire(cache, "end_user_restricted_registry")
        with pytest.raises(LastGoodEntryMissing):
            await self._customer(database, cache, "c-restricted")

    @pytest.mark.asyncio
    async def test_a_registry_read_that_hangs_is_answered_by_the_last_good_registry(self, monkeypatch):
        """A database that stops answering parks the restricted-customer registry query: the lookup deadline ends
        it, and the last good registry answers instead of every request that names a customer waiting for ever."""
        from litellm.proxy._types import LiteLLM_EndUserTable

        database = _LimitsDatabase()
        database.customers.rows["c-restricted"] = LiteLLM_EndUserTable(user_id="c-restricted", blocked=True)
        cache = _serving_cache()
        assert await self._customer(database, cache, "c-free") is None
        await self._expire(cache, "end_user_restricted_registry")
        never: Final = asyncio.Event()

        async def parked(where: object = None, take: int | None = None) -> list[BaseModel]:
            await never.wait()
            return []

        database.customers.find_many = parked
        monkeypatch.setattr("litellm.proxy.db.db_lookup_gate.PROXY_DB_LOOKUP_DEADLINE_SECONDS", 0.05)

        assert await asyncio.wait_for(self._customer(database, cache, "c-free"), timeout=5) is None

    @pytest.mark.asyncio
    async def test_blocked_customer_stays_blocked_in_the_blocked_customer_hook(self):
        from fastapi import HTTPException
        from litellm.proxy._types import LiteLLM_EndUserTable
        from litellm.proxy.common_utils.user_api_key_cache import end_user_cache_key

        from enterprise.enterprise_hooks.blocked_user_list import ENTERPRISE_BlockedUserList

        database = _LimitsDatabase()
        database.customers.rows["c-blocked"] = LiteLLM_EndUserTable(user_id="c-blocked", blocked=True)
        cache = _serving_cache()
        await self._customer(database, cache, "c-blocked")
        await self._expire(cache, end_user_cache_key("c-blocked"), "end_user_restricted_registry")
        database.stopped_with = _STOPPED
        hook = ENTERPRISE_BlockedUserList(prisma_client=database)  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient

        async def call(user: str) -> None:
            await hook.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(), cache=cache, data={"user": user}, call_type="completion"
            )

        with pytest.raises(HTTPException) as blocked:
            await call("c-blocked")
        assert blocked.value.status_code == 400
        await call("c-free")
        await self._expire(cache, "end_user_restricted_registry")
        cache.last_good_memory.clear()
        with pytest.raises(HTTPException) as refused:
            await call("c-free")
        assert refused.value.status_code == 503

    @pytest.mark.asyncio
    async def test_default_customer_budget_comes_from_the_last_good_entry(self, monkeypatch):
        from litellm.proxy._types import LiteLLM_BudgetTable
        from litellm.proxy.auth.auth_checks import get_default_end_user_budget

        monkeypatch.setattr(litellm, "max_end_user_budget_id", "b-default")
        database = _LimitsDatabase()
        database.budgets.rows["b-default"] = LiteLLM_BudgetTable(budget_id="b-default", max_budget=10.0)
        cache = _serving_cache()

        async def default_budget() -> LiteLLM_BudgetTable | None:
            return await get_default_end_user_budget(prisma_client=database, user_api_key_cache=cache)  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient

        await default_budget()
        await self._expire(cache, "default_end_user_budget:b-default")
        database.stopped_with = _STOPPED
        assert (await default_budget()).max_budget == 10.0

        await self._expire(cache, "default_end_user_budget:b-default")
        cache.last_good_memory.clear()
        with pytest.raises(LastGoodEntryMissing):
            await default_budget()

    @pytest.mark.asyncio
    async def test_team_member_default_budget_comes_from_the_last_good_entry(self):
        from litellm.proxy._types import LiteLLM_BudgetTable
        from litellm.proxy.auth.auth_checks import get_team_member_default_budget

        database = _LimitsDatabase()
        database.budgets.rows["b-member"] = LiteLLM_BudgetTable(budget_id="b-member", max_budget=10.0)
        cache = _serving_cache()

        async def member_budget(budget_id: str) -> LiteLLM_BudgetTable | None:
            return await get_team_member_default_budget(budget_id=budget_id, prisma_client=database, user_api_key_cache=cache)  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient

        await member_budget("b-member")
        await self._expire(cache, "team_member_default_budget:b-member")
        database.stopped_with = _STOPPED
        assert (await member_budget("b-member")).max_budget == 10.0
        with pytest.raises(LastGoodEntryMissing):
            await member_budget("b-unknown")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cache_key",
        [
            "project_id:p-mutated",
            "u-mutated",
            "end_user_id:c-mutated",
            "end_user_restricted_registry",
            "tag:t-mutated",
            "tag_registry",
        ],
    )
    async def test_a_mutation_that_evicts_the_cached_object_drops_its_last_good_entry(self, cache_key: str):
        from litellm.proxy._types import LiteLLM_UserTable
        from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import evict_and_broadcast
        from litellm.proxy.common_utils.user_api_key_cache import recall_last_good

        redis = fake_redis_cache()
        cache = _serving_cache(redis_cache=redis.cache)
        await remember_last_good(cache, cache_key, LiteLLM_UserTable(user_id="u-mutated"))

        await evict_and_broadcast(cache_keys=(cache_key,), user_api_key_cache=cache)

        assert await recall_last_good(cache, cache_key, LiteLLM_UserTable) is None
        assert await redis.cache.init_async_client().keys("last_good:*") == []

    @staticmethod
    async def _tag_budget_check(database: _LimitsDatabase, cache: UserApiKeyCache, tags: tuple[str, ...]) -> None:
        from litellm.proxy.auth.auth_checks import tag_max_budget_check_for_tags

        await tag_max_budget_check_for_tags(
            tags=tags,
            prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            user_api_key_cache=cache,
            proxy_logging_obj=_limits_proxy_logging(),
            valid_token=None,
        )

    @pytest.mark.asyncio
    async def test_over_budget_tag_stays_over_budget(self):
        from litellm.proxy._types import LiteLLM_BudgetTable, LiteLLM_TagTable
        from litellm.proxy.common_utils.user_api_key_cache import tag_cache_key, tag_registry_cache_key

        database = _LimitsDatabase()
        database.tags.rows["t-spent"] = LiteLLM_TagTable(
            tag_name="t-spent", spend=50.0, litellm_budget_table=LiteLLM_BudgetTable(max_budget=10.0)
        )
        cache = _serving_cache()
        with pytest.raises(litellm.BudgetExceededError):
            await self._tag_budget_check(database, cache, ("t-spent", "t-free"))
        await self._expire(cache, tag_cache_key("t-spent"), tag_registry_cache_key())
        database.stopped_with = _STOPPED

        with pytest.raises(litellm.BudgetExceededError):
            await self._tag_budget_check(database, cache, ("t-spent", "t-free"))

    @pytest.mark.asyncio
    async def test_tag_never_read_refuses_the_request(self):
        database = _LimitsDatabase()
        database.stopped_with = _STOPPED

        with pytest.raises(LastGoodEntryMissing):
            await self._tag_budget_check(database, _serving_cache(), ("t-unknown",))

    @pytest.mark.asyncio
    async def test_model_access_group_budget_comes_from_the_last_good_entry(self):
        from litellm.proxy.auth.auth_checks import get_model_access_group_budgets_batch
        from litellm.proxy.common_utils.user_api_key_cache import model_access_group_cache_key

        database = _LimitsDatabase()
        database.access_groups.rows["g-budget"] = SimpleNamespace(  # pyright: ignore[reportArgumentType]  # a prisma row needs only these fields
            access_group_name="g-budget", spend=50.0, litellm_budget_table=SimpleNamespace(max_budget=10.0)
        )
        cache = _serving_cache()

        async def budgets(*groups: str):
            return await get_model_access_group_budgets_batch(
                access_group_names=groups,
                prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
                user_api_key_cache=cache,
            )

        await budgets("g-budget")
        await self._expire(cache, model_access_group_cache_key("g-budget"))
        database.stopped_with = _STOPPED

        served = (await budgets("g-budget"))["g-budget"]

        assert (served.spend, served.max_budget) == (50.0, 10.0)
        with pytest.raises(LastGoodEntryMissing):
            await budgets("g-unknown")

    @pytest.mark.asyncio
    async def test_global_proxy_spend_comes_from_the_last_good_entry(self):
        from litellm.constants import GLOBAL_PROXY_SPEND_CACHE_KEY, LITELLM_PROXY_BUDGET_NAME
        from litellm.proxy._types import LiteLLM_UserTable
        from litellm.proxy.auth.user_api_key_auth import fetch_global_spend_with_event_coordination

        database = _LimitsDatabase()
        database.users.rows[LITELLM_PROXY_BUDGET_NAME] = LiteLLM_UserTable(user_id=LITELLM_PROXY_BUDGET_NAME, spend=99.0)
        cache = _serving_cache()

        async def global_spend(cache: UserApiKeyCache) -> float | None:
            return await fetch_global_spend_with_event_coordination(
                cache_key=GLOBAL_PROXY_SPEND_CACHE_KEY,
                user_api_key_cache=cache,
                prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
            )

        assert await global_spend(cache) == 99.0
        await self._expire(cache, GLOBAL_PROXY_SPEND_CACHE_KEY)
        database.stopped_with = _STOPPED

        assert await global_spend(cache) == 99.0
        with pytest.raises(LastGoodEntryMissing):
            await global_spend(_serving_cache())

    @pytest.mark.asyncio
    async def test_requests_waiting_on_a_global_spend_load_that_finds_no_entry_are_refused_too(self):
        """Concurrent requests wait on one global spend load. With no entry kept, each of them is refused like the
        request that ran the load, where they used to read the empty cache and skip the proxy budget."""
        from litellm.constants import GLOBAL_PROXY_SPEND_CACHE_KEY
        from litellm.proxy.auth.user_api_key_auth import fetch_global_spend_with_event_coordination

        database = _LimitsDatabase()
        loads: Final = asyncio.Queue[object]()
        load_started: Final = asyncio.Event()
        release: Final = asyncio.Event()

        async def stalled(where: dict[str, str], include: object = None) -> BaseModel | None:
            loads.put_nowait(where)
            load_started.set()
            await release.wait()
            raise _STOPPED

        database.users.find_unique = stalled
        cache = _serving_cache()
        requests: Final = [
            asyncio.create_task(
                fetch_global_spend_with_event_coordination(
                    cache_key=GLOBAL_PROXY_SPEND_CACHE_KEY,
                    user_api_key_cache=cache,
                    prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
                )
            )
            for _ in range(3)
        ]
        await load_started.wait()
        for _ in range(5):
            await asyncio.sleep(0)
        release.set()

        outcomes: Final = await asyncio.gather(*requests, return_exceptions=True)

        assert [type(outcome) for outcome in outcomes] == [LastGoodEntryMissing] * 3
        assert loads.qsize() == 1

    @pytest.mark.asyncio
    async def test_user_the_database_reported_missing_refuses_the_request(self, monkeypatch):
        from litellm.proxy.auth import auth_checks
        from litellm.types.proxy.auth.auth_checks import UserNotFoundError

        database = _LimitsDatabase()
        database.users.rows["u-gone"] = LiteLLM_UserTable(user_id="u-gone", max_budget=10.0, spend=50.0)
        cache = _serving_cache()
        await self._user(database, cache, "u-gone")
        del database.users.rows["u-gone"]
        await self._expire(cache, "u-gone")
        with pytest.raises(UserNotFoundError):
            await self._user(database, cache, "u-gone")
        monkeypatch.setattr(auth_checks, "db_cache_expiry", 0)
        database.stopped_with = _STOPPED

        with pytest.raises(LastGoodEntryMissing):
            await self._user(database, cache, "u-gone")

    @pytest.mark.asyncio
    async def test_customer_the_database_reported_missing_refuses_the_request(self):
        """A token carrying a customer budget reads the customer's own row, so the restricted registry cannot vouch for
        it, and a row the database reported missing must not come back from its last good entry."""
        from litellm.proxy.common_utils.user_api_key_cache import end_user_cache_key

        database = _LimitsDatabase()
        database.customers.rows["c-gone"] = LiteLLM_EndUserTable(user_id="c-gone", blocked=True)
        cache = _serving_cache()

        async def customer() -> LiteLLM_EndUserTable | None:
            return await get_end_user_object(
                end_user_id="c-gone",
                prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
                user_api_key_cache=cache,
                token_end_user_max_budget=10.0,
            )

        held: Final = await customer()
        assert held is not None and held.blocked
        del database.customers.rows["c-gone"]
        await self._expire(cache, end_user_cache_key("c-gone"))
        assert await customer() is None
        database.stopped_with = _STOPPED

        with pytest.raises(LastGoodEntryMissing):
            await customer()

    @staticmethod
    def _stop_answering(rows: _LimitRows, monkeypatch) -> None:
        """The database stops answering the table's batch reads, which only the lookup deadline ends."""
        never: Final = asyncio.Event()

        async def parked(where: object = None, take: int | None = None, include: object = None) -> list[BaseModel]:
            await never.wait()
            return []

        rows.find_many = parked
        monkeypatch.setattr("litellm.proxy.db.db_lookup_gate.PROXY_DB_LOOKUP_DEADLINE_SECONDS", 0.05)

    @pytest.mark.asyncio
    async def test_a_tag_read_that_hangs_is_answered_by_the_last_good_entry(self, monkeypatch):
        from litellm.proxy._types import LiteLLM_TagTable
        from litellm.proxy.common_utils.user_api_key_cache import tag_cache_key, tag_registry_cache_key

        database = _LimitsDatabase()
        database.tags.rows["t-spent"] = LiteLLM_TagTable(
            tag_name="t-spent", spend=50.0, litellm_budget_table=LiteLLM_BudgetTable(max_budget=10.0)
        )
        cache = _serving_cache()
        with pytest.raises(litellm.BudgetExceededError):
            await self._tag_budget_check(database, cache, ("t-spent",))
        await self._expire(cache, tag_cache_key("t-spent"), tag_registry_cache_key())
        self._stop_answering(database.tags, monkeypatch)

        with pytest.raises(litellm.BudgetExceededError):
            await asyncio.wait_for(self._tag_budget_check(database, cache, ("t-spent",)), timeout=5)

    @pytest.mark.asyncio
    async def test_a_model_access_group_read_that_hangs_is_answered_by_the_last_good_entry(self, monkeypatch):
        from litellm.proxy.auth.auth_checks import get_model_access_group_budgets_batch
        from litellm.proxy.common_utils.user_api_key_cache import model_access_group_cache_key

        database = _LimitsDatabase()
        database.access_groups.rows["g-budget"] = SimpleNamespace(  # pyright: ignore[reportArgumentType]  # a prisma row needs only these fields
            access_group_name="g-budget", spend=50.0, litellm_budget_table=SimpleNamespace(max_budget=10.0)
        )
        cache = _serving_cache()

        async def budgets():
            return await get_model_access_group_budgets_batch(
                access_group_names=("g-budget",),
                prisma_client=database,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
                user_api_key_cache=cache,
            )

        await budgets()
        await self._expire(cache, model_access_group_cache_key("g-budget"))
        self._stop_answering(database.access_groups, monkeypatch)

        served = (await asyncio.wait_for(budgets(), timeout=5))["g-budget"]

        assert (served.spend, served.max_budget) == (50.0, 10.0)
