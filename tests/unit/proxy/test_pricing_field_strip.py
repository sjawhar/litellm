"""Proxy strips client-supplied pricing parameters from request bodies.

`litellm.completion` accepts pricing fields (`input_cost_per_token`,
`output_cost_per_token`, the rest of `CustomPricingLiteLLMParams`,
`metadata.model_info`) as part of its kwarg surface. On direct SDK use that
is intentional. On the proxy, those same fields would let any caller rewrite
their own per-request cost and — via `litellm.register_model` — mutate
`litellm.model_cost` for every subsequent caller in the worker. The proxy
strips them at the boundary; an opt-in key/team flag preserves the override
for operators who actually want it.
"""

import asyncio
import copy
from unittest.mock import MagicMock

import pytest
from fastapi import Request

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import (
    _CLIENT_PRICING_CONTROL_FIELDS,
    _CLIENT_PRICING_METADATA_FIELDS,
    _strip_client_pricing_overrides,
    add_litellm_data_to_request,
    strip_client_guardrail_information,
)
from litellm.types.router import RouterRateLimitError
from litellm.types.utils import CustomPricingLiteLLMParams


def _make_request_mock() -> Request:
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"
    return request_mock


def _user_api_key_auth(metadata=None, team_metadata=None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed-key",
        metadata=metadata or {},
        team_metadata=team_metadata or {},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )


class TestStripClientPricingOverrides:
    def test_pricing_field_set_tracks_pydantic_model(self):
        # The strip set is built from the model so additions are picked up
        # automatically — this test guards against the model and the strip
        # set drifting apart if someone replaces the auto-derivation later.
        assert _CLIENT_PRICING_CONTROL_FIELDS >= frozenset(CustomPricingLiteLLMParams.model_fields.keys())
        # Sanity: the obvious top-level pricing fields are in the set.
        for field in (
            "input_cost_per_token",
            "output_cost_per_token",
            "cost_per_second",
            "input_cost_per_second",
            "cache_creation_input_token_cost",
        ):
            assert field in _CLIENT_PRICING_CONTROL_FIELDS

    def test_root_pricing_fields_dropped(self):
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "cache_creation_input_token_cost": 0.0,
        }
        _strip_client_pricing_overrides(data)
        assert data == {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_root_base_model_dropped(self):
        # ``base_model`` picks the cost-map entry a spend row is priced with
        # (success rows via the standard logging payload, billed failure rows
        # via the failure lift), so a caller naming a cheaper model lowers
        # their own recorded spend.
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "base_model": "gpt-4o-mini",
        }
        _strip_client_pricing_overrides(data)
        assert data == {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_metadata_model_info_dropped(self):
        data = {
            "model": "gpt-4",
            "metadata": {
                "user_session": "keep-me",
                "model_info": {"input_cost_per_token": 0.0},
            },
            "litellm_metadata": {
                "model_info": {"output_cost_per_token": 0.0},
            },
        }
        _strip_client_pricing_overrides(data)
        assert data["metadata"] == {"user_session": "keep-me"}
        assert data["litellm_metadata"] == {}

    def test_metadata_guardrail_information_dropped(self):
        # Client-seeded guardrail entries would otherwise be summed into
        # response_cost and spend, letting a caller forge (even negative)
        # guardrail cost against their own budget.
        data = {
            "model": "gpt-4",
            "metadata": {
                "user_session": "keep-me",
                "standard_logging_guardrail_information": [
                    {
                        "guardrail_name": "forged",
                        "guardrail_status": "success",
                        "guardrail_cost": -0.005,
                    }
                ],
            },
            "litellm_metadata": {
                "standard_logging_guardrail_information": [{"guardrail_cost": 5.0}],
            },
        }
        strip_client_guardrail_information(data)
        assert data["metadata"] == {"user_session": "keep-me"}
        assert data["litellm_metadata"] == {}

    def test_non_pricing_fields_untouched(self):
        data = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100,
            "tools": [{"type": "function"}],
            "metadata": {"trace_id": "abc"},
        }
        snapshot = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100,
            "tools": [{"type": "function"}],
            "metadata": {"trace_id": "abc"},
        }
        _strip_client_pricing_overrides(data)
        assert data == snapshot

    def test_metadata_strip_handles_non_dict_metadata(self):
        # Defensive — Pydantic validation would normally reject non-dict
        # metadata, but the strip mustn't crash if a malformed body sneaks in.
        _strip_client_pricing_overrides({"metadata": "not-a-dict"})
        _strip_client_pricing_overrides({"metadata": None})
        _strip_client_pricing_overrides({"litellm_metadata": ["a", "b"]})

    def test_metadata_field_set_contains_model_info(self):
        assert "model_info" in _CLIENT_PRICING_METADATA_FIELDS

    def test_strip_emits_debug_log_listing_dropped_fields(self, caplog):
        # Operators need a paper trail so they can diagnose why a previously
        # working override stopped applying after the strip landed.
        import logging

        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.setLevel(logging.DEBUG)
        with caplog.at_level(logging.DEBUG, logger=verbose_proxy_logger.name):
            _strip_client_pricing_overrides(
                {
                    "model": "gpt-4",
                    "input_cost_per_token": 0.0,
                    "metadata": {"model_info": {"output_cost_per_token": 0.0}},
                }
            )
        log_text = " ".join(record.getMessage() for record in caplog.records)
        assert "input_cost_per_token" in log_text
        assert "metadata.model_info" in log_text
        assert "allow_client_pricing_override" in log_text

    def test_strip_does_not_log_when_no_fields_present(self, caplog):
        # No-op strips must stay silent so the log isn't filled with noise on
        # every legitimate request.
        import logging

        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.setLevel(logging.DEBUG)
        with caplog.at_level(logging.DEBUG, logger=verbose_proxy_logger.name):
            _strip_client_pricing_overrides({"model": "gpt-4", "temperature": 0.7})
        assert not any("pricing" in record.getMessage().lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_root_pricing_fields():
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }

    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert "input_cost_per_token" not in updated
    assert "output_cost_per_token" not in updated


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_preserves_caller_streaming_request():
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "is_streaming_request": True,
    }

    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert updated["is_streaming_request"] is True


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_client_disconnect_metadata():
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {
            "client_disconnected": True,
            "error_information": {
                "error_code": "499",
                "error_message": "Client disconnected the request",
                "error_class": "ClientDisconnected",
            },
        },
    }

    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert "client_disconnected" not in updated.get("metadata", {})
    assert "error_information" not in updated.get("metadata", {})


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_metadata_model_info():
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"model_info": {"input_cost_per_token": 0.0}},
    }

    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert "model_info" not in updated.get("metadata", {})


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_skips_strip_with_key_opt_in():
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "input_cost_per_token": 0.0001,
        "base_model": "gpt-4o-mini",
        "metadata": {"model_info": {"output_cost_per_token": 0.0002}},
    }

    user_auth = _user_api_key_auth(metadata={"allow_client_pricing_override": True})
    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=user_auth,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert updated["input_cost_per_token"] == 0.0001
    assert updated["metadata"]["model_info"] == {"output_cost_per_token": 0.0002}
    assert updated["base_model"] == "gpt-4o-mini"


@pytest.mark.asyncio
@pytest.mark.parametrize("opt_in_field", ["metadata", "team_metadata"])
async def test_add_litellm_data_to_request_strips_guardrail_information_despite_pricing_opt_in(opt_in_field):
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "input_cost_per_token": 0.0001,
        "metadata": {
            "user_session": "keep-me",
            "standard_logging_guardrail_information": [{"guardrail_name": "forged", "guardrail_cost": 5.0}],
        },
    }

    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(**{opt_in_field: {"allow_client_pricing_override": True}}),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert updated["input_cost_per_token"] == 0.0001
    assert updated["metadata"]["user_session"] == "keep-me"
    assert "standard_logging_guardrail_information" not in updated["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_client_pricing_override", [False, True])
async def test_add_litellm_data_to_request_strips_root_model_info_and_cooldown_time(allow_client_pricing_override):
    """The router writes a dispatched call's ``model_info`` and takes ``cooldown_time`` from the deployment, and its
    failure callbacks count against and cool the deployment a logged ``model_info.id`` names. A body's root value of
    either never reaches a logging object, whatever the pricing opt-in, which still decides ``metadata.model_info``."""
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "model_info": {"id": "victim-deployment"},
        "cooldown_time": 3600,
        "metadata": {"model_info": {"output_cost_per_token": 0.0002}},
    }

    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(metadata={"allow_client_pricing_override": allow_client_pricing_override}),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert {
        "root": {field: updated.get(field) for field in ("model_info", "cooldown_time")},
        "metadata_model_info": updated["metadata"].get("model_info"),
    } == {
        "root": {"model_info": None, "cooldown_time": None},
        "metadata_model_info": {"output_cost_per_token": 0.0002} if allow_client_pricing_override else None,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_client_pricing_override", [False, True])
@pytest.mark.parametrize("bucket", ["metadata", "litellm_metadata"])
async def test_add_litellm_data_to_request_strips_a_body_litellm_params_and_metadata_deployment(
    bucket, allow_client_pricing_override
):
    """The SDK spreads a body's root ``litellm_params`` over the parameters its callbacks read, and the router writes
    ``deployment`` on dispatch, so the request setup removes both whatever the pricing opt-in. ``model_group`` stays:
    A2A, the MCP REST tool call and search write their own before the setup. ``model_info`` still follows the opt-in."""
    updated = await add_litellm_data_to_request(
        data={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "litellm_params": {"model_info": {"id": "victim-deployment"}, "cooldown_time": 86400},
            bucket: {
                "deployment": "openai/victim-model",
                "model_group": "a2a_agent/route-written",
                "model_info": {"output_cost_per_token": 0.0002},
            },
        },
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(metadata={"allow_client_pricing_override": allow_client_pricing_override}),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert {
        "litellm_params": updated.get("litellm_params"),
        "metadata": {field: updated["metadata"].get(field) for field in ("deployment", "model_group", "model_info")},
    } == {
        "litellm_params": None,
        "metadata": {
            "deployment": None,
            "model_group": "a2a_agent/route-written",
            "model_info": {"output_cost_per_token": 0.0002} if allow_client_pricing_override else None,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("slot", "sampling_rate"), [("metadata", 0.25), ("litellm_params", None)])
async def test_a_callback_param_is_read_from_metadata_and_no_longer_from_a_body_litellm_params(slot, sampling_rate):
    """The SDK reads the callback parameters a caller may set (``langsmith_sampling_rate`` among them) from
    ``metadata``, ``litellm_metadata`` and ``litellm_params.metadata``. The request setup removes a body's root
    ``litellm_params``, so on a proxy request only the first two still take effect."""
    from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
        initialize_standard_callback_dynamic_params,
    )

    claim = {"langsmith_sampling_rate": 0.25}
    updated = await add_litellm_data_to_request(
        data={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            **({"metadata": claim} if slot == "metadata" else {"litellm_params": {"metadata": claim}}),
        },
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert initialize_standard_callback_dynamic_params(updated).get("langsmith_sampling_rate") == sampling_rate


@pytest.mark.asyncio
async def test_a_failure_after_dispatch_counts_and_cools_the_deployment_the_router_chose_not_the_one_the_body_names():
    """The positive control for the strip above: a provider failure after dispatch still counts against, and cools
    for the router's own ``cooldown_time``, the deployment the router chose, while the deployment and the time the
    body names get nothing."""
    from litellm.router_utils.router_callbacks.track_deployment_metrics import (
        get_deployment_failures_for_current_minute,
    )

    deployment_ids = ("chosen-a", "chosen-b", "body-named")
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4", "api_key": "sk-test"},
                "model_info": {"id": "chosen-a"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                "model_info": {"id": "chosen-b"},
            },
            {
                "model_name": "other-model",
                "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                "model_info": {"id": "body-named"},
            },
        ],
        cooldown_time=13,
        num_retries=0,
    )
    data = await add_litellm_data_to_request(
        data={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "model_info": {"id": "body-named"},
            "cooldown_time": 86400,
        },
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    with pytest.raises(litellm.RateLimitError):
        router.completion(**data, mock_response="litellm.RateLimitError")

    failures = {
        deployment_id: get_deployment_failures_for_current_minute(
            litellm_router_instance=router, deployment_id=deployment_id
        )
        for deployment_id in deployment_ids
    }
    cooldowns = {
        deployment_id: cooldown["cooldown_time"]
        for deployment_id, cooldown in router.cooldown_cache.get_active_cooldowns(
            model_ids=list(deployment_ids), parent_otel_span=None
        )
    }
    chosen = [deployment_id for deployment_id, count in failures.items() if count]
    assert chosen in (["chosen-a"], ["chosen-b"])
    assert {"failures": failures, "cooldowns": cooldowns} == {
        "failures": {deployment_id: int(deployment_id in chosen) for deployment_id in deployment_ids},
        "cooldowns": {chosen[0]: 13},
    }


def _deployment_counters(router: litellm.Router, deployment_ids: tuple[str, ...]) -> dict[str, object]:
    """What the router's failure callbacks wrote for each deployment: its failures this minute, its active cooldown,
    and the deployments an RPM usage key names."""
    from litellm.router_utils.router_callbacks.track_deployment_metrics import (
        get_deployment_failures_for_current_minute,
    )

    return {
        "failures": {
            deployment_id: get_deployment_failures_for_current_minute(
                litellm_router_instance=router, deployment_id=deployment_id
            )
            for deployment_id in deployment_ids
        },
        "cooldowns": {
            deployment_id: cooldown["cooldown_time"]
            for deployment_id, cooldown in router.cooldown_cache.get_active_cooldowns(
                model_ids=list(deployment_ids), parent_otel_span=None
            )
        },
        "rpm_usage": sorted(
            {
                key.split(":")[1]
                for key in router.cache.in_memory_cache.cache_dict
                if isinstance(key, str) and key.startswith("global_router:") and ":rpm:" in key
            }
            & set(deployment_ids)
        ),
    }


@pytest.mark.asyncio
async def test_a_provider_error_counts_against_the_deployment_the_router_chose_not_one_a_body_litellm_params_names():
    """On a provider that is not OpenAI the SDK copies the request parameters it does not know into the parameters
    its failure callbacks read, so a body's root ``litellm_params`` replaced the ``litellm_params`` the router's
    callbacks take the deployment and the cooldown from. The request setup removes it: a provider error counts
    against, cools for the router's own ``cooldown_time`` and adds usage to the deployment the router chose, and the
    deployment the body named gets no failure, no cooldown and no usage."""

    deployment_ids = ("chosen-a", "chosen-b", "body-named")
    router = litellm.Router(
        model_list=[
            {
                "model_name": "claude",
                "litellm_params": {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-test"},
                "model_info": {"id": "chosen-a"},
            },
            {
                "model_name": "claude",
                "litellm_params": {"model": "anthropic/claude-3-5-haiku-20241022", "api_key": "sk-test"},
                "model_info": {"id": "chosen-b"},
            },
            {
                "model_name": "other-model",
                "litellm_params": {"model": "anthropic/claude-3-opus-20240229", "api_key": "sk-test"},
                "model_info": {"id": "body-named"},
            },
        ],
        cooldown_time=13,
        num_retries=0,
    )
    data = await add_litellm_data_to_request(
        data={
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "litellm_params": {
                "model_info": {"id": "body-named"},
                "cooldown_time": 86400,
                "metadata": {"model_group": "other-model", "deployment": "anthropic/claude-3-opus-20240229"},
            },
        },
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(**data, mock_response="litellm.RateLimitError")
    pending = asyncio.all_tasks() - {asyncio.current_task()}
    await asyncio.gather(*pending, return_exceptions=True)

    counters = _deployment_counters(router, deployment_ids)
    chosen = next(
        (deployment_id for deployment_id in ("chosen-a", "chosen-b") if counters["failures"][deployment_id]), None
    )
    assert counters == {
        "failures": {deployment_id: int(deployment_id == chosen) for deployment_id in deployment_ids},
        "cooldowns": {chosen: 13},
        "rpm_usage": [chosen],
    }


def _cooling_primary_router(backup_mock_response: str, cooldown_time: float | None = None) -> litellm.Router:
    """A ``primary`` group whose only deployment is cooling, so a request for it reaches its fallback without any
    provider call; a ``backup`` group of two Anthropic deployments answering ``backup_mock_response``; and another
    group of two deployments a body may name."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                "model_info": {"id": "primary-a"},
            },
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "api_key": "sk-test",
                    "mock_response": backup_mock_response,
                },
                "model_info": {"id": "chosen-a"},
            },
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-haiku-20241022",
                    "api_key": "sk-test",
                    "mock_response": backup_mock_response,
                },
                "model_info": {"id": "chosen-b"},
            },
            {
                "model_name": "other",
                "litellm_params": {"model": "anthropic/claude-3-opus-20240229", "api_key": "sk-test"},
                "model_info": {"id": "body-named"},
            },
            {
                "model_name": "other",
                "litellm_params": {"model": "anthropic/claude-3-haiku-20240307", "api_key": "sk-test"},
                "model_info": {"id": "body-named-sibling"},
            },
        ],
        num_retries=0,
        **({"cooldown_time": cooldown_time} if cooldown_time is not None else {}),
    )
    router.cooldown_cache.add_deployment_to_cooldown(
        model_id="primary-a",
        original_exception=litellm.RateLimitError("cooling", "openai", "gpt-4.1"),
        exception_status=429,
        cooldown_time=600.0,
    )
    return router


@pytest.mark.asyncio
async def test_a_dict_fallback_target_cannot_count_cool_or_add_usage_to_a_deployment_its_litellm_params_names():
    """The router merges a dict fallback target into the call's parameters when it falls back to it, after the
    request setup ran on the root. The primary reaches no provider, so the only provider call is the backup's: its
    error counts against, cools for the router's own ``cooldown_time`` and adds usage to the backup deployment the
    router chose, and the deployment the target's ``litellm_params`` names gets no failure, no cooldown and no usage."""
    router = _cooling_primary_router(backup_mock_response="litellm.RateLimitError", cooldown_time=13)
    data = await add_litellm_data_to_request(
        data={
            "model": "primary",
            "messages": [{"role": "user", "content": "hi"}],
            "fallbacks": [
                {
                    "model": "backup",
                    "litellm_params": {
                        "model_info": {"id": "body-named"},
                        "cooldown_time": 86400,
                        "metadata": {"model_group": "other", "deployment": "anthropic/claude-3-opus-20240229"},
                    },
                }
            ],
        },
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    with pytest.raises((litellm.RateLimitError, RouterRateLimitError)):
        await router.acompletion(**data)
    pending = asyncio.all_tasks() - {asyncio.current_task()}
    await asyncio.gather(*pending, return_exceptions=True)

    deployment_ids = ("chosen-a", "chosen-b", "body-named", "body-named-sibling")
    counters = _deployment_counters(router, deployment_ids)
    chosen = next(
        (deployment_id for deployment_id in ("chosen-a", "chosen-b") if counters["failures"][deployment_id]), None
    )
    assert counters == {
        "failures": {deployment_id: int(deployment_id == chosen) for deployment_id in deployment_ids},
        "cooldowns": {chosen: 13},
        "rpm_usage": [chosen],
    }


@pytest.mark.asyncio
async def test_a_served_fallback_keeps_the_requests_own_identity_not_one_its_dict_target_names(monkeypatch):
    """A dict fallback target's ``metadata`` replaced the request's own when the router merged the target into the
    call, so a served fallback call carried the key, team and user the body named, and its success spend row is
    filed under the identity it reads from that metadata. The request setup removes a target's metadata buckets: the
    fallback hop keeps the identity the proxy stamped."""
    router = _cooling_primary_router(backup_mock_response="served by the fallback")
    identity_fields = ("user_api_key", "user_api_key_team_id", "user_api_key_user_id")
    named_identity = dict(zip(identity_fields, ("victim-key-hash", "victim-team", "victim-user")))
    data = await add_litellm_data_to_request(
        data={
            "model": "primary",
            "messages": [{"role": "user", "content": "hi"}],
            "fallbacks": [{"model": "backup", "metadata": named_identity}],
        },
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth().model_copy(update={"team_id": "own-team", "user_id": "own-user"}),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )
    stamped = {field: data["metadata"].get(field) for field in identity_fields}
    sdk_calls: list[dict[str, object]] = []
    sdk_acompletion = litellm.acompletion

    async def recording_acompletion(**kwargs):
        sdk_calls.append(kwargs)
        return await sdk_acompletion(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", recording_acompletion)

    response = await router.acompletion(**data)

    fallback_hop_metadata = sdk_calls[-1]["metadata"]
    assert {
        "content": response.choices[0].message.content,
        "fallback_hop_identity": {field: fallback_hop_metadata.get(field) for field in identity_fields},
    } == {
        "content": "served by the fallback",
        "fallback_hop_identity": {**stamped, "user_api_key_team_id": "own-team", "user_api_key_user_id": "own-user"},
    }


def _target_with_caller_overrides() -> dict[str, object]:
    return {
        "model": "backup",
        "messages": [{"role": "user", "content": "hi"}],
        "model_info": {"id": "victim"},
        "cooldown_time": 86400,
        "litellm_params": {"model_info": {"id": "victim"}},
        "metadata": {"user_api_key_team_id": "victim-team", "deployment": "openai/victim"},
        "litellm_metadata": {"user_api_key_team_id": "victim-team"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "path"),
    [
        ({"fallbacks": ["backup-string", _target_with_caller_overrides()]}, ("fallbacks",)),
        ({"fallbacks": [{"gpt-4": ["backup-string", _target_with_caller_overrides()]}]}, ("fallbacks", 0, "gpt-4")),
        (
            {"context_window_fallbacks": [{"gpt-4": ["backup-string", _target_with_caller_overrides()]}]},
            ("context_window_fallbacks", 0, "gpt-4"),
        ),
        (
            {"content_policy_fallbacks": [{"gpt-4": ["backup-string", _target_with_caller_overrides()]}]},
            ("content_policy_fallbacks", 0, "gpt-4"),
        ),
        (
            {
                "router_settings_override": {
                    "fallbacks": [{"gpt-4": ["backup-string", _target_with_caller_overrides()]}]
                }
            },
            ("router_settings_override", "fallbacks", 0, "gpt-4"),
        ),
        (
            {"fallbacks": [{"model": "outer", "fallbacks": ["backup-string", _target_with_caller_overrides()]}]},
            ("fallbacks", 0, "fallbacks"),
        ),
    ],
    ids=["target-list", "group-map", "context-window", "content-policy", "router-settings-override", "nested"],
)
async def test_add_litellm_data_to_request_strips_caller_overrides_from_every_dict_fallback_target(body, path):
    """Every dict fallback target a body supplies, in each shape the router reads, loses the router-owned fields, a
    ``litellm_params`` and both metadata buckets; its other keys and every string target stay as sent."""
    updated = await add_litellm_data_to_request(
        data={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], **copy.deepcopy(body)},
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    targets: object = updated
    for step in path:
        targets = targets[step]
    assert targets == ["backup-string", {"model": "backup", "messages": [{"role": "user", "content": "hi"}]}]



@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_json_string_litellm_metadata():
    """``litellm_metadata`` may arrive as a JSON-encoded string (multipart/
    form-data or ``extra_body``). The strip has to run after the proxy parses
    it into a dict but before the chat-route fold into ``metadata``; otherwise
    ``model_info`` survives via the string path and lands in the folded bucket.
    """
    import json

    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_metadata": json.dumps({"model_info": {"input_cost_per_token": 0.0}}),
    }

    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert "litellm_metadata" not in updated
    assert "model_info" not in updated["metadata"]


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_skips_strip_with_team_opt_in():
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "input_cost_per_token": 0.0001,
    }

    user_auth = _user_api_key_auth(team_metadata={"allow_client_pricing_override": True})
    updated = await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=user_auth,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert updated["input_cost_per_token"] == 0.0001


@pytest.mark.asyncio
async def test_global_model_cost_unmutated_after_stripped_request(monkeypatch):
    """After a stripped request, ``litellm.model_cost`` must not carry the
    caller's submitted pricing for the model. The mutation only happens when
    the pricing fields reach ``litellm.completion``; the strip prevents that."""
    snapshot = dict(litellm.model_cost)
    data = {
        "model": "test-pricing-canary-model",
        "messages": [{"role": "user", "content": "hi"}],
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }

    await add_litellm_data_to_request(
        data=data,
        request=_make_request_mock(),
        user_api_key_dict=_user_api_key_auth(),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    # The strip prevents the pricing fields from ever reaching the path that
    # would mutate the global model_cost map.
    assert "test-pricing-canary-model" not in litellm.model_cost
    # And no other entries were mutated as a side effect.
    assert litellm.model_cost == snapshot
