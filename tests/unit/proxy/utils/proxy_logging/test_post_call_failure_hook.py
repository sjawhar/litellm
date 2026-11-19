"""Pin ``ProxyLogging.post_call_failure_hook``, ``_is_proxy_only_llm_api_error``,
and ``_handle_logging_proxy_only_error``."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.constants import PROXY_REJECTED_BEFORE_ROUTING_KEY
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import ProxyErrorTypes, UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.utils import ProxyLogging
from litellm.types.utils import CachingDetails, CallTypes


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


# ---------------------------------------------------------------------------
# _is_proxy_only_llm_api_error
# ---------------------------------------------------------------------------


def test_is_proxy_only_llm_api_truth_table(proxy_logging):
    """Pin the truth table of ``_is_proxy_only_llm_api_error`` in a single
    snapshot. Covers no-route, non-LLM route, HTTPException on LLM route,
    and auth-error short-circuit."""
    snapshot = {
        "no_route": proxy_logging._is_proxy_only_llm_api_error(original_exception=Exception(), route=None),
        "non_llm_route": proxy_logging._is_proxy_only_llm_api_error(
            original_exception=HTTPException(status_code=429, detail="rate"),
            route="/random/path",
        ),
        "http_on_llm_route": proxy_logging._is_proxy_only_llm_api_error(
            original_exception=HTTPException(status_code=429, detail="rate"),
            route="/chat/completions",
        ),
        "auth_short_circuit": proxy_logging._is_proxy_only_llm_api_error(
            original_exception=Exception("auth"),
            error_type=ProxyErrorTypes.auth_error,
            route="/chat/completions",
        ),
        "guardrail_raised_on_llm_route": proxy_logging._is_proxy_only_llm_api_error(
            original_exception=GuardrailRaisedException(guardrail_name="g", message="blocked"),
            route="/chat/completions",
        ),
    }
    assert snapshot == {
        "no_route": False,
        "non_llm_route": False,
        "http_on_llm_route": True,
        "auth_short_circuit": True,
        "guardrail_raised_on_llm_route": True,
    }


def test_is_proxy_only_llm_api_missing_exception_raises(proxy_logging):
    """Passing nothing should TypeError on the missing positional kwarg."""
    with pytest.raises(TypeError):
        proxy_logging._is_proxy_only_llm_api_error()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# post_call_failure_hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_failure_hook_no_callbacks_returns_none(
    proxy_logging, make_user_api_key_auth, mock_callbacks_disabled
):
    proxy_logging.alert_types = []
    request_data = {"litellm_call_id": "abc", "model": "m", "messages": []}
    out = await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=ValueError("oops"),
        user_api_key_dict=make_user_api_key_auth(),
    )
    snapshot = {
        "out_is_none": out is None,
        "litellm_logging_obj_popped": "litellm_logging_obj" not in request_data,
        "call_id_preserved": request_data["litellm_call_id"] == "abc",
        "first_api_call_start_time_present": "first_api_call_start_time" in request_data,
    }
    assert snapshot == {
        "out_is_none": True,
        "litellm_logging_obj_popped": True,
        "call_id_preserved": True,
        "first_api_call_start_time_present": False,
    }


@pytest.mark.asyncio
async def test_post_call_failure_hook_attributes_single_router_deployment(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"provider": "acme"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "internal-model", "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=403, detail="blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs["custom_llm_provider"] == "openai"
    assert kwargs["litellm_params"]["custom_llm_provider"] == "openai"
    assert kwargs["litellm_params"]["metadata"]["model_info"]["provider"] == "acme"
    assert kwargs["litellm_params"]["metadata"]["deployment"] == "openai/gpt-4.1"
    assert kwargs["litellm_params"][PROXY_REJECTED_BEFORE_ROUTING_KEY] is True
    assert kwargs["standard_logging_object"]["custom_llm_provider"] == "openai"
    assert (
        kwargs["standard_logging_object"]["model_id"] == proxy_server.llm_router.get_model_list()[0]["model_info"]["id"]
    )


@pytest.mark.asyncio
async def test_pre_routing_reject_spend_log_keeps_public_model_group(proxy_logging, make_user_api_key_auth, monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "internal-model", "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=401, detail="blocked key"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    assert recorded[0]["standard_logging_object"]["model_group"] == "internal-model"
    now: Final = datetime.now()
    payload = get_logging_payload(
        kwargs={**recorded[0], "completion_start_time": now}, response_obj=None, start_time=now, end_time=now
    )
    assert payload["model"] == "openai/gpt-4.1"
    assert payload["model_group"] == "internal-model"


@pytest.mark.asyncio
async def test_post_call_failure_hook_keeps_router_stamped_metadata_for_post_call_failures(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A post-call guardrail block arrives after the provider handoff with the router's own
    ``model_info`` in the request metadata. The pre-routing flag must stay off so deployment
    metrics keep attributing the failure to the deployment that actually served the call."""
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"id": "routed-deployment"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    request_data = {
        "litellm_call_id": "post-call-guardrail",
        "model": "internal-model",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"model_info": {"id": "routed-deployment", "served": True}},
    }
    logging_obj, request_data = litellm.utils.function_setup(
        original_function="acompletion", rules_obj=litellm.utils.Rules(), start_time=datetime.now(), **request_data
    )
    logging_obj.model_call_details["first_api_call_start_time"] = datetime.now()
    request_data["litellm_logging_obj"] = logging_obj

    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=GuardrailRaisedException(guardrail_name="g", message="response blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs["litellm_params"]["metadata"]["model_info"] == {"id": "routed-deployment", "served": True}
    assert PROXY_REJECTED_BEFORE_ROUTING_KEY not in kwargs["litellm_params"]
    assert kwargs["standard_logging_object"]["model_id"] == "routed-deployment"


@pytest.mark.asyncio
async def test_post_call_failure_hook_keeps_deployment_attribution_for_cache_hit_post_call_failures(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A post-call guardrail blocks a response served from the litellm cache. No provider call was made,
    so ``first_api_call_start_time`` is unset, but the router did pick the deployment: the pre-routing
    flag must stay off so ``litellm_deployment_failure_responses`` keeps its model_id and provider labels."""
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"id": "routed-deployment"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    request_data = {
        "litellm_call_id": "cache-hit-post-call-guardrail",
        "model": "internal-model",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"model_info": {"id": "routed-deployment"}},
    }
    logging_obj, request_data = litellm.utils.function_setup(
        original_function="acompletion", rules_obj=litellm.utils.Rules(), start_time=datetime.now(), **request_data
    )
    logging_obj.caching_details = CachingDetails(cache_hit=True, cache_duration_ms=1.0)
    request_data["litellm_logging_obj"] = logging_obj

    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=GuardrailRaisedException(guardrail_name="g", message="response blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert PROXY_REJECTED_BEFORE_ROUTING_KEY not in kwargs["litellm_params"], kwargs["litellm_params"]
    assert kwargs["standard_logging_object"]["model_id"] == "routed-deployment"
    assert kwargs["standard_logging_object"]["custom_llm_provider"] == "openai"
    assert kwargs["model"] == "internal-model"
    assert kwargs["litellm_params"]["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_post_call_failure_hook_flags_pre_routing_reject_despite_caller_model_info(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A key allowed to override pricing keeps caller-supplied ``metadata.model_info``. A reject
    before any provider handoff must still carry the pre-routing flag so deployment metrics do
    not record an outage for a deployment the request never reached."""
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"id": "real-deployment"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={
            "model": "internal-model",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"model_info": {"id": "spoofed-deployment"}},
        },
        original_exception=HTTPException(status_code=429, detail="key over limit"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs["litellm_params"][PROXY_REJECTED_BEFORE_ROUTING_KEY] is True
    assert kwargs["litellm_params"]["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_field", ["metadata", "litellm_metadata"])
async def test_pre_routing_reject_logs_the_configured_deployment_over_caller_metadata(
    proxy_logging, make_user_api_key_auth, monkeypatch, metadata_field: str
):
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []
    request_data: Final[dict[str, object]] = {
        "model": "internal-model",
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_call_id": "caller-routing-metadata",
        metadata_field: {"deployment": "openai/caller-chosen", "model_group": "caller-group"},
    }
    logging_obj, _ = litellm.utils.function_setup(
        original_function="acompletion", rules_obj=litellm.utils.Rules(), start_time=datetime.now(), **request_data
    )

    await proxy_logging.post_call_failure_hook(
        request_data={**request_data, "litellm_logging_obj": logging_obj},
        original_exception=HTTPException(status_code=400, detail="Content blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    now: Final = datetime.now()
    payload: Final = get_logging_payload(
        kwargs={**recorded[0], "completion_start_time": now}, response_obj=None, start_time=now, end_time=now
    )
    standard_logging_object: Final = recorded[0]["standard_logging_object"]
    assert {
        "logged_model": standard_logging_object["model"],
        "logged_model_group": standard_logging_object["model_group"],
        "spend_model": payload["model"],
        "spend_model_group": payload["model_group"],
    } == {
        "logged_model": "openai/gpt-4.1",
        "logged_model_group": "internal-model",
        "spend_model": "openai/gpt-4.1",
        "spend_model_group": "internal-model",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("error_type", "logged_tags"), [(ProxyErrorTypes.auth_error, []), (None, ["checkout"])])
async def test_proxy_only_failure_log_keeps_caller_tags_only_after_auth(
    proxy_logging, monkeypatch, error_type: ProxyErrorTypes | None, logged_tags: list[str]
):
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={
            "model": "internal-model",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"tags": ["checkout"]},
        },
        original_exception=HTTPException(status_code=401, detail="Invalid proxy server token passed"),
        user_api_key_dict=UserAPIKeyAuth(request_route="/chat/completions"),
        error_type=error_type,
        route="/chat/completions",
    )

    assert len(recorded) == 1
    assert recorded[0]["standard_logging_object"]["request_tags"] == logged_tags


@pytest.mark.asyncio
async def test_post_call_failure_hook_attribution_does_not_count_against_the_deployment(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """The router's failure callbacks run on this path too. A proxy-side reject must not
    bump the deployment's failure or rpm counters, or a key hitting its own limit
    could cool down the only deployment for everyone."""
    from litellm.proxy import proxy_server

    router = litellm.Router(
        model_list=[
            {
                "model_name": "internal-model",
                "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test", "rpm": 100},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    proxy_logging.alert_types = []
    deployment_id = router.get_model_list()[0]["model_info"]["id"]

    for status in (403, 429):
        await proxy_logging.post_call_failure_hook(
            request_data={"model": "internal-model", "messages": [{"role": "user", "content": "hi"}]},
            original_exception=HTTPException(status_code=status, detail="blocked"),
            user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
            route="/chat/completions",
        )
    pending = asyncio.all_tasks() - {asyncio.current_task()}
    await asyncio.gather(*pending, return_exceptions=True)

    deployment_keys = [key for key in router.cache.in_memory_cache.cache_dict if deployment_id in key]
    assert deployment_keys == [], f"proxy reject was counted against the deployment: {deployment_keys}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("original_exception", "error_type", "authenticated", "past_pre_call"),
    (
        (ProxyRateLimitError(detail="key over its rate limit"), None, True, False),
        (HTTPException(status_code=401, detail="Invalid virtual key"), ProxyErrorTypes.auth_error, False, False),
        (ProxyRateLimitError(detail="key over its rate limit"), None, True, True),
    ),
    ids=("rate-limited-key", "invalid-key", "rate-limited-key-with-logging-object"),
)
async def test_proxy_only_failure_does_not_count_or_cool_the_deployment_the_body_names(
    proxy_logging,
    make_user_api_key_auth,
    monkeypatch,
    original_exception: Exception,
    error_type: ProxyErrorTypes | None,
    authenticated: bool,
    past_pre_call: bool,
):
    """``model_info`` and ``cooldown_time`` are logged litellm_params names, and the router's failure callbacks
    take the deployment and the cooldown from them. A body naming a real deployment must not make a key's own
    429, or the 401 of a caller with no valid key, count against that deployment or cool it down. A key refused
    in the pre-call hooks already has the logging object the proxy built from that body, and it holds both."""
    from litellm.proxy import proxy_server
    from litellm.router_utils.router_callbacks.track_deployment_metrics import (
        get_deployment_failures_for_current_minute,
    )

    named_deployment: Final = "body-named-deployment"
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "internal-model",
                "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                "model_info": {"id": named_deployment},
            },
            {
                "model_name": "internal-model",
                "litellm_params": {"model": "openai/gpt-4.1-mini", "api_key": "sk-test"},
                "model_info": {"id": "sibling-deployment"},
            },
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    proxy_logging.alert_types = []
    started: list[threading.Thread] = []

    class _JoinableThread(threading.Thread):
        def start(self) -> None:
            started.append(self)
            super().start()

    monkeypatch.setattr("litellm.proxy.utils.threading", SimpleNamespace(Thread=_JoinableThread))

    body: Final[dict[str, object]] = {
        "model": "internal-model",
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_call_id": "body-named-deployment-call",
        "metadata": {"model_group": "internal-model"},
        "model_info": {"id": named_deployment},
        "cooldown_time": 86400,
    }
    request_data: Final[dict[str, object]] = (
        {
            **body,
            "litellm_logging_obj": litellm.utils.function_setup(
                original_function="acompletion", rules_obj=litellm.utils.Rules(), start_time=datetime.now(), **body
            )[0],
        }
        if past_pre_call
        else dict(body)
    )

    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=original_exception,
        user_api_key_dict=(
            make_user_api_key_auth(request_route="/chat/completions")
            if authenticated
            else UserAPIKeyAuth(request_route="/chat/completions")
        ),
        error_type=error_type,
        route="/chat/completions",
    )
    for thread in started:
        thread.join(timeout=5)
    pending = asyncio.all_tasks() - {asyncio.current_task()}
    await asyncio.gather(*pending, return_exceptions=True)

    assert {
        "sync_failure_handlers": len(started),
        "failures": get_deployment_failures_for_current_minute(
            litellm_router_instance=router, deployment_id=named_deployment
        ),
        "cooldowns": router.cooldown_cache.get_active_cooldowns(model_ids=[named_deployment], parent_otel_span=None),
        "counters": [key for key in router.cache.in_memory_cache.cache_dict if named_deployment in key],
    } == {"sync_failure_handlers": 1, "failures": 0, "cooldowns": [], "counters": []}


@pytest.mark.asyncio
async def test_post_call_failure_hook_attributes_the_keys_team_deployment_over_the_global_group(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A team key requesting its team public model name must be attributed to the team's
    deployment, not to a global group that happens to share the public name."""
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "shared-name",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"id": "global-deployment"},
                },
                {
                    "model_name": "shared-name_test-team_deadbeef",
                    "litellm_params": {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-test"},
                    "model_info": {
                        "id": "team-deployment",
                        "team_id": "test-team",
                        "team_public_model_name": "shared-name",
                    },
                },
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "shared-name", "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=429, detail="rate limited"),
        user_api_key_dict=make_user_api_key_auth(team_id="test-team", request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs["custom_llm_provider"] == "anthropic"
    assert kwargs["litellm_params"]["metadata"]["deployment"] == "anthropic/claude-sonnet-4-5"
    assert kwargs["standard_logging_object"]["model_id"] == "team-deployment"


@pytest.mark.asyncio
async def test_post_call_failure_hook_omits_provider_for_mixed_router_deployments(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                },
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-test"},
                },
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "internal-model", "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=403, detail="blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs.get("custom_llm_provider") is None
    assert "model_info" not in (kwargs["litellm_params"].get("metadata") or {})


@pytest.mark.asyncio
async def test_post_call_failure_hook_omits_provider_when_a_deployment_does_not_resolve(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """One deployment resolves to openai and its sibling resolves to nothing: the group
    is not known to be single-provider, so no provider is stamped on the failure."""
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    router = MagicMock()
    router.get_model_list.return_value = [
        {"model_name": "internal-model", "litellm_params": {"model": "openai/gpt-4.1"}},
        {"model_name": "internal-model", "litellm_params": {"model": "unmapped-model-with-no-provider"}},
    ]
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "internal-model", "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=403, detail="blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs.get("custom_llm_provider") is None
    assert kwargs["litellm_params"].get("custom_llm_provider") is None


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_attributes_with_read_only_metadata(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """With a logging object already on the request, allowlisted metadata is kept,
    a read-only mapping must not crash the stamp, and the failure handler receives provider attribution."""
    from litellm.proxy import proxy_server

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"provider": "acme"},
                }
            ]
        ),
    )
    logging_obj = MagicMock()
    logging_obj.call_type = "acompletion"
    logging_obj.model_call_details = {}
    logging_obj.litellm_params = {"metadata": {"preexisting": "logged"}}
    logging_obj.async_failure_handler = AsyncMock()
    guardrail_info: Final = [
        {
            "guardrail_name": "local-block",
            "guardrail_status": "guardrail_intervened",
            "guardrail_cost": 0.25,
        }
    ]

    await proxy_logging._handle_logging_proxy_only_error(
        request_data={
            "litellm_logging_obj": logging_obj,
            "model": "internal-model",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": MappingProxyType(
                {"tags": ["checkout"], "standard_logging_guardrail_information": guardrail_info}
            ),
        },
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions", key_alias="frozen"),
        route="/chat/completions",
        original_exception=HTTPException(status_code=403, detail="blocked"),
    )

    assert logging_obj.async_failure_handler.called
    update_kwargs = logging_obj.update_environment_variables.call_args.kwargs
    assert update_kwargs["custom_llm_provider"] == "openai"
    assert update_kwargs["litellm_params"]["custom_llm_provider"] == "openai"
    assert {
        key: update_kwargs["litellm_params"]["metadata"][key] for key in ("preexisting", "user_api_key_alias")
    } == {"preexisting": "logged", "user_api_key_alias": "frozen"}
    assert update_kwargs["litellm_params"]["metadata"]["tags"] == ["checkout"]
    assert update_kwargs["litellm_params"]["metadata"]["standard_logging_guardrail_information"] == guardrail_info


@pytest.mark.asyncio
async def test_post_call_failure_hook_fires_without_router_attribution(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "different-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": "internal-model", "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=403, detail="blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs.get("custom_llm_provider") is None
    assert "model_info" not in (kwargs["litellm_params"].get("metadata") or {})


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [123, ["internal-model"], {"name": "internal-model"}, None])
async def test_post_call_failure_hook_fires_for_non_string_model(
    proxy_logging, make_user_api_key_auth, monkeypatch, model: object
):
    """A body whose ``model`` is not a string is rejected by the proxy before routing; its
    failure callback must still fire, unattributed, instead of a TypeError escaping the hook."""
    from litellm.proxy import proxy_server

    recorded: list[dict] = []

    class _RecordingLogger(CustomLogger):
        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            recorded.append(kwargs)

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                }
            ]
        ),
    )
    monkeypatch.setattr(litellm, "callbacks", [_RecordingLogger()])
    proxy_logging.alert_types = []

    await proxy_logging.post_call_failure_hook(
        request_data={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        original_exception=HTTPException(status_code=400, detail="'model' must be a string."),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        route="/chat/completions",
    )

    assert len(recorded) == 1
    kwargs = recorded[0]
    assert kwargs.get("custom_llm_provider") is None
    assert "model_info" not in (kwargs["litellm_params"].get("metadata") or {})


@pytest.mark.asyncio
async def test_post_call_failure_hook_callback_returns_http_exception(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    transformed = HTTPException(status_code=418, detail="teapot")

    class _Cb(CustomLogger):
        async def async_post_call_failure_hook(self, **kwargs):  # type: ignore[override]
            return transformed

    monkeypatch.setattr(litellm, "callbacks", [_Cb()])
    proxy_logging.alert_types = []
    out = await proxy_logging.post_call_failure_hook(
        request_data={"litellm_call_id": "abc"},
        original_exception=ValueError("oops"),
        user_api_key_dict=make_user_api_key_auth(),
    )
    assert out is transformed


@pytest.mark.asyncio
async def test_post_call_failure_hook_callback_raises_http_exception_first_wins(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    err = HTTPException(status_code=418, detail="raised teapot")

    class _Cb(CustomLogger):
        async def async_post_call_failure_hook(self, **kwargs):  # type: ignore[override]
            raise err

    monkeypatch.setattr(litellm, "callbacks", [_Cb()])
    proxy_logging.alert_types = []
    out = await proxy_logging.post_call_failure_hook(
        request_data={"litellm_call_id": "abc"},
        original_exception=ValueError("oops"),
        user_api_key_dict=make_user_api_key_auth(),
    )
    assert out is err


@pytest.mark.asyncio
async def test_post_call_failure_hook_non_http_exception_in_callback_swallowed(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    class _Cb(CustomLogger):
        async def async_post_call_failure_hook(self, **kwargs):  # type: ignore[override]
            raise RuntimeError("non-http inside cb")

    monkeypatch.setattr(litellm, "callbacks", [_Cb()])
    proxy_logging.alert_types = []
    out = await proxy_logging.post_call_failure_hook(
        request_data={"litellm_call_id": "abc"},
        original_exception=ValueError("oops"),
        user_api_key_dict=make_user_api_key_auth(),
    )
    assert out is None


@pytest.mark.asyncio
@pytest.mark.parametrize("logging_value", (None, "caller-controlled", {"baseline_cache_context": "untrusted"}))
async def test_terminal_baseline_cleanup_ignores_missing_or_untrusted_logging(
    proxy_logging: ProxyLogging, monkeypatch: pytest.MonkeyPatch, logging_value: object
) -> None:
    monkeypatch.setattr(litellm, "callbacks", ())
    proxy_logging.alert_types = []  # rebind-ok: isolate the fixture-owned alert configuration
    request_data: Final = {"litellm_call_id": "untrusted-logging", "litellm_logging_obj": logging_value}
    result: Final = await proxy_logging.post_call_failure_hook(  # pyright: ignore[reportUnknownMemberType]  # exercise the existing proxy terminal owner with its legacy request dictionary contract
        request_data=request_data,
        original_exception=ValueError("original provider failure"),
        user_api_key_dict=UserAPIKeyAuth(request_route="/v1/messages"),
    )
    assert result is None
    assert "litellm_logging_obj" not in request_data


# ---------------------------------------------------------------------------
# _handle_logging_proxy_only_error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_uses_existing_logging_obj(proxy_logging, make_user_api_key_auth):
    logging_obj = MagicMock()
    logging_obj.call_type = "acompletion"
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock()

    request_data = {
        "litellm_logging_obj": logging_obj,
        "messages": [{"role": "user", "content": "x"}],
        "model": "m",
        "metadata": {},
    }
    await proxy_logging._handle_logging_proxy_only_error(
        request_data=request_data,
        user_api_key_dict=make_user_api_key_auth(),
        route="/chat/completions",
        original_exception=HTTPException(status_code=429, detail="rate"),
    )
    from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL

    snapshot = {
        "input_logged": "messages" in logging_obj.model_call_details,
        "call_type_normalized": logging_obj.call_type,
        "marker_present": logging_obj.model_call_details.get(LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL) is True,
        "async_failure_called": logging_obj.async_failure_handler.called,
    }
    assert snapshot == {
        "input_logged": True,
        "call_type_normalized": "acompletion",
        "marker_present": True,
        "async_failure_called": True,
    }


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_skips_for_pass_through(proxy_logging, make_user_api_key_auth):
    from litellm.types.utils import CallTypes

    logging_obj = MagicMock()
    logging_obj.call_type = CallTypes.pass_through.value
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock()
    logging_obj.pre_call = MagicMock()
    route_litellm_params = {"api_base": "https://upstream.invalid", "metadata": {"model_info": {"id": "routed"}}}
    request_data = {
        "litellm_logging_obj": logging_obj,
        "messages": [{"role": "user"}],
        "model": "m",
        "litellm_params": route_litellm_params,
    }
    await proxy_logging._handle_logging_proxy_only_error(
        request_data=request_data,
        user_api_key_dict=make_user_api_key_auth(),
        route="/chat/completions",
        original_exception=HTTPException(status_code=429, detail="rate"),
    )
    logging_obj.pre_call.assert_not_called()
    logging_obj.async_failure_handler.assert_not_called()
    assert request_data["litellm_params"] is route_litellm_params


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_no_logging_obj_creates_one(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    fake_logging_obj = MagicMock()
    fake_logging_obj.call_type = "acompletion"
    fake_logging_obj.model_call_details = {}
    fake_logging_obj.async_failure_handler = AsyncMock()

    def fake_function_setup(**kwargs):
        return fake_logging_obj, {}

    monkeypatch.setattr(litellm.utils, "function_setup", fake_function_setup)
    request_data = {"messages": [{"role": "user"}], "model": "m"}
    await proxy_logging._handle_logging_proxy_only_error(
        request_data=request_data,
        user_api_key_dict=make_user_api_key_auth(),
        route="/chat/completions",
        original_exception=HTTPException(status_code=429, detail="rate"),
    )
    assert "litellm_call_id" in request_data
    fake_logging_obj.async_failure_handler.assert_called_once()


class _FailureEventRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged_litellm_params: list[dict] = []
        self.logged_payload_metadata: list[dict] = []

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.logged_litellm_params.append(kwargs["litellm_params"])
        self.logged_payload_metadata.append((kwargs.get("standard_logging_object") or {}).get("metadata") or {})


@pytest.mark.asyncio
async def test_auth_gate_failure_logs_the_authenticated_identity_instead_of_caller_user_api_key_fields(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    recorder = _FailureEventRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    forged = {
        "user_api_key": "f" * 64,
        "user_api_key_team_id": "forged-team",
        "user_api_key_team_metadata": {"priority": "forged"},
        "user_api_key_metadata": {"priority": "forged"},
    }

    authenticated = make_user_api_key_auth(
        team_metadata={"priority": "high", "callback_settings": {"success_callback": ["langfuse"]}}
    )
    await proxy_logging._handle_logging_proxy_only_error(
        request_data={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {**forged, "department": "engineering"},
            "litellm_metadata": {**forged, "trace_id": "trace-1"},
        },
        user_api_key_dict=authenticated,
        route="/chat/completions",
        original_exception=litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0),
    )

    assert len(recorder.logged_litellm_params) == 1
    logged_metadata = recorder.logged_litellm_params[0]["metadata"]
    logged_litellm_metadata = recorder.logged_litellm_params[0]["litellm_metadata"]
    assert {
        "metadata": {key: logged_metadata.get(key) for key in ("department", *forged)},
        "litellm_metadata": {key: logged_litellm_metadata.get(key) for key in ("trace_id", *forged)},
        "logged_key_hash": recorder.logged_payload_metadata[0].get("user_api_key_hash"),
    } == {
        "metadata": {
            "department": None,
            "user_api_key": None,
            "user_api_key_team_id": authenticated.team_id,
            "user_api_key_team_metadata": {"priority": "high"},
            "user_api_key_metadata": None,
        },
        "litellm_metadata": {
            "trace_id": None,
            "user_api_key": None,
            "user_api_key_team_id": authenticated.team_id,
            "user_api_key_team_metadata": None,
            "user_api_key_metadata": None,
        },
        "logged_key_hash": authenticated.api_key,
    }


@pytest.mark.asyncio
async def test_failure_after_add_litellm_data_to_request_keeps_the_fields_the_proxy_stamped(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    recorder = _FailureEventRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    authenticated = make_user_api_key_auth()
    stamped = {
        "user_api_key_auth": authenticated,
        "user_api_key_model_max_budget": {"gpt-4o": 5.0},
        "user_api_key_budget_reservation": {"id": "reservation-1"},
        "user_api_key_metadata": {"owner": "platform"},
    }

    await proxy_logging._handle_logging_proxy_only_error(
        request_data={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": dict(stamped),
        },
        user_api_key_dict=authenticated,
        route="/assistants",
        original_exception=litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0),
    )

    assert len(recorder.logged_litellm_params) == 1
    logged_metadata = recorder.logged_litellm_params[0]["metadata"]
    assert {key: logged_metadata.get(key) for key in (*stamped, "user_api_key_team_id")} == {
        **stamped,
        "user_api_key_team_id": authenticated.team_id,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forged_auth",
    ({"team_id": "evil"}, "UserAPIKeyAuth", [{"team_id": "evil"}], None),
    ids=("dict", "string", "list", "null"),
)
async def test_a_body_shaped_like_the_proxy_stamp_does_not_buy_the_caller_its_identity(
    proxy_logging, make_user_api_key_auth, monkeypatch, forged_auth
):
    recorder = _FailureEventRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    authenticated = make_user_api_key_auth()

    await proxy_logging._handle_logging_proxy_only_error(
        request_data={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {
                "user_api_key_auth": forged_auth,
                "user_api_key": "f" * 64,
                "user_api_key_team_id": "forged-team",
            },
        },
        user_api_key_dict=authenticated,
        route="/chat/completions",
        original_exception=litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0),
    )

    assert len(recorder.logged_litellm_params) == 1
    logged_metadata = recorder.logged_litellm_params[0]["metadata"]
    assert {
        "user_api_key_auth": logged_metadata.get("user_api_key_auth"),
        "user_api_key": logged_metadata.get("user_api_key"),
        "user_api_key_team_id": logged_metadata.get("user_api_key_team_id"),
        "logged_key_hash": recorder.logged_payload_metadata[0].get("user_api_key_hash"),
    } == {
        "user_api_key_auth": None,
        "user_api_key": None,
        "user_api_key_team_id": authenticated.team_id,
        "logged_key_hash": authenticated.api_key,
    }


@pytest.mark.asyncio
async def test_auth_gate_failure_drops_the_team_the_caller_named(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """The ClickHouse spend row, the Datadog loggers, New Relic and the OTel span model read
    team_id and team_alias as the team whenever the authenticated key has none, so a rejected
    caller's own values must not reach a logger."""
    recorder = _FailureEventRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    forged = {"team_id": "victim-team", "team_alias": "victim-team-alias"}
    authenticated = make_user_api_key_auth(team_id=None, team_alias=None)

    await proxy_logging._handle_logging_proxy_only_error(
        request_data={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {**forged, "department": "engineering"},
            "litellm_metadata": {**forged, "trace_id": "trace-1"},
        },
        user_api_key_dict=authenticated,
        route="/chat/completions",
        original_exception=litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0),
    )

    assert len(recorder.logged_litellm_params) == 1
    logged_metadata = recorder.logged_litellm_params[0]["metadata"]
    logged_litellm_metadata = recorder.logged_litellm_params[0]["litellm_metadata"]
    assert {
        "metadata": {key: logged_metadata.get(key) for key in ("department", *forged)},
        "litellm_metadata": {key: logged_litellm_metadata.get(key) for key in ("trace_id", *forged)},
        "payload": {key: recorder.logged_payload_metadata[0].get(key) for key in forged},
    } == {
        "metadata": dict.fromkeys(("department", *forged)),
        "litellm_metadata": dict.fromkeys(("trace_id", *forged)),
        "payload": dict.fromkeys(forged),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("bucket", ("metadata", "litellm_metadata"))
async def test_auth_gate_failure_ignores_the_identity_a_caller_nests_in_litellm_params(
    proxy_logging, make_user_api_key_auth, monkeypatch, bucket: str
):
    """A body key named litellm_params replaced the litellm_params the logging object built, and the
    standard logging payload reads identity from both of its metadata buckets, so a rejected caller
    could name the key, team and user of its own failure there."""
    recorder = _FailureEventRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    forged = {
        "user_api_key_hash": "f" * 64,
        "user_api_key_team_id": "victim-team",
        "user_api_key_user_id": "victim-user",
        "team_id": "victim-team",
    }
    authenticated = make_user_api_key_auth()

    await proxy_logging._handle_logging_proxy_only_error(
        request_data={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "litellm_params": {bucket: dict(forged)},
        },
        user_api_key_dict=authenticated,
        route="/chat/completions",
        original_exception=litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0),
    )

    assert len(recorder.logged_litellm_params) == 1
    logged_litellm_params = recorder.logged_litellm_params[0]
    authenticated_identity = {
        "user_api_key_hash": authenticated.api_key,
        "user_api_key_team_id": authenticated.team_id,
        "user_api_key_user_id": authenticated.user_id,
        "team_id": None,
    }
    assert {
        "metadata": {key: (logged_litellm_params.get("metadata") or {}).get(key) for key in forged},
        "litellm_metadata": {key: (logged_litellm_params.get("litellm_metadata") or {}).get(key) for key in forged},
        "payload": {key: recorder.logged_payload_metadata[0].get(key) for key in forged},
    } == {
        "metadata": authenticated_identity,
        "litellm_metadata": dict.fromkeys(forged),
        "payload": authenticated_identity,
    }


@pytest.mark.asyncio
async def test_failure_after_add_litellm_data_to_request_keeps_the_stamp_over_a_caller_litellm_params(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    recorder = _FailureEventRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    authenticated = make_user_api_key_auth()
    stamped = {
        "user_api_key_auth": authenticated,
        "user_api_key_metadata": {"owner": "platform"},
    }

    await proxy_logging._handle_logging_proxy_only_error(
        request_data={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": dict(stamped),
            "litellm_params": {
                "metadata": {"user_api_key_team_id": "victim-team", "user_api_key_metadata": {"owner": "victim"}}
            },
        },
        user_api_key_dict=authenticated,
        route="/chat/completions",
        original_exception=litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0),
    )

    assert len(recorder.logged_litellm_params) == 1
    logged_metadata = recorder.logged_litellm_params[0]["metadata"]
    assert {
        "metadata": {key: logged_metadata.get(key) for key in (*stamped, "user_api_key_team_id")},
        "payload_team_id": recorder.logged_payload_metadata[0].get("user_api_key_team_id"),
    } == {
        "metadata": {**stamped, "user_api_key_team_id": authenticated.team_id},
        "payload_team_id": authenticated.team_id,
    }


def _record_failure_spend_rows(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Runs the failure spend logger in the callback loop and captures the rows it would write."""
    from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger

    update_database: Final = AsyncMock()
    monkeypatch.setattr("litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter.update_database", update_database)
    monkeypatch.setattr(litellm, "callbacks", [ProxyDBLogger()])
    return update_database


@pytest.mark.asyncio
@pytest.mark.parametrize("bucket", ("metadata", "litellm_metadata"))
async def test_auth_gate_failure_spend_row_bills_no_guardrail_the_body_names(
    proxy_logging, monkeypatch, bucket: str
):
    """An auth-gate rejection never reaches add_litellm_data_to_request, which strips a caller's guardrail record,
    and no guardrail ran on it. The failure spend row bills the guardrail cost it finds in either bucket, so the
    body could charge any amount to the key, user, team and proxy budget the row is filed under."""
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    update_database: Final = _record_failure_spend_rows(monkeypatch)
    proxy_logging.alert_types = []
    forged_record: Final = [
        {"guardrail_name": "never-ran", "guardrail_status": "guardrail_intervened", "guardrail_cost": 25.0}
    ]

    await proxy_logging.post_call_failure_hook(
        request_data={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            bucket: {"standard_logging_guardrail_information": forged_record},
        },
        original_exception=HTTPException(status_code=401, detail="Invalid virtual key"),
        user_api_key_dict=UserAPIKeyAuth(request_route="/chat/completions"),
        error_type=ProxyErrorTypes.auth_error,
        route="/chat/completions",
    )

    update_database.assert_awaited_once()
    row: Final = update_database.call_args.kwargs
    row_metadata: Final = row["kwargs"]["litellm_params"]["metadata"]
    now: Final = datetime.now()
    payload: Final = get_logging_payload(
        kwargs={**row["kwargs"], "completion_start_time": now}, response_obj=None, start_time=now, end_time=now
    )
    assert {
        "spend": row["response_cost"],
        "record_in_row_metadata": "standard_logging_guardrail_information" in row_metadata,
        "row_guardrail_information": json.loads(payload["metadata"]).get("guardrail_information"),
    } == {"spend": 0.0, "record_in_row_metadata": False, "row_guardrail_information": None}


@pytest.mark.asyncio
async def test_guardrail_refusal_past_pre_call_keeps_the_guardrail_cost_on_the_spend_row(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A guardrail that ran in the pre-call hooks wrote its record after add_litellm_data_to_request stripped the
    caller's and stamped the request, so a refusal it raised still bills what it cost."""
    update_database: Final = _record_failure_spend_rows(monkeypatch)
    proxy_logging.alert_types = []
    authenticated: Final = make_user_api_key_auth(request_route="/chat/completions")
    request_data: Final[dict[str, object]] = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_call_id": "guardrail-refusal",
        "metadata": {"user_api_key_auth": authenticated},
    }
    logging_obj, _ = litellm.utils.function_setup(
        original_function="acompletion", rules_obj=litellm.utils.Rules(), start_time=datetime.now(), **request_data
    )
    request_data["metadata"]["standard_logging_guardrail_information"] = [
        {"guardrail_name": "local-block", "guardrail_status": "guardrail_intervened", "guardrail_cost": 0.25}
    ]

    await proxy_logging.post_call_failure_hook(
        request_data={**request_data, "litellm_logging_obj": logging_obj},
        original_exception=GuardrailRaisedException(guardrail_name="local-block", message="blocked"),
        user_api_key_dict=authenticated,
        route="/chat/completions",
    )

    update_database.assert_awaited_once()
    assert update_database.call_args.kwargs["response_cost"] == pytest.approx(0.25)


_AUTH_GATE_BODY_CLAIMS: Final = (
    (
        "root",
        "litellm_params",
        {"api_base": "https://victim.invalid", "litellm_metadata": {"model_info": {"id": "victim-deployment"}}},
    ),
    ("root", "model_info", {"id": "victim-deployment"}),
    ("root", "custom_llm_provider", "victim-provider"),
    ("root", "agent_id", "victim-agent"),
    (
        "root",
        "response_headers",
        {"x-ms-is-spilled-over": "true", "x-ms-spillover-from-deployment": "victim-deployment"},
    ),
    ("root", "proxy_server_request", {"url": "https://victim.invalid", "body": {"user": "victim-customer"}}),
    ("metadata", "model_info", {"id": "victim-deployment"}),
    ("metadata", "model_group", "victim-group"),
    ("metadata", "deployment", "openai/victim-model"),
    ("metadata", "original_model_group", "victim-group"),
    ("metadata", "routing_decision", {"router_model_name": "victim-router", "routed_model": "victim-model"}),
    ("metadata", "agent_id", "victim-agent"),
    ("metadata", "actor_agent_id", "victim-agent"),
    ("metadata", "target_agent_id", "victim-agent"),
    ("metadata", "billing_agent_id", "victim-agent"),
    ("metadata", "agent_execution_mode", "victim-mode"),
    ("metadata", "verified_human_user_id", "victim-user"),
    ("metadata", "internal_call_origin", "victim-origin"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("place", "field", "claim"),
    _AUTH_GATE_BODY_CLAIMS,
    ids=[f"{place}.{field}" for place, field, _ in _AUTH_GATE_BODY_CLAIMS],
)
async def test_auth_gate_failure_spend_row_takes_its_deployment_from_the_router_not_the_body(
    proxy_logging, monkeypatch, place: str, field: str, claim: object
):
    """Nobody is authenticated at the auth gate, so nothing the body says about the provider, deployment, group,
    agent or origin, at the top level or in either metadata bucket, may reach the failure row it is filed under."""
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"id": "configured-deployment"},
                }
            ]
        ),
    )
    update_database: Final = _record_failure_spend_rows(monkeypatch)
    proxy_logging.alert_types = []
    claimed: Final = (
        {field: claim} if place == "root" else {bucket: {field: claim} for bucket in ("metadata", "litellm_metadata")}
    )

    await proxy_logging.post_call_failure_hook(
        request_data={
            "model": "internal-model",
            "messages": [{"role": "user", "content": "hi"}],
            **claimed,
        },
        original_exception=HTTPException(status_code=401, detail="Invalid virtual key"),
        user_api_key_dict=UserAPIKeyAuth(request_route="/chat/completions"),
        error_type=ProxyErrorTypes.auth_error,
        route="/chat/completions",
    )

    update_database.assert_awaited_once()
    row_kwargs: Final = update_database.call_args.kwargs["kwargs"]
    now: Final = datetime.now()
    payload: Final = get_logging_payload(
        kwargs={**row_kwargs, "completion_start_time": now}, response_obj=None, start_time=now, end_time=now
    )
    attribution_columns: Final = ("model_id", "model_group", "model", "api_base", "custom_llm_provider", "agent_id")
    assert {
        "attribution": {key: payload[key] for key in attribution_columns},
        "names_the_claim": "victim" in json.dumps({"row": payload, "request_data": row_kwargs}, default=str),
    } == {
        "attribution": {
            "model_id": "configured-deployment",
            "model_group": "internal-model",
            "model": "openai/gpt-4.1",
            "api_base": "",
            "custom_llm_provider": "openai",
            "agent_id": None,
        },
        "names_the_claim": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original_exception",
    (
        GuardrailRaisedException(guardrail_name="local-block", message="blocked"),
        ProxyRateLimitError(detail="key over its rate limit"),
    ),
    ids=("guardrail", "rate-limited-key"),
)
async def test_pre_call_refusal_spend_row_takes_no_deployment_from_a_body_litellm_params(
    proxy_logging, make_user_api_key_auth, monkeypatch, original_exception: Exception
):
    """A key refused in the pre-call hooks reached no deployment, so a root litellm_params in its body must not
    name the API base, deployment or group of its failure row; the spend logger prefers it over the logged payload."""
    from litellm.proxy import proxy_server
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    monkeypatch.setattr(
        proxy_server,
        "llm_router",
        litellm.Router(
            model_list=[
                {
                    "model_name": "internal-model",
                    "litellm_params": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
                    "model_info": {"id": "configured-deployment"},
                }
            ]
        ),
    )
    update_database: Final = _record_failure_spend_rows(monkeypatch)
    proxy_logging.alert_types = []
    authenticated: Final = make_user_api_key_auth(request_route="/chat/completions")
    request_data: Final[dict[str, object]] = {
        "model": "internal-model",
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_call_id": "pre-call-refusal",
        "metadata": {"user_api_key_auth": authenticated},
        "litellm_params": {
            "api_base": "https://victim.invalid",
            "litellm_metadata": {"model_info": {"id": "victim-deployment"}, "model_group": "victim-group"},
        },
    }
    logging_obj, _ = litellm.utils.function_setup(
        original_function="acompletion", rules_obj=litellm.utils.Rules(), start_time=datetime.now(), **request_data
    )

    await proxy_logging.post_call_failure_hook(
        request_data={**request_data, "litellm_logging_obj": logging_obj},
        original_exception=original_exception,
        user_api_key_dict=authenticated,
        route="/chat/completions",
    )

    update_database.assert_awaited_once()
    row_kwargs: Final = update_database.call_args.kwargs["kwargs"]
    now: Final = datetime.now()
    payload: Final = get_logging_payload(
        kwargs={**row_kwargs, "completion_start_time": now}, response_obj=None, start_time=now, end_time=now
    )
    assert {
        "attribution": {key: payload[key] for key in ("model_id", "model_group", "api_base")},
        "names_the_claim": "victim" in json.dumps({"row": payload, "request_data": row_kwargs}, default=str),
    } == {
        "attribution": {"model_id": "configured-deployment", "model_group": "internal-model", "api_base": ""},
        "names_the_claim": False,
    }


_LOGGING_OBJECT_FIELDS: Final = (
    "litellm_params",
    "optional_params",
    "start_time",
    "call_type",
    "completion_start_time",
    "standard_callback_dynamic_params",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type, passed_as_optional_params",
    (("acompletion", False), (CallTypes.pass_through.value, True)),
    ids=("body", "pass-through"),
)
async def test_handle_logging_proxy_only_path_keeps_body_keys_off_the_logging_object_fields(
    proxy_logging, make_user_api_key_auth, call_type: str, passed_as_optional_params: bool
):
    """update_environment_variables spreads optional_params over the fields it writes from the logging
    object, so a body key of one of those names must not get there. A pass-through route's request_data
    carries the litellm_params and call_type the route built itself, and keeps them."""
    logging_obj = MagicMock()
    logging_obj.call_type = call_type
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock()

    await proxy_logging._handle_logging_proxy_only_error(
        request_data={
            "litellm_logging_obj": logging_obj,
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            **{name: {"from": "request_data"} for name in _LOGGING_OBJECT_FIELDS},
        },
        user_api_key_dict=make_user_api_key_auth(),
        route="/chat/completions",
        original_exception=HTTPException(status_code=429, detail="rate"),
    )

    optional_params = logging_obj.update_environment_variables.call_args.kwargs["optional_params"]
    assert {name: name in optional_params for name in _LOGGING_OBJECT_FIELDS} == dict.fromkeys(
        _LOGGING_OBJECT_FIELDS, passed_as_optional_params
    )


@pytest.mark.asyncio
async def test_handle_logging_proxy_only_path_propagates_async_failure_raises(proxy_logging, make_user_api_key_auth):
    logging_obj = MagicMock()
    logging_obj.call_type = "acompletion"
    logging_obj.model_call_details = {}
    logging_obj.async_failure_handler = AsyncMock(side_effect=RuntimeError("boom"))
    request_data = {
        "litellm_logging_obj": logging_obj,
        "messages": [{"role": "user"}],
        "model": "m",
    }
    with pytest.raises(RuntimeError):
        await proxy_logging._handle_logging_proxy_only_error(
            request_data=request_data,
            user_api_key_dict=make_user_api_key_auth(),
            route="/chat/completions",
            original_exception=Exception("x"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route, request_data, expected_call_type",
    [
        ("/v1/chat/completions", {}, "acompletion"),
        ("/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, "acompletion"),
        ("/v1/messages", {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, "anthropic_messages"),
        ("/v1/responses", {"model": "m", "input": "hi"}, "aresponses"),
        ("/v1/embeddings", {"model": "m", "input": ["hi"]}, "aembedding"),
        ("/model/info", {}, "/model/info"),
    ],
)
async def test_post_call_failure_hook_lifts_route_call_type_for_gate_rejections(
    proxy_logging, make_user_api_key_auth, route, request_data, expected_call_type
):
    """Regression for LIT-5884: the matched route, not the body shape, sets the
    spend-log call_type for requests rejected before dispatch."""
    proxy_logging.alert_types = []
    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=Exception("Authentication Error, No api key passed in."),
        user_api_key_dict=make_user_api_key_auth(request_route=route),
        error_type=ProxyErrorTypes.auth_error,
        route=route,
    )
    assert request_data["call_type"] == expected_call_type
    assert "start_time" in request_data


@pytest.mark.asyncio
async def test_post_call_failure_hook_falls_back_to_body_shape_without_a_route(proxy_logging, make_user_api_key_auth):
    proxy_logging.alert_types = []
    request_data = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=Exception("Authentication Error, No api key passed in."),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
        error_type=ProxyErrorTypes.auth_error,
    )
    assert request_data["call_type"] == "acompletion"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/v1/files", "/files/file-abc", "/v1/containers"])
async def test_post_call_failure_hook_keeps_the_route_for_multi_operation_routes(
    proxy_logging, make_user_api_key_auth, route
):
    """Routes shared by several operations (POST create vs GET list) cannot be attributed without the
    method, so a rejected request there is filed under its route, not under whichever operation the
    mapping lists first."""
    proxy_logging.alert_types = []
    request_data: dict = {}
    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=Exception("Authentication Error, No api key passed in."),
        user_api_key_dict=make_user_api_key_auth(request_route=route),
        error_type=ProxyErrorTypes.auth_error,
        route=route,
    )
    assert request_data["call_type"] == route


@pytest.mark.asyncio
async def test_post_call_failure_hook_guardrail_block_fires_failure_callback(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A ``GuardrailRaisedException`` on an LLM route must reach the logging
    object's ``async_failure_handler`` so custom loggers see a ``failure``
    status - without this, guardrail blocks produce only
    ``post_call_failure_hook`` and no failure logging event."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    recorded: list[object] = []

    class _StatusRecorder(CustomLogger):
        async def async_log_failure_event(
            self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
        ) -> None:
            standard_logging_object = kwargs.get("standard_logging_object")
            recorded.append(standard_logging_object.get("status") if isinstance(standard_logging_object, dict) else None)

    monkeypatch.setattr(litellm, "_async_failure_callback", [_StatusRecorder()])
    logging_obj = LiteLLMLoggingObj(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="test_guardrail_block_failure_cb",
        function_id="test_guardrail_block_failure_cb",
    )
    request_data = {
        "litellm_logging_obj": logging_obj,
        "litellm_call_id": "test_guardrail_block_failure_cb",
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {},
    }
    proxy_logging.alert_types = []
    await proxy_logging.post_call_failure_hook(
        request_data=request_data,
        original_exception=GuardrailRaisedException(guardrail_name="g", message="blocked"),
        user_api_key_dict=make_user_api_key_auth(request_route="/chat/completions"),
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert recorded == ["failure"]
