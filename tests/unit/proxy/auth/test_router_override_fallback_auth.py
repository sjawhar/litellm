"""
VERIA-44: ``router_settings_override.fallbacks`` must be validated
against the API key's model allowlist at auth time. Without this, the
override is promoted to per-request kwargs after auth and lets a caller
execute requests against models their API key cannot call.
"""

import ast
import copy
import importlib
import json
import sys
from collections.abc import Callable, Coroutine, Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.requests import Request

from litellm import Router
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ModelAccessDeniedProxyException,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.agent_endpoints.auth.agent_access_groups import AgentAccessGroupCeiling, CeilingResolver
from litellm.proxy.auth.auth_utils import key_fallback_model_names
from litellm.proxy.auth.user_api_key_auth import enforce_key_and_fallback_model_access
from litellm.router_utils.fallback_event_handlers import FallbackCall, FallbackReading, iter_body_fallback_entries
from litellm.types.agents import AgentCaller, AgentResponse


def _fallback_model_names(fallbacks):
    """Model names the auth check validates for a top-level ``fallbacks`` value."""
    return list(key_fallback_model_names({"fallbacks": fallbacks}, route="/v1/chat/completions", request=None).litellm)


def _key_with_models(models: list[str]) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed",
        user_id="caller",
        user_role=LitellmUserRoles.INTERNAL_USER,
        models=models,
    )


# ── fallback model-name extraction ───────────────────────────────────────────


def test_fallback_model_names_router_config_shape():
    """Router-config shape: ``[{primary: [fallback_list]}]``."""
    assert _fallback_model_names([{"gpt-3.5-turbo": ["gpt-4", "claude-3"]}, {"gpt-4o": ["o1"]}]) == [
        "gpt-4",
        "claude-3",
        "o1",
    ]


def test_fallback_model_names_simple_string_shape():
    """Simple top-level shape: list of strings."""
    assert _fallback_model_names(["gpt-4", "claude-3"]) == ["gpt-4", "claude-3"]


def test_fallback_model_names_client_side_shape():
    """ClientSideFallbackModel shape: ``[{"model": "..."}]``."""
    assert _fallback_model_names([{"model": "gpt-4"}, {"model": "claude-3"}]) == [
        "gpt-4",
        "claude-3",
    ]


def test_fallback_model_names_nested_deployment_fallbacks():
    """A deployment target's own nested fallback field is unrolled too."""
    assert _fallback_model_names([{"primary": [{"model": "gpt-4", "fallbacks": [{"gpt-4": ["deepseek-chat"]}]}]}]) == [
        "gpt-4",
        "deepseek-chat",
    ]


def test_fallback_model_names_target_and_group_named_model_shapes():
    """An entry with a string ``model`` is one target in any key order; ``{"model": [...]}`` maps a group named so."""
    assert _fallback_model_names([{"requested": ["forbidden"], "model": "other"}]) == ["other"]
    assert _fallback_model_names([{"model": ["gpt-4", "claude-3"]}]) == ["gpt-4", "claude-3"]


def test_fallback_model_names_read_a_parameter_only_entry_as_a_target():
    """The Router merges every mapping of a list of targets into the call, so its list values name no model."""
    stop: Final = {"stop": ["END"]}
    call: Final = FallbackCall(model="", reading=FallbackReading.LITELLM)
    entries: Final = iter_body_fallback_entries({"fallbacks": [{"model": "gpt-4"}, stop]}, call)
    assert [entry.target for entry in entries] == [{"model": "gpt-4"}, stop]
    assert _fallback_model_names([{"model": "gpt-4"}, stop]) == ["gpt-4"]
    assert _fallback_model_names([{"gpt-3.5-turbo": ["gpt-4", stop]}]) == ["gpt-4"]


def test_fallback_model_names_read_a_targets_model_switching_fields_as_a_bodys():
    """A target's context-window swap and its own router_settings_override name models as the body's do."""
    swap: Final = {"model": "gpt-4", "context_window_fallback_dict": {"gpt-4": "gpt-4-32k"}}
    override: Final = {"model": "gpt-4", "router_settings_override": {"fallbacks": ["o1"]}}
    assert _fallback_model_names([swap]) == ["gpt-4", "gpt-4-32k"]
    assert _fallback_model_names([{"primary": [override]}]) == ["gpt-4", "o1"]


@pytest.mark.parametrize("beta", [False, True], ids=["litellm-fallbacks", "anthropic-fallbacks"])
def test_a_nested_list_as_a_fallback_entry_is_refused_in_either_reading(beta: bool):
    beta_headers: Final = {"extra_headers": {"anthropic-beta": _SERVER_SIDE_FALLBACK_BETA}} if beta else {}
    with pytest.raises(ValueError, match="a fallback list entry must be a model name or a mapping, got list"):
        key_fallback_model_names({"fallbacks": [["other"]], **beta_headers}, route="/v1/messages", request=None)


def test_a_targets_model_list_is_refused_as_the_bodys_is_even_with_client_side_credentials():
    """litellm.acompletion picks its deployment from a model_list, which the body may never carry"""
    from litellm.proxy.auth.auth_utils import is_request_body_safe

    target: Final = {"model": "primary", "model_list": [{"model_name": "primary", "litellm_params": {"model": "x"}}]}
    with pytest.raises(ValueError, match="model_list is not allowed"):
        is_request_body_safe(
            {"model": "primary", "fallbacks": [target]},
            {"allow_client_side_credentials": True},
            _router_without_fallback_group(),
            "primary",
        )



def test_fallback_model_names_empty_or_none():
    assert _fallback_model_names(None) == []
    assert _fallback_model_names([]) == []
    assert _fallback_model_names("not a list") == []


# ── _enforce_key_and_fallback_model_access ────────────────────────────────────


@pytest.mark.asyncio
async def test_router_override_fallbacks_validated_against_key_allowlist():
    """A fallback nested inside ``router_settings_override`` is validated
    against the API key's allowed models — not just the top-level
    ``fallbacks`` field."""
    valid_token = _key_with_models(["gpt-3.5-turbo"])
    request_data = {
        "model": "gpt-3.5-turbo",
        "router_settings_override": {
            "fallbacks": [{"gpt-3.5-turbo": ["unauthorized-model"]}],
        },
    }

    seen_models: list[str] = []

    async def fake_can_key_call_model(model, llm_model_list, valid_token, llm_router):
        seen_models.append(model)

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            side_effect=fake_can_key_call_model,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
            new=AsyncMock(),
        ),
    ):
        await enforce_key_and_fallback_model_access(
            valid_token=valid_token,
            request_data=request_data,
            route="/v1/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )

    # Both the primary model and the override-nested fallback must be
    # checked against the API key's allowlist.
    assert "gpt-3.5-turbo" in seen_models
    assert "unauthorized-model" in seen_models


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallback_field",
    [
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
    ],
)
async def test_router_override_all_fallback_fields_validated(fallback_field):
    """All three fallback fields the router accepts as per-request kwargs
    are validated — context_window_fallbacks and content_policy_fallbacks
    are promoted in route_llm_request.py too."""
    valid_token = _key_with_models(["gpt-3.5-turbo"])
    request_data = {
        "model": "gpt-3.5-turbo",
        "router_settings_override": {
            fallback_field: [{"gpt-3.5-turbo": ["smuggled-model"]}],
        },
    }

    seen: list[str] = []

    async def fake_can_key_call_model(model, llm_model_list, valid_token, llm_router):
        seen.append(model)

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            side_effect=fake_can_key_call_model,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
            new=AsyncMock(),
        ),
    ):
        await enforce_key_and_fallback_model_access(
            valid_token=valid_token,
            request_data=request_data,
            route="/v1/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )

    assert "smuggled-model" in seen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallback_field",
    [
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
    ],
)
async def test_top_level_fallback_fields_validated(fallback_field):
    """All three top-level fallback fields are forwarded to the router as
    per-request kwargs, so all three must be validated against the API
    key's allowlist. Greptile P1 follow-up: previously only the
    ``fallbacks`` field was walked at the top level."""
    valid_token = _key_with_models(["gpt-3.5-turbo"])
    request_data = {
        "model": "gpt-3.5-turbo",
        fallback_field: [{"gpt-3.5-turbo": ["top-level-smuggled"]}],
    }

    seen: list[str] = []

    async def fake_can_key_call_model(model, llm_model_list, valid_token, llm_router):
        seen.append(model)

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            side_effect=fake_can_key_call_model,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
            new=AsyncMock(),
        ),
    ):
        await enforce_key_and_fallback_model_access(
            valid_token=valid_token,
            request_data=request_data,
            route="/v1/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )

    assert "top-level-smuggled" in seen


@pytest.mark.asyncio
async def test_nested_deployment_fallback_inner_model_validated():
    """A model name nested several fallback rounds deep, inside a deployment
    target's own ``fallbacks``, is extracted and passed to can_key_call_model."""
    valid_token = _key_with_models(["gpt-3.5-turbo"])
    request_data = {
        "model": "gpt-3.5-turbo",
        "fallbacks": [
            {
                "gpt-3.5-turbo": [
                    {
                        "model": "gpt-3.5-turbo",
                        "fallbacks": [{"gpt-3.5-turbo": ["deep-smuggled-model"]}],
                    }
                ]
            }
        ],
    }

    seen: list[str] = []

    async def fake_can_key_call_model(model, llm_model_list, valid_token, llm_router):
        seen.append(model)

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            side_effect=fake_can_key_call_model,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
            new=AsyncMock(),
        ),
    ):
        await enforce_key_and_fallback_model_access(
            valid_token=valid_token,
            request_data=request_data,
            route="/v1/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )

    assert "deep-smuggled-model" in seen


@pytest.mark.asyncio
async def test_model_less_fallback_dict_is_skipped_never_passed_as_none():
    """A fallback target dict without a ``model`` key is skipped, never passed
    as ``None`` into can_key_call_model / is_valid_fallback_model."""
    valid_token = _key_with_models(["gpt-3.5-turbo"])
    request_data = {
        "model": "gpt-3.5-turbo",
        "fallbacks": [
            {
                "gpt-3.5-turbo": [
                    {"model": "real-fallback"},
                    {"api_base": "http://attacker"},
                    "string-fallback",
                ]
            }
        ],
    }

    seen: list[str] = []

    async def fake_can_key_call_model(model, llm_model_list, valid_token, llm_router):
        seen.append(model)

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            side_effect=fake_can_key_call_model,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
            new=AsyncMock(),
        ),
    ):
        await enforce_key_and_fallback_model_access(
            valid_token=valid_token,
            request_data=request_data,
            route="/v1/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )

    assert None not in seen
    assert seen == ["gpt-3.5-turbo", "real-fallback", "string-fallback"]


@pytest.mark.asyncio
async def test_router_override_without_fallbacks_does_not_break_auth():
    """``router_settings_override`` set without any fallback fields is a
    no-op for the auth check — only the primary model is validated."""
    valid_token = _key_with_models(["gpt-3.5-turbo"])
    request_data = {
        "model": "gpt-3.5-turbo",
        "router_settings_override": {"num_retries": 3, "timeout": 30},
    }

    seen: list[str] = []

    async def fake_can_key_call_model(model, llm_model_list, valid_token, llm_router):
        seen.append(model)

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.can_key_call_model",
            side_effect=fake_can_key_call_model,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.is_valid_fallback_model",
            new=AsyncMock(),
        ),
    ):
        await enforce_key_and_fallback_model_access(
            valid_token=valid_token,
            request_data=request_data,
            route="/v1/chat/completions",
            request=None,
            llm_model_list=None,
            llm_router=None,
        )

    assert seen == ["gpt-3.5-turbo"]


# ── Anthropic's server-side `fallbacks` on /v1/messages ──────────────────────


def _messages_request(anthropic_beta: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [(b"anthropic-beta", anthropic_beta.encode())],
        }
    )


def _router_without_fallback_group() -> Router:
    return Router(
        model_list=[
            {"model_name": "primary", "litellm_params": {"model": "anthropic/claude-fable-5", "api_key": "fake"}}
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("anthropic_beta", "key_models", "allowed"),
    [
        pytest.param("server-side-fallback-2026-07-01", ["primary", "claude-opus-4-8"], True, id="beta-key-allows"),
        pytest.param("server-side-fallback-2026-07-01", ["primary"], False, id="beta-key-denies"),
        pytest.param("context-management-2025-06-27", ["primary", "claude-opus-4-8"], False, id="no-beta-needs-group"),
    ],
)
async def test_anthropic_server_side_fallbacks_need_key_access_but_no_router_group(
    anthropic_beta: str, key_models: list[str], allowed: bool
):
    router = _router_without_fallback_group()
    check = enforce_key_and_fallback_model_access(
        valid_token=_key_with_models(key_models),
        request_data={"model": "primary", "fallbacks": [{"model": "claude-opus-4-8"}]},
        route="/v1/messages",
        request=_messages_request(anthropic_beta),
        llm_model_list=router.get_model_list(),
        llm_router=router,
    )
    if allowed:
        await check
    else:
        with pytest.raises(Exception, match="claude-opus-4-8"):
            await check


def test_body_router_fallbacks_get_the_banned_param_check():
    from litellm.proxy.auth.auth_utils import is_request_body_safe

    with pytest.raises(ValueError, match="api_base"):
        is_request_body_safe(
            {"model": "primary", "router_fallbacks": [{"model": "primary", "api_base": "https://attacker.example"}]},
            {},
            _router_without_fallback_group(),
            "primary",
        )


@pytest.mark.asyncio
async def test_body_router_fallbacks_need_key_model_access():
    router = _router_without_fallback_group()
    with pytest.raises(ModelAccessDeniedProxyException, match="'claude-opus-4-8'") as exc_info:
        await enforce_key_and_fallback_model_access(
            valid_token=_key_with_models(["primary"]),
            request_data={"model": "primary", "router_fallbacks": [{"primary": ["claude-opus-4-8"]}]},
            route="/v1/messages",
            request=_messages_request("server-side-fallback-2026-07-01"),
            llm_model_list=router.get_model_list(),
            llm_router=router,
        )

    # The key refuses the target; the Router-group check, which would also fail on it, never runs
    assert exc_info.value.type == ProxyErrorTypes.key_model_access_denied


@pytest.mark.asyncio
async def test_client_router_settings_override_fallbacks_are_checked_when_anthropic_owns_the_body_field():
    router = _router_without_fallback_group()
    with pytest.raises(ModelAccessDeniedProxyException, match="'claude-opus-4-8'") as exc_info:
        await enforce_key_and_fallback_model_access(
            valid_token=_key_with_models(["primary"]),
            request_data={
                "model": "primary",
                "fallbacks": "default",
                "router_settings_override": {"fallbacks": [{"primary": ["claude-opus-4-8"]}]},
            },
            route="/v1/messages",
            request=_messages_request("server-side-fallback-2026-07-01"),
            llm_model_list=router.get_model_list(),
            llm_router=router,
        )

    # The key refuses the target; the Router-group check, which would also fail on it, never runs
    assert exc_info.value.type == ProxyErrorTypes.key_model_access_denied


# ── Every fallback field, end to end through a proxy with a mock upstream ────
#
# One in-process proxy app (real auth, real pre-call processing, real Router) with three upstream hosts: the
# requested deployment answers 500, so the request falls back; `other` is a model the key may call; the attacker
# host is where a caller's endpoint param would send the deployment's API key.

_REQUESTED_HOST = "https://requested.upstream.test"
_OTHER_HOST = "https://other.upstream.test"
_FORBIDDEN_HOST = "https://forbidden.upstream.test"
_ATTACKER_HOST = "https://attacker.example.test"
_DEPLOYMENT_KEY = "sk-ant-deployment-secret"
_CALLER_KEY = "sk-caller-key"
_SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _messages_reply(model: str) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


# The fields a fallback target can arrive in, stated once here so each is tested even if a list in the code drops it
_REQUIRED_LIST_FIELDS = (
    "fallbacks",
    "router_fallbacks",
    "context_window_fallbacks",
    "content_policy_fallbacks",
    "default_fallbacks",
)


def _list_fields() -> tuple[str, ...]:
    """The fallback list fields: the ones the checks read, and the ones this suite requires them to read."""
    from litellm.router_utils.fallback_event_handlers import FALLBACK_LIST_FIELDS

    return tuple(dict.fromkeys((*FALLBACK_LIST_FIELDS, *_REQUIRED_LIST_FIELDS)))


def _target_fields() -> tuple[str, ...]:
    """Every request field a fallback target can come from: the list a rebase must keep the Router's reads inside."""
    return (*_list_fields(), "context_window_fallback_dict")


@pytest.mark.parametrize("fallback_field", ("context_window_fallbacks", "content_policy_fallbacks"))
def test_auth_rejects_non_list_error_fallbacks(fallback_field: str):
    from litellm.proxy.auth.auth_utils import is_request_body_safe

    with pytest.raises(ValueError, match=f"{fallback_field} must be a list"):
        is_request_body_safe(
            {"model": "primary", fallback_field: {"forbidden": []}},
            {},
            _router_without_fallback_group(),
            "primary",
        )


# The header carriers and stored-credential name that swap in a caller's key on the deployment's own endpoint
_REQUIRED_CREDENTIAL_PARAMS = ("headers", "provider_specific_header", "litellm_credential_name")
_REQUIRED_MOCK_PARAMS = ("mock_response", "mock_tool_calls")


def _credential_and_endpoint_params() -> tuple[str, ...]:
    from litellm.proxy.auth.auth_utils import _BANNED_REQUEST_BODY_PARAMS
    from litellm.router_utils import clientside_credential_handler as credentials

    return tuple(
        sorted(
            {
                *_BANNED_REQUEST_BODY_PARAMS,
                *credentials.clientside_credential_keys,
                *credentials._ADMIN_CONFIG_FIELDS_TO_CLEAR_ON_BASE_OVERRIDE,
                *_REQUIRED_CREDENTIAL_PARAMS,
            }
        )
    )


def _mock_params() -> tuple[str, ...]:
    from litellm.proxy.route_llm_request import GATED_MOCK_PARAM_NAMES

    return tuple(sorted({*GATED_MOCK_PARAM_NAMES, *_REQUIRED_MOCK_PARAMS}))


class _Gateway:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        requested_error: tuple[int, str, str] = (500, "api_error", "x"),
        *,
        requested_answers: bool = False,
        other_error: tuple[int, str, str] | None = None,
    ) -> None:
        self._monkeypatch = monkeypatch
        self._requested_error = requested_error
        # The requested deployment answers instead of failing, as Anthropic does when it serves a fallback itself
        self._requested_answers = requested_answers
        # The `other` deployment fails too, so a hop to it falls back again
        self._other_error = other_error

    async def send(
        self,
        body_fields: dict[str, object],
        *,
        beta: bool = False,
        key_router_settings: dict[str, object] | None = None,
        team_router_settings: dict[str, object] | None = None,
        key_models: tuple[str, ...] = ("requested", "other"),
        credentials_allowed: bool = False,
        key_aliases: Mapping[str, str] | None = None,
        team_aliases: Mapping[str, str] | None = None,
        router_aliases: Mapping[str, str] | None = None,
        extra_deployments: Sequence[Mapping[str, object]] = (),
        routing_groups: Sequence[Mapping[str, object]] = (),
        token_team_models: tuple[str, ...] | None = None,
        allow_requests_on_db_unavailable: bool = False,
        user_id: str | None = None,
        team_models: tuple[str, ...] = ("requested", "other"),
        router_fallbacks: Sequence[Mapping[str, object]] = (),
        enforce_fallback_model_access: bool = False,
        # general_settings.allow_client_side_credentials, the proxy-wide opt-in the body's checks read
        client_side_credentials: bool = False,
        # The key's own config, a legacy per-key setting the key's owner can write
        key_config: Mapping[str, object] | None = None,
        route: str = "/v1/messages",
    ) -> "_GatewayResult":
        import httpx
        import respx

        import litellm
        from litellm.caching.dual_cache import DualCache
        from litellm.proxy import proxy_server
        from litellm.proxy._types import LiteLLM_TeamTableCachedObj
        from litellm.proxy.auth.fallback_model_access import router_fallback_access_check
        from litellm.proxy.proxy_server import app, hash_token
        from litellm.proxy.utils import ProxyLogging

        router = Router(
            model_list=[
                {
                    "model_name": name,
                    # Each group sends its own name to Anthropic, so a name Anthropic gets means the same model here
                    "litellm_params": {
                        "model": f"anthropic/{name}",
                        "api_key": _DEPLOYMENT_KEY,
                        "api_base": host,
                    },
                }
                for name, host in (
                    ("requested", _REQUESTED_HOST),
                    ("other", _OTHER_HOST),
                    ("forbidden", _FORBIDDEN_HOST),
                )
            ]
            + [copy.deepcopy(dict(deployment)) for deployment in extra_deployments],
            num_retries=0,
            model_group_alias=dict(router_aliases or {}),
            routing_groups=[dict(group) for group in routing_groups] or None,
            fallbacks=[dict(chain) for chain in router_fallbacks] or None,
            fallback_access_check=router_fallback_access_check if enforce_fallback_model_access else None,
        )
        router.fallback_target_credentials_allowed = (lambda: True) if credentials_allowed else None
        cache = DualCache()
        team_id = "fallback-team" if team_router_settings is not None else None
        token = UserAPIKeyAuth(
            api_key=_CALLER_KEY,
            token=hash_token(_CALLER_KEY),
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=list(key_models),
            team_id=team_id,
            user_id=user_id,
            team_models=(
                list(token_team_models) if token_team_models is not None else list(team_models) if team_id else []
            ),
            router_settings=key_router_settings,
            config=dict(key_config or {}),
            aliases=dict(key_aliases or {}),
            team_model_aliases=dict(team_aliases) if team_aliases is not None else None,
        )
        await cache.async_set_cache(key=hash_token(_CALLER_KEY), value=token)
        if user_id is not None:
            await cache.async_set_cache(key=user_id, value=LiteLLM_UserTable(user_id=user_id))
        if team_id is not None:
            await cache.async_set_cache(
                key=f"team_id:{team_id}",
                value=LiteLLM_TeamTableCachedObj(
                    team_id=team_id, models=list(team_models), router_settings=team_router_settings
                ),
            )
        for name, value in {
            "master_key": "sk-master",
            "prisma_client": MagicMock(),
            "user_api_key_cache": cache,
            "llm_router": router,
            "llm_model_list": router.get_model_list(),
            "general_settings": {
                **({"allow_requests_on_db_unavailable": True} if allow_requests_on_db_unavailable else {}),
                **({"enforce_fallback_model_access": True} if enforce_fallback_model_access else {}),
                **({"allow_client_side_credentials": True} if client_side_credentials else {}),
            },
            "user_custom_auth": None,
            "proxy_logging_obj": ProxyLogging(user_api_key_cache=cache),
        }.items():
            self._monkeypatch.setattr(proxy_server, name, value)
        self._monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        payload = {
            "model": "requested",
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "hi"}],
            **body_fields,
        }
        headers = {
            "Authorization": f"Bearer {_CALLER_KEY}",
            **({"anthropic-beta": _SERVER_SIDE_FALLBACK_BETA} if beta else {}),
        }

        def error_reply(error: tuple[int, str, str]) -> "httpx.Response":
            return httpx.Response(error[0], json={"type": "error", "error": {"type": error[1], "message": error[2]}})

        with respx.mock(assert_all_called=False) as upstream:
            routes = {
                "requested": upstream.post(url__startswith=_REQUESTED_HOST).mock(
                    return_value=httpx.Response(200, json=_messages_reply("claude-requested"))
                    if self._requested_answers
                    else error_reply(self._requested_error)
                ),
                "other": upstream.post(url__startswith=_OTHER_HOST).mock(
                    return_value=httpx.Response(200, json=_messages_reply("claude-other"))
                    if self._other_error is None
                    else error_reply(self._other_error)
                ),
                "forbidden": upstream.post(url__startswith=_FORBIDDEN_HOST).mock(
                    return_value=httpx.Response(200, json=_messages_reply("claude-forbidden"))
                ),
                "attacker": upstream.post(url__startswith=_ATTACKER_HOST).mock(
                    return_value=httpx.Response(200, json=_messages_reply("attacker"))
                ),
            }
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy.test") as client:
                response = await client.post(route, json=payload, headers=headers)
        return _GatewayResult(
            response=response,
            calls={name: route.call_count for name, route in routes.items()},
            attacker_credentials=tuple(
                value
                for call in routes["attacker"].calls
                for value in (call.request.headers.get("x-api-key"), call.request.headers.get("authorization"))
                if value is not None
            ),
            requested_bodies=tuple(json.loads(call.request.read()) for call in routes["requested"].calls),
        )


@dataclass(frozen=True, slots=True)
class _GatewayResult:
    response: "httpx.Response"
    calls: dict[str, int]
    attacker_credentials: tuple[str, ...]
    # The JSON bodies the requested deployment received
    requested_bodies: tuple[object, ...]

    def deployment_key_reached_the_attacker(self) -> bool:
        return any(_DEPLOYMENT_KEY in value for value in self.attacker_credentials)


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> _Gateway:
    return _Gateway(monkeypatch)


def _placements(make_value):
    """Every place a fallback target can enter a /v1/messages request: (id, send kwargs)."""
    body_rows = tuple(
        (f"body.{field}{'+beta' if beta else ''}", {"body_fields": {field: make_value(field)}, "beta": beta})
        for field in _target_fields()
        for beta in (False, True)
    )
    override_rows = tuple(
        (
            f"{source}.router_settings.{field}",
            {"body_fields": {"router_settings_override": {field: make_value(field)}}, "beta": True}
            if source == "body"
            else {f"{source}_router_settings": {field: make_value(field)}, "body_fields": {}, "beta": True},
        )
        for source in ("body", "key", "team")
        for field in _list_fields()
    )
    return (*body_rows, *override_rows)


_ENDPOINT_PARAMS = ("api_base", "base_url", "api_key", "headers", "provider_specific_header")
_CALLER_SUPPLIED_KEY = "sk-caller-supplied"
_PARAM_VALUES = {
    "api_base": _ATTACKER_HOST,
    "base_url": _ATTACKER_HOST,
    "api_key": _CALLER_SUPPLIED_KEY,
    "headers": {"x-api-key": _CALLER_SUPPLIED_KEY},
    "provider_specific_header": {
        "custom_llm_provider": "anthropic",
        "extra_headers": {"x-api-key": _CALLER_SUPPLIED_KEY},
    },
}
# A client-side target names one model; a Router-style chain names the failing group and its targets
_SHAPES = ("flat", "chain")


def _shaped(target: dict, shape: str) -> list:
    return [target] if shape == "flat" else [{"requested": [target]}]


def _endpoint_target(param: str, shape: str):
    def make(field: str):
        if field == "context_window_fallback_dict":
            return {"anthropic/requested": _ATTACKER_HOST}
        return _shaped({"model": "other", param: _PARAM_VALUES[param]}, shape)

    return make


def _mock_target(shape: str):
    def make(field: str):
        if field == "context_window_fallback_dict":
            return {"anthropic/requested": "other"}
        return _shaped({"model": "other", "mock_response": "served by a mock"}, shape)

    return make


# The Router iterates whatever value a field carries, so a mapping serves its keys and a chain serves its lists
_FORBIDDEN_SHAPES: Final[Mapping[str, object]] = MappingProxyType[str, object](
    {
        "chain": [{"requested": ["forbidden"]}],
        "list": ["forbidden"],
        "client-target": [{"model": "forbidden"}],
        "dict-key": {"forbidden": []},
        "dict-chain": {"requested": ["forbidden"]},
        "chain-of-dict": [{"requested": {"forbidden": []}}],
    }
)


def _forbidden_target(shape: str) -> Callable[[str], object]:
    def make(field: str) -> object:
        if field == "context_window_fallback_dict":
            return {"anthropic/requested": "forbidden"}
        return _FORBIDDEN_SHAPES[shape]

    return make


_ENDPOINT_ROWS = tuple(
    pytest.param(kwargs, id=f"{row_id}-{param}-{shape}")
    for param in _ENDPOINT_PARAMS
    for shape in _SHAPES
    for row_id, kwargs in _placements(_endpoint_target(param, shape))
)
_MOCK_ROWS = tuple(
    pytest.param(kwargs, id=f"{row_id}-{shape}")
    for shape in _SHAPES
    for row_id, kwargs in _placements(_mock_target(shape))
    if "context_window_fallback_dict" not in row_id
)
_FORBIDDEN_ROWS = tuple(
    pytest.param(kwargs, id=f"{row_id}-{shape}")
    for shape in _FORBIDDEN_SHAPES
    for row_id, kwargs in _placements(_forbidden_target(shape))
    if shape == "chain" or "context_window_fallback_dict" not in row_id
)


@pytest.mark.asyncio
@pytest.mark.parametrize("send_kwargs", _ENDPOINT_ROWS)
async def test_a_caller_endpoint_or_credential_on_any_fallback_target_is_refused(gateway: _Gateway, send_kwargs):
    result = await gateway.send(**send_kwargs)

    assert not result.deployment_key_reached_the_attacker(), "the deployment key reached the attacker host"
    assert result.attacker_credentials == (), "a request reached the attacker host"
    assert result.response.status_code in (400, 401, 403), result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("send_kwargs", _MOCK_ROWS)
async def test_a_mock_param_on_any_fallback_target_is_refused(gateway: _Gateway, send_kwargs):
    result = await gateway.send(**send_kwargs)

    assert result.response.status_code in (400, 401, 403), result.response.text
    assert "served by a mock" not in result.response.text
    assert result.calls["other"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("credentials_allowed", "leaks"),
    [(False, False), (True, True)],
    ids=["merge-point-refuses", "control-hop-allowed-credentials-leaks"],
)
async def test_a_fallback_kwarg_auth_never_sees_cannot_carry_a_credential_past_the_merge_point(
    gateway: _Gateway, monkeypatch: pytest.MonkeyPatch, credentials_allowed: bool, leaks: bool
):
    """
    A future Router change that reads its fallback list from a kwarg auth does not know (here `backup_models`) skips
    every auth check, so the hop itself, where every target is laid over the deployment's params, refuses a
    credential or endpoint. The control shows the same request leaks the deployment key when the hop allows one.
    """
    _router_reads_fallbacks_from_backup_models(monkeypatch)
    result = await gateway.send(
        {"backup_models": [{"requested": [{"model": "other", "api_base": _ATTACKER_HOST}]}]},
        credentials_allowed=credentials_allowed,
    )

    assert result.calls["requested"] == 1
    assert result.calls["other"] == 0
    assert result.calls["attacker"] == (1 if leaks else 0)
    assert result.deployment_key_reached_the_attacker() is leaks


def _router_reads_fallbacks_from_backup_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm import router as router_module

    reads_the_list_from: Final = router_module.request_fallbacks

    def reads_backup_models(kwargs, router_fallbacks, *, anthropic_messages_call=False):
        backup = kwargs.get("backup_models")
        if isinstance(backup, list):
            return backup
        return reads_the_list_from(kwargs, router_fallbacks, anthropic_messages_call=anthropic_messages_call)

    monkeypatch.setattr(router_module, "request_fallbacks", reads_backup_models)


# Each is neither a credential nor an endpoint, yet picks where or how the deployment's key is sent: the provider (and
# so its default host), a LiteLLM proxy host, a deployment by id, the deployment identity, a file-id map, a request
# path, the no-deployment passthrough, the TLS check, or a token endpoint and its token.
_INDIRECT_ROUTING_VALUES: Final = MappingProxyType[str, object](
    {
        "custom_llm_provider": "openai",
        "azure": True,
        "use_litellm_proxy": True,
        "specific_deployment": True,
        "model_info": {"id": "forbidden-deployment"},
        "model_file_id_mapping": {"file-1": {"forbidden-deployment": "file-2"}},
        "endpoint": "/v1/other",
        "passthrough_on_no_deployment": True,
        "ssl_verify": False,
        "gigachat_auth_url": _ATTACKER_HOST,
        "gigachat_access_token": _CALLER_SUPPLIED_KEY,
        "gigachat_scope": "GIGACHAT_API_CORP",
        "use_xai_oauth": True,
    }
)


@pytest.mark.asyncio
@pytest.mark.parametrize("layer", ["auth", "merge-point"])
@pytest.mark.parametrize("field", sorted(_INDIRECT_ROUTING_VALUES))
async def test_a_field_that_steers_the_deployment_key_indirectly_is_refused_on_a_fallback_target(
    gateway: _Gateway, monkeypatch: pytest.MonkeyPatch, field: str, layer: str
):
    chain: Final = [{"requested": [{"model": "other", field: _INDIRECT_ROUTING_VALUES[field]}]}]
    if layer == "auth":
        result = await gateway.send({"fallbacks": chain})
        assert result.response.status_code in (400, 401, 403), result.response.text
        assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}
        return
    _router_reads_fallbacks_from_backup_models(monkeypatch)
    result = await gateway.send({"backup_models": chain})
    assert result.calls == {"requested": 1, "other": 0, "forbidden": 0, "attacker": 0}, result.response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("credentials_allowed", "leaks"),
    [(False, False), (True, True)],
    ids=["merge-point-refuses", "control-hop-allowed-credentials-leaks"],
)
async def test_a_fallback_field_the_proxy_renames_after_auth_still_stops_at_the_merge_point(
    gateway: _Gateway, monkeypatch: pytest.MonkeyPatch, credentials_allowed: bool, leaks: bool
):
    """
    A mutant proxy step that runs after auth renames a field auth does not read into a fallback field the Router does,
    so auth never sees the target. The hop still refuses its endpoint, and the control proves the mutant reaches it.
    """
    from litellm.proxy import common_request_processing

    snapshots_the_body: Final = common_request_processing.refresh_proxy_server_request_body_snapshot

    def renames_then_snapshots(data: MutableMapping[str, object], *, guardrails_applied: bool = False) -> None:
        if "backup_models" in data:
            data["fallbacks"] = data.pop("backup_models")
        snapshots_the_body(data, guardrails_applied=guardrails_applied)

    monkeypatch.setattr(common_request_processing, "refresh_proxy_server_request_body_snapshot", renames_then_snapshots)
    result = await gateway.send(
        {"backup_models": [{"requested": [{"model": "other", "api_base": _ATTACKER_HOST}]}]},
        credentials_allowed=credentials_allowed,
    )

    assert result.calls["requested"] == 1, result.response.text
    assert result.calls["other"] == 0
    assert result.calls["attacker"] == (1 if leaks else 0)
    assert result.deployment_key_reached_the_attacker() is leaks


def _has_unreadable_litellm_list(send_kwargs: Mapping[str, object]) -> bool:
    """
    Whether a LiteLLM fallback field holds a value that is not a list, which every LiteLLM reader iterates as one: any
    field of a body, of its router_settings_override and of saved settings, except a body's own `fallbacks` under the
    beta, which is Anthropic's
    """
    body: Final = send_kwargs.get("body_fields")
    override: Final = body.get("router_settings_override") if isinstance(body, Mapping) else None
    sources: Final = (
        (body, send_kwargs.get("beta") is True),
        (override, False),
        (send_kwargs.get("key_router_settings"), False),
        (send_kwargs.get("team_router_settings"), False),
    )
    return any(
        field in source and not isinstance(source[field], list)
        for source, anthropic_owns_fallbacks in sources
        if isinstance(source, Mapping)
        for field in _list_fields()
        if not (anthropic_owns_fallbacks and field == "fallbacks")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("send_kwargs", _FORBIDDEN_ROWS)
async def test_every_fallback_target_must_be_a_model_the_key_may_call(gateway: _Gateway, send_kwargs):
    result = await gateway.send(**send_kwargs)
    rejects_non_list_value = _has_unreadable_litellm_list(send_kwargs)
    # A malformed list is refused with one status from every source
    expected_statuses = (400,) if rejects_non_list_value else (401, 403)

    assert result.response.status_code in expected_statuses, result.response.text
    if rejects_non_list_value:
        assert "must be a list" in result.response.text
    assert result.calls["forbidden"] == 0
    assert result.calls["attacker"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallbacks",
    [
        pytest.param([{"model": "other"}, {"requested": ["forbidden"]}], id="group-map-beside-a-target"),
        pytest.param([{"requested": [{"grp": ["forbidden"]}]}], id="group-map-in-a-group-list"),
    ],
)
async def test_every_name_in_anthropics_fallbacks_is_checked_in_any_shape(
    gateway: _Gateway, fallbacks: Sequence[object]
) -> None:
    """Anthropic reads only the `model` of each entry, but a name a caller puts in any other shape is checked too"""
    result: Final = await gateway.send({"fallbacks": list(fallbacks)}, beta=True)

    assert result.response.status_code in (401, 403), result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "requested_error"),
    [
        pytest.param(
            "context_window_fallbacks",
            (400, "invalid_request_error", "prompt is too long: 300000 tokens > 200000 maximum"),
            id="context-window",
        ),
        pytest.param(
            "content_policy_fallbacks",
            (400, "invalid_request_error", "Output blocked by content filtering policy"),
            id="content-policy",
        ),
    ],
)
@pytest.mark.parametrize("shape", ["dict-key", "dict-chain", "chain-of-dict", "list"])
async def test_a_fallback_the_upstream_error_triggers_never_reaches_a_model_the_key_may_not_call(
    monkeypatch: pytest.MonkeyPatch, field: str, requested_error: tuple[int, str, str], shape: str
):
    value: Final = _FORBIDDEN_SHAPES[shape]
    result = await _Gateway(monkeypatch, requested_error).send({field: value})

    if not isinstance(value, list):
        # The Router refuses an error field that is not a list, and no other reader reads one
        assert result.response.status_code == 400, result.response.text
        assert f"{field} must be a list" in result.response.text
        assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}
        return
    assert result.response.status_code in (401, 403), result.response.text
    assert result.calls["forbidden"] == 0


# The upstream error each LiteLLM fallback field falls back on; `fallbacks` takes any error
_FALLBACK_FIELD_ERRORS: Final = (
    pytest.param(
        "context_window_fallbacks",
        (400, "invalid_request_error", "prompt is too long: 300000 tokens > 200000 maximum"),
        id="context-window",
    ),
    pytest.param(
        "content_policy_fallbacks",
        (400, "invalid_request_error", "Output blocked by content filtering policy"),
        id="content-policy",
    ),
    pytest.param("fallbacks", (500, "api_error", "x"), id="fallbacks"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["body", "router-settings-override"])
@pytest.mark.parametrize(
    "nested_target",
    [
        pytest.param(MappingProxyType({"model": "forbidden"}), id="model-the-key-may-not-call"),
        pytest.param(MappingProxyType({"model": "other", "mock_response": "served by a mock"}), id="mock-param"),
    ],
)
async def test_a_fallback_entry_that_names_a_model_is_never_read_as_a_model_group(
    monkeypatch: pytest.MonkeyPatch, placement: str, nested_target: Mapping[str, object]
) -> None:
    """
    The entry names the model `other`, so the fallback checks read it as one target and never read the target nested
    under its first key. The Router reads it the same way, never as the requested group mapped to that target.
    """
    entry: Final = {"requested": [dict(nested_target)], "model": "other"}
    # A string keeps the list in the form the Router resolves by group, and serves from `other`
    settings: Final = {"fallbacks": ["other", entry]}
    body: Final = settings if placement == "body" else {"router_settings_override": settings}
    result: Final = await _Gateway(monkeypatch).send(body)

    assert result.response.status_code == 200, result.response.text
    assert "served by a mock" not in result.response.text
    assert result.calls == {"requested": 1, "other": 1, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(("field", "requested_error"), _FALLBACK_FIELD_ERRORS[:2])
@pytest.mark.parametrize("placement", ["body", "router-settings-override"])
@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(MappingProxyType({"requested": ["other"], "model": "other"}), id="target-with-a-group-key"),
        pytest.param(MappingProxyType({"model": "other"}), id="target"),
    ],
)
async def test_a_target_in_an_error_field_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    requested_error: tuple[int, str, str],
    placement: str,
    entry: Mapping[str, object],
) -> None:
    """The two error fields map model groups to their fallbacks, where a target would match no group and do nothing"""
    entries: Final = [dict(entry)]
    body: Final = {field: entries} if placement == "body" else {"router_settings_override": {field: entries}}
    result: Final = await _Gateway(monkeypatch, requested_error).send(body)

    assert result.response.status_code == 400, result.response.text
    assert f"LiteLLM {field} maps model groups to their fallbacks, so it cannot hold a target" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(("field", "requested_error"), _FALLBACK_FIELD_ERRORS)
async def test_an_empty_mapping_in_a_list_of_group_maps_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch, field: str, requested_error: tuple[int, str, str]
) -> None:
    """The Router reads a group map by its first key, which an empty mapping does not have"""
    result: Final = await _Gateway(monkeypatch, requested_error).send({field: ["other", {}]})

    assert result.response.status_code == 400, result.response.text
    assert f"LiteLLM {field} must not hold an empty mapping" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


def _lists_holding(entry: Mapping[str, object]) -> tuple[object, ...]:
    """(field, requested error, fallbacks): the entry beside a target, and in the list a group maps to, in each field"""
    beside_a_target: Final = pytest.param(
        "fallbacks", (500, "api_error", "x"), [{"model": "other"}, dict(entry)], id="fallbacks-beside-a-target"
    )
    in_a_group_list: Final = tuple(
        pytest.param(*row.values, [{"requested": ["other", dict(entry)]}], id=f"{row.id}-group-list")
        for row in _FALLBACK_FIELD_ERRORS
    )
    return (beside_a_target, *in_a_group_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["body", "router-settings-override", "key", "team"])
@pytest.mark.parametrize(
    ("field", "requested_error", "fallbacks"), _lists_holding(MappingProxyType({"mock_tool_calls": [{"function": {}}]}))
)
async def test_a_parameter_only_entry_the_router_merges_gets_the_target_checks(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    field: str,
    requested_error: tuple[int, str, str],
    fallbacks: Sequence[object],
) -> None:
    """The Router merges every mapping of a list of targets into the call, with or without a model"""
    settings: Final = {field: list(fallbacks)}
    body_fields: Final = {"body": settings, "router-settings-override": {"router_settings_override": settings}}
    result: Final = await _Gateway(monkeypatch, requested_error).send(
        body_fields.get(source, {}),
        key_router_settings=settings if source == "key" else None,
        team_router_settings=settings if source == "team" else None,
    )

    assert result.response.status_code in (400, 401), result.response.text
    assert "a fallback target may not carry mock_tool_calls" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(("field", "requested_error", "fallbacks"), _lists_holding(MappingProxyType({"stop": ["END"]})))
async def test_a_parameter_only_entry_the_router_merges_names_no_model(
    monkeypatch: pytest.MonkeyPatch, field: str, requested_error: tuple[int, str, str], fallbacks: Sequence[object]
) -> None:
    """`END` is a stop sequence the Router merges into the call, not a model the key must be able to call"""
    result: Final = await _Gateway(monkeypatch, requested_error).send({field: list(fallbacks)})

    assert result.response.status_code == 200, result.response.text
    assert result.calls == {"requested": 1, "other": 1, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "requested_error", "fallbacks"),
    [
        pytest.param("fallbacks", (500, "api_error", "x"), ["other"], id="list-of-strings"),
        pytest.param("fallbacks", (500, "api_error", "x"), [{"model": "other", "max_tokens": 5}], id="list-of-targets"),
        pytest.param(
            "fallbacks",
            (500, "api_error", "x"),
            [{"max_tokens": 5, "messages": [{"content": "hi", "role": "user"}], "model": "other"}],
            id="target-with-sorted-keys",
        ),
        pytest.param(
            "fallbacks",
            (500, "api_error", "x"),
            [{"stop": ["\n\nHuman:"], "modalities": ["text"], "model": "other"}],
            id="target-with-string-lists-before-its-model",
        ),
        pytest.param("fallbacks", (500, "api_error", "x"), [{"requested": ["other"]}], id="group-map"),
        pytest.param(
            "fallbacks", (500, "api_error", "x"), [{"requested": [{"model": "other"}]}], id="group-map-of-targets"
        ),
        pytest.param(
            "context_window_fallbacks",
            (400, "invalid_request_error", "prompt is too long: 300000 tokens > 200000 maximum"),
            [{"requested": ["other"]}],
            id="context-window-group-map",
        ),
        pytest.param(
            "content_policy_fallbacks",
            (400, "invalid_request_error", "Output blocked by content filtering policy"),
            [{"requested": ["other"]}],
            id="content-policy-group-map",
        ),
    ],
)
async def test_a_documented_fallback_shape_still_serves_the_request(
    monkeypatch: pytest.MonkeyPatch, field: str, requested_error: tuple[int, str, str], fallbacks: Sequence[object]
) -> None:
    result: Final = await _Gateway(monkeypatch, requested_error).send({field: list(fallbacks)})

    assert result.response.status_code == 200, result.response.text
    assert result.calls == {"requested": 1, "other": 1, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
async def test_a_group_named_model_still_falls_back_through_its_group_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """`{"model": [...]}` names no model: its `model` is a list, so it maps the group named `model` to its fallbacks"""
    group_named_model: Final = {
        "model_name": "model",
        "litellm_params": {"model": "anthropic/model", "api_key": _DEPLOYMENT_KEY, "api_base": _REQUESTED_HOST},
    }
    result: Final = await _Gateway(monkeypatch).send(
        {"model": "model", "fallbacks": [{"model": ["other"]}]},
        key_models=("model", "other"),
        extra_deployments=[group_named_model],
    )

    assert result.response.status_code == 200, result.response.text
    assert result.calls == {"requested": 1, "other": 1, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(("field", "requested_error"), _FALLBACK_FIELD_ERRORS)
@pytest.mark.parametrize("placement", ["body", "router-settings-override"])
@pytest.mark.parametrize(
    "chain",
    [
        pytest.param("abc", id="string"),
        pytest.param(MappingProxyType({"other": []}), id="mapping"),
        pytest.param(None, id="none"),
    ],
)
async def test_a_group_mapped_to_a_non_list_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    requested_error: tuple[int, str, str],
    placement: str,
    chain: str | Mapping[str, object] | None,
) -> None:
    """The Router iterates a group's value: a string calls each character as a model, which no fallback check reads"""
    entries: Final = [{"requested": dict(chain) if isinstance(chain, Mapping) else chain}]
    body: Final = {field: entries} if placement == "body" else {"router_settings_override": {field: entries}}
    result: Final = await _Gateway(monkeypatch, requested_error).send(body)

    assert result.response.status_code == 400, result.response.text
    assert f"LiteLLM {field} must map model group 'requested' to a list of model groups" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
async def test_a_beta_header_only_in_the_body_still_makes_the_fallbacks_anthropics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The Router reads the beta from the body's extra_headers too and sends `fallbacks` to Anthropic, so auth checks its
    names as Anthropic's: a name that has an alias is refused
    """
    result: Final = await _Gateway(monkeypatch, requested_answers=True).send(
        {
            "fallbacks": [{"model": _ALIASED_PROVIDER_FALLBACK}],
            "extra_headers": {"anthropic-beta": _SERVER_SIDE_FALLBACK_BETA},
        },
        router_aliases={_ALIASED_PROVIDER_FALLBACK: "other"},
    )

    assert result.response.status_code == 403, result.response.text
    assert "key_model_access_denied" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


# A param the body check refuses, inside each config or metadata container a body or a target nests
_NESTED_BANNED_PARAMS: Final = (
    pytest.param({"metadata": {"api_base": _ATTACKER_HOST}}, id="metadata"),
    pytest.param({"litellm_metadata": {"api_base": _ATTACKER_HOST}}, id="litellm-metadata"),
    pytest.param({"extra_body": {"api_base": _ATTACKER_HOST}}, id="extra-body"),
    pytest.param({"litellm_embedding_config": {"api_base": _ATTACKER_HOST}}, id="litellm-embedding-config"),
    pytest.param({"litellm_params": {"metadata": {"api_base": _ATTACKER_HOST}}}, id="litellm-params-metadata"),
)


@pytest.mark.parametrize("nested", _NESTED_BANNED_PARAMS)
@pytest.mark.parametrize("field", _list_fields())
def test_a_param_the_body_check_refuses_in_a_targets_nested_config_is_refused(field: str, nested: dict) -> None:
    """The hop merges the target into the call, so what it nests reaches the call as the body's own does"""
    from litellm.proxy.auth.auth_utils import is_request_body_safe

    router: Final = _router_without_fallback_group()
    target: Final = {"model": "primary", **nested}

    assert not _body_is_accepted(is_request_body_safe, router, nested)
    assert not _body_is_accepted(is_request_body_safe, router, {field: [target]})
    assert not _body_is_accepted(is_request_body_safe, router, {"router_settings_override": {field: [target]}})


_CONTEXT_WINDOW_ERROR: Final = (400, "invalid_request_error", "prompt is too long: 300000 tokens > 200000 maximum")
# What a refusal adds for an entry the fallback checks read both as a model group map and as a parameter override
_READ_BOTH_WAYS_NOTE: Final = (
    "The fallbacks entry {entry} names no model, so it was checked both as a model group the Router maps to its "
    "fallbacks and as a parameter override litellm's own fallback loop merges into the call."
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target_metadata",
    [
        # /v1/messages stamps the caller in litellm_metadata, which an empty one replaced
        pytest.param({"litellm_metadata": {}}, id="empty-litellm-metadata"),
        # An identity key in a target's metadata is refused before any call, a STRIPPED row of the table below; the
        # Router's own drop of one at the merge is covered in tests/unit/router_utils/test_fallback_event_handlers.py
    ],
)
async def test_a_targets_metadata_keeps_the_callers_identity_for_a_later_hops_access_check(
    monkeypatch: pytest.MonkeyPatch, target_metadata: dict
) -> None:
    """
    With enforce_fallback_model_access, a hop to `other` that fails falls back on the configured `forbidden`, which
    the access check refuses only while the hop's metadata still carries the caller
    """
    target: Final = {"model": "other", **target_metadata}
    result: Final = await _Gateway(monkeypatch, _CONTEXT_WINDOW_ERROR, other_error=(500, "api_error", "x")).send(
        {"context_window_fallbacks": [{"requested": [target]}]},
        router_fallbacks=[{"other": ["forbidden"]}],
        enforce_fallback_model_access=True,
    )

    assert result.response.status_code != 200, result.response.text
    assert result.calls["requested"] == 1
    assert result.calls["forbidden"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["key", "team"])
async def test_a_saved_settings_target_with_tags_in_its_metadata_is_still_served(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    settings: Final = {"fallbacks": [{"requested": [{"model": "other", "litellm_metadata": {"tags": ["saved"]}}]}]}
    result: Final = await _Gateway(monkeypatch).send(
        {},
        key_router_settings=settings if source == "key" else None,
        team_router_settings=settings if source == "team" else None,
    )

    assert result.response.status_code == 200, result.response.text
    assert result.calls == {"requested": 1, "other": 1, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("swap_to", "served"),
    [pytest.param("forbidden", False, id="model-the-key-may-not-call"), pytest.param("requested", True, id="allowed")],
)
async def test_a_targets_context_window_swap_on_a_router_hop_is_checked_as_the_bodys(
    monkeypatch: pytest.MonkeyPatch, swap_to: str, served: bool
) -> None:
    """The Router's hop leaves the swap unread, but the target is checked as a body is: a forbidden swap is refused"""
    target: Final = {"model": "other", "context_window_fallback_dict": {"other": swap_to}}
    result: Final = await _Gateway(monkeypatch, _CONTEXT_WINDOW_ERROR).send(
        {"context_window_fallbacks": [{"requested": [target]}]}
    )

    if served:
        assert result.response.status_code == 200, result.response.text
        assert result.calls == {"requested": 1, "other": 1, "forbidden": 0, "attacker": 0}
        return
    assert result.response.status_code in (401, 403), result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
async def test_a_targets_own_fallbacks_under_the_beta_get_the_refusals_of_anthropics_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under the beta a hop sends the target's `fallbacks` to Anthropic, so a name there with an alias is refused"""
    target: Final = {"model": "other", "fallbacks": [{"model": _ALIASED_PROVIDER_FALLBACK}]}
    result: Final = await _Gateway(monkeypatch, _CONTEXT_WINDOW_ERROR).send(
        {"context_window_fallbacks": [{"requested": [target]}]},
        beta=True,
        router_aliases={_ALIASED_PROVIDER_FALLBACK: "other"},
    )

    assert result.response.status_code == 403, result.response.text
    assert "key_model_access_denied" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
async def test_a_model_group_refused_as_a_parameter_override_says_it_was_read_both_ways(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group named `api_base` is a model group to the Router and a refused parameter to litellm's own loop"""
    group_named_api_base: Final = {
        "model_name": "api_base",
        "litellm_params": {"model": "anthropic/api_base", "api_key": _DEPLOYMENT_KEY, "api_base": _REQUESTED_HOST},
    }
    result: Final = await _Gateway(monkeypatch).send(
        {"model": "api_base", "fallbacks": [{"api_base": ["other"]}]},
        key_models=("api_base", "other"),
        extra_deployments=[group_named_api_base],
    )

    assert result.response.status_code in (400, 401), result.response.text
    assert "api_base is not allowed in request body" in result.response.text
    assert _READ_BOTH_WAYS_NOTE.format(entry="'api_base'") in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(("field", "requested_error"), _FALLBACK_FIELD_ERRORS)
@pytest.mark.parametrize("source", ["body", "router-settings-override", "key", "team"])
async def test_a_nested_list_as_a_fallback_entry_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch, field: str, requested_error: tuple[int, str, str], source: str
) -> None:
    settings: Final = {field: [["other"]]}
    body_fields: Final = {"body": settings, "router-settings-override": {"router_settings_override": settings}}
    result: Final = await _Gateway(monkeypatch, requested_error).send(
        body_fields.get(source, {}),
        key_router_settings=settings if source == "key" else None,
        team_router_settings=settings if source == "team" else None,
    )

    assert result.response.status_code == 400, result.response.text
    assert "a fallback list entry must be a model name or a mapping, got list" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["key", "team"])
async def test_a_saved_targets_own_fallbacks_under_the_beta_get_the_refusals_of_anthropics_list(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """
    Saved key and team router settings go through the body's checks: under the beta a hop sends a saved target's own
    `fallbacks` to Anthropic, so a name there that has an alias is refused. The refusal comes after authentication, so
    /v1/messages answers it in Anthropic's error shape.
    """
    target: Final = {"model": "other", "fallbacks": [{"model": _ALIASED_PROVIDER_FALLBACK}]}
    settings: Final = {"context_window_fallbacks": [{"requested": [target]}]}
    result: Final = await _Gateway(monkeypatch, _CONTEXT_WINDOW_ERROR).send(
        {},
        beta=True,
        key_router_settings=settings if source == "key" else None,
        team_router_settings=settings if source == "team" else None,
        router_aliases={_ALIASED_PROVIDER_FALLBACK: "other"},
    )

    assert result.response.status_code == 403, result.response.text
    assert f"The requested model '{_ALIASED_PROVIDER_FALLBACK}' is not available" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


def _error_of(response: "httpx.Response") -> tuple[int, object, object]:
    body: Final = response.json()
    error: Final = body.get("error") if isinstance(body, dict) else None
    return (
        response.status_code,
        error.get("type") if isinstance(error, dict) else None,
        error.get("param") if isinstance(error, dict) else None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["key", "team"])
async def test_a_saved_target_is_refused_on_a_chat_route_as_the_bodys_with_the_same_type_and_param(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """On a chat route the error envelope keeps `type` and `param`, so a saved target's refusal matches the body's"""
    settings: Final = {"fallbacks": [{"requested": ["forbidden"]}]}
    from_body: Final = await _Gateway(monkeypatch).send(dict(settings), route="/v1/chat/completions")
    from_saved: Final = await _Gateway(monkeypatch).send(
        {},
        key_router_settings=settings if source == "key" else None,
        team_router_settings=settings if source == "team" else None,
        route="/v1/chat/completions",
    )

    assert from_body.response.status_code in (401, 403), from_body.response.text
    assert _error_of(from_saved.response) == _error_of(from_body.response), from_saved.response.text
    assert from_saved.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["body", "key"])
async def test_a_keys_own_config_does_not_turn_off_the_fallback_checks(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """A key's owner writes its config, so a config on the key never lets a fallback reach a model the key may not"""
    settings: Final = {"fallbacks": [{"requested": ["forbidden"]}]}
    result: Final = await _Gateway(monkeypatch).send(
        dict(settings) if source == "body" else {},
        key_router_settings=settings if source == "key" else None,
        key_config={"max_parallel_requests": 5},
    )

    assert result.response.status_code in (401, 403), result.response.text
    assert result.calls["forbidden"] == 0


@pytest.mark.asyncio
async def test_a_target_that_brings_the_beta_gets_the_refusals_of_anthropics_list_for_its_own_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    With client-side credentials a target may carry the beta header, which the hop merges into the call, so the Router
    sends that target's own `fallbacks` to Anthropic while the request's own are LiteLLM's
    """
    target: Final = {
        "model": "other",
        "extra_headers": {"anthropic-beta": _SERVER_SIDE_FALLBACK_BETA},
        "fallbacks": [{"model": _ALIASED_PROVIDER_FALLBACK}],
    }
    result: Final = await _Gateway(monkeypatch).send(
        {"fallbacks": [target]},
        client_side_credentials=True,
        credentials_allowed=True,
        router_aliases={_ALIASED_PROVIDER_FALLBACK: "other"},
    )

    assert result.response.status_code == 403, result.response.text
    assert "key_model_access_denied" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body_fields",
    [
        pytest.param({"deployment_id": "restricted"}, id="root"),
        pytest.param({"fallbacks": [{"model": "other", "deployment_id": "restricted"}]}, id="target"),
    ],
)
async def test_a_deployment_id_is_refused_at_the_root_of_a_body_and_in_a_target(
    gateway: _Gateway, body_fields: dict[str, object]
) -> None:
    """litellm.completion calls the deployment a deployment_id names as the model, past every model check"""
    result: Final = await gateway.send(copy.deepcopy(body_fields))

    assert result.response.status_code == 401, result.response.text
    assert "deployment_id is not allowed in request body" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body_fields", "requested_answers", "calls"),
    [
        pytest.param(
            {"metadata": {"deployment_id": "my-app"}},
            True,
            {"requested": 1, "other": 0, "forbidden": 0, "attacker": 0},
            id="body",
        ),
        pytest.param(
            {"fallbacks": [{"model": "other", "metadata": {"deployment_id": "my-app"}}]},
            False,
            {"requested": 1, "other": 1, "forbidden": 0, "attacker": 0},
            id="target",
        ),
    ],
)
async def test_a_deployment_id_label_in_metadata_is_still_served(
    monkeypatch: pytest.MonkeyPatch, body_fields: dict[str, object], requested_answers: bool, calls: dict[str, int]
) -> None:
    """No call reads a deployment_id in metadata as its parameter, so a client's own label of that name is served"""
    result: Final = await _Gateway(monkeypatch, requested_answers=requested_answers).send(copy.deepcopy(body_fields))

    assert result.response.status_code == 200, result.response.text
    assert result.calls == calls


@pytest.mark.asyncio
@pytest.mark.parametrize(("first_brings_the_beta", "served"), [(True, False), (False, True)], ids=["beta", "no-beta"])
async def test_a_later_targets_own_fallbacks_are_read_both_ways_when_an_earlier_target_brings_the_beta(
    monkeypatch: pytest.MonkeyPatch, first_brings_the_beta: bool, served: bool
) -> None:
    """
    The Router keeps a target's header fields in the call on later hops, so the second target's own `fallbacks` goes to
    Anthropic once the first brought the beta: a name there with an alias is refused before any call
    """
    first: Final = {"model": "other", "extra_headers": {"anthropic-beta": _SERVER_SIDE_FALLBACK_BETA}}
    second: Final = {"model": "other", "max_tokens": 5, "fallbacks": [{"model": _ALIASED_PROVIDER_FALLBACK}]}
    result: Final = await _Gateway(monkeypatch).send(
        {"fallbacks": [first if first_brings_the_beta else {"model": "other"}, second]},
        client_side_credentials=True,
        credentials_allowed=True,
        router_aliases={_ALIASED_PROVIDER_FALLBACK: "other"},
    )

    if served:
        assert result.response.status_code == 200, result.response.text
        assert result.calls == {"requested": 1, "other": 1, "forbidden": 0, "attacker": 0}
        return
    assert result.response.status_code == 403, result.response.text
    assert "key_model_access_denied" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("bucket", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("value", [7, ["tags"], "not json"], ids=["number", "list", "string"])
async def test_a_metadata_bucket_that_is_not_an_object_is_refused_in_a_body_and_in_a_target(
    monkeypatch: pytest.MonkeyPatch, bucket: str, value: object
) -> None:
    """The request setup's own type check of a body's metadata runs on a target's, with the same 400"""
    for body_fields in ({bucket: value}, {"fallbacks": [{"model": "other", bucket: value}]}):
        result = await _Gateway(monkeypatch).send(copy.deepcopy(body_fields))
        assert result.response.status_code == 400, (body_fields, result.response.text)
        assert f"Invalid type for '{bucket}'" in result.response.text
        assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


# ── A fallback target is never validated less than a request body ────────────
#
# Each row is a key, and the outcome of a /v1/messages body that carries it at its root against the outcome of one
# whose fallback target carries it: in a list of targets, in the list a model group maps to, and in a parameter-only
# mapping. Whatever a body's root refuses, checks or strips, a target refuses: no key is accepted in a target that a
# body is refused for. SAME rows get the body's outcome; CLAIM rows are what this pull request's claim refuses in a
# target (credential, endpoint, routing and mock-response parameters), which a body's root may carry; STRIPPED rows are
# keys the request setup drops from a body's root, which a target merged into the call on a hop may not bring back;
# STAMPED rows are keys the setup writes over a body's own value, which a target on a hop would replace.


class _Rule(Enum):
    SAME = "the body's outcome"
    CLAIM = "refused in a target by the claim"
    STRIPPED = "dropped from a body, refused in a target"
    STAMPED = "written by the setup at a body's root, refused in a target"


@dataclass(frozen=True, slots=True)
class _Row:
    key: str
    value: object
    rule: _Rule
    # For a SAME row: whether a body that carries the key is refused, and so a target that carries it. For any other
    # row: whether a body's root is refused for it after authentication (the mock testing params are)
    refused: bool = True


_MOCK_TESTING_PARAMS: Final = (
    "mock_testing_fallbacks",
    "mock_testing_context_fallbacks",
    "mock_testing_content_policy_fallbacks",
    "mock_testing_rate_limit_error",
    "mock_timeout",
)
_TARGET_TABLE: Final = (
    _Row("model", "forbidden", _Rule.SAME),
    _Row("fallbacks", ["forbidden"], _Rule.SAME),
    _Row("context_window_fallbacks", [{"requested": ["forbidden"]}], _Rule.SAME),
    _Row("content_policy_fallbacks", [{"requested": ["forbidden"]}], _Rule.SAME),
    _Row("default_fallbacks", ["forbidden"], _Rule.SAME),
    _Row("context_window_fallback_dict", {"anthropic/requested": "forbidden"}, _Rule.SAME),
    _Row("router_settings_override", {"num_retries": 0}, _Rule.STAMPED, refused=False),
    _Row("model_list", [{"model_name": "forbidden", "litellm_params": {"model": "anthropic/forbidden"}}], _Rule.SAME),
    _Row("user_config", {"model_list": []}, _Rule.SAME),
    _Row("api_base", _ATTACKER_HOST, _Rule.SAME),
    _Row("base_url", _ATTACKER_HOST, _Rule.SAME),
    _Row("aws_sts_endpoint", _ATTACKER_HOST, _Rule.SAME),
    _Row("aws_profile_name", "admin", _Rule.SAME),
    _Row("deployment_id", "restricted", _Rule.SAME),
    _Row("fastest_response", True, _Rule.SAME, refused=False),
    _Row("deployment", "anthropic/other", _Rule.SAME, refused=False),
    # The prompt management feature's own interface at a body's root
    _Row("prompt_id", "stored-prompt", _Rule.SAME, refused=False),
    _Row("prompt_label", "production", _Rule.SAME, refused=False),
    _Row("prompt_version", 2, _Rule.SAME, refused=False),
    _Row("_excluded_deployment_ids", ["absent-deployment"], _Rule.SAME, refused=False),
    _Row("_retry_skipped_deployment_ids", ["absent-deployment"], _Rule.SAME, refused=False),
    _Row("router_fallbacks", ["other"], _Rule.STRIPPED, refused=False),
    _Row("_target_order", [0], _Rule.STRIPPED, refused=False),
    _Row("fallback_depth", 0, _Rule.STRIPPED, refused=False),
    _Row("fallback_model_group", ["other"], _Rule.STRIPPED, refused=False),
    _Row("callbacks", ["langfuse"], _Rule.STRIPPED, refused=False),
    _Row("client_side_timeout", True, _Rule.STRIPPED, refused=False),
    _Row("base_model", "anthropic/other", _Rule.STRIPPED, refused=False),
    _Row("metadata", {"user_api_key_user_id": "someone-else"}, _Rule.STRIPPED, refused=False),
    _Row("litellm_metadata", {"user_api_key_team_id": "another-team"}, _Rule.STRIPPED, refused=False),
    _Row("guardrails", [], _Rule.STRIPPED, refused=False),
    _Row("policies", [], _Rule.STRIPPED, refused=False),
    _Row("include_fallback_errors", True, _Rule.STRIPPED, refused=False),
    # A per-request callback param, refused at a body's root with the observability params
    _Row("turn_off_message_logging", False, _Rule.SAME),
    _Row("allowed_model_region", "eu", _Rule.STAMPED, refused=False),
    _Row("litellm_call_id", "caller-call-id", _Rule.STAMPED, refused=False),
    # The setup copies these from the caller's own headers: the caller's parameters, in a target as in a body
    _Row("litellm_trace_id", "caller-trace-id", _Rule.SAME, refused=False),
    _Row("litellm_session_id", "caller-session-id", _Rule.SAME, refused=False),
    _Row("timeout", 30, _Rule.SAME, refused=False),
    _Row("stream_timeout", 30, _Rule.SAME, refused=False),
    _Row("num_retries", 0, _Rule.SAME, refused=False),
    _Row("keepalive_seconds", 5, _Rule.SAME, refused=False),
    _Row("ttl", 60, _Rule.SAME, refused=False),
    # Written over a body's only by a proxy started with --request_timeout
    _Row("request_timeout", 30, _Rule.SAME, refused=False),
    _Row("cache", {"no-cache": True}, _Rule.STAMPED, refused=False),
    _Row("disable_fallbacks", False, _Rule.STAMPED, refused=False),
    _Row("success_callback", [], _Rule.STAMPED, refused=False),
    _Row("failure_callback", [], _Rule.STAMPED, refused=False),
    _Row("specific_deployment", True, _Rule.CLAIM, refused=False),
    _Row("custom_llm_provider", "openai", _Rule.CLAIM, refused=False),
    _Row("model_info", {"id": "forbidden-deployment"}, _Rule.CLAIM, refused=False),
    _Row("model_file_id_mapping", {"file-1": {"forbidden-deployment": "file-2"}}, _Rule.CLAIM, refused=False),
    _Row("endpoint", "/v1/other", _Rule.CLAIM, refused=False),
    _Row("azure", True, _Rule.CLAIM, refused=False),
    _Row("use_litellm_proxy", True, _Rule.CLAIM, refused=False),
    _Row("passthrough_on_no_deployment", True, _Rule.CLAIM, refused=False),
    _Row("ssl_verify", False, _Rule.CLAIM, refused=False),
    _Row("api_key", _CALLER_SUPPLIED_KEY, _Rule.CLAIM, refused=False),
    _Row("api_version", "2024-10-21", _Rule.CLAIM, refused=False),
    _Row("organization", "org-other", _Rule.CLAIM, refused=False),
    _Row("litellm_credential_name", "stored-credential", _Rule.CLAIM, refused=False),
    _Row("headers", _PARAM_VALUES["headers"], _Rule.CLAIM, refused=False),
    _Row("extra_headers", _PARAM_VALUES["headers"], _Rule.CLAIM, refused=False),
    _Row("provider_specific_header", _PARAM_VALUES["provider_specific_header"], _Rule.CLAIM, refused=False),
    _Row("extra_body", {"top_k": 1}, _Rule.CLAIM, refused=False),
    _Row("mock_response", "served by a mock", _Rule.CLAIM, refused=False),
    _Row("mock_tool_calls", [{"function": {"name": "f", "arguments": "{}"}}], _Rule.CLAIM, refused=False),
    # A body's root is refused for these by the mock testing gate, unless an admin opted in
    *(_Row(name, True, _Rule.CLAIM) for name in _MOCK_TESTING_PARAMS),
    _Row("mock_delay", 1, _Rule.CLAIM),
)


def _refused(result: _GatewayResult) -> bool:
    """Refused before any call: a 4xx, and no upstream reached"""
    return result.response.status_code // 100 == 4 and not any(result.calls.values())


def _error_param(response: "httpx.Response") -> object:
    try:
        body: Final = response.json()
    except ValueError:
        return None
    error: Final = body.get("error") if isinstance(body, dict) else None
    return error.get("param") if isinstance(error, dict) else None


def _not_refused_by_authentication(result: _GatewayResult) -> bool:
    """What reading shows of a body whose outcome after authentication depends on the provider: no auth refusal"""
    return result.response.status_code not in (400, 401, 403) or any(result.calls.values())


def _target_placements(row: _Row) -> tuple[tuple[str, list[object]], ...]:
    """The `fallbacks` lists whose target carries the row's key, by where the target stands"""
    target: Final = {"model": "other", row.key: row.value}
    return (
        ("list-of-targets", [copy.deepcopy(target)]),
        ("group-list", [{"requested": [copy.deepcopy(target)]}]),
        ("parameter-only", [{"model": "other"}, {row.key: copy.deepcopy(row.value)}]),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("row", _TARGET_TABLE, ids=lambda row: row.key)
async def test_a_fallback_target_is_never_validated_less_than_a_request_body(
    monkeypatch: pytest.MonkeyPatch, row: _Row
) -> None:
    in_body: Final = await _Gateway(monkeypatch).send({row.key: copy.deepcopy(row.value)})
    if row.rule is _Rule.SAME or row.refused:
        assert _refused(in_body) is row.refused, in_body.response.text
    else:
        # Accepted or dropped at a body's root, so a target's refusal is the claim's, not one the body shares
        assert _not_refused_by_authentication(in_body), in_body.response.text
    for placement, fallbacks in _target_placements(row):
        in_target = await _Gateway(monkeypatch).send({"fallbacks": fallbacks})
        seen = f"{placement}: {in_target.response.status_code} {in_target.response.text}"
        # Never looser: a key a body's root is refused for, a target is refused for
        assert _refused(in_target) or not _refused(in_body), seen
        if row.rule is not _Rule.SAME:
            assert _refused(in_target), seen
            continue
        assert _refused(in_target) is row.refused, seen
        if row.refused:
            assert in_target.response.status_code // 100 == in_body.response.status_code // 100, seen
            assert _error_param(in_target.response) == _error_param(in_body.response), seen


# The rows whose refusal in a target is a model name's, which both readings of a mapping name alike
_NAME_ROWS: Final = frozenset(
    {
        "model",
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "default_fallbacks",
        "context_window_fallback_dict",
    }
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [row for row in _TARGET_TABLE if (row.rule is not _Rule.SAME or row.refused) and row.key not in _NAME_ROWS],
    ids=lambda row: row.key,
)
async def test_a_mapping_read_both_ways_is_refused_as_the_parameter_override_it_is_to_litellms_own_loop(
    monkeypatch: pytest.MonkeyPatch, row: _Row
) -> None:
    """
    `["other", {key: value}]` is a `fallbacks` list in the Router's group format, so the mapping without a `model` is a
    model group map to the Router and a parameter override to litellm's own fallback loop. A key a target is refused
    for is a parameter's, so the override reading refuses the mapping, and the refusal says it was read both ways. Rows
    a body accepts, and rows refused for a model name both readings name, prove nothing here and are left out.
    """
    result: Final = await _Gateway(monkeypatch).send({"fallbacks": ["other", {row.key: copy.deepcopy(row.value)}]})

    assert _refused(result), result.response.text
    assert row.key in result.response.text
    assert _READ_BOTH_WAYS_NOTE.format(entry=repr(row.key)) in result.response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("row", [row for row in _TARGET_TABLE if row.rule is not _Rule.SAME or row.refused], ids=str)
async def test_a_target_litellms_own_fallback_loop_merges_is_refused_as_on_the_routers_hop(
    monkeypatch: pytest.MonkeyPatch, row: _Row
) -> None:
    """
    Where no Router serves the request, litellm.acompletion's own fallback loop merges the target, with no merge-point
    guard of the Router's: the entry check alone refuses it, before any call
    """
    target: Final = {"model": "openai/other", row.key: copy.deepcopy(row.value)}
    response, called = await _chat_served_by_litellm(
        monkeypatch, [target], router=True, key_models=("openai/requested", "openai/other")
    )

    assert response.status_code in (400, 401, 403), response.text
    assert called == ()


# The reader's backup group: one deployment, outside the region an end user may be restricted to
_BACKUP_IN_US: Final = MappingProxyType(
    {
        "model_name": "backup",
        "litellm_params": {
            "model": "anthropic/backup",
            "api_key": _DEPLOYMENT_KEY,
            "api_base": _OTHER_HOST,
            "region_name": "us",
        },
    }
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "served"),
    [
        pytest.param({"model": "backup", "allowed_model_region": None}, False, id="clears-the-region"),
        pytest.param({"model": "backup"}, True, id="control"),
    ],
)
async def test_a_target_may_not_replace_the_region_the_proxy_stamps(
    monkeypatch: pytest.MonkeyPatch, target: Mapping[str, object], served: bool
) -> None:
    """
    The request setup writes the key's or end user's region over a body's own, and the Router's region check reads it
    to keep the call in that region; a target merged on a hop would replace it with its own, null included.

    What this proves: a target that carries `allowed_model_region` is refused at the entry check, before any call, so
    the hop to the `us` deployment is never made. What it does not: the Router's region path itself. The harness has no
    end user restricted to a region and runs no pre-call checks, so the control is served whatever its region.
    """
    result: Final = await _Gateway(monkeypatch).send(
        {"fallbacks": [dict(target)]},
        key_models=("requested", "backup"),
        extra_deployments=[_BACKUP_IN_US],
    )

    if served:
        assert result.response.status_code == 200, result.response.text
        assert result.calls == {"requested": 1, "other": 1, "forbidden": 0, "attacker": 0}
        return
    assert result.response.status_code == 401, result.response.text
    assert "allowed_model_region" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
async def test_a_targets_own_timeout_and_retries_are_the_callers_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The setup copies the caller's own timeout and retry headers over its body's values, which applies no one else's
    limit, so a target may set its own, as a body may
    """
    result: Final = await _Gateway(monkeypatch).send(
        {"fallbacks": [{"model": "other", "timeout": 30, "num_retries": 0}]}
    )

    assert result.response.status_code == 200, result.response.text
    assert result.calls == {"requested": 1, "other": 1, "forbidden": 0, "attacker": 0}


async def _chat_served_by_litellm(
    monkeypatch: pytest.MonkeyPatch,
    fallbacks: Sequence[object],
    *,
    router: bool,
    key_models: tuple[str, ...] = (),
    requested_error: tuple[int, str] = (500, "x"),
) -> tuple["httpx.Response", tuple[str, ...]]:
    """
    /v1/chat/completions on a proxy where no Router serves `openai/requested`: one started for that model with no
    Router, or a Router that passes models it does not list through to litellm. Either way litellm.acompletion's own
    fallback loop reads the body's `fallbacks`. The upstream fails `requested` with `requested_error` unless a fallback
    set its temperature; returns the response and the model of each upstream call.
    """
    import httpx
    import respx

    import litellm
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import app, hash_token
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.router import RouterGeneralSettings

    llm_router: Final = (
        Router(
            model_list=[{"model_name": "listed", "litellm_params": {"model": "openai/listed"}}],
            router_general_settings=RouterGeneralSettings(pass_through_all_models=True),
        )
        if router
        else None
    )
    cache: Final = DualCache()
    await cache.async_set_cache(
        key=hash_token(_CALLER_KEY),
        value=UserAPIKeyAuth(
            api_key=_CALLER_KEY,
            token=hash_token(_CALLER_KEY),
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=list(key_models),
        ),
    )
    for name, value in {
        "master_key": "sk-master",
        "prisma_client": MagicMock(),
        "user_api_key_cache": cache,
        "llm_router": llm_router,
        "llm_model_list": llm_router.get_model_list() if llm_router is not None else None,
        "user_model": None if router else "openai/requested",
        "user_api_base": None,
        "general_settings": {},
        "user_custom_auth": None,
        "proxy_logging_obj": ProxyLogging(user_api_key_cache=cache),
    }.items():
        monkeypatch.setattr(proxy_server, name, value)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("OPENAI_API_KEY", _DEPLOYMENT_KEY)
    called: Final[list[str]] = []

    def upstream_reply(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.read())
        called.append(sent["model"])
        if sent["model"] == "requested" and "temperature" not in sent:
            status, error_message = requested_error
            return httpx.Response(status, json={"error": {"message": error_message, "type": "invalid_request_error"}})
        message: Final = {"role": "assistant", "content": "served"}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": sent["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    payload: Final = {
        "model": "openai/requested",
        "messages": [{"role": "user", "content": "hi"}],
        "fallbacks": list(fallbacks),
    }
    with respx.mock(assert_all_called=False) as upstream:
        upstream.post("https://api.openai.com/v1/chat/completions").mock(side_effect=upstream_reply)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy.test") as client:
            response = await client.post(
                "/v1/chat/completions", json=payload, headers={"Authorization": f"Bearer {_CALLER_KEY}"}
            )
    return response, tuple(called)


_MOCK_TOOL_CALLS: Final = MappingProxyType({"mock_tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("router", "fallbacks", "refused_param"),
    [
        # With no Router every named fallback fails the routing check, so only a list that names none gets through
        pytest.param(False, [dict(_MOCK_TOOL_CALLS)], "mock_tool_calls", id="no-router"),
        pytest.param(True, ["a", {"mock_response": ["x"]}], "mock_response", id="pass-through-router"),
        pytest.param(True, ["a", dict(_MOCK_TOOL_CALLS)], "mock_tool_calls", id="pass-through-router-tool-calls"),
    ],
)
async def test_a_mapping_the_direct_completion_fallback_merges_gets_the_target_checks(
    monkeypatch: pytest.MonkeyPatch, router: bool, fallbacks: Sequence[object], refused_param: str
) -> None:
    """
    The Router would read the mapping as a model group map and never merge it, but litellm.acompletion's own fallback
    loop merges every mapping of its list into the call
    """
    response, called = await _chat_served_by_litellm(monkeypatch, fallbacks, router=router)

    assert response.status_code in (400, 401), response.text
    assert f"a fallback target may not carry {refused_param}" in response.text
    assert called == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("router", "fallbacks"),
    [
        pytest.param(False, [{"temperature": 0}], id="no-router-parameter-override"),
        pytest.param(True, ["openai/other"], id="pass-through-router-model"),
        pytest.param(True, ["a", {"temperature": 0}], id="pass-through-router-parameter-override"),
    ],
)
async def test_a_documented_fallback_shape_is_still_served_where_no_router_serves_the_request(
    monkeypatch: pytest.MonkeyPatch, router: bool, fallbacks: Sequence[object]
) -> None:
    response, called = await _chat_served_by_litellm(monkeypatch, fallbacks, router=router)

    assert response.status_code == 200, response.text
    assert "served" in response.text
    assert called[0] == "requested"


@pytest.mark.asyncio
async def test_a_targets_context_window_swap_on_the_direct_path_names_a_checked_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    litellm.acompletion's own fallback loop merges the target, and on a context-window error the completion wrapper
    calls the model the target's context_window_fallback_dict names, so that model is checked as the body's would be
    """
    swap: Final = {"openai/requested": "openai/forbidden"}
    target: Final = {"model": "openai/requested", "context_window_fallback_dict": swap}
    response, called = await _chat_served_by_litellm(
        monkeypatch,
        [target],
        router=True,
        key_models=("openai/requested",),
        requested_error=(400, "This model's maximum context length is 8192 tokens."),
    )

    assert response.status_code in (401, 403), response.text
    assert "'openai/forbidden'" in response.text
    assert called == ()


@pytest.mark.asyncio
async def test_a_parameter_overrides_context_window_swap_on_the_direct_path_names_a_checked_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A mapping without a `model` in `fallbacks` is a parameter override to litellm's own fallback loop, which merges it
    into the call, so the model its context_window_fallback_dict swaps in is checked as the body's would be
    """
    override: Final = {"context_window_fallback_dict": {"openai/requested": "openai/forbidden"}}
    response, called = await _chat_served_by_litellm(
        monkeypatch,
        [override],
        router=True,
        key_models=("openai/requested",),
        requested_error=(400, "This model's maximum context length is 8192 tokens."),
    )

    assert response.status_code in (401, 403), response.text
    assert "'openai/forbidden'" in response.text
    assert _READ_BOTH_WAYS_NOTE.format(entry="'context_window_fallback_dict'") in response.text
    assert called == ()


@pytest.mark.asyncio
async def test_a_parameter_override_refused_as_a_model_group_says_it_was_read_both_ways(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`{"stop": ["END"]}` beside a name is a group map to the Router and an override to litellm's own loop"""
    response, called = await _chat_served_by_litellm(
        monkeypatch,
        ["openai/other", {"stop": ["END"]}],
        router=True,
        key_models=("openai/requested", "openai/other"),
    )

    assert response.status_code in (401, 403), response.text
    assert "'END'" in response.text
    assert _READ_BOTH_WAYS_NOTE.format(entry="'stop'") in response.text
    assert called == ()


# A fallback target outside the team's models, in LiteLLM's own list and in Anthropic's list under the beta
_FALLBACKS_OUTSIDE_THE_TEAM: Final = (
    pytest.param([{"requested": ["forbidden"]}], False, id="litellm-fallbacks"),
    pytest.param([{"model": "forbidden"}], True, id="anthropic-fallbacks"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("fallbacks", "beta"), _FALLBACKS_OUTSIDE_THE_TEAM)
async def test_an_all_team_models_key_gets_its_fallback_targets_checked_against_the_team(
    gateway: _Gateway, fallbacks: Sequence[object], beta: bool
) -> None:
    result = await gateway.send(
        {"fallbacks": fallbacks},
        beta=beta,
        team_router_settings={},
        key_models=("all-team-models",),
    )

    assert result.response.status_code in (401, 403), result.response.text
    assert result.calls["forbidden"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("fallbacks", "beta"), _FALLBACKS_OUTSIDE_THE_TEAM)
async def test_a_key_without_a_model_list_gets_its_fallback_targets_checked_against_the_team(
    gateway: _Gateway, fallbacks: Sequence[object], beta: bool
) -> None:
    """The key may call every model, so the team's model list is the restriction a fallback target must pass"""
    result: Final = await gateway.send({"fallbacks": fallbacks}, beta=beta, team_router_settings={}, key_models=())

    assert result.response.status_code == 403, result.response.text
    assert "team_model_access_denied" in result.response.text
    assert "The requested model 'forbidden' is not available" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fallbacks", "beta", "calls"),
    [
        # The Router serves LiteLLM's list from the next group; under the beta Anthropic serves its own list itself
        pytest.param(
            [{"requested": ["other"]}],
            False,
            {"requested": 1, "other": 1, "forbidden": 0, "attacker": 0},
            id="litellm-fallbacks",
        ),
        pytest.param(
            [{"model": "other"}],
            True,
            {"requested": 1, "other": 0, "forbidden": 0, "attacker": 0},
            id="anthropic-fallbacks",
        ),
    ],
)
async def test_a_fallback_target_in_the_team_still_serves_a_key_without_a_model_list(
    monkeypatch: pytest.MonkeyPatch, fallbacks: Sequence[object], beta: bool, calls: Mapping[str, int]
) -> None:
    result: Final = await _Gateway(monkeypatch, requested_answers=beta).send(
        {"fallbacks": fallbacks}, beta=beta, team_router_settings={}, key_models=()
    )

    assert result.response.status_code == 200, result.response.text
    assert result.calls == calls


# A name Anthropic would serve as itself, aliased to `other`, a model the key and the team may call
_ALIASED_PROVIDER_FALLBACK: Final = "claude-opus-aliased"


@pytest.mark.asyncio
@pytest.mark.parametrize("alias_kind", ["key", "team", "router", "global"])
async def test_a_name_in_anthropics_fallbacks_that_has_an_alias_is_refused(
    monkeypatch: pytest.MonkeyPatch, alias_kind: str
) -> None:
    """Every model check would pass the name as its alias target, while Anthropic gets it as named"""
    import litellm

    aliases: Final = {_ALIASED_PROVIDER_FALLBACK: "other"}
    if alias_kind == "global":
        monkeypatch.setattr(litellm, "model_alias_map", aliases)
    result: Final = await _Gateway(monkeypatch, requested_answers=True).send(
        {"fallbacks": [{"model": _ALIASED_PROVIDER_FALLBACK}]},
        beta=True,
        team_router_settings={} if alias_kind == "team" else None,
        key_aliases=aliases if alias_kind == "key" else None,
        team_aliases=aliases if alias_kind == "team" else None,
        router_aliases=aliases if alias_kind == "router" else None,
    )

    assert result.response.status_code == 403, result.response.text
    assert "key_model_access_denied" in result.response.text
    assert f"The requested model '{_ALIASED_PROVIDER_FALLBACK}' is not available" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


# A name Anthropic serves as itself, and the deployments the router serves it from
_OPUS: Final = "claude-opus-4-8"


def _opus_deployment(
    litellm_model: str,
    access_group: str,
    model_name: str = _OPUS,
    *,
    team_id: str | None = None,
    deployment_id: str | None = None,
    custom_llm_provider: str | None = None,
) -> Mapping[str, object]:
    """
    A deployment under ``model_name`` in one access group; its host counts as the forbidden one, which nothing calls. A
    team's deployment is published under that name for the team, the way the proxy stores one.
    """
    team_fields: Final = {"team_id": team_id, "team_public_model_name": model_name} if team_id is not None else {}
    id_field: Final = {"id": deployment_id} if deployment_id is not None else {}
    provider_field: Final = {"custom_llm_provider": custom_llm_provider} if custom_llm_provider is not None else {}
    return MappingProxyType[str, object](
        {
            "model_name": model_name if team_id is None else f"model_name_{team_id}_0",
            "litellm_params": {
                "model": litellm_model,
                "api_key": _DEPLOYMENT_KEY,
                "api_base": _FORBIDDEN_HOST,
                **provider_field,
            },
            "model_info": {"access_groups": [access_group], **team_fields, **id_field},
        }
    )


# Opus in `expensive` and Haiku in `cheap` under the one name: Anthropic answers the name with Opus
_MIXED_GROUP: Final = (
    _opus_deployment(f"anthropic/{_OPUS}", "expensive"),
    _opus_deployment("anthropic/claude-haiku-4-5", "cheap"),
)
# The same model on Anthropic and on Bedrock: the Bedrock deployment sends Bedrock's id, never the name
_ANTHROPIC_AND_BEDROCK_GROUP: Final = (
    _opus_deployment(f"anthropic/{_OPUS}", "expensive"),
    _opus_deployment("bedrock/us.anthropic.claude-opus-4-8-v1:0", "cheap"),
)
_KEY_WITH_CHEAP: Final = ("requested", "cheap")
_KEY_WITH_THE_NAME: Final = ("requested", _OPUS)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployments", "key_models", "key_aliases", "routing_groups"),
    [
        pytest.param(
            (_opus_deployment("anthropic/claude-haiku-4-5", "cheap"),),
            _KEY_WITH_CHEAP,
            {},
            (),
            id="one-deployment-of-another-model",
        ),
        pytest.param(_MIXED_GROUP, _KEY_WITH_CHEAP, {}, (), id="mixed-group"),
        pytest.param(_MIXED_GROUP, _KEY_WITH_THE_NAME, {}, (), id="mixed-group-key-holds-the-name"),
        pytest.param(
            _MIXED_GROUP,
            ("requested", "cheap", "claude-haiku-4-5"),
            {"expensive": "claude-haiku-4-5"},
            (),
            id="mixed-group-key-alias-named-like-a-group",
        ),
        pytest.param(
            (
                _opus_deployment(f"anthropic/{_OPUS}", "claude-haiku-grant"),
                _opus_deployment("anthropic/claude-haiku-4-5", "cheap"),
            ),
            ("requested", "cheap", "claude-haiku-*"),
            {},
            (),
            id="mixed-group-key-wildcard-matching-a-group",
        ),
        pytest.param(_ANTHROPIC_AND_BEDROCK_GROUP, _KEY_WITH_CHEAP, {}, (), id="other-provider"),
        pytest.param(_ANTHROPIC_AND_BEDROCK_GROUP, _KEY_WITH_THE_NAME, {}, (), id="other-provider-key-holds-the-name"),
        pytest.param(
            (_opus_deployment("anthropic/claude-haiku-*", "cheap", model_name="claude-opus-*"),),
            _KEY_WITH_CHEAP,
            {},
            (),
            id="pattern-route-to-another-model",
        ),
        pytest.param(
            (_opus_deployment("anthropic/claude-haiku-4-5", "cheap", model_name="claude-opus-*"),),
            _KEY_WITH_CHEAP,
            {},
            (),
            id="pattern-route-to-a-fixed-model",
        ),
        pytest.param(
            (
                _opus_deployment(f"anthropic/{_OPUS}", "expensive"),
                _opus_deployment("anthropic/claude-haiku-4-5", "cheap", team_id="haiku-team"),
            ),
            ("requested", "expensive"),
            {},
            (),
            id="another-teams-deployment-under-the-name",
        ),
        pytest.param(
            (_opus_deployment("anthropic/claude-haiku-*", "cheap", model_name="claude-opus-*", team_id="haiku-team"),),
            _KEY_WITH_THE_NAME,
            {},
            (),
            id="another-teams-pattern-route",
        ),
        pytest.param(
            (_opus_deployment("anthropic/claude-haiku-4-5", "cheap", model_name="haiku-member"),),
            _KEY_WITH_THE_NAME,
            {},
            ({"group_name": _OPUS, "models": ["haiku-member"], "routing_strategy": "simple-shuffle"},),
            id="routing-group-of-another-model",
        ),
        pytest.param(
            (_opus_deployment(f"anthropic/{_OPUS}", "expensive", model_name="opus-member"),),
            _KEY_WITH_THE_NAME,
            {},
            ({"group_name": _OPUS, "models": ["opus-member"], "routing_strategy": "simple-shuffle"},),
            # A routing group is refused as such, as an alias is, even when its members all send the name
            id="routing-group-whose-member-sends-the-name",
        ),
        pytest.param(
            (
                _opus_deployment(
                    "anthropic/claude-haiku-4-5", "cheap", model_name="haiku-deployment", deployment_id=_OPUS
                ),
            ),
            _KEY_WITH_THE_NAME,
            {},
            (),
            id="deployment-id-of-another-model",
        ),
        pytest.param(
            (_opus_deployment(_OPUS, "cheap", model_name="opus-elsewhere", custom_llm_provider="openai"),),
            _KEY_WITH_THE_NAME,
            {},
            (),
            id="provider-model-on-another-provider",
        ),
    ],
)
async def test_a_name_in_anthropics_fallbacks_the_router_can_serve_as_another_model_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    deployments: Sequence[Mapping[str, object]],
    key_models: tuple[str, ...],
    key_aliases: Mapping[str, str],
    routing_groups: Sequence[Mapping[str, object]],
) -> None:
    """
    A deployment under the name, for any team or through any route, sends another model: a check could grant the name
    through it while Anthropic serves the name, so the name is refused whatever the key holds
    """
    result: Final = await _Gateway(monkeypatch, requested_answers=True).send(
        {"fallbacks": [{"model": _OPUS}]},
        beta=True,
        key_models=key_models,
        key_aliases=key_aliases,
        extra_deployments=deployments,
        routing_groups=routing_groups,
    )

    assert result.response.status_code == 403, result.response.text
    assert "key_model_access_denied" in result.response.text
    assert '"param":"fallbacks"' in result.response.text.replace(" ", "")
    assert f"The requested model '{_OPUS}' is not available" in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
async def test_a_routing_group_in_anthropics_fallbacks_is_refused_where_a_team_resolves_its_member_to_another_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The proxy-wide member sends the name, but the caller's team publishes a deployment of another model under the
    member's name, which the router serves the group from for that team
    """
    result: Final = await _Gateway(monkeypatch, requested_answers=True).send(
        {"fallbacks": [{"model": _OPUS}]},
        beta=True,
        team_router_settings={},
        key_models=_KEY_WITH_THE_NAME,
        team_models=_KEY_WITH_THE_NAME,
        extra_deployments=(
            _opus_deployment(f"anthropic/{_OPUS}", "expensive", model_name="member"),
            _opus_deployment("anthropic/claude-haiku-4-5", "cheap", model_name="member", team_id="fallback-team"),
        ),
        routing_groups=({"group_name": _OPUS, "models": ["member"], "routing_strategy": "simple-shuffle"},),
    )

    assert result.response.status_code == 403, result.response.text
    assert "key_model_access_denied" in result.response.text
    assert '"param":"fallbacks"' in result.response.text.replace(" ", "")
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployments", "key_models"),
    [
        pytest.param(
            (_opus_deployment(f"anthropic/{_OPUS}", "expensive"), _opus_deployment(f"anthropic/{_OPUS}", "premium")),
            ("requested", "expensive"),
            id="two-anthropic-deployments-key-holds-one-group",
        ),
        pytest.param(
            (_opus_deployment(f"anthropic/{_OPUS}", "expensive"), _opus_deployment(f"anthropic/{_OPUS}", "premium")),
            ("requested", "premium"),
            id="two-anthropic-deployments-key-holds-the-other-group",
        ),
        pytest.param((), _KEY_WITH_THE_NAME, id="no-deployment-key-holds-the-name"),
        pytest.param(
            (_opus_deployment("anthropic/*", "every-claude", model_name="anthropic/*"),),
            ("requested", "every-claude"),
            id="pattern-route-that-forwards-the-name",
        ),
        pytest.param(
            (_opus_deployment(f"anthropic/{_OPUS}", "expensive", model_name="opus-by-id", deployment_id=_OPUS),),
            _KEY_WITH_THE_NAME,
            id="deployment-id-that-sends-the-name",
        ),
    ],
)
async def test_a_name_in_anthropics_fallbacks_that_means_one_model_still_serves_the_request(
    monkeypatch: pytest.MonkeyPatch, deployments: Sequence[Mapping[str, object]], key_models: tuple[str, ...]
) -> None:
    result: Final = await _Gateway(monkeypatch, requested_answers=True).send(
        {"fallbacks": [{"model": _OPUS}]}, beta=True, key_models=key_models, extra_deployments=deployments
    )

    assert result.response.status_code == 200, result.response.text
    assert result.calls == {"requested": 1, "other": 0, "forbidden": 0, "attacker": 0}
    # Anthropic gets the list as sent, so the refusals above are the one place the name could switch models
    assert [body["fallbacks"] for body in result.requested_bodies if isinstance(body, dict)] == [[{"model": _OPUS}]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fallback", "key_aliases", "status", "error_type"),
    [
        # The alias refusal runs in the key's own check, before the team check that would answer 503
        pytest.param(_ALIASED_PROVIDER_FALLBACK, {_ALIASED_PROVIDER_FALLBACK: "other"}, 403, "key_model_access_denied"),
        # A clean name reaches the team check, whose 503 is no longer recovered into a fallback identity
        pytest.param("other", {}, 503, "no_db_connection"),
    ],
    ids=["aliased-name", "clean-name"],
)
async def test_a_name_in_anthropics_fallbacks_is_refused_while_a_team_without_a_recorded_grant_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, fallback: str, key_aliases: Mapping[str, str], status: int, error_type: str
) -> None:
    """
    The team lookup is swallowed into no team and the token records no team models, so the team check answers 503;
    allow_requests_on_db_unavailable does not recover a request whose Anthropic fallbacks name a model
    """
    from litellm.proxy.auth import user_api_key_auth

    safe_fetch: Final = user_api_key_auth._safe_fetch

    async def swallows_the_team_lookup(label: str, awaitable: Coroutine[object, object, object]) -> object:
        if label == "team":
            awaitable.close()
            return None
        return await safe_fetch(label, awaitable)

    monkeypatch.setattr(user_api_key_auth, "_safe_fetch", swallows_the_team_lookup)
    result: Final = await _Gateway(monkeypatch, requested_answers=True).send(
        {"fallbacks": [{"model": fallback}]},
        beta=True,
        team_router_settings={},
        token_team_models=(),
        key_aliases=key_aliases,
        allow_requests_on_db_unavailable=True,
    )

    assert result.response.status_code == status, result.response.text
    assert error_type in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


def _membership_read_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """common_checks' read of the caller's team membership, before any fallback check, misses its deadline"""
    from litellm.proxy.auth import auth_checks
    from litellm.proxy.db.db_lookup_gate import DBLookupDeadlineExceeded

    monkeypatch.setattr(
        auth_checks,
        "get_team_membership",
        AsyncMock(side_effect=DBLookupDeadlineExceeded("team membership", 1.0)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fallback", "deployments", "status", "error_type"),
    [
        # The targets cannot be checked while the membership cannot be read, so the outage answers as unrecovered
        pytest.param("other", (), 503, "no_db_connection", id="clean-name"),
        # A virtual key's own check runs both refusals before common_checks reads the membership
        pytest.param(
            _OPUS,
            (_opus_deployment("anthropic/claude-haiku-4-5", "cheap"),),
            403,
            "key_model_access_denied",
            id="name-the-router-serves-as-another-model",
        ),
    ],
)
async def test_a_database_outage_is_not_recovered_for_a_request_that_names_models_in_anthropics_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    fallback: str,
    deployments: Sequence[Mapping[str, object]],
    status: int,
    error_type: str,
) -> None:
    _membership_read_times_out(monkeypatch)
    result: Final = await _Gateway(monkeypatch, requested_answers=True).send(
        {"fallbacks": [{"model": fallback}]},
        beta=True,
        team_router_settings={},
        key_models=("requested", "other", _OPUS),
        user_id="caller",
        extra_deployments=deployments,
        allow_requests_on_db_unavailable=True,
    )

    assert result.response.status_code == status, result.response.text
    assert error_type in result.response.text
    assert result.calls == {"requested": 0, "other": 0, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body_fields", "beta", "forwarded_fallbacks"),
    [
        pytest.param({}, True, None, id="no-fallbacks"),
        pytest.param({"fallbacks": "default"}, True, "default", id="default-names-no-model"),
        pytest.param({"fallbacks": [{"requested": ["other"]}]}, False, None, id="litellm-fallbacks-without-the-beta"),
    ],
)
async def test_a_database_outage_still_recovers_a_request_whose_anthropic_fallbacks_name_no_model(
    monkeypatch: pytest.MonkeyPatch, body_fields: dict[str, object], beta: bool, forwarded_fallbacks: str | None
) -> None:
    """The membership read fails as above; with no model in Anthropic's fallbacks the fallback identity still serves"""
    _membership_read_times_out(monkeypatch)
    result: Final = await _Gateway(monkeypatch, requested_answers=True).send(
        body_fields,
        beta=beta,
        team_router_settings={},
        user_id="caller",
        allow_requests_on_db_unavailable=True,
    )

    assert result.response.status_code == 200, result.response.text
    assert result.calls == {"requested": 1, "other": 0, "forbidden": 0, "attacker": 0}
    sent: Final = [body for body in result.requested_bodies if isinstance(body, dict)]
    assert [body.get("fallbacks") for body in sent] == [forwarded_fallbacks]


@pytest.mark.asyncio
async def test_common_checks_refuses_an_aliased_name_in_anthropics_fallbacks_before_the_unread_team_check() -> None:
    """The same order where common_checks is the only check, as for a JWT: the alias refusal comes before the 503"""
    team_key: Final = UserAPIKeyAuth(team_id="fallback-team", aliases={_ALIASED_PROVIDER_FALLBACK: "requested"})

    with pytest.raises(ModelAccessDeniedProxyException, match=f"'{_ALIASED_PROVIDER_FALLBACK}'") as exc_info:
        await _common_checks_with_fallbacks(
            [_ALIASED_PROVIDER_FALLBACK], team_key, request=_messages_request(_SERVER_SIDE_FALLBACK_BETA)
        )

    assert exc_info.value.type == ProxyErrorTypes.key_model_access_denied
    assert exc_info.value.code == "403"


@pytest.mark.parametrize(
    ("deployment", "sends"),
    [
        pytest.param({"litellm_params": {"model": f"anthropic/{_OPUS}"}}, True, id="anthropic-prefix"),
        pytest.param(
            {"litellm_params": {"model": _OPUS, "custom_llm_provider": "anthropic"}}, True, id="declared-provider"
        ),
        pytest.param({"litellm_params": {"model": "anthropic/claude-haiku-4-5"}}, False, id="another-model"),
        pytest.param({"litellm_params": {"model": "bedrock/us.anthropic.claude-opus-4-8-v1:0"}}, False, id="bedrock"),
        pytest.param({"litellm_params": {"model": "unmapped-provider/xyz"}}, False, id="unreadable-provider"),
        pytest.param({"litellm_params": {"model": 42}}, False, id="unreadable-model"),
        pytest.param({"litellm_params": {}}, False, id="no-model"),
        pytest.param({"litellm_params": None}, False, id="unreadable-params"),
        pytest.param({}, False, id="no-params"),
    ],
)
def test_a_deployment_sends_a_name_to_anthropic_only_when_its_provider_and_model_read_as_that_name(
    deployment: object, sends: bool
) -> None:
    from litellm.proxy.auth.auth_checks import _deployment_sends_name_to_anthropic

    assert _deployment_sends_name_to_anthropic(deployment, _OPUS) is sends


async def _common_checks_with_fallbacks(
    fallbacks: Sequence[str],
    valid_token: UserAPIKeyAuth,
    *,
    team_object: LiteLLM_TeamTable | None = None,
    membership: LiteLLM_TeamMembership | None = None,
    user_object: LiteLLM_UserTable | None = None,
    project_object: LiteLLM_ProjectTableCachedObj | None = None,
    request: Request | None = None,
) -> bool:
    """
    common_checks on a /v1/messages request for `requested` that names `fallbacks`, or none when it is empty; they are
    Anthropic's own list when `request` carries the server-side-fallback beta
    """
    from litellm.proxy.auth.auth_checks import common_checks

    with patch("litellm.proxy.auth.auth_checks.get_team_membership", new=AsyncMock(return_value=membership)):
        return await common_checks(
            request_body={
                "model": "requested",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "hi"}],
                **({"fallbacks": [{"model": name} for name in fallbacks]} if fallbacks else {}),
            },
            team_object=team_object,
            user_object=user_object,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/v1/messages",
            llm_router=None,
            proxy_logging_obj=MagicMock(),
            valid_token=valid_token,
            request=request if request is not None else MagicMock(spec=Request),
            skip_budget_checks=True,
            project_object=project_object,
        )


# A forbidden target alone and after an allowed one, so a check that reads only the first target fails the second
_FORBIDDEN_FALLBACKS: Final = (
    pytest.param(["other"], id="forbidden-only"),
    pytest.param(["requested", "other"], id="allowed-then-forbidden"),
)


@pytest.mark.asyncio
async def test_a_fallback_target_outside_the_team_members_own_models_is_refused() -> None:
    member: Final = UserAPIKeyAuth(user_id="caller", team_id="fallback-team")
    team: Final = LiteLLM_TeamTable(team_id="fallback-team", models=["requested", "other"])
    membership: Final = LiteLLM_TeamMembership(
        user_id="caller",
        team_id="fallback-team",
        litellm_budget_table=LiteLLM_BudgetTable(allowed_models=["requested"]),
    )

    assert await _common_checks_with_fallbacks(["requested"], member, team_object=team, membership=membership) is True
    with pytest.raises(ModelAccessDeniedProxyException, match="'other'") as exc_info:
        await _common_checks_with_fallbacks(["other"], member, team_object=team, membership=membership)

    assert exc_info.value.type == ProxyErrorTypes.team_model_access_denied
    assert exc_info.value.code == "403"


@pytest.mark.asyncio
async def test_a_fallback_target_outside_a_personal_keys_user_models_is_refused() -> None:
    personal_key: Final = UserAPIKeyAuth(user_id="caller")
    user: Final = LiteLLM_UserTable(user_id="caller", models=["requested"])

    assert await _common_checks_with_fallbacks(["requested"], personal_key, user_object=user) is True
    with pytest.raises(ModelAccessDeniedProxyException, match="'other'") as exc_info:
        await _common_checks_with_fallbacks(["other"], personal_key, user_object=user)

    assert exc_info.value.type == ProxyErrorTypes.user_model_access_denied
    assert exc_info.value.code == "403"


@pytest.mark.asyncio
async def test_a_fallback_target_outside_the_projects_models_is_refused() -> None:
    project_key: Final = UserAPIKeyAuth(project_id="fallback-project")
    project: Final = LiteLLM_ProjectTableCachedObj(project_id="fallback-project", models=["requested"])

    assert await _common_checks_with_fallbacks(["requested"], project_key, project_object=project) is True
    with pytest.raises(ModelAccessDeniedProxyException, match="'other'") as exc_info:
        await _common_checks_with_fallbacks(["other"], project_key, project_object=project)

    assert exc_info.value.type == ProxyErrorTypes.project_model_access_denied
    assert exc_info.value.code == "403"


@pytest.mark.asyncio
@pytest.mark.parametrize("forbidden", _FORBIDDEN_FALLBACKS)
async def test_a_team_key_whose_team_cannot_be_read_is_held_to_the_team_models_its_token_records(
    forbidden: Sequence[str],
) -> None:
    """The team lookup failed, so the team object is absent; the team models on the token still bind every target"""
    team_key: Final = UserAPIKeyAuth(team_id="fallback-team", team_models=["requested"])

    assert await _common_checks_with_fallbacks(["requested"], team_key) is True
    with pytest.raises(ModelAccessDeniedProxyException, match="'other'") as exc_info:
        await _common_checks_with_fallbacks(forbidden, team_key)

    assert exc_info.value.type == ProxyErrorTypes.team_model_access_denied
    assert exc_info.value.code == "403"


@pytest.mark.asyncio
async def test_fallback_targets_are_refused_while_a_team_without_a_recorded_grant_cannot_be_read() -> None:
    team_key: Final = UserAPIKeyAuth(team_id="fallback-team")

    assert await _common_checks_with_fallbacks([], team_key) is True
    with pytest.raises(ProxyException, match="cannot be checked while Team=fallback-team cannot be read") as exc_info:
        await _common_checks_with_fallbacks(["requested"], team_key)

    assert exc_info.value.type == ProxyErrorTypes.no_db_connection
    assert exc_info.value.code == "503"


def _agent_access_group_ceiling(models: frozenset[str] | None) -> CeilingResolver:
    """An agent's attached access groups with a fixed model ceiling, or no attached group when ``models`` is None"""

    async def resolve(_agent_id: str) -> AgentAccessGroupCeiling | None:
        if models is None:
            return None
        return AgentAccessGroupCeiling(
            access_group_ids=("fallback-group",), models=models, mcp_server_ids=frozenset(), agent_ids=frozenset()
        )

    return resolve


@pytest.mark.asyncio
@pytest.mark.parametrize("forbidden", _FORBIDDEN_FALLBACKS)
async def test_a_fallback_target_outside_a_managed_agents_model_grants_is_refused(forbidden: Sequence[str]) -> None:
    managed: Final = UserAPIKeyAuth(token="managed-token")
    managed.managed_agent_policy = AgentResponse(
        agent_id="managed", agent_name="Managed", agent_card_params={}, object_permission={"models": ["requested"]}
    )

    assert await _common_checks_with_fallbacks(["requested"], managed) is True
    with pytest.raises(ModelAccessDeniedProxyException, match="'other'") as exc_info:
        await _common_checks_with_fallbacks(forbidden, managed)

    assert exc_info.value.type == ProxyErrorTypes.agent_model_access_denied
    assert exc_info.value.code == "403"


@pytest.mark.asyncio
@pytest.mark.parametrize("forbidden", _FORBIDDEN_FALLBACKS)
async def test_a_fallback_target_outside_an_agents_access_groups_is_refused(forbidden: Sequence[str]) -> None:
    from litellm.proxy.auth import auth_checks

    agent_key: Final = UserAPIKeyAuth(token="agent-token", agent_id="agent-1")
    ceiling: Final = _agent_access_group_ceiling(frozenset({"requested"}))

    with patch(
        "litellm.proxy.auth.auth_checks._check_agent_access_group_model_access",
        new=partial(auth_checks._check_agent_access_group_model_access, resolve_ceiling=ceiling),
    ):
        assert await _common_checks_with_fallbacks(["requested"], agent_key) is True
        with pytest.raises(ModelAccessDeniedProxyException, match="'other'") as exc_info:
            await _common_checks_with_fallbacks(forbidden, agent_key)

    assert exc_info.value.type == ProxyErrorTypes.agent_model_access_denied
    assert exc_info.value.code == "403"


@pytest.mark.asyncio
@pytest.mark.parametrize("forbidden", _FORBIDDEN_FALLBACKS)
async def test_a_fallback_target_outside_the_invoking_teams_models_is_refused(forbidden: Sequence[str]) -> None:
    """An agent key acting for a caller is held to the caller's team, so a target outside that team is refused"""
    from litellm.proxy.auth import auth_checks

    agent_key: Final = UserAPIKeyAuth(token="agent-token", agent_id="agent-1")
    agent_key.agent_caller = AgentCaller(user_id="caller", team_id="caller-team")
    caller_team: Final = LiteLLM_TeamTable(team_id="caller-team", models=["requested"])

    async def load_team(_valid_token: UserAPIKeyAuth) -> LiteLLM_TeamTable | None:
        return caller_team

    async def load_user(_valid_token: UserAPIKeyAuth) -> LiteLLM_UserTable | None:
        return None

    with (
        patch(
            "litellm.proxy.auth.auth_checks._check_agent_access_group_model_access",
            new=partial(
                auth_checks._check_agent_access_group_model_access, resolve_ceiling=_agent_access_group_ceiling(None)
            ),
        ),
        patch(
            "litellm.proxy.auth.auth_checks._check_agent_caller_model_access",
            new=partial(auth_checks._check_agent_caller_model_access, load_team=load_team, load_user=load_user),
        ),
    ):
        assert await _common_checks_with_fallbacks(["requested"], agent_key) is True
        with pytest.raises(ModelAccessDeniedProxyException, match="'other'") as exc_info:
            await _common_checks_with_fallbacks(forbidden, agent_key)

    assert exc_info.value.type == ProxyErrorTypes.team_model_access_denied
    assert exc_info.value.code == "403"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "send_kwargs",
    [
        pytest.param({"body_fields": {"fallbacks": [{"requested": ["other"]}]}}, id="body-fallbacks"),
        pytest.param(
            {
                "body_fields": {"fallbacks": "default"},
                "beta": True,
                "key_router_settings": {"fallbacks": [{"requested": ["other"]}]},
            },
            id="key-router-settings-beside-anthropic-fallbacks",
        ),
        pytest.param(
            {"body_fields": {}, "team_router_settings": {"fallbacks": [{"requested": ["other"]}]}},
            id="team-router-settings",
        ),
    ],
)
async def test_an_allowed_plain_fallback_target_still_serves_the_request(gateway: _Gateway, send_kwargs):
    result = await gateway.send(**send_kwargs)

    assert result.response.status_code == 200, result.response.text
    assert result.calls == {"requested": 1, "other": 1, "forbidden": 0, "attacker": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(("beta", "other_calls"), [(True, 0), (False, 1)], ids=["beta", "no-beta"])
async def test_a_client_router_settings_override_carries_no_litellm_fallbacks_under_the_beta(
    gateway: _Gateway, beta: bool, other_calls: int
):
    result = await gateway.send({"router_settings_override": {"fallbacks": [{"requested": ["other"]}]}}, beta=beta)

    assert result.calls == {"requested": 1, "other": other_calls, "forbidden": 0, "attacker": 0}, result.response.text


@pytest.mark.parametrize("field", _target_fields())
def test_every_credential_endpoint_or_mock_param_on_a_fallback_target_is_refused(field: str):
    from litellm.proxy.auth.auth_utils import is_request_body_safe

    router = _router_without_fallback_group()
    params = (*_credential_and_endpoint_params(), *_mock_params())
    if field == "context_window_fallback_dict":
        assert not _body_is_accepted(is_request_body_safe, router, {field: {"anthropic/x": "https://attacker"}})
        return
    accepted = tuple(
        param
        for param in params
        if _body_is_accepted(is_request_body_safe, router, {field: [{"model": "primary", param: "x"}]})
        or _body_is_accepted(
            is_request_body_safe, router, {"router_settings_override": {field: [{"model": "primary", param: "x"}]}}
        )
    )
    assert accepted == ()


def _body_is_accepted(is_request_body_safe, router: Router, extra: dict) -> bool:
    try:
        is_request_body_safe({"model": "primary", **extra}, {}, router, "primary")
    except ValueError:
        return False
    return True


def test_fallback_key_scan_covers_renamed_router_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The scan must catch M2's renamed key, M3's helper parameter, and M4's proxy rename."""
    module_name: Final = "fallback_key_scan_fixture"
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    router_source = tmp_path / "router_source.py"
    router_source.write_text('def request_fallbacks(request_data):\n    return request_data.get("spare_models")\n')
    proxy_source = tmp_path / "proxy_source.py"
    proxy_source.write_text('def transform(data):\n    data["fallbacks"] = data.pop("backup_fallbacks")\n')

    read = _fallback_keys_read(module_name, router_source)

    assert read == frozenset({"spare_models"})
    assert _proxy_renamed_fallback_input_fields((proxy_source,), read | frozenset({"fallbacks"})) == frozenset(
        {"backup_fallbacks"}
    )


# What the request setup writes at a body's root and leaves the caller's own: it fills the key in only when the body
# lacks it, or merges the body's value with its own. A key here may ride on a fallback target.
_SETUP_DEFAULTS: Final = MappingProxyType(
    {
        "model": "the body's own model, resolved by alias; a target's own model gets the model checks",
        "drop_params": "set only when the body has none (should_auto_drop_params_for_agentic_cli)",
        "stream_options": "merged: include_usage is added; fallback_target_params_over_call keeps the call's keys",
        "metadata": "merged: fallback_target_params_over_call keeps every key the call holds",
        "litellm_metadata": "merged: fallback_target_params_over_call keeps every key the call holds",
    }
)
# What the setup copies from the caller's own request headers over its body's values: no one else's limit applies to
# the write, so a target may set its own, as a body may. Each with why.
_SETUP_CALLERS_OWN: Final = MappingProxyType(
    {
        "timeout": "the caller's x-litellm-timeout header, no limit; the merge marks a target's client_side_timeout",
        "stream_timeout": "the caller's x-litellm-stream-timeout header, no limit; marked as timeout is",
        "num_retries": "the caller's x-litellm-num-retries header, no limit",
        "keepalive_seconds": "the caller's keepalive header; the deployment's opt-in gates the root's value when read",
        "ttl": "the s-maxage of the caller's Cache-Control header, no limit",
        "litellm_trace_id": "the caller's trace headers, or one the setup adds when the body has none",
        "litellm_session_id": "the caller's session header or metadata; the session policy only adds a missing one",
    }
)
# Written over a body's own only under an administrator's setting, so refused in a target then: each key, with the
# module and global that switch it on (the command line's values, and the key hash as `user`)
_SETUP_PINS: Final = MappingProxyType(
    {
        "temperature": ("litellm.proxy.proxy_server", "user_temperature"),
        "max_tokens": ("litellm.proxy.proxy_server", "user_max_tokens"),
        "api_base": ("litellm.proxy.proxy_server", "user_api_base"),
        "request_timeout": ("litellm.proxy.proxy_server", "user_request_timeout"),
        "user": ("litellm", "overwrite_user_with_key_hash"),
    }
)
# The setup's keys a proxy-wide allow_client_side_credentials admits on a target, by the claim's own rule
_ADMITTED_BY_THE_CREDENTIAL_OPT_IN: Final = frozenset(
    {"api_key", "api_base", "api_version", "organization", "headers", "extra_headers", "provider_specific_header"}
)


def _computed_setup_keys() -> Mapping[str, frozenset[str]]:
    """The keys behind each name the setup computes a root key from, resolved by reading its line"""
    from litellm.proxy import litellm_pre_call_utils as setup
    from litellm.types.utils import StandardCallbackDynamicParams

    buckets: Final = frozenset({"metadata", "litellm_metadata"})
    return MappingProxyType(
        {
            "_metadata_variable_name": buckets,
            "metadata_variable_name": buckets,
            "_metadata_field": buckets,
            "_internal_key": frozenset(setup._UNTRUSTED_ROOT_CONTROL_FIELDS),  # pyright: ignore[reportPrivateUsage]
            "field": frozenset(setup._CLIENT_PRICING_CONTROL_FIELDS),  # pyright: ignore[reportPrivateUsage]
            "k": frozenset(StandardCallbackDynamicParams.__annotations__),  # the team's callback vars
            "key": frozenset({"api_base", "api_key", "api_version"}),  # a model's credential config, when absent
        }
    )


def _request_setup_root_keys() -> tuple[frozenset[str], frozenset[str]]:
    """
    The root keys the request setup writes, sets by default or drops, read from its source: litellm_pre_call_utils, and
    the pre-call logic of common_request_processing. Returns the keys it names and the names it computes.
    """
    import litellm

    root: Final = Path(litellm.__file__).parent / "proxy"
    sources: Final = (
        ("litellm.proxy.litellm_pre_call_utils", root / "litellm_pre_call_utils.py", None),
        (
            "litellm.proxy.common_request_processing",
            root / "common_request_processing.py",
            frozenset({"common_processing_pre_call_logic"}),
        ),
    )
    named: set[str] = set()
    computed: set[str] = set()
    for module_name, path, functions in sources:
        tree = ast.parse(path.read_text())
        module = importlib.import_module(module_name)
        defs = {
            node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        def record(node: ast.expr, module: ModuleType = module) -> None:
            value = node.value if isinstance(node, ast.Constant) else None
            if isinstance(node, ast.Name):
                value = getattr(module, node.id, None)
                if not isinstance(value, str):
                    computed.add(node.id)
                    return
            if isinstance(value, str):
                named.add(value)
            else:
                computed.add(ast.unparse(node))

        def is_data(node: ast.expr) -> bool:
            return (isinstance(node, ast.Name) and node.id in ("data", "request_data")) or (
                isinstance(node, ast.Attribute)
                and node.attr == "data"
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            )

        def scan(function: ast.AST, seen: set[str], defs: Mapping[str, ast.AST] = defs) -> None:
            for node in ast.walk(function):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else (node.target,)
                    if isinstance(node, (ast.AnnAssign, ast.AugAssign))
                    else ()
                )
                for target in targets:
                    if isinstance(target, ast.Subscript) and is_data(target.value):
                        record(target.slice)
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args):
                    continue
                if not is_data(node.func.value):
                    continue
                if node.func.attr in ("setdefault", "pop"):
                    record(node.args[0])
                if node.func.attr == "update":
                    # The keys of the dict a helper returns into the request data
                    called = node.args[0]
                    callee = (
                        called.func.attr
                        if isinstance(called, ast.Call) and isinstance(called.func, ast.Attribute)
                        else called.func.id
                        if isinstance(called, ast.Call) and isinstance(called.func, ast.Name)
                        else None
                    )
                    helper = defs.get(callee) if callee is not None else None
                    if helper is None:
                        computed.add(ast.unparse(called))
                        continue
                    if callee not in seen:
                        seen.add(callee)
                        scan(helper, seen)
                    for returned in ast.walk(helper):
                        if isinstance(returned, ast.Return) and isinstance(returned.value, ast.Dict):
                            for key in returned.value.keys:
                                if key is not None:
                                    record(key)

        for name, function in defs.items():
            if functions is None or name in functions:
                scan(function, set())
    return frozenset(named), frozenset(computed)


def test_every_root_key_the_request_setup_writes_or_drops_is_refused_in_a_target_or_the_callers_own(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    A key the request setup writes over a body's own with a value that is not the caller's (the key's, team's or end
    user's settings, the proxy's own), or drops from it, would be replaced or brought back by a target merged on a hop,
    so the entry check refuses it in a target, naming it; a key the setup copies from the caller's own headers, only
    defaults or merges stays the caller's own. A new key the setup writes fails this test until it is classified.
    """
    from litellm.proxy.auth.auth_utils import (
        _proxy_pinned_call_params,  # pyright: ignore[reportPrivateUsage]
        is_request_body_safe,
    )

    named, computed = _request_setup_root_keys()
    resolved: Final = _computed_setup_keys()
    assert computed <= frozenset(resolved), computed - frozenset(resolved)
    keys: Final = named.union(*(resolved[name] for name in computed))
    router: Final = _router_without_fallback_group()

    def refusal(key: str, general_settings: dict) -> str:
        target = {"model": "primary", key: {}}  # a mapping, so no key is refused for its value's shape
        try:
            is_request_body_safe({"model": "primary", "fallbacks": [target]}, general_settings, router, "primary")
        except (ValueError, ProxyException) as error:
            return str(getattr(error, "message", error))
        return ""

    callers_own: Final = frozenset(_SETUP_CALLERS_OWN)
    pins: Final = frozenset(_SETUP_PINS)
    refused_keys: Final = keys - frozenset(_SETUP_DEFAULTS) - callers_own - pins
    assert {"allowed_model_region", "router_settings_override", "cache", "litellm_call_id"} <= refused_keys
    assert [key for key in sorted(refused_keys) if key not in refusal(key, {})] == []
    with_opt_in: Final = {"allow_client_side_credentials": True}
    assert [
        key for key in sorted(refused_keys - _ADMITTED_BY_THE_CREDENTIAL_OPT_IN) if key not in refusal(key, with_opt_in)
    ] == []

    assert callers_own <= keys
    assert [key for key in sorted(callers_own) if refusal(key, {})] == []

    # api_base is the claim's without the credential opt-in, whatever the command line says
    switched: Final = sorted(pins - _ADMITTED_BY_THE_CREDENTIAL_OPT_IN)
    assert pins <= keys
    assert _proxy_pinned_call_params() == frozenset()
    assert [key for key in switched if refusal(key, {})] == []
    for module_name, global_name in _SETUP_PINS.values():
        monkeypatch.setattr(importlib.import_module(module_name), global_name, True)
    assert _proxy_pinned_call_params() == pins
    assert [key for key in switched if key not in refusal(key, {})] == []


def test_the_router_reads_no_fallback_kwarg_that_auth_does_not_check():
    """
    Every kwarg the Router or proxy promotes into a fallback target must be a field auth checks, so a rebase cannot
    open a path auth misses. The merge-point test covers aliased or computed keys that this static scan cannot see.
    """

    import litellm
    from litellm.proxy.litellm_pre_call_utils import _UNTRUSTED_ROOT_CONTROL_FIELDS
    from litellm.proxy.route_llm_request import GATED_MOCK_PARAM_NAMES

    root = Path(litellm.__file__).parent
    modules: dict[str, Path] = {
        "litellm.router": root / "router.py",
        "litellm.proxy.route_llm_request": root / "proxy" / "route_llm_request.py",
        "litellm.utils": root / "utils.py",
        "litellm.main": root / "main.py",
        "litellm.proxy.litellm_pre_call_utils": root / "proxy" / "litellm_pre_call_utils.py",
        "litellm.proxy.common_request_processing": root / "proxy" / "common_request_processing.py",
        "litellm.messages.dispatch": root / "messages" / "dispatch.py",
        **{
            f"litellm.router_utils.{path.stem}": path
            for path in (root / "router_utils").glob("*.py")
            if path.stem != "__init__"
        },
    }
    router_keys: frozenset[str] = frozenset().union(
        *(_fallback_keys_read(name, path) for name, path in modules.items())
    )
    stripped_router_state = frozenset({"fallback_depth", "fallback_model_group", "_mid_stream_fallback_controls"})
    caller_flags = frozenset({"disable_fallbacks", "_disable_fallbacks", "max_fallbacks", "include_fallback_errors"})
    not_a_kwarg = frozenset({"_ageneric_api_call_with_fallbacks"})
    hop_counters = frozenset({"attempted_fallbacks"})
    # A target's own model, read to tell a target from a model group map: a name the walker yields for the model
    # checks, never a kwarg that holds fallbacks (as a request kwarg it is the requested model, checked as such)
    target_model_name = frozenset({"model"})

    from litellm.router_utils.fallback_event_handlers import CONTEXT_WINDOW_FALLBACK_DICT_FIELD, FALLBACK_LIST_FIELDS

    auth_checked = frozenset((*FALLBACK_LIST_FIELDS, CONTEXT_WINDOW_FALLBACK_DICT_FIELD))
    proxy_renamed_keys = _proxy_renamed_fallback_input_fields(
        (root / "proxy").rglob("*.py"), (router_keys - target_model_name) | auth_checked
    )
    read = router_keys | proxy_renamed_keys
    assert stripped_router_state <= frozenset(_UNTRUSTED_ROOT_CONTROL_FIELDS)
    exempt: Final = stripped_router_state | caller_flags | not_a_kwarg | hop_counters | target_model_name
    unchecked = read - auth_checked - exempt
    assert unchecked - frozenset(GATED_MOCK_PARAM_NAMES) == frozenset()
    assert "router_fallbacks" in read


def _fallback_keys_read(module_name: str, path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text())
    module_constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    }

    def resolve(node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            value = module_constants.get(node.id, getattr(importlib.import_module(module_name), node.id, None))
            return value if isinstance(value, str) else None
        return None

    def mapping_key(node: ast.AST, parameter_names: frozenset[str]) -> str | None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            owner = node.func.value
            if node.func.attr in ("get", "pop", "setdefault") and isinstance(owner, ast.Name):
                if owner.id in parameter_names:
                    return resolve(node.args[0])
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in parameter_names:
            return resolve(node.slice)
        return None

    def output_values(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.expr]:
        for node in ast.walk(function):
            if isinstance(node, ast.Return) and node.value is not None:
                yield node.value
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                if any(isinstance(target, ast.Name) and "fallback" in target.id.lower() for target in targets):
                    yield node.value

    def function_keys(function: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
        arguments = (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
        parameter_names = frozenset(argument.arg for argument in arguments)

        def output_keys() -> Iterator[str]:
            for value in output_values(function):
                for node in ast.walk(value):
                    key = mapping_key(node, parameter_names)
                    if key is not None:
                        yield key

        return frozenset(output_keys())

    fallback_function_keys = tuple(
        function_keys(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and "fallback" in node.name.lower()
    )
    literal_keys = (
        resolve(element)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set))
        for element in node.elts
        if isinstance(element, (ast.Constant, ast.Name))
    )
    named_module_constants = (value for name, value in module_constants.items() if "fallback" in name.lower())
    named_keys = frozenset(
        key
        for key in (*literal_keys, *named_module_constants)
        if isinstance(key, str) and "fallback" in key and key.isidentifier()
    )
    return named_keys | frozenset().union(*fallback_function_keys)


def _proxy_renamed_fallback_input_fields(paths: Iterable[Path], fallback_fields: frozenset[str]) -> frozenset[str]:
    def literal_key(node: ast.AST) -> str | None:
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    def source_key(value: ast.expr) -> str | None:
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.args:
            if value.func.attr in ("get", "pop", "setdefault"):
                return literal_key(value.args[0])
        return None

    def renamed_keys(path: Path) -> frozenset[str]:
        tree = ast.parse(path.read_text())

        def assignment_keys() -> Iterator[str]:
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                for target in targets:
                    if not isinstance(target, ast.Subscript):
                        continue
                    target_key = literal_key(target.slice)
                    renamed_key = source_key(node.value)
                    if target_key in fallback_fields and renamed_key is not None:
                        yield renamed_key

        return frozenset(assignment_keys())

    return frozenset().union(*(renamed_keys(path) for path in paths))


class _ProbeUpstream:
    """The probe's three real HTTP upstreams (gate780a probe_exfil.py): requested fails, other and attacker answer."""

    def __init__(self) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.seen: list[tuple[str, str | None]] = []
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                import json

                sent = json.loads(self.rfile.read(int(self.headers["content-length"])))
                name = self.server.upstream_name  # pyright: ignore[reportAttributeAccessIssue]  # set below
                upstream.seen.append((name, self.headers.get("x-api-key")))
                status = 500 if name == "requested" else 200
                reply = (
                    {"type": "error", "error": {"type": "api_error", "message": "mock outage"}}
                    if status == 500
                    else _messages_reply(sent.get("model", ""))
                )
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(reply).encode())

            def log_message(self, *args: object) -> None:
                pass

        self._servers = tuple(ThreadingHTTPServer(("127.0.0.1", 0), Handler) for _ in range(3))
        for server, name in zip(self._servers, ("requested", "other", "attacker")):
            server.upstream_name = name  # pyright: ignore[reportAttributeAccessIssue]  # read by the handler
            threading.Thread(target=server.serve_forever, daemon=True).start()
        self.requested, self.other, self.attacker = (f"http://127.0.0.1:{s.server_address[1]}" for s in self._servers)

    def close(self) -> None:
        for server in self._servers:
            server.shutdown()
            server.server_close()


@pytest.mark.asyncio
async def test_probe_exfil_router_fallbacks_api_base_never_gets_the_deployment_key(monkeypatch: pytest.MonkeyPatch):
    from unittest.mock import MagicMock as _MagicMock

    import litellm
    from litellm.proxy.auth.auth_utils import is_request_body_safe
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy.route_llm_request import route_request

    upstream = _ProbeUpstream()
    try:
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        router = Router(
            model_list=[
                {
                    "model_name": name,
                    "litellm_params": {
                        "model": f"anthropic/claude-{name}",
                        "api_key": _DEPLOYMENT_KEY,
                        "api_base": host,
                    },
                }
                for name, host in (("requested", upstream.requested), ("other", upstream.other))
            ],
            num_retries=0,
        )
        body = {
            "model": "requested",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "router_fallbacks": [{"model": "other", "api_base": upstream.attacker}],
        }
        for field in ("fallbacks", "router_fallbacks"):
            probe_body = {**{k: v for k, v in body.items() if k != "router_fallbacks"}, field: body["router_fallbacks"]}
            with pytest.raises(ValueError, match="api_base"):
                is_request_body_safe(request_body=probe_body, general_settings={}, llm_router=router, model="requested")

        request = _MagicMock(spec=Request)
        request.url = _MagicMock()
        request.url.path = "/v1/messages"
        request.url.__str__.return_value = "http://proxy.test/v1/messages"
        request.method = "POST"
        request.query_params = {}
        request.headers = {"anthropic-beta": _SERVER_SIDE_FALLBACK_BETA}
        request.client = _MagicMock()
        request.client.host = "127.0.0.1"
        data = await add_litellm_data_to_request(
            data=dict(body),
            request=request,
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key"),
            proxy_config=_MagicMock(),
            general_settings={},
            version="test",
        )
        with pytest.raises(litellm.InternalServerError):
            await (await route_request(data=data, llm_router=router, user_model=None, route_type="anthropic_messages"))
    finally:
        upstream.close()

    assert upstream.seen == [("requested", _DEPLOYMENT_KEY)]
