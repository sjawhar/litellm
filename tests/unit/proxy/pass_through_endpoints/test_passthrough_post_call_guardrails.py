"""
Tests for post-call guardrail invocation on pass-through endpoints.

Verifies that apply_guardrail(input_type="response") is called for
non-streaming pass-through responses. Addresses issue #20270.
"""

import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.proxy._types import ProxyException

_PT_MOD = "litellm.proxy.pass_through_endpoints.pass_through_endpoints"
_COLLECT = "litellm.proxy.pass_through_endpoints.passthrough_guardrails.PassthroughGuardrailHandler.collect_guardrails"

_GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [{"text": "Hello"}],
            }
        }
    ]
}


def _make_user_api_key_dict(**overrides):
    d = MagicMock()
    d.api_key = "sk-test"
    d.user_id = "user-1"
    d.team_id = "team-1"
    d.org_id = None
    d.request_route = "/vertex_ai/v1/projects/p/locations/l/publishers/google/models/gemini:generateContent"
    for k, v in overrides.items():
        setattr(d, k, v)
    return d


def _make_httpx_response(body: dict, status_code: int = 200) -> httpx.Response:
    content = json.dumps(body).encode("utf-8")
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=content,
        request=httpx.Request("POST", "https://example.com/v1/generateContent"),
    )


def _make_mock_request():
    mock_request = MagicMock()
    mock_request.method = "POST"
    mock_request.query_params = {}
    mock_request.headers = MagicMock()
    mock_request.headers.copy.return_value = {}
    return mock_request


from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    pass_through_request,
)


def _common_patches(mock_proxy_logging, mock_response):
    """Return a combined context manager for the patches shared by all tests."""
    mock_async_client = AsyncMock()
    mock_async_client_obj = MagicMock()
    mock_async_client_obj.client = mock_async_client

    mock_pt_logging = MagicMock()
    mock_pt_logging.pass_through_async_success_handler = AsyncMock()

    patches = [
        patch(
            f"{_PT_MOD}.HttpPassThroughEndpointHelpers.non_streaming_http_request_handler",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
        patch(f"{_PT_MOD}._is_streaming_response", return_value=False),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging),
        patch(f"{_PT_MOD}.pass_through_endpoint_logging", mock_pt_logging),
        patch(f"{_PT_MOD}.get_async_httpx_client", return_value=mock_async_client_obj),
        patch(f"{_PT_MOD}.read_request_body", new_callable=AsyncMock, return_value={}),
        patch(f"{_PT_MOD}.safe_get_request_headers", return_value={}),
    ]

    stack = ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


@pytest.mark.asyncio
class TestPassthroughPostCallGuardrails:

    @patch(_COLLECT, return_value=["rubrik"])
    async def test_post_call_success_hook_called_when_guardrails_configured(
        self,
        mock_collect,
    ):
        """post_call_success_hook should fire when guardrails are configured."""
        mock_response = _make_httpx_response(_GEMINI_RESPONSE)

        mock_proxy_logging = MagicMock()
        mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
        mock_proxy_logging.post_call_success_hook = AsyncMock(
            return_value=_GEMINI_RESPONSE
        )
        mock_proxy_logging.post_call_response_headers_hook = AsyncMock(return_value={})

        with _common_patches(mock_proxy_logging, mock_response):
            await pass_through_request(
                request=_make_mock_request(),
                target="https://example.com/v1/generateContent",
                custom_headers={"Content-Type": "application/json"},
                user_api_key_dict=_make_user_api_key_dict(),
                stream=False,
            )

        mock_proxy_logging.post_call_success_hook.assert_awaited_once()
        call_kwargs = mock_proxy_logging.post_call_success_hook.call_args
        assert call_kwargs.kwargs["response"] == _GEMINI_RESPONSE

    @patch(_COLLECT, return_value=[])
    async def test_post_call_success_hook_skipped_when_no_guardrails(
        self,
        mock_collect,
    ):
        """post_call_success_hook should NOT fire when no guardrails are configured."""
        mock_response = _make_httpx_response(_GEMINI_RESPONSE)

        mock_proxy_logging = MagicMock()
        mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
        mock_proxy_logging.post_call_success_hook = AsyncMock()
        mock_proxy_logging.post_call_response_headers_hook = AsyncMock(return_value={})

        with _common_patches(mock_proxy_logging, mock_response):
            result = await pass_through_request(
                request=_make_mock_request(),
                target="https://example.com/v1/generateContent",
                custom_headers={"Content-Type": "application/json"},
                user_api_key_dict=_make_user_api_key_dict(),
                stream=False,
            )

        mock_proxy_logging.post_call_success_hook.assert_not_awaited()
        assert result.status_code == 200

    @patch(_COLLECT, return_value=["rubrik"])
    async def test_modify_response_exception_returns_error(
        self,
        mock_collect,
    ):
        """ModifyResponseException from guardrail should return 200 with provider-agnostic error."""
        response_body = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"name": "dangerous_tool", "args": {}}}
                        ],
                    }
                }
            ]
        }
        mock_response = _make_httpx_response(response_body)

        mock_proxy_logging = MagicMock()
        mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
        mock_proxy_logging.post_call_success_hook = AsyncMock(
            side_effect=ModifyResponseException(
                message="Tool dangerous_tool blocked by policy",
                model="gemini-2.0-flash",
                request_data={},
                guardrail_name="rubrik",
            )
        )
        mock_proxy_logging.post_call_failure_hook = AsyncMock()

        with _common_patches(mock_proxy_logging, mock_response):
            result = await pass_through_request(
                request=_make_mock_request(),
                target="https://example.com/v1/generateContent",
                custom_headers={"Content-Type": "application/json"},
                user_api_key_dict=_make_user_api_key_dict(),
                stream=False,
            )

        mock_proxy_logging.post_call_failure_hook.assert_awaited_once()
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["error"]["type"] == "content_filter"
        assert body["error"]["message"] == "Tool dangerous_tool blocked by policy"
        assert body["error"]["guardrail_name"] == "rubrik"
        assert body["error"]["model"] == "gemini-2.0-flash"

    @patch(_COLLECT, return_value=["rubrik"])
    async def test_deny_forwards_guardrail_logging_info_to_failure_hook(
        self,
        mock_collect,
    ):
        """A post-call guardrail deny (non-ModifyResponseException) records its
        standard_logging_guardrail_information on the hook_data dict; the failure
        handler must forward that info to post_call_failure_hook so downstream
        loggers (e.g. the otel guardrail span) still see it. Regression for the
        block path dropping it."""
        mock_response = _make_httpx_response(_GEMINI_RESPONSE)

        def _block(*, data, user_api_key_dict, response):
            metadata = data.setdefault("metadata", {})
            metadata.setdefault("standard_logging_guardrail_information", []).append(
                {"guardrail_name": "rubrik", "guardrail_status": "guardrail_intervened"}
            )
            raise HTTPException(status_code=400, detail={"error": "blocked"})

        captured = {}

        async def _capture_failure(**kwargs):
            captured.update(kwargs)

        mock_proxy_logging = MagicMock()
        mock_proxy_logging.pre_call_hook = AsyncMock(return_value={})
        mock_proxy_logging.post_call_success_hook = AsyncMock(side_effect=_block)
        mock_proxy_logging.post_call_failure_hook = AsyncMock(
            side_effect=_capture_failure
        )

        with _common_patches(mock_proxy_logging, mock_response):
            with pytest.raises(ProxyException):
                await pass_through_request(
                    request=_make_mock_request(),
                    target="https://example.com/v1/generateContent",
                    custom_headers={"Content-Type": "application/json"},
                    user_api_key_dict=_make_user_api_key_dict(),
                    stream=False,
                )

        mock_proxy_logging.post_call_failure_hook.assert_awaited_once()
        entries = captured["request_data"]["metadata"][
            "standard_logging_guardrail_information"
        ]
        assert any(e.get("guardrail_name") == "rubrik" for e in entries)


@pytest.mark.asyncio
class TestUnifiedGuardrailCallTypeResolution:

    async def test_pass_through_call_type_resolved_from_logging_obj(self):
        """Unified guardrail should resolve call_type from logging_obj for pass-through."""
        from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
            UnifiedLLMGuardrails,
        )

        unified = UnifiedLLMGuardrails()

        mock_guardrail = MagicMock(spec=CustomGuardrail)
        mock_guardrail.guardrail_name = "test-guardrail"
        mock_guardrail.should_run_guardrail.return_value = True

        mock_logging_obj = MagicMock()
        mock_logging_obj.call_type = "pass_through_endpoint"

        user_api_key_dict = _make_user_api_key_dict()

        data = {
            "guardrail_to_apply": mock_guardrail,
            "litellm_logging_obj": mock_logging_obj,
        }

        response_body = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}

        mock_handler_instance = AsyncMock()
        mock_handler_instance.process_output_response = AsyncMock(
            return_value=response_body
        )
        mock_handler_class = MagicMock(return_value=mock_handler_instance)

        from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail import (
            unified_guardrail as unified_guardrail_module,
        )
        from litellm.types.utils import CallTypes

        with patch.object(
            unified_guardrail_module,
            "load_guardrail_translation_mappings",
            lambda: {CallTypes.pass_through: mock_handler_class},
        ):
            result = await unified.async_post_call_success_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                response=response_body,
            )

        mock_handler_instance.process_output_response.assert_awaited_once()


def test_modify_response_exception_importable_from_both_paths():
    """ModifyResponseException re-export from custom_guardrail must stay in sync."""
    from litellm.exceptions import ModifyResponseException as FromExceptions
    from litellm.integrations.custom_guardrail import (
        ModifyResponseException as FromGuardrail,
    )

    assert FromExceptions is FromGuardrail


_CLIENT_BUCKETS = {
    "metadata": {"user_id": "s1", "model_group": "client-chosen"},
    "litellm_metadata": {"model_group": "client-chosen"},
}
_PROVIDER_ROUTES = (
    pytest.param(
        "/anthropic/v1/messages",
        "https://api.anthropic.com/v1/messages",
        "anthropic",
        {"model": "claude-ok", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]},
        "claude-ok",
        id="anthropic",
    ),
    pytest.param(
        "/gemini/v1beta/models/gemini-2.5-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        "gemini",
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        "gemini-2.5-flash",
        id="gemini",
    ),
)


async def _failure_row_model_group(request_data, route):
    from datetime import datetime

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    writer = MagicMock(spec=DBSpendUpdateWriter)
    writer.update_database = AsyncMock()
    await ProxyDBLogger(spend_writer=lambda: writer).async_post_call_failure_hook(
        request_data=request_data,
        original_exception=Exception("refused"),
        user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key", request_route=route),
    )
    now = datetime.now()
    return get_logging_payload(
        kwargs=writer.update_database.call_args.kwargs["kwargs"], response_obj=None, start_time=now, end_time=now
    )["model_group"]


async def _drive_provider_pass_through(mock_proxy_logging, upstream_response, path, target, provider, body):
    from fastapi import Request

    from litellm.proxy._types import UserAPIKeyAuth

    request = Request({"type": "http", "method": "POST", "path": path, "headers": [], "query_string": b""})
    with _common_patches(mock_proxy_logging, upstream_response):
        with patch(f"{_PT_MOD}.read_request_body", new_callable=AsyncMock, return_value=body):
            return await pass_through_request(
                request=request,
                target=target,
                custom_headers={"Content-Type": "application/json"},
                user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key", request_route=path),
                stream=False,
                custom_llm_provider=provider,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("path, target, provider, request_body, route_model", _PROVIDER_ROUTES)
@patch(_COLLECT, return_value=["rubrik"])
async def test_response_guardrail_block_failure_row_takes_the_routes_model_group(
    mock_collect, path, target, provider, request_body, route_model
):
    """A response guardrail that blocks a built-in provider pass-through reply hands back the hook's own
    request dict, so the failure spend row must take model_group from the route, not from the body"""

    async def block(data, user_api_key_dict, response):
        raise ModifyResponseException(message="blocked", model=route_model, request_data=data, guardrail_name="rubrik")

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.pre_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["data"])
    mock_proxy_logging.post_call_success_hook = AsyncMock(side_effect=block)
    mock_proxy_logging.post_call_failure_hook = AsyncMock()
    mock_proxy_logging.post_call_response_headers_hook = AsyncMock(return_value={})

    result = await _drive_provider_pass_through(
        mock_proxy_logging, _make_httpx_response({"ok": True}), path, target, provider, {**request_body, **_CLIENT_BUCKETS}
    )

    assert json.loads(result.body)["error"]["type"] == "content_filter"
    request_data = mock_proxy_logging.post_call_failure_hook.call_args.kwargs["request_data"]
    assert await _failure_row_model_group(request_data, path) == route_model


@pytest.mark.asyncio
@pytest.mark.parametrize("path, target, provider, request_body, route_model", _PROVIDER_ROUTES)
@patch(_COLLECT, return_value=[])
async def test_upstream_error_failure_row_takes_the_routes_model_group(
    mock_collect, path, target, provider, request_body, route_model
):
    """An upstream 4xx on a built-in provider pass-through is logged from the body plus the pass-through
    kwargs, so the failure spend row must take model_group from the route, not from the body"""
    mock_proxy_logging = MagicMock()
    mock_proxy_logging.pre_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["data"])
    mock_proxy_logging.post_call_failure_hook = AsyncMock()
    mock_proxy_logging.post_call_response_headers_hook = AsyncMock(return_value={})

    await _drive_provider_pass_through(
        mock_proxy_logging,
        _make_httpx_response({"error": {"type": "invalid_request_error"}}, 400),
        path,
        target,
        provider,
        {**request_body, **_CLIENT_BUCKETS},
    )

    request_data = mock_proxy_logging.post_call_failure_hook.call_args.kwargs["request_data"]
    assert await _failure_row_model_group(request_data, path) == route_model
