"""Tests for litellm.litellm_core_utils.fallback_utils."""

from typing import Final

import pytest
import httpx

import litellm
from litellm.litellm_core_utils.core_helpers import fallback_target_params_over_call, process_response_headers
from litellm.litellm_core_utils.fallback_utils import (
    async_completion_with_fallbacks,
)


@pytest.mark.asyncio
async def test_fallback_dict_not_mutated(monkeypatch):
    fallback_dict = {"model": "fallback-model", "temperature": 0.2}
    original_fallback_dict = dict(fallback_dict)

    attempted_models: list[str] = []

    async def _fake_acompletion(*, model: str, **kwargs):
        attempted_models.append(model)
        if model == "primary-model":
            raise Exception("primary failed")
        return {"model": model, "temperature": kwargs.get("temperature")}

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    # Call 1: primary fails, fallback dict succeeds
    response_1 = await async_completion_with_fallbacks(
        model="primary-model",
        kwargs={"fallbacks": [fallback_dict]},
    )
    assert response_1["model"] == "fallback-model"
    assert fallback_dict == original_fallback_dict

    # Call 2: re-use the same dict object; it should still work and remain unchanged
    response_2 = await async_completion_with_fallbacks(
        model="primary-model",
        kwargs={"fallbacks": [fallback_dict]},
    )
    assert response_2["model"] == "fallback-model"
    assert fallback_dict == original_fallback_dict

    assert attempted_models == [
        "primary-model",
        "fallback-model",
        "primary-model",
        "fallback-model",
    ]


@pytest.mark.asyncio
async def test_async_completion_with_fallbacks_sets_attempted_fallbacks_header():
    """
    When a fallback succeeds, the response must carry the
    `x-litellm-attempted-fallbacks` header so the proxy and other callers can
    detect that a fallback occurred. Without it,
    `_override_openai_response_model` stamps the requested model back over the
    fallback model used. See issue #28241.
    """
    response = await async_completion_with_fallbacks(
        model="openai/primary-llm",
        messages=[{"role": "user", "content": "hi"}],
        api_key="fake-key",
        mock_response=Exception("forced failure"),
        kwargs={
            "fallbacks": [
                {
                    "model": "openai/backup-llm",
                    "api_key": "fake-key",
                    "mock_response": "backup-resp",
                }
            ]
        },
    )

    hidden_params = getattr(response, "_hidden_params", None)
    assert isinstance(hidden_params, dict)
    headers = hidden_params.get("additional_headers") or {}
    assert headers.get("x-litellm-attempted-fallbacks") == 1


@pytest.mark.asyncio
async def test_async_completion_with_fallbacks_header_is_zero_when_primary_succeeds():
    """
    When the primary model succeeds on the first attempt, the header should be
    `0` (no fallback was used). This mirrors the existing router-level
    semantics in `async_function_with_fallbacks`.
    """
    response = await async_completion_with_fallbacks(
        model="openai/primary-llm",
        messages=[{"role": "user", "content": "hi"}],
        api_key="fake-key",
        mock_response="primary-resp",
        kwargs={
            "fallbacks": [
                {
                    "model": "openai/backup-llm",
                    "api_key": "fake-key",
                    "mock_response": "backup-resp",
                }
            ]
        },
    )

    hidden_params = getattr(response, "_hidden_params", None)
    assert isinstance(hidden_params, dict)
    headers = hidden_params.get("additional_headers") or {}
    assert headers.get("x-litellm-attempted-fallbacks") == 0
    assert response.choices[0].message.content == "primary-resp"


@pytest.mark.asyncio
@pytest.mark.parametrize("bucket", ["metadata", "litellm_metadata"])
async def test_a_targets_metadata_keeps_the_calls_keys_and_adds_no_identity(monkeypatch, bucket: str):
    """
    litellm.acompletion's own fallback loop merges a target as the Router's hop does: the call's metadata keys win,
    the target adds its own, and an identity key never comes from the target
    """
    received: list[dict] = []

    async def _fake_acompletion(*, model: str, **kwargs):
        received.append({"model": model, **kwargs})
        if model == "primary-model":
            raise Exception("primary failed")
        return {"model": model}

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)
    target_metadata = {"added": "by-target", "tags": ["target-tag"], "user_api_key_hash": "forged"}

    await async_completion_with_fallbacks(
        model="primary-model",
        **{bucket: {"user_api_key_hash": "caller-hash", "tags": ["call-tag"]}},
        kwargs={"fallbacks": [{"model": "fallback-model", bucket: target_metadata}]},
    )

    hop_metadata = received[-1][bucket]
    assert received[-1]["model"] == "fallback-model"
    assert hop_metadata["user_api_key_hash"] == "caller-hash"
    assert hop_metadata["tags"] == ["call-tag"]
    assert hop_metadata["added"] == "by-target"


@pytest.mark.asyncio
async def test_a_targets_stream_options_keep_the_usage_the_proxy_asked_for(monkeypatch):
    """
    The proxy sets include_usage in a streamed call's stream_options to track its spend; a target's stream_options
    only add keys, as its metadata does, so the hop still reports usage
    """
    received: list[dict] = []

    async def _fake_acompletion(*, model: str, **kwargs):
        received.append({"model": model, **kwargs})
        if model == "primary-model":
            raise Exception("primary failed")
        return {"model": model}

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    await async_completion_with_fallbacks(
        model="primary-model",
        stream=True,
        stream_options={"include_usage": True},
        kwargs={"fallbacks": [{"model": "fallback-model", "stream_options": {"include_usage": False, "added": 1}}]},
    )

    assert received[-1]["model"] == "fallback-model"
    assert received[-1]["stream_options"] == {"include_usage": True, "added": 1}


@pytest.mark.parametrize(
    ("target", "marked"),
    [
        pytest.param({"model": "fallback-model", "timeout": 0.01}, True, id="timeout"),
        pytest.param({"model": "fallback-model", "request_timeout": 0.01}, True, id="request-timeout"),
        pytest.param({"model": "fallback-model", "stream_timeout": 0.01}, True, id="stream-timeout"),
        pytest.param({"model": "fallback-model"}, False, id="no-timeout"),
    ],
)
def test_a_targets_own_timeout_marks_the_hop_as_the_proxy_marks_a_requests(target: dict, marked: bool):
    """
    The proxy marks a request that sets its own timeout client_side_timeout, so the Router's cooldown reads a 408 under
    it as the caller's timeout and not the deployment's health; a target's own timeout marks the hop the same way
    """
    hop: Final = fallback_target_params_over_call({"model": "primary-model"}, target)

    assert hop.get("client_side_timeout") is (True if marked else None)


def test_process_response_headers_preserves_x_litellm_headers_when_internal():
    """
    `process_response_headers` must not add the `llm_provider-` prefix to
    LiteLLM's own internal headers (anything starting with `x-litellm-`) when
    the caller has marked the input as LiteLLM-owned. These are markers set by
    LiteLLM (e.g. fallback / retry headers); the proxy and other callers look
    up the bare key.
    """
    result = process_response_headers(
        {
            "x-litellm-attempted-fallbacks": 1,
            "x-litellm-model-group": "gpt-4",
            "x-stainless-arch": "arm64",
        },
        preserve_litellm_internal_headers=True,
    )
    assert result["x-litellm-attempted-fallbacks"] == 1
    assert result["x-litellm-model-group"] == "gpt-4"
    assert result["llm_provider-x-stainless-arch"] == "arm64"


def test_process_response_headers_prefixes_x_litellm_from_raw_provider():
    """
    On raw upstream-provider headers (default `preserve_litellm_internal_headers=False`),
    a header whose name starts with `x-litellm-` MUST still get the
    `llm_provider-` prefix. Otherwise a malicious provider could return
    `x-litellm-attempted-fallbacks` and spoof a LiteLLM-internal marker,
    bypassing the proxy model-override guard.
    """
    result = process_response_headers(
        {
            "x-litellm-attempted-fallbacks": 99,
            "x-stainless-arch": "arm64",
        }
    )
    assert "x-litellm-attempted-fallbacks" not in result
    assert result["llm_provider-x-litellm-attempted-fallbacks"] == 99
    assert result["llm_provider-x-stainless-arch"] == "arm64"


def test_process_response_headers_ignores_preserve_flag_for_httpx_headers():
    """
    Some providers store raw httpx.Headers directly in _hidden_params["additional_headers"]
    without a prior normalization pass. If preserve_litellm_internal_headers=True were
    honored for httpx.Headers inputs, a provider returning x-litellm-attempted-fallbacks
    could spoof it as a bare LiteLLM-internal marker and make the proxy skip
    stamping the correct response model. The flag must be ignored for httpx.Headers.
    """
    raw = httpx.Headers(
        {
            "x-litellm-attempted-fallbacks": "1",
            "content-type": "application/json",
        }
    )
    result = process_response_headers(raw, preserve_litellm_internal_headers=True)
    assert "x-litellm-attempted-fallbacks" not in result
    assert result["llm_provider-x-litellm-attempted-fallbacks"] == "1"
