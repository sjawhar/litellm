import copy
import json
from typing import Final

import httpx
import pytest

from litellm.llms.anthropic.pass_through.messages.transformation import AnthropicMessagesConfig

_BOUND_BLOCK: Final = (
    "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation."
)
_BAD_SIGNATURE: Final = "messages.1.content.0: Invalid `signature` in `thinking` block"
_NULL_SIGNATURE: Final = "messages.1.content.0.thinking.signature.str: Input should be a valid string"
_EMPTY_THINKING: Final = "messages.1.content.0.thinking: each thinking block must contain thinking"


def _invalid_thinking_400(message: str) -> httpx.HTTPStatusError:
    request: Final = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response: Final = httpx.Response(
        400,
        request=request,
        text=json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": message}}),
    )
    return httpx.HTTPStatusError("bad request", request=request, response=response)


def _request(prefix_mismatch_behavior: str) -> dict[str, object]:
    return {
        "model": "claude-opus-5-5",
        "max_tokens": 16,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "signature": "sig"}]},
            {"role": "user", "content": "again"},
        ],
        "thinking": {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": prefix_mismatch_behavior}},
    }


_INVALID_THINKING_ERRORS: Final = pytest.mark.parametrize(
    "error_message",
    [_BOUND_BLOCK, _BAD_SIGNATURE, _NULL_SIGNATURE, _EMPTY_THINKING],
    ids=["bound-to-other-conversation", "invalid-signature", "null-signature", "empty-thinking"],
)


@_INVALID_THINKING_ERRORS
def test_a_request_asking_for_the_binding_error_is_never_stripped_or_retried(error_message: str):
    config: Final = AnthropicMessagesConfig()
    error: Final = _invalid_thinking_400(error_message)
    request: Final = _request("error")
    sent: Final = copy.deepcopy(request)

    retried: Final = config.should_retry_anthropic_messages_on_http_error(error, {}, request_body=request)
    config.transform_anthropic_messages_request_on_http_error(error, request)

    assert retried is False
    assert request == sent


@_INVALID_THINKING_ERRORS
def test_a_request_dropping_unbound_blocks_still_gets_the_strip_retry(error_message: str):
    config: Final = AnthropicMessagesConfig()
    error: Final = _invalid_thinking_400(error_message)
    request: Final = _request("drop_block")

    retried: Final = config.should_retry_anthropic_messages_on_http_error(error, {}, request_body=request)
    config.transform_anthropic_messages_request_on_http_error(error, request)

    assert retried is True
    assert "thinking" not in request
    assert [message["role"] for message in request["messages"]] == ["user", "user"]
