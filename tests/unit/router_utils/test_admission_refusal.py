from collections.abc import Callable, Mapping

import pytest

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.admission_refusal import mark_admission_refusal
from litellm.router_utils.cooldown_handlers import get_cooldown_deployments

REFUSED_DEPLOYMENT = "org-a"
OTHER_DEPLOYMENT = "org-b"


class _RefusesOneDeployment(CustomLogger):
    def __init__(self, refusal: Callable[[], Exception]) -> None:
        super().__init__()
        self.refusal = refusal
        self.attempts: tuple[str, ...] = ()
        self.raised: tuple[Exception, ...] = ()

    async def async_pre_call_check(self, deployment: Mapping[str, object], parent_otel_span: object) -> None:
        match deployment:
            case {"model_info": {"id": str() as deployment_id}}:
                self.attempts = (*self.attempts, deployment_id)
            case _:
                raise AssertionError(f"router passed a deployment without an id: {deployment}")
        if deployment_id != REFUSED_DEPLOYMENT:
            return None
        refusal = self.refusal()
        self.raised = (*self.raised, refusal)
        raise refusal


class _AdmitsOnlyTheFirstAttempt(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.attempts: tuple[str, ...] = ()
        self.raised: tuple[Exception, ...] = ()

    async def async_pre_call_check(self, deployment: Mapping[str, object], parent_otel_span: object) -> None:
        match deployment:
            case {"model_info": {"id": str() as deployment_id}}:
                self.attempts = (*self.attempts, deployment_id)
            case _:
                raise AssertionError(f"router passed a deployment without an id: {deployment}")
        if len(self.attempts) == 1:
            return None
        refusal = _marked(litellm.RateLimitError(message="bucket full", llm_provider="openai", model="gpt-5.6"))
        self.raised = (*self.raised, refusal)
        raise refusal


def _mock_deployment(model_name: str, deployment_id: str, mock_response: str, weight: int = 1) -> dict[str, object]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": "openai/gpt-5.6",
            "api_key": "sk-fake",
            "mock_response": mock_response,
            "weight": weight,
        },
        "model_info": {"id": deployment_id},
    }


def _router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "claude",
                "litellm_params": {"model": "openai/gpt-5.6", "api_key": "sk-fake", "weight": 1, "mock_response": "a"},
                "model_info": {"id": REFUSED_DEPLOYMENT},
            },
            {
                "model_name": "claude",
                "litellm_params": {"model": "openai/gpt-5.6", "api_key": "sk-fake", "weight": 0, "mock_response": "b"},
                "model_info": {"id": OTHER_DEPLOYMENT},
            },
            {
                "model_name": "claude-fallback",
                "litellm_params": {"model": "openai/gpt-5.6", "api_key": "sk-fake", "mock_response": "fallback"},
                "model_info": {"id": "fallback"},
            },
        ],
        num_retries=2,
        retry_after=0,
        fallbacks=[{"claude": ["claude-fallback"]}],
    )


def _marked(exception: Exception) -> Exception:
    mark_admission_refusal(exception)
    return exception


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refusal",
    (
        pytest.param(
            lambda: _marked(litellm.RateLimitError(message="bucket full", llm_provider="openai", model="gpt-5.6")),
            id="rate_limit",
        ),
        pytest.param(
            lambda: _marked(
                litellm.ServiceUnavailableError(message="limiter unverifiable", llm_provider="openai", model="gpt-5.6")
            ),
            id="service_unavailable",
        ),
    ),
)
async def test_marked_admission_refusal_is_final_with_no_retry_fallback_or_cooldown(
    monkeypatch: pytest.MonkeyPatch, refusal: Callable[[], Exception]
):
    admission = _RefusesOneDeployment(refusal)
    monkeypatch.setattr(litellm, "callbacks", [admission])
    router = _router()

    with pytest.raises(type(refusal())) as refused:
        await router.acompletion(model="claude", messages=[{"role": "user", "content": "hi"}])

    assert admission.raised == (refused.value,)
    assert admission.attempts == (REFUSED_DEPLOYMENT,)
    assert get_cooldown_deployments(router, parent_otel_span=None) == []


@pytest.mark.asyncio
async def test_unmarked_pre_call_rate_limit_still_cools_down_and_moves_to_another_deployment(
    monkeypatch: pytest.MonkeyPatch,
):
    admission = _RefusesOneDeployment(
        lambda: litellm.RateLimitError(message="deployment rpm", llm_provider="openai", model="gpt-5.6")
    )
    monkeypatch.setattr(litellm, "callbacks", [admission])
    router = _router()

    response = await router.acompletion(model="claude", messages=[{"role": "user", "content": "hi"}])

    assert response.choices[0].message.content == "b"
    assert admission.attempts == (REFUSED_DEPLOYMENT, OTHER_DEPLOYMENT)
    assert get_cooldown_deployments(router, parent_otel_span=None) == [REFUSED_DEPLOYMENT]


@pytest.mark.asyncio
async def test_refusal_on_a_retry_after_an_upstream_failure_ends_the_request_with_no_further_attempt(
    monkeypatch: pytest.MonkeyPatch,
):
    admission = _AdmitsOnlyTheFirstAttempt()
    monkeypatch.setattr(litellm, "callbacks", [admission])
    router = Router(
        model_list=[
            _mock_deployment("claude", "upstream-fails", "litellm.InternalServerError"),
            _mock_deployment("claude-fallback", "fallback", "fallback"),
        ],
        num_retries=2,
        retry_after=0,
        fallbacks=[{"claude": ["claude-fallback"]}],
    )

    with pytest.raises(litellm.RateLimitError) as refused:
        await router.acompletion(model="claude", messages=[{"role": "user", "content": "hi"}])

    assert admission.raised == (refused.value,)
    assert admission.attempts == ("upstream-fails", "upstream-fails")


@pytest.mark.asyncio
async def test_refusal_on_a_fallback_hop_ends_the_request_with_no_further_fallback_or_cooldown(
    monkeypatch: pytest.MonkeyPatch,
):
    admission = _RefusesOneDeployment(
        lambda: _marked(litellm.RateLimitError(message="bucket full", llm_provider="openai", model="gpt-5.6"))
    )
    monkeypatch.setattr(litellm, "callbacks", [admission])
    router = Router(
        model_list=[
            _mock_deployment("claude", "upstream-fails", "litellm.InternalServerError"),
            _mock_deployment("claude-fallback", REFUSED_DEPLOYMENT, "fallback"),
            _mock_deployment("claude-last", "last-resort", "last"),
        ],
        num_retries=0,
        fallbacks=[{"claude": ["claude-fallback", "claude-last"]}],
    )

    with pytest.raises(litellm.RateLimitError) as refused:
        await router.acompletion(model="claude", messages=[{"role": "user", "content": "hi"}])

    assert admission.raised == (refused.value,)
    assert admission.attempts == ("upstream-fails", REFUSED_DEPLOYMENT)
    assert REFUSED_DEPLOYMENT not in get_cooldown_deployments(router, parent_otel_span=None)


@pytest.mark.asyncio
async def test_refusal_on_a_weighted_failover_pick_ends_the_request_with_no_cross_group_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    admission = _RefusesOneDeployment(
        lambda: _marked(litellm.RateLimitError(message="bucket full", llm_provider="openai", model="gpt-5.6"))
    )
    monkeypatch.setattr(litellm, "callbacks", [admission])
    router = Router(
        model_list=[
            _mock_deployment("claude", "upstream-fails", "litellm.InternalServerError"),
            _mock_deployment("claude", REFUSED_DEPLOYMENT, "never served", weight=0),
            _mock_deployment("claude-fallback", "fallback", "fallback"),
        ],
        num_retries=0,
        enable_weighted_failover=True,
        fallbacks=[{"claude": ["claude-fallback"]}],
    )

    with pytest.raises(litellm.RateLimitError) as refused:
        await router.acompletion(model="claude", messages=[{"role": "user", "content": "hi"}])

    assert admission.raised == (refused.value,)
    assert admission.attempts == ("upstream-fails", REFUSED_DEPLOYMENT)
