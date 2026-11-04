import datetime
import logging
import sys
from itertools import chain
from typing import Final

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth


def _clear_prometheus_registry() -> None:
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)


def _create_prometheus_logger_with_custom_labels(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        litellm,
        "custom_prometheus_metadata_labels",
        ["metadata.department", "metadata.environment"],
    )
    _clear_prometheus_registry()
    return PrometheusLogger()


def _standard_logging_payload_with_requester_metadata() -> dict:
    return {
        "model_id": "model-123",
        "model_group": "gpt-4o-mini",
        "api_base": "https://api.openai.com",
        "custom_llm_provider": "openai",
        "metadata": {
            "user_api_key_hash": "test-hash",
            "user_api_key_alias": "test-alias",
            "user_api_key_team_id": "test-team",
            "user_api_key_team_alias": "test-team-alias",
            "user_api_key_user_id": "test-user",
            "user_api_key_user_email": "test@example.com",
            "user_api_key_org_id": None,
            "requester_metadata": {
                "department": "engineering",
                "environment": "production",
            },
            "user_api_key_auth_metadata": None,
            "spend_logs_metadata": None,
        },
        "request_tags": [],
        "completion_tokens": 0,
        "total_tokens": 0,
        "response_cost": 0,
    }


def _metric_samples(metric_name: str):
    return [
        sample
        for metric in REGISTRY.collect()
        for sample in metric.samples
        if sample.name == metric_name
    ]


@pytest.mark.asyncio
async def test_async_log_failure_event_accepts_custom_metadata_labels(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    kwargs = {
        "model": "gpt-4o-mini",
        "litellm_params": {
            "metadata": {
                "user_api_key_end_user_id": "test-end-user",
            }
        },
        "standard_logging_object": _standard_logging_payload_with_requester_metadata(),
    }

    with caplog.at_level(logging.ERROR):
        await prometheus_logger.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

    assert "Incorrect label count" not in caplog.text
    samples = _metric_samples("litellm_llm_api_failed_requests_metric_total")
    assert any(
        sample.labels.get("metadata_department") == "engineering"
        and sample.labels.get("metadata_environment") == "production"
        for sample in samples
    )


_PRIORITY_CASES = (
    ("high", None, None, "high"),
    (None, "high", None, ""),
    (None, None, "high", ""),
    ("high", "forged-requester", "forged-key", "high"),
)
_PRIORITY_ROUTES = (
    pytest.param("metadata", id="chat_completions"),
    pytest.param("litellm_metadata", id="messages"),
)


def _priority_litellm_params(
    trusted_metadata_key: str,
    team_priority: str | None,
    requester_priority: str | None,
    key_priority: str | None,
) -> dict:
    caller_metadata = {"priority": requester_priority} if requester_priority is not None else {}
    trusted_metadata = {
        "user_api_key_metadata": {"priority": key_priority} if key_priority is not None else {},
        "user_api_key_team_metadata": {"priority": team_priority} if team_priority is not None else {},
    }
    if trusted_metadata_key == "metadata":
        return {"metadata": {**caller_metadata, **trusted_metadata}}
    return {"metadata": caller_metadata, "litellm_metadata": trusted_metadata}


def _priority_standard_logging_payload(requester_priority: str | None, key_priority: str | None) -> dict:
    payload = _standard_logging_payload_with_requester_metadata()
    payload["metadata"]["requester_metadata"] = (
        {"priority": requester_priority} if requester_priority is not None else {}
    )
    payload["metadata"]["user_api_key_auth_metadata"] = {"priority": key_priority} if key_priority is not None else {}
    return payload


@pytest.mark.asyncio
@pytest.mark.parametrize("trusted_metadata_key", _PRIORITY_ROUTES)
@pytest.mark.parametrize(("team_priority", "requester_priority", "key_priority", "expected_priority"), _PRIORITY_CASES)
async def test_priority_label_uses_team_metadata_for_failure_counters_and_saturation_gauge(
    monkeypatch: pytest.MonkeyPatch,
    trusted_metadata_key: str,
    team_priority: str | None,
    requester_priority: str | None,
    key_priority: str | None,
    expected_priority: str,
):
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", ["priority", "metadata.department"])
    _clear_prometheus_registry()
    prometheus_logger = PrometheusLogger()
    litellm_params = _priority_litellm_params(trusted_metadata_key, team_priority, requester_priority, key_priority)
    kwargs = {
        "model": "gpt-4o-mini",
        "litellm_params": litellm_params,
        "standard_logging_object": _priority_standard_logging_payload(requester_priority, key_priority),
        "exception": RuntimeError("provider failed"),
    }

    await prometheus_logger.async_log_failure_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
    )
    await prometheus_logger.async_post_call_failure_hook(
        request_data={"model": "gpt-4o-mini", **litellm_params},
        original_exception=RuntimeError("proxy failed"),
        user_api_key_dict=UserAPIKeyAuth(
            api_key="test-key",
            metadata={"priority": key_priority} if key_priority is not None else {},
            team_metadata={"priority": team_priority} if team_priority is not None else {},
        ),
    )
    prometheus_logger.set_model_group_saturation_ratio(
        model_group="gpt-4o-mini",
        saturation=0.75,
        litellm_params=litellm_params,
    )

    for metric_name in (
        "litellm_llm_api_failed_requests_metric_total",
        "litellm_deployment_failure_responses_total",
        "litellm_proxy_failed_requests_metric_total",
    ):
        samples = _metric_samples(metric_name)
        assert len(samples) == 1, metric_name
        assert samples[0].labels["priority"] == expected_priority, metric_name

    saturation_samples = _metric_samples("litellm_model_group_saturation_ratio")
    assert len(saturation_samples) == 1
    assert saturation_samples[0].labels == {"model_group": "gpt-4o-mini", "priority": expected_priority}
    assert saturation_samples[0].value == 0.75


@pytest.mark.asyncio
@pytest.mark.parametrize("trusted_metadata_key", _PRIORITY_ROUTES)
@pytest.mark.parametrize(
    ("team_priority", "expected_priority"),
    (("high", "high"), (None, "")),
)
async def test_priority_label_uses_team_metadata_for_success_and_virtual_key_metrics(
    monkeypatch: pytest.MonkeyPatch,
    trusted_metadata_key: str,
    team_priority: str | None,
    expected_priority: str,
):
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", ["priority"])
    _clear_prometheus_registry()
    prometheus_logger = PrometheusLogger()
    litellm_params = _priority_litellm_params(trusted_metadata_key, team_priority, "forged-requester", "forged-key")
    standard_logging_payload = {
        **_priority_standard_logging_payload("forged-requester", "forged-key"),
        "id": "priority-success",
        "call_type": "completion",
        "status": "success",
        "total_tokens": 30,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "response_cost": 0.001,
        "startTime": 1.0,
        "endTime": 2.0,
        "completionStartTime": 1.5,
        "model": "gpt-4o-mini",
        "end_user": None,
        "cache_hit": False,
        "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": None},
    }
    kwargs = {
        "model": "gpt-4o-mini",
        "litellm_params": litellm_params,
        "standard_logging_object": standard_logging_payload,
    }

    now = datetime.datetime.now()
    await prometheus_logger.async_log_success_event(kwargs, None, now, now)
    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata={"model_group": "gpt-4o-mini", "litellm-key-remaining-requests-gpt-4o-mini": 3},
        model_id="model-123",
    )

    for metric_name in (
        "litellm_total_tokens_metric_total",
        "litellm_remaining_api_key_requests_for_model",
    ):
        samples = _metric_samples(metric_name)
        assert samples, metric_name
        assert {sample.labels["priority"] for sample in samples} == {expected_priority}, metric_name


@pytest.mark.parametrize("team_priority", ("high", None))
def test_priority_label_on_gauges_without_a_request_is_empty(monkeypatch: pytest.MonkeyPatch, team_priority: str | None):
    from litellm.proxy._types import LiteLLM_TeamTable

    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", ["priority"])
    _clear_prometheus_registry()
    prometheus_logger = PrometheusLogger()
    team_metadata = {"priority": team_priority} if team_priority is not None else {}

    prometheus_logger._set_team_budget_metrics(
        LiteLLM_TeamTable(team_id="team-1", metadata=team_metadata, max_budget=10.0, spend=1.0)
    )
    prometheus_logger._set_key_budget_metrics(
        UserAPIKeyAuth(token="hashed-key", key_alias="key-1", team_metadata=team_metadata, max_budget=5.0, spend=1.0)
    )
    prometheus_logger._set_deployment_tpm_rpm_limit_metrics(
        model_info={"tpm": 100, "rpm": 10},
        litellm_params={},
        litellm_model_name="gpt-4o-mini",
        model_id="model-123",
        api_base="https://api.openai.com",
        llm_provider="openai",
        model_group="gpt-4o-mini",
    )

    for metric_name in (
        "litellm_remaining_team_budget_metric",
        "litellm_remaining_api_key_budget_metric",
        "litellm_deployment_tpm_limit",
        "litellm_deployment_rpm_limit",
    ):
        samples = _metric_samples(metric_name)
        assert [sample.labels["priority"] for sample in samples] == [""], metric_name


def _body_forging_team_priority() -> dict:
    forged = {"user_api_key_team_metadata": {"priority": "forged-by-caller"}}
    return {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": dict(forged),
        "litellm_metadata": dict(forged),
    }


def _priority_labels_by_metric() -> dict[str, set[str]]:
    samples = chain.from_iterable(metric.samples for metric in REGISTRY.collect())
    labelled = [sample for sample in samples if "priority" in sample.labels and not sample.name.endswith("_created")]
    return {
        name: {sample.labels["priority"] for sample in labelled if sample.name == name}
        for name in {sample.name for sample in labelled}
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("team_priority", "expected_priority"), (("high", "high"), (None, "")))
async def test_client_failure_priority_label_ignores_team_metadata_forged_in_the_body(
    monkeypatch: pytest.MonkeyPatch, team_priority: str | None, expected_priority: str
):
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", ["priority"])
    _clear_prometheus_registry()
    prometheus_logger = PrometheusLogger()

    await prometheus_logger.async_post_call_failure_hook(
        request_data=_body_forging_team_priority(),
        original_exception=litellm.AuthenticationError(message="Invalid key", llm_provider="", model="gpt-4o-mini"),
        user_api_key_dict=UserAPIKeyAuth(team_metadata={"priority": team_priority} if team_priority else None),
    )

    assert _priority_labels_by_metric() == {
        "litellm_proxy_failed_requests_metric_total": {expected_priority},
        "litellm_proxy_total_requests_metric_total": {expected_priority},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("team_priority", "expected_priority"), (("high", "high"), (None, "")))
async def test_auth_gate_rejection_labels_every_failure_series_with_the_key_team_class(
    monkeypatch: pytest.MonkeyPatch, team_priority: str | None, expected_priority: str
):
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import ProxyErrorTypes
    from litellm.proxy.utils import ProxyLogging

    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", ["priority"])
    _clear_prometheus_registry()
    monkeypatch.setattr(litellm, "callbacks", [PrometheusLogger()])

    await ProxyLogging(user_api_key_cache=DualCache()).post_call_failure_hook(
        request_data=_body_forging_team_priority(),
        original_exception=litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0),
        user_api_key_dict=UserAPIKeyAuth(
            token="hashed-key",
            team_id="team-1",
            team_metadata={"priority": team_priority} if team_priority else None,
            request_route="/chat/completions",
        ),
        error_type=ProxyErrorTypes.auth_error,
        route="/chat/completions",
    )

    labels_by_metric = _priority_labels_by_metric()
    assert {
        "litellm_proxy_failed_requests_metric_total",
        "litellm_llm_api_failed_requests_metric_total",
    } <= labels_by_metric.keys()
    assert {name: {expected_priority} for name in labels_by_metric} == labels_by_metric


def _client_labels_by_metric() -> dict[str, set[tuple[str, str]]]:
    samples: Final = chain.from_iterable(metric.samples for metric in REGISTRY.collect())
    labelled: Final = [
        sample for sample in samples if "client_ip" in sample.labels and not sample.name.endswith("_created")
    ]
    return {
        name: {(sample.labels["client_ip"], sample.labels["user_agent"]) for sample in labelled if sample.name == name}
        for name in {sample.name for sample in labelled}
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_bucket", ("metadata", "litellm_metadata"))
async def test_client_failure_ip_and_agent_labels_come_from_the_bucket_the_proxy_stamped(proxy_bucket: str):
    """Routes such as /v1/messages keep the proxy's own metadata under litellm_metadata and leave the caller's
    provider metadata at the top level, so reading the two client labels from metadata would take the caller's
    strings on a post-auth failure and mint one series each."""

    _clear_prometheus_registry()
    prometheus_logger: Final = PrometheusLogger()
    stamped: Final = {
        "user_api_key_auth": UserAPIKeyAuth(token="hashed-key"),
        "requester_ip_address": "10.1.2.3",
        "user_agent": "real-client/1.0",
    }
    forged: Final = {"requester_ip_address": "9.9.9.9", "user_agent": "forged-agent"}
    request_data: Final = {
        "model": "gpt-4o-mini",
        proxy_bucket: stamped,
        **({"metadata": forged} if proxy_bucket == "litellm_metadata" else {}),
    }

    await prometheus_logger.async_post_call_failure_hook(
        request_data=request_data,
        original_exception=litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0),
        user_api_key_dict=UserAPIKeyAuth(token="hashed-key"),
    )

    expected: Final = {("10.1.2.3", "real-client/1.0")}
    assert _client_labels_by_metric() == {
        name: expected
        for name in ("litellm_proxy_failed_requests_metric_total", "litellm_proxy_total_requests_metric_total")
    }


def test_virtual_key_rate_limit_metrics_accept_custom_metadata_labels(
    monkeypatch: pytest.MonkeyPatch,
):
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    metadata = {
        "model_group": "gpt-4o-mini",
        "litellm-key-remaining-requests-gpt-4o-mini": 3,
        "litellm-key-remaining-tokens-gpt-4o-mini": 200,
    }
    kwargs = {
        "litellm_params": {
            "metadata": metadata,
        },
        "standard_logging_object": _standard_logging_payload_with_requester_metadata(),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    samples = _metric_samples("litellm_remaining_api_key_requests_for_model")
    assert any(
        sample.labels.get("metadata_department") == "engineering"
        and sample.labels.get("metadata_environment") == "production"
        and sample.value == 3
        for sample in samples
    )


def test_virtual_key_rate_limit_metrics_preserve_zero_remaining_values(
    monkeypatch: pytest.MonkeyPatch,
):
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    metadata = {
        "model_group": "gpt-4o-mini",
        "litellm-key-remaining-requests-gpt-4o-mini": 0,
        "litellm-key-remaining-tokens-gpt-4o-mini": 0,
    }
    kwargs = {
        "litellm_params": {
            "metadata": metadata,
        },
        "standard_logging_object": _standard_logging_payload_with_requester_metadata(),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    request_samples = _metric_samples("litellm_remaining_api_key_requests_for_model")
    token_samples = _metric_samples("litellm_remaining_api_key_tokens_for_model")

    assert any(sample.value == 0 for sample in request_samples)
    assert any(sample.value == 0 for sample in token_samples)
    assert not any(sample.value == sys.maxsize for sample in request_samples)
    assert not any(sample.value == sys.maxsize for sample in token_samples)
