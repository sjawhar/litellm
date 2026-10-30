"""
Dynamic rate limiter v3 - Saturation-aware priority-based rate limiting per provider bucket
"""

import os
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException
from pydantic import ConfigDict, TypeAdapter, ValidationError, with_config
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm import ModelResponse
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import (
    ProxyRateLimitError,
    map_v3_rate_limit_type,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (  # noqa: F401  # legacy module exports
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitDescriptorRateLimitObject,
    RateLimitResponse,
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    claim_request_stash_for_data,
    get_or_create_request_stash,
)
from litellm.proxy.hooks.rate_limiter_utils import (
    convert_priority_to_percent,
    resolve_llm_provider_for_rate_limit,
)
from litellm.proxy.utils import InternalUsageCache
from litellm.router_utils.add_retry_fallback_headers import (
    ensure_response_additional_headers,
    response_has_hidden_params,
)
from litellm.router_utils.admission_refusal import mark_admission_refusal
from litellm.types.integrations.prometheus import priority_of_team_metadata, team_priority_label
from litellm.types.router import ModelGroupInfo
from litellm.types.utils import CallTypesLiteral, LLMResponseTypes

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.types.utils import PriorityReservationSettings


def _get_priority_settings() -> "PriorityReservationSettings":
    """
    Get the priority reservation settings, guaranteed to be non-None.

    The settings are lazy-loaded in litellm.__init__ and always return an instance.
    This helper provides proper type narrowing for mypy.
    """
    settings: Final = litellm.priority_reservation_settings
    if settings is None:
        # This should never happen due to lazy loading, but satisfy mypy
        from litellm.types.utils import PriorityReservationSettings

        return PriorityReservationSettings()
    return settings


def _is_latin1_encodable(value: object) -> bool:
    return all(ord(char) < 256 for char in str(value))


@with_config(ConfigDict(extra="ignore"))
class _DeploymentModelInfo(TypedDict, total=False):
    id: ReadOnly[str]
    bucket_class: ReadOnly[object]
    rpm: ReadOnly[float | None]
    tpm: ReadOnly[float | None]


@with_config(ConfigDict(extra="ignore"))
class _DeploymentParams(TypedDict, total=False):
    model: ReadOnly[str]
    anthropic_organization_id: ReadOnly[str | None]
    rpm: ReadOnly[float | None]
    tpm: ReadOnly[float | None]


@with_config(ConfigDict(extra="ignore"))
class _RoutedDeployment(TypedDict):
    model_name: ReadOnly[str]
    litellm_params: ReadOnly[_DeploymentParams]
    model_info: ReadOnly[_DeploymentModelInfo]


@with_config(ConfigDict(extra="ignore"))
class _TrustedMetadata(TypedDict, total=False):
    deployment: ReadOnly[str]


@with_config(ConfigDict(extra="ignore"))
class _LoggedCall(TypedDict, total=False):
    model_info: ReadOnly[_DeploymentModelInfo | None]
    anthropic_organization_id: ReadOnly[str | None]
    litellm_metadata: ReadOnly[_TrustedMetadata | None]
    metadata: ReadOnly[_TrustedMetadata | None]


_ROUTED_DEPLOYMENT: Final = TypeAdapter(_RoutedDeployment)
_LOGGED_CALL: Final = TypeAdapter(_LoggedCall)
_BUCKET_CLASSES: Final = TypeAdapter(Mapping[str, str])


@dataclass(frozen=True, slots=True)
class _Bucket:
    """One organization's capacity for one model class. Its limits are the serving deployment's."""

    key: str
    capacity: ModelGroupInfo


@dataclass(frozen=True, slots=True)
class _AdmissionRequest:
    model_group: str
    team_class: str | None
    request_data: Mapping[str, object]


_admission_request: Final[ContextVar[_AdmissionRequest | None]] = ContextVar(
    "litellm_dynamic_rate_limiter_v3_admission_request", default=None
)


def _bucket_class(served_model: str, model_info: _DeploymentModelInfo) -> str:
    try:
        bucket_classes: Final = _BUCKET_CLASSES.validate_python(model_info.get("bucket_class", MappingProxyType({})))
    except ValidationError:
        return served_model
    return bucket_classes.get(served_model) or served_model


def _bucket_key(deployment_model: str, organization_id: str | None, model_info: _DeploymentModelInfo) -> str | None:
    organization: Final = organization_id or model_info.get("id")
    if not organization:
        return None
    served_model, _ = resolve_llm_provider_for_rate_limit(deployment_model)
    return f"{organization}:{_bucket_class(served_model, model_info)}"


def _deployment_limit(*candidates: float | None) -> int | None:
    return next((int(limit) for limit in candidates if limit is not None and limit > 0), None)


def _admission_bucket(deployment: _RoutedDeployment) -> _Bucket | None:
    litellm_params: Final = deployment["litellm_params"]
    model_info: Final = deployment["model_info"]
    deployment_model: Final = litellm_params.get("model")
    if deployment_model is None:
        return None
    key: Final = _bucket_key(deployment_model, litellm_params.get("anthropic_organization_id"), model_info)
    rpm: Final = _deployment_limit(litellm_params.get("rpm"), model_info.get("rpm"))
    tpm: Final = _deployment_limit(litellm_params.get("tpm"), model_info.get("tpm"))
    if key is None or (rpm is None and tpm is None):
        return None
    return _Bucket(key=key, capacity=ModelGroupInfo(model_group=key, providers=[], rpm=rpm, tpm=tpm))


def _served_bucket_key(litellm_params: object) -> str | None:
    try:
        logged: Final = _LOGGED_CALL.validate_python(litellm_params)
    except ValidationError:
        return None
    model_info: Final = logged.get("model_info")
    deployment_model: Final = (logged.get("litellm_metadata") or {}).get("deployment") or (
        logged.get("metadata") or {}
    ).get("deployment")
    if model_info is None or deployment_model is None:
        return None
    return _bucket_key(deployment_model, logged.get("anthropic_organization_id"), model_info)


def _model_window_saturation(rate_limit_response: RateLimitResponse, pre_read_saturation: float) -> float:
    return max(
        (
            (status["current_limit"] - status["limit_remaining"]) / status["current_limit"]
            for status in rate_limit_response["statuses"]
            if status["descriptor_key"] == "model_saturation_check" and status["current_limit"] > 0
        ),
        default=pre_read_saturation,
    )


def _record_model_group_saturation(
    model: str,
    pre_read_saturation: float,
    admission: RateLimitResponse | None,
    data: Mapping[str, object],
) -> None:
    try:
        from litellm.integrations.prometheus import PrometheusLogger

        prometheus_logger: Final = PrometheusLogger.get_instance()
        if prometheus_logger is None:
            return
        prometheus_logger.set_model_group_saturation_ratio(
            model_group=model,
            saturation=(
                pre_read_saturation if admission is None else _model_window_saturation(admission, pre_read_saturation)
            ),
            litellm_params=data,
        )
    except Exception as e:  # noqa: BLE001  # a metrics failure must never change rate-limit enforcement
        verbose_proxy_logger.debug("Error recording model group saturation metric: %s", e)


def _reserved_priority(priority: str | None) -> str | None:
    if priority is None or litellm.priority_reservation is None or priority not in litellm.priority_reservation:
        return None
    return priority


def _priority_pool_key(bucket_key: str, priority: str | None) -> str:
    reserved_priority: Final = _reserved_priority(priority)
    return f"{bucket_key}:default_pool" if reserved_priority is None else f"{bucket_key}:{reserved_priority}"


class _ModelCapacityRateLimitError(ProxyRateLimitError):
    """The bucket's own limit refused the request, so its window is full."""


class _PROXY_DynamicRateLimitHandlerV3(CustomLogger):
    """
    Saturation-aware priority-based rate limiter using v3 infrastructure.

    Saturation and the priority pools are keyed by provider bucket: the organization and the model class of the
    deployment the router picked. The router admits each call in ``async_pre_call_check``, once it holds that
    deployment and before the provider call; ``async_pre_call_hook`` only records the calling team's class.

    Key features:
    1. Bucket capacity ALWAYS enforced at 100% (prevents over-allocation)
    2. Priority usage tracked from first request (accurate accounting)
    3. Priority limits only enforced when saturated >= threshold
    4. Atomic check-and-increment prevents partial counter increments
    5. Reuses v3 limiter's Redis-based tracking (multi-instance safe)
    """

    def __init__(
        self,
        internal_usage_cache: DualCache,
        time_provider: Callable[[], datetime] | None = None,
    ):
        self.internal_usage_cache = InternalUsageCache(dual_cache=internal_usage_cache)
        self.v3_limiter = PROXY_MaxParallelRequestsHandler_v3(self.internal_usage_cache, time_provider=time_provider)

    def _get_saturation_check_cache_ttl(self) -> int:
        """Get the configurable TTL for local cache when reading saturation values."""
        return _get_priority_settings().saturation_check_cache_ttl

    async def _get_saturation_value_from_cache(
        self,
        counter_key: str,
    ) -> str | None:
        """
        Get saturation value with configurable local cache TTL.

        Uses DualCache with configurable TTL for local cache storage.
        TTL is configurable via litellm.priority_reservation_settings.saturation_check_cache_ttl

        Args:
            counter_key: The cache key for the saturation counter

        Returns:
            Counter value as string, or None if not found
        """
        local_cache_ttl: Final = self._get_saturation_check_cache_ttl()

        return await self.internal_usage_cache.async_get_cache(
            key=counter_key,
            litellm_parent_otel_span=None,
            local_only=False,
            ttl=local_cache_ttl,
        )

    def _get_priority_weight(self, priority: str | None, model_info: ModelGroupInfo | None = None) -> float:
        """Get the weight for a given priority from litellm.priority_reservation"""
        weight: float = _get_priority_settings().default_priority
        if litellm.priority_reservation is None or priority not in litellm.priority_reservation:
            verbose_proxy_logger.debug("Priority Reservation not set for the given priority.")
        elif priority is not None and litellm.priority_reservation is not None:
            if os.getenv("LITELLM_LICENSE", None) is None:
                verbose_proxy_logger.error(
                    "PREMIUM FEATURE: Reserving tpm/rpm by priority is a premium feature. Please add a 'LITELLM_LICENSE' to your .env to enable this.\nGet a license: https://docs.litellm.ai/docs/proxy/enterprise."
                )
            else:
                value: Final = litellm.priority_reservation[priority]
                weight = convert_priority_to_percent(value, model_info)
        return weight

    def _get_priority_from_user_api_key_dict(self, user_api_key_dict: UserAPIKeyAuth) -> str | None:
        """The calling team's ``metadata.priority``. A key's own metadata never sets the class."""
        return priority_of_team_metadata(user_api_key_dict.team_metadata) or None

    def _normalize_priority_weights(self, model_info: ModelGroupInfo) -> dict[str, float]:
        """
        Normalize priority weights if they sum to > 1.0

        Handles over-allocation: {key_a: 0.60, key_b: 0.80} -> {key_a: 0.43, key_b: 0.57}
        Converts absolute rpm/tpm values to percentages based on bucket capacity.
        """
        if litellm.priority_reservation is None:
            return {}

        # Convert all values to percentages first
        weights: Final[dict[str, float]] = {}
        for k, v in litellm.priority_reservation.items():
            weights[k] = convert_priority_to_percent(v, model_info)

        total_weight: Final = sum(weights.values())

        if total_weight > 1.0:
            normalized: Final = {k: v / total_weight for k, v in weights.items()}
            verbose_proxy_logger.debug("Normalized over-allocated priorities: %s -> %s", weights, normalized)
            return normalized

        return weights

    def _get_priority_allocation(
        self,
        bucket_key: str,
        priority: str | None,
        normalized_weights: dict[str, float],
        model_info: ModelGroupInfo | None = None,
    ) -> tuple[float, str]:
        """
        Get priority weight and pool key for a given priority.

        For explicit priorities: returns specific allocation and unique pool key
        For default priority: returns default allocation and shared pool key
        """
        reserved_priority: Final = _reserved_priority(priority)
        if reserved_priority is None:
            return _get_priority_settings().default_priority, _priority_pool_key(bucket_key, priority)
        normalized_weight: Final = normalized_weights.get(reserved_priority)
        priority_weight: Final = (
            self._get_priority_weight(reserved_priority, model_info) if normalized_weight is None else normalized_weight
        )
        return priority_weight, _priority_pool_key(bucket_key, priority)

    async def _check_bucket_saturation(self, bucket: _Bucket) -> float:
        """
        Check current saturation by directly querying v3 limiter's cache keys.

        Reuses v3 limiter's Redis-based tracking (works across multiple instances).
        Reads counters WITHOUT incrementing them.

        Returns:
            float: Saturation ratio (0.0 = empty, 1.0 = at capacity, >1.0 = over)
        """
        try:
            saturation: Final = max(
                await self._window_saturation(bucket.key, "requests", bucket.capacity.rpm),
                await self._window_saturation(bucket.key, "tokens", bucket.capacity.tpm),
            )
        except Exception as e:
            verbose_proxy_logger.error("Error checking saturation for %s: %s", bucket.key, e)
            # Fail open: assume not saturated on error
            return 0.0
        verbose_proxy_logger.debug("Bucket %s overall saturation: %.1f%%", bucket.key, saturation * 100)
        return saturation

    async def _window_saturation(
        self, bucket_key: str, rate_limit_type: Literal["requests", "tokens"], limit: int | None
    ) -> float:
        if limit is None or limit <= 0:
            return 0.0
        counter_value: Final = await self._get_saturation_value_from_cache(
            counter_key=self.v3_limiter.create_rate_limit_keys(
                key="model_saturation_check", value=bucket_key, rate_limit_type=rate_limit_type
            )
        )
        return 0.0 if counter_value is None else float(counter_value) / limit

    def _create_priority_based_descriptors(
        self,
        bucket: _Bucket,
        priority: str | None,
    ) -> tuple[RateLimitDescriptor, ...]:
        """
        Create the priority pool descriptor of the bucket, with normalized priority weights.

        For explicit priorities: each priority gets its own pool (e.g., prod gets 75%)
        For default priority: ALL teams without a reserved priority share ONE pool (e.g., all share 25%)
        """
        if litellm.priority_reservation is None:
            return ()

        capacity: Final = bucket.capacity
        normalized_weights: Final = self._normalize_priority_weights(capacity)
        priority_weight, priority_key = self._get_priority_allocation(
            bucket_key=bucket.key,
            priority=priority,
            normalized_weights=normalized_weights,
            model_info=capacity,
        )

        rate_limit_config: Final[RateLimitDescriptorRateLimitObject] = {}

        if capacity.tpm is not None:
            rate_limit_config["tokens_per_unit"] = int(capacity.tpm * priority_weight)

        if capacity.rpm is not None:
            rate_limit_config["requests_per_unit"] = int(capacity.rpm * priority_weight)

        if not rate_limit_config:
            return ()
        rate_limit_config["window_size"] = self.v3_limiter.window_size
        return (RateLimitDescriptor(key="priority_model", value=priority_key, rate_limit=rate_limit_config),)

    def _create_model_tracking_descriptor(self, bucket: _Bucket) -> RateLimitDescriptor:
        """Create the descriptor that tracks and caps the bucket's whole window."""
        return RateLimitDescriptor(
            key="model_saturation_check",
            value=bucket.key,
            rate_limit={
                "requests_per_unit": bucket.capacity.rpm,
                "tokens_per_unit": bucket.capacity.tpm,
                "window_size": self.v3_limiter.window_size,
            },
        )

    async def _check_rate_limits(
        self,
        bucket: _Bucket,
        model_group: str,
        deployment_model: str,
        priority: str | None,
        saturation: float,
        parent_otel_span: "Span | None" = None,
    ) -> RateLimitResponse:
        """
        Atomically check and increment the bucket's counters.

        Key behaviors:
        - Only increment counters if request will be allowed
        - Bucket capacity: Always enforced at 100%
        - Priority limits: Only enforced when saturated >= threshold
        - Both counters tracked from first request (accurate accounting)

        Raises:
            HTTPException: If any limit is exceeded
        """
        import json

        saturation_threshold: Final = _get_priority_settings().saturation_threshold
        should_enforce_priority: Final = saturation >= saturation_threshold

        model_wide_descriptor: Final = self._create_model_tracking_descriptor(bucket)
        priority_descriptors: Final = self._create_priority_based_descriptors(bucket=bucket, priority=priority)

        # Atomic check-and-increment for the ENFORCED descriptor set:
        # - model_saturation_check is always enforced
        # - priority_model is enforced only when saturation crosses threshold
        #
        # Backed by a Redis Lua script (multi-process atomic) with an
        # asyncio.Lock + in-memory fallback for single-process deployments.
        # All-or-nothing: if any enforced descriptor would exceed its limit,
        # no counter is modified and the response carries "OVER_LIMIT".
        enforced_descriptors: Final = (
            model_wide_descriptor,
            *(priority_descriptors if should_enforce_priority else ()),
        )
        per_request_increment: Final[dict[Literal["requests", "tokens"], int]] = {"requests": 1, "tokens": 0}
        atomic_response: Final = await self.v3_limiter.atomic_check_and_increment_by_n(
            descriptors=list(enforced_descriptors),
            increments=[per_request_increment for _ in enforced_descriptors],
            parent_otel_span=parent_otel_span,
        )

        verbose_proxy_logger.debug(
            "Atomic check+increment response: %s", json.dumps(atomic_response, indent=2, default=list)
        )

        capacity: Final = bucket.capacity
        model_tpm: Final = capacity.tpm if capacity.tpm is not None else "not configured"
        model_rpm: Final = capacity.rpm if capacity.rpm is not None else "not configured"
        if atomic_response["overall_code"] == "OVER_LIMIT":
            resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(deployment_model)
            for status in atomic_response["statuses"]:
                if status["code"] != "OVER_LIMIT":
                    continue
                descriptor_key = status["descriptor_key"]
                if descriptor_key == "model_saturation_check":
                    raise _ModelCapacityRateLimitError(
                        detail={
                            "error": f"Model capacity reached for {model_group}. "
                            f"Priority: {priority}, "
                            f"Rate limit type: {status['rate_limit_type']}, "
                            f"Model TPM: {model_tpm}, "
                            f"Model RPM: {model_rpm}, "
                            f"Remaining: {status['limit_remaining']}"
                        },
                        headers={
                            "retry-after": str(self.v3_limiter.window_size),
                            "rate_limit_type": str(status["rate_limit_type"]),
                            "x-litellm-priority": priority or "default",
                        },
                        rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
                        model=resolved_model,
                        llm_provider=llm_provider,
                    )
                if descriptor_key == "priority_model":
                    verbose_proxy_logger.debug(
                        "Enforcing priority limits for bucket %s, saturation: %.1f%%, priority: %s",
                        bucket.key,
                        saturation * 100,
                        priority,
                    )
                    raise ProxyRateLimitError(
                        detail={
                            "error": f"Priority-based rate limit exceeded. "
                            f"Model: {model_group}, "
                            f"Priority: {priority}, "
                            f"Rate limit type: {status['rate_limit_type']}, "
                            f"Model TPM: {model_tpm}, "
                            f"Model RPM: {model_rpm}, "
                            f"Remaining: {status['limit_remaining']}, "
                            f"Model saturation: {saturation:.1%}"
                        },
                        headers={
                            "retry-after": str(self.v3_limiter.window_size),
                            "rate_limit_type": str(status["rate_limit_type"]),
                            "x-litellm-priority": priority or "default",
                            "x-litellm-saturation": f"{saturation:.2%}",
                        },
                        rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
                        model=resolved_model,
                        llm_provider=llm_provider,
                    )

            # Fail-closed guard: overall_code says OVER_LIMIT but no status
            # matched a descriptor key we know how to translate into a 429.
            # Refuse the request rather than silently fall through and let an
            # over-limit request proceed to the model. Without this, a future
            # caller wiring an unfamiliar descriptor into enforced_descriptors
            # would silently bypass the rate limit.
            offending: Final = next(
                (s for s in atomic_response["statuses"] if s["code"] == "OVER_LIMIT"),
                None,
            )
            verbose_proxy_logger.error(
                "Dynamic rate limiter: OVER_LIMIT response with unknown descriptor_key(s) — refusing request. response=%s",
                atomic_response,
            )
            raise ProxyRateLimitError(
                detail={
                    "error": "Rate limit exceeded",
                    "descriptor_key": (offending["descriptor_key"] if offending else "unknown"),
                    "rate_limit_type": (str(offending["rate_limit_type"]) if offending else "unknown"),
                },
                rate_limit_type=map_v3_rate_limit_type(offending["rate_limit_type"] if offending else None),
                headers={
                    "retry-after": str(self.v3_limiter.window_size),
                    "x-litellm-priority": priority or "default",
                },
                model=resolved_model,
                llm_provider=llm_provider,
            )

        # If priority is NOT enforced (saturation below threshold) but
        # priority_descriptors exist, increment them for tracking only — no
        # check, no rollback. This matches the prior tracking semantics.
        #
        # Using the non-atomic should_rate_limit (instead of
        # atomic_check_and_increment_by_n) is intentional here: we don't want
        # to enforce the limit, we only want to bump the counter so the
        # priority allocation has accurate usage when it later becomes
        # enforced. The increment-then-check semantics of should_rate_limit
        # are fine because we ignore the OVER_LIMIT response.
        if priority_descriptors and not should_enforce_priority:
            priority_tracking_response: Final = await self.v3_limiter.should_rate_limit(
                descriptors=priority_descriptors,
                parent_otel_span=parent_otel_span,
                read_only=False,
            )
            get_or_create_request_stash().rate_limit_response = RateLimitResponse(
                overall_code=atomic_response["overall_code"],
                statuses=atomic_response["statuses"] + priority_tracking_response["statuses"],
            )
        else:
            get_or_create_request_stash().rate_limit_response = atomic_response
        return atomic_response

    @with_service_target("rate_limits")
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Exception | str | dict | None:
        """
        Record the calling team's class for ``async_pre_call_check``, which admits the request against the bucket
        of the deployment the router picks. No counter moves here.
        """
        match data.get("model"):
            case str() as model_group:
                claim_request_stash_for_data(data)
                _admission_request.set(
                    _AdmissionRequest(
                        model_group=model_group,
                        team_class=self._get_priority_from_user_api_key_dict(user_api_key_dict=user_api_key_dict),
                        request_data=data,
                    )
                )
            case _:
                pass
        return None

    @with_service_target("rate_limits")
    async def async_pre_call_check(self, deployment: Mapping[str, object], parent_otel_span: "Span | None") -> None:
        """
        Admit the call against the bucket of the deployment the router picked, before the provider call.

        A refusal reaches the caller as is: the router neither retries it, falls back, nor cools the deployment down.
        """
        try:
            await self._admit(deployment=deployment, parent_otel_span=parent_otel_span)
        except HTTPException as refusal:
            mark_admission_refusal(refusal)
            raise

    async def _admit(self, deployment: Mapping[str, object], parent_otel_span: "Span | None") -> None:
        try:
            routed: Final = _ROUTED_DEPLOYMENT.validate_python(deployment)
        except ValidationError as e:
            verbose_proxy_logger.debug("Dynamic rate limiter: deployment has no bucket fields, serving: %s", e)
            return
        bucket: Final = _admission_bucket(routed)
        if bucket is None:
            verbose_proxy_logger.debug("Dynamic rate limiter: no bucket limits for %s, serving", routed["model_name"])
            return

        request: Final = _admission_request.get()
        model_group: Final = request.model_group if request is not None else routed["model_name"]
        priority: Final = request.team_class if request is not None else None
        request_data: Final[Mapping[str, object]] = (
            request.request_data if request is not None else MappingProxyType({})
        )

        saturation: Final = await self._check_bucket_saturation(bucket)
        try:
            verbose_proxy_logger.debug(
                "[Dynamic Rate Limiter] Bucket=%s, Saturation=%.1f%%, Threshold=%.1f%%, Priority=%s",
                bucket.key,
                saturation * 100,
                _get_priority_settings().saturation_threshold * 100,
                priority,
            )
            atomic_response: Final = await self._check_rate_limits(
                bucket=bucket,
                model_group=model_group,
                deployment_model=routed["litellm_params"].get("model", model_group),
                priority=priority,
                saturation=saturation,
                parent_otel_span=parent_otel_span,
            )
        except _ModelCapacityRateLimitError:
            _record_model_group_saturation(
                model=model_group, pre_read_saturation=1.0, admission=None, data=request_data
            )
            raise
        except ProxyRateLimitError:
            _record_model_group_saturation(
                model=model_group, pre_read_saturation=saturation, admission=None, data=request_data
            )
            raise
        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error("Error in dynamic rate limiter: %s, allowing request", e)
            _record_model_group_saturation(
                model=model_group, pre_read_saturation=saturation, admission=None, data=request_data
            )
            return

        _record_model_group_saturation(
            model=model_group, pre_read_saturation=saturation, admission=atomic_response, data=request_data
        )

    @with_service_target("rate_limits")
    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ) -> LLMResponseTypes:
        """
        Post-call hook to add rate limit headers to response.
        Leverages v3 limiter's post-call hook functionality.
        """
        try:
            # Call v3 limiter's post-call hook to add standard rate limit headers
            await self.v3_limiter.async_post_call_success_hook(
                data=data, user_api_key_dict=user_api_key_dict, response=response
            )

            if response_has_hidden_params(response):
                priority: Final = self._get_priority_from_user_api_key_dict(user_api_key_dict=user_api_key_dict)
                additional_headers: Final = ensure_response_additional_headers(response)
                priority_header: Final = priority or "default"
                if _is_latin1_encodable(priority_header):
                    additional_headers["x-litellm-priority"] = priority_header
                else:
                    verbose_proxy_logger.debug(
                        "Skipping x-litellm-priority header: priority %r is not Latin-1 encodable", priority
                    )
                additional_headers["x-litellm-rate-limiter-version"] = "v3"

            return response

        except Exception as e:
            verbose_proxy_logger.exception("Error in dynamic rate limiter v3 post-call hook: %s", e)
            return response

    @with_service_target("rate_limits")
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """
        Charge the tokens of a served call to the bucket of the deployment that served it.

        Increments token counters for:
        - model_saturation_check: the bucket's whole window
        - priority_model: the team class's pool in that bucket
        """
        from litellm.litellm_core_utils.core_helpers import (
            get_parent_otel_span_from_kwargs,
        )
        from litellm.types.caching import RedisPipelineIncrementOperation
        from litellm.types.utils import Usage

        try:
            verbose_proxy_logger.debug("INSIDE dynamic rate limiter ASYNC SUCCESS LOGGING")

            litellm_parent_otel_span: Final = get_parent_otel_span_from_kwargs(kwargs)

            bucket_key: Final = _served_bucket_key(kwargs.get("litellm_params"))
            if bucket_key is None:
                return

            priority: Final = team_priority_label(kwargs.get("litellm_params")) or None

            # Get total tokens from response
            total_tokens = 0
            rate_limit_type: Final = self.v3_limiter.get_rate_limit_type()

            if isinstance(response_obj, ModelResponse):
                _usage: Final = getattr(response_obj, "usage", None)
                if _usage and isinstance(_usage, Usage):
                    if rate_limit_type == "output":
                        total_tokens = _usage.completion_tokens
                    elif rate_limit_type == "input":
                        total_tokens = _usage.prompt_tokens
                    elif rate_limit_type == "total":
                        total_tokens = _usage.total_tokens

            if total_tokens == 0:
                return

            pipeline_operations: Final = [
                RedisPipelineIncrementOperation(
                    key=self.v3_limiter.create_rate_limit_keys(
                        key=descriptor_key, value=descriptor_value, rate_limit_type="tokens"
                    ),
                    increment_value=total_tokens,
                    ttl=self.v3_limiter.window_size,
                )
                for descriptor_key, descriptor_value in (
                    ("model_saturation_check", bucket_key),
                    ("priority_model", _priority_pool_key(bucket_key, priority)),
                )
            ]

            await self.v3_limiter.async_increment_tokens_with_ttl_preservation(
                pipeline_operations=pipeline_operations,
                parent_otel_span=litellm_parent_otel_span,
            )

            # Only log 'priority' if it's known safe; otherwise, redact.
            SAFE_PRIORITIES: Final = {"low", "medium", "high", "default"}
            logged_priority: Final = priority if priority in SAFE_PRIORITIES else "REDACTED"
            verbose_proxy_logger.debug(
                "[Dynamic Rate Limiter] Incremented tokens by %s for bucket=%s, priority=%s",
                total_tokens,
                bucket_key,
                logged_priority,
            )

        except Exception as e:
            verbose_proxy_logger.exception("Error in dynamic rate limiter success event: %s", e)


PROXY_DynamicRateLimitHandlerV3: Final = _PROXY_DynamicRateLimitHandlerV3
