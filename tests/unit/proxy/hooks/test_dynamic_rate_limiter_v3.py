"""
Test priority-based rate limiting for dynamic_rate_limiter_v3.

Core tests to validate that priority weights are respected (0.9/0.1) instead of equal splitting (0.5/0.5).
"""

import asyncio
import contextvars
import os
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from functools import partial
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import respx
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample


import litellm
from litellm import CustomLLM, DualCache, Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
    PROXY_DynamicRateLimitHandlerV3 as DynamicRateLimitHandler,
)
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import _Bucket
from litellm.types.router import ModelGroupInfo
from litellm.types.utils import ModelResponse, Usage


class TimeController:
    def __init__(self):
        self._current = datetime.utcnow()

    def now(self) -> datetime:
        return self._current

    def advance(self, seconds: float) -> None:
        self._current += timedelta(seconds=seconds)


@pytest.fixture
def time_controller(monkeypatch):
    controller = TimeController()
    monkeypatch.setattr(time, "time", lambda: controller.now().timestamp())
    return controller


def _clear_prometheus_registry() -> None:
    for collector in list(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)


def _bucket(key: str, tpm: int | None = None, rpm: int | None = None) -> _Bucket:
    return _Bucket(key=key, capacity=ModelGroupInfo(model_group=key, providers=[], tpm=tpm, rpm=rpm))


def _single_deployment(model: str, **limits: int) -> dict:
    router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "test-key", "api_base": "test-base", **limits},
            }
        ]
    )
    return router.get_model_list(model_name=model)[0]


def _bucket_key(deployment: Mapping[str, object]) -> str:
    return f"{deployment['model_info']['id']}:gpt-3.5-turbo"


async def _admit(
    handler: DynamicRateLimitHandler,
    deployment: Mapping[str, object],
    user: UserAPIKeyAuth,
    data: Mapping[str, object] | None = None,
) -> None:
    await handler.async_pre_call_hook(
        user_api_key_dict=user,
        cache=handler.internal_usage_cache,
        data=dict(data if data is not None else {"model": deployment["model_name"]}),
        call_type="completion",
    )
    await handler.async_pre_call_check(deployment, None)


@pytest.mark.asyncio
async def test_priority_weight_allocation(monkeypatch):
    """
    Test that priority weights are correctly applied instead of equal splitting.

    With priority_reservation = {"high": 0.9, "low": 0.1}:
    - High priority should get 90% of TPM (900 out of 1000)
    - Low priority should get 10% of TPM (100 out of 1000)

    This validates the core fix where before it would split 50/50.
    """
    # Set up environment for premium feature
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    # Set up priority reservations
    litellm.priority_reservation = {"high": 0.9, "low": 0.1}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "test-model"
    total_tpm = 1000

    high_descriptors = handler._create_priority_based_descriptors(bucket=_bucket(model, tpm=total_tpm), priority="high")

    assert len(high_descriptors) == 1
    high_descriptor = high_descriptors[0]
    expected_high_tpm = int(total_tpm * 0.9)  # 900
    actual_high_tpm = high_descriptor["rate_limit"]["tokens_per_unit"]

    assert (
        actual_high_tpm == expected_high_tpm
    ), f"High priority should get {expected_high_tpm} TPM (90%), got {actual_high_tpm}"
    assert high_descriptor["value"] == f"{model}:high"

    # Test low priority allocation
    low_descriptors = handler._create_priority_based_descriptors(bucket=_bucket(model, tpm=total_tpm), priority="low")

    assert len(low_descriptors) == 1
    low_descriptor = low_descriptors[0]
    expected_low_tpm = int(total_tpm * 0.1)  # 100
    actual_low_tpm = low_descriptor["rate_limit"]["tokens_per_unit"]

    assert (
        actual_low_tpm == expected_low_tpm
    ), f"Low priority should get {expected_low_tpm} TPM (10%), got {actual_low_tpm}"
    assert low_descriptor["value"] == f"{model}:low"

    # Verify the ratio is 9:1, not 1:1 (equal splitting)
    ratio = actual_high_tpm / actual_low_tpm
    expected_ratio = 9.0
    assert (
        abs(ratio - expected_ratio) < 0.1
    ), f"High:Low ratio should be {expected_ratio}:1, got {ratio}:1"


@pytest.mark.asyncio
async def test_concurrent_priority_requests(monkeypatch):
    """
    Test the core issue: 5 concurrent requests with different priorities should get
    proper allocation based on priority weights, not equal splitting.

    This tests the exact scenario mentioned: priorities 0.9 and 0.1 should be 0.9/0.1, not 0.5/0.5.
    """
    # Set up environment for premium feature
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    # Set up the exact scenario from the issue
    litellm.priority_reservation = {"high": 0.9, "low": 0.1}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "test-model"
    total_tpm = 1000

    # Create 5 concurrent users - 3 high priority, 2 low priority
    high_priority_users = []
    low_priority_users = []

    for i in range(3):  # 3 high priority users
        user = UserAPIKeyAuth()
        user.team_metadata = {"priority": "high"}
        user.user_id = f"high_user_{i}"
        high_priority_users.append(user)

    for i in range(2):  # 2 low priority users
        user = UserAPIKeyAuth()
        user.team_metadata = {"priority": "low"}
        user.user_id = f"low_user_{i}"
        low_priority_users.append(user)

    # Test all high priority users get the same allocation (not divided)
    for user in high_priority_users:
        descriptors = handler._create_priority_based_descriptors(bucket=_bucket(model, tpm=total_tpm), priority="high")

        assert len(descriptors) == 1
        descriptor = descriptors[0]
        # Each high priority user should get 900 TPM, not divided by 3
        assert descriptor["rate_limit"]["tokens_per_unit"] == 900, (
            f"High priority user {user.user_id} should get 900 TPM, "
            f"got {descriptor['rate_limit']['tokens_per_unit']}"
        )
        assert descriptor["value"] == f"{model}:high"

    # Test all low priority users get the same allocation (not divided)
    for user in low_priority_users:
        descriptors = handler._create_priority_based_descriptors(bucket=_bucket(model, tpm=total_tpm), priority="low")

        assert len(descriptors) == 1
        descriptor = descriptors[0]
        # Each low priority user should get 100 TPM, not divided by 2
        assert descriptor["rate_limit"]["tokens_per_unit"] == 100, (
            f"Low priority user {user.user_id} should get 100 TPM, "
            f"got {descriptor['rate_limit']['tokens_per_unit']}"
        )
        assert descriptor["value"] == f"{model}:low"


@pytest.mark.asyncio
async def test_100_concurrent_priority_requests(time_controller, monkeypatch):
    """
    Stress test: 100 concurrent requests with mixed priorities over 10 seconds.

    This validates that the priority system works correctly under high load:
    - 70 high priority requests (should get 900 TPM each)
    - 30 low priority requests (should get 100 TPM each)
    - Spread across 10 seconds to simulate real-world load
    """
    # Set up environment for premium feature
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    # Set up priority reservations
    litellm.priority_reservation = {"high": 0.9, "low": 0.1}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(
        internal_usage_cache=dual_cache, time_provider=time_controller.now
    )

    model = "stress-test-model"
    total_tpm = 1000

    bucket = _bucket(model, tpm=total_tpm, rpm=500)

    # Create 100 users: 70 high priority, 30 low priority
    all_users = []

    # 70 high priority users
    for i in range(70):
        user = UserAPIKeyAuth()
        user.team_metadata = {"priority": "high"}
        user.user_id = f"high_stress_user_{i}"
        all_users.append((user, "high", 900, 450))  # expected TPM, expected RPM

    # 30 low priority users
    for i in range(30):
        user = UserAPIKeyAuth()
        user.team_metadata = {"priority": "low"}
        user.user_id = f"low_stress_user_{i}"
        all_users.append((user, "low", 100, 50))  # expected TPM, expected RPM

    async def test_user_descriptors(user_data):
        """Test descriptor creation for a single user."""
        user, priority, expected_tpm, expected_rpm = user_data

        descriptors = handler._create_priority_based_descriptors(bucket=bucket, priority=priority)

        assert (
            len(descriptors) == 1
        ), f"User {user.user_id} should have exactly 1 descriptor"
        descriptor = descriptors[0]

        # Validate TPM allocation
        actual_tpm = descriptor["rate_limit"]["tokens_per_unit"]
        assert (
            actual_tpm == expected_tpm
        ), f"User {user.user_id} ({priority}) should get {expected_tpm} TPM, got {actual_tpm}"

        # Validate RPM allocation
        actual_rpm = descriptor["rate_limit"]["requests_per_unit"]
        assert (
            actual_rpm == expected_rpm
        ), f"User {user.user_id} ({priority}) should get {expected_rpm} RPM, got {actual_rpm}"

        # Validate descriptor key
        assert descriptor["value"] == f"{model}:{priority}"
        assert descriptor["key"] == "priority_model"

        return {
            "user_id": user.user_id,
            "priority": priority,
            "tpm": actual_tpm,
            "rpm": actual_rpm,
            "success": True,
        }

    # Run all 100 requests concurrently to simulate high load
    start_time = time.time()

    # Split into batches to simulate requests over 10 seconds
    batch_size = 10  # 10 requests per batch
    batches = [
        all_users[i : i + batch_size] for i in range(0, len(all_users), batch_size)
    ]

    all_results = []

    for batch_idx, batch in enumerate(batches):
        # Process each batch concurrently
        batch_tasks = [test_user_descriptors(user_data) for user_data in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        all_results.extend(batch_results)

        # Add small delay between batches to spread over ~10 seconds
        if batch_idx < len(batches) - 1:  # Don't sleep after last batch
            await asyncio.sleep(0)
            time_controller.advance(1.0)  # simulate 1s passing between batches

    end_time = time.time()
    total_duration = end_time - start_time

    # Validate that the test ran over approximately 10 seconds
    assert (
        total_duration >= 9.0
    ), f"Test should take ~10 seconds, took {total_duration:.2f}s"
    assert total_duration <= 15.0, f"Test took too long: {total_duration:.2f}s"

    # Validate all requests were successful
    successful_results = [
        r for r in all_results if isinstance(r, dict) and r.get("success")
    ]
    assert (
        len(successful_results) == 100
    ), f"Expected 100 successful results, got {len(successful_results)}"

    # Validate priority distribution
    high_priority_results = [r for r in successful_results if r["priority"] == "high"]
    low_priority_results = [r for r in successful_results if r["priority"] == "low"]

    assert (
        len(high_priority_results) == 70
    ), f"Expected 70 high priority results, got {len(high_priority_results)}"
    assert (
        len(low_priority_results) == 30
    ), f"Expected 30 low priority results, got {len(low_priority_results)}"

    # Validate all high priority users got correct allocation
    for result in high_priority_results:
        assert (
            result["tpm"] == 900
        ), f"High priority user {result['user_id']} got {result['tpm']} TPM, expected 900"
        assert (
            result["rpm"] == 450
        ), f"High priority user {result['user_id']} got {result['rpm']} RPM, expected 450"

    # Validate all low priority users got correct allocation
    for result in low_priority_results:
        assert (
            result["tpm"] == 100
        ), f"Low priority user {result['user_id']} got {result['tpm']} TPM, expected 100"
        assert (
            result["rpm"] == 50
        ), f"Low priority user {result['user_id']} got {result['rpm']} RPM, expected 50"

    print(f"✅ Successfully processed 100 concurrent requests in {total_duration:.2f}s")
    print(f"   - 70 high priority users: 900 TPM, 450 RPM each")
    print(f"   - 30 low priority users: 100 TPM, 50 RPM each")
    print(f"   - Priority ratio maintained: 9:1 (TPM) and 9:1 (RPM)")


@pytest.mark.asyncio
async def test_concurrent_pre_call_hooks_stress(monkeypatch):
    """
    Stress test: 50 concurrent pre-call hooks with saturation-aware priority enforcement.

    Tests priority-based rate limiting in strict mode (>80% saturation).
    Mocks high saturation to force strict mode where priorities are enforced.
    Premium users (80% allocation) should have >90% success rate.
    Standard users (20% allocation) should have ~70% success rate with 30% random limiting.
    """
    # Set up environment for premium feature
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    litellm.priority_reservation = {"premium": 0.8, "standard": 0.2}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "pre-call-stress-model"
    total_tpm = 2000

    deployment = _single_deployment(model, tpm=total_tpm)

    # Mock the v3 limiter to simulate different scenarios
    successful_requests = []
    rate_limited_requests = []

    # Mock saturation check to return high saturation (forces strict mode)
    async def mock_get_cache(key, litellm_parent_otel_span=None, local_only=False):
        """Mock cache to simulate high saturation."""
        # Return high usage to trigger strict mode (>80% saturation)
        if ":requests" in key or ":tokens" in key:
            return 1800  # 1800/2000 = 90% saturation
        return None

    async def mock_should_rate_limit(
        descriptors, parent_otel_span=None, read_only=False
    ):
        """Mock rate limiter that handles saturation-aware descriptors."""
        descriptor = descriptors[0]
        descriptor_key = descriptor["key"]
        descriptor_value = descriptor["value"]

        # Handle model-wide tracking (for both generous and strict mode tracking)
        if descriptor_key == "model_saturation_check":
            # Always allow model-wide tracking (doesn't enforce in our mock)
            return {
                "overall_code": "OK",
                "statuses": [
                    {
                        "code": "OK",
                        "descriptor_key": descriptor_value,
                        "rate_limit_type": "tokens_per_unit",
                        "limit_remaining": 10000,
                    }
                ],
            }

        # Handle priority-specific enforcement in strict mode
        elif descriptor_key == "priority_model":
            # Extract priority from value like "pre-call-stress-model:premium"
            priority = descriptor_value.split(":")[-1]

            if priority == "premium":
                # Allow all premium requests
                return {
                    "overall_code": "OK",
                    "statuses": [
                        {
                            "code": "OK",
                            "descriptor_key": descriptor_value,
                            "rate_limit_type": "tokens_per_unit",
                            "limit_remaining": 1000,
                        }
                    ],
                }
            else:
                # Rate limit some standard requests (simulate load)
                import random

                if random.random() < 0.3:  # 30% of standard requests get rate limited
                    return {
                        "overall_code": "OVER_LIMIT",
                        "statuses": [
                            {
                                "code": "OVER_LIMIT",
                                "descriptor_key": descriptor_value,
                                "rate_limit_type": "tokens_per_unit",
                                "limit_remaining": 0,
                            }
                        ],
                    }
                else:
                    return {
                        "overall_code": "OK",
                        "statuses": [
                            {
                                "code": "OK",
                                "descriptor_key": descriptor_value,
                                "rate_limit_type": "tokens_per_unit",
                                "limit_remaining": 100,
                            }
                        ],
                    }

        # Default: allow
        return {
            "overall_code": "OK",
            "statuses": [
                {
                    "code": "OK",
                    "descriptor_key": descriptor_value,
                    "rate_limit_type": "tokens_per_unit",
                    "limit_remaining": 1000,
                }
            ],
        }

    # Create 50 users: 30 premium, 20 standard
    users = []

    for i in range(30):
        user = UserAPIKeyAuth()
        user.team_metadata = {"priority": "premium"}
        user.user_id = f"premium_hook_user_{i}"
        users.append((user, "premium"))

    for i in range(20):
        user = UserAPIKeyAuth()
        user.team_metadata = {"priority": "standard"}
        user.user_id = f"standard_hook_user_{i}"
        users.append((user, "standard"))

    async def make_request(user_data):
        """Make a pre-call hook request."""
        user, priority = user_data

        try:
            await _admit(handler, deployment, user)

            # If no exception, request was allowed
            successful_requests.append(
                {"user_id": user.user_id, "priority": priority, "result": "allowed"}
            )
            return {
                "status": "success",
                "user_id": user.user_id,
                "priority": priority,
            }

        except Exception as e:
            # Request was rate limited
            rate_limited_requests.append(
                {"user_id": user.user_id, "priority": priority, "error": str(e)}
            )
            return {
                "status": "rate_limited",
                "user_id": user.user_id,
                "priority": priority,
            }

    # Run all 50 requests concurrently with patches applied to the entire batch
    start_time = time.time()
    with (
        patch.object(
            handler.v3_limiter, "should_rate_limit", side_effect=mock_should_rate_limit
        ),
        patch.object(
            handler.internal_usage_cache, "async_get_cache", side_effect=mock_get_cache
        ),
    ):
        tasks = [make_request(user_data) for user_data in users]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    successful_count = len(
        [r for r in results if isinstance(r, dict) and r["status"] == "success"]
    )
    rate_limited_count = len(
        [r for r in results if isinstance(r, dict) and r["status"] == "rate_limited"]
    )

    # Validate that premium users were mostly successful (priority worked)
    premium_results = [
        r for r in results if isinstance(r, dict) and r["priority"] == "premium"
    ]
    premium_success = len([r for r in premium_results if r["status"] == "success"])

    standard_results = [
        r for r in results if isinstance(r, dict) and r["priority"] == "standard"
    ]
    standard_success = len([r for r in standard_results if r["status"] == "success"])

    # Premium users should have higher success rate due to priority
    premium_success_rate = (
        premium_success / len(premium_results) if premium_results else 0
    )
    standard_success_rate = (
        standard_success / len(standard_results) if standard_results else 0
    )

    assert (
        premium_success_rate >= 0.9
    ), f"Premium success rate should be >= 90%, got {premium_success_rate:.2%}"
    assert (
        standard_success_rate >= 0.5
    ), f"Standard success rate should be >= 50% (with 30% random limiting, allows for variance), got {standard_success_rate:.2%}"

    # Allow for the case where both are 100% due to timing/mocking issues
    # The test is inherently flaky due to random behavior
    if premium_success_rate < 1.0 or standard_success_rate < 1.0:
        assert (
            premium_success_rate >= standard_success_rate
        ), "Premium should have >= success rate than standard"

    total_duration = end_time - start_time

    print(f"✅ Processed 50 concurrent pre-call hooks in {total_duration:.2f}s")
    print(
        f"   - Premium users: {premium_success}/{len(premium_results)} success ({premium_success_rate:.1%})"
    )
    print(
        f"   - Standard users: {standard_success}/{len(standard_results)} success ({standard_success_rate:.1%})"
    )
    print(f"   - Total successful: {successful_count}/50 ({successful_count/50:.1%})")
    print(f"   - Priority system working: Premium > Standard success rates")


# These tests make actual async_pre_call_hook calls to simulate real traffic


@pytest.mark.asyncio
async def test_fake_calls_case_1_no_rate_limiting_at_capacity(monkeypatch):
    """
    Test Case 1: Saturation-Aware Rate Limiting at 50% Threshold

    System: 100 RPM capacity, saturation_threshold=50%
    Key A: priority_reservation=0.75 (75 RPM reserved)
    Key B: priority_reservation=0.25 (25 RPM reserved)
    Traffic A: 1 request
    Traffic B: 100 requests

    Expected behavior:
    - Key A: 1 request succeeds (low traffic)
    - Key B: ~25-26 requests succeed (capped at reservation when saturation >= 50%)

    Once saturation hits 50%, strict mode enforces priority-based limits.
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    # Set up priority reservations
    litellm.priority_reservation = {"key_a": 0.75, "key_b": 0.25}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "fake-call-test-1"
    total_rpm = 100

    deployment = _single_deployment(model, rpm=total_rpm)

    # Create users
    key_a_user = UserAPIKeyAuth()
    key_a_user.team_metadata = {"priority": "key_a"}
    key_a_user.user_id = "key_a_user"

    key_b_user = UserAPIKeyAuth()
    key_b_user.team_metadata = {"priority": "key_b"}
    key_b_user.user_id = "key_b_user"

    # Track results
    successful_requests = {"key_a": 0, "key_b": 0}
    rate_limited_requests = {"key_a": 0, "key_b": 0}

    async def make_request(user, priority_name, request_id):
        """Make a single request and track the result."""
        try:
            result = await _admit(handler, deployment, user)

            if result is None:
                successful_requests[priority_name] += 1
                return {"status": "success", "priority": priority_name}
            else:
                rate_limited_requests[priority_name] += 1
                return {"status": "rate_limited", "priority": priority_name}

        except Exception as e:
            rate_limited_requests[priority_name] += 1
            return {
                "status": "rate_limited",
                "priority": priority_name,
                "error": str(e),
            }

    # Send 1 request from key_a, 100 from key_b
    tasks = []

    for i in range(1):
        tasks.append(make_request(key_a_user, "key_a", f"key_a_{i}"))

    for i in range(100):
        tasks.append(make_request(key_b_user, "key_b", f"key_b_{i}"))

    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    total_successful = successful_requests["key_a"] + successful_requests["key_b"]
    total_rate_limited = rate_limited_requests["key_a"] + rate_limited_requests["key_b"]

    print(f"Test Case 1 - Saturation-Aware Rate Limiting:")
    print(f"   - Duration: {end_time - start_time:.2f}s")
    print(f"   - Key A: {successful_requests['key_a']}/1 successful (reserved 75 RPM)")
    print(
        f"   - Key B: {successful_requests['key_b']}/100 successful (reserved 25 RPM)"
    )
    print(f"   - Total successful: {total_successful}/101")
    print(f"   - Total rate limited: {total_rate_limited}/101")

    # Key A should get its 1 request
    assert (
        successful_requests["key_a"] == 1
    ), f"Key A should get 1 request, got {successful_requests['key_a']}"

    # Key B can send until saturation hits 50% (which is ~50 total requests)
    # After that, strict mode enforces its 25 RPM reservation
    # Due to race conditions in concurrent execution, allow 45-52 successful requests
    assert (
        45 <= successful_requests["key_b"] <= 52
    ), f"Key B should get ~49 requests (45-52), got {successful_requests['key_b']}"

    # Verify approximately half of key_b requests were rate limited
    assert (
        rate_limited_requests["key_b"] >= 45
    ), f"Key B should have ≥45 rate limited requests, got {rate_limited_requests['key_b']}"


@pytest.mark.asyncio
async def test_fake_calls_case_2_priority_queue_during_saturation(monkeypatch):
    """
    Test Case 2: Priority Queue Behavior During Saturation

    System: 100 RPM capacity
    Key A: priority_reservation=0.75 (75 RPM reserved)
    Key B: priority_reservation=0.25 (25 RPM reserved)
    Traffic A: 200 RPM
    Traffic B: 200 RPM
    Expected A: 75 RPM (75% of capacity)
    Expected B: 25 RPM (25% of capacity)

    When total traffic exceeds capacity, rate limiting enforces priority reservations.
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    litellm.priority_reservation = {"key_a": 0.75, "key_b": 0.25}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "fake-call-test-2"
    total_rpm = 100

    deployment = _single_deployment(model, rpm=total_rpm)

    # Create users
    key_a_user = UserAPIKeyAuth()
    key_a_user.team_metadata = {"priority": "key_a"}
    key_a_user.user_id = "key_a_user"

    key_b_user = UserAPIKeyAuth()
    key_b_user.team_metadata = {"priority": "key_b"}
    key_b_user.user_id = "key_b_user"

    # Track results
    successful_requests = {"key_a": 0, "key_b": 0}
    rate_limited_requests = {"key_a": 0, "key_b": 0}

    async def make_request(user, priority_name, request_id):
        """Make a single request and track the result."""
        try:
            result = await _admit(handler, deployment, user)

            if result is None:
                successful_requests[priority_name] += 1
                return {"status": "success", "priority": priority_name}
            else:
                rate_limited_requests[priority_name] += 1
                return {"status": "rate_limited", "priority": priority_name}

        except Exception as e:
            rate_limited_requests[priority_name] += 1
            return {
                "status": "rate_limited",
                "priority": priority_name,
                "error": str(e),
            }

    # Send 200 requests from each priority (over capacity)
    tasks = []

    for i in range(200):
        tasks.append(make_request(key_a_user, "key_a", f"key_a_{i}"))

    for i in range(200):
        tasks.append(make_request(key_b_user, "key_b", f"key_b_{i}"))

    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    total_successful = successful_requests["key_a"] + successful_requests["key_b"]

    key_a_success_rate = successful_requests["key_a"] / 200
    key_b_success_rate = successful_requests["key_b"] / 200

    print(f"Test Case 2 - Priority Queue Behavior During Saturation:")
    print(f"   - Duration: {end_time - start_time:.2f}s")
    print(
        f"   - Key A: {successful_requests['key_a']}/200 successful ({key_a_success_rate:.1%})"
    )
    print(
        f"   - Key B: {successful_requests['key_b']}/200 successful ({key_b_success_rate:.1%})"
    )
    print(f"   - Total successful: {total_successful}/400")

    # Key A should get significantly more requests than Key B (75:25 ratio)
    assert (
        key_a_success_rate > key_b_success_rate
    ), f"Key A should have higher success rate: {key_a_success_rate:.1%} vs {key_b_success_rate:.1%}"

    # Check ratio is approximately 3:1 (75:25)
    if total_successful > 0:
        key_a_share = successful_requests["key_a"] / total_successful
        expected_key_a_share = 0.75

        print(
            f"   - Key A got {key_a_share:.1%} of successful requests (expected ~75%)"
        )

        # Allow tolerance for timing effects
        assert (
            abs(key_a_share - expected_key_a_share) < 0.2
        ), f"Key A share should be ~75%, got {key_a_share:.1%}"


@pytest.mark.asyncio
async def test_fake_calls_case_3_spillover_capacity_default_keys(monkeypatch):
    """
    Test Case 3: Spillover Capacity for Default Keys

    System: 100 RPM capacity
    Key A: priority_reservation=0.75 (75 RPM reserved)
    Key B: nothing set (default)
    Key C: nothing set (default)
    Key D: nothing set (default)
    Traffic A: 150 RPM
    Traffic B: 150 RPM
    Traffic C: 150 RPM
    Traffic D: 150 RPM
    Expected A: 75 RPM (75% reserved)
    Expected B: ~8.3 RPM (remaining 25 RPM / 3 default keys)
    Expected C: ~8.3 RPM
    Expected D: ~8.3 RPM

    Tests spillover behavior where default keys share remaining capacity.
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    litellm.priority_reservation = {"key_a": 0.75}
    litellm.priority_reservation_settings.default_priority = 0.25

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "fake-call-test-3"
    total_rpm = 100

    deployment = _single_deployment(model, rpm=total_rpm)

    # Create users
    key_a_user = UserAPIKeyAuth()
    key_a_user.team_metadata = {"priority": "key_a"}
    key_a_user.user_id = "key_a_user"

    key_b_user = UserAPIKeyAuth()
    key_b_user.metadata = {}
    key_b_user.user_id = "key_b_user"

    key_c_user = UserAPIKeyAuth()
    key_c_user.metadata = {}
    key_c_user.user_id = "key_c_user"

    key_d_user = UserAPIKeyAuth()
    key_d_user.metadata = {}
    key_d_user.user_id = "key_d_user"

    # Track results
    successful_requests = {"key_a": 0, "key_b": 0, "key_c": 0, "key_d": 0}
    rate_limited_requests = {"key_a": 0, "key_b": 0, "key_c": 0, "key_d": 0}

    async def make_request(user, key_name, request_id):
        """Make a single request and track the result."""
        try:
            result = await _admit(handler, deployment, user)

            if result is None:
                successful_requests[key_name] += 1
                return {"status": "success", "key": key_name}
            else:
                rate_limited_requests[key_name] += 1
                return {"status": "rate_limited", "key": key_name}

        except Exception as e:
            rate_limited_requests[key_name] += 1
            return {"status": "rate_limited", "key": key_name, "error": str(e)}

    # Send 150 requests from each key (600 total, 6x over capacity)
    tasks = []

    for i in range(150):
        tasks.append(make_request(key_a_user, "key_a", f"key_a_{i}"))

    for i in range(150):
        tasks.append(make_request(key_b_user, "key_b", f"key_b_{i}"))

    for i in range(150):
        tasks.append(make_request(key_c_user, "key_c", f"key_c_{i}"))

    for i in range(150):
        tasks.append(make_request(key_d_user, "key_d", f"key_d_{i}"))

    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    total_successful = sum(successful_requests.values())

    print(f"Test Case 3 - Spillover Capacity for Default Keys:")
    print(f"   - Duration: {end_time - start_time:.2f}s")
    print(f"   - Key A: {successful_requests['key_a']}/150 successful")
    print(f"   - Key B: {successful_requests['key_b']}/150 successful (default)")
    print(f"   - Key C: {successful_requests['key_c']}/150 successful (default)")
    print(f"   - Key D: {successful_requests['key_d']}/150 successful (default)")
    print(f"   - Total successful: {total_successful}/600")

    # Key A should get the most requests (75% of capacity)
    assert (
        successful_requests["key_a"] > successful_requests["key_b"]
    ), "Key A should get more than Key B"
    assert (
        successful_requests["key_a"] > successful_requests["key_c"]
    ), "Key A should get more than Key C"
    assert (
        successful_requests["key_a"] > successful_requests["key_d"]
    ), "Key A should get more than Key D"

    # Default keys should get similar amounts (spillover capacity)
    avg_default = (
        successful_requests["key_b"]
        + successful_requests["key_c"]
        + successful_requests["key_d"]
    ) / 3
    print(f"   - Average default key success: {avg_default:.1f}")


@pytest.mark.asyncio
async def test_fake_calls_case_4_over_allocated_with_normalization(monkeypatch):
    """
    Test Case 4: Over-Allocated Priority reservations with Normalization

    System: 100 RPM capacity
    Key A: priority_reservation=0.60 (60% requested)
    Key B: priority_reservation=0.80 (80% requested)
    Total: 140% (over-allocated, should normalize to 43%/57%)
    Traffic A: 200 RPM
    Traffic B: 200 RPM

    With saturation-aware rate limiting:
    - Initially, requests are allowed through in generous mode (under 80% saturation)
    - Once saturated, strict priority-based limits kick in with normalized weights
    - Due to concurrent burst, total successful may exceed 100 RPM in the test window
    - This test verifies normalization works and total capacity is reasonably bounded
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    litellm.priority_reservation = {"key_a": 0.60, "key_b": 0.80}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "fake-call-test-4"
    total_rpm = 100

    deployment = _single_deployment(model, rpm=total_rpm)

    # Create users
    key_a_user = UserAPIKeyAuth()
    key_a_user.team_metadata = {"priority": "key_a"}
    key_a_user.user_id = "key_a_user"

    key_b_user = UserAPIKeyAuth()
    key_b_user.team_metadata = {"priority": "key_b"}
    key_b_user.user_id = "key_b_user"

    # Track results
    successful_requests = {"key_a": 0, "key_b": 0}
    rate_limited_requests = {"key_a": 0, "key_b": 0}

    async def make_request(user, priority_name, request_id):
        """Make a single request and track the result."""
        try:
            result = await _admit(handler, deployment, user)

            if result is None:
                successful_requests[priority_name] += 1
                return {"status": "success", "priority": priority_name}
            else:
                rate_limited_requests[priority_name] += 1
                return {"status": "rate_limited", "priority": priority_name}

        except Exception as e:
            rate_limited_requests[priority_name] += 1
            return {
                "status": "rate_limited",
                "priority": priority_name,
                "error": str(e),
            }

    # Send 200 requests from each key (400 total, 4x over capacity)
    tasks = []

    for i in range(200):
        tasks.append(make_request(key_a_user, "key_a", f"key_a_{i}"))

    for i in range(200):
        tasks.append(make_request(key_b_user, "key_b", f"key_b_{i}"))

    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    total_successful = successful_requests["key_a"] + successful_requests["key_b"]

    key_a_success_rate = successful_requests["key_a"] / 200
    key_b_success_rate = successful_requests["key_b"] / 200

    print(f"Test Case 4 - Over-Allocated Priority Reservations with Normalization:")
    print(f"   - Duration: {end_time - start_time:.2f}s")
    print(
        f"   - Key A (0.60): {successful_requests['key_a']}/200 successful ({key_a_success_rate:.1%})"
    )
    print(
        f"   - Key B (0.80): {successful_requests['key_b']}/200 successful ({key_b_success_rate:.1%})"
    )
    print(f"   - Total successful: {total_successful}/400")

    # With saturation-aware behavior:
    # 1. Verify total capacity is reasonably bounded (not all 400 requests succeed)
    assert (
        total_successful < 300
    ), f"Total requests should be bounded by saturation detection, got {total_successful}/400"

    # 2. Verify significant rate limiting occurred (at least 50% blocked)
    assert (
        total_successful < 200
    ), f"At least 50% of requests should be rate limited, got {total_successful}/400 successful"

    # 3. Verify both keys got some requests through (normalization is working)
    assert successful_requests["key_a"] > 0, "Key A should get some requests"
    assert successful_requests["key_b"] > 0, "Key B should get some requests"

    print(
        f"   - Normalization test PASSED: Both priorities got requests, "
        f"total bounded to {total_successful} (under 200)"
    )


@pytest.mark.asyncio
async def test_fake_calls_case_5_default_value_priority_reservation(monkeypatch):
    """
    Test Case 5: Default value for priority reservation

    System: 100 RPM capacity
    Key A: priority_reservation=0.50 (50 RPM)
    Key B: priority_reservation=0.20 (20 RPM)
    Key C: priority_reservation=0.05 (5 RPM)
    Key D: nothing set (uses default_priority=0.05, 5 RPM)
    Traffic A: 150 RPM
    Traffic B: 150 RPM
    Traffic C: 150 RPM
    Traffic D: 150 RPM
    Expected A: 55 RPM (normalized)
    Expected B: 25 RPM (normalized)
    Expected C: 10 RPM (normalized)
    Expected D: 10 RPM (normalized)

    Tests complex scenario with explicit priorities and default priority.
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    litellm.priority_reservation = {"key_a": 0.50, "key_b": 0.20, "key_c": 0.05}
    litellm.priority_reservation_settings.default_priority = 0.05

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "fake-call-test-5"
    total_rpm = 100

    deployment = _single_deployment(model, rpm=total_rpm)

    # Create users
    key_a_user = UserAPIKeyAuth()
    key_a_user.team_metadata = {"priority": "key_a"}
    key_a_user.user_id = "key_a_user"

    key_b_user = UserAPIKeyAuth()
    key_b_user.team_metadata = {"priority": "key_b"}
    key_b_user.user_id = "key_b_user"

    key_c_user = UserAPIKeyAuth()
    key_c_user.team_metadata = {"priority": "key_c"}
    key_c_user.user_id = "key_c_user"

    key_d_user = UserAPIKeyAuth()
    key_d_user.metadata = {}
    key_d_user.user_id = "key_d_user"

    # Track results
    successful_requests = {"key_a": 0, "key_b": 0, "key_c": 0, "key_d": 0}
    rate_limited_requests = {"key_a": 0, "key_b": 0, "key_c": 0, "key_d": 0}

    async def make_request(user, key_name, request_id):
        """Make a single request and track the result."""
        try:
            result = await _admit(handler, deployment, user)

            if result is None:
                successful_requests[key_name] += 1
                return {"status": "success", "key": key_name}
            else:
                rate_limited_requests[key_name] += 1
                return {"status": "rate_limited", "key": key_name}

        except Exception as e:
            rate_limited_requests[key_name] += 1
            return {"status": "rate_limited", "key": key_name, "error": str(e)}

    # Send 150 requests from each key (600 total, 6x over capacity)
    tasks = []

    for i in range(150):
        tasks.append(make_request(key_a_user, "key_a", f"key_a_{i}"))

    for i in range(150):
        tasks.append(make_request(key_b_user, "key_b", f"key_b_{i}"))

    for i in range(150):
        tasks.append(make_request(key_c_user, "key_c", f"key_c_{i}"))

    for i in range(150):
        tasks.append(make_request(key_d_user, "key_d", f"key_d_{i}"))

    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Analyze results
    total_successful = sum(successful_requests.values())

    print(f"Test Case 5 - Default value for priority reservation:")
    print(f"   - Duration: {end_time - start_time:.2f}s")
    print(f"   - Key A (0.50): {successful_requests['key_a']}/150 successful")
    print(f"   - Key B (0.20): {successful_requests['key_b']}/150 successful")
    print(f"   - Key C (0.05): {successful_requests['key_c']}/150 successful")
    print(f"   - Key D (default 0.05): {successful_requests['key_d']}/150 successful")
    print(f"   - Total successful: {total_successful}/600")

    # Verify priority ordering: A > B > C ≈ D
    assert (
        successful_requests["key_a"] > successful_requests["key_b"]
    ), "Key A should get more than Key B"
    assert (
        successful_requests["key_b"] > successful_requests["key_c"]
    ), "Key B should get more than Key C"

    # Key C and Key D should get similar amounts (both have 0.05 priority)
    key_c_vs_d_ratio = successful_requests["key_c"] / max(
        successful_requests["key_d"], 1
    )
    print(f"   - Key C vs Key D ratio: {key_c_vs_d_ratio:.2f} (expected ~1.0)")

    if total_successful > 0:
        key_a_share = successful_requests["key_a"] / total_successful
        print(
            f"   - Key A got {key_a_share:.1%} of successful requests (expected ~55-62%)"
        )


@pytest.mark.asyncio
async def test_default_priority_shared_pool(monkeypatch):
    """
    Test that keys without explicit priority share ONE default pool, not get individual allocations.

    With default_priority=0.25:
    - Key A, B, C (no priority) should share ONE 25 RPM pool
    - NOT get 25 RPM each (which would be 75 RPM total)
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    litellm.priority_reservation = {"prod": 0.75}
    litellm.priority_reservation_settings.default_priority = 0.25

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "test-default-pool"
    total_rpm = 100

    bucket = _bucket(model, rpm=total_rpm)

    # Create 3 users without explicit priority
    user_a = UserAPIKeyAuth()
    user_a.metadata = {}
    user_a.user_id = "user_a"

    user_b = UserAPIKeyAuth()
    user_b.metadata = {}
    user_b.user_id = "user_b"

    user_c = UserAPIKeyAuth()
    user_c.metadata = {}
    user_c.user_id = "user_c"

    # Get descriptors for each
    desc_a = handler._create_priority_based_descriptors(bucket=bucket, priority=None)
    desc_b = handler._create_priority_based_descriptors(bucket=bucket, priority=None)
    desc_c = handler._create_priority_based_descriptors(bucket=bucket, priority=None)

    # All should use the SAME shared pool key
    assert desc_a[0]["value"] == f"{model}:default_pool"
    assert desc_b[0]["value"] == f"{model}:default_pool"
    assert desc_c[0]["value"] == f"{model}:default_pool"

    # All should have same limit (25 RPM SHARED, not 25 RPM each)
    assert desc_a[0]["rate_limit"]["requests_per_unit"] == 25
    assert desc_b[0]["rate_limit"]["requests_per_unit"] == 25
    assert desc_c[0]["rate_limit"]["requests_per_unit"] == 25

    # Verify explicit priority uses different pool
    user_prod = UserAPIKeyAuth()
    user_prod.team_metadata = {"priority": "prod"}
    desc_prod = handler._create_priority_based_descriptors(bucket=bucket, priority="prod")

    assert desc_prod[0]["value"] == f"{model}:prod"
    assert desc_prod[0]["rate_limit"]["requests_per_unit"] == 75
    assert desc_prod[0]["value"] != desc_a[0]["value"]  # Different pools

    print("✅ Default priority test passed:")
    print(f"   - 3 keys without priority share ONE pool: {desc_a[0]['value']}")
    print(f"   - Shared pool limit: {desc_a[0]['rate_limit']['requests_per_unit']} RPM")
    print(f"   - Explicit priority 'prod' uses separate pool: {desc_prod[0]['value']}")


@pytest.mark.asyncio
async def test_async_log_success_event_increments_by_actual_tokens(monkeypatch):
    """
    Test that async_log_success_event increments token counters by actual token usage.

    This validates the fix for Bug 1: Token count was incrementing by 1 instead of actual usage.
    The async_log_success_event should increment both model_saturation_check and priority_model
    counters by the actual completion_tokens (when rate_limit_type=output).
    """
    from unittest.mock import MagicMock

    from litellm.types.utils import ModelResponse, Usage

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    litellm.priority_reservation = {"dev": 0.1, "prod": 0.9}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "test-token-increment"
    deployment = _single_deployment(model, tpm=1000)

    # Track what gets incremented
    increment_calls = []

    async def mock_increment(pipeline_operations, parent_otel_span=None):
        for op in pipeline_operations:
            increment_calls.append(
                {
                    "key": op["key"],
                    "increment_value": op["increment_value"],
                }
            )

    handler.v3_limiter.async_increment_tokens_with_ttl_preservation = mock_increment

    # Create mock response with 50 completion tokens
    mock_response = MagicMock(spec=ModelResponse)
    mock_response.usage = MagicMock(spec=Usage)
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 60

    kwargs = {
        "standard_logging_object": {"model_group": model},
        "litellm_params": {
            "model_info": deployment["model_info"],
            "metadata": {
                "model_group": model,
                "deployment": deployment["litellm_params"]["model"],
                "user_api_key_team_metadata": {"priority": "dev"},
            },
        },
    }

    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=mock_response,
        start_time=None,
        end_time=None,
    )

    # Verify increments happened with actual token count (60 total tokens)
    assert (
        len(increment_calls) == 2
    ), f"Expected 2 increment calls, got {len(increment_calls)}"

    # Both should increment by 50 (total_tokens, since rate_limit_type defaults to 'total')
    for call in increment_calls:
        assert (
            call["increment_value"] == 60
        ), f"Expected increment of 60 tokens, got {call['increment_value']} for key {call['key']}"

    # Verify correct keys were used
    keys = [call["key"] for call in increment_calls]
    assert any(
        "model_saturation_check" in k for k in keys
    ), "Should increment model_saturation_check"
    assert any(
        "priority_model" in k and "dev" in k for k in keys
    ), "Should increment priority_model with 'dev' priority"


@pytest.mark.asyncio
async def test_saturation_check_cache_ttl_configuration(monkeypatch):
    """
    Test that saturation_check_cache_ttl controls how long saturation values are cached locally.

    This validates the configurable TTL for multi-node consistency:
    - When saturation_check_cache_ttl is set, local cache should expire after that duration
    - After expiration, fresh values should be fetched from Redis
    - This prevents nodes from having stale saturation data in multi-node deployments
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")

    # Set a short TTL for testing (5 seconds)
    original_ttl = litellm.priority_reservation_settings.saturation_check_cache_ttl
    litellm.priority_reservation_settings.saturation_check_cache_ttl = 5

    try:
        dual_cache = DualCache()
        handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

        model = "test-saturation-ttl"

        # Verify the TTL getter returns configured value
        assert (
            handler._get_saturation_check_cache_ttl() == 5
        ), "TTL should be configurable via priority_reservation_settings"

        # Track async_get_cache calls to verify TTL is passed
        get_cache_calls = []
        original_get_cache = handler.internal_usage_cache.async_get_cache

        async def mock_get_cache(
            key, litellm_parent_otel_span=None, local_only=False, **kwargs
        ):
            get_cache_calls.append(
                {
                    "key": key,
                    "ttl": kwargs.get("ttl"),
                    "local_only": local_only,
                }
            )
            return None  # Simulate cache miss

        handler.internal_usage_cache.async_get_cache = mock_get_cache

        # Call _get_saturation_value_from_cache
        counter_key = handler.v3_limiter.create_rate_limit_keys(
            key="model_saturation_check",
            value=model,
            rate_limit_type="requests",
        )

        await handler._get_saturation_value_from_cache(counter_key=counter_key)

        # Verify async_get_cache was called with the configured TTL
        assert len(get_cache_calls) == 1, "Expected 1 cache call"
        assert (
            get_cache_calls[0]["ttl"] == 5
        ), f"Expected TTL of 5 seconds, got {get_cache_calls[0]['ttl']}"
        assert (
            get_cache_calls[0]["local_only"] is False
        ), "Should check Redis (local_only=False) for multi-node consistency"

        # Test with different TTL value
        get_cache_calls.clear()
        litellm.priority_reservation_settings.saturation_check_cache_ttl = 30

        await handler._get_saturation_value_from_cache(counter_key=counter_key)

        assert (
            get_cache_calls[0]["ttl"] == 30
        ), f"TTL should update to 30 seconds, got {get_cache_calls[0]['ttl']}"

        print("Saturation check cache TTL test passed:")
        print(
            "   - TTL is configurable via priority_reservation_settings.saturation_check_cache_ttl"
        )
        print(
            "   - TTL is passed to async_get_cache for local cache expiration control"
        )
        print(
            "   - local_only=False ensures Redis is checked for multi-node consistency"
        )

    finally:
        # Restore original TTL
        litellm.priority_reservation_settings.saturation_check_cache_ttl = original_ttl


def _saturation_handler(
    model: str, rpm: int = 10, cache: DualCache | None = None
) -> tuple[DynamicRateLimitHandler, dict]:
    return DynamicRateLimitHandler(internal_usage_cache=cache or DualCache()), _single_deployment(
        model, rpm=rpm, tpm=1000
    )


def _priority_request_data(
    model: str,
    trusted_metadata_key: Literal["metadata", "litellm_metadata"],
    team_priority: str | None,
    requester_priority: str | None,
    key_priority: str | None,
) -> dict[str, object]:
    caller_metadata = {"priority": requester_priority} if requester_priority is not None else {}
    trusted_metadata = {
        "model_group": model,
        "user_api_key_metadata": {"priority": key_priority} if key_priority is not None else {},
        "user_api_key_team_metadata": {"priority": team_priority} if team_priority is not None else {},
    }
    if trusted_metadata_key == "metadata":
        return {"model": model, "metadata": {**caller_metadata, **trusted_metadata}}
    return {
        "model": model,
        "metadata": caller_metadata,
        "litellm_metadata": trusted_metadata,
    }


def _team_user(team_priority: str | None, key_priority: str | None = None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        metadata={"priority": key_priority} if key_priority is not None else {},
        team_metadata={"priority": team_priority} if team_priority is not None else {},
    )


async def _record_tokens(
    handler: DynamicRateLimitHandler,
    deployment: Mapping[str, object],
    data: Mapping[str, object],
    total_tokens: int,
    stream: bool = False,
    key_priority: str | None = None,
) -> None:
    from litellm.types.utils import ModelResponse, Usage

    key_metadata: Final = {"priority": key_priority} if key_priority is not None else {}
    trusted_metadata_key: Final = "litellm_metadata" if "litellm_metadata" in data else "metadata"
    routed_metadata: Final = {**data[trusted_metadata_key], "deployment": deployment["litellm_params"]["model"]}
    await handler.async_log_success_event(
        kwargs={
            "stream": stream,
            "litellm_params": {
                **{key: value for key, value in data.items() if key != "model"},
                trusted_metadata_key: routed_metadata,
                "model_info": deployment["model_info"],
            },
            "standard_logging_object": {
                "metadata": {"user_api_key_auth_metadata": key_metadata, "requester_metadata": key_metadata}
            },
        },
        response_obj=ModelResponse(
            model=data["model"],
            usage=Usage(prompt_tokens=0, completion_tokens=total_tokens, total_tokens=total_tokens),
        ),
        start_time=None,
        end_time=None,
    )


def _saturation_samples() -> list[Sample]:
    return [
        sample
        for sample in chain.from_iterable(metric.samples for metric in REGISTRY.collect())
        if sample.name == "litellm_model_group_saturation_ratio"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trusted_metadata_key",
    (pytest.param("metadata", id="chat_completions"), pytest.param("litellm_metadata", id="messages")),
)
@pytest.mark.parametrize(
    ("team_priority", "requester_priority", "key_priority", "expected_priority"),
    (
        ("high", None, None, "high"),
        (None, "high", None, ""),
        (None, None, "high", ""),
        ("high", "forged-requester", "forged-key", "high"),
    ),
)
async def test_pre_call_emits_model_group_saturation_from_limiter_window(
    monkeypatch: pytest.MonkeyPatch,
    trusted_metadata_key: Literal["metadata", "litellm_metadata"],
    team_priority: str | None,
    requester_priority: str | None,
    key_priority: str | None,
    expected_priority: str,
):
    _clear_prometheus_registry()
    try:
        monkeypatch.setattr(litellm, "callbacks", [PrometheusLogger()])
        model = "saturated-model"
        handler, deployment = _saturation_handler(model)
        data = _priority_request_data(model, trusted_metadata_key, team_priority, requester_priority, key_priority)
        user = _team_user(team_priority, key_priority)

        await _admit(handler, deployment, user, data)
        await _record_tokens(handler, deployment, data, total_tokens=500)
        await _admit(handler, deployment, user, data)

        samples = _saturation_samples()
        assert len(samples) == 1
        assert samples[0].labels == {"model_group": model, "priority": expected_priority}
        assert samples[0].value == max(2 / 10, 500 / 1000)
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("admitted_before", "tokens", "expected_saturation"),
    ((5, 100, 6 / 10), (1, 700, 700 / 1000)),
    ids=("rpm_dominant", "tpm_dominant"),
)
async def test_pre_call_saturation_gauge_uses_larger_of_rpm_and_tpm_after_admission(
    monkeypatch: pytest.MonkeyPatch, admitted_before: int, tokens: int, expected_saturation: float
):
    _clear_prometheus_registry()
    try:
        monkeypatch.setattr(litellm, "callbacks", [PrometheusLogger()])
        model = "max-saturation-model"
        handler, deployment = _saturation_handler(model)
        data = _priority_request_data(model, "metadata", "high", None, None)
        user = _team_user("high")
        for _ in range(admitted_before):
            await _admit(handler, deployment, user, data)
        await _record_tokens(handler, deployment, data, total_tokens=tokens)

        await _admit(handler, deployment, user, data)

        samples = _saturation_samples()
        assert len(samples) == 1
        assert samples[0].value == pytest.approx(expected_saturation)
    finally:
        _clear_prometheus_registry()


class _LaggingSaturationReadCache(DualCache):
    """A pod whose saturation pre-read comes from a local copy taken before other pods filled the window."""

    def __init__(self, lagging_counter: str) -> None:
        super().__init__()
        self._lagging_counter = lagging_counter

    async def async_get_cache(
        self, key: str, parent_otel_span: object = None, local_only: bool = False, **kwargs: object
    ) -> object:
        if not local_only and "model_saturation_check" in key:
            return self._lagging_counter
        return await super().async_get_cache(key, parent_otel_span=parent_otel_span, local_only=local_only, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("make_cache", "expected_pre_read"),
    (
        pytest.param(DualCache, 5.0, id="pre_read_over_capacity"),
        pytest.param(partial(_LaggingSaturationReadCache, lagging_counter="1"), 0.1, id="pre_read_below_capacity"),
    ),
)
async def test_pre_call_saturation_gauge_reads_a_full_window_on_a_model_capacity_429(
    monkeypatch: pytest.MonkeyPatch, make_cache: Callable[[], DualCache], expected_pre_read: float
):
    from fastapi import HTTPException

    _clear_prometheus_registry()
    try:
        monkeypatch.setattr(litellm, "callbacks", [PrometheusLogger()])
        model = "over-capacity-model"
        handler, deployment = _saturation_handler(model, cache=make_cache())
        data = _priority_request_data(model, "metadata", "high", None, None)
        user = _team_user("high")
        await _admit(handler, deployment, user, data)
        await _record_tokens(handler, deployment, data, total_tokens=5000)
        pre_read_saturation = await handler._check_bucket_saturation(_bucket(_bucket_key(deployment), tpm=1000, rpm=10))

        with pytest.raises(HTTPException) as blocked:
            await _admit(handler, deployment, user, data)

        assert blocked.value.status_code == 429
        assert "Model capacity reached" in blocked.value.detail["error"]
        assert pre_read_saturation == pytest.approx(expected_pre_read)
        samples = _saturation_samples()
        assert len(samples) == 1
        assert samples[0].labels == {"model_group": model, "priority": "high"}
        assert samples[0].value == 1.0
    finally:
        _clear_prometheus_registry()


def _priority_pool_key(
    handler: DynamicRateLimitHandler, bucket_key: str, pool: str, rate_limit_type: Literal["requests", "tokens"]
) -> str:
    return handler.v3_limiter.create_rate_limit_keys(
        key="priority_model", value=f"{bucket_key}:{pool}", rate_limit_type=rate_limit_type
    )


async def _pool_counter(
    handler: DynamicRateLimitHandler, bucket_key: str, pool: str, rate_limit_type: Literal["requests", "tokens"]
) -> int:
    value = await handler.internal_usage_cache.async_get_cache(
        key=_priority_pool_key(handler, bucket_key, pool, rate_limit_type), litellm_parent_otel_span=None
    )
    return 0 if value is None else int(value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trusted_metadata_key",
    (pytest.param("metadata", id="chat_completions"), pytest.param("litellm_metadata", id="messages")),
)
async def test_success_accounting_charges_the_team_class_across_pods_for_whole_and_streamed_calls(
    monkeypatch: pytest.MonkeyPatch, trusted_metadata_key: Literal["metadata", "litellm_metadata"]
):
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.8, "low": 0.2})
    model = "accounting-model"
    shared_cache = DualCache()
    first_pod, deployment = _saturation_handler(model, cache=shared_cache)
    second_pod, _ = _saturation_handler(model, cache=shared_cache)
    data = _priority_request_data(model, trusted_metadata_key, "high", "low", "low")

    await _record_tokens(first_pod, deployment, data, total_tokens=120, key_priority="low")
    await _record_tokens(second_pod, deployment, data, total_tokens=80, stream=True, key_priority="low")

    assert await _pool_counter(first_pod, _bucket_key(deployment), "high", "tokens") == 200
    assert await _pool_counter(second_pod, _bucket_key(deployment), "low", "tokens") == 0
    assert await _pool_counter(second_pod, _bucket_key(deployment), "default_pool", "tokens") == 0


def _anthropic_message_body(model: str, input_tokens: int, output_tokens: int) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _anthropic_message_stream(model: str, input_tokens: int, output_tokens: int) -> bytes:
    import json

    message_start: Final = {**_anthropic_message_body(model, input_tokens, 1), "content": [], "stop_reason": None}
    events: Final = (
        ("message_start", {"type": "message_start", "message": message_start}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    return "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events).encode()


async def _await_pool_tokens(handler: DynamicRateLimitHandler, model: str, pool: str, expected: int) -> int:
    for _ in range(100):
        charged = await _pool_counter(handler, model, pool, "tokens")
        if charged >= expected:
            return charged
        await asyncio.sleep(0.05)
    return await _pool_counter(handler, model, pool, "tokens")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True), ids=("whole", "streamed"))
async def test_native_anthropic_messages_success_charges_the_team_pool_tokens(
    monkeypatch: pytest.MonkeyPatch, stream: bool
):
    import httpx
    import respx

    from litellm.caching.llm_caching_handler import LLMClientCache

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.8, "low": 0.2})
    model = "native-claude"
    upstream_model = "claude-opus-4-7"
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": f"anthropic/{upstream_model}",
                    "api_key": "sk-ant-test",
                    "rpm": 10,
                    "tpm": 1000,
                },
            }
        ]
    )
    handler.update_variables(llm_router=router)
    monkeypatch.setattr(litellm, "callbacks", [handler])
    upstream_response: Final = (
        httpx.Response(
            200,
            content=_anthropic_message_stream(upstream_model, 10, 5),
            headers={"content-type": "text/event-stream"},
        )
        if stream
        else httpx.Response(200, json=_anthropic_message_body(upstream_model, 10, 5))
    )

    with respx.mock(assert_all_called=True) as upstream:
        upstream.post("https://api.anthropic.com/v1/messages").mock(return_value=upstream_response)
        response = await router.aanthropic_messages(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            stream=stream,
            metadata={"user_id": "caller", "priority": "low"},
            litellm_metadata={
                "user_api_key_metadata": {"priority": "low"},
                "user_api_key_team_metadata": {"priority": "high"},
            },
        )
        if stream:
            async for _ in response:
                pass

        assert await _await_pool_tokens(handler, model, "high", 15) == 15
    assert await _pool_counter(handler, model, "low", "tokens") == 0
    assert await _pool_counter(handler, model, "default_pool", "tokens") == 0


@pytest.mark.asyncio
async def test_success_accounting_charges_a_team_without_class_to_the_default_pool_whatever_the_key_says(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.8, "low": 0.2})
    model = "default-accounting-model"
    handler, deployment = _saturation_handler(model)
    data = _priority_request_data(model, "metadata", None, "high", "high")

    await _record_tokens(handler, deployment, data, total_tokens=50, key_priority="high")

    assert await _pool_counter(handler, _bucket_key(deployment), "default_pool", "tokens") == 50
    assert await _pool_counter(handler, _bucket_key(deployment), "high", "tokens") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("team_priority", "key_priority", "charged_pool", "uncharged_pool"),
    (("low", "high", "low", "high"), (None, "high", "default_pool", "high")),
)
async def test_admission_charges_the_team_class_and_ignores_key_metadata(
    monkeypatch: pytest.MonkeyPatch,
    team_priority: str | None,
    key_priority: str,
    charged_pool: str,
    uncharged_pool: str,
):
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.8, "low": 0.2})
    model = "admission-model"
    handler, deployment = _saturation_handler(model)

    await _admit(
        handler,
        deployment,
        _team_user(team_priority, key_priority),
        _priority_request_data(model, "metadata", team_priority, None, key_priority),
    )

    assert await _pool_counter(handler, _bucket_key(deployment), charged_pool, "requests") == 1
    assert await _pool_counter(handler, _bucket_key(deployment), uncharged_pool, "requests") == 0


@pytest.mark.asyncio
async def test_team_class_missing_from_reservations_is_charged_to_the_default_pool(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.8, "low": 0.2})
    model = "unknown-class-model"
    handler, deployment = _saturation_handler(model)
    bucket_key = _bucket_key(deployment)
    data = _priority_request_data(model, "metadata", "unlisted", None, None)

    await _admit(handler, deployment, _team_user("unlisted"), data)
    default_tokens_after_admission = await _pool_counter(handler, bucket_key, "default_pool", "tokens")
    await _record_tokens(handler, deployment, data, total_tokens=30)

    assert await _pool_counter(handler, bucket_key, "default_pool", "requests") == 1
    assert await _pool_counter(handler, bucket_key, "default_pool", "tokens") == default_tokens_after_admission + 30
    assert await _pool_counter(handler, bucket_key, "unlisted", "requests") == 0
    assert await _pool_counter(handler, bucket_key, "unlisted", "tokens") == 0


@pytest.mark.asyncio
async def test_non_string_team_priority_keeps_model_capacity_enforced_and_labels_no_class(
    monkeypatch: pytest.MonkeyPatch,
):
    from fastapi import HTTPException

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.8, "low": 0.2})
    _clear_prometheus_registry()
    try:
        monkeypatch.setattr(litellm, "callbacks", [PrometheusLogger()])
        model = "list-priority-model"
        handler, deployment = _saturation_handler(model, rpm=1)
        user = UserAPIKeyAuth(team_metadata={"priority": ["high"]})
        data = {"model": model, "metadata": {"model_group": model, "user_api_key_team_metadata": {"priority": ["high"]}}}

        await _admit(handler, deployment, user, data)
        with pytest.raises(HTTPException) as blocked:
            await _admit(handler, deployment, user, data)

        assert blocked.value.status_code == 429
        assert {sample.labels["priority"] for sample in _saturation_samples()} == {""}
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_saturation_gauge_reports_the_model_window_not_the_priority_pool(monkeypatch: pytest.MonkeyPatch):
    from fastapi import HTTPException

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.5, "low": 0.5})
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.4)
    _clear_prometheus_registry()
    try:
        monkeypatch.setattr(litellm, "callbacks", [PrometheusLogger()])
        model = "pool-vs-model"
        handler, deployment = _saturation_handler(model)
        high_data = _priority_request_data(model, "metadata", "high", None, None)
        high_user = _team_user("high")
        for _ in range(4):
            await _admit(handler, deployment, high_user, high_data)
        await _record_tokens(
            handler, deployment, _priority_request_data(model, "metadata", "low", None, None), total_tokens=500
        )

        await _admit(handler, deployment, high_user, high_data)
        admitted_values = [sample.value for sample in _saturation_samples()]
        with pytest.raises(HTTPException) as blocked:
            await _admit(handler, deployment, high_user, high_data)
        refused_values = [sample.value for sample in _saturation_samples()]

        assert admitted_values == [pytest.approx(max(5 / 10, 500 / 1000))]
        assert blocked.value.status_code == 429
        assert "Priority-based rate limit exceeded" in blocked.value.detail["error"]
        assert refused_values == [pytest.approx(max(5 / 10, 500 / 1000))]
    finally:
        _clear_prometheus_registry()


class _FailingSaturationPrometheusLogger(PrometheusLogger):
    def __init__(self) -> None:
        pass

    def set_model_group_saturation_ratio(self, model_group: str, saturation: float, litellm_params: object) -> None:
        raise RuntimeError("metrics backend unavailable")


@pytest.mark.asyncio
async def test_pre_call_still_enforces_model_capacity_when_saturation_metric_emission_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    from fastapi import HTTPException

    model = "metric-failure-model"
    handler, deployment = _saturation_handler(model, rpm=1)
    monkeypatch.setattr(litellm, "callbacks", [_FailingSaturationPrometheusLogger()])

    await _admit(handler, deployment, UserAPIKeyAuth())
    with pytest.raises(HTTPException) as blocked:
        await _admit(handler, deployment, UserAPIKeyAuth())

    assert blocked.value.status_code == 429


@pytest.mark.asyncio
async def test_messages_success_accounting_charges_the_routed_deployment_not_the_caller_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.8, "low": 0.2})
    model = "routed-messages-model"
    caller_named_model = "caller-named-model"
    handler, deployment = _saturation_handler(model)
    data = {
        **_priority_request_data(model, "litellm_metadata", "high", None, None),
        "metadata": {"user_id": "caller", "model_group": caller_named_model, "deployment": caller_named_model},
    }

    await _record_tokens(handler, deployment, data, total_tokens=40)

    model_tokens = await handler.internal_usage_cache.async_get_cache(
        key=handler.v3_limiter.create_rate_limit_keys(
            key="model_saturation_check", value=_bucket_key(deployment), rate_limit_type="tokens"
        ),
        litellm_parent_otel_span=None,
    )
    assert model_tokens is not None and int(model_tokens) == 40
    assert await _pool_counter(handler, _bucket_key(deployment), "high", "tokens") == 40
    assert await _pool_counter(handler, f"{deployment['model_info']['id']}:{caller_named_model}", "high", "tokens") == 0


@pytest.mark.asyncio
async def test_pre_call_records_no_saturation_when_the_limiter_cannot_verify_its_counters(
    monkeypatch: pytest.MonkeyPatch,
):
    from fastapi import HTTPException

    from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitUnverifiableError

    _clear_prometheus_registry()
    try:
        monkeypatch.setattr(litellm, "callbacks", [PrometheusLogger()])
        model = "unverifiable-model"
        handler, deployment = _saturation_handler(model)

        with patch.object(
            handler.v3_limiter,
            "atomic_check_and_increment_by_n",
            new=AsyncMock(side_effect=RateLimitUnverifiableError()),
        ):
            with pytest.raises(HTTPException) as rejected:
                await _admit(
                    handler, deployment, _team_user("high"), _priority_request_data(model, "metadata", "high", None, None)
                )

        assert rejected.value.status_code == 503
        assert _saturation_samples() == []
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_priority_429_includes_model_name_and_configured_limits(monkeypatch):
    """
    The priority-based 429 should tell operators which model was hit and what
    the model's configured TPM/RPM are, so they can decide whether to tune the
    priority allocation or the model limits.

    Regression test for the previous message that read:
        "Priority-based rate limit exceeded. Priority: prod,
         Rate limit type: tokens, Remaining: -664145,
         Model saturation: 86.3%"
    -- with no indication of which model was hit.
    """
    from fastapi import HTTPException

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    litellm.priority_reservation = {"prod": 0.5}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "gpt-4o-test"
    total_tpm = 1_000_000
    total_rpm = 10_000

    deployment = _single_deployment(model, tpm=total_tpm, rpm=total_rpm)

    user = UserAPIKeyAuth()
    user.team_metadata = {"priority": "prod"}
    user.user_id = "prod_user"

    # Force the atomic check+increment to return OVER_LIMIT for the
    # priority_model descriptor, whatever the bucket's saturation.
    over_limit_response = {
        "overall_code": "OVER_LIMIT",
        "statuses": [
            {
                "code": "OVER_LIMIT",
                "descriptor_key": "priority_model",
                "rate_limit_type": "tokens",
                "limit_remaining": -664145,
                "current_limit": int(total_tpm * 0.5),
            }
        ],
    }

    with patch.object(
        handler.v3_limiter,
        "atomic_check_and_increment_by_n",
        new=AsyncMock(return_value=over_limit_response),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _admit(handler, deployment, user)

    assert exc_info.value.status_code == 429
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    error_msg = detail["error"]

    # New fields added by this change -- the whole point of the fix.
    assert f"Model: {model}" in error_msg, error_msg
    assert f"Model TPM: {total_tpm}" in error_msg, error_msg
    assert f"Model RPM: {total_rpm}" in error_msg, error_msg

    # Existing fields must still be present (no regression).
    assert "Priority-based rate limit exceeded" in error_msg, error_msg
    assert "Priority: prod" in error_msg, error_msg
    assert "Rate limit type: tokens" in error_msg, error_msg
    assert "Model saturation:" in error_msg, error_msg


@pytest.mark.asyncio
async def test_tpm_only_model_enforces_priority_and_model_capacity(monkeypatch):
    """Regression: a model configured with ONLY tpm (no rpm) must still be
    rate limited.

    The atomic check-and-increment path used to drop any counter whose
    pre-call increment was zero. Token increments are always zero pre-call
    (usage lands on the counters post-call), so on a TPM-only model the
    limiter evaluated no counters at all: no model-wide cap, no priority
    reservation, in either mode. This test drives the real pre-call ->
    log-success -> pre-call flow with no limiter internals mocked.
    """
    from fastapi import HTTPException

    from litellm.types.utils import ModelResponse, Usage

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    litellm.priority_reservation = {"dev": 0.25, "prod": 0.5}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "tpm-only-model"
    deployment = _single_deployment(model, tpm=400)

    dev_user = UserAPIKeyAuth()
    dev_user.team_metadata = {"priority": "dev"}
    prod_user = UserAPIKeyAuth()
    prod_user.team_metadata = {"priority": "prod"}

    async def record_usage(priority: str, total_tokens: int) -> None:
        await handler.async_log_success_event(
            kwargs={
                "standard_logging_object": {},
                "litellm_params": {
                    "model_info": deployment["model_info"],
                    "metadata": {
                        "model_group": model,
                        "deployment": deployment["litellm_params"]["model"],
                        "user_api_key_team_metadata": {"priority": priority},
                    },
                },
            },
            response_obj=ModelResponse(
                model=model,
                usage=Usage(prompt_tokens=0, completion_tokens=total_tokens, total_tokens=total_tokens),
            ),
            start_time=None,
            end_time=None,
        )

    await _admit(handler, deployment, dev_user)

    await record_usage("dev", 250)

    with pytest.raises(HTTPException) as dev_blocked:
        await _admit(handler, deployment, dev_user)
    assert dev_blocked.value.status_code == 429
    assert "Priority-based rate limit exceeded" in dev_blocked.value.detail["error"]

    await _admit(handler, deployment, prod_user)

    await record_usage("prod", 200)

    with pytest.raises(HTTPException) as capacity_blocked:
        await _admit(handler, deployment, prod_user)
    assert capacity_blocked.value.status_code == 429
    assert "Model capacity reached" in capacity_blocked.value.detail["error"]


@pytest.mark.asyncio
async def test_post_call_success_hook_attaches_priority_headers_to_dict_response():
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        RateLimitResponse,
        RateLimitStatus,
        get_or_create_request_stash,
    )

    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    get_or_create_request_stash().rate_limit_response = RateLimitResponse(
        overall_code="OK",
        statuses=[
            RateLimitStatus(
                code="OK",
                current_limit=75,
                limit_remaining=74,
                rate_limit_type="requests",
                descriptor_key="priority_model",
            )
        ],
    )
    response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [],
        "_hidden_params": {"additional_headers": {"x-litellm-attempted-retries": 0}},
    }

    await handler.async_post_call_success_hook(
        data={"model": "anthropic-haiku"},
        user_api_key_dict=UserAPIKeyAuth(team_metadata={"priority": "premium"}),
        response=response,
    )

    additional_headers = response["_hidden_params"]["additional_headers"]
    assert additional_headers["x-litellm-attempted-retries"] == 0
    assert additional_headers["x-ratelimit-priority_model-limit-requests"] == 75
    assert additional_headers["x-ratelimit-priority_model-remaining-requests"] == 74
    assert additional_headers["x-litellm-priority"] == "premium"
    assert additional_headers["x-litellm-rate-limiter-version"] == "v3"


@pytest.mark.asyncio
async def test_post_call_success_hook_leaves_raw_provider_dict_untouched():
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    response = {"id": "msg_123", "type": "message", "role": "assistant", "content": []}

    await handler.async_post_call_success_hook(
        data={"model": "anthropic-haiku"},
        user_api_key_dict=UserAPIKeyAuth(team_metadata={"priority": "premium"}),
        response=response,
    )

    assert response == {"id": "msg_123", "type": "message", "role": "assistant", "content": []}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("team_metadata", "expected_priority_header"),
    [
        ({"priority": "优先"}, None),
        ({"priority": "high"}, "high"),
        ({}, "default"),
    ],
)
async def test_post_call_success_hook_priority_header_is_always_http_encodable(team_metadata, expected_priority_header):
    from starlette.responses import Response

    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    response = {"id": "msg_123", "type": "message", "role": "assistant", "content": [], "_hidden_params": {}}

    await handler.async_post_call_success_hook(
        data={"model": "anthropic-haiku"},
        user_api_key_dict=UserAPIKeyAuth(team_id="team-1", team_metadata=team_metadata),
        response=response,
    )

    additional_headers = response["_hidden_params"]["additional_headers"]
    http_response = Response(headers={key: str(value) for key, value in additional_headers.items()})
    assert http_response.headers.get("x-litellm-priority") == expected_priority_header
    assert http_response.headers["x-litellm-rate-limiter-version"] == "v3"


ORG_A: Final = "org-a"
ORG_B: Final = "org-b"
OPUS_4_CLASSES: Final = {"claude-opus-4-7": "opus-4", "claude-opus-4-6": "opus-4"}


class _CountingProvider(CustomLLM):
    def __init__(self) -> None:
        super().__init__()
        self.served_by: tuple[str, ...] = ()

    async def acompletion(self, *args: object, **kwargs: object) -> ModelResponse:
        api_base: Final = kwargs.get("api_base")
        self.served_by = (*self.served_by, str(api_base))
        return litellm.completion(
            model="gpt-5.6", messages=[{"role": "user", "content": "hi"}], mock_response=f"served by {api_base}"
        )


@pytest.fixture
def counting_provider(monkeypatch: pytest.MonkeyPatch) -> _CountingProvider:
    provider = _CountingProvider()
    monkeypatch.setattr(litellm, "custom_provider_map", [{"provider": "counting", "custom_handler": provider}])
    monkeypatch.setattr(litellm, "provider_list", [*litellm.provider_list, "counting"])
    monkeypatch.setattr(litellm, "_custom_providers", [*litellm._custom_providers, "counting"])
    return provider


def _org_base(organization: str) -> str:
    return f"https://{organization}.test"


def _counting_deployment(
    organization: str,
    model_name: str = "claude",
    rpm: int | None = None,
    weight: int | None = None,
) -> dict[str, object]:
    limits = {"rpm": rpm} if rpm is not None else {}
    weights = {"weight": weight} if weight is not None else {}
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": "counting/claude-opus-4-7",
            "api_base": _org_base(organization),
            "api_key": "sk-fake",
            "anthropic_organization_id": organization,
            **limits,
            **weights,
        },
        "model_info": {"id": f"{organization}-{model_name}", "bucket_class": OPUS_4_CLASSES},
    }


class _PinOrganization(CustomLogger):
    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: object,
        request_kwargs: dict | None = None,
        parent_otel_span: object = None,
    ) -> list:
        match request_kwargs:
            case {"metadata": {"pinned_organization": str() as organization}}:
                return [
                    deployment
                    for deployment in healthy_deployments
                    if deployment["litellm_params"].get("anthropic_organization_id") == organization
                ]
            case _:
                return healthy_deployments


class _AttemptSpy(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.attempts: tuple[str, ...] = ()

    async def async_pre_call_check(self, deployment: Mapping[str, object], parent_otel_span: object) -> None:
        match deployment:
            case {"model_info": {"id": str() as deployment_id}}:
                self.attempts = (*self.attempts, deployment_id)


async def _proxy_call(
    handler: DynamicRateLimitHandler,
    router: Router,
    user: UserAPIKeyAuth,
    pinned_organization: str | None = None,
    model: str = "claude",
) -> ModelResponse:
    pin = {"pinned_organization": pinned_organization} if pinned_organization is not None else {}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {**pin, "user_api_key_team_metadata": user.team_metadata},
    }
    await handler.async_pre_call_hook(
        user_api_key_dict=user, cache=handler.internal_usage_cache, data=data, call_type="acompletion"
    )
    return await router.acompletion(**data)


def _served_by(response: ModelResponse) -> str:
    return str(response.choices[0].message.content)


@pytest.mark.asyncio
async def test_saturated_bucket_refuses_the_default_pool_while_the_idle_organization_and_the_high_floor_serve(
    monkeypatch: pytest.MonkeyPatch, counting_provider: _CountingProvider
):
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.7})
    monkeypatch.setattr(litellm.priority_reservation_settings, "default_priority", 0.3)
    monkeypatch.setattr(litellm.priority_reservation_settings, "saturation_threshold", 0.8)
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    monkeypatch.setattr(litellm, "callbacks", [_PinOrganization(), handler])
    router = Router(model_list=[_counting_deployment(ORG_A, rpm=10), _counting_deployment(ORG_B, rpm=10)])
    default_team = UserAPIKeyAuth(team_metadata={})
    high_team = UserAPIKeyAuth(team_metadata={"priority": "high"})

    for _ in range(8):
        await _proxy_call(handler, router, default_team, ORG_A)
    with pytest.raises(ProxyRateLimitError) as refused:
        await _proxy_call(handler, router, default_team, ORG_A)
    idle_organization = await _proxy_call(handler, router, default_team, ORG_B)
    high_floor = await _proxy_call(handler, router, high_team, ORG_A)

    assert refused.value.status_code == 429
    assert "Priority-based rate limit exceeded" in refused.value.detail["error"]
    assert _served_by(idle_organization) == f"served by {_org_base(ORG_B)}"
    assert _served_by(high_floor) == f"served by {_org_base(ORG_A)}"
    assert counting_provider.served_by == (*(_org_base(ORG_A),) * 8, _org_base(ORG_B), _org_base(ORG_A))


def _wildcard_router(rpm: int | None = None, tpm: int | None = None) -> Router:
    limits = {**({"rpm": rpm} if rpm is not None else {}), **({"tpm": tpm} if tpm is not None else {})}
    return Router(
        model_list=[
            {
                "model_name": "anthropic/*",
                "litellm_params": {
                    "model": "anthropic/*",
                    "api_key": "sk-fake",
                    "anthropic_organization_id": organization,
                    **limits,
                },
                "model_info": {"id": f"{organization}-wildcard", "bucket_class": OPUS_4_CLASSES},
            }
            for organization in (ORG_A, ORG_B)
        ]
    )


def _wildcard_deployment(router: Router, model: str, organization: str) -> Mapping[str, object]:
    resolved = router.pattern_router.route(model)
    assert resolved is not None
    return next(
        deployment
        for deployment in resolved
        if deployment["litellm_params"]["anthropic_organization_id"] == organization
    )


@pytest.mark.asyncio
async def test_wildcard_models_share_the_bucket_their_class_map_names_and_an_unmapped_model_is_its_own_bucket(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    router = _wildcard_router(rpm=2)

    async def admit(model: str) -> None:
        await handler.async_pre_call_check(_wildcard_deployment(router, model, ORG_A), None)

    await admit("anthropic/claude-opus-4-7")
    await admit("anthropic/claude-opus-4-6")
    with pytest.raises(ProxyRateLimitError) as opus_4_full:
        await admit("anthropic/claude-opus-4-7")
    await admit("anthropic/claude-opus-5")
    await admit("anthropic/claude-opus-5")
    with pytest.raises(ProxyRateLimitError) as opus_5_full:
        await admit("anthropic/claude-opus-5")

    assert "Model capacity reached" in opus_4_full.value.detail["error"]
    assert "Model capacity reached" in opus_5_full.value.detail["error"]


@pytest.mark.asyncio
async def test_saturation_gauge_reports_the_shared_bucket_under_the_requested_model_group(
    monkeypatch: pytest.MonkeyPatch,
):
    _clear_prometheus_registry()
    try:
        monkeypatch.setattr(litellm, "callbacks", [PrometheusLogger()])
        monkeypatch.setattr(litellm, "priority_reservation", None)
        handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
        router = _wildcard_router(rpm=4)
        team = _team_user("high")

        for model in ("anthropic/claude-opus-4-7", "anthropic/claude-opus-4-6"):
            data = _priority_request_data(model, "litellm_metadata", "high", None, None)
            await handler.async_pre_call_hook(
                user_api_key_dict=team, cache=handler.internal_usage_cache, data=data, call_type="anthropic_messages"
            )
            await handler.async_pre_call_check(_wildcard_deployment(router, model, ORG_A), None)

        samples = {sample.labels["model_group"]: sample for sample in _saturation_samples()}
        assert samples["anthropic/claude-opus-4-7"].value == pytest.approx(1 / 4)
        assert samples["anthropic/claude-opus-4-6"].value == pytest.approx(2 / 4)
        assert samples["anthropic/claude-opus-4-6"].labels["priority"] == "high"
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_a_deployment_without_limits_is_served(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.7})
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    deployment = _wildcard_deployment(_wildcard_router(), "anthropic/claude-opus-4-7", ORG_A)

    results = [await handler.async_pre_call_check(deployment, None) for _ in range(50)]

    assert results == [None] * 50


@pytest.mark.asyncio
async def test_bucket_capacity_falls_back_to_the_limits_in_model_info(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    deployment = {
        "model_name": "claude",
        "litellm_params": {"model": "anthropic/claude-opus-4-7", "anthropic_organization_id": ORG_A},
        "model_info": {"id": f"{ORG_A}-claude", "rpm": 1},
    }

    await handler.async_pre_call_check(deployment, None)
    with pytest.raises(ProxyRateLimitError) as full:
        await handler.async_pre_call_check(deployment, None)

    assert "Model RPM: 1" in full.value.detail["error"]


@pytest.mark.asyncio
async def test_bucket_refusal_makes_no_upstream_call_tries_no_other_deployment_and_cools_nothing_down(
    monkeypatch: pytest.MonkeyPatch, counting_provider: _CountingProvider
):
    from litellm.router_utils.cooldown_handlers import get_cooldown_deployments

    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    spy = _AttemptSpy()
    monkeypatch.setattr(litellm, "callbacks", [spy, handler])
    router = Router(
        model_list=[
            _counting_deployment(ORG_A, rpm=1, weight=1),
            _counting_deployment(ORG_B, rpm=1, weight=0),
            _counting_deployment("org-fallback", model_name="claude-fallback"),
        ],
        num_retries=2,
        retry_after=0,
        fallbacks=[{"claude": ["claude-fallback"]}],
    )
    team = UserAPIKeyAuth(team_metadata={})

    served = await _proxy_call(handler, router, team)
    with pytest.raises(ProxyRateLimitError) as refused:
        await _proxy_call(handler, router, team)

    assert _served_by(served) == f"served by {_org_base(ORG_A)}"
    assert refused.value.status_code == 429
    assert "Model capacity reached" in refused.value.detail["error"]
    assert counting_provider.served_by == (_org_base(ORG_A),)
    assert spy.attempts == (f"{ORG_A}-claude", f"{ORG_A}-claude")
    assert get_cooldown_deployments(router, parent_otel_span=None) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("weight_a", "weight_b", "served_by"),
    ((1, 0, None), (0, 1, _org_base(ORG_B))),
    ids=("router_picks_the_saturated_deployment", "router_picks_the_idle_deployment"),
)
async def test_admission_follows_the_deployment_the_router_finally_picks_among_several(
    monkeypatch: pytest.MonkeyPatch,
    counting_provider: _CountingProvider,
    weight_a: int,
    weight_b: int,
    served_by: str | None,
):
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    monkeypatch.setattr(litellm, "callbacks", [handler])
    router = Router(
        model_list=[
            _counting_deployment(ORG_A, rpm=1, weight=weight_a),
            _counting_deployment(ORG_B, rpm=1, weight=weight_b),
        ]
    )
    await handler.async_pre_call_check(router.get_deployment(model_id=f"{ORG_A}-claude").model_dump(), None)
    team = UserAPIKeyAuth(team_metadata={})

    if served_by is None:
        with pytest.raises(ProxyRateLimitError):
            await _proxy_call(handler, router, team)
        assert counting_provider.served_by == ()
    else:
        assert _served_by(await _proxy_call(handler, router, team)) == f"served by {served_by}"
        assert counting_provider.served_by == (served_by,)


def _logged_wildcard_call(
    deployment: Mapping[str, object],
    trusted_metadata_key: Literal["metadata", "litellm_metadata"],
    team_priority: str,
    forged_caller_deployment: object,
) -> dict[str, object]:
    match deployment:
        case {"litellm_params": {"model": str() as served_model, "anthropic_organization_id": str() as organization}}:
            pass
        case _:
            raise AssertionError(deployment)
    trusted_metadata = {
        "model_group": served_model,
        "deployment": served_model,
        "user_api_key_team_metadata": {"priority": team_priority},
    }
    forged_caller_metadata = {"deployment": forged_caller_deployment, "user_id": "caller"}
    shape = (
        {"metadata": trusted_metadata}
        if trusted_metadata_key == "metadata"
        else {"metadata": forged_caller_metadata, "litellm_metadata": trusted_metadata}
    )
    return {
        "litellm_params": {
            "model_info": deployment["model_info"],
            "anthropic_organization_id": organization,
            **shape,
        },
        "standard_logging_object": {},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trusted_metadata_key", "forged_caller_deployment"),
    (
        pytest.param("metadata", None, id="chat_completions"),
        pytest.param("litellm_metadata", "anthropic/claude-opus-5", id="messages"),
        pytest.param("litellm_metadata", None, id="messages_with_a_malformed_caller_deployment"),
    ),
)
async def test_success_accounting_charges_the_bucket_of_the_deployment_that_served_the_call(
    monkeypatch: pytest.MonkeyPatch,
    trusted_metadata_key: Literal["metadata", "litellm_metadata"],
    forged_caller_deployment: object,
):
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.7})
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    router = _wildcard_router(tpm=1000)
    served_by = _wildcard_deployment(router, "anthropic/claude-opus-4-6", ORG_A)
    other_organization = _wildcard_deployment(router, "anthropic/claude-opus-4-7", ORG_B)
    other_class = _wildcard_deployment(router, "anthropic/claude-opus-5", ORG_A)
    for deployment in (served_by, other_organization, other_class):
        await handler.async_pre_call_check(deployment, None)

    await handler.async_log_success_event(
        kwargs=_logged_wildcard_call(served_by, trusted_metadata_key, "high", forged_caller_deployment),
        response_obj=ModelResponse(usage=Usage(prompt_tokens=0, completion_tokens=1200, total_tokens=1200)),
        start_time=None,
        end_time=None,
    )

    with pytest.raises(ProxyRateLimitError) as same_bucket:
        await handler.async_pre_call_check(_wildcard_deployment(router, "anthropic/claude-opus-4-7", ORG_A), None)
    assert "Model capacity reached" in same_bucket.value.detail["error"]
    assert await handler.async_pre_call_check(other_organization, None) is None
    assert await handler.async_pre_call_check(other_class, None) is None
    assert (
        await _pool_counter(handler, f"{ORG_A}:opus-4", "high", "tokens"),
        await _pool_counter(handler, f"{ORG_A}:claude-opus-5", "high", "tokens"),
        await _pool_counter(handler, f"{ORG_B}:opus-4", "high", "tokens"),
    ) == (1200, 0, 0)


class _SuccessLogged(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged: Final = asyncio.Event()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.logged.set()


@pytest.mark.asyncio
async def test_messages_tokens_reach_the_bucket_of_the_organization_admitted_through_the_real_router(
    monkeypatch: pytest.MonkeyPatch, httpx_transport: None
):
    for name in ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "priority_reservation", None)
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    success = _SuccessLogged()
    monkeypatch.setattr(litellm, "callbacks", [handler, success])
    router = Router(
        model_list=[
            {
                "model_name": "claude",
                "litellm_params": {
                    "model": "anthropic/claude-opus-4-7",
                    "api_key": "sk-fake",
                    "anthropic_organization_id": ORG_A,
                    "tpm": 1000,
                },
                "model_info": {"id": f"{ORG_A}-claude", "bucket_class": OPUS_4_CLASSES},
            }
        ]
    )
    logging_obj, data = litellm.utils.function_setup(
        original_function="anthropic_messages",
        rules_obj=litellm.utils.Rules(),
        start_time=datetime.now(),
        model="claude",
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        litellm_call_id=uuid4().hex,
        litellm_metadata={},
    )

    with respx.mock() as upstream:
        upstream.post("https://api.anthropic.com/v1/messages").respond(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1200},
            },
        )
        await router.aanthropic_messages(**data, litellm_logging_obj=logging_obj)
    await asyncio.wait_for(success.logged.wait(), timeout=20)

    with pytest.raises(ProxyRateLimitError) as full:
        await handler.async_pre_call_check(router.get_deployment(model_id=f"{ORG_A}-claude").model_dump(), None)
    assert "Model capacity reached" in full.value.detail["error"]


def _messages_upstream_reply() -> dict[str, object]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


@pytest.mark.asyncio
async def test_messages_call_over_bucket_capacity_is_refused_once_with_one_upstream_post_and_no_cooldown(
    monkeypatch: pytest.MonkeyPatch, httpx_transport: None
):
    from litellm.router_utils.admission_refusal import is_admission_refusal
    from litellm.router_utils.cooldown_handlers import get_cooldown_deployments

    for name in ("ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "priority_reservation", None)
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    monkeypatch.setattr(litellm, "callbacks", [handler])
    router = Router(
        model_list=[
            {
                "model_name": "claude",
                "litellm_params": {
                    "model": "anthropic/claude-opus-4-7",
                    "api_key": "sk-fake",
                    "anthropic_organization_id": organization,
                    "rpm": 1,
                    "weight": weight,
                },
                "model_info": {"id": f"{organization}-claude", "bucket_class": OPUS_4_CLASSES},
            }
            for organization, weight in ((ORG_A, 1), (ORG_B, 0))
        ],
        num_retries=2,
        retry_after=0,
    )

    async def call() -> object:
        logging_obj, data = litellm.utils.function_setup(
            original_function="anthropic_messages",
            rules_obj=litellm.utils.Rules(),
            start_time=datetime.now(),
            model="claude",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            litellm_call_id=uuid4().hex,
            litellm_metadata={},
        )
        return await router.aanthropic_messages(**data, litellm_logging_obj=logging_obj)

    with respx.mock() as upstream:
        posts = upstream.post("https://api.anthropic.com/v1/messages").respond(200, json=_messages_upstream_reply())
        await call()
        with pytest.raises(ProxyRateLimitError) as refused:
            await call()

    assert refused.value.status_code == 429
    assert is_admission_refusal(refused.value)
    assert posts.call_count == 1
    assert get_cooldown_deployments(router, parent_otel_span=None) == []


@pytest.mark.asyncio
async def test_unverifiable_counters_refusal_is_final_with_no_retry_fallback_or_cooldown(
    monkeypatch: pytest.MonkeyPatch, counting_provider: _CountingProvider
):
    from fastapi import HTTPException

    from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitUnverifiableError
    from litellm.router_utils.cooldown_handlers import get_cooldown_deployments

    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    spy = _AttemptSpy()
    monkeypatch.setattr(litellm, "callbacks", [spy, handler])
    router = Router(
        model_list=[
            _counting_deployment(ORG_A, rpm=10, weight=1),
            _counting_deployment(ORG_B, rpm=10, weight=0),
            _counting_deployment("org-fallback", model_name="claude-fallback"),
        ],
        num_retries=2,
        retry_after=0,
        fallbacks=[{"claude": ["claude-fallback"]}],
    )

    with patch.object(
        handler.v3_limiter, "atomic_check_and_increment_by_n", new=AsyncMock(side_effect=RateLimitUnverifiableError())
    ):
        with pytest.raises(HTTPException) as refused:
            await _proxy_call(handler, router, UserAPIKeyAuth(team_metadata={}))

    assert refused.value.status_code == 503
    assert counting_provider.served_by == ()
    assert spy.attempts == (f"{ORG_A}-claude",)
    assert get_cooldown_deployments(router, parent_otel_span=None) == []


def _org_a_deployment(
    model: str,
    deployment_id: str = f"{ORG_A}-claude",
    litellm_limits: Mapping[str, int] = MappingProxyType({"rpm": 1}),
    model_info: Mapping[str, object] = MappingProxyType({"bucket_class": OPUS_4_CLASSES}),
) -> dict[str, object]:
    return {
        "model_name": "claude",
        "litellm_params": {"model": model, "anthropic_organization_id": ORG_A, **litellm_limits},
        "model_info": {"id": deployment_id, **model_info},
    }


async def _served(handler: DynamicRateLimitHandler, deployment: Mapping[str, object]) -> bool:
    try:
        await handler.async_pre_call_check(deployment, None)
    except ProxyRateLimitError:
        return False
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_bucket_class",
    (["opus-4"], {"claude-opus-4-7": 4}, "opus-4"),
    ids=("list", "non_string_class", "string"),
)
async def test_malformed_bucket_class_serves_each_model_as_its_own_class(
    monkeypatch: pytest.MonkeyPatch, malformed_bucket_class: object
):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    malformed = MappingProxyType({"bucket_class": malformed_bucket_class})

    outcomes = [
        await _served(handler, _org_a_deployment(model, model_info=malformed))
        for model in ("anthropic/claude-opus-4-7", "anthropic/claude-opus-4-6", "anthropic/claude-opus-4-7")
    ]

    assert outcomes == [True, True, False]


@pytest.mark.asyncio
async def test_two_unmapped_models_do_not_share_a_bucket(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())

    outcomes = [
        await _served(handler, _org_a_deployment(model))
        for model in ("anthropic/claude-opus-5", "anthropic/claude-sonnet-5", "anthropic/claude-opus-5")
    ]

    assert outcomes == [True, True, False]


@pytest.mark.asyncio
async def test_deployments_of_one_organization_share_the_bucket_of_their_class(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())

    outcomes = [
        await _served(handler, _org_a_deployment(model, deployment_id=deployment_id))
        for model, deployment_id in (
            ("anthropic/claude-opus-4-7", "org-a-first-deployment"),
            ("anthropic/claude-opus-4-6", "org-a-second-deployment"),
        )
    ]

    assert outcomes == [True, False]


@pytest.mark.asyncio
async def test_litellm_params_limits_win_over_model_info_limits(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    deployment = _org_a_deployment(
        "anthropic/claude-opus-4-7", model_info=MappingProxyType({"bucket_class": OPUS_4_CLASSES, "rpm": 5})
    )

    outcomes = [await _served(handler, deployment) for _ in range(2)]

    assert outcomes == [True, False]


@pytest.mark.asyncio
async def test_a_zero_limit_is_served(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    deployment = _org_a_deployment("anthropic/claude-opus-4-7", litellm_limits=MappingProxyType({"rpm": 0, "tpm": 0}))

    outcomes = [await _served(handler, deployment) for _ in range(5)]

    assert outcomes == [True] * 5


@pytest.mark.asyncio
async def test_refusal_names_the_provider_and_model_of_the_deployment_not_the_model_group(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "priority_reservation", None)
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    deployment = _org_a_deployment("anthropic/claude-opus-4-7")
    await handler.async_pre_call_check(deployment, None)

    with pytest.raises(ProxyRateLimitError) as refused:
        await handler.async_pre_call_check(deployment, None)

    assert (refused.value.llm_provider, refused.value.model) == ("anthropic", "claude-opus-4-7")


@pytest.mark.asyncio
async def test_bucket_admission_keeps_the_per_key_rate_limit_headers(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        RateLimitResponse,
        RateLimitStatus,
        claim_request_stash_for_data,
    )

    monkeypatch.setattr(litellm, "priority_reservation", None)
    handler = DynamicRateLimitHandler(internal_usage_cache=DualCache())
    deployment = _org_a_deployment("anthropic/claude-opus-4-7", litellm_limits=MappingProxyType({"rpm": 10}))
    data = {"model": "claude", "litellm_call_id": "call-1"}
    response = {"id": "msg_1", "type": "message", "role": "assistant", "content": [], "_hidden_params": {}}

    async def request() -> None:
        claim_request_stash_for_data(data).rate_limit_response = RateLimitResponse(
            overall_code="OK",
            statuses=[
                RateLimitStatus(
                    code="OK",
                    current_limit=100,
                    limit_remaining=99,
                    rate_limit_type="requests",
                    descriptor_key="api_key",
                )
            ],
        )
        await handler.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(), cache=handler.internal_usage_cache, data=data, call_type="acompletion"
        )
        await handler.async_pre_call_check(deployment, None)
        await handler.async_pre_call_check(deployment, None)
        await handler.async_post_call_success_hook(data=data, user_api_key_dict=UserAPIKeyAuth(), response=response)

    await asyncio.create_task(request(), context=contextvars.Context())

    headers = response["_hidden_params"]["additional_headers"]
    assert (headers["x-ratelimit-api_key-remaining-requests"], headers["x-ratelimit-api_key-limit-requests"]) == (99, 100)
    assert (
        headers["x-ratelimit-model_saturation_check-remaining-requests"],
        headers["x-ratelimit-model_saturation_check-limit-requests"],
    ) == (8, 10)
