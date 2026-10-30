# Dynamic Rate Limiter v3 - Saturation-Aware Priority-Based Rate Limiting

## Overview

The v3 dynamic rate limiter implements saturation-aware rate limiting with priority-based allocation. It balances resource efficiency (allowing unused capacity to be borrowed) with fairness guarantees (enforcing priorities during high load).

Saturation and the priority pools are kept per provider bucket: the organization and the model class of the deployment that serves a call, not the LiteLLM model group. The proxy's pre-call hook only records the calling team's class. The router admits the call in `async_pre_call_check`, once it holds the deployment it picked and before the provider call

**Key Behavior:**
- When the bucket is under `saturation_threshold`: Generous mode - allows priority borrowing
- When the bucket is at/above `saturation_threshold`: Strict mode - enforces normalized priority limits

## How It Works

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Incoming Request                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Check Bucket Saturation                                  │
│     - Query v3 limiter's Redis counters for the bucket       │
│     - Calculate: current_usage / capacity                    │
│     - Returns: 0.0 (empty) to 1.0+ (saturated)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                ┌────────┴────────┐
                │  Saturation?    │
                └────────┬────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
   < threshold (Generous)          >= threshold (Strict)
         │                               │
         ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│  Generous Mode      │         │  Strict Mode        │
│                     │         │                     │
│  - Enforce bucket-  │         │  - Normalize        │
│    wide capacity    │         │    priority weights │
│  - No priority      │         │    (if over 1.0)    │
│    restrictions     │         │                     │
│  - Allows borrowing │         │  - Create priority- │
│                     │         │    specific         │
│  - First-come-      │         │    descriptors      │
│    first-served     │         │                     │
│    until capacity   │         │  - Enforce strict   │
│                     │         │    limits per       │
│                     │         │    priority         │
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           │                               ▼
           │                    ┌──────────────────────┐
           │                    │  Track model usage   │
           │                    │  for future          │
           │                    │  saturation checks   │
           │                    └──────────┬───────────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  v3 Limiter  │
                    │  Check       │
                    └──────┬───────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
     OVER_LIMIT                        OK
           │                               │
           ▼                               ▼
   Return 429 Error              Allow Request
```

## Configuration

### Priority Reservation

Set priority weights in your proxy configuration:

```python
litellm.priority_reservation = {
    "premium": 0.75,    # 75% of capacity
    "standard": 0.25    # 25% of capacity
}
```

### Priority Reservation Settings

Configure saturation-aware behavior:

```python
litellm.priority_reservation_settings = PriorityReservationSettings(
    default_priority=0.25,          # Weight of the pool shared by teams without a reserved priority
    saturation_threshold=0.50,      # 50% - threshold for strict mode enforcement
    saturation_check_cache_ttl=60,  # Seconds a pod caches a saturation read
)
```

**Settings:**
- `default_priority` (default: 0.25) - Priority weight for teams without a priority
- `saturation_threshold` (default: 0.50) - Saturation level (0.0-1.0) at which strict priority enforcement begins
- `saturation_check_cache_ttl` (default: 60) - Seconds a saturation value read from Redis is cached locally

### Team Priority Assignment

A request's priority is its team's `metadata.priority`, set with `PATCH /team/{team_id}` and the body `{"metadata": {"priority": "premium"}}`. A key's own metadata and the request body never set it. A request from a team with no priority, or with a priority missing from `priority_reservation`, shares the default pool.

### Provider Buckets

A bucket is one organization's capacity for one model class. The organization is the deployment's `litellm_params.anthropic_organization_id`, or its `model_info.id` when that is not set. The class comes from `model_info.bucket_class`, a map from the provider model name a call targets (provider prefix stripped) to a class name. A deployment with no map, or a model with no entry, is its own class, named after the model. On a wildcard deployment the model is the one the request resolved to

```yaml
model_list:
  - model_name: anthropic/*
    litellm_params:
      model: anthropic/*
      anthropic_organization_id: org-a
      tpm: 2000000
    model_info:
      id: org-a-claude
      bucket_class:
        claude-opus-4-7: opus-4
        claude-opus-4-6: opus-4
```

Here calls for `claude-opus-4-7` and `claude-opus-4-6` share the `org-a` / `opus-4` bucket, and `claude-opus-5` gets its own. The bucket's capacity is the serving deployment's `tpm` and `rpm`, read from `litellm_params` and then `model_info`. A deployment with neither limit is not admitted against, so its calls are always served

A refusal is a 429 with the usual priority, model, limit type, capacity and saturation fields. The router passes it straight to the caller: no provider call is made, no other deployment or fallback group is tried, and no deployment is cooled down. Success logging charges the tokens to the bucket of the deployment that served the call, read from the logged `model_info`, `anthropic_organization_id` and routed deployment. The `litellm_model_group_saturation_ratio{model_group,priority}` gauge keeps the requested model group as its label and reports the saturation of the bucket the call was admitted against

## Priority Weight Normalization

If priorities sum to > 1.0, they are automatically normalized:

```
Input:  {key_a: 0.60, key_b: 0.80} = 1.40 total
Output: {key_a: 0.43, key_b: 0.57} = 1.00 total
```

This ensures total allocation never exceeds model capacity.

## Implementation Details

### Saturation Detection

- Queries v3 limiter's Redis counters for the bucket's whole window
- Checks both RPM and TPM, returns higher saturation value
- Non-blocking reads (doesn't increment counters)

### Mode Selection

**Generous Mode (below `saturation_threshold`):**
- Counts the priority pools without enforcing them
- Enforces total capacity only
- Allows any priority to use available capacity
- Prevents over-subscription via the bucket-wide limit

**Strict Mode (at or above `saturation_threshold`):**
- Creates priority-specific descriptors with normalized weights
- Each priority gets its reserved allocation
- Keeps enforcing the bucket-wide limit at 100%
- Ensures fairness under load

Test scenarios covered:
1. No rate limiting when under capacity
2. Priority queue behavior during saturation
3. Spillover capacity for default keys
4. Over-allocated priorities with normalization
5. Default priority value handling


### `_PROXY_DynamicRateLimitHandlerV3`

Main handler class inheriting from `CustomLogger`.

**Key Methods:**
- `async_pre_call_hook()` - Records the calling team's class for the router stage
- `async_pre_call_check()` - Admits the call against the bucket of the deployment the router picked
- `async_log_success_event()` - Charges served tokens to the bucket and the team class's pool
- `_check_bucket_saturation()` - Queries Redis for current bucket usage
- `_normalize_priority_weights()` - Handles over-allocation
- `_create_priority_based_descriptors()` - Creates the priority pool descriptor of a bucket

