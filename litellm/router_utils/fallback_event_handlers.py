import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from typing_extensions import TypeIs  # noqa: TID251  # TypeIs reaches typing only on 3.13

import litellm
from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    fallback_target_params_over_call,
    get_metadata_variable_name_from_kwargs,
    is_object_dict,
    is_object_list,
    safe_deep_copy,
)
from litellm.litellm_core_utils.get_llm_provider_logic import inferred_provider
from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure
from litellm.llms.anthropic.common_utils import OBJECT_LIST_ADAPTER, anthropic_owns_fallbacks
from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    get_fallback_error_info,
)
from litellm.router_utils.batch_utils import get_router_metadata_variable_name
from litellm.router_utils.clientside_credential_handler import (
    reject_fallback_target_credentials,
    reject_fallback_target_routing,
)
from litellm.router_utils.cooldown_handlers import (
    _first_present,  # pyright: ignore[reportPrivateUsage] - shared internal helper, used across router_utils
    cast_exception_status_to_int,
    is_advisor_orchestration_failure,
    is_caller_timeout_408,
    set_cooldown_deployments,
)
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    increment_deployment_failures_for_current_minute,
)
from litellm.types.router import LiteLLMParamsTypedDict, reject_server_owned_wif_params

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any

# Status codes a generic API call's caller-supplied resource id can trigger on its own
# (e.g. a nonexistent file/batch/thread id), independent of the selected deployment's health.
_REQUEST_SCOPED_STATUS_CODES: Final = frozenset((404,))
_NO_MODEL_CALL_DETAILS: Final[Mapping[str, object]] = MappingProxyType({})


def _trigger_cooldown_for_failed_deployment(
    litellm_router: LitellmRouter,
    kwargs: Mapping[str, object],
    exception: Exception,
    model_call_details: Mapping[str, object] = _NO_MODEL_CALL_DETAILS,
) -> None:
    """
    Trigger cooldown for a failed fallback deployment.

    In the fallback path the normal failure-callback cooldown is skipped because the
    Logging object sets has_logged_async_failure=True after the first failure and
    blocks all subsequent failure callbacks. This helper ensures every failed
    fallback deployment is evaluated for cooldown regardless.
    """
    try:
        if is_advisor_orchestration_failure(exception):
            verbose_router_logger.debug(
                "Not triggering cooldown for fallback deployment: failure originated "
                "from advisor orchestration, not the selected deployment."
            )
            return

        exception_status: Final[str | int] = getattr(exception, "status_code", "")

        # Generic API calls (files, batches, threads, rerank, ...) take a caller-supplied
        # resource id, so a 404 there usually means "that id doesn't exist" rather than
        # "this deployment is unhealthy". Left unguarded, one bad id would 404 every
        # deployment in the fallback chain and cool all of them down from a single request.
        if (
            kwargs.get("original_generic_function") is not None
            and cast_exception_status_to_int(exception_status) in _REQUEST_SCOPED_STATUS_CODES
        ):
            verbose_router_logger.debug(
                "Not triggering cooldown for fallback deployment: status %s on a generic API "
                "call is caller-attributable, not a deployment health signal.",
                exception_status,
            )
            return

        # The proxy's `x-litellm-timeout` header lets a caller set an arbitrarily short
        # timeout, which litellm.Timeout reports as status 408 regardless of the deployment's
        # actual health. Left unguarded, a caller could force a 408 on every deployment in
        # the fallback chain from a single request with a near-zero timeout.
        if is_caller_timeout_408(
            model_call_details,
            exception_status,
            ended=datetime.now(),  # noqa: DTZ005  # naive to match the logging pipeline's api_call_start_time
        ):
            verbose_router_logger.debug(
                "Not triggering cooldown for fallback deployment: a caller-supplied "
                "x-litellm-timeout caused this 408, not deployment health."
            )
            return

        # Only Router._set_failed_deployment_id_on_exception()'s server-stamped id is
        # trusted here: a metadata-bucket lookup (e.g. "metadata"/"litellm_metadata")
        # can't reliably tell a caller-supplied bucket from a router-authored one
        # without knowing this call's function_name, so a client with permission to
        # set metadata could otherwise get an arbitrary deployment cooled down.
        deployment_id: Final[str | None] = getattr(exception, "failed_deployment_id", None)

        if deployment_id is None:
            verbose_router_logger.debug("Cannot trigger cooldown for fallback: no failed_deployment_id on exception")
            return

        # Priority: deployment config > response header > router default, matching
        # Router.deployment_callback_on_failure's precedence for the primary path.
        deployment_dict: Final = litellm_router.get_model_info(id=deployment_id)
        deployment_cooldown: Final = (
            _first_present(
                deployment_dict.get("model_info"), deployment_dict.get("litellm_params"), key="cooldown_time"
            )
            if deployment_dict is not None
            else None
        )
        exception_headers: Final = litellm.litellm_core_utils.exception_mapping_utils._get_response_headers(
            original_exception=exception
        )
        _get_retry_after: Final = (
            litellm.utils._get_retry_after_from_exception_header  # pyright: ignore[reportPrivateUsage] - as router.py
        )
        header_cooldown: Final = (
            _get_retry_after(response_headers=exception_headers) if exception_headers is not None else None
        )
        time_to_cooldown: Final = (
            deployment_cooldown
            if deployment_cooldown is not None and deployment_cooldown >= 0
            else (
                header_cooldown
                if header_cooldown is not None and header_cooldown >= 0
                else litellm_router.cooldown_time
            )
        )

        increment_deployment_failures_for_current_minute(
            litellm_router_instance=litellm_router,
            deployment_id=deployment_id,
        )
        set_cooldown_deployments(
            litellm_router_instance=litellm_router,
            exception_status=exception_status,
            original_exception=exception,
            deployment=deployment_id,
            time_to_cooldown=time_to_cooldown,
        )

        verbose_router_logger.debug("Triggered cooldown for fallback deployment %s", deployment_id)
    except Exception as e:  # noqa: BLE001 - best-effort cooldown trigger must never break the fallback response itself
        verbose_router_logger.debug("Error triggering cooldown for fallback deployment: %s", e)


def fallback_attempt_key(fallback_target: object) -> str | None:
    """
    Identity of one fallback attempt, so the same attempt is never made twice per request.

    A bare model group name and a `{"model": name}` entry describe the same attempt. An
    entry carrying anything else describes a different one and keeps its own identity: a
    client-side fallback list overrides request params such as `messages`, and the router
    re-targets the group that just failed by attaching `_target_order` or
    `_excluded_deployment_ids` to select a different set of deployments inside it. The
    payload is hashed rather than kept, so a large `messages` override does not make the
    request hold a second copy of itself.

    Returns None for a shape with no usable identity, which is never skipped.
    """
    if isinstance(fallback_target, str):
        return fallback_target
    if not isinstance(fallback_target, dict):
        return None
    model: Final = fallback_target.get("model")
    if tuple(fallback_target) == ("model",) and isinstance(model, str):
        return model
    serialized: Final = json.dumps(fallback_target, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass(slots=True)
class AttemptedFallbackTargets:
    """
    The fallback attempts a single request has already made.

    One instance is created on the first fallback hop and shared by reference for the rest
    of the walk, so an attempt made in one branch is not repeated in a sibling branch.
    Without it the walk enumerates paths rather than attempts: a fallback graph containing
    a cycle retries one deterministic failure once per path through the cycle, and a
    client-side fallback list is re-walked at every level of the recursion.
    """

    keys: frozenset[str] = frozenset()

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def record(self, key: str) -> None:
        self.keys = self.keys | frozenset((key,))


def has_unattempted_fallback_target(
    fallback_model_group: Sequence[object] | None, kwargs: Mapping[str, object]
) -> bool:
    """Whether a resolved chain still holds an entry this request has not tried."""
    if fallback_model_group is None:
        return False
    attempted: Final = kwargs.get("attempted_targets")
    if not isinstance(attempted, AttemptedFallbackTargets):
        return True
    return any((key := fallback_attempt_key(target)) is None or key not in attempted for target in fallback_model_group)


def _check_stripped_model_group(model_group: str, fallback_key: str) -> bool:
    """
    Handles wildcard routing scenario

    where fallbacks set like:
    [{"gpt-3.5-turbo": ["claude-3-haiku"]}]

    but model_group is like:
    "openai/gpt-3.5-turbo"

    Returns:
    - True if the stripped model group == fallback_key
    """
    for provider in litellm.provider_list:
        if isinstance(provider, Enum):
            _provider = provider.value
        else:
            _provider = provider
        if model_group.startswith(f"{_provider}/"):
            stripped_model_group = model_group.replace(f"{_provider}/", "")
            if stripped_model_group == fallback_key:
                return True
    return False


def _provider_prefixed_model_group(model_group: str, fallback_keys: Sequence[str]) -> str | None:
    if "/" in model_group or not any(key.endswith(f"/{model_group}") for key in fallback_keys):
        return None
    provider: Final = inferred_provider(model_group)
    return f"{provider}/{model_group}" if provider else None


PRE_ROUTING_SELECTED_MODEL_KEY: Final = "pre_routing_selected_model"
_ROUTER_METADATA_BUCKETS: Final = ("metadata", "litellm_metadata")


def record_pre_routing_selection(request_kwargs: Mapping[str, object] | None, selected_model: str) -> None:
    """
    Remember which model a pre-routing hook picked, so fallback lookup can key off it.

    Fallback resolution runs on an outer kwargs dict that ``**kwargs`` already copied, so
    writing the model there is invisible by the time routing picks a tier. The metadata
    buckets are nested dicts shared by reference across those copies, which is how the
    router already carries values back up.

    The write goes through the proxy-internal bucket resolver, never into both buckets:
    on /v1/messages the top-level ``metadata`` dict is the provider's own request field,
    so a blanket write would forward the tier stamp upstream.
    """
    if request_kwargs is None:
        return
    bucket: Final = request_kwargs.get(get_metadata_variable_name_from_kwargs(request_kwargs))
    if isinstance(bucket, dict):
        bucket[PRE_ROUTING_SELECTED_MODEL_KEY] = selected_model


def clear_pre_routing_selection(request_kwargs: Mapping[str, object] | None) -> None:
    """
    Drop any selection the router did not make itself on this hop.

    The buckets carry whatever the caller sent, so an inbound value is the caller
    choosing a fallback chain rather than the router choosing a tier. A fallback hop
    also inherits the previous hop's selection, which would key its own failure off
    the tier that already failed. Clearing at the start of every hop leaves only a
    value the pre-routing hook wrote while routing that hop.
    """
    if request_kwargs is None:
        return
    for bucket in (request_kwargs.get(name) for name in _ROUTER_METADATA_BUCKETS):
        if isinstance(bucket, dict) and PRE_ROUTING_SELECTED_MODEL_KEY in bucket:
            del bucket[PRE_ROUTING_SELECTED_MODEL_KEY]


def get_pre_routing_selection(kwargs: Mapping[str, object]) -> str | None:
    """The model a pre-routing hook selected for this request, if one did."""
    buckets: Final = (kwargs.get(name) for name in _ROUTER_METADATA_BUCKETS)
    selections: Final = (bucket.get(PRE_ROUTING_SELECTED_MODEL_KEY) for bucket in buckets if isinstance(bucket, dict))
    return next((selected for selected in selections if isinstance(selected, str) and selected), None)


def carry_over_pre_routing_selection(live_kwargs: Mapping[str, object], snapshot: Mapping[str, object]) -> None:
    """
    Replace whatever selection the snapshot carries with the one the pre-routing hook stamped
    into the live kwargs while routing this attempt, so a mid-stream fallback keys its lookup
    off the tier this attempt actually routed to.
    """
    clear_pre_routing_selection(snapshot)
    live_selection: Final = get_pre_routing_selection(live_kwargs)
    if live_selection is not None:
        record_pre_routing_selection(snapshot, live_selection)


MID_STREAM_FALLBACK_CONTROLS_KEY: Final = "_mid_stream_fallback_controls"
ROUTER_FALLBACKS_KWARG: Final = "router_fallbacks"
_PER_REQUEST_FALLBACK_CONTROL_KEYS: Final = (
    "fallbacks",
    ROUTER_FALLBACKS_KWARG,
    "context_window_fallbacks",
    "content_policy_fallbacks",
    "num_retries",
    "model_group_retry_policy",
)


def provider_owns_fallbacks_field(kwargs: Mapping[str, object], *, anthropic_messages_call: bool = False) -> bool:
    """
    Whether this anthropic_messages request's `fallbacks` is Anthropic's server-side fallback list, which then goes to
    Anthropic as is while the Router's own per-request list rides in ROUTER_FALLBACKS_KWARG. A caller that knows the
    call is anthropic_messages says so, since not every kwargs carries ``original_generic_function``.
    """
    function_name: Final[object] = getattr(kwargs.get("original_generic_function"), "__name__", None)
    return (anthropic_messages_call or function_name == "anthropic_messages") and anthropic_owns_fallbacks(kwargs)


def _router_fallbacks_key(kwargs: Mapping[str, object], anthropic_messages_call: bool) -> str:
    return (
        ROUTER_FALLBACKS_KWARG
        if provider_owns_fallbacks_field(kwargs, anthropic_messages_call=anthropic_messages_call)
        else "fallbacks"
    )


def fallback_entries_are_targets(fallbacks: Sequence[object], field: str) -> bool:
    """
    Whether the Router calls every entry of this fallback field's list as a target, a string as a model and a mapping
    merged into the call with or without a ``model``, rather than reading each mapping as a model group mapped to the
    list of targets it calls once that group fails: so only for a ``fallbacks`` list in the client-side format. The
    list a model group is mapped to is always a list of targets.
    """
    return field in ("fallbacks", ROUTER_FALLBACKS_KWARG) and check_non_standard_fallback_format(fallbacks)


def _unsupported_group_map_entry(fallbacks: Sequence[object], field: str) -> str | None:
    # The Router reads a mapping of a list in the group format by its first key and iterates the value it finds there:
    # a string would be called character by character, a mapping key by key, and an empty mapping has no key. An entry,
    # and an entry of a group's list, that is neither a model name nor a mapping is called as neither.
    if fallback_entries_are_targets(fallbacks, field):
        return None
    for entry in fallbacks:
        if isinstance(entry, str):
            continue
        if not is_object_dict(entry):
            return f"must hold model names and mappings, got {type(entry).__name__}"
        if not entry:
            return "must not hold an empty mapping"
        if fallback_entry_is_target(entry):
            if field in ("fallbacks", ROUTER_FALLBACKS_KWARG):
                continue
            return f"maps model groups to their fallbacks, so it cannot hold a target naming model {entry['model']!r}"
        for group, chain in entry.items():
            if not is_object_list(chain):
                return f"must map model group {group!r} to a list of model groups, got {type(chain).__name__}"
            unsupported = tuple(type(item).__name__ for item in chain if not isinstance(item, (str, dict)))
            if unsupported:
                return f"must map model group {group!r} to model names and mappings, got {unsupported[0]}"
    return None


# The client wrapper swaps a model for this mapping's value on a context-window error (litellm/utils.py)
CONTEXT_WINDOW_FALLBACK_DICT_FIELD: Final = "context_window_fallback_dict"
ROUTER_SETTINGS_OVERRIDE_KWARG: Final = "router_settings_override"
ERROR_FALLBACK_FIELDS: Final = ("context_window_fallbacks", "content_policy_fallbacks")
# Every request field that holds a LiteLLM fallback list, in a request body and in each target merged into a call
FALLBACK_LIST_FIELDS: Final = (
    "fallbacks",
    ROUTER_FALLBACKS_KWARG,
    "context_window_fallbacks",
    "content_policy_fallbacks",
    # A Router init setting, never read per request; read so a body copy can never carry a target past a check.
    "default_fallbacks",
)
# The kwargs that carry a call's headers to the provider; a target merged into the call replaces each one it carries.
HEADER_CARRIER_KWARGS: Final = ("headers", "extra_headers", "provider_specific_header")


class FallbackListShapeError(ValueError):
    """
    A fallback list no reader can call as sent: an entry that is neither a model name nor a mapping, an error field
    that is not a list, or nesting deeper than any reader follows. ``param`` names the field the list came in.
    """

    def __init__(self, message: str, field: str | None) -> None:
        super().__init__(message)
        self.param: Final[str] = field or "fallbacks"


def _is_mapping(value: object) -> TypeIs[Mapping[object, object]]:  # guard-ok: isinstance cannot type a mapping's keys
    return isinstance(value, Mapping)


def _is_entry_mapping(
    value: object,
) -> TypeIs[Mapping[str, object]]:  # guard-ok: an entry comes from a JSON body or a call's kwargs, keyed by strings
    return isinstance(value, Mapping)


def fallback_target_model_name(target: object) -> str | None:
    """The model a fallback entry names: a string entry, or a mapping's string ``model``"""
    if isinstance(target, str):
        return target
    if _is_mapping(target):
        model: Final = target.get("model")
        if isinstance(model, str):
            return model
    return None


class FallbackReading(Enum):
    """Whose list each ``fallbacks`` field of a request and of its targets is, on every hop the request can take"""

    LITELLM = "litellm"
    ANTHROPIC = "anthropic"
    # A hop can carry the beta or lose it, by the header fields an earlier target left in the call: read both ways
    BOTH = "both"


class FallbackBudgetError(FallbackListShapeError):
    """
    Fallback lists that hold more entries in all than their budget, refused when the walk reaches the list that passes
    it, before it reads that list's entries
    """


@dataclass(frozen=True, slots=True)
class FallbackEntry:
    """
    One entry of a fallback list: a model name, or a mapping a reader merges into the call. ``group`` is the key of the
    model group map the entry is or stands under in a ``fallbacks`` list in the Router's group format, a map the Router
    reads as that group's fallbacks while litellm.acompletion's own fallback loop, which serves a request no Router
    serves, merges it into the call as a parameter override; ``is_group_map`` marks the map itself. ``anthropic`` marks
    an entry of a list Anthropic reads. ``via_litellm_fallbacks`` marks an entry reached through a ``fallbacks`` field
    read as LiteLLM's by a walk that reads it both ways.
    """

    target: str | Mapping[str, object]
    group: str | None = None
    is_group_map: bool = False
    anthropic: bool = False
    via_litellm_fallbacks: bool = False


@dataclass(frozen=True, slots=True)
class FallbackListRead:
    """A value a LiteLLM reader iterates as the fallback list ``field`` holds"""

    value: object
    field: str
    via_litellm_fallbacks: bool = False


def _in_reading(event: FallbackEntry | FallbackListRead, reading: FallbackReading) -> bool:
    """Whether a walk that reads every ``fallbacks`` field both ways reaches ``event`` under ``reading`` too"""
    if reading is FallbackReading.LITELLM:
        return isinstance(event, FallbackListRead) or not event.anthropic
    if reading is FallbackReading.ANTHROPIC:
        return not event.via_litellm_fallbacks
    return True


@dataclass(frozen=True, slots=True)
class FallbackChecks:
    """
    What a reader checks of what a walk reached: ``entry`` sees every entry, a target before the lists it carries, and
    ``litellm_list`` sees every list LiteLLM reads, with the field it is the value of.
    """

    entry: Callable[[FallbackEntry], None] | None = None
    litellm_list: Callable[[object, str], None] | None = None


_NO_FALLBACK_CHECKS: Final = FallbackChecks()


@dataclass(frozen=True, slots=True)
class FallbackWalk:
    """
    One walk of fallback lists: whose list each ``fallbacks`` field is, what its readers reach in walk order (every
    entry, and every list LiteLLM iterates), and the shape error the walk stopped at
    """

    reading: FallbackReading
    events: tuple[FallbackEntry | FallbackListRead, ...]
    error: FallbackListShapeError | None = None

    @property
    def entries(self) -> tuple[FallbackEntry, ...]:
        return tuple(event for event in self.events if isinstance(event, FallbackEntry))

    def check(self, checks: FallbackChecks = _NO_FALLBACK_CHECKS) -> tuple[FallbackEntry, ...]:
        """
        Runs ``checks`` on what the walk reached, in its order, then raises the shape error it stopped at, so a check
        refuses an entry before the walk's refusal of a later one. Returns the entries.
        """
        for event in self.events:
            if isinstance(event, FallbackListRead):
                if checks.litellm_list is not None:
                    checks.litellm_list(event.value, event.field)
            elif checks.entry is not None:
                checks.entry(event)
        if self.error is not None:
            raise self.error
        return self.entries


@dataclass(slots=True)
class _FallbackRead:
    """
    The state of one walk. It reads each list, and each mapping merged into the call, once per way of reading it: the
    field it is read as, its depth, whose list it is, the group key above it, and whether a both-ways ``fallbacks``
    read lies above it. A walk is so linear in the entries. ``budget`` caps the entries of all the lists, each counted
    once however many ways it is read; None leaves them uncounted.
    """

    reading: FallbackReading
    budget: int | None
    read: set[tuple[object, ...]]  # mutable-ok: one walk's own record of what it read, never handed out
    counted: set[int]  # mutable-ok: one walk's own record of the lists it counted, never handed out
    entries: int = 0

    def first_read(self, way: tuple[object, ...]) -> bool:
        if way in self.read:
            return False
        self.read.add(way)
        return True

    def count(self, container: object, entries: int, field: str | None) -> None:
        if id(container) in self.counted:
            return
        self.counted.add(id(container))
        self.entries += entries
        if self.budget is not None and self.entries > self.budget:
            raise FallbackBudgetError(
                f"Rejected Request: the fallback lists hold more than {self.budget} entries in all.", field
            )


def _until_shape_error(
    events: Iterator[FallbackEntry | FallbackListRead],
) -> Iterator[FallbackEntry | FallbackListRead | FallbackListShapeError]:
    """What a walk reached, then the shape error it stopped at; a list past the budget is refused at once"""
    try:
        yield from events
    except FallbackListShapeError as shape_error:
        if isinstance(shape_error, FallbackBudgetError):
            raise
        yield shape_error


def _recorded(events: Iterator[FallbackEntry | FallbackListRead], reading: FallbackReading) -> FallbackWalk:
    """What a walk reached and the shape error it stopped at, for the checks to read in the walk's order"""
    reached: Final = tuple(_until_shape_error(events))
    stopped: Final = reached[-1] if reached and isinstance(reached[-1], FallbackListShapeError) else None
    return FallbackWalk(
        reading, tuple(event for event in reached if not isinstance(event, FallbackListShapeError)), stopped
    )


def walk_fallback_lists(
    body: Mapping[str, object],
    *,
    carriers: Mapping[str, object],
    anthropic_messages_call: bool,
    budget: int | None = None,
) -> FallbackWalk:
    """
    One walk of every fallback list a request body carries, with whose list each ``fallbacks`` field is on every hop.
    ``carriers`` holds the request's header fields, which the Router's provider_owns_fallbacks_field reads for the
    beta. A target that carries a header field replaces the call's on its hop and the Router keeps it on later hops, so
    when any target carries one and any hop can carry the beta, the order of hops decides and every ``fallbacks`` field
    is read both ways. An anthropic_messages call's walk reads both ways from the start, decides the reading from the
    targets it reached, and keeps what that reading reaches: one walk serves the reading and every check. ``budget``
    caps the entries of all the lists.
    """
    if not anthropic_messages_call:
        return _recorded(
            iter_body_fallback_entries(body, _FallbackRead(FallbackReading.LITELLM, budget, set(), set())),
            FallbackReading.LITELLM,
        )
    both_ways: Final = _recorded(
        iter_body_fallback_entries(body, _FallbackRead(FallbackReading.BOTH, budget, set(), set())),
        FallbackReading.BOTH,
    )
    if both_ways.error is not None:
        # The checks refuse the lists; nothing is known of their hops until then
        return both_ways
    request_beta: Final = provider_owns_fallbacks_field(carriers, anthropic_messages_call=True)
    carrying: Final = tuple(
        entry.target
        for entry in both_ways.entries
        if not isinstance(entry.target, str) and any(key in entry.target for key in HEADER_CARRIER_KWARGS)
    )
    target_beta: Final = any(provider_owns_fallbacks_field(target, anthropic_messages_call=True) for target in carrying)
    reading: Final = (
        (FallbackReading.ANTHROPIC if request_beta else FallbackReading.LITELLM)
        if not carrying
        else (FallbackReading.BOTH if request_beta or target_beta else FallbackReading.LITELLM)
    )
    return FallbackWalk(reading, tuple(event for event in both_ways.events if _in_reading(event, reading)))


def walk_fallback_list(value: object, field: str, reading: FallbackReading) -> FallbackWalk:
    """One walk of the fallback list ``field`` holds, its targets' own ``fallbacks`` read as ``reading`` says"""
    return _recorded(iter_fallback_entries(value, field, _FallbackRead(reading, None, set(), set())), reading)


def _list_readings(
    reading: FallbackReading, *, own_fallbacks: bool, widest_reading: bool, via_litellm_fallbacks: bool
) -> tuple[tuple[bool, bool], ...]:
    """
    For each reading of a list: whether it is Anthropic's, read in the widest way, and whether a both-ways
    ``fallbacks`` field read as LiteLLM's lies above what it reaches
    """
    if widest_reading:
        return ((True, via_litellm_fallbacks),)
    if not own_fallbacks or reading is FallbackReading.LITELLM:
        return ((False, via_litellm_fallbacks),)
    if reading is FallbackReading.ANTHROPIC:
        return ((True, via_litellm_fallbacks),)
    return ((False, True), (True, via_litellm_fallbacks))


def iter_fallback_entries(
    value: object,
    field: str | None,
    walk: _FallbackRead,
    *,
    depth: int = 0,
    widest_reading: bool = False,
    group: str | None = None,
    via_litellm_fallbacks: bool = False,
) -> Iterator[FallbackEntry | FallbackListRead]:
    """
    The entries of one fallback list, read as its reader reads it: ``field`` is the fallback field whose value the list
    is, and None marks a list of targets, such as the one a model group is mapped to. Every mapping merged into the
    call has its fallback lists read as a request body's are, by iter_merged_target. A ``fallbacks`` list in the group
    format is read both ways: a model group map there is also merged as a parameter override, and its names are tagged
    with its key, which ``group`` carries down. ``widest_reading`` reads every mapping without a string ``model`` as a
    group map, at every level, for a list Anthropic reads, whose names no LiteLLM reader calls, so a name in any shape
    gets the checks. A list the walk read before in the same way yields nothing again.
    """
    if depth > 2 * litellm.ROUTER_MAX_FALLBACKS:
        raise FallbackListShapeError("Rejected Request: fallback nesting exceeds the allowed validation depth.", field)
    if not walk.first_read(("list", id(value), field, depth, widest_reading, group, via_litellm_fallbacks)):
        return
    if field is not None and not widest_reading:
        yield FallbackListRead(value, field, via_litellm_fallbacks)
    if _is_mapping(value):
        walk.count(value, len(value), field)
        # The Router iterates a fallback list, and iterating a mapping walks its keys, so each key is a target
        for key in value:
            if isinstance(key, str):
                yield FallbackEntry(key, group, anthropic=widest_reading, via_litellm_fallbacks=via_litellm_fallbacks)
        for target_list in value.values():
            yield from iter_fallback_entries(
                target_list,
                None,
                walk,
                depth=depth + 1,
                widest_reading=widest_reading,
                group=group,
                via_litellm_fallbacks=via_litellm_fallbacks,
            )
    elif is_object_list(value):
        walk.count(value, len(value), field)
        yield from _iter_list_entries(
            value,
            field,
            walk,
            depth=depth,
            widest_reading=widest_reading,
            group=group,
            via_litellm_fallbacks=via_litellm_fallbacks,
        )


def _iter_list_entries(
    items: Sequence[object],
    field: str | None,
    walk: _FallbackRead,
    *,
    depth: int,
    widest_reading: bool,
    group: str | None,
    via_litellm_fallbacks: bool,
) -> Iterator[FallbackEntry | FallbackListRead]:
    entries_are_targets: Final = not widest_reading and (field is None or fallback_entries_are_targets(items, field))
    read_both_ways: Final = field == "fallbacks" and not widest_reading and not entries_are_targets
    for item in items:
        if isinstance(item, str):
            yield FallbackEntry(item, group, anthropic=widest_reading, via_litellm_fallbacks=via_litellm_fallbacks)
        elif not _is_entry_mapping(item):
            # Every reader takes an entry for a model name or a mapping, so a nested list is called as neither
            raise FallbackListShapeError(
                "Rejected Request: a fallback list entry must be a model name or a mapping, "
                f"got {type(item).__name__}.",
                field,
            )
        elif entries_are_targets or fallback_entry_is_target(item):
            yield from iter_merged_target(
                item,
                walk,
                depth=depth + 1,
                widest_reading=widest_reading,
                group=group,
                via_litellm_fallbacks=via_litellm_fallbacks,
            )
        else:
            # A model group mapped to the targets the Router calls once that group fails. In `fallbacks`,
            # litellm.acompletion's own fallback loop merges the mapping itself into the call instead.
            item_group = ", ".join(str(key) for key in item) if read_both_ways and group is None else group
            if read_both_ways:
                yield from iter_merged_target(
                    item,
                    walk,
                    depth=depth + 1,
                    group=item_group,
                    is_group_map=True,
                    via_litellm_fallbacks=via_litellm_fallbacks,
                )
            for target_list in item.values():
                yield from iter_fallback_entries(
                    target_list,
                    None,
                    walk,
                    depth=depth + 1,
                    widest_reading=widest_reading,
                    group=item_group,
                    via_litellm_fallbacks=via_litellm_fallbacks,
                )


def iter_merged_target(
    target: Mapping[str, object],
    walk: _FallbackRead,
    *,
    depth: int,
    widest_reading: bool = False,
    group: str | None = None,
    is_group_map: bool = False,
    via_litellm_fallbacks: bool = False,
) -> Iterator[FallbackEntry | FallbackListRead]:
    """
    One mapping a reader merges into the call, its fallback lists read as a request body's are: the mapping itself,
    then iter_body_fallback_entries reads every list through which the call it lands in picks another model, the
    mapping's own ``fallbacks`` as Anthropic's or LiteLLM's by the walk's reading.
    """
    yield FallbackEntry(target, group, is_group_map, widest_reading, via_litellm_fallbacks)
    yield from iter_body_fallback_entries(
        target,
        walk,
        depth=depth,
        widest_reading=widest_reading,
        group=group,
        via_litellm_fallbacks=via_litellm_fallbacks,
    )


def iter_body_fallback_entries(
    body: Mapping[str, object],
    walk: _FallbackRead,
    *,
    depth: int = 0,
    widest_reading: bool = False,
    group: str | None = None,
    via_litellm_fallbacks: bool = False,
) -> Iterator[FallbackEntry | FallbackListRead]:
    """
    The entries of every fallback list a request body carries, the request's own or a target's merged into the call:
    its fallback fields, those of its router_settings_override, and the models its context_window_fallback_dict swaps
    in. Its own ``fallbacks`` is read as the walk's reading says: Anthropic's, read in the widest way, LiteLLM's, or
    both. A body the walk read before in the same way yields nothing again.
    """
    if not walk.first_read(("body", id(body), depth, widest_reading, group, via_litellm_fallbacks)):
        return
    override: Final = body.get(ROUTER_SETTINGS_OVERRIDE_KWARG)
    for source in (body, override):
        if not _is_mapping(source):
            continue
        for field in FALLBACK_LIST_FIELDS:
            value = source.get(field)
            own_fallbacks = source is body and field == "fallbacks"
            for anthropic_list, via in _list_readings(
                walk.reading,
                own_fallbacks=own_fallbacks,
                widest_reading=widest_reading,
                via_litellm_fallbacks=via_litellm_fallbacks,
            ):
                yield from iter_fallback_entries(
                    value,
                    field,
                    walk,
                    depth=depth,
                    widest_reading=anthropic_list,
                    group=group,
                    via_litellm_fallbacks=via,
                )
    context_window_swaps: Final = body.get(CONTEXT_WINDOW_FALLBACK_DICT_FIELD)
    if _is_mapping(context_window_swaps):
        # Each value is the model the client wrapper swaps in, so the values are a list of targets
        walk.count(context_window_swaps, len(context_window_swaps), CONTEXT_WINDOW_FALLBACK_DICT_FIELD)
        yield from _iter_list_entries(
            tuple(context_window_swaps.values()),
            None,
            walk,
            depth=depth,
            widest_reading=widest_reading,
            group=group,
            via_litellm_fallbacks=via_litellm_fallbacks,
        )


# A Router init setting, never read per call, so no Router rule applies to it
_ROUTER_UNREAD_LIST_FIELDS: Final = frozenset({"default_fallbacks"})


def _reject_router_unreadable_list(fallbacks: object, field: str, *, model: object) -> None:
    if fallbacks is None or field in _ROUTER_UNREAD_LIST_FIELDS:
        return
    problem: Final = (
        f"must be a list of model groups, got {type(fallbacks).__name__}"
        if not isinstance(fallbacks, list)
        else _unsupported_group_map_entry(fallbacks, field)
    )
    if problem is not None:
        raise litellm.BadRequestError(
            message=f"LiteLLM {field} {problem}",
            model=model if isinstance(model, str) else "",
            llm_provider="",
        )


def reject_non_list_litellm_fallbacks(
    fallbacks: object,
    model: object,
    *,
    field: str = "fallbacks",
    list_of_targets: bool = False,
    call_kwargs: Mapping[str, object] | None = None,
    anthropic_messages_call: bool = False,
) -> None:
    """
    LiteLLM fallback fields hold lists of model groups. In the Router's formats a list of group maps maps each group to
    such a list, and an error field's list holds no target. The same holds for every list a target carries, which the
    Router reads once a hop merges that target, so each is checked here, through the walk the proxy's checks read the
    lists with, before the first call. ``call_kwargs`` are the kwargs of the call, whose header carriers decide whether
    a target's own ``fallbacks`` is Anthropic's; without them no LiteLLM rule is applied to it. ``list_of_targets``
    marks a list whose reader calls every entry as a target, as the direct completion fallback does, so only its type
    is checked. A hop, a call that carries its request's AttemptedFallbackTargets, is not read again: each list in its
    kwargs is one its request's first call checked, or the own list of a target merged into the call, which that check
    walked.
    """
    if list_of_targets:
        if fallbacks is not None and not isinstance(fallbacks, list):
            _reject_router_unreadable_list(fallbacks, field, model=model)
        return
    if call_kwargs is not None and isinstance(call_kwargs.get("attempted_targets"), AttemptedFallbackTargets):
        return
    checks: Final = FallbackChecks(
        litellm_list=lambda value, value_field: _reject_router_unreadable_list(value, value_field, model=model)
    )
    try:
        walk_fallback_list(
            fallbacks, field, _router_reading(call_kwargs, fallbacks, field, anthropic_messages_call)
        ).check(checks)
    except FallbackListShapeError as shape_error:
        raise litellm.BadRequestError(
            message=str(shape_error),
            model=model if isinstance(model, str) else "",
            llm_provider="",
        ) from shape_error


def _router_reading(
    kwargs: Mapping[str, object] | None, fallbacks: object, field: str, anthropic_messages_call: bool
) -> FallbackReading:
    """Whose list each ``fallbacks`` field is, for a Router check of ``fallbacks``, the list ``field`` holds"""
    if kwargs is None:
        return FallbackReading.BOTH
    function_name: Final[object] = getattr(kwargs.get("original_generic_function"), "__name__", None)
    return walk_fallback_lists(
        {**kwargs, field: fallbacks},
        carriers=kwargs,
        anthropic_messages_call=anthropic_messages_call or function_name == "anthropic_messages",
    ).reading


def request_fallbacks(
    kwargs: Mapping[str, object], router_fallbacks: Sequence[object] | None, *, anthropic_messages_call: bool = False
) -> list[object] | None:  # mutable-ok: the Router hands it on to its fallback functions' `list | None` parameter
    """The Router's fallback list for this request: the per-request override, else the Router's own"""
    fallbacks: Final = kwargs.get(_router_fallbacks_key(kwargs, anthropic_messages_call), router_fallbacks)
    reject_non_list_litellm_fallbacks(
        fallbacks, kwargs.get("model"), call_kwargs=kwargs, anthropic_messages_call=anthropic_messages_call
    )
    return None if fallbacks is None else OBJECT_LIST_ADAPTER.validate_python(fallbacks)


def pop_request_fallbacks(
    kwargs: dict[str, object],  # mutable-ok: async_function_with_retries pops its controls before the attempt runs
    router_fallbacks: Sequence[object] | None,
) -> list[object] | None:  # mutable-ok: the Router hands it on to its fallback functions' `list | None` parameter
    fallbacks: Final = kwargs.pop(_router_fallbacks_key(kwargs, False), router_fallbacks)
    reject_non_list_litellm_fallbacks(fallbacks, kwargs.get("model"), call_kwargs=kwargs)
    return None if fallbacks is None else OBJECT_LIST_ADAPTER.validate_python(fallbacks)


@dataclass(frozen=True, slots=True)
class MidStreamFallbackControls:
    """
    The per-request fallback and retry overrides every streaming attempt must see again.

    async_function_with_retries pops them before the attempt function runs, so without this
    carrier a fallback hop's own mid-stream re-entry would fall back to the router-level settings.
    """

    overrides: Mapping[str, object]


_NO_FALLBACK_CONTROLS: Final = MidStreamFallbackControls(MappingProxyType({}))


def per_request_fallback_controls(kwargs: Mapping[str, object]) -> MidStreamFallbackControls:
    return MidStreamFallbackControls(
        MappingProxyType({key: kwargs[key] for key in _PER_REQUEST_FALLBACK_CONTROL_KEYS if key in kwargs})
    )


def mid_stream_fallback_snapshot_kwargs(
    model: str,
    controls: object,
    kwargs: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: the streaming iterators rewrite it in place when they re-enter the chain
    """
    The kwargs a completion attempt's stream re-enters the fallback chain with if it fails.

    async_function_with_retries popped the per-request fallback lists before the attempt ran, so
    the carrier restores them here and rides along into every hop this re-entry opens. A shallow
    copy keeps the metadata buckets shared with the live kwargs, the way the attempt's own
    in-place bucket writes expect.
    """
    hop_controls: Final = controls if isinstance(controls, MidStreamFallbackControls) else _NO_FALLBACK_CONTROLS
    return {
        **kwargs,
        **hop_controls.overrides,
        MID_STREAM_FALLBACK_CONTROLS_KEY: hop_controls,
        "model": model,
    }


def mid_stream_fallback_hop_kwargs(
    model: str,
    original_generic_function: Callable[..., object],
    controls: object,
    kwargs: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: the streaming iterators rewrite it in place when they re-enter the chain
    """
    The kwargs one generic-endpoint streaming attempt re-enters the fallback chain with if its stream fails.

    A shallow copy keeps ``attempted_targets`` shared with the outer chain, so entries this
    request already tried are never retried; the metadata buckets are copied key by key because
    the attempt writes deployment-specific fields into them in place.
    """
    copied_buckets: Final = MappingProxyType(
        {name: safe_deep_copy(kwargs[name]) for name in _ROUTER_METADATA_BUCKETS if isinstance(kwargs.get(name), dict)}
    )
    return {
        **mid_stream_fallback_snapshot_kwargs(model=model, controls=controls, kwargs=kwargs),
        **copied_buckets,
        "original_generic_function": original_generic_function,
    }


_MID_STREAM_RETRY_STRIPPED_KEYS: Final = (*_PER_REQUEST_FALLBACK_CONTROL_KEYS, "original_function")
_PROVIDER_FALLBACKS_RETRY_STRIPPED_KEYS: Final = tuple(
    key for key in _MID_STREAM_RETRY_STRIPPED_KEYS if key != "fallbacks"
)
_MID_STREAM_RETRY_ATTEMPTED_KEY: Final = "attempted_retries"
_MID_STREAM_RETRY_BUDGET_KEY: Final = "max_retries"


def mid_stream_retry_kwargs(
    hop_kwargs: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: unpacked as **kwargs into the attempt function, which pops its controls carrier
    """
    The kwargs a same-group retry re-enters the attempt function with. async_function_with_retries
    pops the per-request controls and the chain's original_function before any attempt runs, and
    the controls carrier the snapshot still holds restores the overrides into the retry's own hop.
    """
    stripped: Final = (
        _PROVIDER_FALLBACKS_RETRY_STRIPPED_KEYS
        if provider_owns_fallbacks_field(hop_kwargs)
        else _MID_STREAM_RETRY_STRIPPED_KEYS
    )
    return {key: value for key, value in hop_kwargs.items() if key not in stripped}


def _request_metadata_bucket(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    bucket: Final = kwargs.get(get_metadata_variable_name_from_kwargs(kwargs))
    return bucket if isinstance(bucket, Mapping) else None


def attempted_retries_for_request(kwargs: Mapping[str, object]) -> int:
    """How many same-group retries async_function_with_retries, or a mid-stream retry, already spent on this request."""
    bucket: Final = _request_metadata_bucket(kwargs)
    attempted: Final = bucket.get(_MID_STREAM_RETRY_ATTEMPTED_KEY) if bucket is not None else None
    return attempted if type(attempted) is int and attempted > 0 else 0


def committed_retry_budget_for_request(kwargs: Mapping[str, object]) -> int | None:
    """The budget the first retry of this request committed to, kept by every later attempt the way the
    pre-stream retry loop keeps its own; None until a retry has run."""
    if attempted_retries_for_request(kwargs) == 0:
        return None
    bucket: Final = _request_metadata_bucket(kwargs)
    budget: Final = bucket.get(_MID_STREAM_RETRY_BUDGET_KEY) if bucket is not None else None
    return budget if type(budget) is int else None


def record_retry_attempt(kwargs: Mapping[str, object], attempted_retries: int, max_retries: int) -> None:
    """Stamp the attempt about to run the way async_function_with_retries does before each of its retries."""
    bucket: Final = kwargs.get(get_metadata_variable_name_from_kwargs(kwargs))
    if not isinstance(bucket, dict):
        return
    bucket[_MID_STREAM_RETRY_ATTEMPTED_KEY] = attempted_retries
    bucket[_MID_STREAM_RETRY_BUDGET_KEY] = max_retries


def _routed_model_info(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    bucket: Final = _request_metadata_bucket(kwargs)
    model_info: Final = bucket.get("model_info") if bucket is not None else None
    return model_info if isinstance(model_info, Mapping) else None


def routed_deployment_id(kwargs: Mapping[str, object]) -> str | None:
    model_info: Final = _routed_model_info(kwargs)
    deployment_id: Final = model_info.get("id") if model_info is not None else None
    return deployment_id if isinstance(deployment_id, str) else None


def carry_over_routed_deployment(live_kwargs: Mapping[str, object], snapshot: Mapping[str, object]) -> None:
    """
    Copy the deployment this attempt routed to into the snapshot's metadata bucket, which was
    taken before routing: a same-group retry reads the deployment's own num_retries off it and
    records which deployment failed.
    """
    snapshot_bucket: Final = snapshot.get(get_metadata_variable_name_from_kwargs(snapshot))
    model_info: Final = _routed_model_info(live_kwargs)
    if not isinstance(snapshot_bucket, dict) or model_info is None:
        return
    snapshot_bucket["model_info"] = dict(model_info)


DISABLE_FALLBACKS_METADATA_KEY: Final = "_disable_fallbacks"


def record_disable_fallbacks(request_kwargs: Mapping[str, object] | None, disabled: bool) -> None:
    """
    Write-or-clear the request's disable_fallbacks verdict into the router-internal metadata
    bucket. The wrapper pops the raw kwarg before any downstream frame runs, so the refusal
    gate (which decides whether to convert a refusal into a recoverable error) needs this
    carrier to know recovery is impossible.
    """
    from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs

    if request_kwargs is None:
        return
    bucket: Final = request_kwargs.get(get_metadata_variable_name_from_kwargs(request_kwargs))
    if not isinstance(bucket, dict):
        return
    if disabled:
        bucket[DISABLE_FALLBACKS_METADATA_KEY] = True
    else:
        bucket.pop(DISABLE_FALLBACKS_METADATA_KEY, None)


def fallbacks_disabled_for_request(kwargs: Mapping[str, object]) -> bool:
    """True when this request opted out of fallbacks, read from the raw kwarg (pre-pop
    snapshots keep it) or the router-internal bucket the wrapper stamps after popping it."""
    if kwargs.get("disable_fallbacks") is True:
        return True
    buckets: Final = (kwargs.get(name) for name in _ROUTER_METADATA_BUCKETS)
    return any(isinstance(bucket, dict) and bucket.get(DISABLE_FALLBACKS_METADATA_KEY) is True for bucket in buckets)


def fallback_lookup_groups(kwargs: Mapping[str, object], model_group: str | None) -> tuple[str, ...]:
    """
    Ordered keys for resolving a fallback chain: the tier a pre-routing hook selected wins,
    then the routed group, then the requested group, then the group the request was
    originally for. The routed group differs when Claude Code session affinity remaps a
    subagent's concrete model to its bound router. The original group differs on a fallback
    hop that fails after `run_async_fallback` already returned its stream: the hop has no
    chain of its own, so it resumes the original group's chain, and `attempted_targets` keeps
    the entries already tried from being repeated.
    """
    metadata: Final = kwargs.get(get_metadata_variable_name_from_kwargs(kwargs))
    routed_group_value: Final = metadata.get("model_group") if isinstance(metadata, Mapping) else None
    routed_group: Final = routed_group_value if isinstance(routed_group_value, str) else None
    original_group_value: Final = metadata.get("original_model_group") if isinstance(metadata, Mapping) else None
    original_group: Final = original_group_value if isinstance(original_group_value, str) else None
    ordered: Final = (get_pre_routing_selection(kwargs), routed_group, model_group, original_group)
    return tuple(dict.fromkeys(group for group in ordered if group))


def _resolved_a_specific_chain(
    fallbacks: list[Any],  # mutable-ok: mirrors get_fallback_model_group's contract
    result: tuple[list[str] | None, int | None],  # mutable-ok: mirrors get_fallback_model_group's contract
) -> bool:
    resolved, generic_idx = result
    if resolved is None:
        return False
    return generic_idx is None or resolved is not fallbacks[generic_idx]["*"]


def get_fallback_model_group_for_lookup_groups(
    fallbacks: list[Any],  # mutable-ok: mirrors get_fallback_model_group's contract
    lookup_groups: tuple[str, ...],
) -> tuple[list[str] | None, int | None]:  # mutable-ok: mirrors get_fallback_model_group's contract
    """
    First lookup group with a specifically-keyed chain wins; the generic "*" chain applies
    only after every group missed, so a catch-all cannot shadow a later group's own chain.
    """
    results: Final = tuple(get_fallback_model_group(fallbacks=fallbacks, model_group=group) for group in lookup_groups)
    specific: Final = next((result for result in results if _resolved_a_specific_chain(fallbacks, result)), None)
    if specific is not None:
        return specific
    return next((result for result in results if result[0] is not None), (None, None))


def fallback_entry_is_target(entry: object) -> bool:
    """
    A fallback entry that names a string ``model`` is one target, whatever the order of its keys, never a model group
    mapped to the fallbacks under it. The Router's fallback resolution and the proxy's fallback checks both read an
    entry by this, so they agree on which entries are targets.
    """
    return fallback_target_model_name(entry) is not None


def get_fallback_model_group(fallbacks: list[Any], model_group: str) -> tuple[list[str] | None, int | None]:
    """
    Returns:
    - fallback_model_group: List[str] of fallback model groups. example: ["gpt-4", "gpt-3.5-turbo"]
    - generic_fallback_idx: int of the index of the generic fallback in the fallbacks list.

    Checks:
    - exact match
    - stripped model group match
    - provider-prefixed model group match
    - generic fallback

    An entry that names a string model is a target, not a group map, so it matches no group here.
    """
    generic_fallback_idx: int | None = None
    stripped_model_fallback: list[str] | None = None
    fallback_model_group: list[str] | None = None
    fallback_keys: Final = tuple(
        next(iter(item)) for item in fallbacks if isinstance(item, dict) and item and not fallback_entry_is_target(item)
    )
    prefixed_model_group: Final = _provider_prefixed_model_group(model_group, fallback_keys)
    ## check for specific model group-specific fallbacks
    for idx, item in enumerate(fallbacks):
        if isinstance(item, dict) and not fallback_entry_is_target(item):
            fallback_key = next(iter(item))
            if fallback_key == model_group:  # check exact match
                fallback_model_group = item[model_group]
                break
            elif fallback_key == prefixed_model_group or _check_stripped_model_group(
                model_group=model_group, fallback_key=fallback_key
            ):  # check generic fallback
                stripped_model_fallback = item[fallback_key]
            elif fallback_key == "*":  # check generic fallback
                generic_fallback_idx = idx
        elif isinstance(item, str):
            fallback_model_group = [item]
    ## if none, check for generic fallback
    if fallback_model_group is None:
        if stripped_model_fallback is not None:
            fallback_model_group = stripped_model_fallback
        elif generic_fallback_idx is not None:
            fallback_model_group = fallbacks[generic_fallback_idx]["*"]

    return fallback_model_group, generic_fallback_idx


PROVIDER_SCOPED_RESOURCE_KEYS: Final = ("input_file_id", "training_file", "batch_id", "file_id", "fine_tuning_job_id")
PROVIDER_SCOPED_RESOURCE_FUNCTION_NAMES: Final = frozenset(
    {
        "_acreate_batch",
        "_acancel_batch",
        "acreate_fine_tuning_job",
        "acancel_fine_tuning_job",
        "aretrieve_fine_tuning_job",
        "afile_content",
        "afile_delete",
    }
)
PROVIDER_SCOPED_CREATION_FUNCTION_NAMES: Final = frozenset({"_acreate_file"})


def _get_fallback_target_model_group(fallback_entry: str | Mapping[str, object]) -> str | None:
    if isinstance(fallback_entry, str):
        return fallback_entry
    target: Final = fallback_entry.get("model")
    return target if isinstance(target, str) else None


async def _is_fallback_target_authorized(
    litellm_router: LitellmRouter,
    fallback_entry: str | Mapping[str, object],
    original_model_group: str,
    kwargs: Mapping[str, object],
) -> bool:
    access_check: Final = litellm_router.fallback_access_check
    target: Final = _get_fallback_target_model_group(fallback_entry)
    if (
        access_check is None
        or target is None
        or target == original_model_group
        or target == get_pre_routing_selection(kwargs)
    ):
        return True
    if await access_check(model=target, request_kwargs=kwargs, llm_router=litellm_router):
        return True
    verbose_router_logger.info(
        "Skipping fallback to model_group = %s: caller is not authorized to call it",
        mask_sensitive_structure(fallback_entry),
    )
    return False


async def _is_fallback_target_within_budget(
    litellm_router: LitellmRouter,
    fallback_entry: str | Mapping[str, object],
    original_model_group: str,
    kwargs: Mapping[str, object],
) -> bool:
    budget_check: Final = litellm_router.fallback_budget_check
    target: Final = _get_fallback_target_model_group(fallback_entry)
    if budget_check is None or target is None or target == original_model_group:
        return True
    if await budget_check(model=target, request_kwargs=kwargs, llm_router=litellm_router):
        return True
    verbose_router_logger.info(
        "Skipping fallback to model_group = %s: caller is over budget",
        mask_sensitive_structure(fallback_entry),
    )
    return False


def references_provider_scoped_resource(kwargs: Mapping[str, object]) -> bool:
    """
    True when a file, batch, or fine-tuning job operation names an id that only exists
    under one provider's credentials.

    Each of those ids lives in the account of the deployment that issued it. Handing it to
    a different model group asks a provider about an id it never issued, which costs an
    extra round trip that can only answer not-found. Generic calls dispatched through
    `Router._ageneric_api_call_with_fallbacks` carry the real handler in
    `original_generic_function`, so both slots are checked. Gating on the handler name
    keeps completion-style requests eligible for cross-group fallback even when a caller
    passes a stray extra body field that happens to share one of these key names.
    """
    handler_names: Final = tuple(
        getattr(kwargs.get(function_key), "__name__", None)
        for function_key in ("original_function", "original_generic_function")
    )
    if all(name not in PROVIDER_SCOPED_RESOURCE_FUNCTION_NAMES for name in handler_names):
        return False
    return any(kwargs.get(key) for key in PROVIDER_SCOPED_RESOURCE_KEYS)


def creates_provider_scoped_resource(kwargs: Mapping[str, object]) -> bool:
    """
    True when the request creates a resource that will live under one provider's credentials.

    A file uploaded for batches or fine-tuning is stored in the account of the deployment
    that handled it, and its id is only usable against the model group the caller named.
    Letting the upload fall back to a different model group silently stores the file with
    the wrong provider, and every later use of the returned id fails.
    """
    return getattr(kwargs.get("original_function"), "__name__", None) in PROVIDER_SCOPED_CREATION_FUNCTION_NAMES


async def run_async_fallback(
    *args: object,
    litellm_router: LitellmRouter,
    fallback_model_group: list[str],
    original_model_group: str,
    original_exception: Exception,
    max_fallbacks: int,
    fallback_depth: int,
    include_fallback_errors: bool = False,
    **kwargs,
) -> Any:
    """
    Loops through all the fallback model groups and calls kwargs["original_function"] with the arguments and keyword arguments provided.

    If the call is successful, it logs the success and returns the response.
    If the call fails, it logs the failure and continues to the next fallback model group.
    If all fallback model groups fail, it raises the most recent exception.

    Args:
        litellm_router: The litellm router instance.
        *args: Positional arguments.
        fallback_model_group: List[str] of fallback model groups. example: ["gpt-4", "gpt-3.5-turbo"]
        original_model_group: The original model group. example: "gpt-3.5-turbo"
        original_exception: The original exception.
        **kwargs: Keyword arguments. `attempted_targets` carries the fallback attempts
            already made for this request, created on the first hop and shared by reference
            for the rest of the walk. A target already in it is skipped, so neither a
            fallback graph that loops back on itself nor a client-side fallback list
            re-walked at each level can repeat an attempt that has already failed. Identity
            comes from `fallback_attempt_key`, so an entry that overrides request params or
            re-targets the failed group with a different deployment selection stays distinct
            from a bare name.

    Returns:
        The response from the successful fallback model group.
    Raises:
        The most recent exception if all fallback model groups fail.
    """

    ### BASE CASE ### MAX FALLBACK DEPTH REACHED
    if fallback_depth >= max_fallbacks:
        raise original_exception

    error_from_fallbacks = original_exception
    fallback_errors = (get_fallback_error_info(original_exception),)
    metadata_variable_name: Final = get_router_metadata_variable_name(
        function_name=getattr(kwargs.get("original_function"), "__name__", None)
    )
    same_model_group_only: Final = references_provider_scoped_resource(kwargs) or creates_provider_scoped_resource(
        kwargs
    )
    # Read out of kwargs and narrowed here rather than declared as a parameter: every caller
    # reaches this function by spreading a loosely-typed kwargs dict, so a declared parameter
    # would carry an annotation that no call site can actually be checked against.
    carried_targets: Final = kwargs.get("attempted_targets")
    attempted: Final = (
        carried_targets if isinstance(carried_targets, AttemptedFallbackTargets) else AttemptedFallbackTargets()
    )
    failed_model_group: Final = get_pre_routing_selection(kwargs) or original_model_group
    attempted.record(failed_model_group)

    # A dict target is merged straight into kwargs below, and kwargs win over the deployment's own params, so a target
    # could otherwise set a federation field, or a credential, endpoint or header that runs the hop under the
    # deployment's identity, whichever request field or setting the list came from. Checked here rather than at the
    # merge: inside the loop the refusal would be caught as a per-target failure and quietly skipped to the next one.
    credentials_allowed: Final[Callable[[], bool] | None] = getattr(
        litellm_router, "fallback_target_credentials_allowed", None
    )
    refuses_credentials: Final = credentials_allowed is None or not credentials_allowed()
    for target in fallback_model_group:
        if isinstance(target, dict):
            reject_server_owned_wif_params(target)
            reject_fallback_target_routing(target)
            if refuses_credentials:
                reject_fallback_target_credentials(target)

    for mg in fallback_model_group:
        if mg == failed_model_group:
            continue
        if same_model_group_only and _get_fallback_target_model_group(mg) != original_model_group:
            verbose_router_logger.info(
                "Skipping fallback to model_group = %s: request names a resource owned by model_group = %s",
                mask_sensitive_structure(mg),
                original_model_group,
            )
            continue
        if not await _is_fallback_target_authorized(litellm_router, mg, original_model_group, kwargs):
            continue
        if not await _is_fallback_target_within_budget(litellm_router, mg, original_model_group, kwargs):
            continue
        attempt_key = fallback_attempt_key(mg)
        if attempt_key is not None:
            if attempt_key in attempted:
                verbose_router_logger.info(
                    "Skipping fallback to model_group = %s, already attempted for this request",
                    mask_sensitive_structure(mg),
                )
                continue
            attempted.record(attempt_key)
        try:
            # LOGGING
            kwargs = litellm_router.log_retry(kwargs=kwargs, e=original_exception)
            verbose_router_logger.info("Falling back to model_group = %s", mask_sensitive_structure(mg))
            kwargs.pop("_target_order", None)
            if isinstance(mg, str):
                kwargs["model"] = mg
            elif isinstance(mg, dict):
                kwargs.update(fallback_target_params_over_call(kwargs, mg))
            fallback_depth = fallback_depth + 1
            _hop_metadata = dict(kwargs.get(metadata_variable_name) or {})
            _original_model_group_stamp = _hop_metadata.pop("original_model_group", original_model_group)
            _hop_metadata.pop("model_group", None)
            _hop_metadata.pop("attempted_fallbacks", None)
            _hop_metadata["original_model_group"] = _original_model_group_stamp
            _hop_metadata["model_group"] = kwargs.get("model", None)
            _hop_metadata["attempted_fallbacks"] = fallback_depth
            kwargs[metadata_variable_name] = _hop_metadata
            kwargs["fallback_depth"] = fallback_depth
            kwargs["max_fallbacks"] = max_fallbacks
            kwargs["attempted_targets"] = attempted
            if include_fallback_errors:
                kwargs["include_fallback_errors"] = include_fallback_errors
            response = await litellm_router.async_function_with_fallbacks(*args, **kwargs)
            verbose_router_logger.info("Successful fallback b/w models.")
            response = add_fallback_headers_to_response(
                response=response,
                attempted_fallbacks=fallback_depth,
                fallback_errors=(list(fallback_errors) if include_fallback_errors else None),
            )
            # callback for successfull_fallback_event():
            await log_success_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
            return response
        except Exception as e:
            error_from_fallbacks = e
            fallback_errors = fallback_errors + (get_fallback_error_info(e),)
            await log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
            logging_obj = kwargs.get("litellm_logging_obj")
            if logging_obj is not None and logging_obj.model_call_details.get("has_logged_async_failure", False):
                _trigger_cooldown_for_failed_deployment(
                    litellm_router=litellm_router,
                    kwargs=kwargs,
                    exception=e,
                    model_call_details=logging_obj.model_call_details,
                )
    raise error_from_fallbacks


async def log_success_fallback_event(original_model_group: str, kwargs: dict, original_exception: Exception):
    """
    Log a successful fallback event to all registered callbacks.

    Uses LoggingCallbackManager.get_custom_loggers_for_type() to get deduplicated
    CustomLogger instances from all callback lists.

    Args:
        original_model_group (str): The original model group before fallback.
        kwargs (dict): kwargs for the request

    Note:
        Errors during logging are caught and reported but do not interrupt the process.
    """
    # Get deduplicated CustomLogger instances from all callback lists
    custom_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)

    for _callback_custom_logger in custom_loggers:
        try:
            await _callback_custom_logger.log_success_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            verbose_router_logger.error("Error in log_success_fallback_event: %s", e)


async def log_failure_fallback_event(original_model_group: str, kwargs: dict, original_exception: Exception):
    """
    Log a failed fallback event to all registered callbacks.

    Uses LoggingCallbackManager.get_custom_loggers_for_type() to get deduplicated
    CustomLogger instances from all callback lists.

    Args:
        original_model_group (str): The original model group before fallback.
        kwargs (dict): kwargs for the request

    Note:
        Errors during logging are caught and reported but do not interrupt the process.
    """
    # Get deduplicated CustomLogger instances from all callback lists
    custom_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)

    for _callback_custom_logger in custom_loggers:
        try:
            await _callback_custom_logger.log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            verbose_router_logger.error("Error in log_failure_fallback_event: %s", e)


def check_non_standard_fallback_format(fallbacks: Sequence[object] | None) -> bool:
    """
    Checks if the fallbacks list is a list of strings or a list of dictionaries.

    If
    - List[str]: e.g. ["claude-3-haiku", "openai/o-1"]
    - List[Dict[<LiteLLMParamsTypedDict>, Any]]: e.g. [{"model": "claude-3-haiku", "messages": [{"role": "user", "content": "Hey, how's it going?"}]}]

    If [{"gpt-3.5-turbo": ["claude-3-haiku"]}] then standard format.
    """
    if fallbacks is None or not isinstance(fallbacks, list) or len(fallbacks) == 0:
        return False
    if all(isinstance(item, str) for item in fallbacks):
        return True
    dict_entries: Final = tuple(item for item in fallbacks if isinstance(item, dict))
    if len(dict_entries) == len(fallbacks):
        for item in dict_entries:
            for key in LiteLLMParamsTypedDict.__annotations__:
                if key in item:
                    # If the value is a list, it's likely a standard fallback model group mapping
                    # (e.g. {"model": ["backup"]}) rather than a parameter override.
                    if not isinstance(item[key], list):
                        return True

    return False


_check_non_standard_fallback_format = check_non_standard_fallback_format


def run_non_standard_fallback_format(fallbacks: Sequence[str] | Sequence[Mapping[str, object]], model_group: str):
    pass
