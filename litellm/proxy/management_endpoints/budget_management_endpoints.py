"""
BUDGET MANAGEMENT

All /budget management endpoints

/budget/new
/budget/info
/budget/update
/budget/delete
/budget/settings
/budget/list
"""

#### BUDGET TABLE MANAGEMENT ####
import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final, TypeVar

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.common_utils.user_api_key_cache import (
    default_end_user_budget_cache_key,
    end_user_cache_key,
    model_access_group_cache_key,
    project_cache_key,
    revoke_last_good,
    serves_last_good_entries,
    tag_cache_key,
    team_member_default_budget_cache_key,
    team_membership_auth_cache_key,
)
from litellm.proxy.management_endpoints.common_utils import (  # noqa: F401  # legacy module exports
    _user_has_admin_view,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    user_api_key_has_admin_view,
    validate_budget_duration,
)
from litellm.proxy.utils import PrismaClient, jsonify_object
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.organization_repository import OrganizationRepository
from litellm.repositories.project_repository import ProjectRepository
from litellm.repositories.table_repositories import (
    EndUserRepository,
    ModelAccessGroupBudgetRepository,
    TagRepository,
    TeamMembershipRepository,
)
from litellm.repositories.verification_token_repository import VerificationTokenRepository

router: Final = APIRouter()


@router.post(
    "/budget/new",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_budget(
    budget_obj: BudgetNewRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new budget object. Can apply this to teams, orgs, end-users, keys.

    Parameters:
    - budget_duration: Optional[str] - Budget reset period ("30d", "1h", etc.)
    - budget_id: Optional[str] - The id of the budget. If not provided, a new id will be generated.
    - max_budget: Optional[float] - The max budget for the budget.
    - soft_budget: Optional[float] - The soft budget for the budget.
    - max_parallel_requests: Optional[int] - The max number of parallel requests for the budget.
    - tpm_limit: Optional[int] - The tokens per minute limit for the budget.
    - rpm_limit: Optional[int] - The requests per minute limit for the budget.
    - tpd_limit: Optional[int] - The tokens per day limit for the budget. Charged by batch submissions instead of tpm_limit/rpm_limit.
    - model_max_budget: Optional[dict] - Specify max budget for a given model. Example: {"openai/gpt-4o-mini": {"max_budget": 100.0, "budget_duration": "1d", "tpm_limit": 100000, "rpm_limit": 100000}}
    - budget_reset_at: Optional[datetime] - Datetime when the initial budget is reset. Default is now.
    """
    from prisma.errors import UniqueViolationError

    from litellm.proxy.proxy_server import litellm_proxy_admin_name, prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    # Validate budget values are not negative
    if budget_obj.max_budget is not None and (not math.isfinite(budget_obj.max_budget) or budget_obj.max_budget < 0):
        raise HTTPException(
            status_code=400,
            detail={"error": f"max_budget must be a non-negative finite number. Received: {budget_obj.max_budget}"},
        )
    if budget_obj.soft_budget is not None and (not math.isfinite(budget_obj.soft_budget) or budget_obj.soft_budget < 0):
        raise HTTPException(
            status_code=400,
            detail={"error": f"soft_budget must be a non-negative finite number. Received: {budget_obj.soft_budget}"},
        )

    validate_budget_duration(budget_obj.budget_duration)

    # Validate model_max_budget if present
    if budget_obj.model_max_budget is not None and len(budget_obj.model_max_budget) > 0:
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            validate_model_max_budget,
        )

        try:
            validate_model_max_budget(budget_obj.model_max_budget)
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})

    # if no budget_reset_at date is set, but a budget_duration is given, then set budget_reset_at initially to the first completed duration interval in future
    if budget_obj.budget_reset_at is None and budget_obj.budget_duration is not None:
        budget_obj.budget_reset_at = get_budget_reset_time(budget_duration=budget_obj.budget_duration)

    budget_obj_json: Final = budget_obj.model_dump(exclude_none=True)
    budget_obj_jsonified: Final[dict[str, object]] = jsonify_object(budget_obj_json)  # mutable-ok: prisma create input
    try:
        response: Final = await BudgetRepository(prisma_client).table.create(
            data={
                **budget_obj_jsonified,
                "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
                "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
            }
        )
    except Exception as e:
        if not isinstance(e, UniqueViolationError):
            raise
        raise HTTPException(
            status_code=400,
            detail={"error": f"Budget with id '{budget_obj.budget_id}' already exists."},
        )

    return response


@router.post(
    "/budget/update",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_budget(
    budget_obj: BudgetNewRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update an existing budget object.

    Parameters:
    - budget_duration: Optional[str] - Budget reset period ("30d", "1h", etc.)
    - budget_id: Optional[str] - The id of the budget. If not provided, a new id will be generated.
    - max_budget: Optional[float] - The max budget for the budget.
    - soft_budget: Optional[float] - The soft budget for the budget.
    - max_parallel_requests: Optional[int] - The max number of parallel requests for the budget.
    - tpm_limit: Optional[int] - The tokens per minute limit for the budget.
    - rpm_limit: Optional[int] - The requests per minute limit for the budget.
    - tpd_limit: Optional[int] - The tokens per day limit for the budget. Charged by batch submissions instead of tpm_limit/rpm_limit.
    - model_max_budget: Optional[dict] - Specify max budget for a given model. Example: {"openai/gpt-4o-mini": {"max_budget": 100.0, "budget_duration": "1d", "tpm_limit": 100000, "rpm_limit": 100000}}
    - budget_reset_at: Optional[datetime] - Update the Datetime when the budget was last reset.
    """
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    if budget_obj.budget_id is None:
        raise HTTPException(status_code=400, detail={"error": "budget_id is required"})

    budget_obj_jsonified: Final = budget_update_data(budget_obj, user_api_key_dict)
    # Every holder's last good entry is revoked before the write, so a lowered limit cannot stay in one
    holders: Final = await read_budget_holders(budget_obj.budget_id)
    await revoke_last_good(user_api_key_cache, holders.last_good_keys)
    return await write_budget_update(budget_obj.budget_id, budget_obj_jsonified, holders)


def budget_update_data(budget_obj: BudgetNewRequest, user_api_key_dict: UserAPIKeyAuth) -> Mapping[str, object]:
    """The budget row columns ``/budget/update`` writes for ``budget_obj``. A value it refuses raises 400 before
    anything is read or written, so a caller that revokes before its first write validates with this first."""
    from litellm.proxy.proxy_server import litellm_proxy_admin_name

    # Validate budget values are not negative
    if budget_obj.max_budget is not None and (not math.isfinite(budget_obj.max_budget) or budget_obj.max_budget < 0):
        raise HTTPException(
            status_code=400,
            detail={"error": f"max_budget must be a non-negative finite number. Received: {budget_obj.max_budget}"},
        )
    if budget_obj.soft_budget is not None and (not math.isfinite(budget_obj.soft_budget) or budget_obj.soft_budget < 0):
        raise HTTPException(
            status_code=400,
            detail={"error": f"soft_budget must be a non-negative finite number. Received: {budget_obj.soft_budget}"},
        )

    validate_budget_duration(budget_obj.budget_duration)

    # Validate model_max_budget if present in update
    if budget_obj.model_max_budget is not None and len(budget_obj.model_max_budget) > 0:
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            validate_model_max_budget,
        )

        try:
            validate_model_max_budget(budget_obj.model_max_budget)
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})

    # recompute budget_reset_at when the duration changes, unless the caller pinned a reset time explicitly
    recomputed_reset_at: Final = (
        {"budget_reset_at": get_budget_reset_time(budget_duration=budget_obj.budget_duration)}
        if budget_obj.budget_duration is not None and "budget_reset_at" not in budget_obj.model_fields_set
        else MappingProxyType({"budget_reset_at": None})
        if "budget_duration" in budget_obj.model_fields_set
        and budget_obj.budget_duration is None
        and "budget_reset_at" not in budget_obj.model_fields_set
        else {}
    )

    budget_obj_jsonified: Final[Mapping[str, object]] = jsonify_object(
        {
            **budget_obj.model_dump(exclude_unset=True),
            **recomputed_reset_at,
            "updated_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
        }
    )
    return budget_obj_jsonified


async def write_budget_update(budget_id: str, data: Mapping[str, object], holders: "BudgetHolders"):
    """Write ``data`` from ``budget_update_data`` to budget ``budget_id``, then evict ``holders``. The caller read
    them with ``read_budget_holders`` and revoked their ``last_good_keys`` before the first write of its request."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    response: Final = await BudgetRepository(prisma_client).table.update(
        where={"budget_id": budget_id},
        data=data,
    )
    await evict_budget_holders(budget_id, holders)

    return response


_EvictionStep = TypeVar("_EvictionStep")


async def _best_effort(subject: str, what: str, step: Callable[[], Awaitable[_EvictionStep]]) -> _EvictionStep | None:
    try:
        return await step()
    except Exception as e:  # noqa: BLE001  # one failed read or eviction must not skip the others
        verbose_proxy_logger.warning(
            "Could not evict %s of %s; they may enforce the old limits until they expire: %s",
            what,
            subject,
            e,
        )
        return None


async def read_holders(
    subject: str, what: str, read: Callable[[], Awaitable[_EvictionStep]], *, required: bool
) -> _EvictionStep | None:
    """``required`` while last good entries are served: a holder that cannot be listed cannot be revoked, so its read
    failing refuses the change before anything is written. Otherwise best effort, as the eviction always was."""
    if not required:
        return await _best_effort(subject, what, read)
    try:
        return await read()
    except Exception as e:
        verbose_proxy_logger.error(
            "Refused to change %s: could not read its %s to revoke their last good auth entries: %s", subject, what, e
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": f"Could not read the {what} of {subject}, so their last good auth entries could not be "
                "revoked and nothing was changed. Retry once the database answers."
            },
        ) from e


@dataclass(frozen=True, slots=True)
class BudgetHolders:
    """The cached objects that carry a budget, read once before it changes, so that what is revoked before the write
    and what is evicted after it cannot drift apart."""

    # (user_id, team_id) of each team membership
    memberships: tuple[tuple[str, str], ...] = ()
    cache_keys: tuple[str, ...] = ()
    hashed_tokens: tuple[str, ...] = ()

    @property
    def last_good_keys(self) -> tuple[str, ...]:
        """Every key a last good entry of these holders is kept under."""
        return (
            *self.cache_keys,
            *(
                team_membership_auth_cache_key(team_id=team_id, user_id=user_id)
                for user_id, team_id in self.memberships
            ),
            *self.hashed_tokens,
        )


async def _team_member_keys(prisma_client: PrismaClient, memberships: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """Hashed tokens of the keys each ``(user_id, team_id)`` member holds in that team; a key's entry carries the
    rate limits of its member budget. One read per team, filtered by it, so a member's keys in other teams are not
    loaded."""
    members_by_team: Final[dict[str, list[str]]] = {}  # mutable-ok: grouped once, read below
    for user_id, team_id in dict.fromkeys(memberships):
        members_by_team.setdefault(team_id, []).append(user_id)
    table: Final = VerificationTokenRepository(prisma_client).table
    return tuple(
        [
            key.token
            for team_id, user_ids in members_by_team.items()
            for key in await find_many_in(table, "user_id", user_ids, where={"team_id": team_id})
        ]
    )


async def read_budget_holders(budget_id: str) -> BudgetHolders:
    """Keys, team memberships, organizations, projects, customers, tags and model access groups carry their budget row,
    a key also carries the budget of its organization and of its team membership, and the default customer and
    team-member budgets are cached by id. While last good entries are served, a holder table that cannot be read
    refuses the change with 503; otherwise each read stands on its own, so one unreadable table still leaves the
    other holders listed."""
    from litellm.proxy.auth.auth_checks import organization_cache_keys
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        return BudgetHolders()
    subject: Final = f"budget {budget_id}"
    required: Final = serves_last_good_entries(user_api_key_cache)
    on_budget: Final = {"budget_id": budget_id}
    memberships, organizations, projects, customers, tags, access_groups = await asyncio.gather(
        read_holders(
            subject,
            "team memberships",
            lambda: TeamMembershipRepository(prisma_client).table.find_many(where=on_budget),
            required=required,
        ),
        read_holders(
            subject,
            "organizations",
            lambda: OrganizationRepository(prisma_client).table.find_many(where=on_budget),
            required=required,
        ),
        read_holders(
            subject,
            "projects",
            lambda: ProjectRepository(prisma_client).table.find_many(where=on_budget),
            required=required,
        ),
        read_holders(
            subject,
            "customers",
            lambda: EndUserRepository(prisma_client).table.find_many(where=on_budget),
            required=required,
        ),
        read_holders(
            subject, "tags", lambda: TagRepository(prisma_client).table.find_many(where=on_budget), required=required
        ),
        read_holders(
            subject,
            "model access groups",
            lambda: ModelAccessGroupBudgetRepository(prisma_client).table.find_many(where=on_budget),
            required=required,
        ),
    )
    member_pairs: Final = tuple(dict.fromkeys((m.user_id, m.team_id) for m in memberships or ()))
    keys_on_budget, keys_of_its_organizations, keys_of_its_members = await asyncio.gather(
        read_holders(
            subject,
            "keys",
            lambda: VerificationTokenRepository(prisma_client).table.find_many(where=on_budget),
            required=required,
        ),
        read_holders(
            subject,
            "keys of its organizations",
            lambda: find_many_in(
                VerificationTokenRepository(prisma_client).table,
                "organization_id",
                tuple(org.organization_id for org in organizations or ()),
            ),
            required=required,
        ),
        read_holders(
            subject,
            "keys of its team members",
            # Only the revocation needs them: invalidate_team_member_spend_state evicts a member's keys after the write
            lambda: _team_member_keys(prisma_client, member_pairs if required else ()),
            required=required,
        ),
    )
    return BudgetHolders(
        memberships=member_pairs,
        cache_keys=(
            default_end_user_budget_cache_key(budget_id),
            team_member_default_budget_cache_key(budget_id),
            *chain.from_iterable(organization_cache_keys(org.organization_id) for org in organizations or ()),
            *(project_cache_key(project.project_id) for project in projects or ()),
            *(end_user_cache_key(customer.user_id) for customer in customers or ()),
            *(tag_cache_key(tag.tag_name) for tag in tags or ()),
            *(model_access_group_cache_key(group.access_group_name) for group in access_groups or ()),
        ),
        hashed_tokens=tuple(
            dict.fromkeys(
                (
                    *(key.token for key in keys_on_budget or ()),
                    *(key.token for key in keys_of_its_organizations or ()),
                    *(keys_of_its_members or ()),
                )
            )
        ),
    )


async def read_team_member_holders(team_id: str, user_ids: Sequence[str]) -> BudgetHolders:
    """The entries that carry these members' budgets in ``team_id``: each membership and, while last good entries are
    served, the keys it holds in the team, read like ``read_budget_holders``."""
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    memberships: Final = tuple(dict.fromkeys((user_id, team_id) for user_id in user_ids))
    required: Final = serves_last_good_entries(user_api_key_cache)
    if prisma_client is None or not memberships or not required:
        return BudgetHolders(memberships=memberships)
    keys: Final = await read_holders(
        f"team {team_id}", "keys of its members", lambda: _team_member_keys(prisma_client, memberships), required=True
    )
    return BudgetHolders(memberships=memberships, hashed_tokens=keys or ())


async def evict_budget_holders(budget_id: str, holders: BudgetHolders) -> None:
    """Evict every cached copy of the holders, last good entries included, once their budget changed. Each eviction
    stands on its own and is best effort, like every eviction after a committed write."""
    from litellm.proxy.auth.auth_checks import delete_cache_key_objects, invalidate_team_member_spend_state
    from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import evict_and_broadcast
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    subject: Final = f"budget {budget_id}"
    for user_id, team_id in holders.memberships:
        await _best_effort(
            subject,
            f"the spend state of team member {user_id}",
            lambda u=user_id, t=team_id: invalidate_team_member_spend_state(
                user_id=u, team_id=t, user_api_key_cache=user_api_key_cache
            ),
        )
    await _best_effort(
        subject,
        "the cached budget holders",
        lambda: evict_and_broadcast(cache_keys=holders.cache_keys, user_api_key_cache=user_api_key_cache),
    )
    await _best_effort(
        subject,
        "the cached keys",
        lambda: delete_cache_key_objects(
            hashed_tokens=holders.hashed_tokens,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        ),
    )


@router.post(
    "/budget/info",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def info_budget(data: BudgetRequest):
    """
    Get the budget id specific information

    Parameters:
    - budgets: List[str] - The list of budget ids to get information for
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No db connected"})

    if len(data.budgets) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Specify list of budget id's to query. Passed in={data.budgets}"},
        )
    response: Final = await BudgetRepository(prisma_client).table.find_many(
        where={"budget_id": {"in": data.budgets}},
    )

    return response


@router.get(
    "/budget/settings",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def budget_settings(
    budget_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get list of configurable params + current value for a budget item + description of each field

    Used on Admin UI.

    Query Parameters:
    - budget_id: str - The budget id to get information for
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if not user_api_key_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=400,
            detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
        )

    ## get budget item from db
    db_budget_row: Final = await BudgetRepository(prisma_client).table.find_first(where={"budget_id": budget_id})

    if db_budget_row is not None:
        db_budget_row_dict = db_budget_row.model_dump(exclude_none=True)
    else:
        db_budget_row_dict = {}

    allowed_args: Final = {
        "max_parallel_requests": {"type": "Integer"},
        "tpm_limit": {"type": "Integer"},
        "rpm_limit": {"type": "Integer"},
        "tpd_limit": {"type": "Integer"},
        "budget_duration": {"type": "String"},
        "max_budget": {"type": "Float"},
        "soft_budget": {"type": "Float"},
        "model_max_budget": {"type": "Object"},
    }

    return_val: Final = []

    for field_name, field_info in BudgetNewRequest.model_fields.items():
        if field_name in allowed_args:
            _stored_in_db = True

            _response_obj = ConfigList(
                field_name=field_name,
                field_type=allowed_args[field_name]["type"],
                field_description=field_info.description or "",
                field_value=db_budget_row_dict.get(field_name, None),
                stored_in_db=_stored_in_db,
                field_default_value=field_info.default,
            )
            return_val.append(_response_obj)

    return return_val


@router.get(
    "/budget/list",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_budget(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List all the created budgets in proxy db. Used on Admin UI."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=400,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if not user_api_key_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=400,
            detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
        )

    response: Final = await BudgetRepository(prisma_client).table.find_many()

    return response


@router.post(
    "/budget/delete",
    tags=["budget management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_budget(
    data: BudgetDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete budget

    Parameters:
    - id: str - The budget id to delete
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=400,
            detail={"error": f"{CommonProxyErrors.not_allowed_access.value}, your role={user_api_key_dict.user_role}"},
        )

    response: Final = await BudgetRepository(prisma_client).table.delete(where={"budget_id": data.id})

    return response
