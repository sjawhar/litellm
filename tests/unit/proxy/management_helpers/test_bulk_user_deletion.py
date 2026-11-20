import asyncio
import copy
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.constants import LAST_GOOD_TOMBSTONE_TTL_SECONDS
from litellm.proxy._types import LiteLLM_TeamTable, LitellmTableNames, LitellmUserRoles, Member, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import jwt_key_mapping_cache_key
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, team_membership_auth_cache_key
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.list_api.common import ManagementProblem
from litellm.proxy.management_helpers.bulk_user_deletion import bulk_delete_users, bulk_remove_team_members
from litellm.types.proxy.management_endpoints.internal_user_endpoints import BulkDeleteUserRequest
from litellm.types.proxy.management_endpoints.team_endpoints import BulkTeamMemberDeleteRequest, TeamMemberRef

if TYPE_CHECKING:
    from tests.unit.caching.fake_redis_cache import FakeRedisCache

ADMIN: Final = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin")
INTERNAL: Final = UserAPIKeyAuth(user_id="someone", user_role=LitellmUserRoles.INTERNAL_USER)
ORG_ADMIN: Final = UserAPIKeyAuth(user_id="org-admin", user_role=LitellmUserRoles.ORG_ADMIN)


class _UserRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_id: str
    user_email: str | None = None
    teams: list[str] = []


class _Record(BaseModel):
    """Attribute access like a Prisma row, over whatever columns the test seeded."""

    model_config = ConfigDict(extra="allow")


def _in(where: Mapping[str, object], field: str) -> set[str] | None:
    clause = where.get(field)
    if isinstance(clause, dict) and "in" in clause:
        return set(clause["in"])
    if isinstance(clause, str):
        return {clause}
    return None


def _matches(row: Mapping[str, object], where: Mapping[str, object]) -> bool:
    if "OR" in where:
        return any(_matches(row, clause) for clause in where["OR"])
    return all((wanted := _in(where, field)) is not None and row.get(field) in wanted for field in where)


class _Rows:
    """A list-backed Prisma table supporting the `in`/equality/OR filters the helper issues. It has `create` because a
    `RoutingPrismaWrapper` routes reads only for a table that has `find_many` and `create`."""

    def __init__(self, rows: Sequence[Mapping[str, object]] = ()) -> None:
        self.rows: list[dict[str, object]] = [dict(r) for r in rows]

    async def find_many(self, where: Mapping[str, object]) -> list[_Record]:
        return [_Record.model_validate(r) for r in self.rows if _matches(r, where)]

    async def delete_many(self, where: Mapping[str, object]) -> int:
        before = len(self.rows)
        self.rows = [r for r in self.rows if not _matches(r, where)]
        return before - len(self.rows)

    async def create_many(self, data: Sequence[Mapping[str, object]]) -> int:
        self.rows.extend(dict(r) for r in data)
        return len(data)

    async def create(self, data: Mapping[str, object]) -> _Record:
        self.rows.append(dict(data))
        return _Record.model_validate(dict(data))


class _UserTable:
    def __init__(self, users: Sequence[_UserRow]) -> None:
        self.rows: dict[str, _UserRow] = {u.user_id: u for u in users}

    async def find_many(self, where: Mapping[str, object]) -> list[_UserRow]:
        return [u for u in self.rows.values() if _matches(u.model_dump(), where)]

    async def update(self, where: Mapping[str, str], data: Mapping[str, Mapping[str, Sequence[str]]]) -> _UserRow:
        row = self.rows[where["user_id"]]
        updated = row.model_copy(update={"teams": list(data["teams"]["set"])})
        self.rows[row.user_id] = updated
        return updated

    async def delete_many(self, where: Mapping[str, object]) -> int:
        doomed = [uid for uid, u in self.rows.items() if _matches(u.model_dump(), where)]
        for uid in doomed:
            del self.rows[uid]
        return len(doomed)

    async def create(self, data: Mapping[str, object]) -> _UserRow:
        row = _UserRow.model_validate(dict(data))
        self.rows[row.user_id] = row
        return row


class _TeamTable:
    def __init__(self, teams: Sequence[LiteLLM_TeamTable]) -> None:
        self.rows: dict[str, LiteLLM_TeamTable] = {t.team_id: t for t in teams}
        self.update_calls = 0

    async def find_unique(self, where: Mapping[str, str]) -> LiteLLM_TeamTable | None:
        return self.rows.get(where["team_id"])

    async def find_many(self, where: Mapping[str, object]) -> list[LiteLLM_TeamTable]:
        return [t for t in self.rows.values() if _matches({"team_id": t.team_id}, where)]

    async def update(self, where: Mapping[str, str], data: Mapping[str, str]) -> LiteLLM_TeamTable:
        self.update_calls += 1
        team = self.rows[where["team_id"]]
        team.members_with_roles = [Member(**m) for m in json.loads(data["members_with_roles"])]
        return team

    async def create(self, data: Mapping[str, object]) -> LiteLLM_TeamTable:
        team = LiteLLM_TeamTable.model_validate(dict(data))
        self.rows[team.team_id] = team
        return team


class _Db:
    def __init__(
        self,
        users: Sequence[_UserRow],
        teams: Sequence[LiteLLM_TeamTable],
        memberships: Sequence[tuple[str, str]] = (),
        tokens: Sequence[Mapping[str, object]] = (),
        invitations: Sequence[Mapping[str, object]] = (),
        org_memberships: Sequence[Mapping[str, object]] = (),
        jwt_mappings: Sequence[Mapping[str, object]] = (),
    ) -> None:
        self.litellm_usertable = _UserTable(users)
        self.litellm_teamtable = _TeamTable(teams)
        self.litellm_teammembership = _Rows([{"team_id": t, "user_id": u} for t, u in memberships])
        self.litellm_verificationtoken = _Rows(tokens)
        self.litellm_deletedverificationtoken = _Rows()
        self.litellm_invitationlink = _Rows(invitations)
        self.litellm_organizationmembership = _Rows(org_memberships)
        self.litellm_jwtkeymapping = _Rows(jwt_mappings)


class _Tx:
    def __init__(self, db: _Db, on_lock: Callable[[str], None], fail_locks: frozenset[str]) -> None:
        self.litellm_teamtable = db.litellm_teamtable
        self.litellm_usertable = db.litellm_usertable
        self.litellm_teammembership = db.litellm_teammembership
        self.litellm_verificationtoken = db.litellm_verificationtoken
        self.litellm_deletedverificationtoken = db.litellm_deletedverificationtoken
        self.litellm_invitationlink = db.litellm_invitationlink
        self.litellm_organizationmembership = db.litellm_organizationmembership
        self.litellm_jwtkeymapping = db.litellm_jwtkeymapping
        self._on_lock = on_lock
        self._fail_locks = fail_locks
        self.locks: list[str] = []
        self.roster_reads: list[str] = []

    async def query_raw(self, sql: str, *args: object) -> list[dict[str, object]]:
        team_id = str(args[0])
        if "pg_advisory_xact_lock" in sql:
            if team_id in self._fail_locks:
                raise RuntimeError("lock timeout")
            self.locks.append(team_id)
            self._on_lock(team_id)
            return []
        assert team_id in self.locks, "roster must be read under this team's advisory lock"
        self.roster_reads.append(team_id)
        team = self.litellm_teamtable.rows.get(team_id)
        if team is None:
            return []
        return [{"members_with_roles": json.dumps([m.model_dump() for m in team.members_with_roles])}]


class _FakePrisma:
    def __init__(
        self,
        users: Sequence[_UserRow] = (),
        teams: Sequence[LiteLLM_TeamTable] = (),
        memberships: Sequence[tuple[str, str]] = (),
        tokens: Sequence[Mapping[str, object]] = (),
        invitations: Sequence[Mapping[str, object]] = (),
        org_memberships: Sequence[Mapping[str, object]] = (),
        jwt_mappings: Sequence[Mapping[str, object]] = (),
        on_lock: Callable[[str], None] = lambda _: None,
        fail_locks: frozenset[str] = frozenset(),
        fail_commit: bool = False,
    ) -> None:
        self.db = _Db(users, teams, memberships, tokens, invitations, org_memberships, jwt_mappings)
        self._on_lock = on_lock
        self._fail_locks = fail_locks
        self._fail_commit = fail_commit
        self.locks: list[str] = []
        self.roster_reads: list[str] = []

    @asynccontextmanager
    async def tx(self, *, timeout: object = None):
        snapshot = copy.deepcopy(self.db)
        tx = _Tx(self.db, self._on_lock, self._fail_locks)
        try:
            yield tx
            if self._fail_commit:
                raise RuntimeError("connection reset")
        except BaseException:
            self.db.__dict__.update(snapshot.__dict__)
            raise
        self.locks.extend(tx.locks)
        self.roster_reads.extend(tx.roster_reads)


def _team(team_id: str, *members: str, org: str | None = None) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(
        team_id=team_id,
        organization_id=org,
        members_with_roles=[Member(user_id=m, user_email=f"{m}@example.com", role="user") for m in members],
    )


def _user(user_id: str, *teams: str) -> _UserRow:
    return _UserRow(user_id=user_id, user_email=f"{user_id}@example.com", teams=list(teams))


def _roster(prisma: _FakePrisma, team_id: str) -> list[str | None]:
    return [m.user_id for m in prisma.db.litellm_teamtable.rows[team_id].members_with_roles]


def _cache_with(*hashed_tokens: str) -> UserApiKeyCache:
    cache = UserApiKeyCache()
    for token in hashed_tokens:
        cache.set_cache(key=token, value=UserAPIKeyAuth(token=token))
    return cache


def _jwt_mapping(token: str, claim_value: str, issuer: str | None = None) -> Mapping[str, object]:
    return {"token": token, "jwt_claim_name": "sub", "jwt_claim_value": claim_value, "jwt_issuer": issuer}


def _cache_with_jwt_mapping_keys(*cache_keys: str) -> UserApiKeyCache:
    cache = UserApiKeyCache()
    for key in cache_keys:
        cache.set_cache(key=key, value={"cache_key": key})
    return cache


async def _delete(
    prisma: "_FakePrisma | _ReplicatedPrisma | _OneConnectionPrisma",
    user_ids: Sequence[str],
    caller: UserAPIKeyAuth = ADMIN,
    cache: UserApiKeyCache | None = None,
):
    return await bulk_delete_users(
        data=BulkDeleteUserRequest(user_ids=tuple(user_ids)),
        user_api_key_dict=caller,
        prisma_client=prisma,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
        user_api_key_cache=cache or UserApiKeyCache(),
        proxy_logging_obj=None,
        litellm_proxy_admin_name="default_user_id",
        litellm_changed_by=None,
    )


async def _remove(
    prisma: "_FakePrisma | _OneConnectionPrisma",
    team_id: str,
    members: Sequence[Mapping[str, str]],
    caller: UserAPIKeyAuth = ADMIN,
    cache: UserApiKeyCache | None = None,
):
    return await bulk_remove_team_members(
        team_id=team_id,
        data=BulkTeamMemberDeleteRequest(members=tuple(TeamMemberRef(**m) for m in members)),
        user_api_key_dict=caller,
        prisma_client=prisma,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
        user_api_key_cache=cache or UserApiKeyCache(),
        proxy_logging_obj=None,
    )


@pytest.mark.asyncio
async def test_bulk_delete_removes_users_from_every_team_and_store():
    prisma = _FakePrisma(
        users=[_user("u1", "t1", "t2"), _user("u2", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "u2", "keep"), _team("t2", "u1", "other")],
        memberships=[("t1", "u1"), ("t2", "u1"), ("t1", "u2"), ("t1", "keep")],
        tokens=[{"token": "k1", "user_id": "u1", "team_id": "t1"}, {"token": "k2", "user_id": "keep"}],
        invitations=[
            {"id": "i1", "user_id": "u2", "created_by": "admin", "updated_by": "admin"},
            {"id": "i2", "user_id": "keep", "created_by": "u1", "updated_by": "admin"},
            {"id": "i3", "user_id": "keep", "created_by": "admin", "updated_by": "admin"},
        ],
        org_memberships=[{"user_id": "u1", "organization_id": "o1", "user_role": "internal_user"}],
    )

    results = await _delete(prisma, ["u1", "u2"])

    assert len(results) == 2
    assert [(r.user_id, r.user_email, r.success, r.teams_removed) for r in results] == [
        ("u1", "u1@example.com", True, ("t1", "t2")),
        ("u2", "u2@example.com", True, ("t1",)),
    ]
    assert _roster(prisma, "t1") == ["keep"] and _roster(prisma, "t2") == ["other"]
    assert set(prisma.db.litellm_usertable.rows) == {"keep"}
    assert prisma.db.litellm_teammembership.rows == [{"team_id": "t1", "user_id": "keep"}]
    assert [t["token"] for t in prisma.db.litellm_verificationtoken.rows] == ["k2"]
    assert [t["token"] for t in prisma.db.litellm_deletedverificationtoken.rows] == ["k1"]
    assert [i["id"] for i in prisma.db.litellm_invitationlink.rows] == ["i3"]
    assert prisma.db.litellm_organizationmembership.rows == []
    assert prisma.locks == ["t1", "t2"] and prisma.roster_reads == ["t1", "t2"]


@pytest.mark.asyncio
async def test_bulk_delete_leaves_teammates_who_share_the_deleted_users_email_alone():
    twin = _UserRow(user_id="twin", user_email="u1@example.com", teams=["t1"])
    team = LiteLLM_TeamTable(
        team_id="t1",
        members_with_roles=[
            Member(user_id="u1", user_email="u1@example.com", role="user"),
            Member(user_id="twin", user_email="u1@example.com", role="user"),
        ],
    )
    prisma = _FakePrisma(
        users=[_user("u1", "t1"), twin],
        teams=[team],
        memberships=[("t1", "u1"), ("t1", "twin")],
        tokens=[
            {"token": "k1", "user_id": "u1", "team_id": "t1"},
            {"token": "k-twin", "user_id": "twin", "team_id": "t1"},
        ],
    )

    results = await _delete(prisma, ["u1"])

    assert [(r.success, r.teams_removed) for r in results] == [(True, ("t1",))]
    assert _roster(prisma, "t1") == ["twin"]
    assert set(prisma.db.litellm_usertable.rows) == {"twin"} and prisma.db.litellm_usertable.rows["twin"].teams == [
        "t1"
    ]
    assert prisma.db.litellm_teammembership.rows == [{"team_id": "t1", "user_id": "twin"}]
    assert [t["token"] for t in prisma.db.litellm_verificationtoken.rows] == ["k-twin"]


@pytest.mark.asyncio
async def test_bulk_delete_removes_the_deleted_users_email_only_roster_entry():
    team = LiteLLM_TeamTable(
        team_id="t1",
        members_with_roles=[
            Member(user_id=None, user_email="u1@example.com", role="user"),
            Member(user_id="keep", user_email="keep@example.com", role="user"),
        ],
    )
    prisma = _FakePrisma(users=[_user("u1", "t1"), _user("keep", "t1")], teams=[team])

    results = await _delete(prisma, ["u1"])

    assert [(r.success, r.teams_removed) for r in results] == [(True, ("t1",))]
    assert _roster(prisma, "t1") == ["keep"]
    assert set(prisma.db.litellm_usertable.rows) == {"keep"}


@pytest.mark.asyncio
async def test_bulk_delete_finds_teams_through_membership_rows_when_user_teams_array_is_stale():
    prisma = _FakePrisma(
        users=[_user("u1")],
        teams=[_team("t1", "u1", "keep")],
        memberships=[("t1", "u1")],
    )

    results = await _delete(prisma, ["u1"])

    assert results[0].teams_removed == ("t1",)
    assert _roster(prisma, "t1") == ["keep"]
    assert prisma.db.litellm_teammembership.rows == []


@pytest.mark.asyncio
async def test_bulk_delete_reads_roster_under_lock_so_a_concurrent_add_survives():
    team = _team("t1", "u1")

    def concurrent_member_add(team_id: str) -> None:
        team.members_with_roles.append(Member(user_id="late", role="user"))

    prisma = _FakePrisma(users=[_user("u1", "t1")], teams=[team], on_lock=concurrent_member_add)

    results = await _delete(prisma, ["u1"])

    assert results[0].success is True
    assert _roster(prisma, "t1") == ["late"]


@pytest.mark.asyncio
async def test_bulk_delete_reports_missing_and_duplicate_ids_per_item_and_still_deletes_the_rest():
    prisma = _FakePrisma(users=[_user("u1")])

    results = await _delete(prisma, ["u1", "ghost", "u1"])

    assert [r.success for r in results].count(True) == 1
    assert [(r.user_id, r.success, r.error) for r in results] == [
        ("u1", True, None),
        ("ghost", False, "User id=ghost not found"),
        ("u1", False, "Duplicate user_id in request: u1"),
    ]
    assert prisma.db.litellm_usertable.rows == {}


@pytest.mark.asyncio
async def test_bulk_delete_rolls_back_every_team_and_user_when_one_team_rewrite_fails():
    prisma = _FakePrisma(
        users=[_user("u1", "a-good", "z-bad"), _user("u2", "a-good")],
        teams=[_team("a-good", "u1", "u2"), _team("z-bad", "u1")],
        tokens=[{"token": "k1", "user_id": "u1", "team_id": "a-good"}],
        fail_locks=frozenset({"z-bad"}),
    )
    cache = _cache_with("k1")

    results = await _delete(prisma, ["u1", "u2"], cache=cache)

    assert [(r.user_id, r.success, r.teams_removed, r.error) for r in results] == [
        ("u1", False, (), "Failed to delete user: lock timeout"),
        ("u2", False, (), "Failed to delete user: lock timeout"),
    ]
    assert set(prisma.db.litellm_usertable.rows) == {"u1", "u2"}
    assert _roster(prisma, "a-good") == ["u1", "u2"] and _roster(prisma, "z-bad") == ["u1"]
    assert [t["token"] for t in prisma.db.litellm_verificationtoken.rows] == ["k1"]
    assert cache.get_cache(key="k1") is not None


@pytest.mark.asyncio
async def test_bulk_delete_skips_teams_the_user_still_names_but_which_no_longer_exist():
    prisma = _FakePrisma(users=[_user("u1", "gone", "t1")], teams=[_team("t1", "u1", "keep")])

    results = await _delete(prisma, ["u1"])

    assert [(r.success, r.teams_removed) for r in results] == [(True, ("t1",))]
    assert prisma.db.litellm_usertable.rows == {} and _roster(prisma, "t1") == ["keep"]
    assert prisma.locks == ["t1"]


@pytest.mark.asyncio
async def test_bulk_delete_rolls_back_every_user_row_and_reports_it_per_row_when_the_delete_fails():
    prisma = _FakePrisma(
        users=[_user("u1", "t1"), _user("u2")],
        teams=[_team("t1", "u1")],
        tokens=[{"token": "k1", "user_id": "u1"}],
        fail_commit=True,
    )
    cache = _cache_with("k1")

    results = await _delete(prisma, ["u1", "u2", "ghost"], cache=cache)

    assert [(r.user_id, r.success, r.error) for r in results] == [
        ("u1", False, "Failed to delete user: connection reset"),
        ("u2", False, "Failed to delete user: connection reset"),
        ("ghost", False, "User id=ghost not found"),
    ]
    assert set(prisma.db.litellm_usertable.rows) == {"u1", "u2"}
    assert [t["token"] for t in prisma.db.litellm_verificationtoken.rows] == ["k1"]
    assert prisma.db.litellm_deletedverificationtoken.rows == []
    assert cache.get_cache(key="k1") is not None


@pytest.mark.asyncio
async def test_bulk_delete_evicts_deleted_keys_and_users_from_the_auth_cache():
    prisma = _FakePrisma(
        users=[_user("u1", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "keep")],
        tokens=[
            {"token": "team-key", "user_id": "u1", "team_id": "t1"},
            {"token": "personal-key", "user_id": "u1"},
            {"token": "keep-key", "user_id": "keep", "team_id": "t1"},
        ],
    )
    cache = _cache_with("team-key", "personal-key", "keep-key")
    cache.set_cache(key="u1", value={"user_id": "u1"})

    await _delete(prisma, ["u1"], cache=cache)

    assert cache.get_cache(key="team-key") is None and cache.get_cache(key="personal-key") is None
    assert cache.get_cache(key="u1") is None
    assert cache.get_cache(key="keep-key") is not None


@pytest.mark.asyncio
async def test_bulk_delete_evicts_jwt_key_mappings_of_the_deleted_users_keys():
    issuer: Final = "https://issuer.example"
    doomed_global: Final = jwt_key_mapping_cache_key("sub", "alice")
    doomed_scoped: Final = jwt_key_mapping_cache_key("sub", "alice", issuer)
    kept: Final = jwt_key_mapping_cache_key("sub", "bob")
    prisma = _FakePrisma(
        users=[_user("u1", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "keep")],
        tokens=[
            {"token": "team-key", "user_id": "u1", "team_id": "t1"},
            {"token": "personal-key", "user_id": "u1"},
            {"token": "keep-key", "user_id": "keep", "team_id": "t1"},
        ],
        jwt_mappings=[
            _jwt_mapping("personal-key", "alice"),
            _jwt_mapping("team-key", "alice", issuer=issuer),
            _jwt_mapping("keep-key", "bob"),
        ],
    )
    cache = _cache_with_jwt_mapping_keys(doomed_global, doomed_scoped, kept)

    await _delete(prisma, ["u1"], cache=cache)

    assert cache.get_cache(key=doomed_global) is None and cache.get_cache(key=doomed_scoped) is None
    assert cache.get_cache(key=kept) is not None


@pytest.mark.asyncio
async def test_bulk_delete_rejects_non_admin_callers_before_touching_the_db():
    prisma = _FakePrisma(users=[_user("u1")])

    with pytest.raises(ManagementProblem) as exc:
        await _delete(prisma, ["u1"], caller=INTERNAL)

    assert exc.value.problem.status == 403
    assert set(prisma.db.litellm_usertable.rows) == {"u1"}


@pytest.mark.asyncio
async def test_org_admin_deletes_only_users_fully_inside_their_orgs():
    prisma = _FakePrisma(
        users=[_user("inside"), _user("straddles"), _user("orgless")],
        org_memberships=[
            {"user_id": "org-admin", "organization_id": "o1", "user_role": LitellmUserRoles.ORG_ADMIN.value},
            {"user_id": "inside", "organization_id": "o1", "user_role": "internal_user"},
            {"user_id": "straddles", "organization_id": "o1", "user_role": "internal_user"},
            {"user_id": "straddles", "organization_id": "o2", "user_role": "internal_user"},
        ],
    )

    results = await _delete(prisma, ["inside", "straddles", "orgless"], caller=ORG_ADMIN)

    assert [r.success for r in results] == [True, False, False]
    assert all("not within your admin scope" in (r.error or "") for r in results[1:])
    assert set(prisma.db.litellm_usertable.rows) == {"straddles", "orgless"}
    assert {(m["user_id"], m["organization_id"]) for m in prisma.db.litellm_organizationmembership.rows} == {
        ("org-admin", "o1"),
        ("straddles", "o1"),
        ("straddles", "o2"),
    }


@pytest.mark.asyncio
async def test_bulk_member_delete_removes_by_id_and_email_and_keeps_the_rest():
    prisma = _FakePrisma(
        users=[_user("u1", "t1", "t2"), _user("u2", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "u2", "keep")],
        memberships=[("t1", "u1"), ("t1", "u2"), ("t1", "keep")],
        tokens=[
            {"token": "team-key", "user_id": "u1", "team_id": "t1"},
            {"token": "other-team-key", "user_id": "u1", "team_id": "t2"},
            {"token": "keep-key", "user_id": "keep", "team_id": "t1"},
        ],
    )

    results = await _remove(prisma, "t1", [{"user_id": "u1"}, {"user_email": "u2@example.com"}])

    assert [(r.user_id, r.user_email, r.success) for r in results] == [
        ("u1", None, True),
        (None, "u2@example.com", True),
    ]
    assert _roster(prisma, "t1") == ["keep"]
    users = prisma.db.litellm_usertable.rows
    assert users["u1"].teams == ["t2"] and users["u2"].teams == [] and users["keep"].teams == ["t1"]
    assert prisma.db.litellm_teammembership.rows == [{"team_id": "t1", "user_id": "keep"}]
    assert sorted(t["token"] for t in prisma.db.litellm_verificationtoken.rows) == ["keep-key", "other-team-key"]
    assert [t["token"] for t in prisma.db.litellm_deletedverificationtoken.rows] == ["team-key"]
    assert prisma.locks == ["t1"] and prisma.roster_reads == ["t1"]


@pytest.mark.asyncio
async def test_bulk_member_delete_reports_members_not_on_the_team_without_rewriting_the_roster():
    prisma = _FakePrisma(users=[_user("u1", "t1"), _user("elsewhere")], teams=[_team("t1", "u1")])

    results = await _remove(prisma, "t1", [{"user_id": "elsewhere"}, {"user_email": "nobody@example.com"}])

    assert [(r.success, r.error) for r in results] == [
        (False, "User not found in team"),
        (False, "User not found in team"),
    ]
    assert prisma.db.litellm_teamtable.update_calls == 0
    assert _roster(prisma, "t1") == ["u1"]


@pytest.mark.asyncio
async def test_bulk_member_delete_leaves_keys_and_memberships_of_unmatched_members_alone():
    prisma = _FakePrisma(
        users=[_user("u1", "t1"), _user("elsewhere")],
        teams=[_team("t1", "u1")],
        memberships=[("t1", "elsewhere")],
        tokens=[{"token": "orphan-key", "user_id": "elsewhere", "team_id": "t1"}],
    )

    results = await _remove(prisma, "t1", [{"user_id": "elsewhere"}])

    assert results[0].success is False
    assert prisma.db.litellm_teammembership.rows == [{"team_id": "t1", "user_id": "elsewhere"}]
    assert [t["token"] for t in prisma.db.litellm_verificationtoken.rows] == ["orphan-key"]


@pytest.mark.asyncio
async def test_bulk_member_delete_reports_repeated_members_as_duplicates_and_removes_them_once():
    prisma = _FakePrisma(users=[_user("u1", "t1"), _user("u2", "t1")], teams=[_team("t1", "u1", "u2", "keep")])

    results = await _remove(
        prisma, "t1", [{"user_id": "u1"}, {"user_id": "u1"}, {"user_email": "u1@example.com"}, {"user_id": "u2"}]
    )

    assert [(r.success, r.error) for r in results] == [
        (True, None),
        (False, "Duplicate member in request"),
        (True, None),
        (True, None),
    ]
    assert _roster(prisma, "t1") == ["keep"]


@pytest.mark.asyncio
async def test_bulk_member_delete_evicts_the_removed_team_keys_from_the_auth_cache():
    prisma = _FakePrisma(
        users=[_user("u1", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "keep")],
        tokens=[
            {"token": "team-key", "user_id": "u1", "team_id": "t1"},
            {"token": "keep-key", "user_id": "keep", "team_id": "t1"},
        ],
    )
    cache = _cache_with("team-key", "keep-key")

    await _remove(prisma, "t1", [{"user_id": "u1"}], cache=cache)

    assert cache.get_cache(key="team-key") is None
    assert cache.get_cache(key="keep-key") is not None


@pytest.mark.asyncio
async def test_bulk_member_delete_evicts_jwt_key_mappings_of_the_removed_team_keys():
    issuer: Final = "https://issuer.example"
    doomed: Final = jwt_key_mapping_cache_key("sub", "alice", issuer)
    kept: Final = jwt_key_mapping_cache_key("sub", "bob")
    prisma = _FakePrisma(
        users=[_user("u1", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "keep")],
        tokens=[
            {"token": "team-key", "user_id": "u1", "team_id": "t1"},
            {"token": "keep-key", "user_id": "keep", "team_id": "t1"},
        ],
        jwt_mappings=[_jwt_mapping("team-key", "alice", issuer=issuer), _jwt_mapping("keep-key", "bob")],
    )
    cache = _cache_with_jwt_mapping_keys(doomed, kept)

    await _remove(prisma, "t1", [{"user_id": "u1"}], cache=cache)

    assert cache.get_cache(key=doomed) is None
    assert cache.get_cache(key=kept) is not None


@pytest.mark.asyncio
async def test_bulk_member_delete_cleans_a_user_whose_teams_array_still_names_the_team():
    prisma = _FakePrisma(users=[_user("stale", "t1")], teams=[_team("t1", "other")], memberships=[("t1", "stale")])

    results = await _remove(prisma, "t1", [{"user_id": "stale"}])

    assert results[0].success is True
    assert prisma.db.litellm_usertable.rows["stale"].teams == []
    assert prisma.db.litellm_teammembership.rows == []
    assert _roster(prisma, "t1") == ["other"] and prisma.db.litellm_teamtable.update_calls == 0


@pytest.mark.asyncio
async def test_bulk_member_delete_by_id_removes_the_members_email_only_roster_entry():
    team = LiteLLM_TeamTable(
        team_id="t1",
        members_with_roles=[
            Member(user_id=None, user_email="u1@example.com", role="user"),
            Member(user_id="twin", user_email="u1@example.com", role="user"),
            Member(user_id="keep", user_email="keep@example.com", role="user"),
        ],
    )
    twin = _UserRow(user_id="twin", user_email="u1@example.com", teams=["t1"])
    prisma = _FakePrisma(users=[_user("u1", "t1"), twin, _user("keep", "t1")], teams=[team])

    results = await _remove(prisma, "t1", [{"user_id": "u1"}])

    assert [(r.success, r.error) for r in results] == [(True, None)]
    assert _roster(prisma, "t1") == ["twin", "keep"]
    users = prisma.db.litellm_usertable.rows
    assert users["u1"].teams == [] and users["twin"].teams == ["t1"]


@pytest.mark.asyncio
async def test_bulk_member_delete_by_id_of_a_non_member_leaves_a_same_email_users_roster_entry():
    team = LiteLLM_TeamTable(
        team_id="t1",
        members_with_roles=[
            Member(user_id=None, user_email="shared@example.com", role="user"),
            Member(user_id="keep", user_email="keep@example.com", role="user"),
        ],
    )
    outsider = _UserRow(user_id="outsider", user_email="shared@example.com", teams=[])
    member = _UserRow(user_id="member", user_email="shared@example.com", teams=["t1"])
    prisma = _FakePrisma(users=[outsider, member, _user("keep", "t1")], teams=[team])

    results = await _remove(prisma, "t1", [{"user_id": "outsider"}])

    assert [(r.success, r.error) for r in results] == [(False, "User not found in team")]
    assert _roster(prisma, "t1") == [None, "keep"]
    assert prisma.db.litellm_usertable.rows["member"].teams == ["t1"]


@pytest.mark.asyncio
async def test_bulk_member_delete_rejects_unknown_team_and_unauthorized_callers():
    prisma = _FakePrisma(users=[_user("u1", "t1")], teams=[_team("t1", "u1")])

    with pytest.raises(ManagementProblem) as missing:
        await _remove(prisma, "nope", [{"user_id": "u1"}])
    with pytest.raises(ManagementProblem) as forbidden:
        await _remove(prisma, "t1", [{"user_id": "u1"}], caller=INTERNAL)

    assert missing.value.problem.status == 404
    assert forbidden.value.problem.status == 403
    assert _roster(prisma, "t1") == ["u1"] and prisma.locks == []


@pytest.mark.asyncio
async def test_team_admin_may_bulk_remove_members():
    team = _team("t1", "lead", "u1")
    team.members_with_roles[0].role = "admin"
    prisma = _FakePrisma(users=[_user("lead", "t1"), _user("u1", "t1")], teams=[team])

    results = await _remove(prisma, "t1", [{"user_id": "u1"}], caller=UserAPIKeyAuth(user_id="lead"))

    assert results[0].success is True
    assert _roster(prisma, "t1") == ["lead"]


def test_request_models_enforce_batch_bounds():
    with pytest.raises(ValidationError):
        BulkDeleteUserRequest(user_ids=())
    with pytest.raises(ValidationError):
        BulkDeleteUserRequest(user_ids=tuple(f"u{i}" for i in range(501)))
    with pytest.raises(ValidationError):
        BulkTeamMemberDeleteRequest(members=())
    with pytest.raises(ValidationError):
        BulkTeamMemberDeleteRequest(members=tuple(TeamMemberRef(user_id=f"u{i}") for i in range(501)))
    assert len(BulkDeleteUserRequest(user_ids=tuple(f"u{i}" for i in range(500))).user_ids) == 500


def test_bulk_member_delete_request_requires_exactly_one_identifier_per_member():
    with pytest.raises(ValidationError, match="exactly one of user_id or user_email"):
        BulkTeamMemberDeleteRequest.model_validate({"members": [{"user_id": "u1", "user_email": "other@example.com"}]})
    with pytest.raises(ValidationError):
        BulkTeamMemberDeleteRequest.model_validate({"members": [{}]})
    assert BulkTeamMemberDeleteRequest(members=(TeamMemberRef(user_id="u1"),)).members[0].user_id == "u1"


def test_request_models_reject_unknown_fields():
    with pytest.raises(ValidationError, match="team_id"):
        BulkTeamMemberDeleteRequest.model_validate({"team_id": "t1", "members": [{"user_id": "u1"}]})
    with pytest.raises(ValidationError, match="role"):
        BulkTeamMemberDeleteRequest.model_validate({"members": [{"user_id": "u1", "role": "admin"}]})
    with pytest.raises(ValidationError, match="dry_run"):
        BulkDeleteUserRequest.model_validate({"user_ids": ["u1"], "dry_run": True})


@pytest.mark.asyncio
async def test_bulk_delete_writes_deleted_audit_log_for_deleted_keys(mocker):
    prisma = _FakePrisma(
        users=[_user("u1", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "keep")],
        tokens=[
            {"token": "team-key", "user_id": "u1", "team_id": "t1"},
            {"token": "personal-key", "user_id": "u1"},
            {"token": "keep-key", "user_id": "keep", "team_id": "t1"},
        ],
    )

    captured: Final[list] = []

    async def _capture(request_data):
        captured.append(request_data)

    mocker.patch("litellm.store_audit_logs", True)
    mocker.patch(
        "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
        new=_capture,
    )
    await _delete(prisma, ["u1"])
    for _ in range(100):
        if len([r for r in captured if r.table_name == LitellmTableNames.KEY_TABLE_NAME]) >= 2:
            break
        await asyncio.sleep(0.01)

    key_rows: Final = [r for r in captured if r.table_name == LitellmTableNames.KEY_TABLE_NAME]
    assert {r.object_id for r in key_rows} == {"team-key", "personal-key"}
    assert {r.action for r in key_rows} == {"deleted"}
    assert all(r.changed_by for r in key_rows)
    assert {json.loads(r.before_value)["token"] for r in key_rows} == {"team-key", "personal-key"}


@pytest.mark.asyncio
async def test_bulk_member_delete_writes_deleted_audit_log_for_removed_team_keys(mocker):
    prisma = _FakePrisma(
        users=[_user("u1", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "keep")],
        tokens=[
            {"token": "team-key", "user_id": "u1", "team_id": "t1"},
            {"token": "keep-key", "user_id": "keep", "team_id": "t1"},
        ],
    )

    captured: Final[list] = []

    async def _capture(request_data):
        captured.append(request_data)

    mocker.patch("litellm.store_audit_logs", True)
    mocker.patch(
        "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
        new=_capture,
    )
    await _remove(prisma, "t1", [{"user_id": "u1"}])
    for _ in range(100):
        if captured:
            break
        await asyncio.sleep(0.01)

    key_rows: Final = [r for r in captured if r.table_name == LitellmTableNames.KEY_TABLE_NAME]
    assert len(key_rows) == 1
    audit_row: Final = key_rows[0]
    assert audit_row.action == "deleted"
    assert audit_row.object_id == "team-key"
    assert audit_row.changed_by
    assert json.loads(audit_row.before_value)["token"] == "team-key"


@pytest.mark.asyncio
async def test_bulk_delete_skips_the_key_audit_log_when_the_tx_rolls_back(mocker):
    prisma = _FakePrisma(
        users=[_user("u1", "a-good", "z-bad"), _user("u2", "a-good")],
        teams=[_team("a-good", "u1", "u2"), _team("z-bad", "u1")],
        tokens=[{"token": "k1", "user_id": "u1", "team_id": "a-good"}],
        fail_locks=frozenset({"z-bad"}),
    )
    cache = _cache_with("k1")

    mocker.patch("litellm.store_audit_logs", True)
    mock_audit_write = mocker.patch(
        "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
        new=mocker.AsyncMock(),
    )
    results = await _delete(prisma, ["u1", "u2"], cache=cache)

    assert [(r.user_id, r.success) for r in results] == [("u1", False), ("u2", False)]
    assert [t["token"] for t in prisma.db.litellm_verificationtoken.rows] == ["k1"]
    mock_audit_write.assert_not_called()


def _unreachable_redis_cache(*, serve_last_good_entries: bool) -> UserApiKeyCache:
    from tests.unit.caching.fake_redis_cache import fake_redis_cache

    redis: Final = fake_redis_cache()
    redis.server.connected = False
    return UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=serve_last_good_entries)


@pytest.mark.asyncio
@pytest.mark.parametrize("serves_last_good", [True, False], ids=["serving", "not_serving"])
async def test_bulk_member_delete_answers_503_and_writes_nothing_when_redis_cannot_record_the_revocation(
    serves_last_good: bool,
):
    """The team's entry lists the member on its roster, the user's entry lists the team and the membership's entry
    carries the member's limits, so a removal Redis cannot revoke them for is a 503 problem with nothing written, even
    when the member holds no key; with last good entries off it removes as before."""
    prisma: Final = _FakePrisma(
        users=[_user("u1", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "keep")],
        memberships=[("t1", "u1"), ("t1", "keep")],
    )
    cache: Final = _unreachable_redis_cache(serve_last_good_entries=serves_last_good)

    if not serves_last_good:
        results: Final = await _remove(prisma, "t1", [{"user_id": "u1"}], cache=cache)
        assert results[0].success is True and _roster(prisma, "t1") == ["keep"]
        return
    with pytest.raises(ManagementProblem) as refused:
        await _remove(prisma, "t1", [{"user_id": "u1"}], cache=cache)

    assert refused.value.problem.status == 503
    assert _roster(prisma, "t1") == ["u1", "keep"]
    assert prisma.db.litellm_usertable.rows["u1"].teams == ["t1"]
    assert prisma.db.litellm_teammembership.rows == [
        {"team_id": "t1", "user_id": "u1"},
        {"team_id": "t1", "user_id": "keep"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("serves_last_good", [True, False], ids=["serving", "not_serving"])
async def test_bulk_user_delete_fails_the_row_and_deletes_nothing_when_redis_cannot_record_the_revocation(
    serves_last_good: bool,
):
    """A deleted user's own entry is revoked before the batch writes, even for a user with no key and no team: Redis
    unable to record it fails that row and the batch deletes nothing; with last good entries off it deletes as
    before."""
    prisma: Final = _FakePrisma(users=[_user("u1"), _user("keep")])
    cache: Final = _unreachable_redis_cache(serve_last_good_entries=serves_last_good)

    results: Final = await _delete(prisma, ["u1"], cache=cache)

    if not serves_last_good:
        assert [(r.user_id, r.success) for r in results] == [("u1", True)]
        assert sorted(prisma.db.litellm_usertable.rows) == ["keep"]
        return
    assert [(r.user_id, r.success) for r in results] == [("u1", False)]
    assert results[0].error is not None and "could not record this revocation" in results[0].error
    assert sorted(prisma.db.litellm_usertable.rows) == ["keep", "u1"]


def _tombstones_standing_in(redis: "FakeRedisCache") -> Callable[[Sequence[str]], Awaitable[tuple[str, ...]]]:
    async def tombstoned(cache_keys: Sequence[str]) -> tuple[str, ...]:
        client: Final = redis.cache.init_async_client()
        return tuple([key for key in cache_keys if await client.exists(f"last_good_tombstone:{key}") == 1])

    return tombstoned


class _TombstonedRows(_Rows):
    """Rows whose every delete records the cache keys of the rows it removes, with those of them that had a tombstone in
    Redis as it ran. ``tombstoned`` is a plain function, which a transaction's snapshot copies by reference."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        cache_key_of: Callable[[Mapping[str, object]], str],
        tombstoned: Callable[[Sequence[str]], Awaitable[tuple[str, ...]]],
    ) -> None:
        super().__init__(rows)
        self._cache_key_of = cache_key_of
        self._tombstoned = tombstoned
        self.deletes: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    async def delete_many(self, where: Mapping[str, object]) -> int:
        cache_keys: Final = tuple(self._cache_key_of(r) for r in self.rows if _matches(r, where))
        self.deletes.append((cache_keys, await self._tombstoned(cache_keys)))
        return await super().delete_many(where)


class _ReplicatedPrisma:
    """A client with ``DATABASE_URL_READ_REPLICA`` set: ``db`` sends reads to ``replica`` unless the context is pinned
    to the writer, and writes and transactions go to ``writer``."""

    def __init__(self, writer: _FakePrisma, replica: _FakePrisma) -> None:
        self.db = RoutingPrismaWrapper(
            writer=writer.db,  # pyright: ignore[reportArgumentType]  # a fake database stands in for PrismaWrapper
            reader=replica.db,  # pyright: ignore[reportArgumentType]  # a fake database stands in for PrismaWrapper
        )
        self.writer_db = writer.db
        self.tx = writer.tx


@pytest.mark.asyncio
@pytest.mark.parametrize("serves_last_good", [True, False], ids=["serving_pinned", "not_serving"])
async def test_bulk_user_delete_revokes_the_memberships_a_lagging_replica_lacks_before_deleting_them(
    serves_last_good: bool,
) -> None:
    """The replica lags behind u1 joining t1 and taking a membership of t2, a team since deleted: its u1 row names no
    team, t1's roster there lacks u1, and it has neither membership. While last good entries are served auth pins the
    request's reads to the writer (``reads_pinned_to_writer`` stands for it here), so the batch takes u1 off t1, and
    every membership and key delete runs with the tombstones of what it removes standing in Redis, the t2 membership's
    too, though no roster or user row names t2. With the setting off nothing is pinned: the users and memberships are
    read from the replica, so no team is rewritten, and the deletes by user id remove u1's key and memberships,
    revoking nothing."""
    from litellm.proxy.db.routing_prisma_wrapper import reads_pinned_to_writer
    from tests.unit.caching.fake_redis_cache import fake_redis_cache

    redis: Final = fake_redis_cache()
    tombstoned: Final = _tombstones_standing_in(redis)
    memberships: Final = _TombstonedRows(
        [{"team_id": "t1", "user_id": "u1"}, {"team_id": "t1", "user_id": "keep"}, {"team_id": "t2", "user_id": "u1"}],
        lambda row: team_membership_auth_cache_key(team_id=str(row["team_id"]), user_id=str(row["user_id"])),
        tombstoned,
    )
    keys: Final = _TombstonedRows(
        [{"token": "k1", "user_id": "u1", "team_id": "t1"}], lambda row: str(row["token"]), tombstoned
    )
    writer: Final = _FakePrisma(users=[_user("u1", "t1"), _user("keep", "t1")], teams=[_team("t1", "u1", "keep")])
    writer.db.litellm_teammembership = memberships
    writer.db.litellm_verificationtoken = keys
    replica: Final = _FakePrisma(
        users=[_user("u1"), _user("keep", "t1")], teams=[_team("t1", "keep")], memberships=[("t1", "keep")]
    )
    cache: Final = UserApiKeyCache(redis_cache=redis.cache, serve_last_good_entries=serves_last_good)

    with reads_pinned_to_writer(serves_last_good):
        results: Final = await _delete(_ReplicatedPrisma(writer, replica), ["u1"], cache=cache)

    in_t1: Final = team_membership_auth_cache_key(team_id="t1", user_id="u1")
    in_t2: Final = team_membership_auth_cache_key(team_id="t2", user_id="u1")
    assert (memberships.rows, keys.rows) == ([{"team_id": "t1", "user_id": "keep"}], [])
    assert set(writer.db.litellm_usertable.rows) == {"keep"}
    if serves_last_good:
        assert [(r.success, r.teams_removed) for r in results] == [(True, ("t1",))]
        assert _roster(writer, "t1") == ["keep"]
        assert memberships.deletes == [((in_t1,), (in_t1,)), ((in_t2,), (in_t2,))]
        assert keys.deletes == [(("k1",), ("k1",))]
        return
    assert [(r.success, r.teams_removed) for r in results] == [(True, ())]
    assert _roster(writer, "t1") == ["u1", "keep"]
    assert (memberships.deletes, keys.deletes) == ([((in_t1, in_t2), ())], [(("k1",), ())])


class _OneConnectionPrisma:
    """A client whose writer pool holds one connection: while its transaction holds it, a read through ``db`` would
    wait for it forever, so here that read fails at once."""

    def __init__(self, prisma: _FakePrisma) -> None:
        self._prisma = prisma
        self._in_tx = False

    @property
    def db(self) -> _Db:
        assert not self._in_tx, "a read through the client waits for the one connection its open transaction holds"
        return self._prisma.db

    @asynccontextmanager
    async def tx(self, *, timeout: object = None) -> AsyncIterator[_Tx]:
        async with self._prisma.tx(timeout=timeout) as tx:
            self._in_tx = True
            try:
                yield tx
            finally:
                self._in_tx = False


@pytest.mark.asyncio
@pytest.mark.parametrize("serves_last_good", [True, False], ids=["serving", "not_serving"])
@pytest.mark.parametrize("path", ["bulk_user_delete", "bulk_member_delete"])
async def test_bulk_deletes_read_the_deleted_keys_jwt_mappings_through_their_transaction(
    path: str, serves_last_good: bool
) -> None:
    """The JWT mappings of the keys a batch deletes go with those keys, so their cache keys are read inside the batch's
    transaction, and through it: with one connection in the pool, a read through the client would wait for the one
    the open transaction holds."""
    from tests.unit.caching.fake_redis_cache import fake_redis_cache

    doomed: Final = jwt_key_mapping_cache_key("sub", "alice")
    kept: Final = jwt_key_mapping_cache_key("sub", "bob")
    prisma: Final = _FakePrisma(
        users=[_user("u1", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "keep")],
        tokens=[
            {"token": "team-key", "user_id": "u1", "team_id": "t1"},
            {"token": "keep-key", "user_id": "keep", "team_id": "t1"},
        ],
        jwt_mappings=[_jwt_mapping("team-key", "alice"), _jwt_mapping("keep-key", "bob")],
    )
    cache: Final = UserApiKeyCache(
        redis_cache=fake_redis_cache().cache if serves_last_good else None, serve_last_good_entries=serves_last_good
    )
    for cache_key in (doomed, kept):
        cache.set_cache(key=cache_key, value={"cache_key": cache_key})
    one_connection: Final = _OneConnectionPrisma(prisma)

    results: Final = (
        await _delete(one_connection, ["u1"], cache=cache)
        if path == "bulk_user_delete"
        else await _remove(one_connection, "t1", [{"user_id": "u1"}], cache=cache)
    )

    assert [r.success for r in results] == [True]
    assert [row["token"] for row in prisma.db.litellm_verificationtoken.rows] == ["keep-key"]
    assert cache.get_cache(key=doomed) is None and cache.get_cache(key=kept) is not None


class _DrivenClock:
    """The last good entries' clock, which a test moves instead of waiting. ``wall`` moves with it, for
    ``last_good_wall_clock`` and for fake Redis (``drive_fake_redis_time``), so a tombstone lapses once the test moves
    the clock past its time."""

    def __init__(self) -> None:
        self.now: float = 1000.0

    def __call__(self) -> float:
        return self.now

    def wall(self) -> float:
        return self.now + 1_700_000_000.0


class _SlowRosterWrites(_TeamTable):
    """A team table whose every roster write takes ``seconds`` on ``clock``."""

    def __init__(self, teams: Sequence[LiteLLM_TeamTable], clock: _DrivenClock, seconds: float) -> None:
        super().__init__(teams)
        self._clock = clock
        self._seconds = seconds

    async def update(self, where: Mapping[str, str], data: Mapping[str, str]) -> LiteLLM_TeamTable:
        self._clock.now += self._seconds
        return await super().update(where, data)


class _TombstonesSeenAt(_Rows):
    """Rows whose ``at`` method, ``find_many``, ``create_many`` or ``delete_many``, records on each call, under
    ``call``, which of ``cache_keys`` had a tombstone in Redis as it ran. ``tombstoned`` is a plain function, which a
    transaction's snapshot copies by reference."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        at: str,
        call: str,
        cache_keys: Sequence[str],
        tombstoned: Callable[[Sequence[str]], Awaitable[tuple[str, ...]]],
        seen: list[tuple[str, tuple[str, ...]]],
    ) -> None:
        super().__init__(rows)
        self._at = at
        self._call = call
        self._cache_keys = tuple(cache_keys)
        self._tombstoned = tombstoned
        self._seen = seen

    async def _record(self, method: str) -> None:
        if method == self._at:
            self._seen.append((self._call, await self._tombstoned(self._cache_keys)))

    async def find_many(self, where: Mapping[str, object]) -> list[_Record]:
        await self._record("find_many")
        return await super().find_many(where)

    async def create_many(self, data: Sequence[Mapping[str, object]]) -> int:
        await self._record("create_many")
        return await super().create_many(data)

    async def delete_many(self, where: Mapping[str, object]) -> int:
        await self._record("delete_many")
        return await super().delete_many(where)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["bulk_user_delete", "bulk_member_delete"])
async def test_bulk_deletes_read_a_key_rounds_mappings_and_archive_its_keys_before_the_rounds_revocation(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """The batch's roster write takes past the tombstone time, so the key round revokes u1's key again in Redis. The
    round reads the key's JWT mappings and archives the key before that revocation, which the key's delete then
    follows with nothing awaited between: the key's tombstone has lapsed at the read and at the archive, and stands
    again at the delete."""
    from tests.unit.caching.fake_redis_cache import drive_fake_redis_time, fake_redis_cache

    clock: Final = _DrivenClock()
    drive_fake_redis_time(monkeypatch, clock.wall)
    redis: Final = fake_redis_cache()
    tombstoned: Final = _tombstones_standing_in(redis)
    seen: Final[list[tuple[str, tuple[str, ...]]]] = []  # mutable-ok: the key's tombstone at each call, in order
    prisma: Final = _FakePrisma(users=[_user("u1", "t1"), _user("keep", "t1")])
    prisma.db.litellm_teamtable = _SlowRosterWrites(
        [_team("t1", "u1", "keep")], clock, LAST_GOOD_TOMBSTONE_TTL_SECONDS + 1
    )
    prisma.db.litellm_jwtkeymapping = _TombstonesSeenAt(
        [_jwt_mapping("k1", "alice")],
        at="find_many",
        call="JWT mapping read",
        cache_keys=("k1",),
        tombstoned=tombstoned,
        seen=seen,
    )
    prisma.db.litellm_deletedverificationtoken = _TombstonesSeenAt(
        [], at="create_many", call="archive", cache_keys=("k1",), tombstoned=tombstoned, seen=seen
    )
    prisma.db.litellm_verificationtoken = _TombstonesSeenAt(
        [{"token": "k1", "user_id": "u1", "team_id": "t1"}],
        at="delete_many",
        call="key delete",
        cache_keys=("k1",),
        tombstoned=tombstoned,
        seen=seen,
    )
    cache: Final = UserApiKeyCache(
        redis_cache=redis.cache, serve_last_good_entries=True, last_good_clock=clock, last_good_wall_clock=clock.wall
    )

    results: Final = (
        await _delete(prisma, ["u1"], cache=cache)
        if path == "bulk_user_delete"
        else await _remove(prisma, "t1", [{"user_id": "u1"}], cache=cache)
    )

    assert [r.success for r in results] == [True]
    assert seen == [("JWT mapping read", ()), ("archive", ()), ("key delete", ("k1",))]


class _RosterSeenByReads(_Rows):
    """Rows whose every read records ``roster()`` as it stood then. ``roster`` is a plain function, which a
    transaction's snapshot copies by reference."""

    def __init__(self, rows: Sequence[Mapping[str, object]], roster: Callable[[], list[str | None]]) -> None:
        super().__init__(rows)
        self._roster = roster
        self.rosters_seen: list[list[str | None]] = []

    async def find_many(self, where: Mapping[str, object]) -> list[_Record]:
        self.rosters_seen.append(self._roster())
        return await super().find_many(where)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["bulk_user_delete", "bulk_member_delete"])
async def test_bulk_deletes_without_last_good_entries_read_the_keys_mappings_before_the_roster_write(path: str) -> None:
    """With last good entries off nothing is revoked, so a team's removal reads its keys' JWT mappings right after the
    keys, before it rewrites the roster, as it did before them."""
    prisma: Final = _FakePrisma(
        users=[_user("u1", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "keep")],
        tokens=[{"token": "k1", "user_id": "u1", "team_id": "t1"}],
    )
    mappings: Final = _RosterSeenByReads([_jwt_mapping("k1", "alice")], lambda: _roster(prisma, "t1"))
    prisma.db.litellm_jwtkeymapping = mappings

    results: Final = (
        await _delete(prisma, ["u1"])
        if path == "bulk_user_delete"
        else await _remove(prisma, "t1", [{"user_id": "u1"}])
    )

    assert [r.success for r in results] == [True]
    assert mappings.rosters_seen == [["u1", "keep"]]
    assert prisma.db.litellm_verificationtoken.rows == []
