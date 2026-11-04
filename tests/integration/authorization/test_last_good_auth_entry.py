"""R69/A79: while the database is unreachable, the gateway serves each key and team it last read from the database,
whatever the age of its cached entry, enforces the user limits it last read for them, and refuses a caller it never
read or that was deleted. The database and Redis each sit behind a relay the test takes down and brings back."""

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa

from tests.integration._support.client import Gateway, eventually, string_value
from tests.integration._support.database import scratch_database
from tests.integration._support.database_relay import outage_relay
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

pytestmark = pytest.mark.timeout(600)

MODEL: Final = "last-good-model"
KEY_ID: Final = "integration-last-good"
CACHE_TTL_SECONDS: Final = 2
# JWT sign-in is an Enterprise feature: the team-by-id and team-by-alias JWT callers run only with a license.
# Without one, the team path is still exercised through the team key, whose auth reads its team by id.
JWT_LICENSED: Final = bool(os.environ.get("LITELLM_LICENSE"))


def _completion(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=json.dumps({"object": "list", "data": [{"id": MODEL, "object": "model"}]}).encode())
    assert request.method == "POST" and request.target.endswith("/chat/completions"), request
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


@contextmanager
def _issuer() -> Iterator[tuple[rsa.RSAPrivateKey, str]]:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    jwks: Final = json.dumps({"keys": [{**public_jwk, "kid": KEY_ID, "use": "sig", "alg": "RS256"}]}).encode()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=jwks)

    with wire_server(respond) as server:
        yield private_key, server.url


def _token(private_key: rsa.RSAPrivateKey, **claims: str) -> str:
    now: Final = int(time.time())
    return jwt.encode(
        {"sub": f"workload-{uuid.uuid4().hex}", **claims, "iat": now, "exp": now + 600},
        private_key,
        algorithm="RS256",
        headers={"kid": KEY_ID},
    )


def _config(directory: Path, upstream_url: str, redis_port: int) -> Path:
    config: Final = {
        "model_list": [
            {
                "model_name": MODEL,
                "litellm_params": {"model": f"openai/{MODEL}", "api_base": upstream_url, "api_key": "synthetic-key"},
            }
        ],
        "litellm_settings": {"enable_redis_auth_cache": True},
        "general_settings": {
            "master_key": "os.environ/LITELLM_MASTER_KEY",
            "database_url": "os.environ/DATABASE_URL",
            "user_api_key_cache_ttl": CACHE_TTL_SECONDS,
            "serve_last_good_auth_entries": True,
            "coordination_redis": {"host": "127.0.0.1", "port": redis_port},
            "enable_jwt_auth": True,
            "litellm_jwtauth": {"team_id_jwt_field": "team_id", "team_alias_jwt_field": "team_alias"},
        },
    }
    path: Final = directory / "last_good_auth_entry.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _chat(proxy: Gateway, credential: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": "last good control"}]},
        key=credential,
    )


def _outcome(response: httpx.Response) -> int | str:
    return 200 if response.status_code == 200 else f"{response.status_code} {response.text[:400]}"


def _statuses(proxy: Gateway, callers: dict[str, str]) -> dict[str, int | str]:
    return {name: _outcome(_chat(proxy, credential)) for name, credential in callers.items()}


def _outlive_the_cache() -> None:
    time.sleep(CACHE_TTL_SECONDS + 1)


def _user_key(proxy: Gateway, *, max_budget: float | None) -> str:
    """A personal key of a new internal user, whose budget the gateway checks on every request."""
    user_id: Final = f"last-good-user-{uuid.uuid4().hex}"
    proxy.post("/user/new", {"user_id": user_id, "models": [MODEL], "max_budget": max_budget, "auto_create_key": False})
    return string_value(proxy.post("/key/generate", {"user_id": user_id, "models": [MODEL]})["key"])


def _over_budget(response: httpx.Response) -> bool:
    return response.status_code != 200 and "ExceededBudget" in response.text


def test_callers_the_gateway_last_read_are_served_while_the_database_is_stopped(tmp_path: Path) -> None:
    master_key: Final = f"sk-{uuid.uuid4().hex}"
    redis_host: Final = os.environ.get("REDIS_HOST", "127.0.0.1")
    redis_port: Final = int(os.environ.get("REDIS_PORT", "6379"))
    with (
        wire_server(_completion) as upstream,
        _issuer() as (private_key, jwks_url),
        scratch_database() as database_url,
        outage_relay(urlsplit(database_url).hostname or "", urlsplit(database_url).port or 5432) as database,
        outage_relay(redis_host, redis_port) as redis,
        httpx.Client(trust_env=False) as stub_client,
    ):
        relayed_database: Final = (
            urlsplit(database_url)
            ._replace(
                netloc=f"{urlsplit(database_url).username}:{urlsplit(database_url).password}@127.0.0.1:{database.port}"
            )
            .geturl()
        )
        stub: Final = Gateway(stub_client, master_key, upstream.url)
        overrides: Final = {
            "DATABASE_URL": relayed_database,
            "JWT_PUBLIC_KEY_URL": jwks_url,
            "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
        }
        with owned_proxy(stub, tmp_path, overrides, config=_config(tmp_path, upstream.url, redis.port)) as proxy:
            by_id: Final = string_value(
                proxy.post("/team/new", {"team_alias": f"lanes-{uuid.uuid4().hex}", "models": [MODEL]})["team_id"]
            )
            by_alias: Final = f"cuts-{uuid.uuid4().hex}"
            proxy.post("/team/new", {"team_alias": by_alias, "models": [MODEL]})
            jwt_callers: Final = (
                {
                    "team_by_id": _token(private_key, team_id=by_id),
                    "team_by_alias": _token(private_key, team_alias=by_alias),
                }
                if JWT_LICENSED
                else {}
            )
            served: Final = {
                "key": string_value(proxy.post("/key/generate", {"models": [MODEL]})["key"]),
                "team_key": string_value(proxy.post("/key/generate", {"team_id": by_id, "models": [MODEL]})["key"]),
                "user_key": _user_key(proxy, max_budget=None),
                **jwt_callers,
            }
            spent_user_key: Final = _user_key(proxy, max_budget=0.0)
            deleted: Final = string_value(proxy.post("/key/generate", {"models": [MODEL]})["key"])

            assert _statuses(proxy, {**served, "deleted": deleted}) == dict.fromkeys((*served, "deleted"), 200)
            assert _over_budget(_chat(proxy, spent_user_key)), "the user's budget is spent"
            _outlive_the_cache()
            proxy.post("/key/delete", {"keys": [deleted]})
            database.cut()

            assert _statuses(proxy, served) == dict.fromkeys(served, 200), "the database is stopped"
            assert _over_budget(_chat(proxy, spent_user_key)), "the user's budget holds while the database is stopped"
            refused: Final = _statuses(proxy, {"deleted": deleted, "never_read": f"sk-{uuid.uuid4().hex}"})
            assert all(str(outcome).split()[0] in ("401", "503") for outcome in refused.values()), refused

            redis.cut()
            _outlive_the_cache()
            assert _statuses(proxy, served) == dict.fromkeys(served, 200), "the database and Redis are stopped"

            redis.restore()
            database.restore()
            narrowed: Final = {"key": served["key"], "models": ["some-other-model"]}
            eventually(
                lambda: proxy.request("POST", "/key/update", narrowed).status_code,
                lambda status: status == 200,
                seconds=90,
            )
            assert _chat(proxy, served["key"]).status_code in (401, 403), "the database answered again"
            _outlive_the_cache()
            database.cut()
            assert _chat(proxy, served["key"]).status_code in (401, 403), "the refreshed entry is the one served"
            database.restore()
