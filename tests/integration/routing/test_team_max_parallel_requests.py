import json
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx
import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, eventually
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, wire_server

pytestmark: Final = pytest.mark.timeout(6 * graceful_stop_seconds() + 120)

KEY_ID: Final = "integration-team-parallel-signing-key"
TEAM_LIMIT: Final = 2


def _frame(identity: str, delta: dict[str, str], *, finish: str | None = None) -> bytes:
    event: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(event).encode() + b"\n\n"


def _stream_chunks(identity: str) -> tuple[bytes, ...]:
    return (
        _frame(identity, {"role": "assistant", "content": "held"}),
        _frame(identity, {"content": " open"}),
        _frame(identity, {}, finish="stop"),
        b"data: [DONE]\n\n",
    )


@dataclass(frozen=True, slots=True)
class Upstream:
    """A provider that holds a registered stream open after its first frame until the test releases it, so the call
    stays in flight on the gateway for as long as the test needs. A stream registered to fail breaks its connection
    after the second frame, mid-stream, once released. An unregistered stream is answered at once."""

    gates: dict[str, threading.Event] = field(default_factory=dict)
    failing: set[str] = field(default_factory=set)

    def hold(self, marker: str, *, fail: bool) -> threading.Event:
        gate: Final = threading.Event()
        self.gates[marker] = gate
        if fail:
            self.failing.add(marker)
        return gate

    def respond(self, request: Request) -> Reply:
        assert request.target == "/v1/chat/completions", request
        body: Final = json.loads(request.body)
        assert body["stream"] is True, body
        marker: Final = body["messages"][0]["content"]
        return Reply(
            content_type="text/event-stream",
            chunks=_stream_chunks(marker),
            gate_after_first=self.gates.get(marker),
            abort_after=2 if marker in self.failing else None,
        )


@dataclass(frozen=True, slots=True)
class Held:
    response: httpx.Response
    lines: Iterator[str]
    gate: threading.Event


def _open(
    pod: Gateway, model: str, credential: str, upstream: Upstream, name: str, *, fail: bool = False
) -> Held | int:
    marker: Final = f"team-parallel-{name}-{uuid.uuid4().hex}"
    gate: Final = upstream.hold(marker, fail=fail)
    request: Final = pod.client.build_request(
        "POST",
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": marker}], "stream": True},
        headers={"Authorization": f"Bearer {credential}"},
    )
    response: Final = pod.client.send(request, stream=True)
    if response.status_code != 200:
        response.read()
        response.close()
        return response.status_code
    lines: Final = response.iter_lines()
    first: Final = next(line for line in lines if line)
    assert marker in first, first
    return Held(response, lines, gate)


def _admitted(pod: Gateway, model: str, credential: str, upstream: Upstream, name: str) -> Held:
    held: Final = eventually(
        lambda: _open(pod, model, credential, upstream, name),
        lambda outcome: isinstance(outcome, Held),
        seconds=30,
    )
    assert isinstance(held, Held)
    return held


def _refusal(pod: Gateway, model: str, credential: str) -> httpx.Response:
    response: Final = pod.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"refused-{uuid.uuid4().hex}"}], "stream": True},
        key=credential,
    )
    assert response.status_code == 429, response.text
    return response


def _read_to_end(held: Held) -> str:
    try:
        return "\n".join(held.lines)
    except httpx.HTTPError as broken:
        return f"broken: {broken!r}"
    finally:
        held.response.close()


def _config(directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = directory / "team_parallel_requests.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "general_settings": {
                    **config["general_settings"],
                    "enable_jwt_auth": True,
                    "litellm_jwtauth": {"team_id_jwt_field": "team_id"},
                },
            }
        )
    )
    return path


def _jwks_reply(private_key: rsa.RSAPrivateKey) -> Reply:
    public: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    return Reply(body=json.dumps({"keys": [{**public, "kid": KEY_ID, "use": "sig", "alg": "RS256"}]}).encode())


def test_team_parallel_limit_holds_across_pods_and_each_ending_frees_a_slot(gateway: Gateway, tmp_path: Path) -> None:
    """Three proxies share one Redis. A team caller signed in with a token has no key, so its team's
    max_parallel_requests is its only concurrency limit: the team's third call in flight is refused on any pod, a key
    on the same team counts against the same limit, and a stream that completes, one that fails mid-stream and one
    the client abandons each free their slot for the next call"""
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwks_reply: Final = _jwks_reply(private_key)
    upstream_double: Final = Upstream()
    config: Final = _config(tmp_path)

    def jwks(request: Request) -> Reply:
        assert request.method == "GET", request
        return jwks_reply

    with (
        wire_server(jwks) as signing_keys,
        wire_server(upstream_double.respond) as upstream,
        owned_proxy(gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": signing_keys.url}, config=config) as first_pod,
        first_pod.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=upstream.url + "/v1", num_retries=0)
        team: Final = scenario.team(models=[model], max_parallel_requests=TEAM_LIMIT)
        team_key: Final = scenario.key(team_id=team, models=[model])
        issued: Final = int(time.time())
        token: Final = jwt.encode(
            {"sub": f"team-parallel-{uuid.uuid4().hex}", "team_id": team, "iat": issued, "exp": issued + 900},
            private_key,
            algorithm="RS256",
            headers={"kid": KEY_ID},
        )
        environment: Final = {"JWT_PUBLIC_KEY_URL": signing_keys.url}
        with (
            owned_proxy(gateway, tmp_path, environment, config=config) as second_pod,
            owned_proxy(gateway, tmp_path, environment, config=config) as third_pod,
        ):
            completing: Final = _open(first_pod, model, token, upstream_double, "success")
            failing: Final = _open(second_pod, model, team_key, upstream_double, "failure", fail=True)
            assert isinstance(completing, Held) and isinstance(failing, Held), (completing, failing)

            for pod, credential in ((third_pod, token), (first_pod, team_key)):
                refused = _refusal(pod, model, credential).text
                assert f"team: {team}." in refused and "Limit type: max_parallel_requests." in refused, refused

            completing.gate.set()
            assert "data: [DONE]" in _read_to_end(completing)
            abandoned: Final = _admitted(third_pod, model, token, upstream_double, "after-success")
            _refusal(first_pod, model, token)

            failing.gate.set()
            assert "data: [DONE]" not in _read_to_end(failing)
            last: Final = _admitted(first_pod, model, token, upstream_double, "after-failure")
            _refusal(second_pod, model, token)

            abandoned.response.close()
            after_disconnect: Final = _admitted(second_pod, model, token, upstream_double, "after-disconnect")
            _refusal(third_pod, model, team_key)

            for held in (last, after_disconnect):
                held.response.close()
