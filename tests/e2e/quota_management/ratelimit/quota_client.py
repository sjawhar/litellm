"""Client for the quota-management suite: the shared ProxyClient plus raw chat
calls judged by HTTP status, body, and headers (a rate-limit block is a 429
whose body and retry-after header carry the contract, not a typed success
model)."""

from __future__ import annotations

from dataclasses import dataclass

from proxy_client import ProxyClient
from e2e_http import StreamingResponse, unwrap
from e2e_metadata import step
from models import ChatBody, ChatMessage, TeamInfoParams, TeamInfoResponse, TeamMetadata


@dataclass(frozen=True, slots=True)
class QuotaClient:
    proxy: ProxyClient

    @step('Send a /chat/completions request to {model} with the prompt "{content}"')
    def chat(self, key: str, model: str, content: str, *, max_tokens: int = 16) -> StreamingResponse:
        return self.proxy.transport.send(
            "/chat/completions",
            headers=self.proxy.transport.bearer(key),
            json=ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=content)],
                max_tokens=max_tokens,
            ),
        )

    @step("Read the team's metadata from /team/info")
    def team_metadata(self, team_id: str) -> TeamMetadata | None:
        return unwrap(
            self.proxy.transport.get(
                "/team/info",
                headers=self.proxy.management_headers(),
                params=TeamInfoParams(team_id=team_id),
                response_type=TeamInfoResponse,
            )
        ).team_info.metadata


def build_client(proxy: ProxyClient) -> QuotaClient:
    return QuotaClient(proxy=proxy)
