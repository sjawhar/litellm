from dataclasses import dataclass
from typing import Final

VERIFIED_CLIENT_CONTEXT_KEY: Final = "litellm_verified_client_context"


@dataclass(frozen=True, slots=True)
class VerifiedClientContext:
    """The caller's address and User-Agent as the proxy itself resolved them, carried on a request that failed
    before ``add_litellm_data_to_request`` wrote its identity stamp. A JSON body can send the metadata names but
    never this object, so a reader that takes it cannot be given the caller's own strings."""

    requester_ip_address: str | None
    user_agent: str | None
