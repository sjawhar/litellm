"""A refusal a router callback raises when it declines the deployment the router picked, before the provider call.

The decision is about the request, not the deployment's health, so the router passes it straight to the caller: it
does not cool the deployment down, retry the call, fall back to another model group, or log it as a deployment
failure. The exception is tagged rather than wrapped so its type and the client-facing error stay unchanged.
"""

from typing import Final

_ADMISSION_REFUSAL_ATTR: Final = "_litellm_deployment_admission_refusal"


def mark_admission_refusal(exception: BaseException) -> None:
    setattr(exception, _ADMISSION_REFUSAL_ATTR, True)


def is_admission_refusal(exception: BaseException | None) -> bool:
    return getattr(exception, _ADMISSION_REFUSAL_ATTR, False) is True
