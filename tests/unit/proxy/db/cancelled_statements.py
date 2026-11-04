from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

from prisma.engine.utils import handle_response_errors

CancelReason = Literal["statement timeout", "user request", "lock timeout"]

_SQLSTATE: Final[Mapping[CancelReason, str]] = MappingProxyType(
    {"statement timeout": "57014", "user request": "57014", "lock timeout": "55P03"}
)


def _model_query_payload(code: str, message: str) -> Mapping[str, object]:
    connector_error: Final = (
        "Error occurred during query execution:\nConnectorError(ConnectorError { user_facing_error: None, kind: "
        f'QueryError(PostgresError {{ code: "{code}", message: "{message}", severity: "ERROR", detail: None, '
        "column: None, hint: None }), transient: false })"
    )
    return {"error": connector_error, "user_facing_error": {"is_panic": False, "message": connector_error}}


def _raw_query_payload(code: str, message: str) -> Mapping[str, object]:
    raw_message: Final = f"ERROR: {message}"
    return {
        "error": f'Error occurred during query execution:\nConnectorError(ConnectorError {{ kind: RawDatabaseError {{ code: "{code}", message: "{raw_message}" }}, transient: false }})',
        "user_facing_error": {
            "is_panic": False,
            "message": f"Raw query failed. Code: `{code}`. Message: `{raw_message}`",
            "meta": {"code": code, "message": raw_message},
            "error_code": "P2010",
        },
    }


def cancelled_statement(reason: CancelReason, *, raw: bool = False) -> Exception:
    """What a statement Postgres cancelled raises through prisma, built by prisma's own response handling from the
    engine payload. The payloads were captured from prisma-client-py 0.11.0 against Postgres 16.6 on 2026-10-30:
    a model query (``find_unique``) and a raw query, each blocked on a held table lock and then cancelled by
    ``statement_timeout``, ``lock_timeout`` or ``pg_cancel_backend``."""
    build: Final = _raw_query_payload if raw else _model_query_payload
    try:
        handle_response_errors(None, [build(_SQLSTATE[reason], f"canceling statement due to {reason}")])  # pyright: ignore[reportArgumentType]  # the response is read only for an unmapped error
    except Exception as e:  # noqa: BLE001  # prisma raises the mapped error, which is the value under test
        return e
    raise AssertionError("prisma did not raise for an error payload")
