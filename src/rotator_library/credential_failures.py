from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

CredentialFailureCategory: TypeAlias = Literal["proxy_busy", "proxy_all_credentials_exhausted"]


@dataclass(frozen=True, slots=True)
class CredentialFailureClassification:
    error_type: str
    code: str
    status: int
    message: str
    metric_code: str
    monitor_code: str
    retryable: bool


_CREDENTIAL_FAILURES: Final = {
    "proxy_busy": CredentialFailureClassification(
        error_type="proxy_busy",
        code="acquisition_timeout",
        status=503,
        message="No upstream credential is currently available. Retry later.",
        metric_code="proxy_busy",
        monitor_code="acquisition_timeout",
        retryable=True,
    ),
    "proxy_all_credentials_exhausted": CredentialFailureClassification(
        error_type="proxy_all_credentials_exhausted",
        code="all_credentials_exhausted",
        status=503,
        message="No upstream credential is currently available. Retry later.",
        metric_code="proxy_all_credentials_exhausted",
        monitor_code="all_credentials_exhausted",
        retryable=False,
    ),
}

_PUBLIC_STREAM_ERRORS: Final = {
    "quota_exceeded": (
        "rate_limit_error",
        "quota_exceeded",
        429,
        "The upstream service rate limited the request. Retry later.",
    ),
    "internal_error": (
        "proxy_internal_error",
        "stream_error",
        500,
        "The proxy could not complete the streaming request.",
    ),
}


def classify_credential_failure(
    category: CredentialFailureCategory,
) -> CredentialFailureClassification:
    return _CREDENTIAL_FAILURES[category]


def build_public_stream_error(
    category: str,
    retry_after_seconds: int | None = None,
) -> dict[str, dict[str, str | int]]:
    """Return an allowlisted terminal SSE error without provider-controlled values."""
    credential_failure = _CREDENTIAL_FAILURES.get(category)
    if credential_failure is None:
        error_type, code, status, message = _PUBLIC_STREAM_ERRORS.get(
            category,
            _PUBLIC_STREAM_ERRORS["internal_error"],
        )
    else:
        error_type = credential_failure.error_type
        code = credential_failure.code
        status = credential_failure.status
        message = credential_failure.message
    error: dict[str, str | int] = {
        "type": error_type,
        "code": code,
        "status": status,
        "message": message,
    }
    if retry_after_seconds is not None and retry_after_seconds >= 1:
        error["retry_after_seconds"] = int(retry_after_seconds)
    return {"error": error}
