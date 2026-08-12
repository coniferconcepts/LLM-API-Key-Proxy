from __future__ import annotations

import logging
import math
import re
import time
from typing import Any, Literal, Protocol

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rotator_library.error_handler import (
    NoAvailableKeysError,
    build_public_stream_error,
    classify_credential_failure,
)

_PUBLIC_ERRORS: dict[int, tuple[str, str, str]] = {
    400: ("invalid_request", "invalid_request", "The request is invalid."),
    401: ("authentication_error", "authentication_failed", "Authentication failed."),
    409: (
        "conflict_error",
        "conflict",
        "The request conflicts with the current proxy mode.",
    ),
    429: ("rate_limit_error", "rate_limited", "The upstream service rate limited the request."),
    500: ("proxy_internal_error", "internal_error", "The proxy could not complete the request."),
    502: ("upstream_error", "bad_gateway", "The upstream service returned an invalid response."),
    503: ("service_unavailable", "service_unavailable", "The upstream service is unavailable."),
    504: ("timeout_error", "gateway_timeout", "The upstream service timed out."),
}
_SAFE_TYPE = re.compile(r"[^A-Za-z0-9_.-]")
_ANTHROPIC_ERROR_TYPES: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "api_error",
    504: "timeout_error",
}
_ANTHROPIC_PATHS = frozenset({"/v1/messages", "/v1/messages/count_tokens"})


class FinalResponseLogger(Protocol):
    def log_final_response(
        self,
        status_code: int,
        headers: dict[str, Any] | None,
        body: dict[str, Any],
    ) -> None: ...


def public_error_detail(status: int, *, code: str | None = None) -> dict[str, Any]:
    error_type, default_code, message = _PUBLIC_ERRORS.get(status, _PUBLIC_ERRORS[500])
    return {
        "type": error_type,
        "code": code or default_code,
        "status": status,
        "message": message,
    }


def anthropic_error_content(status: int) -> dict[str, Any]:
    """Build the sanitized error envelope required by Anthropic-compatible clients."""
    detail = public_error_detail(status)
    return {
        "type": "error",
        "error": {
            "type": _ANTHROPIC_ERROR_TYPES.get(status, "api_error"),
            "message": detail["message"],
        },
    }


def log_safe_exception(context: str, error: BaseException, status: int) -> None:
    error_type = _SAFE_TYPE.sub("_", type(error).__name__)[:64] or "Exception"
    logging.error("%s failed error_type=%s status=%d", context, error_type, status)


def retry_after_seconds_from_soonest(
    soonest_end: float | None,
    *,
    now: float | None = None,
) -> int | None:
    """Ceil remaining seconds if soonest_end is in the future; else None."""
    if soonest_end is None:
        return None
    remaining = float(soonest_end) - (time.time() if now is None else now)
    if remaining <= 0:
        return None
    return max(1, int(math.ceil(remaining)))


def credential_failure_response(
    category: Literal["proxy_busy", "proxy_all_credentials_exhausted"],
    *,
    content: dict[str, Any] | None = None,
    retry_after_seconds: int | None = None,
) -> JSONResponse:
    classification = classify_credential_failure(category)
    headers = None
    if retry_after_seconds is not None and retry_after_seconds >= 1:
        headers = {"Retry-After": str(int(retry_after_seconds))}
    return JSONResponse(
        status_code=classification.status,
        content=content or build_public_stream_error(category, retry_after_seconds),
        headers=headers,
    )


def handle_credential_failure(
    error: NoAvailableKeysError,
    logger: FinalResponseLogger | None,
) -> JSONResponse:
    classification = classify_credential_failure(error.category)
    log_safe_exception("OpenAI credential acquisition", error, classification.status)
    retry_after = None
    if error.category == "proxy_all_credentials_exhausted":
        retry_after = retry_after_seconds_from_soonest(error.soonest_end)
    public_body = build_public_stream_error(error.category, retry_after)
    response = credential_failure_response(
        error.category,
        content=public_body,
        retry_after_seconds=retry_after,
    )
    if logger:
        logger.log_final_response(
            status_code=classification.status,
            headers={"retry-after": str(retry_after)} if retry_after else None,
            body=public_body,
        )
    return response


def terminal_completion_error_response(response: Any) -> JSONResponse | None:
    if not isinstance(response, dict) or not isinstance(response.get("error"), dict):
        return None
    error = response["error"]
    classification = classify_credential_failure("proxy_all_credentials_exhausted")
    if (
        error.get("type") != classification.error_type
        or error.get("code") != classification.code
        or error.get("status") != classification.status
    ):
        return None
    return credential_failure_response(
        "proxy_all_credentials_exhausted",
        content=response,
    )


class SafeUnhandledErrorMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False
        response_complete = False

        async def track_response(message: Message) -> None:
            nonlocal response_complete, response_started
            response_started = response_started or message["type"] == "http.response.start"
            response_complete = response_complete or (
                message["type"] == "http.response.body" and not message.get("more_body", False)
            )
            await send(message)

        try:
            await self.app(scope, receive, track_response)
        except Exception as exc:  # noqa: BLE001 - ASGI safety boundary
            log_safe_exception("Unhandled HTTP request", exc, 500)
            if response_complete:
                return
            if response_started:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
            response = JSONResponse(
                status_code=500,
                content=(
                    anthropic_error_content(500)
                    if scope.get("path") in _ANTHROPIC_PATHS
                    else {"detail": public_error_detail(500)}
                ),
            )
            await response(scope, receive, send)
