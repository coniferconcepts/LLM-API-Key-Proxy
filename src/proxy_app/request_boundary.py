from __future__ import annotations

import asyncio
import json
import math
import secrets
from collections.abc import Callable
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from proxy_app.request_framing import JSONBoundaryError, parse_content_length
from proxy_app.safe_errors import anthropic_error_content

MAX_JSON_BODY_BYTES = 4_194_304
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 100_000
MAX_JSON_NUMBER_CHARACTERS = 128
BODY_READ_TIMEOUT_SECONDS = 30.0
BODY_INTER_CHUNK_TIMEOUT_SECONDS = 5.0

JSON_BODY_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/embeddings",
        "/v1/quota-stats",
        "/v1/token-count",
        "/v1/cost-estimate",
    }
)
ANTHROPIC_BODY_PATHS = frozenset({"/v1/messages", "/v1/messages/count_tokens"})


def _open_credential_config() -> tuple[None, bool]:
    return None, True


def _authenticate(
    headers: list[tuple[bytes, bytes]],
    path: str,
    credential_config: tuple[str | None, bool],
) -> None:
    proxy_api_key, is_current = credential_config
    if not is_current:
        raise JSONBoundaryError(
            503,
            "security_configuration_changed",
            "Runtime security configuration changed after initialization.",
        )
    if not proxy_api_key:
        return
    authorization = [value for name, value in headers if name == b"authorization"]
    api_keys = [value for name, value in headers if name == b"x-api-key"]
    if len(authorization) > 1 or len(api_keys) > 1 or (authorization and api_keys):
        raise JSONBoundaryError(401, "authentication_failed", "Authentication failed.")

    expected_key = proxy_api_key.encode("utf-8")
    valid = False
    if authorization:
        valid = secrets.compare_digest(authorization[0], b"Bearer " + expected_key)
    elif path in ANTHROPIC_BODY_PATHS and api_keys:
        valid = secrets.compare_digest(api_keys[0], expected_key)
    if not valid:
        raise JSONBoundaryError(401, "authentication_failed", "Authentication failed.")


def _load_bounded_json(
    body: bytes,
    *,
    max_depth: int,
    max_nodes: int,
    max_number_characters: int,
) -> dict[str, Any]:
    def parse_integer(value: str) -> int:
        if len(value) > max_number_characters:
            raise ValueError("integer too long")
        return int(value)

    def parse_float(value: str) -> float:
        if len(value) > max_number_characters:
            raise ValueError("float too long")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite float")
        return parsed

    try:
        parsed = json.loads(
            body,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("non-finite number")),
            parse_float=parse_float,
            parse_int=parse_integer,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise JSONBoundaryError(
            400, "invalid_json", "Request body must be valid bounded JSON."
        ) from exc
    if not isinstance(parsed, dict):
        raise JSONBoundaryError(400, "invalid_json_shape", "Request body must be a JSON object.")

    nodes = 0
    stack: list[tuple[Any, int]] = [(parsed, 1)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise JSONBoundaryError(400, "json_too_complex", "Request JSON is too complex.")
        if depth > max_depth:
            raise JSONBoundaryError(400, "json_too_deep", "Request JSON is too deeply nested.")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)
    return parsed


class BoundedJSONBodyMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        paths: frozenset[str] | set[str] = JSON_BODY_PATHS,
        max_bytes: int = MAX_JSON_BODY_BYTES,
        max_depth: int = MAX_JSON_DEPTH,
        max_nodes: int = MAX_JSON_NODES,
        max_number_characters: int = MAX_JSON_NUMBER_CHARACTERS,
        credential_config_getter: Callable[[], tuple[str | None, bool]] = _open_credential_config,
        body_read_timeout_seconds: float = BODY_READ_TIMEOUT_SECONDS,
        body_inter_chunk_timeout_seconds: float = BODY_INTER_CHUNK_TIMEOUT_SECONDS,
    ) -> None:
        if (
            not math.isfinite(body_read_timeout_seconds)
            or body_read_timeout_seconds <= 0
            or not math.isfinite(body_inter_chunk_timeout_seconds)
            or body_inter_chunk_timeout_seconds <= 0
        ):
            raise ValueError("body read timeouts must be finite positive seconds")
        self.app = app
        self.paths = frozenset(paths)
        self.max_bytes = max_bytes
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_number_characters = max_number_characters
        self.credential_config_getter = credential_config_getter
        self.body_read_timeout_seconds = body_read_timeout_seconds
        self.body_inter_chunk_timeout_seconds = body_inter_chunk_timeout_seconds

    async def _reject(
        self, error: JSONBoundaryError, scope: Scope, receive: Receive, send: Send
    ) -> None:
        path = scope.get("path", "")
        if path in ANTHROPIC_BODY_PATHS:
            response = JSONResponse(
                status_code=error.status,
                content=anthropic_error_content(error.status),
            )
            await response(scope, receive, send)
            return
        response = JSONResponse(
            status_code=error.status,
            content={
                "detail": {
                    "type": "invalid_request",
                    "code": error.code,
                    "status": error.status,
                    "message": error.public_message,
                }
            },
        )
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") not in {"POST", "PUT", "PATCH"}
            or scope.get("path") not in self.paths
        ):
            await self.app(scope, receive, send)
            return

        try:
            _authenticate(
                scope.get("headers", []),
                scope.get("path", ""),
                self.credential_config_getter(),
            )
            declared = parse_content_length(scope.get("headers", []), self.max_bytes)
            chunks: list[bytes] = []
            body_size = 0
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.body_read_timeout_seconds
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise JSONBoundaryError(408, "body_timeout", "Request body read timed out.")
                try:
                    message = await asyncio.wait_for(
                        receive(),
                        timeout=min(remaining, self.body_inter_chunk_timeout_seconds),
                    )
                except asyncio.TimeoutError as exc:
                    raise JSONBoundaryError(
                        408,
                        "body_timeout",
                        "Request body read timed out.",
                    ) from exc
                if message["type"] != "http.request":
                    raise JSONBoundaryError(400, "incomplete_body", "Request body is incomplete.")
                chunk = message.get("body", b"")
                body_size += len(chunk)
                if body_size > self.max_bytes:
                    raise JSONBoundaryError(
                        413,
                        "body_too_large",
                        "Request body exceeds the 4 MiB limit.",
                    )
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            if declared is not None and declared != len(body):
                raise JSONBoundaryError(
                    400, "content_length_mismatch", "Request framing is invalid."
                )
            _load_bounded_json(
                body,
                max_depth=self.max_depth,
                max_nodes=self.max_nodes,
                max_number_characters=self.max_number_characters,
            )
        except (UnicodeDecodeError, JSONBoundaryError) as error:
            boundary_error = (
                error
                if isinstance(error, JSONBoundaryError)
                else JSONBoundaryError(400, "invalid_header", "Request framing is invalid.")
            )
            await self._reject(boundary_error, scope, receive, send)
            return

        replayed = False

        async def replay_body() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay_body, send)
