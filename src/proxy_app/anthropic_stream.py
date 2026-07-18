from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
import json
from typing import Any, Protocol

from proxy_app.safe_errors import anthropic_error_content, log_safe_exception
from proxy_app.stream_bounds import (
    DEFAULT_SSE_STREAM_CAPACITY,
    DEFAULT_SSE_STREAM_POLICY,
    SSE_TERMINAL_RESYNC,
    SSEStreamPolicy,
    bounded_sse_stream,
)
from rotator_library.stream_terminal import require_anthropic_terminal

ANTHROPIC_STREAM_MAX_BYTES = DEFAULT_SSE_STREAM_POLICY.max_bytes
ANTHROPIC_STREAM_MAX_EVENTS = DEFAULT_SSE_STREAM_POLICY.max_events
ANTHROPIC_STREAM_IDLE_TIMEOUT_SECONDS = DEFAULT_SSE_STREAM_POLICY.idle_timeout_seconds
ANTHROPIC_STREAM_TOTAL_TIMEOUT_SECONDS = DEFAULT_SSE_STREAM_POLICY.total_timeout_seconds
ANTHROPIC_STREAM_CLEANUP_TIMEOUT_SECONDS = DEFAULT_SSE_STREAM_POLICY.cleanup_timeout_seconds
ANTHROPIC_STREAM_CAPACITY = DEFAULT_SSE_STREAM_CAPACITY


class DisconnectRequest(Protocol):
    async def is_disconnected(self) -> bool: ...


class FinalResponseLogger(Protocol):
    def log_final_response(
        self,
        status_code: int,
        headers: dict[str, str] | None,
        body: Any,
    ) -> None: ...


def _stream_policy() -> SSEStreamPolicy:
    return SSEStreamPolicy(
        max_bytes=ANTHROPIC_STREAM_MAX_BYTES,
        max_events=ANTHROPIC_STREAM_MAX_EVENTS,
        idle_timeout_seconds=ANTHROPIC_STREAM_IDLE_TIMEOUT_SECONDS,
        total_timeout_seconds=ANTHROPIC_STREAM_TOTAL_TIMEOUT_SECONDS,
        cleanup_timeout_seconds=ANTHROPIC_STREAM_CLEANUP_TIMEOUT_SECONDS,
    )


async def bounded_anthropic_sse_response(
    response_stream: AsyncGenerator[str, None],
    logger: FinalResponseLogger | None = None,
    *,
    request: DisconnectRequest | None = None,
) -> AsyncGenerator[str, None]:
    try:
        bounded_stream = bounded_sse_stream(
            response_stream,
            _stream_policy(),
            ANTHROPIC_STREAM_CAPACITY,
        )
        async with aclosing(bounded_stream):
            disconnect_check = request.is_disconnected if request is not None else None
            async for chunk in require_anthropic_terminal(bounded_stream, disconnect_check):
                yield chunk
    except Exception as exc:
        log_safe_exception("Anthropic response stream", exc, 500)
        payload = anthropic_error_content(500)
        yield f"{SSE_TERMINAL_RESYNC}event: error\ndata: {json.dumps(payload)}\n\n"
        if logger:
            logger.log_final_response(status_code=500, headers=None, body=payload)
