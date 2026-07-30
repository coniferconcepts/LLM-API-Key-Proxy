from __future__ import annotations

from collections.abc import AsyncGenerator

import anyio
import pytest

import proxy_app.anthropic_stream as anthropic_stream
from proxy_app.stream_bounds import SSEStreamLimitExceeded, bounded_openai_sse_stream
from sse_fragmented_event_support import collect, policy

_WIRE_EVENT = "data: value\r\n\r\n"


@pytest.mark.asyncio
async def test_fragmented_data_field_names_cannot_bypass_event_limit() -> None:
    chunks = ("da", "ta: one\n\n", "d", "ata: two\r\n\r\n", "dat", "a: three\r\r")
    with pytest.raises(SSEStreamLimitExceeded, match="events"):
        await collect(chunks, policy())


@pytest.mark.asyncio
@pytest.mark.parametrize("split_at", range(1, len(_WIRE_EVENT)))
async def test_openai_event_bound_survives_every_wire_split(split_at: int) -> None:
    fragment = (_WIRE_EVENT[:split_at], _WIRE_EVENT[split_at:])

    async def source() -> AsyncGenerator[str, None]:
        for chunk in fragment * 3:
            yield chunk

    with pytest.raises(SSEStreamLimitExceeded, match="events"):
        _ = [
            chunk
            async for chunk in bounded_openai_sse_stream(
                source(), policy(), anyio.CapacityLimiter(1)
            )
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("split_at", range(1, len(_WIRE_EVENT)))
async def test_anthropic_event_bound_survives_every_wire_split(
    monkeypatch: pytest.MonkeyPatch, split_at: int
) -> None:
    closed = 0
    fragment = (_WIRE_EVENT[:split_at], _WIRE_EVENT[split_at:])

    async def source() -> AsyncGenerator[str, None]:
        nonlocal closed
        try:
            for chunk in fragment * 3:
                yield chunk
        finally:
            closed += 1

    monkeypatch.setattr(anthropic_stream, "ANTHROPIC_STREAM_MAX_EVENTS", 2)
    body = "".join(
        [chunk async for chunk in anthropic_stream.bounded_anthropic_sse_response(source())]
    )
    assert "event: error" in body
    assert '"type": "api_error"' in body
    assert closed == 1
