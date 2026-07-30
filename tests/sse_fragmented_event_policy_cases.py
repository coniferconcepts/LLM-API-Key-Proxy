from __future__ import annotations

from collections.abc import AsyncGenerator

import anyio
import pytest

from proxy_app.stream_bounds import (
    DEFAULT_SSE_STREAM_POLICY,
    SSEStreamLimitExceeded,
    bounded_sse_stream,
)
from sse_fragmented_event_support import collect, policy


@pytest.mark.asyncio
async def test_fragmented_stream_preserves_accepted_bytes_and_order() -> None:
    chunks = ("eve", "nt: ping\r", "\nda", "ta", ": one\r\n", "data: two\n", "\n")
    accepted = await collect(chunks, policy(max_events=2))
    assert accepted == list(chunks)
    assert "".join(accepted) == "".join(chunks)


@pytest.mark.asyncio
async def test_multiline_and_bare_data_fields_share_one_definition() -> None:
    with pytest.raises(SSEStreamLimitExceeded, match="events"):
        await collect(("data: first\n", "da", "ta\n\n"), policy(max_events=1))


@pytest.mark.asyncio
async def test_non_data_fields_and_prefixes_do_not_consume_the_bound() -> None:
    chunks = (": keepalive\n", "event: ping\r\n", " data: indented\n", "database: value\n\n")
    accepted = await collect(chunks, policy(max_events=0))
    assert accepted == list(chunks)


@pytest.mark.asyncio
async def test_many_tiny_fragments_cannot_turn_event_bound_into_byte_only_bound() -> None:
    chunks = ("da", "ta:\n\n") * 100_001
    with pytest.raises(SSEStreamLimitExceeded, match="events") as raised:
        await collect(chunks, DEFAULT_SSE_STREAM_POLICY)
    assert raised.value.boundary == "events"


@pytest.mark.asyncio
async def test_medium_fragmented_stream_below_default_bounds_is_forwarded() -> None:
    field_count = 50_000
    chunks = ("da", "ta: fragment\n\n") * field_count
    accepted = await collect(chunks, DEFAULT_SSE_STREAM_POLICY)
    assert len(accepted) == field_count * 2
    assert accepted[:2] == ["da", "ta: fragment\n\n"]
    assert accepted[-2:] == ["da", "ta: fragment\n\n"]


@pytest.mark.asyncio
async def test_one_byte_over_limit_reports_byte_boundary() -> None:
    chunk = "data: x\n\n"
    with pytest.raises(SSEStreamLimitExceeded, match="bytes") as raised:
        await collect((chunk,), policy(max_events=1, max_bytes=len(chunk.encode("utf-8")) - 1))
    assert raised.value.boundary == "bytes"


@pytest.mark.asyncio
async def test_fragment_limit_closes_once_and_capacity_recovers() -> None:
    limiter = anyio.CapacityLimiter(1)
    closed = 0

    async def over_limit() -> AsyncGenerator[str, None]:
        nonlocal closed
        try:
            for chunk in ("da", "ta:\n\n") * 3:
                yield chunk
        finally:
            closed += 1

    async def healthy() -> AsyncGenerator[str, None]:
        yield "data: recovered\n\n"

    with pytest.raises(SSEStreamLimitExceeded, match="events"):
        _ = [chunk async for chunk in bounded_sse_stream(over_limit(), policy(), limiter)]
    recovered = [chunk async for chunk in bounded_sse_stream(healthy(), policy(), limiter)]
    assert closed == 1
    assert recovered == ["data: recovered\n\n"]
