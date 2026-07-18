from __future__ import annotations

from collections.abc import AsyncGenerator

import anyio
import pytest

from proxy_app.stream_bounds import (
    OpenAIStreamLimitExceeded,
    OpenAIStreamPolicy,
    OpenAIStreamTimedOut,
    bounded_openai_sse_stream,
)


def _policy(*, max_bytes: int = 100, max_events: int = 10) -> OpenAIStreamPolicy:
    return OpenAIStreamPolicy(
        max_bytes=max_bytes,
        max_events=max_events,
        idle_timeout_seconds=0.05,
        total_timeout_seconds=0.25,
    )


async def _collect(
    source: AsyncGenerator[str, None],
    policy: OpenAIStreamPolicy,
    capacity: anyio.CapacityLimiter | None = None,
) -> list[str]:
    limiter = capacity or anyio.CapacityLimiter(1)
    return [chunk async for chunk in bounded_openai_sse_stream(source, policy, limiter)]


@pytest.mark.asyncio
async def test_exact_utf8_byte_limit_passes_and_plus_one_fails() -> None:
    # Given two chunks whose encoded sizes differ by one byte at a UTF-8 boundary.
    async def exact() -> AsyncGenerator[str, None]:
        yield "data: é\n\n"

    async def over() -> AsyncGenerator[str, None]:
        yield "data: éx\n\n"

    exact_chunk = "data: é\n\n"
    byte_limit = len(exact_chunk.encode("utf-8"))

    # When each stream is evaluated against the exact encoded-byte cap.
    assert await _collect(exact(), _policy(max_bytes=byte_limit)) == [exact_chunk]
    with pytest.raises(OpenAIStreamLimitExceeded, match="bytes"):
        await _collect(over(), _policy(max_bytes=byte_limit))


@pytest.mark.asyncio
async def test_exact_event_limit_passes_and_plus_one_fails() -> None:
    # Given one chunk at the event cap and another containing one extra data event.
    async def exact() -> AsyncGenerator[str, None]:
        yield "data: one\n\ndata: two\n\n"

    async def over() -> AsyncGenerator[str, None]:
        yield "data: one\n\ndata: two\n\ndata: three\n\n"

    # When the data-line counter enforces two events.
    assert len(await _collect(exact(), _policy(max_events=2))) == 1
    with pytest.raises(OpenAIStreamLimitExceeded, match="events"):
        await _collect(over(), _policy(max_events=2))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("idle", "total", "expected"),
    ((0.01, 0.20, "idle"), (0.20, 0.01, "total")),
)
async def test_idle_and_total_deadlines_report_distinct_boundaries(
    idle: float,
    total: float,
    expected: str,
) -> None:
    # Given a source that never produces its first event.
    async def stalled() -> AsyncGenerator[str, None]:
        await anyio.sleep_forever()
        yield "unreachable"

    policy = OpenAIStreamPolicy(100, 10, idle, total)

    # When the earlier policy deadline expires.
    with pytest.raises(OpenAIStreamTimedOut, match=expected) as raised:
        await _collect(stalled(), policy)

    # Then callers can distinguish the exhausted boundary without provider detail.
    assert raised.value.boundary == expected


@pytest.mark.asyncio
async def test_consumer_disconnect_closes_source_and_releases_capacity() -> None:
    # Given a source with a close sentinel and a single shared capacity slot.
    closed = anyio.Event()
    limiter = anyio.CapacityLimiter(1)

    async def source() -> AsyncGenerator[str, None]:
        try:
            yield "data: first\n\n"
            yield "data: must-not-be-read\n\n"
        finally:
            closed.set()

    # When the downstream consumer closes immediately after its first event.
    bounded = bounded_openai_sse_stream(source(), _policy(), limiter)
    assert await bounded.__anext__() == "data: first\n\n"
    await bounded.aclose()

    # Then the provider generator closes and a subsequent stream acquires the slot.
    assert closed.is_set()

    async def healthy() -> AsyncGenerator[str, None]:
        yield "data: recovered\n\n"

    assert await _collect(healthy(), _policy(), limiter) == ["data: recovered\n\n"]


@pytest.mark.asyncio
async def test_total_deadline_releases_capacity_while_consumer_is_paused() -> None:
    closed = anyio.Event()
    limiter = anyio.CapacityLimiter(1)
    policy = OpenAIStreamPolicy(100, 10, 1.0, 0.02)

    async def stalled_after_first() -> AsyncGenerator[str, None]:
        try:
            yield "data: first\n\n"
            await anyio.sleep_forever()
        finally:
            closed.set()

    paused = bounded_openai_sse_stream(stalled_after_first(), policy, limiter)
    assert await paused.__anext__() == "data: first\n\n"
    await anyio.sleep(0.05)

    async def healthy() -> AsyncGenerator[str, None]:
        yield "data: recovered\n\n"

    assert await _collect(healthy(), _policy(), limiter) == ["data: recovered\n\n"]
    assert closed.is_set()
    await paused.aclose()


@pytest.mark.asyncio
async def test_buffered_accepted_chunk_precedes_later_source_error() -> None:
    failed = anyio.Event()

    async def fails_after_two() -> AsyncGenerator[str, None]:
        yield "data: first\n\n"
        yield "data: second\n\n"
        try:
            raise RuntimeError("source failed")
        finally:
            failed.set()

    bounded = bounded_openai_sse_stream(fails_after_two(), _policy(), anyio.CapacityLimiter(1))
    assert await bounded.__anext__() == "data: first\n\n"
    await failed.wait()
    await anyio.sleep(0)

    assert await bounded.__anext__() == "data: second\n\n"
    with pytest.raises(RuntimeError, match="source failed"):
        await bounded.__anext__()


@pytest.mark.asyncio
async def test_hung_source_cleanup_is_bounded_and_capacity_recovers() -> None:
    limiter = anyio.CapacityLimiter(1)
    close_calls = 0

    class HungCloseSource:
        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            await anyio.sleep_forever()
            raise StopAsyncIteration

        async def aclose(self) -> None:
            nonlocal close_calls
            close_calls += 1
            await anyio.sleep_forever()

    policy = OpenAIStreamPolicy(100, 10, 0.20, 0.01, cleanup_timeout_seconds=0.01)
    with anyio.fail_after(0.1):
        with pytest.raises(OpenAIStreamTimedOut, match="total"):
            await _collect(HungCloseSource(), policy, limiter)  # type: ignore[arg-type]

    assert close_calls == 1
    assert limiter.borrowed_tokens == 0

    async def healthy() -> AsyncGenerator[str, None]:
        yield "data: recovered\n\n"

    assert await _collect(healthy(), _policy(), limiter) == ["data: recovered\n\n"]
