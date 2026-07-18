from __future__ import annotations

from collections.abc import AsyncGenerator
import json

import anyio
import pytest

import proxy_app.main as module
import proxy_app.anthropic_stream as anthropic_stream_module
from rotator_library.anthropic_compat.streaming import anthropic_streaming_wrapper


class ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


class DisconnectedRequest:
    async def is_disconnected(self) -> bool:
        return True


def test_main_reexports_extracted_anthropic_stream_wrapper() -> None:
    from proxy_app.anthropic_stream import bounded_anthropic_sse_response

    assert module.anthropic_streaming_response_wrapper is bounded_anthropic_sse_response


def _event(content: str = "ok") -> str:
    payload = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": content}}
    return f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n"


async def _collect(source: AsyncGenerator[str, None], request=None) -> str:
    return "".join(
        [
            chunk
            async for chunk in module.anthropic_streaming_response_wrapper(
                source, request=request or ConnectedRequest()
            )
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("boundary", "value"), (("bytes", 64), ("events", 1)))
async def test_anthropic_stream_enforces_size_boundaries_and_closes(
    monkeypatch, boundary: str, value: int
) -> None:
    closed = 0

    async def source() -> AsyncGenerator[str, None]:
        nonlocal closed
        try:
            if boundary == "bytes":
                yield _event("x" * 128)
            else:
                yield _event("first")
                yield _event("second")
        finally:
            closed += 1

    if boundary == "bytes":
        monkeypatch.setattr(anthropic_stream_module, "ANTHROPIC_STREAM_MAX_BYTES", value)
    else:
        monkeypatch.setattr(anthropic_stream_module, "ANTHROPIC_STREAM_MAX_EVENTS", value)

    body = await _collect(source())

    assert "x" * 128 not in body
    assert "second" not in body
    assert "event: error" in body
    assert '"type": "api_error"' in body
    assert closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("idle", "total"), ((0.01, 0.20), (0.20, 0.01)))
async def test_anthropic_stream_enforces_idle_and_total_deadlines(
    monkeypatch, idle: float, total: float
) -> None:
    closed = 0

    async def stalled() -> AsyncGenerator[str, None]:
        nonlocal closed
        try:
            await anyio.sleep_forever()
            yield "unreachable"
        finally:
            closed += 1

    monkeypatch.setattr(anthropic_stream_module, "ANTHROPIC_STREAM_IDLE_TIMEOUT_SECONDS", idle)
    monkeypatch.setattr(anthropic_stream_module, "ANTHROPIC_STREAM_TOTAL_TIMEOUT_SECONDS", total)

    with anyio.fail_after(0.5):
        body = await _collect(stalled())

    assert "event: error" in body
    assert closed == 1


@pytest.mark.asyncio
async def test_anthropic_stream_capacity_rejects_without_entering_and_recovers(monkeypatch) -> None:
    entered = anyio.Event()
    release = anyio.Event()
    rejected_entered = anyio.Event()
    limiter = anyio.CapacityLimiter(1)

    async def occupied() -> AsyncGenerator[str, None]:
        entered.set()
        await release.wait()
        yield _event("occupied")

    async def candidate() -> AsyncGenerator[str, None]:
        rejected_entered.set()
        yield _event("candidate")

    async def consume_occupied() -> None:
        await _collect(occupied())

    monkeypatch.setattr(anthropic_stream_module, "ANTHROPIC_STREAM_CAPACITY", limiter)
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(consume_occupied)
        await entered.wait()
        rejected = await _collect(candidate())
        release.set()

    assert not rejected_entered.is_set()
    assert "event: error" in rejected
    assert "candidate" in await _collect(candidate())


@pytest.mark.asyncio
async def test_anthropic_disconnect_closes_once_without_forwarding() -> None:
    closed = 0

    async def source() -> AsyncGenerator[str, None]:
        nonlocal closed
        try:
            yield _event("private")
            yield _event("must-not-prefetch")
        finally:
            closed += 1

    body = await _collect(source(), DisconnectedRequest())

    assert body == ""
    assert closed == 1


@pytest.mark.asyncio
async def test_anthropic_stream_drains_accepted_order_before_sanitized_error(caplog) -> None:
    sentinel = "private-provider-sentinel"

    async def source() -> AsyncGenerator[str, None]:
        yield _event("first")
        yield _event("second")
        raise RuntimeError(sentinel)

    body = await _collect(source())

    assert body.index("first") < body.index("second") < body.index("event: error")
    assert sentinel not in body
    assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_anthropic_conversion_sanitizes_failure_and_closes_upstream_once(caplog) -> None:
    sentinel = "nested-private-provider-sentinel"
    closed = 0

    async def source() -> AsyncGenerator[str, None]:
        nonlocal closed
        try:
            raise RuntimeError(sentinel)
            yield "unreachable"
        finally:
            closed += 1

    body = "".join(
        [
            chunk
            async for chunk in anthropic_streaming_wrapper(
                source(), original_model="provider/model"
            )
        ]
    )

    assert "event: error" in body
    assert "Internal server error" in body
    assert sentinel not in body
    assert sentinel not in caplog.text
    assert closed == 1


@pytest.mark.asyncio
async def test_anthropic_conversion_bounds_hung_upstream_cleanup(monkeypatch, caplog) -> None:
    close_calls = 0

    class HungCloseSource:
        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            nonlocal close_calls
            close_calls += 1
            await anyio.sleep_forever()

    monkeypatch.setattr(
        "rotator_library.anthropic_compat.streaming.STREAM_CLEANUP_TIMEOUT_SECONDS", 0.01
    )
    source = HungCloseSource()
    with anyio.fail_after(0.1):
        body = "".join(
            [
                chunk
                async for chunk in anthropic_streaming_wrapper(
                    source, original_model="provider/model"  # type: ignore[arg-type]
                )
            ]
        )

    assert "event: error" in body
    assert "event: message_stop" not in body
    assert close_calls == 1
    assert "cleanup failed" not in caplog.text.lower()


@pytest.mark.asyncio
async def test_anthropic_total_deadline_releases_capacity_while_consumer_pauses(
    monkeypatch,
) -> None:
    closed = anyio.Event()
    limiter = anyio.CapacityLimiter(1)

    async def stalled_after_first() -> AsyncGenerator[str, None]:
        try:
            yield _event("first")
            await anyio.sleep_forever()
        finally:
            closed.set()

    monkeypatch.setattr(anthropic_stream_module, "ANTHROPIC_STREAM_CAPACITY", limiter)
    monkeypatch.setattr(anthropic_stream_module, "ANTHROPIC_STREAM_IDLE_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(anthropic_stream_module, "ANTHROPIC_STREAM_TOTAL_TIMEOUT_SECONDS", 0.02)
    wrapped = module.anthropic_streaming_response_wrapper(
        stalled_after_first(), request=ConnectedRequest()
    )
    assert "first" in await wrapped.__anext__()
    await anyio.sleep(0.05)

    async def healthy() -> AsyncGenerator[str, None]:
        yield _event("recovered")

    assert "recovered" in await _collect(healthy())
    assert closed.is_set()
    await wrapped.aclose()
