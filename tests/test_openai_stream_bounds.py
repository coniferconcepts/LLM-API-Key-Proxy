from __future__ import annotations

from collections.abc import AsyncGenerator
import json

import anyio
import pytest

import proxy_app.main as module


class ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


class DisconnectedRequest:
    async def is_disconnected(self) -> bool:
        return True


async def _collect(stream: AsyncGenerator[str, None]) -> str:
    return "".join([chunk async for chunk in stream])


def _data_event(content: str = "ok") -> str:
    payload = {"choices": [{"delta": {"content": content}}]}
    return f"data: {json.dumps(payload)}\n\n"


@pytest.mark.asyncio
async def test_stream_rejects_oversized_event_and_closes_upstream(monkeypatch) -> None:
    # Given a byte policy smaller than the first upstream event.
    closed = anyio.Event()

    async def oversized_stream() -> AsyncGenerator[str, None]:
        try:
            yield _data_event("x" * 128)
            yield "data: [DONE]\n\n"
        finally:
            closed.set()

    monkeypatch.setattr(module, "OPENAI_STREAM_MAX_BYTES", 64, raising=False)

    # When the terminal wrapper consumes the stream.
    body = await _collect(
        module.streaming_response_wrapper(ConnectedRequest(), {}, oversized_stream())
    )

    # Then no oversized provider event is forwarded and the source is closed.
    assert "x" * 128 not in body
    assert '"code": "stream_error"' in body
    assert body.endswith("data: [DONE]\n\n")
    assert closed.is_set()


@pytest.mark.asyncio
async def test_stream_rejects_too_many_events(monkeypatch) -> None:
    # Given an event policy allowing exactly one data event.
    async def two_event_stream() -> AsyncGenerator[str, None]:
        yield _data_event("first")
        yield _data_event("second")
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(module, "OPENAI_STREAM_MAX_EVENTS", 1, raising=False)

    # When the wrapper sees the second event.
    body = await _collect(
        module.streaming_response_wrapper(ConnectedRequest(), {}, two_event_stream())
    )

    # Then it forwards the allowed event and terminates with a fixed error.
    assert "first" in body
    assert "second" not in body
    assert '"code": "stream_error"' in body
    assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_stream_stall_hits_idle_deadline_and_closes_upstream(monkeypatch) -> None:
    # Given a stream that stalls after one valid event.
    closed = anyio.Event()

    async def stalled_stream() -> AsyncGenerator[str, None]:
        try:
            yield _data_event("first")
            await anyio.sleep_forever()
        finally:
            closed.set()

    monkeypatch.setattr(module, "OPENAI_STREAM_IDLE_TIMEOUT_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(module, "OPENAI_STREAM_TOTAL_TIMEOUT_SECONDS", 0.05, raising=False)

    # When the wrapper waits for the next event.
    with anyio.fail_after(0.25):
        body = await _collect(
            module.streaming_response_wrapper(ConnectedRequest(), {}, stalled_stream())
        )

    # Then it returns a sanitized terminal event within the policy deadline.
    assert "first" in body
    assert '"code": "stream_error"' in body
    assert closed.is_set()


@pytest.mark.asyncio
async def test_stream_capacity_applies_backpressure_and_recovers(monkeypatch) -> None:
    # Given one occupied stream slot and a second source with an observable entry point.
    first_entered = anyio.Event()
    release_first = anyio.Event()
    second_entered = anyio.Event()
    first_body: list[str] = []

    async def blocking_stream() -> AsyncGenerator[str, None]:
        first_entered.set()
        await release_first.wait()
        yield _data_event("first")
        yield "data: [DONE]\n\n"

    async def second_stream() -> AsyncGenerator[str, None]:
        second_entered.set()
        yield _data_event("second")
        yield "data: [DONE]\n\n"

    async def consume_first() -> None:
        first_body.append(
            await _collect(
                module.streaming_response_wrapper(ConnectedRequest(), {}, blocking_stream())
            )
        )

    monkeypatch.setattr(module, "OPENAI_STREAM_CAPACITY", anyio.CapacityLimiter(1), raising=False)

    # When a second stream starts while the only slot is occupied.
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(consume_first)
        await first_entered.wait()
        rejected_body = await _collect(
            module.streaming_response_wrapper(ConnectedRequest(), {}, second_stream())
        )
        release_first.set()

    # Then the second source is not consumed, and a later request remains healthy.
    assert not second_entered.is_set()
    assert '"code": "stream_error"' in rejected_body
    assert "first" in first_body[0]
    healthy_body = await _collect(
        module.streaming_response_wrapper(ConnectedRequest(), {}, second_stream())
    )
    assert "second" in healthy_body
    assert '"code": "stream_error"' not in healthy_body


@pytest.mark.asyncio
async def test_stream_policy_error_never_leaks_upstream_sentinel(monkeypatch, caplog) -> None:
    # Given an upstream exception containing provider-controlled secret material.
    sentinel = "SSE-UPSTREAM-SENTINEL-private-provider-detail"

    async def failing_stream() -> AsyncGenerator[str, None]:
        raise RuntimeError(sentinel)
        yield  # pragma: no cover

    monkeypatch.setattr(module, "OPENAI_STREAM_IDLE_TIMEOUT_SECONDS", 0.05, raising=False)

    # When the wrapper converts the failure to a terminal event.
    body = await _collect(
        module.streaming_response_wrapper(ConnectedRequest(), {}, failing_stream())
    )

    # Then only the fixed public envelope is observable.
    assert sentinel not in body
    assert sentinel not in caplog.text
    assert '"code": "stream_error"' in body
    assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_client_disconnect_closes_upstream_without_forwarding() -> None:
    # Given an upstream generator and a request that is already disconnected.
    closed = anyio.Event()

    async def connected_stream() -> AsyncGenerator[str, None]:
        try:
            yield _data_event("must-not-forward")
        finally:
            closed.set()

    # When the terminal wrapper observes the disconnect.
    body = await _collect(
        module.streaming_response_wrapper(DisconnectedRequest(), {}, connected_stream())
    )

    # Then it closes the source without forwarding provider content.
    assert body == ""
    assert closed.is_set()
