from __future__ import annotations

from collections.abc import AsyncGenerator
import json

import anyio
import pytest
from starlette.requests import Request
from starlette.types import Message, Scope

import proxy_app.anthropic_stream as anthropic_stream
import proxy_app.main as main
from rotator_library.anthropic_compat.streaming import anthropic_streaming_wrapper


def _request(*, disconnected: bool = False) -> Request:
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
    }

    async def receive() -> Message:
        return {"type": "http.disconnect"} if disconnected else {"type": "http.request"}

    return Request(scope, receive)


def _records(body: str) -> list[tuple[str | None, str]]:
    records: list[tuple[str | None, str]] = []
    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    for block in normalized.split("\n\n"):
        event: str | None = None
        data: list[str] = []
        for line in block.splitlines():
            field, separator, value = line.partition(":")
            if not separator:
                continue
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event = value
            elif field == "data":
                data.append(value)
        if event is not None or data:
            records.append((event, "\n".join(data)))
    return records


def _openai_delta(**delta: str | list[dict[str, object]]) -> str:
    payload = {"choices": [{"delta": delta}]}
    return f"data: {json.dumps(payload)}\n\n"


async def _collect(source: AsyncGenerator[str, None]) -> str:
    return "".join([chunk async for chunk in source])


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", (False, True))
async def test_openai_clean_eof_emits_one_sanitized_failure_terminal_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    partial: bool,
) -> None:
    closed = 0
    limiter = anyio.CapacityLimiter(1)

    async def truncated() -> AsyncGenerator[str, None]:
        nonlocal closed
        try:
            if partial:
                yield _openai_delta(content="accepted")
        finally:
            closed += 1

    async def healthy() -> AsyncGenerator[str, None]:
        yield _openai_delta(content="recovered")
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(main, "OPENAI_STREAM_CAPACITY", limiter)
    body = await _collect(main.streaming_response_wrapper(_request(), {}, truncated()))
    recovered = await _collect(main.streaming_response_wrapper(_request(), {}, healthy()))

    records = _records(body)
    assert ("accepted" in body) is partial
    assert sum('"code": "stream_error"' in data for _, data in records) == 1
    assert records[-1] == (None, "[DONE]")
    assert closed == 1
    assert "recovered" in recovered
    assert '"code": "stream_error"' not in recovered


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", (False, True))
async def test_anthropic_clean_eof_emits_error_without_success_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    partial: bool,
) -> None:
    closed = 0
    limiter = anyio.CapacityLimiter(1)

    async def truncated() -> AsyncGenerator[str, None]:
        nonlocal closed
        try:
            if partial:
                yield 'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'
        finally:
            closed += 1

    async def healthy() -> AsyncGenerator[str, None]:
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    monkeypatch.setattr(anthropic_stream, "ANTHROPIC_STREAM_CAPACITY", limiter)
    body = await _collect(
        anthropic_stream.bounded_anthropic_sse_response(truncated(), request=_request())
    )
    recovered = await _collect(
        anthropic_stream.bounded_anthropic_sse_response(healthy(), request=_request())
    )

    records = _records(body)
    assert ("content_block_delta" in body) is partial
    assert [event for event, _ in records if event in {"message_delta", "message_stop"}] == []
    assert records[-1][0] == "error"
    assert json.loads(records[-1][1])["error"]["type"] == "api_error"
    assert closed == 1
    assert _records(recovered)[-1][0] == "message_stop"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delta",
    (
        None,
        {"content": "accepted text"},
        {"reasoning_content": "accepted thought"},
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "tool-safe-id",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ]
        },
    ),
)
async def test_anthropic_conversion_exception_is_unambiguously_failed(
    delta: dict[str, str | list[dict[str, object]]] | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "private-upstream-terminal-sentinel"
    closed = 0

    async def source() -> AsyncGenerator[str, None]:
        nonlocal closed
        try:
            if delta is not None:
                yield _openai_delta(**delta)
            raise RuntimeError(sentinel)
        finally:
            closed += 1

    body = await _collect(anthropic_streaming_wrapper(source(), "provider/model"))
    records = _records(body)
    event_names = [event for event, _ in records]
    starts = [
        json.loads(data)["index"] for event, data in records if event == "content_block_start"
    ]
    stops = [json.loads(data)["index"] for event, data in records if event == "content_block_stop"]

    assert "message_delta" not in event_names
    assert "message_stop" not in event_names
    assert event_names[-1] == "error"
    assert len(starts) == len(set(starts))
    assert stops == starts
    assert sentinel not in body
    assert sentinel not in caplog.text
    assert closed == 1


@pytest.mark.asyncio
async def test_anthropic_conversion_clean_eof_is_an_error_not_success() -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield _openai_delta(content="accepted")

    body = await _collect(anthropic_streaming_wrapper(source(), "provider/model"))
    records = _records(body)
    event_names = [event for event, _ in records]

    assert "accepted" in body
    assert "message_delta" not in event_names
    assert "message_stop" not in event_names
    assert event_names[-1] == "error"


@pytest.mark.asyncio
async def test_in_process_anthropic_pipeline_emits_exactly_one_failure_terminal() -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield _openai_delta(content="accepted")
        raise RuntimeError("private-pipeline-sentinel")

    converted = anthropic_streaming_wrapper(source(), "provider/model")
    body = await _collect(
        anthropic_stream.bounded_anthropic_sse_response(converted, request=_request())
    )
    records = _records(body)
    event_names = [event for event, _ in records]

    assert event_names.count("error") == 1
    assert "message_delta" not in event_names
    assert "message_stop" not in event_names
    assert "private-pipeline-sentinel" not in body


@pytest.mark.asyncio
async def test_disconnect_remains_silent_instead_of_becoming_protocol_failure() -> None:
    async def openai_source() -> AsyncGenerator[str, None]:
        yield _openai_delta(content="must-not-forward")

    async def anthropic_source() -> AsyncGenerator[str, None]:
        yield 'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'

    openai_body = await _collect(
        main.streaming_response_wrapper(_request(disconnected=True), {}, openai_source())
    )
    anthropic_body = await _collect(
        anthropic_stream.bounded_anthropic_sse_response(
            anthropic_source(), request=_request(disconnected=True)
        )
    )

    assert openai_body == ""
    assert anthropic_body == ""


@pytest.mark.asyncio
async def test_openai_terminal_is_recognized_at_every_fragment_boundary() -> None:
    wire = "data: [DONE]\r\n\r\n"
    for split_at in range(1, len(wire)):

        async def source() -> AsyncGenerator[str, None]:
            yield wire[:split_at]
            yield wire[split_at:]

        body = await _collect(main.streaming_response_wrapper(_request(), {}, source()))

        assert _records(body) == [(None, "[DONE]")]
        assert '"code": "stream_error"' not in body


@pytest.mark.asyncio
async def test_anthropic_terminal_is_recognized_at_every_fragment_boundary() -> None:
    wire = 'event: message_stop\r\ndata: {"type":"message_stop"}\r\n\r\n'
    for split_at in range(1, len(wire)):

        async def source() -> AsyncGenerator[str, None]:
            yield wire[:split_at]
            yield wire[split_at:]

        body = await _collect(
            anthropic_stream.bounded_anthropic_sse_response(source(), request=_request())
        )

        assert _records(body)[-1][0] == "message_stop"
        assert "event: error" not in body


@pytest.mark.asyncio
async def test_openai_terminal_rejects_multiline_data_and_discards_trailing_records() -> None:
    async def ambiguous() -> AsyncGenerator[str, None]:
        yield "data: [DONE]\ndata: extra\n\n"

    async def terminal_with_trailer() -> AsyncGenerator[str, None]:
        yield "data: [DONE]\n\ndata: must-not-forward\n\n"

    ambiguous_body = await _collect(main.streaming_response_wrapper(_request(), {}, ambiguous()))
    terminal_body = await _collect(
        main.streaming_response_wrapper(_request(), {}, terminal_with_trailer())
    )

    assert '"code": "stream_error"' in ambiguous_body
    assert "must-not-forward" not in terminal_body
    assert _records(terminal_body) == [(None, "[DONE]")]


@pytest.mark.asyncio
async def test_anthropic_error_terminal_discards_same_chunk_success_trailer() -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield (
            'event: error\ndata: {"type":"error"}\n\n'
            'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )

    body = await _collect(
        anthropic_stream.bounded_anthropic_sse_response(source(), request=_request())
    )

    assert [event for event, _ in _records(body)] == ["error"]
