from __future__ import annotations

from collections.abc import AsyncGenerator
import json

import pytest
from starlette.requests import Request
from starlette.types import Message, Scope

import proxy_app.anthropic_stream as anthropic_stream
import proxy_app.main as main
from rotator_library.anthropic_compat.streaming import anthropic_streaming_wrapper
from rotator_library.stream_terminal import sse_data_content


def _request() -> Request:
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
    }

    async def receive() -> Message:
        return {"type": "http.request"}

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


def _delta(text: str) -> str:
    payload = {"choices": [{"delta": {"content": text}}]}
    return f"data: {json.dumps(payload)}\n\n"


async def _collect(source: AsyncGenerator[str, None]) -> str:
    return "".join([chunk async for chunk in source])


@pytest.mark.asyncio
async def test_openai_incomplete_terminal_line_is_not_completed_by_error_suffix() -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield "data: [DONE]\n"

    body = await _collect(main.streaming_response_wrapper(_request(), {}, source()))
    records = _records(body)

    assert records[-1] == (None, "[DONE]")
    assert sum('"code": "stream_error"' in data for _, data in records) == 1
    assert body.index('"code": "stream_error"') < body.index("data: [DONE]")


@pytest.mark.asyncio
async def test_anthropic_incomplete_terminal_line_is_not_completed_by_error_suffix() -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield "event: message_stop\n"

    body = await _collect(
        anthropic_stream.bounded_anthropic_sse_response(source(), request=_request())
    )

    assert [event for event, _ in _records(body)] == ["error"]


@pytest.mark.asyncio
async def test_converter_incomplete_done_line_is_an_error_not_success() -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield "data: [DONE]\n"

    body = await _collect(anthropic_streaming_wrapper(source(), "provider/model"))
    events = [event for event, _ in _records(body)]

    assert "message_delta" not in events
    assert "message_stop" not in events
    assert events == ["error"]


@pytest.mark.asyncio
async def test_converter_done_is_valid_at_every_fragment_boundary() -> None:
    wire = _delta("accepted") + "data: [DONE]\r\n\r\n"
    for split_at in range(1, len(wire)):

        async def source() -> AsyncGenerator[str, None]:
            yield wire[:split_at]
            yield wire[split_at:]

        body = await _collect(anthropic_streaming_wrapper(source(), "provider/model"))
        events = [event for event, _ in _records(body)]

        assert "accepted" in body, split_at
        assert events[-2:] == ["message_delta", "message_stop"], split_at
        assert "error" not in events, split_at


@pytest.mark.asyncio
async def test_converter_parses_metadata_prefixed_records_at_every_boundary() -> None:
    wire = "id: 7\n" + _delta("accepted") + ": keepalive\ndata: [DONE]\n\n"
    for split_at in range(1, len(wire)):

        async def source() -> AsyncGenerator[str, None]:
            yield wire[:split_at]
            yield wire[split_at:]

        body = await _collect(anthropic_streaming_wrapper(source(), "provider/model"))
        events = [event for event, _ in _records(body)]

        assert "accepted" in body, split_at
        assert events[-2:] == ["message_delta", "message_stop"], split_at
        assert "error" not in events, split_at


@pytest.mark.asyncio
async def test_converter_handles_delta_and_done_in_one_chunk() -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield _delta("first") + _delta("second") + "data: [DONE]\n\n"

    body = await _collect(anthropic_streaming_wrapper(source(), "provider/model"))
    events = [event for event, _ in _records(body)]

    assert body.index("first") < body.index("second")
    assert events[-2:] == ["message_delta", "message_stop"]
    assert "error" not in events


@pytest.mark.asyncio
async def test_empty_disconnected_eof_is_silent_through_composed_pipeline() -> None:
    disconnected_checks = 0

    async def disconnected() -> bool:
        nonlocal disconnected_checks
        disconnected_checks += 1
        return disconnected_checks >= 2

    async def source() -> AsyncGenerator[str, None]:
        if False:
            yield ""

    class DisconnectedRequest:
        is_disconnected = staticmethod(disconnected)

    converted = anthropic_streaming_wrapper(
        source(), "provider/model", is_disconnected=disconnected
    )
    body = await _collect(
        anthropic_stream.bounded_anthropic_sse_response(converted, request=DisconnectedRequest())
    )

    assert body == ""
    assert disconnected_checks == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", ("\n", "\r\n", "\r"))
async def test_bare_data_prevents_false_done_at_every_boundary(newline: str) -> None:
    ambiguous = f"data{newline}data: [DONE]{newline}{newline}"
    for split_at in range(1, len(ambiguous)):

        async def source() -> AsyncGenerator[str, None]:
            yield ambiguous[:split_at]
            yield ambiguous[split_at:]

        body = await _collect(main.streaming_response_wrapper(_request(), {}, source()))

        assert '"code": "stream_error"' in body, split_at
        assert body.index('"code": "stream_error"') < body.rindex("data: [DONE]"), split_at


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", ("\n", "\r\n", "\r"))
async def test_converter_ignores_ambiguous_done_and_consumes_coalesced_terminal(
    newline: str,
) -> None:
    wire = f"data{newline}data: [DONE]{newline}{newline}" + _delta("accepted") + "data: [DONE]\n\n"
    for split_at in range(1, len(wire)):

        async def source() -> AsyncGenerator[str, None]:
            yield wire[:split_at]
            yield wire[split_at:]

        converted = anthropic_streaming_wrapper(source(), "provider/model")
        body = await _collect(
            anthropic_stream.bounded_anthropic_sse_response(converted, request=_request())
        )
        events = [event for event, _ in _records(body)]

        assert "accepted" in body, split_at
        assert events[-2:] == ["message_delta", "message_stop"], split_at
        assert "error" not in events, split_at


def test_bare_data_field_has_an_empty_compatible_value() -> None:
    assert sse_data_content("data\n\n") == ""
    assert sse_data_content("data\ndata: value\n\n") == "\nvalue"


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect_after_record", (False, True))
async def test_exception_rechecks_disconnect_before_protocol_failure(
    disconnect_after_record: bool,
) -> None:
    checks = 0

    async def disconnected() -> bool:
        nonlocal checks
        checks += 1
        return disconnect_after_record and checks >= 2

    async def source() -> AsyncGenerator[str, None]:
        yield _delta("accepted")
        raise RuntimeError("private-sentinel")

    body = await _collect(
        anthropic_streaming_wrapper(source(), "provider/model", is_disconnected=disconnected)
    )
    events = [event for event, _ in _records(body)]

    assert "accepted" in body
    if disconnect_after_record:
        assert "error" not in events
    else:
        assert events[-1] == "error"
