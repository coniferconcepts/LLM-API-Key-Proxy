from __future__ import annotations

from collections.abc import AsyncGenerator
import json

import pytest

import proxy_app.anthropic_stream as anthropic_stream
import proxy_app.main as main
from sse_fragmented_event_support import ConnectedRequest, sse_records


@pytest.mark.asyncio
async def test_openai_fragment_limit_is_sanitized_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = 0

    async def source() -> AsyncGenerator[str, None]:
        nonlocal closed
        try:
            for content in ("first", "second", "third"):
                yield "da"
                yield f'ta: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}\n\n'
        finally:
            closed += 1

    monkeypatch.setattr(main, "OPENAI_STREAM_MAX_EVENTS", 2)
    body = "".join(
        [chunk async for chunk in main.streaming_response_wrapper(ConnectedRequest(), {}, source())]
    )
    assert "first" in body
    assert "second" in body
    assert "third" not in body
    assert '"code": "stream_error"' in body
    assert body.endswith("data: [DONE]\n\n")
    assert closed == 1


@pytest.mark.asyncio
async def test_openai_fragment_limit_resynchronizes_terminal_sse_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield "da"
        yield "ta: over-limit\n\n"

    monkeypatch.setattr(main, "OPENAI_STREAM_MAX_EVENTS", 0)
    body = "".join(
        [chunk async for chunk in main.streaming_response_wrapper(ConnectedRequest(), {}, source())]
    )
    records = sse_records(body)
    error_records = [record for record in records if record.get("data") != "[DONE]"]
    assert len(error_records) == 1
    assert json.loads(error_records[0]["data"])["error"]["code"] == "stream_error"
    assert records[-1] == {"data": "[DONE]"}


@pytest.mark.asyncio
async def test_anthropic_fragment_limit_resynchronizes_terminal_sse_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def source() -> AsyncGenerator[str, None]:
        yield "da"
        yield "ta: over-limit\n\n"

    monkeypatch.setattr(anthropic_stream, "ANTHROPIC_STREAM_MAX_EVENTS", 0)
    body = "".join(
        [chunk async for chunk in anthropic_stream.bounded_anthropic_sse_response(source())]
    )
    records = sse_records(body)
    assert len(records) == 1
    assert records[0]["event"] == "error"
    assert json.loads(records[0]["data"])["error"]["type"] == "api_error"
