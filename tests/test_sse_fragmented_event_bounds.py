from __future__ import annotations

from collections.abc import AsyncGenerator
import json

import anyio
import pytest
from starlette.requests import Request
from starlette.types import Scope

import proxy_app.anthropic_stream as anthropic_stream
import proxy_app.main as main
from proxy_app.stream_bounds import (
    DEFAULT_SSE_STREAM_POLICY,
    SSEStreamLimitExceeded,
    SSEStreamPolicy,
    bounded_openai_sse_stream,
    bounded_sse_stream,
)


def _policy(
    *,
    max_events: int = 2,
    max_bytes: int = 1_000,
    total_timeout_seconds: float = 1.0,
) -> SSEStreamPolicy:
    return SSEStreamPolicy(
        max_bytes=max_bytes,
        max_events=max_events,
        idle_timeout_seconds=0.1,
        total_timeout_seconds=total_timeout_seconds,
    )


async def _collect(chunks: tuple[str, ...], policy: SSEStreamPolicy) -> list[str]:
    async def source() -> AsyncGenerator[str, None]:
        for chunk in chunks:
            yield chunk

    return [
        chunk
        async for chunk in bounded_sse_stream(
            source(),
            policy,
            anyio.CapacityLimiter(1),
        )
    ]


def _sse_records(body: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    for block in normalized.split("\n\n"):
        fields: dict[str, str] = {}
        data: list[str] = []
        for line in block.splitlines():
            if not line or line.startswith(":"):
                continue
            name, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if name == "data":
                data.append(value)
            elif name == "event":
                fields[name] = value
        if data:
            fields["data"] = "\n".join(data)
        if fields:
            records.append(fields)
    return records


@pytest.mark.asyncio
async def test_fragmented_data_field_names_cannot_bypass_event_limit() -> None:
    # Given three valid SSE data fields split inside every field name.
    chunks = ("da", "ta: one\n\n", "d", "ata: two\r\n\r\n", "dat", "a: three\r\r")

    # When the reassembled stream exceeds the two-field policy.
    with pytest.raises(SSEStreamLimitExceeded, match="events"):
        await _collect(chunks, _policy())


_WIRE_EVENT = "data: value\r\n\r\n"


class _ConnectedRequest(Request):
    def __init__(self) -> None:
        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "server": ("testserver", 80),
            "client": ("testclient", 50_000),
            "scheme": "http",
            "method": "POST",
            "root_path": "",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "headers": [],
            "state": {},
        }
        super().__init__(scope)

    async def is_disconnected(self) -> bool:
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize("split_at", range(1, len(_WIRE_EVENT)))
async def test_openai_event_bound_survives_every_wire_split(split_at: int) -> None:
    # Given three events fragmented at one of every possible wire boundaries.
    fragment = (_WIRE_EVENT[:split_at], _WIRE_EVENT[split_at:])
    chunks = fragment * 3

    # When the OpenAI alias enforces a two-field policy.
    async def source() -> AsyncGenerator[str, None]:
        for chunk in chunks:
            yield chunk

    with pytest.raises(SSEStreamLimitExceeded, match="events"):
        _ = [
            chunk
            async for chunk in bounded_openai_sse_stream(
                source(),
                _policy(),
                anyio.CapacityLimiter(1),
            )
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("split_at", range(1, len(_WIRE_EVENT)))
async def test_anthropic_event_bound_survives_every_wire_split(
    monkeypatch: pytest.MonkeyPatch,
    split_at: int,
) -> None:
    # Given an Anthropic stream fragmented at a field, colon, CR, LF, or terminator boundary.
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

    # When the protocol wrapper consumes the fragmented stream.
    body = "".join(
        [chunk async for chunk in anthropic_stream.bounded_anthropic_sse_response(source())]
    )

    # Then it closes once and emits only the sanitized terminal error after the bound.
    assert "event: error" in body
    assert '"type": "api_error"' in body
    assert closed == 1


@pytest.mark.asyncio
async def test_openai_fragment_limit_is_sanitized_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given three fragmented OpenAI fields and a two-field terminal-wrapper policy.
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

    # When the OpenAI terminal wrapper reaches the fragmented third field.
    body = "".join(
        [
            chunk
            async for chunk in main.streaming_response_wrapper(
                _ConnectedRequest(),
                {},
                source(),
            )
        ]
    )

    # Then provider content is withheld, the public error is fixed, and cleanup occurs once.
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
        [
            chunk
            async for chunk in main.streaming_response_wrapper(
                _ConnectedRequest(),
                {},
                source(),
            )
        ]
    )
    records = _sse_records(body)
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
    records = _sse_records(body)

    assert len(records) == 1
    assert records[0]["event"] == "error"
    assert json.loads(records[0]["data"])["error"]["type"] == "api_error"


@pytest.mark.asyncio
async def test_fragmented_stream_preserves_accepted_bytes_and_order() -> None:
    # Given mixed SSE fields split across field names, colons, CRLFs, and blank lines.
    chunks = ("eve", "nt: ping\r", "\nda", "ta", ": one\r\n", "data: two\n", "\n")

    # When the data-field count exactly equals the policy.
    accepted = await _collect(chunks, _policy(max_events=2))

    # Then the byte sequence and chunk order are unchanged.
    assert accepted == list(chunks)
    assert "".join(accepted) == "".join(chunks)


@pytest.mark.asyncio
async def test_multiline_and_bare_data_fields_share_one_definition() -> None:
    # Given one SSE event containing a colon field and a valid bare data field.
    chunks = ("data: first\n", "da", "ta\n\n")

    # When both data fields exceed a one-field policy.
    with pytest.raises(SSEStreamLimitExceeded, match="events"):
        await _collect(chunks, _policy(max_events=1))


@pytest.mark.asyncio
async def test_non_data_fields_and_prefixes_do_not_consume_the_bound() -> None:
    # Given comments, other fields, leading whitespace, and a data-name prefix.
    chunks = (": keepalive\n", "event: ping\r\n", " data: indented\n", "database: value\n\n")

    # When no data fields are allowed.
    accepted = await _collect(chunks, _policy(max_events=0))

    # Then every non-data byte is forwarded without a false-positive limit failure.
    assert accepted == list(chunks)


@pytest.mark.asyncio
async def test_many_tiny_fragments_cannot_turn_event_bound_into_byte_only_bound() -> None:
    # Given 100,001 fields split into 200,002 chunks below the normal byte ceiling.
    chunks = ("da", "ta:\n\n") * 100_001

    # When the stream exceeds the normal 100,000-field policy.
    with pytest.raises(SSEStreamLimitExceeded, match="events"):
        await _collect(
            chunks,
            DEFAULT_SSE_STREAM_POLICY,
        )


@pytest.mark.asyncio
async def test_fragment_limit_closes_once_and_capacity_recovers() -> None:
    # Given a fragmented over-limit stream sharing one capacity slot.
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

    # When the first stream exceeds the bound and the next stream starts.
    with pytest.raises(SSEStreamLimitExceeded, match="events"):
        _ = [chunk async for chunk in bounded_sse_stream(over_limit(), _policy(), limiter)]
    recovered = [chunk async for chunk in bounded_sse_stream(healthy(), _policy(), limiter)]

    # Then cleanup occurs exactly once and the shared slot is reusable.
    assert closed == 1
    assert recovered == ["data: recovered\n\n"]
