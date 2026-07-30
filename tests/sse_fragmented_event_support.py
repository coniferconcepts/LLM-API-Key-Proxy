from __future__ import annotations

from collections.abc import AsyncGenerator

import anyio
from proxy_app.stream_bounds import SSEStreamPolicy, bounded_sse_stream
from starlette.requests import Request
from starlette.types import Scope


def policy(
    *, max_events: int = 2, max_bytes: int = 1_000, total_timeout_seconds: float = 1.0
) -> SSEStreamPolicy:
    return SSEStreamPolicy(
        max_bytes=max_bytes,
        max_events=max_events,
        idle_timeout_seconds=0.1,
        total_timeout_seconds=total_timeout_seconds,
    )


async def collect(chunks: tuple[str, ...], stream_policy: SSEStreamPolicy) -> list[str]:
    async def source() -> AsyncGenerator[str, None]:
        for chunk in chunks:
            yield chunk

    return [
        chunk
        async for chunk in bounded_sse_stream(source(), stream_policy, anyio.CapacityLimiter(1))
    ]


def sse_records(body: str) -> list[dict[str, str]]:
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


class ConnectedRequest(Request):
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
