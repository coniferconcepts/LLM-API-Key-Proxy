from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def boundary_module():
    return importlib.import_module("proxy_app.request_boundary")


async def _invoke(
    middleware,
    body_chunks: list[bytes],
    headers: list[tuple[bytes, bytes]],
) -> tuple[list[dict[str, Any]], list[bytes]]:
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(body_chunks) - 1,
        }
        for index, chunk in enumerate(body_chunks)
    ]
    if not messages:
        messages.append({"type": "http.request", "body": b"", "more_body": False})
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return messages.pop(0)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/json",
        "headers": headers,
    }
    await middleware(scope, receive, send)
    replayed = getattr(middleware.app, "replayed", [])
    return sent, replayed


class RecordingApp:
    def __init__(self) -> None:
        self.calls = 0
        self.replayed: list[bytes] = []

    async def __call__(self, _scope, receive, send) -> None:
        self.calls += 1
        message = await receive()
        self.replayed.append(message.get("body", b""))
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


def _status(messages: list[dict[str, Any]]) -> int:
    return next(
        message["status"] for message in messages if message["type"] == "http.response.start"
    )


@pytest.mark.asyncio
async def test_exact_cap_and_no_content_length_replay_once(boundary_module) -> None:
    body = b'{"value":"' + (b"x" * 51) + b'"}'
    assert len(body) == 63
    app = RecordingApp()
    middleware = boundary_module.BoundedJSONBodyMiddleware(app, paths={"/json"}, max_bytes=63)

    sent, replayed = await _invoke(middleware, [body[:7], body[7:]], [])

    assert _status(sent) == 204
    assert replayed == [body]
    assert app.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body_chunks", "headers", "expected_status"),
    (
        ([b'{"value":"' + (b"x" * 52) + b'"}'], [], 413),
        ([b"{" + (b" " * 31), b" " * 32], [], 413),
        ([b"{}"], [(b"content-length", b"3")], 400),
        (
            [b"{}"],
            [(b"content-length", b"2"), (b"transfer-encoding", b"chunked")],
            400,
        ),
        ([b"{}"], [(b"content-length", b"2"), (b"content-length", b"3")], 400),
    ),
)
async def test_oversize_and_framing_ambiguity_never_dispatch(
    boundary_module,
    body_chunks: list[bytes],
    headers: list[tuple[bytes, bytes]],
    expected_status: int,
) -> None:
    app = RecordingApp()
    middleware = boundary_module.BoundedJSONBodyMiddleware(app, paths={"/json"}, max_bytes=63)

    sent, _replayed = await _invoke(middleware, body_chunks, headers)

    assert _status(sent) == expected_status
    assert app.calls == 0
    assert "SECRET" not in repr(sent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "headers"),
    (
        ("/json", []),
        ("/json", [(b"authorization", b"Bearer wrong")]),
        (
            "/json",
            [
                (b"authorization", b"Bearer proxy-token"),
                (b"authorization", b"Bearer proxy-token"),
            ],
        ),
        (
            "/v1/messages",
            [
                (b"authorization", b"Bearer proxy-token"),
                (b"x-api-key", b"proxy-token"),
            ],
        ),
    ),
)
async def test_authentication_failure_rejects_without_reading_body(
    boundary_module,
    path: str,
    headers: list[tuple[bytes, bytes]],
) -> None:
    app = RecordingApp()
    middleware = boundary_module.BoundedJSONBodyMiddleware(
        app,
        paths={"/json", "/v1/messages"},
        max_bytes=63,
        credential_config_getter=lambda: ("proxy-token", True),
    )
    reads = 0
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal reads
        reads += 1
        return {"type": "http.request", "body": b'{"value":1}', "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(
        {"type": "http", "method": "POST", "path": path, "headers": headers},
        receive,
        send,
    )

    assert _status(sent) == 401
    assert reads == 0
    assert app.calls == 0


@pytest.mark.asyncio
async def test_rotated_credential_config_is_read_at_request_time(boundary_module) -> None:
    state: dict[str, Any] = {"key": None, "current": True}
    app = RecordingApp()
    middleware = boundary_module.BoundedJSONBodyMiddleware(
        app,
        paths={"/json"},
        max_bytes=63,
        credential_config_getter=lambda: (state["key"], state["current"]),
    )
    state["key"] = "new-key"

    sent, replayed = await _invoke(
        middleware,
        [b'{"value":1}'],
        [(b"authorization", b"Bearer new-key")],
    )

    assert _status(sent) == 204
    assert replayed == [b'{"value":1}']

    reads = 0
    rejected: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal reads
        reads += 1
        return {"type": "http.request", "body": b'{"value":1}', "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        rejected.append(message)

    await middleware(
        {
            "type": "http",
            "method": "POST",
            "path": "/json",
            "headers": [(b"authorization", b"Bearer stale-key")],
        },
        receive,
        send,
    )

    assert _status(rejected) == 401
    assert reads == 0
