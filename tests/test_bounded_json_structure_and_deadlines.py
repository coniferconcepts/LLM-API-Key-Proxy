from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
import sys
import time
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def boundary_module():
    return importlib.import_module("proxy_app.request_boundary")


class RecordingApp:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, _scope, _receive, send) -> None:
        self.calls += 1
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


def _status(messages: list[dict[str, Any]]) -> int:
    return next(
        message["status"] for message in messages if message["type"] == "http.response.start"
    )


async def _send_body(middleware, body: bytes) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(
        {"type": "http", "method": "POST", "path": "/json", "headers": []},
        receive,
        send,
    )
    return sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    (
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":1e999}',
        b'{"value":1234567890123456789012345678901234567890}',
        (b'{"x":' * 9) + b"0" + (b"}" * 9),
        b'{"a":[0,0,0,0,0,0,0,0,0,0]}',
    ),
)
async def test_numeric_depth_and_node_limits_reject_before_dispatch(
    boundary_module,
    body: bytes,
) -> None:
    app = RecordingApp()
    middleware = boundary_module.BoundedJSONBodyMiddleware(
        app,
        paths={"/json"},
        max_bytes=1024,
        max_depth=8,
        max_nodes=8,
        max_number_characters=16,
    )

    sent = await _send_body(middleware, body)

    assert _status(sent) == 400
    assert app.calls == 0


@pytest.mark.asyncio
async def test_total_deadline_cancels_an_actively_blocked_receive(
    boundary_module,
) -> None:
    app = RecordingApp()
    middleware = boundary_module.BoundedJSONBodyMiddleware(
        app,
        paths={"/json"},
        max_bytes=63,
        body_read_timeout_seconds=0.02,
        body_inter_chunk_timeout_seconds=1.0,
    )
    sent: list[dict[str, Any]] = []
    reads = 0
    cancelled = False

    async def receive() -> dict[str, Any]:
        nonlocal cancelled, reads
        reads += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(
        {"type": "http", "method": "POST", "path": "/json", "headers": []},
        receive,
        send,
    )

    assert _status(sent) == 408
    assert reads == 1
    assert cancelled is True
    assert app.calls == 0


@pytest.mark.asyncio
async def test_inter_chunk_deadline_cancels_an_actively_blocked_receive(
    boundary_module,
) -> None:
    app = RecordingApp()
    middleware = boundary_module.BoundedJSONBodyMiddleware(
        app,
        paths={"/json"},
        max_bytes=63,
        body_read_timeout_seconds=1.0,
        body_inter_chunk_timeout_seconds=0.02,
    )
    sent: list[dict[str, Any]] = []
    cancelled = False

    async def receive() -> dict[str, Any]:
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(
        {"type": "http", "method": "POST", "path": "/json", "headers": []},
        receive,
        send,
    )

    assert _status(sent) == 408
    assert cancelled is True
    assert app.calls == 0


@pytest.mark.asyncio
async def test_total_deadline_between_completed_receives_rejects_without_dispatch(
    boundary_module,
) -> None:
    app = RecordingApp()
    middleware = boundary_module.BoundedJSONBodyMiddleware(
        app,
        paths={"/json"},
        max_bytes=63,
        body_read_timeout_seconds=0.001,
        body_inter_chunk_timeout_seconds=1.0,
    )
    sent: list[dict[str, Any]] = []
    reads = 0

    async def receive() -> dict[str, Any]:
        nonlocal reads
        reads += 1
        stop = time.perf_counter() + 0.01
        while time.perf_counter() < stop:
            pass
        return {"type": "http.request", "body": b" ", "more_body": True}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(
        {"type": "http", "method": "POST", "path": "/json", "headers": []},
        receive,
        send,
    )

    assert _status(sent) == 408
    assert reads == 1
    assert app.calls == 0


@pytest.mark.asyncio
async def test_disconnect_rejects_without_dispatch(boundary_module) -> None:
    app = RecordingApp()
    middleware = boundary_module.BoundedJSONBodyMiddleware(app, paths={"/json"})
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(
        {"type": "http", "method": "POST", "path": "/json", "headers": []},
        receive,
        send,
    )

    assert _status(sent) == 400
    assert app.calls == 0


@pytest.mark.parametrize("timeout", (0.0, -1.0, float("inf"), float("nan")))
def test_invalid_body_timeout_configuration_is_rejected(boundary_module, timeout: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        boundary_module.BoundedJSONBodyMiddleware(
            RecordingApp(),
            paths={"/json"},
            body_read_timeout_seconds=timeout,
        )
