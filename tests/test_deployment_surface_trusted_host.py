from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from deployment_surface_helpers import ROOT, load_main_symbols as _load_main_symbols


@pytest.mark.parametrize(
    "host_headers",
    [
        [(b"host", b"api.example.com"), (b"host", b"attacker.example")],
        [(b"HOST", b"attacker.example"), (b"Host", b"api.example.com")],
    ],
)
@pytest.mark.parametrize("scope_type", ["http", "websocket"])
@pytest.mark.parametrize("allowed_hosts", [["*.example.com"], ["*"]])
def test_trusted_host_rejects_duplicate_raw_host_headers(
    host_headers: list[tuple[bytes, bytes]],
    scope_type: str,
    allowed_hosts: list[str],
) -> None:
    namespace = _load_main_symbols("TrustedHostMiddleware")
    middleware_type = namespace["TrustedHostMiddleware"]
    downstream_calls: list[str] = []

    async def downstream(_scope, _receive, _send):
        downstream_calls.append(scope_type)

    guarded = middleware_type(downstream, allowed_hosts=allowed_hosts)
    scope = {
        "type": scope_type,
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": host_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("0.0.0.0", 8000),
        "extensions": {"websocket.http.response": {}},
    }
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {
            "type": ("websocket.connect" if scope_type == "websocket" else "http.request"),
            "body": b"",
            "more_body": False,
        }

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    asyncio.run(guarded(scope, receive, send))

    assert downstream_calls == []
    if scope_type == "websocket":
        assert sent == [{"type": "websocket.close", "code": 1008}]
    else:
        assert any(message.get("status") == 400 for message in sent)


@pytest.mark.parametrize(
    "host",
    [
        "localhost:",
        "evil..example",
        ".example",
        "-bad.example",
        "bad-.example",
        "evil%2f.example",
        f"{'a' * 64}.example",
    ],
)
def test_approved_match_all_wildcard_still_rejects_malformed_authority(
    host: str,
) -> None:
    namespace = _load_main_symbols("TrustedHostMiddleware")
    middleware_type = namespace["TrustedHostMiddleware"]

    async def ok(_request):
        return PlainTextResponse("ok")

    test_app = Starlette(routes=[Route("/", ok)])
    guarded = middleware_type(test_app, allowed_hosts=["*"])

    with TestClient(guarded) as client:
        valid = client.get("/", headers={"host": "arbitrary.example"})
        malformed = client.get("/", headers={"host": host})

    assert valid.status_code == 200
    assert malformed.status_code == 400


def test_trusted_host_guard_precedes_cors_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SKIP_OAUTH_INIT_CHECK", "true")
    monkeypatch.setenv("MIRROWEL_LOCAL_TRANSPORT_SAFE_MODE", "true")
    monkeypatch.setenv("MIRROWEL_ALLOW_NETWORK_BIND", "true")
    monkeypatch.setenv("MIRROWEL_ALLOWED_HOSTS", "proxy.example.com")
    monkeypatch.setenv("PROXY_API_KEY", "test-token")
    monkeypatch.setattr(sys, "argv", ["pytest", "--port", "8000"])
    sys.path.insert(0, str(ROOT / "src"))
    try:
        sys.modules.pop("proxy_app.main", None)
        module = importlib.import_module("proxy_app.main")
    finally:
        sys.path.remove(str(ROOT / "src"))

    with TestClient(module.app) as client:
        response = client.options(
            "/v1/models",
            headers={
                "host": "evil..example",
                "origin": "http://localhost",
                "access-control-request-method": "GET",
            },
        )

    assert response.status_code == 400
    assert response.text == "Invalid host header"


@pytest.mark.parametrize("scope_type", ["http", "websocket"])
def test_trusted_host_rejects_single_empty_raw_host_header(scope_type: str) -> None:
    namespace = _load_main_symbols("TrustedHostMiddleware")
    middleware_type = namespace["TrustedHostMiddleware"]
    downstream_calls: list[str] = []

    async def downstream(_scope, _receive, _send):
        downstream_calls.append(scope_type)

    guarded = middleware_type(downstream, allowed_hosts=["proxy.example.com"])
    scope = {
        "type": scope_type,
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"  ")],
        "client": ("127.0.0.1", 12345),
        "server": ("0.0.0.0", 8000),
        "extensions": {"websocket.http.response": {}},
    }
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {
            "type": ("websocket.connect" if scope_type == "websocket" else "http.request"),
            "body": b"",
            "more_body": False,
        }

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    asyncio.run(guarded(scope, receive, send))

    assert downstream_calls == []
    if scope_type == "websocket":
        assert sent == [{"type": "websocket.close", "code": 1008}]
    else:
        assert any(message.get("status") == 400 for message in sent)


def test_trusted_host_rejects_websocket_without_http_response_extension() -> None:
    namespace = _load_main_symbols("TrustedHostMiddleware")
    middleware_type = namespace["TrustedHostMiddleware"]
    downstream_calls = 0

    async def downstream(_scope, _receive, _send):
        nonlocal downstream_calls
        downstream_calls += 1

    guarded = middleware_type(downstream, allowed_hosts=["proxy.example.com"])
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"attacker.example")],
        "client": ("127.0.0.1", 12345),
        "server": ("0.0.0.0", 8000),
        "extensions": {},
    }
    sent: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return {"type": "websocket.connect"}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    asyncio.run(guarded(scope, receive, send))

    assert downstream_calls == 0
    assert sent == [{"type": "websocket.close", "code": 1008}]
