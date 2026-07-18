from __future__ import annotations

import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from starlette.testclient import TestClient

from test_local_transport_safe_mode import (
    FAKE_XAI_AUTHORITY,
    _block_catalog_fetches,
    _fake_openai_upstream,
    _import_proxy_main,
)


@contextmanager
def _hostile_proxy():
    requests: list[dict[str, str | bool]] = []

    class Handler(BaseHTTPRequestHandler):
        def _reject(self) -> None:
            requests.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "authorization_present": "Authorization" in self.headers,
                }
            )
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_CONNECT = _reject
        do_GET = _reject
        do_POST = _reject

        def log_message(self, _format, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("stream", (False, True))
def test_safe_local_xai_transport_ignores_ambient_proxy_environment(
    monkeypatch,
    tmp_path,
    stream: bool,
) -> None:
    with (
        _hostile_proxy() as (proxy_port, proxy_requests),
        _fake_openai_upstream() as (
            upstream_port,
            upstream_requests,
        ),
    ):
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.setenv(key, proxy_url)
        for key in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)

        module = _import_proxy_main(
            monkeypatch,
            tmp_path,
            f"http://{FAKE_XAI_AUTHORITY}:{upstream_port}/v1",
        )
        catalog_attempts = _block_catalog_fetches(monkeypatch)

        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
                json={
                    "model": "xai_oauth/grok-4.5",
                    "messages": [{"role": "user", "content": "proxy isolation"}],
                    "stream": stream,
                },
            )

    assert response.status_code == 200
    assert catalog_attempts == []
    assert proxy_requests == []
    assert len(upstream_requests) == 1
    assert upstream_requests[0]["path"] == "/v1/chat/completions"
    assert upstream_requests[0]["authorization"] == "Bearer fake-xai-key"
    assert upstream_requests[0]["body"].get("stream", False) is stream


@pytest.mark.parametrize("stream", (False, True))
def test_safe_local_xai_transport_does_not_follow_hostile_redirect(
    monkeypatch,
    tmp_path,
    stream: bool,
) -> None:
    with (
        _hostile_proxy() as (hostile_port, hostile_requests),
        _fake_openai_upstream(redirect_url=f"http://127.0.0.1:{hostile_port}/capture") as (
            upstream_port,
            upstream_requests,
        ),
    ):
        module = _import_proxy_main(
            monkeypatch,
            tmp_path,
            f"http://{FAKE_XAI_AUTHORITY}:{upstream_port}/v1",
        )
        catalog_attempts = _block_catalog_fetches(monkeypatch)

        with TestClient(
            module.app,
            base_url="http://127.0.0.1",
            raise_server_exceptions=False,
        ) as client:
            module.app.state.rotating_client.max_retries = 1
            client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
                json={
                    "model": "xai_oauth/grok-4.5",
                    "messages": [{"role": "user", "content": "redirect isolation"}],
                    "stream": stream,
                },
            )

    assert catalog_attempts == []
    assert len(upstream_requests) == 1
    assert upstream_requests[0]["body"].get("stream", False) is stream
    assert hostile_requests == []
