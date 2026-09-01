import json
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
def _trap_upstream():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": json.loads(body.decode("utf-8")),
                }
            )
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, _format, *_args):
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_base", None),
        ("base_url", None),
        ("user_config", {"api_base": None}),
    ],
)
def test_chat_rejects_client_routing_overrides_without_contacting_trap(
    monkeypatch,
    tmp_path,
    field,
    value,
):
    with _trap_upstream() as (trap_port, trap_requests):
        trap_base = f"http://127.0.0.1:{trap_port}/v1"
        override = trap_base if value is None else {"api_base": trap_base}
        module = _import_proxy_main(
            monkeypatch,
            tmp_path,
            f"http://{FAKE_XAI_AUTHORITY}:2465/v1",
        )
        _block_catalog_fetches(monkeypatch)
        dispatches = 0

        async def record_dispatch(*_args, **_kwargs):
            nonlocal dispatches
            dispatches += 1
            raise AssertionError("blocked request dispatched")

        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            monkeypatch.setattr(module.app.state.rotating_client, "acompletion", record_dispatch)
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
                json={
                    "model": "xai_oauth/grok-4.5",
                    "messages": [{"role": "user", "content": "do not dispatch"}],
                    "stream": False,
                    field: override,
                },
            )

    assert response.status_code == 400
    assert field in response.json()["detail"]
    assert dispatches == 0
    assert trap_requests == []
    assert "fake-xai-key" not in "".join(map(str, trap_requests))


@pytest.mark.parametrize(
    "payload",
    [
        {"api_key": "client-key"},
        {"model_list": []},
        {"fallbacks": []},
        {"extra_body": {"api_base": "http://127.0.0.1:9/v1"}},
        {"extra_body": {"base_url": "http://127.0.0.1:9/v1"}},
        {"extra_body": {"user_config": {}}},
        {"extra_body": {"api_key": "client-key"}},
    ],
)
def test_chat_rejects_other_client_connection_overrides(monkeypatch, tmp_path, payload):
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    async def record_dispatch(*_args, **_kwargs):
        nonlocal dispatches
        dispatches += 1
        raise AssertionError("blocked request dispatched")

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", record_dispatch)
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={
                "model": "xai_oauth/grok-4.5",
                "messages": [{"role": "user", "content": "do not dispatch"}],
                **payload,
            },
        )

    assert response.status_code == 400
    assert dispatches == 0


def test_chat_without_override_reaches_operator_configured_upstream(monkeypatch, tmp_path):
    with _fake_openai_upstream() as (port, upstream_requests):
        module = _import_proxy_main(
            monkeypatch,
            tmp_path,
            f"http://{FAKE_XAI_AUTHORITY}:{port}/v1",
        )
        _block_catalog_fetches(monkeypatch)

        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
                json={
                    "model": "xai_oauth/grok-4.5",
                    "messages": [{"role": "user", "content": "control"}],
                    "stream": False,
                },
            )

    assert response.status_code == 200
    assert len(upstream_requests) == 1
    assert upstream_requests[0]["authorization"] == "Bearer fake-xai-key"
