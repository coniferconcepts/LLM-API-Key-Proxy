import importlib
import json
import os
import socket
import sys
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from filelock import FileLock
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FAKE_XAI_AUTHORITY = "[::1]"
FAKE_XAI_LOCK_PATH = Path(tempfile.gettempdir()) / f"mirrowel-xai-{os.getuid()}.lock"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _clear_provider_env(monkeypatch):
    for key in list(os.environ):
        if key != "PROXY_API_KEY" and (
            "_API_KEY" in key or key.endswith("_API_BASE") or key.endswith("_KEY")
        ):
            monkeypatch.delenv(key, raising=False)
    for key in (
        "FIREWORKS_API_V2_KEY",
        "OPENROUTER_FREE_KEY",
        "OPENROUTER_NON_ZDR_KEY",
        "DISABLED_PROVIDERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _import_proxy_main(monkeypatch, tmp_path, api_base: str, *, safe_mode: bool = True):
    _clear_provider_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROXY_API_KEY", "proxy-token")
    monkeypatch.setenv("SKIP_OAUTH_INIT_CHECK", "true")
    monkeypatch.setenv(
        "MIRROWEL_LOCAL_TRANSPORT_SAFE_MODE",
        "true" if safe_mode else "false",
    )
    monkeypatch.setenv("MIRROWEL_ALLOWED_HOSTS", "127.0.0.1,localhost,::1,testserver")
    monkeypatch.setenv(
        "MIRROWEL_CORS_ALLOWED_ORIGINS",
        "http://127.0.0.1,http://localhost,http://[::1],http://allowed.local",
    )
    monkeypatch.setenv("XAI_OAUTH_API_BASE", api_base)
    monkeypatch.setenv("XAI_OAUTH_API_KEY", "fake-xai-key")
    monkeypatch.setenv("GLOBAL_TIMEOUT", "5")
    monkeypatch.setenv("ACQUIRE_TIMEOUT", "2")
    monkeypatch.setenv("OAUTH_REFRESH_INTERVAL", "3600")

    sys.modules.pop("proxy_app.main", None)
    return importlib.import_module("proxy_app.main")


def _block_catalog_fetches(monkeypatch):
    model_info_service = importlib.import_module("rotator_library.model_info_service")
    model_info_service._registry_instance = None
    attempts = []

    def blocked_http_get(_adapter, url: str, timeout: int = 30):
        attempts.append(url)
        raise RuntimeError(f"external startup catalog fetch blocked: {url}")

    monkeypatch.setattr(model_info_service.DataSourceAdapter, "_http_get", blocked_http_get)
    return attempts


@contextmanager
def _fake_openai_upstream(*, redirect_url: str | None = None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            request_body = json.loads(body.decode("utf-8"))
            requests.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": request_body,
                }
            )
            if redirect_url is not None:
                self.send_response(307)
                self.send_header("Location", redirect_url)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if request_body.get("stream"):
                chunks = (
                    {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "grok-4.5",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "fake-ok"},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "grok-4.5",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    },
                )
                encoded = (
                    "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                    + "data: [DONE]\n\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            payload = {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 1,
                "model": "grok-4.5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "fake-ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format, *_args):
            return

    class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
        address_family = socket.AF_INET6

    with FileLock(FAKE_XAI_LOCK_PATH, timeout=120):
        server = IPv6ThreadingHTTPServer(("::1", 2465), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_port, requests
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_local_transport_safe_mode_skips_startup_catalog_fetches(monkeypatch, tmp_path):
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    catalog_attempts = _block_catalog_fetches(monkeypatch)

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        service = getattr(module.app.state, "model_info_service", None)
        if service is not None:
            client.portal.call(service.await_ready, 2.0)

        response = client.get("/health", headers={"Host": "127.0.0.1"})

        assert response.status_code == 200
        assert hasattr(module.app.state, "rotating_client")
        assert catalog_attempts == []


def test_local_transport_safe_mode_enforces_ipv4_ipv6_host_and_cors_matrix(
    monkeypatch,
    tmp_path,
):
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        host_statuses = {
            host: client.get("/health", headers={"Host": host}).status_code
            for host in (
                "127.0.0.1",
                "127.0.0.1:8000",
                "localhost",
                "[::1]",
                "[::1]:8000",
                "evil.example",
            )
        }
        cors_statuses = {
            origin: client.options(
                "/health",
                headers={
                    "Host": "[::1]:8000",
                    "Origin": origin,
                    "Access-Control-Request-Method": "GET",
                },
            )
            for origin in (
                "http://127.0.0.1",
                "http://localhost",
                "http://[::1]",
                "https://evil.example",
            )
        }

    assert host_statuses == {
        "127.0.0.1": 200,
        "127.0.0.1:8000": 200,
        "localhost": 200,
        "[::1]": 200,
        "[::1]:8000": 200,
        "evil.example": 400,
    }
    for origin in ("http://127.0.0.1", "http://localhost", "http://[::1]"):
        assert cors_statuses[origin].status_code == 200
        assert cors_statuses[origin].headers["access-control-allow-origin"] == origin
    assert cors_statuses["https://evil.example"].status_code == 400
    assert "access-control-allow-origin" not in cors_statuses["https://evil.example"].headers
