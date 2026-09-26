from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import litellm
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bounded_campaign_production_support import ProviderScenario, fake_provider
from credential_admission_contract_support import make_client
from test_local_transport_safe_mode import _block_catalog_fetches

from rotator_library.error_handler import NoAvailableKeysError, PreRequestCallbackError
from rotator_library.provider_config import ProviderConfig
from rotator_library.single_dispatch import (
    SingleDispatchGuard,
)


@pytest.fixture(autouse=True)
def isolate_litellm_session(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "aclient_session", None)


@pytest.mark.parametrize("failure", ["429", "500", "disconnect", "first_byte_timeout"])
@pytest.mark.asyncio
async def test_single_dispatch_physical_post_budget_across_litellm_and_rotation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    posts: list[str] = []
    release_response = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            posts.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            if failure == "disconnect":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            if failure == "first_byte_timeout":
                release_response.wait(timeout=2)
                return
            body = json.dumps(
                {"error": {"message": "synthetic rejection", "type": "rate_limit_error"}}
            ).encode()
            self.send_response(int(failure))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("CHUTES_API_BASE", f"http://127.0.0.1:{server.server_port}/v1")
        catalog_attempts = _block_catalog_fetches(monkeypatch)
        client, _manager = make_client(tmp_path, acquire_timeout=0.2)

        async def record_failure(*_args: object, **_kwargs: object) -> None:
            return None

        _manager.record_failure = record_failure
        client.all_credentials = {"chutes": ["synthetic-first", "synthetic-rotated"]}
        client.max_concurrent_requests_per_key = {"chutes": 1}
        client.max_retries = 3
        client.provider_config = ProviderConfig()
        client.global_timeout = 1.5
        client.litellm_provider_params = {"chutes": {}}
        guard = SingleDispatchGuard()
        forwarded: list[tuple[int, int]] = []
        real_completion = litellm.acompletion

        async def observed_completion(**kwargs: object) -> object:
            forwarded.append((kwargs["num_retries"], kwargs["max_retries"]))
            return await real_completion(**kwargs)

        monkeypatch.setattr(litellm, "acompletion", observed_completion)
        try:
            await asyncio.wait_for(
                client.acompletion(
                    pre_request_callback=guard,
                    model="chutes/moonshotai/Kimi-K3-TEE",
                    messages=[{"role": "user", "content": "synthetic"}],
                    stream=False,
                    request_timeout=0.25,
                ),
                timeout=3,
            )
        except (PreRequestCallbackError, NoAvailableKeysError):
            pass
        assert forwarded == [(0, 0)]
        assert posts == ["/v1/chat/completions"]
        assert catalog_attempts == []
    finally:
        release_response.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_disconnected_stream_closes_fake_provider_socket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider_closed = threading.Event()
    posts: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            posts.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": "chatcmpl-synthetic",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "moonshotai/Kimi-K3-TEE",
                "choices": [{"index": 0, "delta": {"content": "first"}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            self.connection.settimeout(3)
            try:
                if self.rfile.read(1) == b"":
                    provider_closed.set()
            except (ConnectionResetError, OSError):
                provider_closed.set()

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("CHUTES_API_BASE", f"http://127.0.0.1:{server.server_port}/v1")
        _block_catalog_fetches(monkeypatch)
        client, _manager = make_client(tmp_path, acquire_timeout=0.2)
        client.all_credentials = {"chutes": ["synthetic-only"]}
        client.max_concurrent_requests_per_key = {"chutes": 1}
        client.provider_config = ProviderConfig()
        client.litellm_provider_params = {"chutes": {}}
        stream = client.acompletion(
            model="chutes/moonshotai/Kimi-K3-TEE",
            messages=[{"role": "user", "content": "synthetic"}],
            stream=True,
            request_timeout=2,
        )
        try:
            assert "first" in await asyncio.wait_for(anext(stream), timeout=3)
        finally:
            await stream.aclose()
        assert await asyncio.to_thread(provider_closed.wait, 2)
        assert posts == ["/v1/chat/completions"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_unflagged_chutes_request_preserves_normal_retry_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with fake_provider(ProviderScenario()) as provider:
        monkeypatch.setenv("CHUTES_API_BASE", provider.api_base)
        catalog_attempts = _block_catalog_fetches(monkeypatch)
        client, _manager = make_client(tmp_path, acquire_timeout=0.2)
        client.all_credentials = {"chutes": ["synthetic-first", "synthetic-rotated"]}
        client.max_concurrent_requests_per_key = {"chutes": 1}
        client.max_retries = 3
        client.provider_config = ProviderConfig()
        client.global_timeout = 1.5
        client.litellm_provider_params = {"chutes": {}}
        forwarded: list[dict[str, object]] = []
        real_completion = litellm.acompletion

        async def observed_completion(**kwargs: object) -> object:
            forwarded.append(kwargs)
            return await real_completion(**kwargs)

        monkeypatch.setattr(litellm, "acompletion", observed_completion)
        await asyncio.wait_for(
            client.acompletion(
                model="chutes/moonshotai/Kimi-K3-TEE",
                messages=[{"role": "user", "content": "synthetic"}],
                stream=False,
                request_timeout=0.25,
            ),
            timeout=2,
        )
        assert len(forwarded) == 1
        assert forwarded[0].get("num_retries") != 0
        assert forwarded[0].get("max_retries") != 0
        assert [post.path for post in provider.posts] == ["/v1/chat/completions"]
        assert catalog_attempts == []
