from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest
from starlette.datastructures import Headers
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bounded_campaign_production_support import ProviderScenario, fake_provider
from credential_admission_contract_support import make_client
from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main

from rotator_library.error_handler import NoAvailableKeysError, PreRequestCallbackError
from rotator_library.provider_config import ProviderConfig
from rotator_library.single_dispatch import (
    SingleDispatchGuard,
    SingleDispatchRejected,
    single_dispatch_requested,
)


def test_single_dispatch_rejection_reason_omits_token(monkeypatch: pytest.MonkeyPatch) -> None:
    canary = "canary-token-value-0123456789abcdef"
    monkeypatch.setenv("MIRROWEL_SINGLE_DISPATCH_TOKEN", "expected-token-value-0123456789abcd")
    headers = Headers(
        {
            "x-mirrowel-single-dispatch": "1",
            "x-mirrowel-single-dispatch-token": canary,
            "x-opencode-alias": "kimi-k3-advisor",
        }
    )
    request = SimpleNamespace(headers=headers, client=SimpleNamespace(host="127.0.0.1"))
    with pytest.raises(SingleDispatchRejected) as raised:
        single_dispatch_requested(request, "chutes/moonshotai/Kimi-K3-TEE", authenticated=True)
    assert raised.value.reason_class == "token_missing_or_invalid"
    assert canary not in str(raised.value)


def test_single_dispatch_requires_loopback_proxy_auth_alias_and_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "synthetic-token-of-at-least-32-characters"
    monkeypatch.setenv("MIRROWEL_SINGLE_DISPATCH_TOKEN", token)
    headers = Headers(
        {
            "x-mirrowel-single-dispatch": "1",
            "x-mirrowel-single-dispatch-token": token,
            "x-opencode-alias": "kimi-k3-advisor",
        }
    )
    request = SimpleNamespace(headers=headers, client=SimpleNamespace(host="127.0.0.1"))
    model = "chutes/moonshotai/Kimi-K3-TEE"
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=False)
    assert single_dispatch_requested(request, model, authenticated=True)
    request.client.host = "192.0.2.1"
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    request.client.host = "127.0.0.1"
    request.headers = Headers(
        {
            "x-mirrowel-single-dispatch": "1",
            "x-mirrowel-single-dispatch-token": token,
            "x-opencode-alias": "kimi-k3",
        }
    )
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    request.headers = Headers(
        {
            "x-mirrowel-single-dispatch": "1",
            "x-mirrowel-single-dispatch-token": "forged",
            "x-opencode-alias": "kimi-k3-advisor",
        }
    )
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    request.headers = Headers(
        raw=[
            (b"x-mirrowel-single-dispatch", b"1"),
            (b"x-mirrowel-single-dispatch-token", b"\xe9" * 40),
            (b"x-opencode-alias", b"kimi-k3-advisor"),
        ]
    )
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    request.headers = headers
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, "chutes/moonshotai/Kimi-K3", authenticated=True)
    request.headers = Headers(
        raw=[
            (b"x-mirrowel-single-dispatch", b"1"),
            (b"x-mirrowel-single-dispatch", b"1"),
            (b"x-mirrowel-single-dispatch-token", token.encode()),
            (b"x-opencode-alias", b"kimi-k3-advisor"),
        ]
    )
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    request.headers = headers
    monkeypatch.delenv("MIRROWEL_SINGLE_DISPATCH_TOKEN")
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    monkeypatch.setenv("MIRROWEL_SINGLE_DISPATCH_TOKEN", token)
    request.client.host = "127.0.0.1"
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, "chutes/other-model", authenticated=True)
    request.headers = Headers({"x-mirrowel-single-dispatch": "yes"})
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    request.headers = Headers({})
    assert not single_dispatch_requested(request, model, authenticated=False)


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


def test_chat_endpoint_single_dispatch_auth_and_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token = "synthetic-router-only-dispatch-token-32"
    with fake_provider(ProviderScenario(status=429)) as provider:
        catalog_attempts = _block_catalog_fetches(monkeypatch)
        from rotator_library import model_info_service

        async def no_background_catalog_load(_registry: object) -> None:
            return None

        monkeypatch.setattr(model_info_service.ModelRegistry, "_load_all_sources", no_background_catalog_load)
        module = _import_proxy_main(monkeypatch, tmp_path, provider.api_base, safe_mode=False)
        monkeypatch.setenv("CHUTES_API_BASE", provider.api_base)
        monkeypatch.setenv("MIRROWEL_SINGLE_DISPATCH_TOKEN", token)
        module.api_keys.clear()
        module.api_keys["chutes"] = ["synthetic-first", "synthetic-second"]
        import rotator_library.providers.chutes_provider as chutes_module

        monkeypatch.setattr(
            chutes_module.ChutesProvider, "get_background_job_config", lambda _self: None
        )
        headers = {
            "Authorization": "Bearer proxy-token",
            "X-Mirrowel-Single-Dispatch": "1",
            "X-Mirrowel-Single-Dispatch-Token": token,
            "X-OpenCode-Alias": "kimi-k3-advisor",
        }
        body = {
            "model": "chutes/moonshotai/Kimi-K3-TEE",
            "messages": [{"role": "user", "content": "synthetic"}],
            "stream": False,
            "max_completion_tokens": 2048,
            "extra_body": {"reasoning_effort": "low"},
        }

        async def loopback_app(scope: dict, receive: object, send: object) -> None:
            if scope["type"] == "http":
                scope = {**scope, "client": ("127.0.0.1", 51337)}
            await module.app(scope, receive, send)

        with TestClient(
            loopback_app,
            base_url="http://127.0.0.1",
            raise_server_exceptions=False,
        ) as http:
            client = module.app.state.rotating_client
            assert len(client.all_credentials["chutes"]) == 2
            assert client.abort_on_callback_error
            client.max_retries = 2
            client.global_timeout = 3
            unauth = http.post(
                "/v1/chat/completions",
                headers={key: value for key, value in headers.items() if key != "Authorization"},
                json=body,
            )
            assert unauth.status_code == 401
            assert len(provider.posts) == 0
            forged = http.post(
                "/v1/chat/completions",
                headers={**headers, "X-Mirrowel-Single-Dispatch-Token": "forged"},
                json=body,
            )
            assert forged.status_code == 403
            alias_mismatch = http.post(
                "/v1/chat/completions",
                headers={**headers, "X-OpenCode-Alias": "kimi-k3"},
                json=body,
            )
            assert alias_mismatch.status_code == 403
            model_mismatch = http.post(
                "/v1/chat/completions",
                headers=headers,
                json={**body, "model": "chutes/moonshotai/Kimi-K3"},
            )
            assert model_mismatch.status_code == 403
            assert len(provider.posts) == 0
            trusted = http.post("/v1/chat/completions", headers=headers, json=body)
            assert trusted.status_code not in {401, 403}
            assert len(provider.posts) == 1, (trusted.status_code, trusted.json().get("detail"))
            assert provider.posts[0].path == "/v1/chat/completions"
            wire = provider.posts[0].body
            assert wire["max_tokens"] == 2048
            assert wire["reasoning_effort"] == "low"
            assert wire.get("stream", False) is False
            assert "tools" not in wire
            assert catalog_attempts == []


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
