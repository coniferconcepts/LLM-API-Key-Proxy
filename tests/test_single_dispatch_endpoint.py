from __future__ import annotations

import sys
from pathlib import Path

import litellm
import pytest
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bounded_campaign_production_support import ProviderScenario, fake_provider
from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main


@pytest.fixture(autouse=True)
def isolate_litellm_session(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "aclient_session", None)


def test_chat_endpoint_single_dispatch_auth_and_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token = "synthetic-router-only-dispatch-token-32"
    with fake_provider(ProviderScenario(status=429)) as provider:
        catalog_attempts = _block_catalog_fetches(monkeypatch)
        from rotator_library import model_info_service

        async def no_background_catalog_load(_registry: object) -> None:
            return None

        monkeypatch.setattr(
            model_info_service.ModelRegistry, "_load_all_sources", no_background_catalog_load
        )
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
