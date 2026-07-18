from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main
from rotator_library.client import RotatingClient
from rotator_library.credential_manager import CredentialManager


def test_explicit_empty_oauth_map_skips_credential_rediscovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: the caller deliberately supplies no OAuth credentials.
    discoveries = 0

    def record_discovery(_manager: CredentialManager) -> dict[str, list[str]]:
        nonlocal discoveries
        discoveries += 1
        return {"remote": ["/private/credential.json"]}

    monkeypatch.setattr(CredentialManager, "discover_and_prepare", record_discovery)

    # When: the client is created with an explicit empty mapping.
    client = RotatingClient(
        api_keys={},
        oauth_credentials={},
        configure_logging=False,
        data_dir=tmp_path,
    )

    # Then: empty remains empty without filesystem discovery or copying.
    assert client.oauth_credentials == {}
    assert client.all_credentials == {}
    assert discoveries == 0


def test_owner_only_provider_environment_wins_over_repo_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: the launcher has imported a validated provider source before startup.
    (tmp_path / ".env").write_text(
        (
            "PROXY_API_KEY=repo-shadow\n"
            "XAI_OAUTH_API_KEY=repo-shadow\n"
            "CHUTES_API_KEY=repo-only-credential\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENCODE_ROUTER_PROVIDER_ENV_PATH", "/private/provider-env")

    # When: Mirrowel loads compatibility dotenv files.
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")

    # Then: inherited provider values remain authoritative.
    assert module.proxy_api_key == "proxy-token"
    assert module.api_keys["xai_oauth"] == ["fake-xai-key"]
    assert "chutes" not in module.api_keys
    assert "CHUTES_API_KEY" not in module.os.environ


@pytest.mark.parametrize(
    ("path", "headers", "body", "dispatch_method"),
    [
        (
            "/v1/chat/completions",
            {"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            {
                "model": "xai_oauth/grok-4.5",
                "messages": [{"role": "user", "content": "stay local"}],
            },
            "acompletion",
        ),
        (
            "/v1/messages",
            {"x-api-key": "proxy-token", "Host": "127.0.0.1"},
            {
                "model": "xai_oauth/grok-4.5",
                "messages": [{"role": "user", "content": "stay local"}],
                "max_tokens": 16,
            },
            "anthropic_messages",
        ),
        (
            "/v1/embeddings",
            {"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            {"model": "xai_oauth/embed", "input": "stay local"},
            "aembedding",
        ),
    ],
)
def test_safe_mode_base_drift_fails_closed_before_inference_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
    headers: dict[str, str],
    body: dict[str, Any],
    dispatch_method: str,
) -> None:
    # Given: safe mode started with the canonical local xAI base.
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    async def record_dispatch(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal dispatches
        dispatches += 1
        raise AssertionError("request dispatched after safe-mode configuration drift")

    # When: the base changes after startup and an inference route is requested.
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, dispatch_method, record_dispatch)
        monkeypatch.setenv("XAI_OAUTH_API_BASE", "https://api.x.ai/v1")
        response = client.post(path, headers=headers, json=body)

    # Then: every inference surface fails closed before dispatch.
    assert response.status_code == 503
    if path == "/v1/messages":
        assert response.json() == {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "The upstream service is unavailable.",
            },
        }
    else:
        assert response.json()["detail"]["code"] == "local_transport_configuration_changed"
    assert dispatches == 0


def test_safe_mode_posture_drift_fails_closed_before_chat_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: safe mode was enabled when the application started.
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    async def record_dispatch(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal dispatches
        dispatches += 1
        raise AssertionError("chat dispatched after safe-mode posture drift")

    # When: the live environment attempts to disable safe mode.
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", record_dispatch)
        monkeypatch.setenv("MIRROWEL_LOCAL_TRANSPORT_SAFE_MODE", "false")
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={
                "model": "xai_oauth/grok-4.5",
                "messages": [{"role": "user", "content": "stay local"}],
            },
        )

    # Then: the startup posture remains authoritative and dispatch is blocked.
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "local_transport_configuration_changed"
    assert dispatches == 0


@pytest.mark.parametrize("method", ["get", "post"])
def test_safe_mode_drift_fails_closed_before_quota_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    method: str,
) -> None:
    # Given: safe mode was frozen at startup.
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    async def record_dispatch(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal dispatches
        dispatches += 1
        raise AssertionError("quota access continued after safe-mode drift")

    # When: the safe-mode posture drifts before a cached or refresh request.
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        rotating_client = module.app.state.rotating_client
        monkeypatch.setattr(rotating_client, "get_quota_stats", record_dispatch)
        monkeypatch.setattr(rotating_client, "force_refresh_quota", record_dispatch)
        monkeypatch.setenv("MIRROWEL_LOCAL_TRANSPORT_SAFE_MODE", "false")
        headers = {"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"}
        if method == "get":
            response = client.get("/v1/quota-stats", headers=headers)
        else:
            response = client.post(
                "/v1/quota-stats",
                headers=headers,
                json={"action": "force_refresh", "scope": "all"},
            )

    # Then: neither cached access nor live refresh reaches the client.
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "local_transport_configuration_changed"
    assert dispatches == 0
