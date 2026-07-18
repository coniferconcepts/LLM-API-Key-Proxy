from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main


@pytest.mark.parametrize(
    ("path", "headers", "body", "dispatch_method"),
    [
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
def test_safe_mode_rejects_non_chat_inference_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
    headers: dict[str, str],
    body: dict[str, Any],
    dispatch_method: str,
) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    async def record_dispatch(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal dispatches
        dispatches += 1
        raise AssertionError("safe mode dispatched outside local xAI chat")

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, dispatch_method, record_dispatch)
        response = client.post(path, headers=headers, json=body)

    assert response.status_code == 409
    if path == "/v1/messages":
        assert response.json() == {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "The request conflicts with the current proxy mode.",
            },
        }
    else:
        assert response.json()["detail"]["code"] == "local_transport_endpoint_disabled"
    assert dispatches == 0


def test_safe_mode_filters_startup_credentials_to_xai_oauth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    module.api_keys = {
        "xai_oauth": ["local-api-key"],
        "remote_provider": ["must-not-enter-client"],
    }

    class CredentialManager:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def discover_and_prepare(self) -> dict[str, list[str]]:
            return {
                "xai_oauth": ["env://local-oauth"],
                "remote_oauth": ["env://must-not-enter-client"],
            }

    monkeypatch.setattr(module, "CredentialManager", CredentialManager)

    with TestClient(module.app, base_url="http://127.0.0.1"):
        client = module.app.state.rotating_client
        assert client.api_keys == {"xai_oauth": ["local-api-key"]}
        assert client.oauth_credentials == {"xai_oauth": ["env://local-oauth"]}


@pytest.mark.parametrize(
    ("path", "headers", "body", "dispatch_method"),
    [
        (
            "/v1/messages",
            {"x-api-key": "proxy-token", "Host": "127.0.0.1"},
            {
                "model": "opencode_go_messages/model",
                "messages": [{"role": "user", "content": "normal mode"}],
                "max_tokens": 16,
            },
            "anthropic_messages",
        ),
        (
            "/v1/embeddings",
            {"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            {"model": "remote_provider/embed", "input": "normal mode"},
            "aembedding",
        ),
    ],
)
def test_normal_mode_preserves_non_chat_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
    headers: dict[str, str],
    body: dict[str, Any],
    dispatch_method: str,
) -> None:
    module = _import_proxy_main(
        monkeypatch,
        tmp_path,
        "http://198.51.100.7:2465/v1",
        safe_mode=False,
    )
    dispatches = 0

    async def record_dispatch(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal dispatches
        dispatches += 1
        raise RuntimeError("synthetic normal-mode dispatch")

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, dispatch_method, record_dispatch)
        response = client.post(path, headers=headers, json=body)

    assert response.status_code == 500
    assert dispatches == 1
