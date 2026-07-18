from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from test_local_transport_safe_mode import (
    FAKE_XAI_AUTHORITY,
    _block_catalog_fetches,
    _fake_openai_upstream,
    _import_proxy_main,
)


def test_credential_bearing_safe_mode_has_zero_startup_egress_before_xai_post(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    startup_attempts: list[str] = []

    class CredentialManager:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def discover_and_prepare(self) -> dict[str, list[str]]:
            return {"fake_oauth": ["env://credential-bearing"]}

    class Provider:
        async def initialize_token(self, _path: str) -> None:
            startup_attempts.append("oauth.initialize_token")

        async def get_user_info(self, _path: str) -> dict[str, str]:
            startup_attempts.append("oauth.get_user_info")
            return {"email": "fixture@example.invalid"}

        async def initialize_credentials(self, _credentials: list[str]) -> None:
            startup_attempts.append("background.initialize_credentials")

        def get_background_job_config(self) -> dict[str, int | bool | str]:
            startup_attempts.append("background.get_job_config")
            return {"name": "fixture", "interval": 3600, "run_on_start": True}

        async def run_background_job(self, _usage_manager, _credentials: list[str]) -> None:
            startup_attempts.append("background.run_provider_job")

        async def proactively_refresh(self, _path: str) -> None:
            startup_attempts.append("background.proactively_refresh")

    with _fake_openai_upstream() as (port, upstream_requests):
        module = _import_proxy_main(
            monkeypatch,
            tmp_path,
            f"http://{FAKE_XAI_AUTHORITY}:{port}/v1",
        )
        catalog_attempts = _block_catalog_fetches(monkeypatch)
        monkeypatch.delenv("SKIP_OAUTH_INIT_CHECK", raising=False)
        monkeypatch.setattr(module, "CredentialManager", CredentialManager)
        monkeypatch.setitem(module.PROVIDER_PLUGINS, "fake_oauth", Provider)

        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            client.portal.call(module.asyncio.sleep, 0.05)
            attempts_before_request = list(startup_attempts)
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
                json={
                    "model": "xai_oauth/grok-4.5",
                    "messages": [{"role": "user", "content": "one intended request"}],
                    "stream": False,
                },
            )

    assert attempts_before_request == []
    assert catalog_attempts == []
    assert response.status_code == 200
    assert len(upstream_requests) == 1
    assert upstream_requests[0]["path"] == "/v1/chat/completions"
    assert upstream_requests[0]["body"]["model"] == "grok-4.5"


def test_normal_mode_keeps_background_refresher_and_catalog_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    module = _import_proxy_main(
        monkeypatch,
        tmp_path,
        "http://198.51.100.7:2465/v1",
        safe_mode=False,
    )
    refresher_class = module.RotatingClient.__init__.__globals__["BackgroundRefresher"]

    class ModelInfoService:
        async def stop(self) -> None:
            events.append("catalog.stop")

    def start(_refresher) -> None:
        events.append("refresher.start")

    async def stop(_refresher) -> None:
        events.append("refresher.stop")

    async def init_model_info_service() -> ModelInfoService:
        events.append("catalog.start")
        return ModelInfoService()

    monkeypatch.setenv("MIRROWEL_LOCAL_TRANSPORT_SAFE_MODE", "false")
    monkeypatch.setattr(refresher_class, "start", start)
    monkeypatch.setattr(refresher_class, "stop", stop)
    monkeypatch.setattr(module, "init_model_info_service", init_model_info_service)

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        response = client.get("/health", headers={"Host": "127.0.0.1"})

    assert response.status_code == 200
    assert events == ["refresher.start", "catalog.start", "refresher.stop", "catalog.stop"]


def test_safe_mode_model_discovery_queries_only_local_xai_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    provider_queries: list[str] = []

    async def get_local_models(provider: str) -> list[str]:
        provider_queries.append(provider)
        return ["xai_oauth/grok-4.5"]

    async def reject_all_provider_discovery(*_args, **_kwargs):
        raise AssertionError("safe mode attempted all-provider model discovery")

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        rotating_client = module.app.state.rotating_client
        monkeypatch.setattr(rotating_client, "get_available_models", get_local_models)
        monkeypatch.setattr(
            rotating_client,
            "get_all_available_models",
            reject_all_provider_discovery,
        )
        response = client.get(
            "/v1/models?enriched=false",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
        )

    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == ["xai_oauth/grok-4.5"]
    assert provider_queries == ["xai_oauth"]


def test_safe_mode_force_quota_refresh_is_rejected_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    refresh_attempts = 0

    async def record_refresh(*_args, **_kwargs):
        nonlocal refresh_attempts
        refresh_attempts += 1
        raise AssertionError("safe mode attempted a live quota refresh")

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(
            module.app.state.rotating_client,
            "force_refresh_quota",
            record_refresh,
        )
        response = client.post(
            "/v1/quota-stats",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={"action": "force_refresh", "scope": "all"},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "type": "conflict_error",
        "code": "local_transport_live_refresh_disabled",
        "status": 409,
        "message": "The request conflicts with the current proxy mode.",
    }
    assert refresh_attempts == 0
