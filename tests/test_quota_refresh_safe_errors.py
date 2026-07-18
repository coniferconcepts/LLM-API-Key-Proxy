from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any

import pytest
from starlette.testclient import TestClient

from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SECRET_MARKER = "SECRET-quota-provider-payload-" + ("x" * 300)
SECRET_PATH = f"/private/oauth/{SECRET_MARKER}.json"


def _bare_client(provider_class):
    client_module = importlib.import_module("rotator_library.client")
    client = object.__new__(client_module.RotatingClient)
    client.all_credentials = {"synthetic": [SECRET_PATH]}
    client._provider_plugins = {"synthetic": provider_class}
    client._provider_instances = {}
    client.usage_manager = object()
    return client


class ProviderResponseFailure:
    async def fetch_initial_baselines(self, credentials: list[str]) -> dict[str, Any]:
        return {
            credentials[0]: {
                "status": "error",
                "error": SECRET_MARKER,
            }
        }


class ProviderException:
    async def fetch_initial_baselines(self, _credentials: list[str]) -> dict[str, Any]:
        raise RuntimeError(SECRET_MARKER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_class", "category"),
    (
        (ProviderResponseFailure, "provider_response_error"),
        (ProviderException, "provider_exception"),
    ),
)
async def test_force_refresh_returns_only_aggregate_failure_categories(
    provider_class,
    category: str,
    monkeypatch,
) -> None:
    client = _bare_client(provider_class)
    log_calls: list[tuple[Any, ...]] = []
    client_module = importlib.import_module("rotator_library.client")
    monkeypatch.setattr(client_module.lib_logger, "error", lambda *args: log_calls.append(args))

    result = await client.force_refresh_quota(provider="synthetic")

    assert result["failed_count"] == 1
    assert result["failure_categories"] == {category: 1}
    assert "errors" not in result
    assert "provider" not in result
    assert "credential" not in result
    assert SECRET_MARKER not in repr(result)
    assert SECRET_PATH not in repr(result)
    assert SECRET_MARKER not in repr(log_calls)


@pytest.mark.parametrize("provider_class", (ProviderResponseFailure, ProviderException))
def test_quota_route_keeps_failed_refresh_response_and_logs_marker_free(
    provider_class,
    monkeypatch,
    tmp_path,
) -> None:
    module = _import_proxy_main(
        monkeypatch,
        tmp_path,
        "http://127.0.0.1:2465/v1",
        safe_mode=False,
    )
    _block_catalog_fetches(monkeypatch)
    log_calls: list[tuple[Any, ...]] = []
    client_module = importlib.import_module("rotator_library.client")
    monkeypatch.setattr(client_module.lib_logger, "error", lambda *args: log_calls.append(args))

    with TestClient(module.app, base_url="http://127.0.0.1") as test_client:
        rotating_client = module.app.state.rotating_client
        rotating_client.all_credentials["synthetic"] = [SECRET_PATH]
        rotating_client._provider_plugins["synthetic"] = provider_class
        rotating_client._provider_instances.pop("synthetic", None)

        async def safe_stats(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"providers": {}}

        monkeypatch.setattr(rotating_client, "get_quota_stats", safe_stats)
        response = test_client.post(
            "/v1/quota-stats",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={"action": "force_refresh", "scope": "provider", "provider": "synthetic"},
        )

    assert response.status_code == 200
    refresh_result = response.json()["refresh_result"]
    assert refresh_result["success"] is False
    assert refresh_result["failed_count"] == 1
    assert set(refresh_result["failure_categories"]) in (
        {"provider_response_error"},
        {"provider_exception"},
    )
    assert "provider" not in refresh_result
    assert "credential" not in refresh_result
    assert SECRET_MARKER not in response.text
    assert SECRET_PATH not in response.text
    assert SECRET_MARKER not in repr(log_calls)
