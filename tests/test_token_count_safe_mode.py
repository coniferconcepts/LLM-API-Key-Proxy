from __future__ import annotations

import importlib
from pathlib import Path
from typing import Final, TypeAlias

import pytest
from starlette.testclient import TestClient

from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main

JSONValue: TypeAlias = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
AUTH_HEADERS: Final = {"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"}


@pytest.mark.parametrize(
    "messages",
    (
        [{"role": "user", "content": "count locally"}],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "count locally"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://public.example/unbounded.png"},
                    },
                ],
            }
        ],
    ),
)
def test_safe_mode_rejects_token_count_before_litellm_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    messages: list[dict[str, JSONValue]],
) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    def record_token_counter(**_kwargs: JSONValue) -> int:
        nonlocal dispatches
        dispatches += 1
        return 37

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        rotating_client_module = importlib.import_module("rotator_library.client")
        monkeypatch.setattr(rotating_client_module, "token_counter", record_token_counter)
        response = client.post(
            "/v1/token-count",
            headers=AUTH_HEADERS,
            json={"model": "xai_oauth/grok-4.5", "messages": messages},
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "local_transport_endpoint_disabled"
    assert dispatches == 0


def test_safe_mode_token_count_revalidates_configuration_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    def record_token_counter(**_kwargs: JSONValue) -> int:
        nonlocal dispatches
        dispatches += 1
        return 37

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        rotating_client_module = importlib.import_module("rotator_library.client")
        monkeypatch.setattr(rotating_client_module, "token_counter", record_token_counter)
        monkeypatch.setenv("XAI_OAUTH_API_BASE", "https://api.x.ai/v1")
        response = client.post(
            "/v1/token-count",
            headers=AUTH_HEADERS,
            json={
                "model": "xai_oauth/grok-4.5",
                "messages": [{"role": "user", "content": "do not dispatch"}],
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "local_transport_configuration_changed"
    assert dispatches == 0


def test_normal_mode_preserves_local_token_count_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_proxy_main(
        monkeypatch,
        tmp_path,
        "http://198.51.100.7:2465/v1",
        safe_mode=False,
    )
    _block_catalog_fetches(monkeypatch)
    recorded: list[dict[str, JSONValue]] = []

    def record_token_counter(**kwargs: JSONValue) -> int:
        recorded.append(kwargs)
        return 37

    body = {
        "model": "remote_provider/model",
        "messages": [{"role": "user", "content": "normal mode"}],
    }

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        rotating_client_module = importlib.import_module("rotator_library.client")
        monkeypatch.setattr(rotating_client_module, "token_counter", record_token_counter)
        response = client.post("/v1/token-count", headers=AUTH_HEADERS, json=body)

    assert response.status_code == 200
    assert response.json() == {"token_count": 37}
    assert recorded == [body]
