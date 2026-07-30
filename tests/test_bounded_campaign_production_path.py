from __future__ import annotations

from pathlib import Path
import sys

import pytest
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bounded_campaign_production_support import (  # noqa: E402
    ProviderScenario,
    fake_provider,
    install_bounded_state,
    payload,
    proxy_module,
    reserve_synthetic_campaign_headroom,
)


@pytest.mark.parametrize(
    ("stream", "tool"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_production_adapter_emits_one_physical_post_for_admitted_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stream: bool,
    tool: bool,
) -> None:
    request = payload(stream=stream, tool=tool)
    _ledger, headers = install_bounded_state(monkeypatch, tmp_path, request)

    with fake_provider(ProviderScenario()) as provider:
        module = proxy_module(monkeypatch, tmp_path, provider)
        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            response = client.post("/v1/chat/completions", headers=headers, json=request)

    assert response.status_code == 200
    assert len(provider.posts) == 1
    assert provider.posts[0].path == "/v1/chat/completions"
    assert provider.posts[0].body.get("stream", False) is stream
    assert ("tools" in provider.posts[0].body) is tool


@pytest.mark.parametrize(
    "scenario",
    [
        ProviderScenario(status=429),
        ProviderScenario(redirect=True),
        ProviderScenario(disconnect=True),
    ],
    ids=["retryable-and-rotation", "redirect", "reconnect"],
)
def test_production_adapter_suppresses_second_post_after_egress_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scenario: ProviderScenario,
) -> None:
    request = payload()
    _ledger, headers = install_bounded_state(monkeypatch, tmp_path, request)

    with fake_provider(scenario) as provider:
        module = proxy_module(monkeypatch, tmp_path, provider)
        with TestClient(
            module.app,
            base_url="http://127.0.0.1",
            raise_server_exceptions=False,
        ) as client:
            assert len(module.app.state.rotating_client.all_credentials["xai_oauth"]) == 2
            response = client.post("/v1/chat/completions", headers=headers, json=request)

    assert response.status_code != 200
    assert len(provider.posts) == 1


def test_duplicate_capability_rejects_before_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = payload()
    _ledger, headers = install_bounded_state(monkeypatch, tmp_path, request)

    with fake_provider(ProviderScenario()) as provider:
        module = proxy_module(monkeypatch, tmp_path, provider)
        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            first = client.post("/v1/chat/completions", headers=headers, json=request)
            duplicate = client.post("/v1/chat/completions", headers=headers, json=request)

    assert first.status_code == 200
    assert duplicate.status_code != 200
    assert len(provider.posts) == 1


def test_restart_recovers_durable_reservation_before_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = payload()
    _ledger, headers = install_bounded_state(monkeypatch, tmp_path, request)

    with fake_provider(ProviderScenario()) as provider:
        module = proxy_module(monkeypatch, tmp_path, provider)
        with TestClient(module.app, base_url="http://127.0.0.1") as client:
            first = client.post("/v1/chat/completions", headers=headers, json=request)
        restarted_module = proxy_module(monkeypatch, tmp_path, provider)
        with TestClient(
            restarted_module.app,
            base_url="http://127.0.0.1",
            raise_server_exceptions=False,
        ) as restarted:
            recovered = restarted.post("/v1/chat/completions", headers=headers, json=request)

    assert first.status_code == 200
    assert recovered.status_code != 200
    assert len(provider.posts) == 1


@pytest.mark.parametrize(
    "ledger_state",
    ["corrupt", "exhausted"],
    ids=["corrupt-ledger", "synthetic-ledger-headroom-79"],
)
def test_invalid_durable_ledger_fails_closed_before_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ledger_state: str,
) -> None:
    request = payload()
    ledger_path, headers = install_bounded_state(monkeypatch, tmp_path, request)
    if ledger_state == "corrupt":
        ledger_path.write_text("not-json\n", encoding="utf-8")
        ledger_path.chmod(0o600)
    else:
        reserve_synthetic_campaign_headroom(ledger_path, 79)

    with fake_provider(ProviderScenario()) as provider:
        module = proxy_module(monkeypatch, tmp_path, provider)
        with TestClient(
            module.app,
            base_url="http://127.0.0.1",
            raise_server_exceptions=False,
        ) as client:
            response = client.post("/v1/chat/completions", headers=headers, json=request)

    assert response.status_code != 200
    assert provider.posts == []
