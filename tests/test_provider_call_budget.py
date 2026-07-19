from __future__ import annotations

from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main

UPSTREAM_SENTINEL = "UPSTREAM-PAYLOAD-SENTINEL"


@pytest.mark.parametrize(
    ("upstream_status", "public_status", "public_code"),
    [
        pytest.param(401, 401, "authentication_failed", id="authentication-401"),
        pytest.param(403, 401, "authentication_failed", id="authorization-403"),
        pytest.param(429, 429, "rate_limited", id="rate-limit-429"),
        pytest.param(408, 504, "gateway_timeout", id="timeout-408"),
        pytest.param(500, 503, "service_unavailable", id="server-500"),
    ],
)
def test_raw_httpx_budget_failure_is_classified_and_sanitized_at_public_boundary(
    monkeypatch,
    tmp_path,
    upstream_status: int,
    public_status: int,
    public_code: str,
) -> None:
    # Given: the first admitted provider call failed with a raw httpx status error.
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    request = httpx.Request("POST", "https://upstream.invalid/v1/chat/completions")
    response = httpx.Response(upstream_status, request=request)
    first_failure = httpx.HTTPStatusError(
        UPSTREAM_SENTINEL,
        request=request,
        response=response,
    )

    async def deny_after_first_failure(**_kwargs: Any) -> Any:
        raise module.ProviderCallBudgetExhausted(first_failure)

    # When: the typed terminal denial crosses the public HTTP boundary.
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(
            module.app.state.rotating_client, "acompletion", deny_after_first_failure
        )
        public_response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer proxy-token",
                "Host": "127.0.0.1",
                "X-OpenCode-Max-Provider-Calls": "1",
            },
            json={
                "model": "xai_oauth/grok-4.5",
                "messages": [{"role": "user", "content": "one dispatch"}],
                "stream": False,
            },
        )

    # Then: response metadata reflects the upstream status class without exposing diagnostics.
    assert public_response.status_code == public_status
    assert public_response.json()["detail"]["code"] == public_code
    assert UPSTREAM_SENTINEL not in public_response.text
    assert "upstream.invalid" not in public_response.text


def test_authenticated_call_ceiling_blocks_same_credential_retry(monkeypatch, tmp_path) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    async def simulate_inner_retry(
        *, request: Any, pre_request_callback: Any = None, **kwargs: Any
    ) -> Any:
        nonlocal dispatches
        budget = kwargs.pop("_provider_call_budget")
        first_failure: Exception | None = None
        for _attempt in range(2):
            if pre_request_callback is not None:
                await pre_request_callback(request, kwargs)
            budget.admit(first_failure)
            dispatches += 1
            first_failure = RuntimeError("synthetic retry failure")
        raise RuntimeError("synthetic retry failure")

    with TestClient(
        module.app,
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", simulate_inner_retry)
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer proxy-token",
                "Host": "127.0.0.1",
                "X-OpenCode-Max-Provider-Calls": "1",
            },
            json={
                "model": "xai_oauth/grok-4.5",
                "messages": [{"role": "user", "content": "one dispatch"}],
                "stream": False,
            },
        )

    assert response.status_code == 500
    assert dispatches == 1


def test_authenticated_call_ceiling_follows_credential_rotation(monkeypatch, tmp_path) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches: list[str] = []

    async def simulate_rotation(
        *, request: Any, pre_request_callback: Any = None, **kwargs: Any
    ) -> Any:
        budget = kwargs.pop("_provider_call_budget")
        first_failure: Exception | None = None
        for credential in ("credential-a", "credential-b"):
            if pre_request_callback is not None:
                await pre_request_callback(request, kwargs)
            budget.admit(first_failure)
            dispatches.append(credential)
            first_failure = RuntimeError("synthetic credential failure")
        raise RuntimeError("synthetic rotation failure")

    with TestClient(
        module.app,
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", simulate_rotation)
        client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer proxy-token",
                "Host": "127.0.0.1",
                "X-OpenCode-Max-Provider-Calls": "1",
            },
            json={
                "model": "xai_oauth/grok-4.5",
                "messages": [{"role": "user", "content": "one credential"}],
                "stream": False,
            },
        )

    assert dispatches == ["credential-a"]


@pytest.mark.parametrize("value", ["", "0", "01", "+1", " 1", "1.0", "1001", "abc"])
def test_authenticated_invalid_call_ceiling_is_rejected(monkeypatch, tmp_path, value) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer proxy-token",
                "Host": "127.0.0.1",
                "X-OpenCode-Max-Provider-Calls": value,
            },
            json={"model": "xai_oauth/grok-4.5", "messages": []},
        )

    assert response.status_code == 400


def test_auth_rejection_precedes_invalid_call_ceiling(monkeypatch, tmp_path) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer wrong-token",
                "Host": "127.0.0.1",
                "X-OpenCode-Max-Provider-Calls": "invalid",
            },
            json={"model": "xai_oauth/grok-4.5", "messages": []},
        )

    assert response.status_code in {401, 403}


def test_duplicate_call_ceiling_is_rejected(monkeypatch, tmp_path) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/v1/chat/completions",
            headers=[
                ("Authorization", "Bearer proxy-token"),
                ("Host", "127.0.0.1"),
                ("X-OpenCode-Max-Provider-Calls", "1"),
                ("x-opencode-max-provider-calls", "1"),
            ],
            json={"model": "xai_oauth/grok-4.5", "messages": []},
        )

    assert response.status_code == 400


def test_streaming_call_ceiling_blocks_second_dispatch(monkeypatch, tmp_path) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    async def simulate_stream(*, request: Any, **kwargs: Any) -> Any:
        nonlocal dispatches
        budget = kwargs.pop("_provider_call_budget")
        first_failure: Exception | None = None
        for _attempt in range(2):
            budget.admit(first_failure)
            dispatches += 1
            first_failure = RuntimeError("synthetic pre-byte failure")
            if False:
                yield "unused"

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", simulate_stream)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer proxy-token",
                "Host": "127.0.0.1",
                "X-OpenCode-Max-Provider-Calls": "1",
            },
            json={
                "model": "xai_oauth/grok-4.5",
                "messages": [],
                "stream": True,
            },
        ) as response:
            list(response.iter_text())

    assert dispatches == 1
