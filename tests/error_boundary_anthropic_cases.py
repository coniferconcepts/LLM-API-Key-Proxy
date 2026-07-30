from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

from error_boundary_regression_support import SECRET_MESSAGE, anthropic_body
from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main


@pytest.mark.parametrize(
    ("exception_name", "status", "anthropic_type"),
    (
        ("InvalidRequestError", 400, "invalid_request_error"),
        ("AuthenticationError", 401, "authentication_error"),
        ("RateLimitError", 429, "rate_limit_error"),
        ("ServiceUnavailableError", 503, "api_error"),
        ("Timeout", 504, "timeout_error"),
        (None, 500, "api_error"),
    ),
)
@pytest.mark.parametrize("path", ("/v1/messages", "/v1/messages/count_tokens"))
def test_anthropic_routes_preserve_sanitized_error_envelope(
    exception_name: str | None, status: int, anthropic_type: str, path: str, monkeypatch, tmp_path
) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1", safe_mode=False)
    _block_catalog_fetches(monkeypatch)

    class SyntheticUpstreamError(Exception):
        pass

    if exception_name is not None:
        monkeypatch.setattr(module.litellm, exception_name, SyntheticUpstreamError)

    async def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise SyntheticUpstreamError(SECRET_MESSAGE)

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        rotating_client = module.app.state.rotating_client
        method = "anthropic_messages" if path == "/v1/messages" else "anthropic_count_tokens"
        monkeypatch.setattr(rotating_client, method, fail)
        body = anthropic_body()
        if path.endswith("count_tokens"):
            body.pop("max_tokens")
            body.pop("stream")
        response = client.post(
            path, headers={"x-api-key": "proxy-token", "Host": "127.0.0.1"}, json=body
        )
    assert response.status_code == status
    payload = response.json()
    assert set(payload) == {"type", "error"}
    assert payload["type"] == "error"
    assert set(payload["error"]) == {"type", "message"}
    assert payload["error"]["type"] == anthropic_type
    assert payload["error"]["message"]
    assert SECRET_MESSAGE not in response.text
    assert "detail" not in response.json()


def test_anthropic_authentication_failure_uses_anthropic_envelope(monkeypatch, tmp_path) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1", safe_mode=False)
    _block_catalog_fetches(monkeypatch)
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        response = client.post("/v1/messages", headers={"Host": "127.0.0.1"}, json=anthropic_body())
    assert response.status_code == 401
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "authentication_error"
    assert "detail" not in response.json()


@pytest.mark.asyncio
async def test_anthropic_stream_failure_uses_sanitized_anthropic_event(caplog) -> None:
    async def failing_stream():
        yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
        raise RuntimeError(SECRET_MESSAGE)

    import proxy_app.main as module

    with caplog.at_level("ERROR"):
        chunks = [
            chunk async for chunk in module.anthropic_streaming_response_wrapper(failing_stream())
        ]
    body = "".join(chunks)
    assert "event: error" in body
    assert '"type": "error"' in body
    assert '"type": "api_error"' in body
    assert SECRET_MESSAGE not in body
    assert SECRET_MESSAGE not in caplog.text
