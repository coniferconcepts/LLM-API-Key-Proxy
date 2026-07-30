from __future__ import annotations

import json
from typing import Any

from starlette.testclient import TestClient

from error_boundary_regression_support import SECRET_CREDENTIAL, SECRET_MESSAGE
from rotator_library.error_handler import (
    ClassifiedError,
    NoAvailableKeysError,
    RequestErrorAccumulator,
    build_public_stream_error,
)
from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main


def test_openai_credential_failures_preserve_http_and_sse_boundaries_without_redispatch(
    monkeypatch, tmp_path
) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1", safe_mode=False)
    _block_catalog_fetches(monkeypatch)
    dispatch_calls: list[str] = []
    accumulator = RequestErrorAccumulator()
    accumulator.record_error(
        SECRET_CREDENTIAL,
        ClassifiedError("quota_exceeded", RuntimeError(SECRET_MESSAGE), status_code=429),
        SECRET_MESSAGE,
    )
    exhausted_response = accumulator.build_client_error_response()

    async def exhausted(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        dispatch_calls.append("exhausted")
        return exhausted_response

    async def busy(*_args: Any, **_kwargs: Any) -> Any:
        dispatch_calls.append("busy")
        raise NoAvailableKeysError(SECRET_MESSAGE, category="proxy_busy")

    def exhausted_stream(*_args: Any, **_kwargs: Any) -> Any:
        dispatch_calls.append("sse_exhausted")

        async def chunks():
            yield f"data: {json.dumps(build_public_stream_error('proxy_all_credentials_exhausted'))}\n\n"
            yield "data: [DONE]\n\n"

        return chunks()

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", exhausted)
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={
                "model": "synthetic/model",
                "messages": [{"role": "user", "content": "safe boundary"}],
                "stream": False,
            },
        )
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", busy)
        busy_response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={
                "model": "synthetic/model",
                "messages": [{"role": "user", "content": "safe boundary"}],
                "stream": False,
            },
        )
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", exhausted_stream)
        sse_response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={
                "model": "synthetic/model",
                "messages": [{"role": "user", "content": "safe boundary"}],
                "stream": True,
            },
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "all_credentials_exhausted"
    assert response.json()["error"]["status"] == 503
    assert busy_response.status_code == 503
    assert busy_response.json() == build_public_stream_error("proxy_busy")
    assert sse_response.status_code == 200
    assert '"type": "proxy_all_credentials_exhausted"' in sse_response.text
    assert '"code": "all_credentials_exhausted"' in sse_response.text
    assert "data: [DONE]" in sse_response.text
    assert dispatch_calls == ["exhausted", "busy", "sse_exhausted"]
    assert SECRET_MESSAGE not in response.text
    assert SECRET_MESSAGE not in busy_response.text
    assert SECRET_MESSAGE not in sse_response.text
    assert SECRET_CREDENTIAL not in response.text
