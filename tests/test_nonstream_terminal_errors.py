"""#47: a non-stream error or empty result must never reach the router as HTTP 200."""

from __future__ import annotations

import json
from typing import Any

import pytest
from starlette.testclient import TestClient

from error_boundary_regression_support import SECRET_CREDENTIAL, SECRET_MESSAGE
from proxy_app.safe_errors import public_error_detail
from rotator_library.error_handler import ClassifiedError, RequestErrorAccumulator
from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main


class _RecordingRawLogger:
    calls: list[tuple[int, dict[str, Any]]] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __getattr__(self, _name: str) -> Any:
        return lambda *_a, **_k: None

    def log_final_response(self, status_code: int, headers: Any, body: Any) -> None:
        _RecordingRawLogger.calls.append((status_code, body))


def _timeout_result() -> dict[str, Any]:
    accumulator = RequestErrorAccumulator()
    accumulator.record_error(
        SECRET_CREDENTIAL,
        ClassifiedError("server_error", RuntimeError(SECRET_MESSAGE), status_code=500),
        SECRET_MESSAGE,
    )
    accumulator.timeout_occurred = True
    return accumulator.build_client_error_response()


def _exhausted_result() -> dict[str, Any]:
    accumulator = RequestErrorAccumulator()
    accumulator.record_error(
        SECRET_CREDENTIAL,
        ClassifiedError("quota_exceeded", RuntimeError(SECRET_MESSAGE), status_code=429),
        SECRET_MESSAGE,
    )
    return accumulator.build_client_error_response()


CASES = [
    ("timeout", _timeout_result, {504}),
    ("empty", lambda: None, {500, 502}),
    (
        "other-error-429",
        lambda: {"error": {"type": "provider_error", "code": "x", "status": 429}},
        {500, 502, 503, 504},
    ),
    (
        "other-error-no-status",
        lambda: {"error": {"type": "provider_error", "message": SECRET_MESSAGE}},
        {500, 502, 503, 504},
    ),
]


def _post(module: Any, monkeypatch: Any, result: Any) -> Any:
    async def completion(*_args: Any, **_kwargs: Any) -> Any:
        return result

    with TestClient(module.app, base_url="http://127.0.0.1", raise_server_exceptions=False) as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", completion)
        return client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={
                "model": "synthetic/model",
                "messages": [{"role": "user", "content": "safe boundary"}],
                "stream": False,
            },
        )


@pytest.mark.parametrize("raw_logging", [False, True], ids=["plain", "raw-logging"])
@pytest.mark.parametrize(("name", "make_result", "allowed"), CASES, ids=[c[0] for c in CASES])
def test_nonstream_terminal_error_gets_its_own_allowlisted_status(
    monkeypatch, tmp_path, raw_logging, name, make_result, allowed
) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1", safe_mode=False)
    _block_catalog_fetches(monkeypatch)
    _RecordingRawLogger.calls = []
    if raw_logging:
        monkeypatch.setattr(module, "ENABLE_RAW_LOGGING", True)
        monkeypatch.setattr(module, "RawIOLogger", _RecordingRawLogger)

    response = _post(module, monkeypatch, make_result())

    assert response.status_code in allowed, name
    assert response.status_code not in {200, 429}
    assert SECRET_MESSAGE not in response.text and SECRET_CREDENTIAL not in response.text
    assert response.json() == {"detail": public_error_detail(response.status_code)}
    if raw_logging:
        assert _RecordingRawLogger.calls, "raw logger saw no final response"
        assert _RecordingRawLogger.calls[-1][0] == response.status_code


def test_exhausted_credentials_response_is_unchanged(monkeypatch, tmp_path) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1", safe_mode=False)
    _block_catalog_fetches(monkeypatch)
    result = _exhausted_result()

    response = _post(module, monkeypatch, result)

    assert response.status_code == 503
    assert response.content == json.dumps(result, separators=(",", ":")).encode()
