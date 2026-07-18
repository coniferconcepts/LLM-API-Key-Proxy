from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

import pytest
from starlette.testclient import TestClient

from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.error_handler import ClassifiedError, RequestErrorAccumulator  # noqa: E402
from rotator_library.error_handler import NoAvailableKeysError  # noqa: E402

SECRET_MESSAGE = "UPSTREAM-PAYLOAD-SENTINEL"
SECRET_CREDENTIAL = "/private/oauth/STABLE-OAUTH-FILENAME-SENTINEL.json"


def _anthropic_body(*, stream: bool = False) -> dict[str, Any]:
    return {
        "model": "xai_oauth/grok-4.5",
        "messages": [{"role": "user", "content": "safe boundary"}],
        "max_tokens": 16,
        "stream": stream,
    }


def test_rotation_error_response_and_log_keep_only_categories_and_counts() -> None:
    # Given a classified credential failure containing upstream and credential sentinels.
    accumulator = RequestErrorAccumulator()
    accumulator.provider = "synthetic"
    accumulator.model = "synthetic/model"
    accumulator.record_error(
        SECRET_CREDENTIAL,
        ClassifiedError("authentication", RuntimeError(SECRET_MESSAGE), status_code=401),
        SECRET_MESSAGE,
    )
    accumulator.record_error(
        f"second-{SECRET_CREDENTIAL}",
        ClassifiedError(SECRET_MESSAGE, RuntimeError(SECRET_MESSAGE), status_code=599),
        SECRET_MESSAGE,
    )

    # When public client and log summaries are built.
    response = accumulator.build_client_error_response()
    log_message = accumulator.build_log_message()

    # Then only stable categories and aggregate counts remain.
    serialized = json.dumps(response)
    assert SECRET_MESSAGE not in serialized
    assert SECRET_CREDENTIAL not in serialized
    assert "STABLE-OAUTH-FILENAME-SENTINEL" not in serialized
    assert SECRET_MESSAGE not in log_message
    assert "STABLE-OAUTH-FILENAME-SENTINEL" not in log_message
    assert response["error"]["details"]["credentials_tried"] == 2
    assert response["error"]["details"]["failure_categories"] == {
        "authentication": 1,
        "unknown": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ("acquisition", "unexpected"))
async def test_terminal_sse_omits_exception_diagnostics_and_credential_identity(
    failure_kind: str,
    caplog,
    monkeypatch,
) -> None:
    # Given a streaming client whose credential acquisition fails with sentinel diagnostics.
    import rotator_library.client as client_module

    class AvailableCooldown:
        async def is_cooling_down(self, _provider: str) -> bool:
            return False

    class FailingUsageManager:
        async def get_credential_availability_stats(self, *_args: Any) -> dict[str, int]:
            return {"available": 1, "on_cooldown": 0, "fair_cycle_excluded": 0}

        async def acquire_key(self, **_kwargs: Any) -> str:
            if failure_kind == "acquisition":
                raise NoAvailableKeysError(
                    SECRET_MESSAGE,
                    code=SECRET_MESSAGE,
                    diagnostics={"provider_payload": SECRET_MESSAGE},
                )
            raise RuntimeError(SECRET_MESSAGE)

    rotating_client = object.__new__(client_module.RotatingClient)
    rotating_client.all_credentials = {"synthetic": [SECRET_CREDENTIAL]}
    rotating_client.global_timeout = 10.0
    rotating_client.acquire_timeout = 1.0
    rotating_client.enable_request_logging = False
    rotating_client.max_concurrent_requests_per_key = {"synthetic": 1}
    rotating_client.cooldown_manager = AvailableCooldown()
    rotating_client.usage_manager = FailingUsageManager()
    rotating_client._apply_routing_policy = lambda model: (model, None)
    rotating_client._get_provider_instance = lambda _provider: None
    rotating_client._resolve_model_id = lambda model, _provider: model
    monkeypatch.setattr(client_module.lib_logger, "propagate", True)

    # When the real terminal-SSE generator handles the failure.
    with caplog.at_level(logging.ERROR, logger="rotator_library"):
        chunks = [
            chunk
            async for chunk in rotating_client._streaming_acompletion_with_retry(
                None,
                model="synthetic/model",
                messages=[],
            )
        ]
    body = "".join(chunks)

    # Then only the fixed public category is exposed and the stream terminates normally.
    assert SECRET_MESSAGE not in body
    assert SECRET_CREDENTIAL not in body
    assert SECRET_MESSAGE not in caplog.text
    assert SECRET_CREDENTIAL not in caplog.text
    expected_status = 503 if failure_kind == "acquisition" else 500
    assert f'"status": {expected_status}' in body
    assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_low_level_stream_buffering_logs_omit_payload_and_credential_identity(
    caplog,
    monkeypatch,
) -> None:
    # Given a malformed upstream stream exception containing provider-controlled JSON.
    import rotator_library.client as client_module

    class UsageManager:
        async def release_key(self, _key: str, _model: str) -> None:
            return

    class FailingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError(f'Received chunk: {{"error":"{SECRET_MESSAGE}"}}')

    rotating_client = object.__new__(client_module.RotatingClient)
    rotating_client.usage_manager = UsageManager()
    monkeypatch.setattr(client_module.lib_logger, "propagate", True)

    # When the low-level stream wrapper promotes the provider error for rotation.
    with caplog.at_level(logging.INFO, logger="rotator_library"):
        with pytest.raises(client_module.StreamedAPIError):
            _ = [
                chunk
                async for chunk in rotating_client._safe_streaming_wrapper(
                    FailingStream(),
                    SECRET_CREDENTIAL,
                    "synthetic/model",
                )
            ]

    # Then logs contain only bounded metadata, never the payload or credential identity.
    assert SECRET_MESSAGE not in caplog.text
    assert SECRET_CREDENTIAL not in caplog.text
    assert "STABLE-OAUTH-FILENAME-SENTINEL" not in caplog.text


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
    exception_name: str | None,
    status: int,
    anthropic_type: str,
    path: str,
    monkeypatch,
    tmp_path,
) -> None:
    # Given an Anthropic-compatible route whose downstream fails in a known category.
    module = _import_proxy_main(
        monkeypatch,
        tmp_path,
        "http://127.0.0.1:2465/v1",
        safe_mode=False,
    )
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
        body = _anthropic_body()
        if path.endswith("count_tokens"):
            body.pop("max_tokens")
            body.pop("stream")
        response = client.post(
            path,
            headers={"x-api-key": "proxy-token", "Host": "127.0.0.1"},
            json=body,
        )

    # Then the status is preserved in Anthropic's error envelope without raw details.
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
    # Given an Anthropic request without the required proxy credential.
    module = _import_proxy_main(
        monkeypatch,
        tmp_path,
        "http://127.0.0.1:2465/v1",
        safe_mode=False,
    )
    _block_catalog_fetches(monkeypatch)

    # When authentication rejects it before endpoint dispatch.
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/v1/messages",
            headers={"Host": "127.0.0.1"},
            json=_anthropic_body(),
        )

    # Then the dependency failure remains Anthropic-compatible and sanitized.
    assert response.status_code == 401
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "authentication_error"
    assert "detail" not in response.json()


@pytest.mark.asyncio
async def test_anthropic_stream_failure_uses_sanitized_anthropic_event(caplog) -> None:
    # Given an Anthropic stream that fails after emitting a normal event.
    async def failing_stream():
        yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
        raise RuntimeError(SECRET_MESSAGE)

    import proxy_app.main as module

    # When the terminal error event is generated.
    with caplog.at_level(logging.ERROR):
        chunks = [
            chunk async for chunk in module.anthropic_streaming_response_wrapper(failing_stream())
        ]
    body = "".join(chunks)

    # Then it uses the Anthropic error event and leaks no exception text to client or logs.
    assert "event: error" in body
    assert '"type": "error"' in body
    assert '"type": "api_error"' in body
    assert SECRET_MESSAGE not in body
    assert SECRET_MESSAGE not in caplog.text
