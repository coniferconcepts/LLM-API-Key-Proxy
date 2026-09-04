from __future__ import annotations

import json
import logging
from typing import Any

from starlette.testclient import TestClient

from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main

SECRET_MARKER = "SECRET-MARKER-provider-payload-" + ("x" * 300)


def test_safe_exception_logger_omits_exception_text(caplog) -> None:
    from proxy_app.safe_errors import log_safe_exception

    with caplog.at_level(logging.ERROR):
        log_safe_exception("Synthetic provider failure", RuntimeError(SECRET_MARKER), 500)

    assert "RuntimeError" in caplog.text
    assert SECRET_MARKER not in caplog.text


class RecordingRawLogger:
    final_responses: list[dict[str, Any]] = []

    def log_request(self, **_kwargs: Any) -> None:
        return

    def log_stream_chunk(self, _chunk: dict[str, Any]) -> None:
        return

    def log_final_response(self, **kwargs: Any) -> None:
        self.final_responses.append(kwargs)


def _openai_body(*, stream: bool) -> dict[str, Any]:
    return {
        "model": "xai_oauth/grok-4.5",
        "messages": [{"role": "user", "content": "safe error contract"}],
        "stream": stream,
    }


def _anthropic_body(*, stream: bool | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "xai_oauth/grok-4.5",
        "messages": [{"role": "user", "content": "safe error contract"}],
        "max_tokens": 16,
    }
    if stream is not None:
        body["stream"] = stream
    return body


def _json_body_at_size(size: int) -> bytes:
    prefix = b'{"model":"xai_oauth/grok-4.5","messages":[{"role":"user","content":"'
    suffix = b'"}],"stream":false}'
    return prefix + (b"x" * (size - len(prefix) - len(suffix))) + suffix


def test_all_json_post_routes_enforce_limit_before_dispatch(monkeypatch, tmp_path) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    dispatches = 0

    async def fail_if_dispatched(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal dispatches
        dispatches += 1
        raise RuntimeError("boundary passed")

    oversized = _json_body_at_size(4_194_305)
    paths = (
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/embeddings",
        "/v1/quota-stats",
        "/v1/token-count",
        "/v1/cost-estimate",
    )
    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", fail_if_dispatched)
        for path in paths:
            headers = {"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"}
            response = client.post(path, headers=headers, content=oversized)
            assert response.status_code == 413

        exact = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            content=_json_body_at_size(4_194_304),
        )

    assert exact.status_code == 500
    assert dispatches == 1


def test_precommit_endpoint_errors_are_low_cardinality_and_marker_free(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    monkeypatch.setattr(module, "ENABLE_RAW_LOGGING", True)
    monkeypatch.setattr(module, "RawIOLogger", RecordingRawLogger)
    RecordingRawLogger.final_responses.clear()

    async def fail_async(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(SECRET_MARKER)

    def fail_sync(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(SECRET_MARKER)

    with TestClient(
        module.app,
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    ) as client:
        rotating_client = module.app.state.rotating_client
        headers = {"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"}
        anthropic_headers = {"x-api-key": "proxy-token", "Host": "127.0.0.1"}
        cases = []

        monkeypatch.setattr(rotating_client, "acompletion", fail_async)
        cases.append(
            client.post("/v1/chat/completions", headers=headers, json=_openai_body(stream=False))
        )

        monkeypatch.setattr(rotating_client, "acompletion", fail_sync)
        cases.append(
            client.post("/v1/chat/completions", headers=headers, json=_openai_body(stream=True))
        )

        monkeypatch.setattr(rotating_client, "anthropic_messages", fail_async)
        cases.append(
            client.post(
                "/v1/messages", headers=anthropic_headers, json=_anthropic_body(stream=False)
            )
        )
        cases.append(
            client.post(
                "/v1/messages", headers=anthropic_headers, json=_anthropic_body(stream=True)
            )
        )

        monkeypatch.setattr(rotating_client, "anthropic_count_tokens", fail_async)
        count_body = _anthropic_body()
        count_body.pop("max_tokens")
        cases.append(
            client.post("/v1/messages/count_tokens", headers=anthropic_headers, json=count_body)
        )

        monkeypatch.setattr(rotating_client, "aembedding", fail_async)
        cases.append(
            client.post(
                "/v1/embeddings",
                headers=headers,
                json={"model": "xai_oauth/embed", "input": "safe error contract"},
            )
        )

        monkeypatch.setattr(rotating_client, "get_quota_stats", fail_async)
        cases.append(client.get("/v1/quota-stats", headers=headers))

        monkeypatch.setattr(rotating_client, "reload_usage_from_disk", fail_async)
        cases.append(client.post("/v1/quota-stats", headers=headers, json={"action": "reload"}))

        monkeypatch.setattr(rotating_client, "token_count", fail_sync)
        cases.append(
            client.post(
                "/v1/token-count",
                headers=headers,
                json={
                    "model": "xai_oauth/grok-4.5",
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        )

        class FailingModelInfoService:
            is_ready = True

            def calculate_cost(self, *_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError(SECRET_MARKER)

            async def stop(self) -> None:
                return

        module.app.state.model_info_service = FailingModelInfoService()
        cases.append(
            client.post(
                "/v1/cost-estimate",
                headers=headers,
                json={"model": "xai_oauth/grok-4.5", "prompt_tokens": 1},
            )
        )

        monkeypatch.setattr(rotating_client, "get_all_available_models", fail_async)
        cases.append(client.get("/v1/models", headers=headers))

    assert [response.status_code for response in cases] == [
        500,
        500,
        409,
        409,
        409,
        409,
        500,
        500,
        409,
        500,
        500,
    ]
    for response in cases[:2] + cases[6:8] + cases[9:]:
        assert response.json()["detail"] == {
            "type": "proxy_internal_error",
            "code": "internal_error",
            "status": 500,
            "message": "The proxy could not complete the request.",
        }
    for response in cases[2:5]:
        assert response.json() == {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "The request conflicts with the current proxy mode.",
            },
        }
    for response in (cases[5], cases[8]):
        assert response.json()["detail"] == {
            "type": "conflict_error",
            "code": "local_transport_endpoint_disabled",
            "status": 409,
            "message": "The request conflicts with the current proxy mode.",
        }
    for response in cases:
        assert SECRET_MARKER not in response.text
    assert SECRET_MARKER not in caplog.text
    assert SECRET_MARKER not in json.dumps(RecordingRawLogger.final_responses)


def test_postcommit_stream_errors_are_fixed_and_marker_free(
    monkeypatch,
    tmp_path,
    caplog,
) -> None:
    module = _import_proxy_main(monkeypatch, tmp_path, "http://127.0.0.1:2465/v1")
    _block_catalog_fetches(monkeypatch)
    monkeypatch.setattr(module, "ENABLE_RAW_LOGGING", True)
    monkeypatch.setattr(module, "RawIOLogger", RecordingRawLogger)
    RecordingRawLogger.final_responses.clear()

    async def failing_stream():
        yield 'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'
        raise RuntimeError(SECRET_MARKER)

    def stream_factory(*_args: Any, **_kwargs: Any):
        return failing_stream()

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        monkeypatch.setattr(module.app.state.rotating_client, "acompletion", stream_factory)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json=_openai_body(stream=True),
        ) as response:
            body = response.read().decode("utf-8")

        async def collect_anthropic_chunks() -> list[str]:
            return [
                chunk
                async for chunk in module.anthropic_streaming_response_wrapper(
                    failing_stream(),
                    RecordingRawLogger(),
                )
            ]

        anthropic_body = "".join(client.portal.call(collect_anthropic_chunks))

    assert response.status_code == 200
    assert '"type": "proxy_internal_error"' in body
    assert '"code": "stream_error"' in body
    assert "data: [DONE]" in body
    assert SECRET_MARKER not in body
    assert "event: error" in anthropic_body
    assert '"type": "error"' in anthropic_body
    assert '"type": "api_error"' in anthropic_body
    assert '"code": "stream_error"' not in anthropic_body
    assert SECRET_MARKER not in anthropic_body
    assert SECRET_MARKER not in caplog.text
    assert [record["status_code"] for record in RecordingRawLogger.final_responses] == [500, 500]
