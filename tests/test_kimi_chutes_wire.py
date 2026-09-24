from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import litellm
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rotator_library.provider_config import ProviderConfig
from rotator_library.request_sanitizer import sanitize_request_payload


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [2048, 4096])
async def test_kimi_chutes_litellm_wire_fields(monkeypatch: pytest.MonkeyPatch, cap: int) -> None:
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append({"path": self.path, "body": body})
            response = json.dumps(
                {
                    "id": "chatcmpl-synthetic",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "moonshotai/Kimi-K3-TEE",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("CHUTES_API_BASE", f"http://127.0.0.1:{server.server_port}/v1")
        model = "chutes/moonshotai/Kimi-K3-TEE"
        request = sanitize_request_payload(
            {
                "model": model,
                "messages": [{"role": "user", "content": "synthetic"}],
                "stream": False,
                "max_completion_tokens": cap,
                "reasoning_effort": "max",
            },
            model,
        )
        kwargs = ProviderConfig().convert_for_litellm(**request)
        assert kwargs["max_completion_tokens"] == cap
        assert kwargs["reasoning_effort"] == "max"
        assert kwargs["api_base"].startswith("http://127.0.0.1:")
        supported = litellm.get_supported_openai_params(model=model)
        assert "max_completion_tokens" in supported
        assert "reasoning_effort" not in supported
        with monkeypatch.context() as scope:
            scope.setattr(litellm, "drop_params", True)
            await litellm.acompletion(
                **kwargs, api_key="synthetic-local-only", num_retries=0, timeout=3
            )
        assert len(received) == 1
        assert received[0]["path"] == "/v1/chat/completions"
        body = received[0]["body"]
        assert body["model"] == "moonshotai/Kimi-K3-TEE"
        assert body.get("stream", False) is False
        assert body.get("max_tokens") == cap, sorted(body)
        assert "max_completion_tokens" not in body
        assert "reasoning_effort" not in body
        assert "tools" not in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_kimi_chutes_thinking_controls_reach_wire_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            response = json.dumps(
                {
                    "id": "chatcmpl-synthetic",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "moonshotai/Kimi-K3-TEE",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("CHUTES_API_BASE", f"http://127.0.0.1:{server.server_port}/v1")
        model = "chutes/moonshotai/Kimi-K3-TEE"

        def request_kwargs() -> dict:
            payload = sanitize_request_payload(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "synthetic"}],
                    "stream": False,
                    "max_tokens": 16,
                },
                model,
            )
            return ProviderConfig().convert_for_litellm(**payload)

        ordinary = request_kwargs()
        disabled = request_kwargs()
        disabled["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
        budgeted = request_kwargs()
        budgeted["extra_body"] = {"reasoning_effort": "low"}

        with monkeypatch.context() as scope:
            scope.setattr(litellm, "drop_params", True)
            for kwargs in (ordinary, disabled, budgeted):
                await litellm.acompletion(
                    **kwargs, api_key="synthetic-local-only", num_retries=0, timeout=3
                )

        assert len(received) == 3
        assert received[0]["model"] == "moonshotai/Kimi-K3-TEE"
        assert "chat_template_kwargs" not in received[0]
        assert "reasoning_effort" not in received[0]
        assert received[1]["model"] == "moonshotai/Kimi-K3-TEE"
        assert received[1]["chat_template_kwargs"] == {"thinking": False}
        assert received[2]["model"] == "moonshotai/Kimi-K3-TEE"
        assert received[2]["reasoning_effort"] == "low"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
