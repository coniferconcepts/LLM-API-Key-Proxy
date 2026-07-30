"""Physical-wire proof: local OpenAI base receives bare catalog model ids."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotator_library.provider_config import ProviderConfig  # noqa: E402


def test_openai_route_identity_posts_bare_catalog_model_to_loopback_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive convert_for_litellm then real HTTP POST body shape to a loopback fake.

    Router/Mirrowel auth still uses openai/gpt-5.6-sol; the wire JSON model must be bare.
    """
    recorded: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            recorded.append({"path": self.path, "body": body})
            payload = {
                "id": "chatcmpl_fixture",
                "object": "chat.completion",
                "created": 1,
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "READY"},
                        "finish_reason": "stop",
                    }
                ],
            }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        api_base = f"http://127.0.0.1:{port}/v1"
        monkeypatch.setenv("OPENAI_API_BASE", api_base)
        config = ProviderConfig()
        # Route identity remains fully qualified (campaign/auth binding).
        route_model = "openai/gpt-5.6-sol"
        litellm_kwargs = config.convert_for_litellm(
            model=route_model,
            api_key="sk-clb-fixture",
            messages=[{"role": "user", "content": "Reply with the single word READY."}],
            stream=False,
        )
        assert litellm_kwargs["model"] == "gpt-5.6-sol"
        assert not str(litellm_kwargs["model"]).startswith("openai/")

        # Physical POST using the same JSON fields LiteLLM would send to the base.
        import urllib.request

        request = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=json.dumps(
                {
                    "model": litellm_kwargs["model"],
                    "messages": litellm_kwargs["messages"],
                    "stream": False,
                }
            ).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer sk-clb-fixture",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200

        assert len(recorded) == 1
        assert recorded[0]["path"] in {"/v1/chat/completions", "/chat/completions"}
        assert recorded[0]["body"]["model"] == "gpt-5.6-sol"
        assert recorded[0]["body"]["model"] != route_model
        assert not str(recorded[0]["body"]["model"]).startswith("openai/")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
