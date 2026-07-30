from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from types import ModuleType
from typing import TypeAlias, cast
import sys
import threading
import time

from filelock import FileLock
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotator_library.bounded_campaign import (  # noqa: E402
    BoundedAuthorization,
    DurableReservationLedger,
    INTERNAL_CAPABILITY_HEADER,
    INTERNAL_ENTRY_HEADER,
)
from proxy_app import local_transport_policy  # noqa: E402
from test_local_transport_safe_mode import (  # noqa: E402
    FAKE_XAI_LOCK_PATH,
    _block_catalog_fetches,
    _import_proxy_main,
)

MODEL = "xai_oauth/grok-4.5"
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class ProviderScenario:
    status: int = 200
    disconnect: bool = False
    redirect: bool = False


@dataclass(frozen=True, slots=True)
class RecordedPost:
    path: str
    body: JSONObject


@dataclass(frozen=True, slots=True)
class FakeProvider:
    api_base: str
    port: int
    posts: list[RecordedPost]


@contextmanager
def fake_provider(scenario: ProviderScenario) -> Iterator[FakeProvider]:
    posts: list[RecordedPost] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            decoded = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            body = cast(JSONObject, decoded)
            posts.append(RecordedPost(path=self.path, body=body))
            if scenario.disconnect:
                self.connection.close()
                return
            if scenario.redirect:
                self.send_response(307)
                self.send_header("Location", "/redirected")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if scenario.status != 200:
                self._write_response(
                    scenario.status,
                    {"error": {"message": "fixture rejection", "type": "rate_limit"}},
                )
                return
            if body.get("stream"):
                encoded = (
                    'data: {"id":"fixture","object":"chat.completion.chunk",'
                    '"created":1,"model":"grok-4.5","choices":[{"index":0,'
                    '"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                    "data: [DONE]\n\n"
                ).encode()
                self._write_encoded(200, "text/event-stream", encoded)
                return
            message: JSONObject = {"role": "assistant", "content": "ok"}
            if body.get("tools"):
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fixture",
                            "type": "function",
                            "function": {"name": "fixture_tool", "arguments": "{}"},
                        }
                    ],
                }
            self._write_response(
                200,
                {
                    "id": "fixture",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "grok-4.5",
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": "stop"}
                    ],
                },
            )

        def _write_response(self, status: int, body: JSONObject) -> None:
            self._write_encoded(status, "application/json", json.dumps(body).encode())

        def _write_encoded(
            self, status: int, content_type: str, body: bytes
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: str | int | float) -> None:
            return

    with FileLock(FAKE_XAI_LOCK_PATH, timeout=120):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        provider = FakeProvider(
            api_base=f"http://127.0.0.1:{port}/v1",
            port=port,
            posts=posts,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield provider
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def payload(*, stream: bool = False, tool: bool = False) -> JSONObject:
    request: JSONObject = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "bounded fixture"}],
        "stream": stream,
    }
    if tool:
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "fixture_tool",
                    "description": "fixture",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        request["tool_choice"] = "required"
    return request


def install_bounded_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: JSONObject,
) -> tuple[Path, dict[str, str]]:
    body_sha256 = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    entries = [
        {
            "entry_id": f"entry-{index:03d}",
            "target": MODEL,
            "method": "POST",
            "path": "/v1/chat/completions",
            "body_sha256": body_sha256,
            "internal_capability": f"internal-{index:03d}-" + "c" * 64,
        }
        for index in range(63)
    ]
    manifest_path = tmp_path / "runtime.json"
    manifest_path.write_text(
        json.dumps(
            {
                "expires_at": time.time() + 60,
                "max_outbound_posts": 79,
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )
    manifest_path.chmod(0o600)
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("OPENCODE_BOUNDED_CAMPAIGN_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("OPENCODE_BOUNDED_CAMPAIGN_LEDGER_PATH", str(ledger_path))
    return ledger_path, {
        "Authorization": "Bearer proxy-token",
        "Host": "127.0.0.1",
        INTERNAL_ENTRY_HEADER: entries[0]["entry_id"],
        INTERNAL_CAPABILITY_HEADER: entries[0]["internal_capability"],
    }


def proxy_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: FakeProvider,
) -> ModuleType:
    monkeypatch.setattr(local_transport_policy, "LOCAL_XAI_PORT", provider.port)
    module = _import_proxy_main(
        monkeypatch,
        tmp_path,
        provider.api_base,
    )
    api_keys = cast(dict[str, list[str]], module.api_keys)
    api_keys["xai_oauth"].append("second-fake-key")
    _block_catalog_fetches(monkeypatch)
    return module


def reserve_synthetic_campaign_headroom(ledger_path: Path, count: int) -> None:
    manifest_path = ledger_path.with_name("runtime.json")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    ledger = DurableReservationLedger(ledger_path)
    for index in range(count):
        ledger.reserve(
            BoundedAuthorization(
                entry_id=f"synthetic-reservation-{index:03d}",
                target=MODEL,
                provider="xai_oauth",
                manifest_sha256=manifest_sha256,
                max_outbound_posts=79,
            )
        )
