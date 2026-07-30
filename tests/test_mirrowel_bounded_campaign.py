import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

bounded_module = importlib.import_module("rotator_library.bounded_campaign")
client_module = importlib.import_module("rotator_library.client")
BoundedAttemptGuard = getattr(bounded_module, "BoundedAttemptGuard")
BoundedAuthorization = getattr(bounded_module, "BoundedAuthorization")
BoundedCampaignError = getattr(bounded_module, "BoundedCampaignError")
DurableReservationLedger = getattr(bounded_module, "DurableReservationLedger")
INTERNAL_CAPABILITY_HEADER = getattr(bounded_module, "INTERNAL_CAPABILITY_HEADER")
INTERNAL_ENTRY_HEADER = getattr(bounded_module, "INTERNAL_ENTRY_HEADER")
validate_internal_request = getattr(bounded_module, "validate_internal_request")
_safe_request_headers = getattr(client_module, "_safe_request_headers")


def _authorization(entry_id: str, manifest_hash: str = "a" * 64) -> BoundedAuthorization:
    return BoundedAuthorization(
        entry_id=entry_id,
        target="fake/model",
        provider="fake",
        manifest_sha256=manifest_hash,
        max_outbound_posts=79,
    )


def test_80_concurrent_reservations_delegate_exactly_79(tmp_path: Path) -> None:
    ledger = DurableReservationLedger(tmp_path / "ledger.jsonl")

    def reserve(index: int) -> bool:
        try:
            ledger.reserve(_authorization(f"entry-{index:03d}"))
        except BoundedCampaignError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=80) as executor:
        accepted = list(executor.map(reserve, range(80)))

    assert sum(accepted) == 79
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 80


def test_ledger_rejects_replay_and_resumes_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    DurableReservationLedger(path).reserve(_authorization("entry-001"))
    restarted = DurableReservationLedger(path)

    with pytest.raises(BoundedCampaignError, match="replay"):
        restarted.reserve(_authorization("entry-001"))

    restarted.reserve(_authorization("entry-002"))
    assert "entry-002" in path.read_text()


@pytest.mark.parametrize("state", ["corrupt", "symlink", "permissions"])
def test_ledger_rejects_unsafe_state_before_reservation(tmp_path: Path, state: str) -> None:
    path = tmp_path / "ledger.jsonl"
    if state == "corrupt":
        path.write_text("not-json\n", encoding="utf-8")
        path.chmod(0o600)
    elif state == "symlink":
        target = tmp_path / "target"
        target.write_text("{}\n", encoding="utf-8")
        target.chmod(0o600)
        path.symlink_to(target)
    else:
        path.write_text("{}\n", encoding="utf-8")
        path.chmod(0o644)

    with pytest.raises(BoundedCampaignError):
        DurableReservationLedger(path).reserve(_authorization("entry-001"))


def test_mirrowel_independently_validates_exact_internal_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"model": "fake/model", "messages": [], "stream": False}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "expires_at": time.time() + 60,
        "max_outbound_posts": 79,
        "entries": [
            {
                "entry_id": f"entry-{index:03d}",
                "target": "fake/model",
                "method": "POST",
                "path": "/v1/chat/completions",
                "body_sha256": hashlib.sha256(canonical).hexdigest(),
                "internal_capability": f"capability-{index:03d}-" + "x" * 64,
            }
            for index in range(63)
        ],
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setenv("OPENCODE_BOUNDED_CAMPAIGN_MANIFEST_PATH", str(path))

    authorization = validate_internal_request(
        {
            INTERNAL_ENTRY_HEADER: "entry-000",
            INTERNAL_CAPABILITY_HEADER: "capability-000-" + "x" * 64,
        },
        payload,
        method="POST",
        path="/v1/chat/completions",
    )

    assert authorization is not None
    assert authorization.entry_id == "entry-000"


@pytest.mark.parametrize(
    ("stream", "tool", "status"),
    [(False, False, 200), (True, False, 200), (True, True, 200), (False, False, 429)],
)
def test_guard_allows_one_physical_post_for_all_response_modes(
    tmp_path: Path, stream: bool, tool: bool, status: int
) -> None:
    received: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            received.append(self.rfile.read(length))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b"data: [DONE]\n\n" if stream else b'{"ok":true}')

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    authorization = _authorization("entry-001")
    guard = BoundedAttemptGuard(
        authorization,
        DurableReservationLedger(tmp_path / "ledger.jsonl"),
    )
    prepared: dict[str, object] = {"stream": stream, "tools": [{}] if tool else []}

    async def attempt_twice() -> None:
        for _attempt in range(2):
            try:
                await guard(None, prepared)
            except BoundedCampaignError:
                break
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions",
                data=b"{}",
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=2) as response:
                    response.read()
            except urllib.error.HTTPError:
                continue

    try:
        asyncio.run(attempt_twice())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert len(received) == 1
    assert prepared["num_retries"] == 0
    assert prepared["max_retries"] == 0
    # Must not inject OpenAI-incompatible body field "retry" (codex-lb 400).
    assert "retry" not in prepared
