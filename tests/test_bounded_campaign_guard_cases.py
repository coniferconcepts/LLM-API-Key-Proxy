import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

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


def test_bounded_guard_strips_preexisting_retry_kwarg(tmp_path: Path) -> None:
    """Guard must remove retry if a caller left it in prepared kwargs."""
    authorization = BoundedAuthorization(
        entry_id="entry-strip-retry",
        target="openai/gpt-5.6-sol",
        provider="openai",
        manifest_sha256="a" * 64,
        max_outbound_posts=79,
    )
    prepared: dict[str, object] = {"retry": 0, "num_retries": 3}
    asyncio.run(
        BoundedAttemptGuard(
            authorization,
            DurableReservationLedger(tmp_path / "ledger.jsonl"),
        )(None, prepared)
    )
    assert "retry" not in prepared
    assert prepared["num_retries"] == 0
    assert prepared["max_retries"] == 0


def test_openai_attempt_header_is_added_only_after_internal_authentication(tmp_path: Path) -> None:
    authorization = BoundedAuthorization(
        entry_id="entry-openai",
        target="openai/gpt-5.6-sol",
        provider="openai",
        manifest_sha256="a" * 64,
        max_outbound_posts=79,
    )
    prepared: dict[str, object] = {}
    asyncio.run(
        BoundedAttemptGuard(
            authorization,
            DurableReservationLedger(tmp_path / "ledger.jsonl"),
        )(None, prepared)
    )
    assert prepared["extra_headers"] == {"X-OpenCode-Bounded-Attempt": "1"}


def test_capability_headers_are_removed_from_failure_logging() -> None:
    class Request:
        headers = {
            "x-opencode-internal-bounded-entry": "entry-001",
            "x-opencode-internal-bounded-capability": "secret-capability",
            "x-request-id": "safe-id",
        }

    safe = _safe_request_headers(Request())

    assert safe == {"x-request-id": "safe-id"}
    assert "secret-capability" not in repr(safe)


def test_wrong_owner_manifest_is_rejected_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "runtime.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setenv("OPENCODE_BOUNDED_CAMPAIGN_MANIFEST_PATH", str(path))
    current_uid = os.geteuid()
    monkeypatch.setattr(bounded_module.os, "geteuid", lambda: current_uid + 1)

    with pytest.raises(BoundedCampaignError, match="owner"):
        validate_internal_request(
            {
                INTERNAL_ENTRY_HEADER: "entry-000",
                INTERNAL_CAPABILITY_HEADER: "not-valid",
            },
            {"model": "fake/model"},
            method="POST",
            path="/v1/chat/completions",
        )


def test_internal_body_tamper_is_rejected_before_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"model": "fake/model", "messages": []}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    entries = [
        {
            "entry_id": f"entry-{index:03d}",
            "target": "fake/model",
            "method": "POST",
            "path": "/v1/chat/completions",
            "body_sha256": hashlib.sha256(canonical).hexdigest(),
            "internal_capability": f"capability-{index:03d}-" + "x" * 64,
        }
        for index in range(63)
    ]
    manifest = {
        "expires_at": time.time() + 60,
        "max_outbound_posts": 79,
        "entries": entries,
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setenv("OPENCODE_BOUNDED_CAMPAIGN_MANIFEST_PATH", str(path))

    with pytest.raises(BoundedCampaignError, match="binding"):
        validate_internal_request(
            {
                INTERNAL_ENTRY_HEADER: "entry-000",
                INTERNAL_CAPABILITY_HEADER: "capability-000-" + "x" * 64,
            },
            {**payload, "temperature": 1},
            method="POST",
            path="/v1/chat/completions",
        )
