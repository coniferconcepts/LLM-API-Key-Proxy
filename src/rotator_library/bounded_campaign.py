from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import time
from typing import Any, Mapping

INTERNAL_ENTRY_HEADER = "x-opencode-internal-bounded-entry"
INTERNAL_CAPABILITY_HEADER = "x-opencode-internal-bounded-capability"
ATTEMPT_HEADER = "x-opencode-bounded-attempt"
MANIFEST_ENV = "OPENCODE_BOUNDED_CAMPAIGN_MANIFEST_PATH"
LEDGER_ENV = "OPENCODE_BOUNDED_CAMPAIGN_LEDGER_PATH"


class BoundedCampaignError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BoundedAuthorization:
    entry_id: str
    target: str
    provider: str
    manifest_sha256: str
    max_outbound_posts: int


def canonical_body_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_owner_file(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise BoundedCampaignError("bounded state unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise BoundedCampaignError("bounded state is not a regular file")
    if metadata.st_uid != os.geteuid():
        raise BoundedCampaignError("bounded state owner mismatch")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise BoundedCampaignError("bounded state permissions are not owner-only")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            return stream.read()
    except OSError as error:
        raise BoundedCampaignError("bounded state cannot be read safely") from error


def _load_manifest(path: Path) -> tuple[dict[str, object], str]:
    raw = _read_owner_file(path)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise BoundedCampaignError("bounded manifest is malformed") from error
    if not isinstance(value, dict):
        raise BoundedCampaignError("bounded manifest root is malformed")
    return value, hashlib.sha256(raw).hexdigest()


def validate_internal_request(
    headers: Mapping[str, str],
    payload: Mapping[str, object],
    *,
    method: str,
    path: str,
    now: float | None = None,
) -> BoundedAuthorization | None:
    entry_id = headers.get(INTERNAL_ENTRY_HEADER)
    capability = headers.get(INTERNAL_CAPABILITY_HEADER)
    attempt = headers.get(ATTEMPT_HEADER)
    if entry_id is None and capability is None:
        if attempt is not None:
            raise BoundedCampaignError("bounded attempt header requires internal authentication")
        return None
    if not entry_id or not capability or attempt is not None:
        raise BoundedCampaignError("internal bounded headers are malformed")
    raw_manifest_path = os.environ.get(MANIFEST_ENV)
    if not raw_manifest_path:
        raise BoundedCampaignError("bounded campaign is disabled")
    manifest, manifest_sha256 = _load_manifest(Path(raw_manifest_path))
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 63:
        raise BoundedCampaignError("bounded manifest entry count mismatch")
    if not all(isinstance(entry, dict) for entry in entries):
        raise BoundedCampaignError("bounded manifest entry is malformed")
    entry_ids = [entry.get("entry_id") for entry in entries]
    internal_capabilities = [entry.get("internal_capability") for entry in entries]
    if any(not isinstance(value, str) or not value for value in entry_ids):
        raise BoundedCampaignError("bounded manifest entry ID is malformed")
    if any(not isinstance(value, str) or len(value) < 64 for value in internal_capabilities):
        raise BoundedCampaignError("bounded manifest capability is malformed")
    if len(set(entry_ids)) != 63 or len(set(internal_capabilities)) != 63:
        raise BoundedCampaignError("bounded manifest identifiers are not unique")
    matches = [
        entry for entry in entries if isinstance(entry, dict) and entry.get("entry_id") == entry_id
    ]
    if len(matches) != 1:
        raise BoundedCampaignError("bounded entry is unknown or duplicated")
    entry = matches[0]
    expected = entry.get("internal_capability")
    target = entry.get("target")
    expires_at = manifest.get("expires_at")
    maximum = manifest.get("max_outbound_posts")
    if not isinstance(expected, str) or not isinstance(target, str):
        raise BoundedCampaignError("bounded entry is malformed")
    effective_now = time.time() if now is None else now
    if not isinstance(expires_at, (int, float)) or effective_now >= expires_at:
        raise BoundedCampaignError("bounded campaign has expired")
    if not isinstance(maximum, int) or maximum != 79:
        raise BoundedCampaignError("bounded campaign limit mismatch")
    if not hmac.compare_digest(capability, expected):
        raise BoundedCampaignError("internal bounded capability rejected")
    bindings = (
        (entry.get("method"), method),
        (entry.get("path"), path),
        (entry.get("body_sha256"), canonical_body_digest(payload)),
        (payload.get("model"), target),
    )
    if any(actual != expected_value for actual, expected_value in bindings):
        raise BoundedCampaignError("internal bounded request binding mismatch")
    return BoundedAuthorization(
        entry_id=entry_id,
        target=target,
        provider=target.split("/", 1)[0],
        manifest_sha256=manifest_sha256,
        max_outbound_posts=maximum,
    )


class DurableReservationLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_name(f"{path.name}.lock")

    @classmethod
    def from_environment(cls) -> DurableReservationLedger:
        raw_path = os.environ.get(LEDGER_ENV)
        if not raw_path:
            raise BoundedCampaignError("bounded ledger is disabled")
        return cls(Path(raw_path))

    def reserve(self, authorization: BoundedAuthorization) -> None:
        lock_descriptor = self._open_owner_file(self.lock_path, create=True)
        with os.fdopen(lock_descriptor, "a+b", buffering=0) as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            entries = self._read_entries(authorization.manifest_sha256)
            if authorization.entry_id in entries:
                raise BoundedCampaignError("bounded entry replay rejected")
            if len(entries) >= authorization.max_outbound_posts:
                raise BoundedCampaignError("bounded campaign physical POST limit reached")
            ledger_descriptor = self._open_owner_file(self.path, create=True)
            with os.fdopen(ledger_descriptor, "a", encoding="utf-8") as ledger:
                if not entries and ledger.tell() == 0:
                    ledger.write(
                        json.dumps({"manifest_sha256": authorization.manifest_sha256}) + "\n"
                    )
                ledger.write(json.dumps({"entry_id": authorization.entry_id}) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())

    def _read_entries(self, manifest_sha256: str) -> set[str]:
        if not self.path.exists():
            return set()
        raw = _read_owner_file(self.path)
        try:
            records = [json.loads(line) for line in raw.splitlines()]
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise BoundedCampaignError("bounded ledger is corrupt") from error
        if not records or records[0] != {"manifest_sha256": manifest_sha256}:
            raise BoundedCampaignError("bounded ledger manifest mismatch")
        entries: set[str] = set()
        for record in records[1:]:
            if not isinstance(record, dict) or set(record) != {"entry_id"}:
                raise BoundedCampaignError("bounded ledger is corrupt")
            entry_id = record["entry_id"]
            if not isinstance(entry_id, str) or not entry_id or entry_id in entries:
                raise BoundedCampaignError("bounded ledger is corrupt")
            entries.add(entry_id)
        return entries

    @staticmethod
    def _open_owner_file(path: Path, *, create: bool) -> int:
        if path.exists() or path.is_symlink():
            _read_owner_file(path)
        flags = os.O_RDWR | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        if create:
            flags |= os.O_CREAT
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as error:
            raise BoundedCampaignError("bounded ledger cannot be opened safely") from error
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            os.close(descriptor)
            raise BoundedCampaignError("bounded ledger ownership or permissions mismatch")
        return descriptor


@dataclass(frozen=True, slots=True)
class BoundedAttemptGuard:
    authorization: BoundedAuthorization
    ledger: DurableReservationLedger

    async def __call__(self, _request: Any, prepared: dict[str, Any]) -> None:
        self.ledger.reserve(self.authorization)
        # LiteLLM client retry knobs only. Never set prepared["retry"]: OpenAI-
        # compatible bases (codex-lb) reject `retry` as an unsupported body field
        # (400 Unsupported parameter: retry), which burns bounded reservations.
        prepared["num_retries"] = 0
        prepared["max_retries"] = 0
        prepared.pop("retry", None)
        if self.authorization.provider == "openai":
            extra_headers = dict(prepared.get("extra_headers") or {})
            extra_headers["X-OpenCode-Bounded-Attempt"] = "1"
            prepared["extra_headers"] = extra_headers
