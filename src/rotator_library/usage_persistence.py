from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
from typing import TypeAlias, TypedDict

from typing_extensions import assert_never


class _ModelStats(TypedDict, total=False):
    window_start_ts: float
    quota_reset_ts: float
    window_started: str | None
    quota_resets: str | None


class _CredentialUsage(TypedDict, total=False):
    models: dict[str, _ModelStats]


_PersistenceScalar: TypeAlias = str | int | float | bool | None
_PersistenceValue: TypeAlias = (
    _PersistenceScalar | list["_PersistenceValue"] | set[str] | dict[str, "_PersistenceValue"]
)


def _credential_fingerprint(credential: str) -> str:
    """Return a stable non-secret identifier for persisted usage state.

    Sixteen SHA-256 hex characters provide a 64-bit identifier. That is not a
    secret, but it is sufficient to avoid practical collisions for local usage
    accounting while keeping raw provider credentials out of persisted state.
    """
    digest = hashlib.sha256(credential.encode("utf-8")).hexdigest()[:16]
    return f"credential_sha256:{digest}"


def _format_timestamp_local(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    try:
        local_datetime = datetime.fromtimestamp(timestamp).astimezone()
        return local_datetime.strftime("%Y-%m-%d %H:%M:%S %z")
    except (OSError, ValueError, OverflowError):
        return None


def add_readable_timestamps(data: dict[str, _CredentialUsage]) -> dict[str, _CredentialUsage]:
    for key_data in data.values():
        for model_stats in key_data.get("models", {}).values():
            if not isinstance(model_stats, dict):
                continue
            window_start = model_stats.get("window_start_ts")
            if window_start:
                model_stats["window_started"] = _format_timestamp_local(window_start)
            else:
                model_stats.pop("window_started", None)

            quota_reset = model_stats.get("quota_reset_ts")
            if quota_reset:
                model_stats["quota_resets"] = _format_timestamp_local(quota_reset)
            else:
                model_stats.pop("quota_resets", None)
    return data


def _safe_cycle_state_for_persistence(value: _PersistenceValue) -> _PersistenceValue:
    match value:
        case dict() as mapping:
            return {key: _safe_cycle_state_for_persistence(item) for key, item in mapping.items()}
        case list() as items:
            return [_credential_fingerprint(str(item)) for item in items]
        case set() as items:
            return [_credential_fingerprint(str(item)) for item in sorted(items)]
        case str() | int() | float() | bool() | None:
            return value
        case unreachable:
            assert_never(unreachable)


def safe_usage_data_for_persistence(
    data: Mapping[str, _PersistenceValue],
) -> dict[str, _PersistenceValue]:
    safe: dict[str, _PersistenceValue] = {}
    for credential, credential_data in data.items():
        if credential == "__fair_cycle__":
            safe[credential] = _safe_cycle_state_for_persistence(credential_data)
        else:
            safe[_credential_fingerprint(credential)] = credential_data
    return safe
