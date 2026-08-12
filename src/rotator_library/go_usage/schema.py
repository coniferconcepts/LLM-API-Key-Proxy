# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Strict GO /usage wire contract. Raises GoUsageError; never SystemExit."""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Final, Mapping

WINDOWS: Final = ("rolling", "weekly", "monthly")


class GoUsageError(ValueError):
    """Malformed GO usage payload."""


def _validate_percent(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GoUsageError("usage window percent must be a number")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= 100:
        raise GoUsageError("usage window percent must be finite and within 0..100")
    return numeric


def parse_resets_at(value: object) -> _dt.datetime:
    if not isinstance(value, str) or not value:
        raise GoUsageError("usage window resetsAt must be a non-empty string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = _dt.datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise GoUsageError("usage window resetsAt must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise GoUsageError("usage window resetsAt must be timezone-aware")
    return parsed


def _normalize_window(name: str, window: object) -> dict[str, Any]:
    if not isinstance(window, dict):
        raise GoUsageError(f"usage window {name} must be an object")
    status = window.get("status")
    if not isinstance(status, str) or not status:
        raise GoUsageError(f"usage window {name} status must be a non-empty string")
    percent = _validate_percent(window.get("percent"))
    resets_at = window.get("resetsAt")
    parse_resets_at(resets_at)
    return {"status": status, "percent": percent, "resetsAt": resets_at}


def normalize_usage(payload: object) -> dict[str, Any]:
    """Parse usage.{rolling,weekly,monthly}.{status,percent,resetsAt}. Extra fields ok."""
    if not isinstance(payload, Mapping):
        raise GoUsageError("usage response must be a JSON object")
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        raise GoUsageError("usage response must contain a usage object")
    normalized: dict[str, Any] = {}
    for name in WINDOWS:
        if name not in usage:
            raise GoUsageError(f"usage response missing required window {name}")
        normalized[name] = _normalize_window(name, usage[name])
    return {"usage": normalized}
