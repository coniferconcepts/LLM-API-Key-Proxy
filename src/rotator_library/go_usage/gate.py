# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Status-authoritative GO usage gating. Never gates on percent >= 100 alone."""

from __future__ import annotations

import datetime as _dt
from typing import Any, Mapping

from .schema import WINDOWS, parse_resets_at


def evaluate_go_usage(
    snapshot: Mapping[str, Any],
    *,
    now: _dt.datetime | None = None,
) -> dict[str, Any]:
    """Decide clear / arm / retain from a normalized usage snapshot.

    - rate-limited + future resetsAt → arm until the latest such reset
    - all windows ok (or past-reset) → clear
    - unknown status → retain prior cooldown
    """
    clock = now or _dt.datetime.now(tz=_dt.timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=_dt.timezone.utc)

    usage = snapshot.get("usage")
    if not isinstance(usage, Mapping):
        return {"action": "retain", "cooldown_until": None, "reason": "malformed"}

    latest_limited: _dt.datetime | None = None
    saw_unknown = False
    for name in WINDOWS:
        window = usage.get(name)
        if not isinstance(window, Mapping):
            saw_unknown = True
            continue
        status = window.get("status")
        if not isinstance(status, str) or not status:
            saw_unknown = True
            continue
        try:
            resets = parse_resets_at(window.get("resetsAt"))
        except ValueError:
            saw_unknown = True
            continue
        if status == "ok":
            continue
        if status == "rate-limited":
            if resets > clock:
                if latest_limited is None or resets > latest_limited:
                    latest_limited = resets
            continue
        saw_unknown = True

    if saw_unknown:
        return {"action": "retain", "cooldown_until": None, "reason": "unknown_status"}
    if latest_limited is not None:
        return {
            "action": "arm",
            "cooldown_until": latest_limited.timestamp(),
            "reason": "rate-limited",
        }
    return {"action": "clear", "cooldown_until": None, "reason": "ok"}
