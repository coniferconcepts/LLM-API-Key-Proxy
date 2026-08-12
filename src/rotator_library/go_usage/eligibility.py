# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Family-wide GO eligibility: unique keys + credential-wide blocks only."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "go_usage_eligibility.v1"
GO_PROVIDER_NAMES = ("opencode_go", "opencode_go_messages")


def unique_credentials(credentials: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for cred in credentials:
        if not cred or cred in seen:
            continue
        seen.add(cred)
        unique.append(cred)
    return unique


def collect_go_credentials(provider_keys: Mapping[str, Iterable[str]]) -> list[str]:
    collected: list[str] = []
    for name in GO_PROVIDER_NAMES:
        collected.extend(provider_keys.get(name) or [])
    return unique_credentials(collected)


def summarize_go_eligibility(
    key_records: Mapping[str, Mapping[str, Any]],
    credentials: Iterable[str],
    *,
    now: float,
) -> dict[str, Any]:
    """Count unique keys using only generic + GO-usage cooldowns.

    Model-specific cooldowns are intentionally ignored so one blocked model
    cannot mark the whole GO family ineligible.
    """
    unique = unique_credentials(credentials)
    eligible = 0
    soonest: float | None = None
    for cred in unique:
        data = key_records.get(cred) or {}
        key_cd = float(data.get("key_cooldown_until") or 0)
        go_cd = float(data.get("go_usage_cooldown_until") or 0)
        block = max(key_cd, go_cd)
        if block > now:
            if soonest is None or block < soonest:
                soonest = block
            continue
        eligible += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "unique_keys": len(unique),
        "eligible": eligible,
        "blocked": len(unique) - eligible,
        "soonest_reset_unix": int(soonest) if soonest is not None else None,
    }
