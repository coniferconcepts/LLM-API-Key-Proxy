# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Background refresh: one GET per unique key, apply UsageManager snapshot."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Iterable, TYPE_CHECKING

import httpx

from .client import DEFAULT_BASE, DEFAULT_TIMEOUT, fetch_usage

if TYPE_CHECKING:
    from ..usage_manager import UsageManager

lib_logger = logging.getLogger("rotator_library")

_MIN_INTERVAL = 60
_DEFAULT_INTERVAL = 300
# Dedup polls of the same key value across opencode_go + opencode_go_messages.
_last_fetch_mono: dict[str, float] = {}
_DEDUP_SECONDS = 30.0


def go_usage_job_config() -> dict[str, Any] | None:
    """None unless GO_QUOTA_REFRESH_ENABLED=1. Enable/disable requires restart."""
    if os.getenv("GO_QUOTA_REFRESH_ENABLED") != "1":
        return None
    raw = os.getenv("GO_QUOTA_REFRESH_INTERVAL", str(_DEFAULT_INTERVAL))
    try:
        interval = int(raw)
    except ValueError:
        interval = _DEFAULT_INTERVAL
    interval = max(_MIN_INTERVAL, interval)
    return {
        "interval": interval,
        "name": "opencode_go_quota_refresh",
        "run_on_start": True,
    }


def _fingerprint(credential: str) -> str:
    return hashlib.sha256(credential.encode("utf-8")).hexdigest()[:16]


def _unique_credentials(credentials: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for cred in credentials:
        if not cred or cred in seen:
            continue
        seen.add(cred)
        unique.append(cred)
    return unique


async def run_go_usage_refresh(
    usage_manager: UsageManager,
    credentials: list[str],
    *,
    base: str | None = None,
) -> None:
    """Fetch usage for each unique credential and apply go_usage_snapshot."""
    api_base = (base or os.getenv("OPENCODE_GO_API_BASE") or DEFAULT_BASE).rstrip("/")
    unique = _unique_credentials(credentials)
    now_mono = time.monotonic()
    to_fetch: list[str] = []
    for cred in unique:
        fp = _fingerprint(cred)
        last = _last_fetch_mono.get(fp)
        if last is not None and (now_mono - last) < _DEDUP_SECONDS:
            continue
        to_fetch.append(cred)

    if not to_fetch:
        return

    async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
        for cred in to_fetch:
            fp = _fingerprint(cred)
            _last_fetch_mono[fp] = time.monotonic()
            result = await fetch_usage(client, base=api_base, api_key=cred, timeout=DEFAULT_TIMEOUT)
            if not result["ok"]:
                lib_logger.info(
                    "GO usage refresh skipped apply category=%s status=%s fp=%s",
                    result["category"],
                    result.get("status"),
                    fp,
                )
                continue
            await usage_manager.update_go_quota(cred, result["snapshot"])
