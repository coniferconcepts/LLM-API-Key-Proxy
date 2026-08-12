# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Async GO /usage fetch. urllib CLI stays separate. No SystemExit."""

from __future__ import annotations

from typing import Any, Final

import httpx

from .classify import classify_http_status
from .schema import GoUsageError, normalize_usage

DEFAULT_BASE: Final = "https://opencode.ai/zen/go/v1"
USER_AGENT: Final = "opencode-router-go-usage-probe/1"
DEFAULT_TIMEOUT: Final = 15.0
DEFAULT_MAX_BYTES: Final = 64 * 1024


def _fail(category: str, exit_code: int, message: str, status: int | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "category": category,
        "exit_code": exit_code,
        "message": message,
        "status": status,
        "snapshot": None,
    }


async def fetch_usage(
    client: httpx.AsyncClient,
    *,
    base: str,
    api_key: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """One GET. Returns {ok, category, exit_code, snapshot?, status?}."""
    url = f"{base.rstrip('/')}/usage"
    try:
        response = await client.get(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": USER_AGENT,
            },
            timeout=timeout,
            follow_redirects=False,
        )
    except httpx.TimeoutException:
        return _fail("request_timeout", 3, "GO usage endpoint exceeded total deadline")
    except httpx.HTTPError:
        return _fail("transport_error", 3, "GO usage endpoint is unreachable")

    content_type = response.headers.get("Content-Type", "")
    payload = response.content
    if response.status_code != 200:
        classified = classify_http_status(response.status_code, content_type)
        return _fail(
            classified["category"],
            classified["exit_code"],
            f"GO usage endpoint returned HTTP {response.status_code}",
            status=response.status_code,
        )
    if len(payload) > max_bytes:
        return _fail("response_too_large", 1, "GO usage response exceeds size limit")
    if content_type and "json" not in content_type.lower():
        return _fail("schema_unknown", 1, "GO usage response is not JSON")
    try:
        decoded = response.json()
    except ValueError:
        return _fail("schema_unknown", 1, "GO usage response is invalid JSON")
    try:
        snapshot = normalize_usage(decoded)
    except GoUsageError:
        return _fail("schema_unknown", 1, "GO usage response failed schema validation")
    return {
        "ok": True,
        "category": "ok",
        "exit_code": 0,
        "message": "",
        "status": 200,
        "snapshot": snapshot,
    }
