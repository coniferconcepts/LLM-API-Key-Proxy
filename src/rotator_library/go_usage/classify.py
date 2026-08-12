# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""HTTP classification for GO /usage. No CLI/SystemExit."""

from __future__ import annotations

from typing import Any


def classify_http_status(code: int, content_type: str = "") -> dict[str, Any]:
    """Map status (+ optional content-type) to category and exit_code.

    Non-JSON 403 is edge_blocked (Cloudflare 1010), not entitlement.
    """
    ctype = content_type.lower()
    if code == 403 and "json" not in ctype:
        return {"category": "edge_blocked", "exit_code": 1, "status": code}
    if code in (401, 403, 404):
        return {"category": "http_error", "exit_code": 1, "status": code}
    if code in (408, 429) or 500 <= code <= 599:
        return {"category": "http_transient", "exit_code": 3, "status": code}
    if 300 <= code < 400:
        return {"category": "redirect_refused", "exit_code": 1, "status": code}
    if code == 200:
        return {"category": "ok", "exit_code": 0, "status": code}
    return {"category": "http_error", "exit_code": 1, "status": code}
