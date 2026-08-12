# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenCode GO usage wire contract (schema + classify + gate). No CLI/SystemExit."""

from .classify import classify_http_status
from .gate import evaluate_go_usage
from .schema import WINDOWS, GoUsageError, normalize_usage

__all__ = [
    "WINDOWS",
    "GoUsageError",
    "classify_http_status",
    "evaluate_go_usage",
    "normalize_usage",
]
