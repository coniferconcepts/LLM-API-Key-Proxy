"""Reject client-controlled provider routing and credential overrides."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_TOP_LEVEL_BLOCKED_FIELDS = frozenset(
    {"api_base", "base_url", "user_config", "api_key", "model_list", "fallbacks"}
)
_EXTRA_BODY_BLOCKED_FIELDS = frozenset({"api_base", "base_url", "user_config", "api_key"})


def find_client_override(payload: Any) -> str | None:
    """Return the first forbidden client-controlled field, if present."""
    if not isinstance(payload, Mapping):
        return None

    for field in sorted(_TOP_LEVEL_BLOCKED_FIELDS):
        if field in payload:
            return field

    extra_body = payload.get("extra_body")
    if isinstance(extra_body, Mapping):
        for field in sorted(_EXTRA_BODY_BLOCKED_FIELDS):
            if field in extra_body:
                return f"extra_body.{field}"

    return None
