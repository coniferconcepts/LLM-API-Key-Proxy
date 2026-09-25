"""Copy request data for logging while limiting recorded HTTP headers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_ALLOWED_HEADERS = frozenset({"content-type", "accept", "user-agent", "x-request-id"})
_DENIED_NAME_PARTS = ("authorization", "auth", "cookie", "credential", "token", "secret", "key")
_HEADER_FIELDS = frozenset({"headers", "extra_headers", "request_headers"})


def filter_log_headers(headers: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only approved HTTP headers in a new dictionary."""
    return {
        name: value
        for name, value in headers.items()
        if isinstance(name, str)
        and isinstance(value, str)
        and name.casefold() in _ALLOWED_HEADERS
        and not any(part in name.casefold() for part in _DENIED_NAME_PARTS)
    }


def redact_request_log_data(value: Any) -> Any:
    """Copy nested request data, filtering header fields at every depth."""
    if isinstance(value, Mapping):
        copied = {}
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in _HEADER_FIELDS:
                copied[key] = filter_log_headers(item) if isinstance(item, Mapping) else {}
            else:
                copied[key] = redact_request_log_data(item)
        return copied
    if isinstance(value, (list, tuple)):
        return [redact_request_log_data(item) for item in value]
    if isinstance(value, BaseException):
        return {"error_type": type(value).__name__}
    return value
