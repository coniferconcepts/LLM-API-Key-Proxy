from __future__ import annotations

import pytest

from rotator_library.go_usage.classify import classify_http_status
from rotator_library.go_usage.schema import GoUsageError, normalize_usage


def _valid() -> dict:
    return {
        "usage": {
            "rolling": {"status": "ok", "percent": 10, "resetsAt": "2026-08-12T20:00:00Z"},
            "weekly": {"status": "ok", "percent": 40, "resetsAt": "2026-08-12T20:00:00Z"},
            "monthly": {"status": "ok", "percent": 70, "resetsAt": "2026-08-12T20:00:00Z"},
        }
    }


def test_normalize_valid_and_extra_fields() -> None:
    payload = _valid()
    payload["extra"] = 1
    payload["usage"]["rolling"]["newField"] = True
    out = normalize_usage(payload)
    assert set(out["usage"]) == {"rolling", "weekly", "monthly"}
    assert out["usage"]["rolling"]["percent"] == 10


def test_normalize_rejects_missing_window() -> None:
    payload = _valid()
    del payload["usage"]["weekly"]
    with pytest.raises(GoUsageError):
        normalize_usage(payload)


def test_normalize_rejects_percent_bool_and_range() -> None:
    payload = _valid()
    payload["usage"]["weekly"]["percent"] = True
    with pytest.raises(GoUsageError):
        normalize_usage(payload)
    payload["usage"]["weekly"]["percent"] = 101
    with pytest.raises(GoUsageError):
        normalize_usage(payload)


def test_normalize_rejects_naive_resets_at() -> None:
    payload = _valid()
    payload["usage"]["weekly"]["resetsAt"] = "2026-08-12T20:00:00"
    with pytest.raises(GoUsageError):
        normalize_usage(payload)


def test_classify_edge_blocked_vs_json_403() -> None:
    assert classify_http_status(403, "text/plain")["category"] == "edge_blocked"
    assert classify_http_status(403, "application/json")["category"] == "http_error"
    assert classify_http_status(429)["exit_code"] == 3
    assert classify_http_status(200)["category"] == "ok"
