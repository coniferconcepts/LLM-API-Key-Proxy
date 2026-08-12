from __future__ import annotations

import datetime as dt

from rotator_library.go_usage.gate import evaluate_go_usage

NOW = dt.datetime(2026, 8, 12, 16, 0, tzinfo=dt.timezone.utc)


def _snap(**windows: dict) -> dict:
    base = {
        "rolling": {"status": "ok", "percent": 0, "resetsAt": "2026-08-12T21:00:00Z"},
        "weekly": {"status": "ok", "percent": 0, "resetsAt": "2026-08-17T00:00:00Z"},
        "monthly": {"status": "ok", "percent": 0, "resetsAt": "2026-09-11T12:00:00Z"},
    }
    base.update(windows)
    return {"usage": base}


def test_all_ok_clears() -> None:
    decision = evaluate_go_usage(_snap(), now=NOW)
    assert decision["action"] == "clear"
    assert decision["cooldown_until"] is None


def test_ok_at_100_percent_does_not_block() -> None:
    decision = evaluate_go_usage(
        _snap(monthly={"status": "ok", "percent": 100, "resetsAt": "2026-09-11T12:00:00Z"}),
        now=NOW,
    )
    assert decision["action"] == "clear"


def test_rate_limited_arms_latest_future_reset() -> None:
    decision = evaluate_go_usage(
        _snap(
            weekly={"status": "rate-limited", "percent": 100, "resetsAt": "2026-08-17T00:00:00Z"},
            monthly={"status": "rate-limited", "percent": 100, "resetsAt": "2026-08-13T15:19:00Z"},
        ),
        now=NOW,
    )
    assert decision["action"] == "arm"
    assert (
        decision["cooldown_until"]
        == dt.datetime(2026, 8, 17, 0, 0, tzinfo=dt.timezone.utc).timestamp()
    )


def test_past_reset_is_treated_as_ok() -> None:
    decision = evaluate_go_usage(
        _snap(
            monthly={"status": "rate-limited", "percent": 100, "resetsAt": "2026-08-11T00:00:00Z"}
        ),
        now=NOW,
    )
    assert decision["action"] == "clear"


def test_unknown_status_retains() -> None:
    decision = evaluate_go_usage(
        _snap(rolling={"status": "weird", "percent": 1, "resetsAt": "2026-08-12T21:00:00Z"}),
        now=NOW,
    )
    assert decision["action"] == "retain"
