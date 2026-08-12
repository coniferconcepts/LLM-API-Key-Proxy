from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from rotator_library.usage_manager import UsageManager

NOW = dt.datetime(2026, 8, 12, 16, 0, tzinfo=dt.timezone.utc)
NOW_TS = NOW.timestamp()


def _snap(monthly_status: str = "ok", monthly_percent: float = 0.0) -> dict:
    return {
        "usage": {
            "rolling": {"status": "ok", "percent": 0, "resetsAt": "2026-08-12T21:00:00Z"},
            "weekly": {"status": "ok", "percent": 0, "resetsAt": "2026-08-17T00:00:00Z"},
            "monthly": {
                "status": monthly_status,
                "percent": monthly_percent,
                "resetsAt": "2026-08-13T15:19:00Z",
            },
        }
    }


@pytest.mark.asyncio
async def test_rate_limited_sets_go_cooldown_not_count_fields(tmp_path: Path) -> None:
    manager = UsageManager(tmp_path / "usage.json")
    cred = "sk-go-test"
    result = await manager.update_go_quota(cred, _snap("rate-limited", 100), now_ts=NOW_TS)
    assert result["action"] == "arm"
    data = (await manager._get_usage_data_snapshot())[cred]  # noqa: SLF001
    assert (
        data["go_usage_cooldown_until"]
        == dt.datetime(2026, 8, 13, 15, 19, tzinfo=dt.timezone.utc).timestamp()
    )
    assert data.get("quota_max_requests") is None
    assert data.get("quota_reset_ts") is None
    assert "request_count" not in data or data.get("request_count") in (0, None)


@pytest.mark.asyncio
async def test_ok_clears_only_go_cooldown(tmp_path: Path) -> None:
    manager = UsageManager(tmp_path / "usage.json")
    cred = "sk-go-test"
    await manager.update_go_quota(cred, _snap("rate-limited", 100), now_ts=NOW_TS)
    data = (await manager._get_usage_data_snapshot())[cred]  # noqa: SLF001
    data["key_cooldown_until"] = NOW_TS + 3600
    data["request_count"] = 7
    await manager.update_go_quota(cred, _snap("ok", 0), now_ts=NOW_TS)
    data = (await manager._get_usage_data_snapshot())[cred]  # noqa: SLF001
    assert data["go_usage_cooldown_until"] is None
    assert data["key_cooldown_until"] == NOW_TS + 3600
    assert data["request_count"] == 7


@pytest.mark.asyncio
async def test_unknown_status_retains_go_cooldown(tmp_path: Path) -> None:
    manager = UsageManager(tmp_path / "usage.json")
    cred = "sk-go-test"
    await manager.update_go_quota(cred, _snap("rate-limited", 100), now_ts=NOW_TS)
    armed = (await manager._get_usage_data_snapshot())[cred][
        "go_usage_cooldown_until"
    ]  # noqa: SLF001
    weird = _snap("ok", 0)
    weird["usage"]["rolling"] = {
        "status": "mystery",
        "percent": 1,
        "resetsAt": "2026-08-12T21:00:00Z",
    }
    await manager.update_go_quota(cred, weird, now_ts=NOW_TS)
    data = (await manager._get_usage_data_snapshot())[cred]  # noqa: SLF001
    assert data["go_usage_cooldown_until"] == armed


@pytest.mark.asyncio
async def test_eligibility_skips_go_cooldown_only(tmp_path: Path) -> None:
    manager = UsageManager(tmp_path / "usage.json")
    blocked = "sk-blocked"
    free = "sk-free"
    await manager.update_go_quota(blocked, _snap("rate-limited", 100), now_ts=NOW_TS)
    available = await manager.get_available_credentials_for_model(
        [blocked, free], "opencode_go/glm-5.2"
    )
    assert available == [free]
