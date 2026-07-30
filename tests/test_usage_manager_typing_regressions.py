from __future__ import annotations

from pathlib import Path

import pytest

from rotator_library.usage_manager import UsageManager
from rotator_library.usage_selection import select_weighted_random


def test_model_reset_uses_window_when_quota_reset_is_none(tmp_path: Path) -> None:
    manager = UsageManager(tmp_path / "usage.json")

    assert manager._should_model_reset(  # noqa: SLF001
        {"quota_reset_ts": None, "window_start_ts": 100.0},
        window_seconds=20,
        now_ts=120.0,
    )


def test_unknown_provider_state_uses_existing_defaults(tmp_path: Path) -> None:
    manager = UsageManager(tmp_path / "usage.json", provider_plugins={})

    assert manager._get_usage_reset_config("unrecognized-credential") is None  # noqa: SLF001
    assert (
        manager._get_model_quota_group("unrecognized-credential", "model") is None
    )  # noqa: SLF001
    assert manager._get_model_usage_weight("unrecognized-credential", "model") == 1  # noqa: SLF001
    assert manager._normalize_model("unrecognized-credential", "model") == "model"  # noqa: SLF001


@pytest.mark.parametrize(
    ("credential", "expected"),
    [
        ("env://antigravity/1", "antigravity"),
        ("oauth_creds/gemini_cli_oauth_1.json", "gemini_cli"),
        ("sk-nano-example", "nanogpt"),
    ],
)
def test_provider_resolution_preserves_supported_credential_formats(
    tmp_path: Path,
    credential: str,
    expected: str,
) -> None:
    manager = UsageManager(tmp_path / "usage.json")

    assert manager._get_provider_from_credential(credential) == expected  # noqa: SLF001


def test_provider_resolution_falls_back_to_stored_model_state(tmp_path: Path) -> None:
    manager = UsageManager(tmp_path / "usage.json")
    manager._usage_data = {"opaque": {"models": {"firmware/model": {}}}}  # noqa: SLF001

    assert manager._get_provider_from_credential("opaque") == "firmware"  # noqa: SLF001


def test_single_weighted_candidate_is_selected_without_randomization() -> None:
    assert select_weighted_random([("credential", 9)], tolerance=4.0) == "credential"


@pytest.mark.asyncio
async def test_exhausted_quota_without_reset_timestamp_skips_cooldown(
    tmp_path: Path,
) -> None:
    manager = UsageManager(tmp_path / "usage.json")

    result = await manager.update_quota_baseline(
        credential="unrecognized-credential",
        model="provider/model",
        remaining_fraction=0.0,
        max_requests=100,
        reset_timestamp=None,
    )

    assert result is None
    snapshot = await manager._get_usage_data_snapshot()  # noqa: SLF001
    assert snapshot["unrecognized-credential"]["model_cooldowns"] == {}
