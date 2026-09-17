from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.cooldown_manager import remaining_budget_seconds  # noqa: E402


def test_remaining_budget_seconds_clamps_past_deadline() -> None:
    assert remaining_budget_seconds(100.0, now=1067.0) == 0.0
    assert remaining_budget_seconds(200.0, now=150.0) == 50.0
    assert remaining_budget_seconds(10.0, now=10.0) == 0.0
