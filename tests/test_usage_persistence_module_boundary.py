from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.usage_manager import project_quota_stats_credentials  # noqa: E402


def test_usage_persistence_has_dedicated_module_boundary() -> None:
    # Given the public quota-stats projection export.
    projection = project_quota_stats_credentials

    # When its implementation owner is inspected.
    owner = projection.__module__

    # Then persistence projection remains outside the oversized usage manager.
    assert owner == "rotator_library.usage_persistence"
