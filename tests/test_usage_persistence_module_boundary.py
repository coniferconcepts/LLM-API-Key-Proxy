from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.usage_manager import _credential_fingerprint  # noqa: E402


def test_usage_persistence_has_dedicated_module_boundary() -> None:
    # Given the credential fingerprint compatibility export.
    fingerprint = _credential_fingerprint

    # When its implementation owner is inspected.
    owner = fingerprint.__module__

    # Then persistence projection remains outside the oversized usage manager.
    assert owner == "rotator_library.usage_persistence"
