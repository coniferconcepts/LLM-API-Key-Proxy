from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.error_handler import classify_credential_failure  # noqa: E402


def test_credential_failure_classification_has_dedicated_module_boundary() -> None:
    # Given the compatibility export used by existing error-handler consumers.
    classifier = classify_credential_failure

    # When its implementation owner is inspected.
    owner = classifier.__module__

    # Then credential-failure policy remains outside the oversized error handler.
    assert owner == "rotator_library.credential_failures"
