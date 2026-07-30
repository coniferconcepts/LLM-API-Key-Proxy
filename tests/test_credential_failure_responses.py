from __future__ import annotations

import json
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proxy_app.safe_errors import (  # noqa: E402
    credential_failure_response,
    terminal_completion_error_response,
)


def test_credential_failure_response_uses_classified_status_and_body() -> None:
    # Given the canonical finite proxy-busy category.
    category = "proxy_busy"

    # When the category crosses the HTTP response boundary.
    response = credential_failure_response(category)

    # Then the response preserves the machine-readable classification.
    assert response.status_code == 503
    assert json.loads(response.body) == {
        "error": {
            "type": "proxy_busy",
            "code": "acquisition_timeout",
            "status": 503,
            "message": "No upstream credential is currently available. Retry later.",
        }
    }


def test_terminal_completion_error_response_rejects_noncanonical_error() -> None:
    # Given an upstream-shaped mapping that is not the canonical terminal exhaustion result.
    response = {"error": {"type": "provider_error", "code": "upstream_failure", "status": 503}}

    # When the completion result is inspected at the HTTP boundary.
    terminal = terminal_completion_error_response(response)

    # Then it remains a normal completion result rather than being promoted to an HTTP error.
    assert terminal is None
