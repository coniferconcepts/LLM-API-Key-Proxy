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


def test_exhausted_response_includes_retry_after_when_provided() -> None:
    response = credential_failure_response(
        "proxy_all_credentials_exhausted",
        retry_after_seconds=42,
    )
    assert response.status_code == 503
    assert response.headers["retry-after"] == "42"
    body = json.loads(response.body)
    assert body["error"]["retry_after_seconds"] == 42
    assert body["error"]["code"] == "all_credentials_exhausted"


def test_busy_response_omits_retry_after_even_if_seconds_passed() -> None:
    from proxy_app.safe_errors import handle_credential_failure
    from rotator_library.error_handler import NoAvailableKeysError

    error = NoAvailableKeysError("busy", category="proxy_busy", soonest_end=9_999_999_999)
    response = handle_credential_failure(error, None)
    assert response.status_code == 503
    assert "retry-after" not in {k.lower() for k in response.headers.keys()}
    assert "retry_after_seconds" not in json.loads(response.body)["error"]


def test_handle_exhausted_emits_retry_after_from_soonest_end() -> None:
    import time

    from proxy_app.safe_errors import handle_credential_failure
    from rotator_library.error_handler import NoAvailableKeysError

    past = NoAvailableKeysError(
        "exhausted",
        category="proxy_all_credentials_exhausted",
        soonest_end=1_000.0,
    )
    past_response = handle_credential_failure(past, None)
    assert "retry-after" not in {k.lower() for k in past_response.headers.keys()}

    future = time.time() + 12.1
    error = NoAvailableKeysError(
        "exhausted",
        category="proxy_all_credentials_exhausted",
        soonest_end=future,
    )
    response = handle_credential_failure(error, None)
    assert response.status_code == 503
    retry_after = int(response.headers["retry-after"])
    assert 12 <= retry_after <= 14
    assert json.loads(response.body)["error"]["retry_after_seconds"] == retry_after


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
