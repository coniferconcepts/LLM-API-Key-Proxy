from __future__ import annotations

from pathlib import Path
import sys

from litellm.exceptions import APIError, RateLimitError

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.error_handler import classify_error  # noqa: E402


def test_litellm_apierror_numeric_403_is_forbidden() -> None:
    error = APIError(
        status_code=403,
        message="upstream rejected",
        llm_provider="openai",
        model="glm-5.3",
    )
    classified = classify_error(error)
    assert classified.error_type == "forbidden"
    assert classified.status_code == 403


def test_litellm_rate_limit_error_stays_rate_limit() -> None:
    error = RateLimitError(
        message="slow down",
        llm_provider="openai",
        model="glm-5.3",
    )
    classified = classify_error(error)
    assert classified.error_type == "rate_limit"
