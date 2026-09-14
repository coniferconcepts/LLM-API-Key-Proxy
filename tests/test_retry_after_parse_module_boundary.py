from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.error_handler import (  # noqa: E402
    _parse_duration_string,
    extract_retry_after_from_body,
)
from rotator_library.retry_after_parse import (  # noqa: E402
    _parse_duration_string as parse_duration,
    extract_retry_after_from_body as extract_retry,
)


def test_retry_after_parsers_remain_importable_from_error_handler() -> None:
    assert extract_retry_after_from_body is extract_retry
    assert _parse_duration_string is parse_duration
    assert extract_retry_after_from_body.__module__ == "rotator_library.retry_after_parse"


def test_parse_duration_string_handles_documented_formats() -> None:
    assert parse_duration("290.979975ms") == 1
    assert parse_duration("2h30m") == 9000
    assert parse_duration("3600s") == 3600
    assert parse_duration("562476") == 562476
    assert parse_duration("") is None


def test_extract_retry_after_from_body_handles_documented_formats() -> None:
    assert extract_retry("Your quota will reset after 39s.") == 39
    assert extract_retry("quota will reset after 156h14m36s") == 562476
    assert extract_retry("retry after 60s") == 60
    assert extract_retry(None) is None
