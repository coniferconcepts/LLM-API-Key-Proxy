"""Retry-after duration parsers extracted from rotator_library.error_handler."""

from __future__ import annotations

import re
from typing import Optional


def _parse_duration_string(duration_str: str) -> Optional[int]:
    """
    Parse duration strings in various formats to total seconds.

    Handles:
    - Milliseconds: '290.979975ms' -> 1 second (rounds up for sub-second values)
    - Compound durations: '156h14m36.752463453s', '2h30m', '45m30s'
    - Simple durations: '562476.752463453s', '3600s', '60m', '2h'
    - Plain seconds (no unit): '562476'

    Args:
        duration_str: Duration string to parse

    Returns:
        Total seconds as integer, or None if parsing fails.
        For sub-second values, returns at least 1 to avoid retry floods.
    """
    if not duration_str:
        return None

    total_seconds = 0.0
    remaining = duration_str.strip().lower()

    # Try parsing as plain number first (no units)
    try:
        return int(float(remaining))
    except ValueError:
        pass

    # Handle pure milliseconds format: "290.979975ms"
    # MUST check this BEFORE checking 'm' for minutes to avoid misinterpreting 'ms'
    ms_match = re.match(r"^([\d.]+)ms$", remaining)
    if ms_match:
        ms_value = float(ms_match.group(1))
        seconds = ms_value / 1000.0
        # Round up to at least 1 second to avoid immediate retry floods
        return max(1, int(seconds)) if seconds > 0 else 0

    # Parse hours component
    hour_match = re.match(r"(\d+)h", remaining)
    if hour_match:
        total_seconds += int(hour_match.group(1)) * 3600
        remaining = remaining[hour_match.end() :]

    # Parse minutes component - use negative lookahead to avoid matching 'ms'
    min_match = re.match(r"(\d+)m(?!s)", remaining)
    if min_match:
        total_seconds += int(min_match.group(1)) * 60
        remaining = remaining[min_match.end() :]

    # Parse seconds component (including decimals like 36.752463453s)
    sec_match = re.match(r"([\d.]+)s", remaining)
    if sec_match:
        total_seconds += float(sec_match.group(1))

    # For sub-second values, round up to at least 1
    if total_seconds > 0:
        return max(1, int(total_seconds))
    return None


def extract_retry_after_from_body(error_body: Optional[str]) -> Optional[int]:
    """
    Extract the retry-after time from an API error response body.

    Handles various error formats including:
    - Gemini CLI: "Your quota will reset after 39s."
    - Antigravity: "quota will reset after 156h14m36s"
    - Generic: "quota will reset after 120s", "retry after 60s"

    Args:
        error_body: The raw error response body

    Returns:
        The retry time in seconds, or None if not found
    """
    if not error_body:
        return None

    # Pattern to match various "reset after" formats - capture the full duration string
    patterns = [
        r"quota will reset after\s*([\dhmso.]+)",  # Matches compound: 156h14m36s or 120s
        r"reset after\s*([\dhmso.]+)",
        r"retry after\s*([\dhmso.]+)",
        r"try again in\s*(\d+)\s*seconds?",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_body, re.IGNORECASE)
        if match:
            duration_str = match.group(1)
            result = _parse_duration_string(duration_str)
            if result is not None:
                return result

    return None
