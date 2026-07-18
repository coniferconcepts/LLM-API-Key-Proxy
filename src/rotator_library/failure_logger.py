# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .secure_logging import OwnerOnlyRotatingFileHandler
from .utils.paths import get_logs_dir, secure_logs_dir

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

# Maximum size of the failure log file before rotation (in bytes)
# Default: 5 MB
FAILURE_LOG_MAX_SIZE: int = 5 * 1024 * 1024

# Number of backup log files to keep
FAILURE_LOG_BACKUP_COUNT: int = 2

SAFE_ERROR_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logs."""

    def format(self, record):
        # The message is already a dict, so we just format it as a JSON string
        return json.dumps(record.msg)


# Module-level state for lazy initialization
_failure_logger: Optional[logging.Logger] = None
_configured_logs_dir: Optional[Path] = None


def configure_failure_logger(logs_dir: Optional[Union[Path, str]] = None) -> None:
    """
    Configure the failure logger to use a specific logs directory.

    Call this before first use if you want to override the default location.
    If not called, the logger will use get_logs_dir() on first use.

    Args:
        logs_dir: Path to the logs directory. If None, uses get_logs_dir().
    """
    global _configured_logs_dir, _failure_logger
    if _failure_logger is not None:
        _close_handlers(_failure_logger)
    _configured_logs_dir = Path(logs_dir) if logs_dir else None
    _failure_logger = None


def _close_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def _setup_failure_logger(logs_dir: Path) -> logging.Logger:
    """
    Sets up a dedicated JSON logger for writing allowlisted failure metadata.

    Args:
        logs_dir: Path to the logs directory.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("failure_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    _close_handlers(logger)

    try:
        secure_logs_dir(logs_dir)

        handler = OwnerOnlyRotatingFileHandler(
            logs_dir / "failures.log",
            maxBytes=FAILURE_LOG_MAX_SIZE,
            backupCount=FAILURE_LOG_BACKUP_COUNT,
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    except (OSError, PermissionError, IOError) as e:
        logging.warning(f"Cannot create failure log file handler: {e}")
        # Add NullHandler to prevent "no handlers" warning
        logger.addHandler(logging.NullHandler())

    return logger


def get_failure_logger() -> logging.Logger:
    """
    Get the failure logger, initializing it lazily if needed.

    Returns:
        The configured failure logger.
    """
    global _failure_logger, _configured_logs_dir

    if _failure_logger is None:
        logs_dir = _configured_logs_dir if _configured_logs_dir else get_logs_dir()
        _failure_logger = _setup_failure_logger(logs_dir)

    return _failure_logger


# Get the main library logger for concise, propagated messages
main_lib_logger = logging.getLogger("rotator_library")


def _safe_error_type(error: Exception) -> str:
    error_type = type(error).__name__
    return error_type if SAFE_ERROR_TYPE.fullmatch(error_type) else "UnknownError"


def _safe_status_code(error: Exception) -> int | None:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and 400 <= status_code <= 599:
        return status_code
    return None


def log_failure(
    api_key: str,
    model: str,
    attempt: int,
    error: Exception,
    request_headers: dict,
    raw_response_text: str | None = None,
):
    """
    Persist safe failure metadata and emit a concise summary.

    Sensitive compatibility inputs are accepted for existing callers but ignored. Only
    the allowlisted schema fields constructed in this function can reach the log file.

    Args:
        api_key: Ignored compatibility input; never persisted
        model: Ignored compatibility input; never persisted
        attempt: The attempt number (1-based)
        error: Used only to derive an allowlisted type name and status code
        request_headers: Ignored compatibility input; never persisted
        raw_response_text: Ignored compatibility input; never persisted
    """
    detailed_log_data = {
        "schema_version": "failure_log.v2",
        "timestamp": datetime.utcnow().isoformat(),
        "attempt_number": attempt,
        "error_type": _safe_error_type(error),
        "status_code": _safe_status_code(error),
    }

    summary_message = f"API call failed. Error type: {_safe_error_type(error)}."

    # Log to failure logger with resilience - if it fails, just continue
    try:
        get_failure_logger().error(detailed_log_data)
    except (OSError, IOError) as e:
        # Log file write failed - log to console instead
        logging.warning(f"Failed to write to failures.log: {e}")

    # Console log always succeeds
    main_lib_logger.error(summary_message)
