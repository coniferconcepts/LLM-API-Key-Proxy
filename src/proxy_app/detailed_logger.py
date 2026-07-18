# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

# src/proxy_app/detailed_logger.py
"""
Raw I/O Logger for the Proxy Layer.

This logger captures the UNMODIFIED HTTP request and response at the proxy boundary.
It is disabled by default and should only be enabled for debugging the proxy itself.

Use this when you need to:
- Verify that requests/responses are not being corrupted
- Debug HTTP-level issues between the client and proxy
- Capture exact payloads as received/sent by the proxy

For normal request/response logging with provider correlation, use the
TransactionLogger in the rotator_library instead (enabled via --enable-request-logging).

Directory structure:
    logs/raw_io/{YYYYMMDD_HHMMSS}_{request_id}/
        request.json           # Unmodified incoming HTTP request
        streaming_chunks.jsonl # If streaming mode
        final_response.json    # Unmodified outgoing HTTP response
        metadata.json          # Summary metadata
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from rotator_library.secure_log_domain import OwnerOnlyLogDomain
from rotator_library.utils.paths import get_logs_dir


class RawIOLogger:
    """
    Logs raw HTTP request/response at the proxy boundary.

    This captures the EXACT data as received from and sent to the client,
    without any transformations. Useful for debugging the proxy itself.

    DISABLED by default. Enable with --enable-raw-logging flag.

    Uses fire-and-forget logging - if disk writes fail, logs are dropped (not buffered)
    to prevent memory issues, especially with streaming responses.
    """

    def __init__(self):
        """
        Initializes the logger for a single request, creating a unique directory
        to store all related log files.
        """
        self.start_time = time.time()
        self.request_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        request_name = f"{timestamp}_{self.request_id}"
        logs_dir = get_logs_dir()
        self.log_dir = logs_dir / "raw_io" / request_name
        self.streaming = False
        self._domain = OwnerOnlyLogDomain(
            root=logs_dir,
            components=("raw_io", request_name),
        )
        try:
            self._domain.ensure()
        except OSError:
            logging.warning("Raw I/O logging is unavailable because its directory is unsafe")
            self._dir_available = False
        else:
            self._dir_available = True

    def _write_json(self, filename: str, data: Dict[str, Any]):
        """Helper to write data to a JSON file in the log directory."""
        try:
            content = json.dumps(data, indent=4, ensure_ascii=False)
            self._domain.write_text(filename, content, append=False)
        except (OSError, TypeError, ValueError):
            self._dir_available = False
            logging.warning("Raw I/O artifact was dropped because its path or content is unsafe")
        else:
            self._dir_available = True

    def log_request(self, headers: Dict[str, Any], body: Dict[str, Any]):
        """Logs the raw incoming request details."""
        self.streaming = body.get("stream", False)
        request_data = {
            "request_id": self.request_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "headers": dict(headers),
            "body": body,
        }
        self._write_json("request.json", request_data)

    def log_stream_chunk(self, chunk: Dict[str, Any]):
        """Logs an individual chunk from a streaming response to a JSON Lines file."""
        try:
            log_entry = {"timestamp_utc": datetime.utcnow().isoformat(), "chunk": chunk}
            content = json.dumps(log_entry, ensure_ascii=False) + "\n"
            self._domain.write_text("streaming_chunks.jsonl", content, append=True)
        except (OSError, TypeError, ValueError):
            self._dir_available = False
            logging.warning("Raw I/O stream chunk was dropped because its path is unsafe")
        else:
            self._dir_available = True

    def log_final_response(
        self, status_code: int, headers: Optional[Dict[str, Any]], body: Dict[str, Any]
    ):
        """Logs the raw outgoing response."""
        end_time = time.time()
        duration_ms = (end_time - self.start_time) * 1000

        response_data = {
            "request_id": self.request_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "status_code": status_code,
            "duration_ms": round(duration_ms),
            "headers": dict(headers) if headers else None,
            "body": body,
        }
        self._write_json("final_response.json", response_data)
        self._log_metadata(response_data)

    def _extract_reasoning(self, response_body: Dict[str, Any]) -> Optional[str]:
        """Recursively searches for and extracts 'reasoning' fields from the response body."""
        reasoning = response_body.get("reasoning")
        if isinstance(reasoning, str):
            return reasoning

        choices = response_body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return None
        for field in ("reasoning", "reasoning_content"):
            reasoning = message.get(field)
            if isinstance(reasoning, str):
                return reasoning

        return None

    def _log_metadata(self, response_data: Dict[str, Any]):
        """Logs a summary of the transaction for quick analysis."""
        body_value = response_data.get("body")
        body = body_value if isinstance(body_value, dict) else {}
        usage_value = body.get("usage")
        usage = usage_value if isinstance(usage_value, dict) else {}
        model_value = body.get("model")
        model = model_value if isinstance(model_value, str) else "N/A"
        finish_reason = "N/A"
        choices = body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_value = choices[0].get("finish_reason")
            if isinstance(finish_value, str):
                finish_reason = finish_value

        metadata = {
            "request_id": self.request_id,
            "timestamp_utc": response_data["timestamp_utc"],
            "duration_ms": response_data["duration_ms"],
            "status_code": response_data["status_code"],
            "model": model,
            "streaming": self.streaming,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "finish_reason": finish_reason,
            "reasoning_found": False,
            "reasoning_content": None,
        }

        reasoning = self._extract_reasoning(body)
        if reasoning:
            metadata["reasoning_found"] = True
            metadata["reasoning_content"] = reasoning

        self._write_json("metadata.json", metadata)


# Backward compatibility alias
DetailedLogger = RawIOLogger
