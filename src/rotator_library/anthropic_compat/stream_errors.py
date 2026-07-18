from __future__ import annotations

import json


def safe_stream_error_events() -> tuple[str, ...]:
    error = {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}
    return (f"event: error\ndata: {json.dumps(error)}\n\n",)
