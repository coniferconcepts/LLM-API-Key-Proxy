from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SECRET_MESSAGE = "UPSTREAM-PAYLOAD-SENTINEL"
SECRET_CREDENTIAL = "/private/oauth/STABLE-OAUTH-FILENAME-SENTINEL.json"


def anthropic_body(*, stream: bool = False) -> dict[str, Any]:
    return {
        "model": "xai_oauth/grok-4.5",
        "messages": [{"role": "user", "content": "safe boundary"}],
        "max_tokens": 16,
        "stream": stream,
    }
