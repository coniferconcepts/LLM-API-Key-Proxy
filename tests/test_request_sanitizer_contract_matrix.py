"""Pin request sanitizer behaviour per provider prefix.

Extension rule: prefix added here only when `make smoke-ladder` records a live
4xx for that prefix with the extra present. Do not widen the Fireworks/GO
message-key strip from this matrix alone.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotator_library.request_sanitizer import sanitize_request_payload  # noqa: E402

GROK_MESSAGE_EXTRAS = ("model_id", "id", "timestamp")
OPENAI_COMPAT_KEYS = {
    "role",
    "content",
    "name",
    "tool_calls",
    "tool_call_id",
    "function_call",
    "refusal",
    "reasoning_content",
}
PASSTHROUGH_MESSAGE_KEYS = {"role", "content", *GROK_MESSAGE_EXTRAS}
STRIPPED_MESSAGE_KEYS = {"role", "content"}

TOOLS = [
    {
        "type": "function",
        "function": {"name": "read", "parameters": {"type": "object"}},
    }
]


def _grok_style_payload() -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": "ok",
                "model_id": "grok-build",
                "id": "msg_1",
                "timestamp": "2026-09-13T00:00:00Z",
            },
            {
                "role": "user",
                "content": "next",
                "model_id": "grok-build",
                "id": "msg_2",
                "timestamp": "2026-09-13T00:00:01Z",
            },
        ],
        "tools": copy.deepcopy(TOOLS),
        "tool_choice": "auto",
        "user": "opencode-session",
    }


# (model, expected_message_keys, extras_stripped, local_openai_base, reason)
_MATRIX = (
    (
        "fireworks/accounts/fireworks/models/glm-5p3-flash",
        STRIPPED_MESSAGE_KEYS,
        True,
        False,
        "Fireworks OpenAI-compat rejects Grok extras (400); strip is live-justified",
    ),
    (
        "opencode_go/glm-5.3-flash",
        STRIPPED_MESSAGE_KEYS,
        True,
        False,
        "OpenCode GO OpenAI-compat chat shares the Fireworks extra-key reject",
    ),
    (
        "opencode_go_messages/minimax-m3",
        PASSTHROUGH_MESSAGE_KEYS,
        False,
        False,
        "Anthropic Messages transform owns shape",
    ),
    (
        "chutes/Qwen/Qwen3.5-TEE",
        PASSTHROUGH_MESSAGE_KEYS,
        False,
        False,
        "no live rejection observed; extend only after smoke-ladder shows 400",
    ),
    (
        "openai/gpt-5.6-sol",
        PASSTHROUGH_MESSAGE_KEYS,
        False,
        True,
        "codex-lb strip drops top-level chat params, not Grok message extras",
    ),
)


@pytest.mark.parametrize(
    "model,expected_message_keys,extras_stripped,local_openai_base,reason",
    _MATRIX,
    ids=("fireworks", "opencode_go", "opencode_go_messages", "chutes", "openai_codex_lb"),
)
def test_sanitizer_contract_matrix_for_grok_style_payload(
    model: str,
    expected_message_keys: set[str],
    extras_stripped: bool,
    local_openai_base: bool,
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if local_openai_base:
        monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:2455/v1")
    else:
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    payload = _grok_style_payload()
    sanitized = sanitize_request_payload(payload, model)

    assert sanitized["tools"] == TOOLS, reason
    assert sanitized["tool_choice"] == "auto", reason
    assert isinstance(sanitized["messages"], list)
    assert sanitized["messages"], reason
    for message in sanitized["messages"]:
        assert set(message) == expected_message_keys, reason
        if extras_stripped:
            for extra in GROK_MESSAGE_EXTRAS:
                assert extra not in message, reason
        else:
            for extra in GROK_MESSAGE_EXTRAS:
                assert extra in message, reason
        assert set(message) <= (OPENAI_COMPAT_KEYS | set(GROK_MESSAGE_EXTRAS))

    if local_openai_base:
        assert "user" not in sanitized, reason
    else:
        assert sanitized.get("user") == "opencode-session", reason
