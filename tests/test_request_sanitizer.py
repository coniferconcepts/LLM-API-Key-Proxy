import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

request_sanitizer = importlib.import_module("rotator_library.request_sanitizer")
sanitize_request_payload = getattr(request_sanitizer, "sanitize_request_payload")


def test_sanitize_request_payload_adds_reasoning_content_for_opencode_go_tool_history():
    payload = {
        "model": "opencode_go/kimi-k2.6",
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "messages": [
            {"role": "user", "content": "Use the tool."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
        ],
    }

    sanitized = sanitize_request_payload(payload, payload["model"])

    assert sanitized["messages"][1]["reasoning_content"] == ""


def test_sanitize_request_payload_preserves_existing_reasoning_content():
    payload = {
        "model": "opencode_go/kimi-k2.6",
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
                "reasoning_content": "plan",
            }
        ],
    }

    sanitized = sanitize_request_payload(payload, payload["model"])

    assert sanitized["messages"][0]["reasoning_content"] == "plan"


def test_sanitize_request_payload_does_not_mutate_other_models():
    payload = {
        "model": "ollama_cloud/kimi-k2.6",
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            }
        ],
    }

    sanitized = sanitize_request_payload(payload, payload["model"])

    assert "reasoning_content" not in sanitized["messages"][0]


def test_sanitize_request_payload_disables_thinking_for_opencode_go_tool_requests():
    payload = {
        "model": "opencode_go/kimi-k2.6",
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "tools": [
            {
                "type": "function",
                "function": {"name": "read", "parameters": {"type": "object"}},
            }
        ],
        "messages": [{"role": "user", "content": "Use the tool."}],
    }

    sanitized = sanitize_request_payload(payload, payload["model"])

    assert "thinking" not in sanitized


def test_sanitize_request_payload_disables_thinking_for_opencode_go_tool_history():
    payload = {
        "model": "opencode_go/kimi-k2.6",
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "messages": [
            {"role": "user", "content": "Use the tool."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ],
    }

    sanitized = sanitize_request_payload(payload, payload["model"])

    assert "thinking" not in sanitized
    assert sanitized["messages"][1]["reasoning_content"] == ""


def test_sanitize_request_payload_keeps_thinking_for_opencode_go_non_tool_requests():
    payload = {
        "model": "opencode_go/kimi-k2.6",
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "messages": [{"role": "user", "content": "Think carefully."}],
    }

    sanitized = sanitize_request_payload(payload, payload["model"])

    assert sanitized["thinking"] == {"type": "enabled", "budget_tokens": 2048}
