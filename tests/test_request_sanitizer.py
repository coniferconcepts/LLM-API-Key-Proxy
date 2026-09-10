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


def test_sanitize_strips_user_for_openai_when_local_api_base(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:2455/v1")
    payload = {
        "model": "openai/gpt-5.6-sol",
        "messages": [{"role": "user", "content": "hi"}],
        "user": "opencode-session",
        "service_tier": "auto",
        "tools": [
            {
                "type": "function",
                "function": {"name": "echo", "parameters": {"type": "object"}},
            }
        ],
    }

    sanitized = sanitize_request_payload(payload, payload["model"])

    assert "user" not in sanitized
    assert "service_tier" not in sanitized
    assert sanitized["tools"]


def test_sanitize_keeps_user_for_openai_without_local_api_base(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    payload = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "user": "platform-user",
    }

    sanitized = sanitize_request_payload(payload, payload["model"])

    assert sanitized["user"] == "platform-user"


def test_sanitize_strips_sampling_and_translates_max_tokens_for_openai_family(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:2455/v1")
    cases = (
        "openai/gpt-6-astra",
        "gpt-6-astra",
        "gpt-6",
        "openai/gpt-5.6-sol",
    )
    for model in cases:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_logprobs": 5,
            "logprobs": True,
            "max_tokens": 128,
        }
        sanitized = sanitize_request_payload(payload, model)
        assert "temperature" not in sanitized, model
        assert "top_p" not in sanitized, model
        assert "top_logprobs" not in sanitized, model
        assert "logprobs" not in sanitized, model
        assert "max_tokens" not in sanitized, model
        assert sanitized["max_completion_tokens"] == 128, model


def test_sanitize_keeps_existing_max_completion_tokens_when_translating(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:2455/v1")
    payload = {
        "model": "openai/gpt-6-astra",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
        "max_completion_tokens": 256,
    }

    sanitized = sanitize_request_payload(payload, payload["model"])

    assert "max_tokens" not in sanitized
    assert sanitized["max_completion_tokens"] == 256


def test_sanitize_keeps_sampling_for_non_openai_with_local_api_base(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:2455/v1")
    payload = {
        "model": "ollama_cloud/glm-5.3-flash:cloud",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "top_p": 0.8,
        "max_tokens": 32,
    }

    sanitized = sanitize_request_payload(payload, payload["model"])

    assert sanitized["temperature"] == 0.2
    assert sanitized["top_p"] == 0.8
    assert sanitized["max_tokens"] == 32
    assert "max_completion_tokens" not in sanitized


def test_sanitize_keeps_user_for_non_openai_with_local_api_base(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:2455/v1")
    payload = {
        "model": "ollama_cloud/kimi-k2.6",
        "messages": [{"role": "user", "content": "hi"}],
        "user": "keep-me",
    }

    sanitized = sanitize_request_payload(payload, payload["model"])

    assert sanitized["user"] == "keep-me"
