"""LiteLLM SDK stream chunks still pass through OpenAI tool-call id normalization."""

from __future__ import annotations

import inspect

from rotator_library.client import RotatingClient
from rotator_library.openai_stream_normalize import JsonObject, OpenAIStreamNormalizer


def test_safe_streaming_wrapper_still_uses_openai_stream_normalizer():
    source = inspect.getsource(RotatingClient._safe_streaming_wrapper)
    assert "OpenAIStreamNormalizer" in source
    assert "stream_normalizer" in source


def test_normalizer_assigns_stable_ids_when_tool_call_ids_are_null():
    normalizer = OpenAIStreamNormalizer()
    first: JsonObject = {
        "id": None,
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"name": "run_terminal_command", "arguments": ""},
                        }
                    ]
                }
            }
        ],
    }
    second: JsonObject = {
        "id": None,
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"name": None, "arguments": "{}"},
                        }
                    ]
                }
            }
        ],
    }
    first_result = normalizer.normalize(first)
    second_result = normalizer.normalize(second)
    first_id = first_result["choices"][0]["delta"]["tool_calls"][0]["id"]
    second_id = second_result["choices"][0]["delta"]["tool_calls"][0]["id"]
    assert isinstance(first_id, str) and first_id
    assert first_id == second_id
    assert first_result["id"] == second_result["id"]
    assert first_result["id"]
