from __future__ import annotations

import json

from rotator_library.openai_stream_normalize import JsonObject, OpenAIStreamNormalizer


def test_normalizer_drops_null_function_name_on_open() -> None:
    normalizer = OpenAIStreamNormalizer()
    chunk: JsonObject = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "type": "function",
                            "function": {"name": None, "arguments": ""},
                        }
                    ]
                }
            }
        ]
    }
    result = normalizer.normalize(chunk)
    tool = result["choices"][0]["delta"]["tool_calls"][0]
    assert isinstance(tool["id"], str) and tool["id"]
    assert "name" not in tool["function"]
    assert tool["function"]["arguments"] == ""
    assert '"name": null' not in json.dumps(result)


def test_normalizer_sticky_function_name_across_null_deltas() -> None:
    normalizer = OpenAIStreamNormalizer()
    first = normalizer.normalize(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": None, "function": {"name": "edit", "arguments": ""}}
                        ]
                    }
                }
            ]
        }
    )
    second = normalizer.normalize(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": None, "function": {"name": None, "arguments": "{"}}
                        ]
                    }
                }
            ]
        }
    )
    assert first["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "edit"
    assert second["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "edit"
    assert (
        second["choices"][0]["delta"]["tool_calls"][0]["id"]
        == first["choices"][0]["delta"]["tool_calls"][0]["id"]
    )


def test_normalizer_coerces_string_tool_index() -> None:
    normalizer = OpenAIStreamNormalizer()
    result = normalizer.normalize(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": "0", "id": None, "function": {"name": "x", "arguments": ""}}
                        ]
                    }
                }
            ]
        }
    )
    tool = result["choices"][0]["delta"]["tool_calls"][0]
    assert tool["index"] == 0
    assert tool["id"] == "call_0"


def test_normalizer_null_arguments_become_empty_string() -> None:
    normalizer = OpenAIStreamNormalizer()
    result = normalizer.normalize(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_x",
                                "function": {"name": "edit", "arguments": None},
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert result["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == ""
