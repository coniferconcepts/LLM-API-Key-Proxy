from __future__ import annotations

from rotator_library.openai_stream_normalize import JsonObject, OpenAIStreamNormalizer


def test_normalizer_reuses_synthetic_id_for_continuation_delta() -> None:
    normalizer = OpenAIStreamNormalizer()
    first: JsonObject = {
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
    second: JsonObject = {
        "choices": [
            {"delta": {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "{"}}]}}
        ]
    }
    first_result = normalizer.normalize(first)
    second_result = normalizer.normalize(second)
    first_id = first_result["choices"][0]["delta"]["tool_calls"][0]["id"]
    second_id = second_result["choices"][0]["delta"]["tool_calls"][0]["id"]
    assert isinstance(first_id, str)
    assert first_id
    assert second_id == first_id


def test_normalizer_assigns_distinct_ids_to_distinct_tool_indexes() -> None:
    normalizer = OpenAIStreamNormalizer()
    chunk: JsonObject = {
        "choices": [{"delta": {"tool_calls": [{"index": 0, "id": None}, {"index": 1, "id": None}]}}]
    }
    result = normalizer.normalize(chunk)
    tool_calls = result["choices"][0]["delta"]["tool_calls"]
    assert tool_calls[0]["id"] != tool_calls[1]["id"]


def test_normalizer_preserves_existing_tool_call_id() -> None:
    normalizer = OpenAIStreamNormalizer()
    chunk: JsonObject = {
        "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_existing"}]}}]
    }
    result = normalizer.normalize(chunk)
    assert result["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_existing"


def test_normalizer_leaves_chunk_without_tool_calls_unchanged() -> None:
    normalizer = OpenAIStreamNormalizer()
    chunk: JsonObject = {"choices": [{"delta": {"content": "hello"}}]}
    result = normalizer.normalize(chunk)
    assert result == {"choices": [{"delta": {"content": "hello"}}]}


def test_normalizer_replaces_null_top_level_chunk_id() -> None:
    normalizer = OpenAIStreamNormalizer()
    chunk: JsonObject = {"id": None, "choices": [{"delta": {"content": "hello"}}]}
    result = normalizer.normalize(chunk)
    assert isinstance(result["id"], str)
    assert result["id"]
