from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import pytest

from rotator_library.client import RotatingClient
from rotator_library.openai_stream_normalize import (
    JsonObject,
    OpenAIStreamNormalizer,
)


class _UsageManager:
    async def record_success(self, _key: str, _model: str) -> None:
        return

    async def release_key(self, _key: str, _model: str) -> None:
        return


async def _upstream_chunks() -> AsyncGenerator[JsonObject, None]:
    yield {
        "id": None,
        "choices": [{"index": 0, "delta": {"content": "planning"}}],
    }
    yield {
        "id": None,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "type": "function",
                            "function": {"name": "edit", "arguments": ""},
                        }
                    ]
                },
            }
        ],
    }
    yield {
        "id": None,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"arguments": "{"},
                        }
                    ]
                },
            }
        ],
    }
    yield {
        "id": None,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": None,
                            "type": "function",
                            "function": {"name": "read", "arguments": ""},
                        }
                    ]
                },
            }
        ],
    }
    yield {
        "id": "chatcmpl_existing",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 2,
                            "id": "call_existing",
                            "type": "function",
                            "function": {"name": "write", "arguments": ""},
                        }
                    ]
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_assigns_stable_ids_when_tool_call_ids_are_null() -> None:
    # Given an OpenAI-compatible stream that opens tools with null IDs.
    rotating_client = object.__new__(RotatingClient)
    rotating_client.usage_manager = _UsageManager()

    # When the shared OpenAI stream wrapper serializes the chunks to SSE.
    events = [
        json.loads(event.removeprefix("data: "))
        async for event in rotating_client._safe_streaming_wrapper(
            _upstream_chunks(),
            "credential",
            "provider/model",
        )
        if event != "data: [DONE]\n\n"
    ]

    # Then each tool index has a stable string ID and valid IDs are preserved.
    first_tool = events[1]["choices"][0]["delta"]["tool_calls"][0]
    continued_tool = events[2]["choices"][0]["delta"]["tool_calls"][0]
    second_tool = events[3]["choices"][0]["delta"]["tool_calls"][0]
    existing_tool = events[4]["choices"][0]["delta"]["tool_calls"][0]

    assert isinstance(first_tool["id"], str)
    assert first_tool["id"]
    assert continued_tool["id"] == first_tool["id"]
    assert second_tool["id"] != first_tool["id"]
    assert existing_tool["id"] == "call_existing"
    assert all(isinstance(event["id"], str) for event in events)
    assert events[0]["choices"][0]["delta"] == {"content": "planning"}


def test_normalizer_reuses_synthetic_id_for_continuation_delta() -> None:
    # Given a stream normalizer and two null-ID deltas for the same tool index.
    normalizer = OpenAIStreamNormalizer()
    first: JsonObject = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"name": "edit", "arguments": ""},
                        }
                    ]
                }
            }
        ]
    }
    second: JsonObject = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "function": {"arguments": "{"},
                        }
                    ]
                }
            }
        ]
    }

    # When both deltas are normalized in stream order.
    first_result = normalizer.normalize(first)
    second_result = normalizer.normalize(second)

    # Then the opening ID is a non-empty string reused by the continuation.
    first_id = first_result["choices"][0]["delta"]["tool_calls"][0]["id"]
    second_id = second_result["choices"][0]["delta"]["tool_calls"][0]["id"]
    assert isinstance(first_id, str)
    assert first_id
    assert second_id == first_id


def test_normalizer_assigns_distinct_ids_to_distinct_tool_indexes() -> None:
    # Given two tool indexes that both open without IDs.
    normalizer = OpenAIStreamNormalizer()
    chunk: JsonObject = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": None},
                        {"index": 1, "id": None},
                    ]
                }
            }
        ]
    }

    # When the shared chunk is normalized.
    result = normalizer.normalize(chunk)

    # Then each tool index receives a different stable identifier.
    tool_calls = result["choices"][0]["delta"]["tool_calls"]
    assert tool_calls[0]["id"] != tool_calls[1]["id"]


def test_normalizer_preserves_existing_tool_call_id() -> None:
    # Given a tool delta with a provider-supplied non-empty ID.
    normalizer = OpenAIStreamNormalizer()
    chunk: JsonObject = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": "call_existing"},
                    ]
                }
            }
        ]
    }

    # When the chunk is normalized.
    result = normalizer.normalize(chunk)

    # Then the provider ID is preserved.
    assert result["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_existing"


def test_normalizer_leaves_chunk_without_tool_calls_unchanged() -> None:
    # Given a chunk that has no tool calls or top-level ID.
    normalizer = OpenAIStreamNormalizer()
    chunk: JsonObject = {"choices": [{"delta": {"content": "hello"}}]}

    # When the chunk is normalized.
    result = normalizer.normalize(chunk)

    # Then the original chunk passes through unchanged.
    assert result == {"choices": [{"delta": {"content": "hello"}}]}


def test_normalizer_replaces_null_top_level_chunk_id() -> None:
    # Given an OpenAI chunk with a null top-level ID.
    normalizer = OpenAIStreamNormalizer()
    chunk: JsonObject = {
        "id": None,
        "choices": [{"delta": {"content": "hello"}}],
    }

    # When the chunk is normalized.
    result = normalizer.normalize(chunk)

    # Then the serialized chunk ID satisfies the OpenAI-compatible string contract.
    assert isinstance(result["id"], str)
    assert result["id"]
