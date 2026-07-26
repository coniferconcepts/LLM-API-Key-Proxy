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
    def __init__(self) -> None:
        self.released: list[tuple[str, str]] = []

    async def record_success(self, _key: str, _model: str) -> None:
        return

    async def release_key(self, key: str, model: str) -> None:
        self.released.append((key, model))


async def _upstream_chunks() -> AsyncGenerator[JsonObject, None]:
    yield {
        "id": None,
        "choices": [{"index": 0, "delta": {"content": "planning"}}],
    }
    # codex-lb style: open with null id AND null name (OpenCode fails both)
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
                            "function": {"name": None, "arguments": ""},
                        }
                    ]
                },
            }
        ],
    }
    # name arrives on a later delta for same index
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
                            "function": {"name": None, "arguments": "{"},
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

    # Then each tool index has a stable string ID and null names are never serialized.
    open_tool = events[1]["choices"][0]["delta"]["tool_calls"][0]
    named_tool = events[2]["choices"][0]["delta"]["tool_calls"][0]
    continued_tool = events[3]["choices"][0]["delta"]["tool_calls"][0]
    second_tool = events[4]["choices"][0]["delta"]["tool_calls"][0]
    existing_tool = events[5]["choices"][0]["delta"]["tool_calls"][0]

    assert isinstance(open_tool["id"], str)
    assert open_tool["id"]
    assert "name" not in open_tool.get("function", {}) or isinstance(
        open_tool["function"].get("name"), str
    )
    assert open_tool["function"].get("name") is not None or "name" not in open_tool["function"]
    assert named_tool["id"] == open_tool["id"]
    assert named_tool["function"]["name"] == "edit"
    assert continued_tool["id"] == open_tool["id"]
    assert continued_tool["function"]["name"] == "edit"
    assert continued_tool["function"].get("name") is not None
    assert second_tool["id"] != open_tool["id"]
    assert existing_tool["id"] == "call_existing"
    assert all(isinstance(event["id"], str) for event in events)
    assert events[0]["choices"][0]["delta"] == {"content": "planning"}
    # JSON must never contain "name": null (OpenCode typeof check)
    for event in events:
        payload = json.dumps(event)
        assert '"name": null' not in payload
        assert '"id": null' not in payload


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_serializes_multi_index_tool_state_without_null_names() -> (
    None
):
    # Given two tool indexes with explicit null names followed by argument-only continuations.
    async def source() -> AsyncGenerator[JsonObject, None]:
        yield {
            "id": None,
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": None,
                                "function": {"name": "read", "arguments": ""},
                            },
                            {
                                "index": 1,
                                "id": None,
                                "function": {"name": "write", "arguments": ""},
                            },
                        ]
                    }
                }
            ],
        }
        yield {
            "id": None,
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": None, "function": {"arguments": "{"}},
                            {
                                "index": 1,
                                "id": None,
                                "function": {"name": None, "arguments": "["},
                            },
                        ]
                    }
                }
            ],
        }

    rotating_client = object.__new__(RotatingClient)
    rotating_client.usage_manager = _UsageManager()

    # When the real streaming wrapper emits serialized SSE records.
    serialized = [
        event
        async for event in rotating_client._safe_streaming_wrapper(
            source(),
            "credential",
            "provider/model",
        )
        if event != "data: [DONE]\n\n"
    ]

    # Then neither explicit null is serialized and each index retains its own identity and name.
    assert all('"name":null' not in event.replace(" ", "") for event in serialized)
    first = json.loads(serialized[0].removeprefix("data: "))["choices"][0]["delta"]["tool_calls"]
    continuation = json.loads(serialized[1].removeprefix("data: "))["choices"][0]["delta"][
        "tool_calls"
    ]
    assert first[0]["id"] != first[1]["id"]
    assert continuation[0]["id"] == first[0]["id"]
    assert continuation[1]["id"] == first[1]["id"]
    assert continuation[0]["function"]["name"] == "read"
    assert continuation[1]["function"]["name"] == "write"


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_releases_key_before_done() -> None:
    # Given a normally completing upstream stream.
    async def source() -> AsyncGenerator[JsonObject, None]:
        yield {"id": "chatcmpl_done", "choices": [{"delta": {"content": "done"}}]}

    usage_manager = _UsageManager()
    rotating_client = object.__new__(RotatingClient)
    rotating_client.usage_manager = usage_manager

    # When the wrapper drains it to its terminal SSE record.
    events = [
        event
        async for event in rotating_client._safe_streaming_wrapper(
            source(),
            "credential",
            "provider/model",
        )
    ]

    # Then the key is released and the public stream terminates with DONE.
    assert usage_manager.released == [("credential", "provider/model")]
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_releases_key_after_upstream_exception() -> None:
    # Given an upstream stream that fails after its first response.
    async def source() -> AsyncGenerator[JsonObject, None]:
        yield {"id": "chatcmpl_error", "choices": [{"delta": {"content": "before"}}]}
        raise RuntimeError("adversarial upstream failure")

    usage_manager = _UsageManager()
    rotating_client = object.__new__(RotatingClient)
    rotating_client.usage_manager = usage_manager

    # When the wrapper propagates the upstream error.
    with pytest.raises(RuntimeError, match="adversarial upstream failure"):
        _ = [
            event
            async for event in rotating_client._safe_streaming_wrapper(
                source(),
                "credential",
                "provider/model",
            )
        ]

    # Then the key is still released without a terminal DONE record.
    assert usage_manager.released == [("credential", "provider/model")]


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_releases_key_after_consumer_cancellation() -> None:
    # Given a stream whose consumer stops after the first response.
    async def source() -> AsyncGenerator[JsonObject, None]:
        while True:
            yield {"id": "chatcmpl_cancel", "choices": [{"delta": {"content": "partial"}}]}

    usage_manager = _UsageManager()
    rotating_client = object.__new__(RotatingClient)
    rotating_client.usage_manager = usage_manager
    wrapped = rotating_client._safe_streaming_wrapper(source(), "credential", "provider/model")

    # When the consumer explicitly cancels the wrapper.
    first = await wrapped.__anext__()
    await wrapped.aclose()

    # Then no terminal record is sent and the key is released for the next stream.
    assert first.startswith("data: ")
    assert usage_manager.released == [("credential", "provider/model")]


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


def test_normalizer_drops_null_function_name_on_open() -> None:
    # Given a first tool delta with null id and null function.name (codex-lb shape).
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
    # Given name on first delta then null name on continuation.
    normalizer = OpenAIStreamNormalizer()
    first = normalizer.normalize(
        {
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
    )
    second = normalizer.normalize(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": None,
                                "function": {"name": None, "arguments": "{"},
                            }
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
