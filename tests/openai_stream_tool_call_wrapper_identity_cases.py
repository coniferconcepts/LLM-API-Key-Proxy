from __future__ import annotations

from collections.abc import AsyncGenerator
import json

import pytest

from openai_stream_tool_call_support import UsageManager, upstream_chunks
from rotator_library.client import RotatingClient
from rotator_library.openai_stream_normalize import JsonObject


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_assigns_stable_ids_when_tool_call_ids_are_null() -> None:
    rotating_client = object.__new__(RotatingClient)
    rotating_client.usage_manager = UsageManager()
    events = [
        json.loads(event.removeprefix("data: "))
        async for event in rotating_client._safe_streaming_wrapper(
            upstream_chunks(), "credential", "provider/model"
        )
        if event != "data: [DONE]\n\n"
    ]
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
    for event in events:
        payload = json.dumps(event)
        assert '"name": null' not in payload
        assert '"id": null' not in payload


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_serializes_multi_index_tool_state_without_null_names() -> (
    None
):
    async def source() -> AsyncGenerator[JsonObject, None]:
        yield {
            "id": None,
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": None, "function": {"name": "read", "arguments": ""}},
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
                            {"index": 1, "id": None, "function": {"name": None, "arguments": "["}},
                        ]
                    }
                }
            ],
        }

    rotating_client = object.__new__(RotatingClient)
    rotating_client.usage_manager = UsageManager()
    serialized = [
        event
        async for event in rotating_client._safe_streaming_wrapper(
            source(), "credential", "provider/model"
        )
        if event != "data: [DONE]\n\n"
    ]
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
