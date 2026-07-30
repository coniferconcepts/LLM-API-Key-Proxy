from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from openai_stream_tool_call_support import UsageManager
from rotator_library.client import RotatingClient
from rotator_library.openai_stream_normalize import JsonObject


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_releases_key_before_done() -> None:
    async def source() -> AsyncGenerator[JsonObject, None]:
        yield {"id": "chatcmpl_done", "choices": [{"delta": {"content": "done"}}]}

    usage_manager = UsageManager()
    rotating_client = object.__new__(RotatingClient)
    rotating_client.usage_manager = usage_manager
    events = [
        event
        async for event in rotating_client._safe_streaming_wrapper(
            source(), "credential", "provider/model"
        )
    ]
    assert usage_manager.released == [("credential", "provider/model")]
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_releases_key_after_upstream_exception() -> None:
    async def source() -> AsyncGenerator[JsonObject, None]:
        yield {"id": "chatcmpl_error", "choices": [{"delta": {"content": "before"}}]}
        raise RuntimeError("adversarial upstream failure")

    usage_manager = UsageManager()
    rotating_client = object.__new__(RotatingClient)
    rotating_client.usage_manager = usage_manager
    with pytest.raises(RuntimeError, match="adversarial upstream failure"):
        _ = [
            event
            async for event in rotating_client._safe_streaming_wrapper(
                source(), "credential", "provider/model"
            )
        ]
    assert usage_manager.released == [("credential", "provider/model")]


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_releases_key_after_consumer_cancellation() -> None:
    async def source() -> AsyncGenerator[JsonObject, None]:
        while True:
            yield {"id": "chatcmpl_cancel", "choices": [{"delta": {"content": "partial"}}]}

    usage_manager = UsageManager()
    rotating_client = object.__new__(RotatingClient)
    rotating_client.usage_manager = usage_manager
    wrapped = rotating_client._safe_streaming_wrapper(source(), "credential", "provider/model")
    first = await wrapped.__anext__()
    await wrapped.aclose()
    assert first.startswith("data: ")
    assert usage_manager.released == [("credential", "provider/model")]
