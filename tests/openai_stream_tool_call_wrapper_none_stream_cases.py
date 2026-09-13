from __future__ import annotations

from collections.abc import AsyncGenerator
import json

import pytest

from openai_stream_tool_call_support import UsageManager
from rotator_library.client import RotatingClient, StreamedAPIError
from rotator_library.error_handler import (
    UpstreamStreamUnavailableError,
    classify_error,
    should_rotate_on_error,
)
from rotator_library.openai_stream_normalize import JsonObject


def _client() -> RotatingClient:
    rotating_client = object.__new__(RotatingClient)
    rotating_client.usage_manager = UsageManager()
    return rotating_client


async def _consume(stream: AsyncGenerator[object, None]) -> list[object]:
    return [event async for event in stream]


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_none_stream_raises_typed_error() -> None:
    rotating_client = _client()
    with pytest.raises(StreamedAPIError) as exc_info:
        await _consume(
            rotating_client._safe_streaming_wrapper(None, "credential", "provider/model")
        )
    inner = exc_info.value.data
    assert isinstance(inner, UpstreamStreamUnavailableError)
    assert inner.status_code == 502
    assert "credential" not in str(inner)
    assert "provider/model" not in str(inner)
    assert not isinstance(exc_info.value, AttributeError)


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_non_iterable_raises_typed_error() -> None:
    rotating_client = _client()
    with pytest.raises(StreamedAPIError) as exc_info:
        await _consume(
            rotating_client._safe_streaming_wrapper(object(), "credential", "provider/model")
        )
    inner = exc_info.value.data
    assert isinstance(inner, UpstreamStreamUnavailableError)
    assert inner.status_code == 502
    assert not isinstance(exc_info.value, AttributeError)


@pytest.mark.asyncio
async def test_safe_streaming_wrapper_async_iterator_streams_chunks_unchanged() -> None:
    payload: JsonObject = {
        "id": "chatcmpl_ok",
        "choices": [{"delta": {"content": "hello-stream"}}],
    }

    async def source() -> AsyncGenerator[JsonObject, None]:
        yield payload

    rotating_client = _client()
    events = [
        event
        async for event in rotating_client._safe_streaming_wrapper(
            source(), "credential", "provider/model"
        )
    ]
    data_events = [event for event in events if event != "data: [DONE]\n\n"]
    assert len(data_events) == 1
    decoded = json.loads(str(data_events[0]).removeprefix("data: "))
    assert decoded["id"] == "chatcmpl_ok"
    assert decoded["choices"][0]["delta"]["content"] == "hello-stream"
    assert events[-1] == "data: [DONE]\n\n"


def test_upstream_stream_unavailable_is_rotatable_server_error() -> None:
    classified = classify_error(UpstreamStreamUnavailableError())
    assert classified.error_type == "server_error"
    assert classified.status_code == 502
    assert should_rotate_on_error(classified) is True
