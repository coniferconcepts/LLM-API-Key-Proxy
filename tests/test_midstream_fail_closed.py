from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from credential_admission_contract_support import _MODEL, make_client
import rotator_library.client as client_module


@pytest.fixture(params=[False, True], ids=["litellm", "custom-provider"])
def streaming_client(request, tmp_path, monkeypatch):
    client, _ = make_client(tmp_path, acquire_timeout=1.0)
    client.all_credentials = {"synthetic": ["first-credential", "second-credential"]}
    client.http_client = object()
    client.usage_manager = SimpleNamespace(
        get_credential_availability_stats=AsyncMock(
            return_value={"available": 2, "on_cooldown": 0, "fair_cycle_excluded": 0}
        ),
        acquire_key=AsyncMock(side_effect=["first-credential", "second-credential"]),
        release_key=AsyncMock(),
        record_failure=AsyncMock(),
        record_success=AsyncMock(),
    )
    upstream = AsyncMock()
    if request.param:
        plugin = SimpleNamespace(has_custom_logic=lambda: True, acompletion=upstream)
        client._get_provider_instance = lambda _provider: plugin
    else:
        client._get_provider_instance = lambda _provider: None
        monkeypatch.setattr(client_module.litellm, "acompletion", upstream)
    monkeypatch.setattr(client_module.random, "shuffle", lambda _items: None)
    return client, upstream


def content(index):
    return {"choices": [{"index": 0, "delta": {"content": f"part-{index}"}}]}


async def response(count, error=None):
    for index in range(count):
        yield content(index)
    if error is not None:
        raise error


def server_error():
    return client_module.litellm.InternalServerError(
        message="private-provider-detail", llm_provider="synthetic", model=_MODEL
    )


async def collect(client):
    return [
        chunk
        async for chunk in client._streaming_acompletion_with_retry(
            None, model=_MODEL, messages=[], stream=True
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("failure", ["server", "quota"])
async def test_midstream_error_stops_without_replaying(streaming_client, count, failure):
    client, upstream = streaming_client
    error = server_error() if failure == "server" else client_module.litellm.RateLimitError(
        message="private-provider-detail quota exceeded",
        llm_provider="synthetic",
        model=_MODEL,
    )
    upstream.side_effect = [response(count, error), response(1)]

    chunks = await collect(client)

    assert len(chunks) == count + 2
    for index, chunk in enumerate(chunks[:count]):
        assert json.loads(chunk.removeprefix("data: "))["choices"][0]["delta"] == {
            "content": f"part-{index}"
        }
    terminal = json.loads(chunks[-2].removeprefix("data: "))
    assert terminal == client_module.build_public_stream_error("internal_error")
    assert terminal["error"]["type"] == "proxy_internal_error"
    assert terminal["error"]["status"] == 500
    assert chunks[-1] == "data: [DONE]\n\n"
    assert "quota_exceeded" not in "".join(chunks)
    assert "private-provider-detail" not in "".join(chunks)
    upstream.assert_awaited_once()
    client.usage_manager.acquire_key.assert_awaited_once()
    client.usage_manager.record_failure.assert_awaited_once()
    args, kwargs = client.usage_manager.record_failure.await_args
    assert args[:2] == ("first-credential", _MODEL)
    assert args[2].original_exception is error
    assert args[2].error_type == ("server_error" if failure == "server" else "quota_exceeded")
    assert kwargs.get("increment_consecutive_failures", True) is (failure != "server")
    client.usage_manager.release_key.assert_awaited_once_with("first-credential", _MODEL)


@pytest.mark.asyncio
async def test_error_before_first_chunk_still_rotates(streaming_client):
    client, upstream = streaming_client
    upstream.side_effect = [response(0, server_error()), response(1)]

    chunks = await collect(client)

    assert len(chunks) == 2
    assert json.loads(chunks[0].removeprefix("data: "))["choices"][0]["delta"] == {
        "content": "part-0"
    }
    assert chunks[1] == "data: [DONE]\n\n"
    assert upstream.await_count == 2
    assert client.usage_manager.acquire_key.await_count == 2
    assert [
        args.kwargs.get("credential_identifier", args.kwargs.get("api_key"))
        for args in upstream.await_args_list
    ] == ["first-credential", "second-credential"]
    client.usage_manager.record_failure.assert_awaited_once()
    assert client.usage_manager.release_key.await_args_list == [
        call("first-credential", _MODEL), call("second-credential", _MODEL)
    ]


@pytest.mark.asyncio
async def test_midstream_cancellation_propagates_and_releases_key(streaming_client):
    client, upstream = streaming_client
    upstream.return_value = response(1, asyncio.CancelledError())
    stream = client._streaming_acompletion_with_retry(None, model=_MODEL, messages=[])
    assert "part-0" in await anext(stream)
    with pytest.raises(asyncio.CancelledError):
        await anext(stream)
    upstream.assert_awaited_once()
    client.usage_manager.record_failure.assert_not_awaited()
    client.usage_manager.release_key.assert_awaited_once_with("first-credential", _MODEL)
