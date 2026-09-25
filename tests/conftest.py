"""Close aiohttp sessions created by tests, including LiteLLM's cached clients."""

import asyncio

import aiohttp
import pytest


@pytest.fixture(autouse=True)
def close_test_aiohttp_clients(monkeypatch: pytest.MonkeyPatch):
    sessions: list[aiohttp.ClientSession] = []
    original_init = aiohttp.ClientSession.__init__

    def tracked_init(session: aiohttp.ClientSession, *args: object, **kwargs: object) -> None:
        original_init(session, *args, **kwargs)
        sessions.append(session)

    monkeypatch.setattr(aiohttp.ClientSession, "__init__", tracked_init)
    yield

    if sessions:
        async def close_sessions() -> None:
            for session in sessions:
                if not session.closed:
                    await session.close()

        asyncio.run(close_sessions())
        import litellm

        cache = getattr(litellm.in_memory_llm_clients_cache, "cache_dict", None)
        if isinstance(cache, dict):
            cache.clear()
