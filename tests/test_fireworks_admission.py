from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from rotator_library.client import RotatingClient
from test_local_transport_safe_mode import _block_catalog_fetches, _import_proxy_main


APPROVED_FIREWORKS_MODEL = "fireworks/accounts/fireworks/models/glm-5p3-flash"
UNAPPROVED_FIREWORKS_MODEL = "fireworks/accounts/fireworks/models/not-approved"
OTHER_FIREWORKS_MODEL = "fireworks/accounts/fireworks/models/second-exact-id"


def _set_fireworks_whitelist(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv("WHITELIST_MODELS_FIREWORKS", raising=False)
    else:
        monkeypatch.setenv("WHITELIST_MODELS_FIREWORKS", value)


def _bare_dispatch_client() -> tuple[RotatingClient, dict[str, int]]:
    client = object.__new__(RotatingClient)
    client.routing_policy = None
    calls = {"nonstream": 0, "stream": 0}

    async def execute_with_retry(*_args, **_kwargs):
        calls["nonstream"] += 1
        return {"ok": True}

    def streaming_with_retry(*_args, **_kwargs):
        calls["stream"] += 1
        return iter(())

    client._execute_with_retry = execute_with_retry
    client._streaming_acompletion_with_retry = streaming_with_retry
    return client, calls


def _dispatch(client: RotatingClient, surface: str, model: str):
    match surface:
        case "completion_nonstream":
            return client.acompletion(model=model, stream=False)
        case "completion_stream":
            return client.acompletion(model=model, stream=True)
        case "embedding":
            return client.aembedding(model=model, input=["admission"])
        case unreachable:
            raise AssertionError(f"unknown test surface: {unreachable}")


@pytest.mark.parametrize(
    ("whitelist", "model", "admitted"),
    [
        (None, APPROVED_FIREWORKS_MODEL, True),
        (None, UNAPPROVED_FIREWORKS_MODEL, False),
        ("", APPROVED_FIREWORKS_MODEL, False),
        ("", UNAPPROVED_FIREWORKS_MODEL, False),
        (APPROVED_FIREWORKS_MODEL, APPROVED_FIREWORKS_MODEL, True),
        (APPROVED_FIREWORKS_MODEL, UNAPPROVED_FIREWORKS_MODEL, False),
    ],
)
@pytest.mark.parametrize(
    "surface",
    ["completion_nonstream", "completion_stream", "embedding"],
)
@pytest.mark.asyncio
async def test_fireworks_dispatch_admission_matrix(
    monkeypatch: pytest.MonkeyPatch,
    whitelist: str | None,
    model: str,
    admitted: bool,
    surface: str,
) -> None:
    # Given the exact Fireworks whitelist environment and an isolated dispatcher.
    _set_fireworks_whitelist(monkeypatch, whitelist)
    client, calls = _bare_dispatch_client()

    # When a completion or embedding request crosses the public dispatcher.
    if admitted:
        result = _dispatch(client, surface, model)
        if inspect.isawaitable(result):
            await result
    else:
        with pytest.raises(ValueError, match="Fireworks model is not admitted"):
            _dispatch(client, surface, model)

    # Then rejection is synchronous and no downstream dispatch path is entered.
    expected_calls = {
        "nonstream": int(admitted and surface != "completion_stream"),
        "stream": int(admitted and surface == "completion_stream"),
    }
    assert calls == expected_calls


@pytest.mark.parametrize(
    "whitelist",
    [
        "*",
        OTHER_FIREWORKS_MODEL,
        f"{APPROVED_FIREWORKS_MODEL},{OTHER_FIREWORKS_MODEL}",
        f" {APPROVED_FIREWORKS_MODEL}",
    ],
)
@pytest.mark.parametrize(
    "surface",
    ["completion_nonstream", "completion_stream", "embedding"],
)
def test_fireworks_dispatch_hard_fails_invalid_widening_configuration(
    monkeypatch: pytest.MonkeyPatch,
    whitelist: str,
    surface: str,
) -> None:
    # Given a Fireworks whitelist value other than the one exact approved ID.
    _set_fireworks_whitelist(monkeypatch, whitelist)
    client, calls = _bare_dispatch_client()

    # When an otherwise approved Fireworks request reaches admission.
    with pytest.raises(ValueError, match="WHITELIST_MODELS_FIREWORKS"):
        _dispatch(client, surface, APPROVED_FIREWORKS_MODEL)

    # Then invalid configuration hard-fails before any dispatch side effect.
    assert calls == {"nonstream": 0, "stream": 0}


@pytest.mark.parametrize(
    ("whitelist", "expected_models", "expected_fetches", "hard_failure"),
    [
        (None, [APPROVED_FIREWORKS_MODEL], 1, False),
        ("", [], 0, False),
        (APPROVED_FIREWORKS_MODEL, [APPROVED_FIREWORKS_MODEL], 1, False),
        ("*", [], 0, True),
        (OTHER_FIREWORKS_MODEL, [], 0, True),
        (f"{APPROVED_FIREWORKS_MODEL},{OTHER_FIREWORKS_MODEL}", [], 0, True),
    ],
)
@pytest.mark.asyncio
async def test_fireworks_catalog_admission_matrix(
    monkeypatch: pytest.MonkeyPatch,
    whitelist: str | None,
    expected_models: list[str],
    expected_fetches: int,
    hard_failure: bool,
) -> None:
    # Given a Fireworks catalog containing both approved and unapproved models.
    _set_fireworks_whitelist(monkeypatch, whitelist)
    fetches = 0

    class FireworksProvider:
        async def get_models(self, _credential: str, _client) -> list[str]:
            nonlocal fetches
            fetches += 1
            return [APPROVED_FIREWORKS_MODEL, UNAPPROVED_FIREWORKS_MODEL]

    client = object.__new__(RotatingClient)
    client._model_list_cache = {}
    client.all_credentials = {"fireworks": ["credential"]}
    client.http_client = None
    client.whitelist_models = {}
    client.ignore_models = {}
    client._get_provider_instance = lambda _provider: FireworksProvider()

    # When the Fireworks provider catalog is requested.
    if hard_failure:
        with pytest.raises(ValueError, match="WHITELIST_MODELS_FIREWORKS"):
            await client.get_all_available_models(grouped=False)
    else:
        assert await client.get_all_available_models(grouped=False) == expected_models

    # Then invalid or empty configuration cannot trigger model discovery.
    assert fetches == expected_fetches


@pytest.mark.asyncio
async def test_non_fireworks_dispatch_and_catalog_are_unchanged_by_fireworks_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given even an invalid Fireworks-specific widening configuration.
    _set_fireworks_whitelist(monkeypatch, "*")
    client, calls = _bare_dispatch_client()

    class OpenAIProvider:
        async def get_models(self, _credential: str, _client) -> list[str]:
            return ["openai/gpt-5.2", "openai/gpt-5.3"]

    client._model_list_cache = {}
    client.all_credentials = {"openai": ["credential"]}
    client.http_client = None
    client.whitelist_models = {}
    client.ignore_models = {}
    client._get_provider_instance = lambda _provider: OpenAIProvider()

    # When OpenAI dispatch and catalog paths are used.
    response = client.acompletion(model="openai/gpt-5.2", stream=False)
    await response
    models = await client.get_available_models("openai")

    # Then the Fireworks policy has no effect on OpenAI admission.
    assert calls == {"nonstream": 1, "stream": 0}
    assert models == ["openai/gpt-5.2", "openai/gpt-5.3"]


@pytest.mark.parametrize("stream", [False, True])
def test_rejected_fireworks_chat_is_http_4xx_before_credentials_or_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stream: bool,
) -> None:
    # Given the real HTTP app with the default fail-closed Fireworks policy.
    _set_fireworks_whitelist(monkeypatch, None)
    module = _import_proxy_main(
        monkeypatch,
        tmp_path,
        "http://127.0.0.1:2465/v1",
        safe_mode=False,
    )
    _block_catalog_fetches(monkeypatch)
    side_effects = {"credential": 0, "provider": 0, "litellm": 0}

    with TestClient(module.app, base_url="http://127.0.0.1") as client:
        rotating_client = module.app.state.rotating_client

        async def acquire_key(*_args, **_kwargs):
            side_effects["credential"] += 1
            raise AssertionError("credential acquisition must not run")

        def get_provider(*_args, **_kwargs):
            side_effects["provider"] += 1
            raise AssertionError("provider lookup must not run")

        async def call_litellm(*_args, **_kwargs):
            side_effects["litellm"] += 1
            raise AssertionError("LiteLLM must not run")

        monkeypatch.setattr(rotating_client.usage_manager, "acquire_key", acquire_key)
        monkeypatch.setattr(rotating_client, "_get_provider_instance", get_provider)
        monkeypatch.setattr(module.litellm, "acompletion", call_litellm)

        # When an unapproved Fireworks chat request reaches port 8000's handler.
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer proxy-token", "Host": "127.0.0.1"},
            json={
                "model": UNAPPROVED_FIREWORKS_MODEL,
                "messages": [{"role": "user", "content": "admission"}],
                "stream": stream,
            },
        )

    # Then both modes fail before HTTP 200/SSE commitment or provider side effects.
    assert 400 <= response.status_code < 500
    assert response.headers["content-type"].startswith("application/json")
    assert "text/event-stream" not in response.headers["content-type"]
    assert "internal_error" not in response.text
    assert side_effects == {"credential": 0, "provider": 0, "litellm": 0}
