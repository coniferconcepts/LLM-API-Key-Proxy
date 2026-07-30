import importlib
import sys
import asyncio
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

client_module = importlib.import_module("rotator_library.client")
routing_policy_module = importlib.import_module("rotator_library.routing_policy")

RotatingClient = getattr(client_module, "RotatingClient")
_merge_openrouter_extra_headers = getattr(client_module, "_merge_openrouter_extra_headers")
RoutingPolicy = getattr(routing_policy_module, "RoutingPolicy")
RoutingPolicyError = getattr(routing_policy_module, "RoutingPolicyError")


def test_openrouter_header_merge_has_dedicated_module_boundary():
    # Given the public helper exposed through the client module.
    helper = _merge_openrouter_extra_headers

    # When its implementation owner is inspected.
    owner = helper.__module__

    # Then header adaptation remains outside the oversized client module.
    assert owner == "rotator_library.openrouter_headers"


def test_client_helper_rewrites_weighted_router_model():
    client = RotatingClient.__new__(RotatingClient)
    client.routing_policy = RoutingPolicy(
        model_overrides={
            "nemotron-3-super": {
                "strategy": "single",
                "primary": "ollama",
                "allowed_providers": ["ollama"],
                "fallback_providers": [],
                "strict": True,
                "allow_global_fallback": False,
            }
        },
        available_providers={"ollama"},
        provider_models={"ollama": {"nemotron-3-super"}},
    )

    model, decision = client._apply_routing_policy("weighted-router/nemotron-3-super")

    assert model == "ollama/nemotron-3-super"
    assert decision is not None
    assert decision.override_applied is True


def test_client_helper_passthrough_without_routing_policy():
    client = RotatingClient.__new__(RotatingClient)
    client.routing_policy = None

    model, decision = client._apply_routing_policy("ollama/nemotron-3-super")

    assert model == "ollama/nemotron-3-super"
    assert decision is None


def test_load_model_routing_overrides_from_env(monkeypatch):
    client = RotatingClient.__new__(RotatingClient)
    monkeypatch.setenv(
        "MODEL_ROUTING_OVERRIDES",
        '{"nemotron-3-super":{"strategy":"single","primary":"ollama","allowed_providers":["ollama"],"fallback_providers":[]}}',
    )

    overrides = client._load_model_routing_overrides_from_env()

    assert overrides["nemotron-3-super"]["primary"] == "ollama_cloud"
    assert overrides["nemotron-3-super"]["allowed_providers"] == ["ollama_cloud"]


def test_invalid_model_routing_overrides_env_fails_closed(monkeypatch):
    client = RotatingClient.__new__(RotatingClient)
    monkeypatch.setenv("MODEL_ROUTING_OVERRIDES", "{invalid")

    with pytest.raises(RoutingPolicyError, match="Invalid JSON"):
        client._load_model_routing_overrides_from_env()


def test_acompletion_rewrites_model_before_dispatch(monkeypatch):
    client = RotatingClient.__new__(RotatingClient)
    client.routing_policy = RoutingPolicy(
        model_overrides={
            "nemotron-3-super": {
                "strategy": "single",
                "primary": "ollama",
                "allowed_providers": ["ollama"],
                "fallback_providers": [],
                "strict": True,
                "allow_global_fallback": False,
            }
        },
        available_providers={"ollama"},
        provider_models={"ollama": {"nemotron-3-super"}},
    )

    captured = {}

    async def fake_execute_with_retry(api_call, request=None, pre_request_callback=None, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(client, "_execute_with_retry", fake_execute_with_retry)
    monkeypatch.setattr(client, "_log_route_decision", lambda decision: None)

    result = asyncio.run(client.acompletion(model="weighted-router/nemotron-3-super", stream=False))

    assert result == {"ok": True}
    assert captured["model"] == "ollama/nemotron-3-super"


def test_client_helper_rewrites_weighted_qwen3_5_model():
    client = RotatingClient.__new__(RotatingClient)
    client.routing_policy = RoutingPolicy(
        model_overrides={
            "qwen3.5": {
                "strategy": "weighted",
                "allowed_providers": ["ollama", "chutes"],
                "weights": {"ollama": 80, "chutes": 20},
                "excluded_providers": ["opencode_go"],
                "fallback_providers": [],
                "strict": True,
                "allow_global_fallback": False,
            }
        },
        available_providers={"ollama", "chutes"},
        provider_models={"ollama": {"qwen3.5"}, "chutes": {"qwen3.5"}},
        known_providers={"ollama", "chutes", "opencode_go"},
        rng=random.Random(1),
    )

    model, decision = client._apply_routing_policy("weighted-router/qwen3.5")

    assert model in {"ollama/qwen3.5", "chutes/qwen3.5"}
    assert decision is not None
    assert decision.strategy == "weighted"
    assert decision.excluded_providers == ["opencode_go"]


def test_unknown_upstream_alias_error_guides_users_to_weighted_router():
    client = RotatingClient.__new__(RotatingClient)

    with pytest.raises(ValueError, match="weighted router at port 8001") as exc_info:
        client._raise_unknown_provider_error("qwen3.5", "qwen3.5")

    message = str(exc_info.value)
    assert "Plain alias models like 'qwen3.5' belong on the weighted router at port 8001" in message
    assert "ollama_cloud/qwen3.5" in message
    assert "opencode_go/..." in message


def test_unknown_prefixed_provider_error_keeps_standard_message():
    client = RotatingClient.__new__(RotatingClient)

    with pytest.raises(
        ValueError,
        match="No API keys or OAuth credentials configured for provider: unknown_provider",
    ):
        client._raise_unknown_provider_error(
            "unknown_provider/qwen3.5",
            "unknown_provider",
        )


def test_merge_openrouter_extra_headers_copies_attribution_from_request():
    class Request:
        headers = {
            "HTTP-Referer": "https://opencode.ai",
            "X-OpenRouter-Title": "OpenCode/opencode-router",
            "X-Title": "OpenCode/opencode-router",
            "X-OpenRouter-Categories": "cli-agent",
        }

    kwargs = _merge_openrouter_extra_headers({"messages": []}, Request())

    assert kwargs["extra_headers"]["HTTP-Referer"] == "https://opencode.ai"
    assert kwargs["extra_headers"]["X-OpenRouter-Title"] == "OpenCode/opencode-router"
    assert kwargs["extra_headers"]["X-Title"] == "OpenCode/opencode-router"
    assert kwargs["extra_headers"]["X-OpenRouter-Categories"] == "cli-agent"


def test_merge_openrouter_extra_headers_preserves_existing_values():
    class Request:
        headers = {
            "HTTP-Referer": "https://opencode.ai",
            "X-OpenRouter-Title": "OpenCode/opencode-router",
        }

    kwargs = _merge_openrouter_extra_headers(
        {
            "extra_headers": {
                "HTTP-Referer": "https://custom.example",
                "Existing": "value",
            }
        },
        Request(),
    )

    assert kwargs["extra_headers"]["HTTP-Referer"] == "https://custom.example"
    assert kwargs["extra_headers"]["X-OpenRouter-Title"] == "OpenCode/opencode-router"
    assert kwargs["extra_headers"]["Existing"] == "value"
