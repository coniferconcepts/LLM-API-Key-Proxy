import os

from rotator_library.provider_config import ProviderConfig, discover_api_keys_from_env


def test_fireworks_v2_api_base_rewrites_fireworks_models(monkeypatch):
    monkeypatch.setenv("FIREWORKS_V2_API_BASE", "https://api.fireworks.ai/inference/v1")

    config = ProviderConfig()

    result = config.convert_for_litellm(
        model="fireworks/accounts/fireworks/routers/kimi-k2p6-turbo",
        api_key="test-v2-key",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result["model"] == "openai/accounts/fireworks/routers/kimi-k2p6-turbo"
    assert result["api_base"] == "https://api.fireworks.ai/inference/v1"
    assert result["custom_llm_provider"] == "openai"
    assert result["api_key"] == "test-v2-key"


def test_fireworks_api_base_alias_is_trimmed_and_not_registered_as_fireworks_v2(
    monkeypatch,
):
    monkeypatch.setenv("FIREWORKS_V2_API_BASE", "https://api.fireworks.ai/inference/v1/")

    config = ProviderConfig()

    assert config.get_api_base("fireworks") == "https://api.fireworks.ai/inference/v1"
    assert config.get_api_base("fireworks_v2") is None
    assert "fireworks" in config.get_custom_providers()


def test_provider_config_does_not_read_legacy_fireworks_keys(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "legacy-metered-key")
    monkeypatch.setenv("FIREWORKS_AI_API_KEY", "legacy-ai-key")
    monkeypatch.setenv("FIREWORKS_API_V2_KEY", "subscription-key")
    monkeypatch.setenv("FIREWORKS_V2_API_BASE", "https://api.fireworks.ai/inference/v1")

    config = ProviderConfig()
    result = config.convert_for_litellm(
        model="fireworks/accounts/fireworks/routers/kimi-k2p6-turbo",
        api_key=os.environ["FIREWORKS_API_V2_KEY"],
    )

    assert result["api_key"] == "subscription-key"
    assert result["api_key"] != os.environ["FIREWORKS_API_KEY"]
    assert result["api_key"] != os.environ["FIREWORKS_AI_API_KEY"]


def test_opencode_go_messages_uses_anthropic_provider_with_unsuffixed_base(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_MESSAGES_API_BASE", "https://opencode.ai/zen/go")

    config = ProviderConfig()

    result = config.convert_for_litellm(
        model="opencode_go_messages/qwen3.7-max",
        api_key="go-key",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result["model"] == "anthropic/qwen3.7-max"
    assert result["api_base"] == "https://opencode.ai/zen/go"
    assert result["custom_llm_provider"] == "anthropic"


def test_opencode_go_messages_strips_v1_base_for_litellm_anthropic(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_MESSAGES_API_BASE", "https://opencode.ai/zen/go/v1")

    config = ProviderConfig()

    result = config.convert_for_litellm(model="opencode_go_messages/qwen3.7-max")

    assert result["api_base"] == "https://opencode.ai/zen/go"


def test_discover_api_keys_maps_fireworks_v2_key_to_fireworks_provider():
    api_keys = discover_api_keys_from_env(
        {
            "PROXY_API_KEY": "proxy-key",
            "FIREWORKS_API_V2_KEY": "subscription-key",
            "FIREWORKS_API_KEY": "legacy-metered-key",
            "FIREWORKS_AI_API_KEY": "legacy-ai-key",
            "OLLAMA_CLOUD_API_KEY": "ollama-key",
        }
    )

    assert api_keys["fireworks"] == ["subscription-key"]
    assert api_keys["ollama_cloud"] == ["ollama-key"]
    assert "fireworks_api_v2" not in api_keys
    assert "proxy" not in api_keys


def test_discover_api_keys_maps_ollama_shell_keys_to_ollama_cloud_provider():
    api_keys = discover_api_keys_from_env(
        {
            "OLLAMA_API_KEY": "canonical-key",
            "OLLAMA_API_KEY_1": "numbered-canonical-key",
            "OLLAMA_CLOUD_API_KEY": "runtime-key",
            "OLLAMA_CLOUD_API_KEY_1": "numbered-runtime-key",
        }
    )

    assert api_keys == {
        "ollama_cloud": [
            "canonical-key",
            "numbered-canonical-key",
            "runtime-key",
            "numbered-runtime-key",
        ]
    }
    assert "ollama" not in api_keys


def test_discover_api_keys_deduplicates_same_value_after_provider_alias_mapping():
    api_keys = discover_api_keys_from_env(
        {
            "OLLAMA_API_KEY": "same-key",
            "OLLAMA_CLOUD_API_KEY": "same-key",
            "OPENROUTER_FREE_KEY": "same-openrouter-key",
            "OPENROUTER_NON_ZDR_API_KEY": "same-openrouter-key",
        }
    )

    assert api_keys == {
        "ollama_cloud": ["same-key"],
        "openrouter_non_zdr": ["same-openrouter-key"],
    }


def test_discover_api_keys_maps_openrouter_free_keys_to_non_zdr_provider():
    api_keys = discover_api_keys_from_env(
        {
            "OPENROUTER_FREE_KEY": "free-key",
            "OPENROUTER_FREE_API_KEY": "free-api-key",
            "OPENROUTER_FREE_API_KEY_1": "numbered-free-api-key",
            "OPENROUTER_NON_ZDR_KEY": "non-zdr-key",
            "OPENROUTER_NON_ZDR_API_KEY": "non-zdr-api-key",
        }
    )

    assert api_keys == {
        "openrouter_non_zdr": [
            "free-key",
            "free-api-key",
            "numbered-free-api-key",
            "non-zdr-key",
            "non-zdr-api-key",
        ]
    }
    assert "openrouter_free" not in api_keys


def test_discover_api_keys_ignores_numbered_non_api_key_aliases():
    api_keys = discover_api_keys_from_env(
        {
            "OPENROUTER_FREE_KEY_1": "numbered-free-key",
            "OPENCODE_GO_KEY_1": "numbered-go-key",
            "FIREWORKS_API_V2_KEY_1": "numbered-fireworks-key",
            "OLLAMA_API_KEY_1": "numbered-ollama-key",
        }
    )

    assert api_keys == {"ollama_cloud": ["numbered-ollama-key"]}


def test_discover_api_keys_skips_empty_values_and_proxy_key():
    api_keys = discover_api_keys_from_env(
        {
            "PROXY_API_KEY": "proxy-key",
            "FIREWORKS_API_V2_KEY": "",
            "OLLAMA_CLOUD_API_KEY": "ollama-key",
        }
    )

    assert api_keys == {"ollama_cloud": ["ollama-key"]}
