import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


def test_fireworks_api_base_alias_is_trimmed_and_not_registered_as_fireworks_v2(monkeypatch):
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


def test_discover_api_keys_skips_empty_values_and_proxy_key():
    api_keys = discover_api_keys_from_env(
        {
            "PROXY_API_KEY": "proxy-key",
            "FIREWORKS_API_V2_KEY": "",
            "OLLAMA_CLOUD_API_KEY": "ollama-key",
        }
    )

    assert api_keys == {"ollama_cloud": ["ollama-key"]}
