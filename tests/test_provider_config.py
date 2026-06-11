import json
import os
import sys
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "src" / "proxy_app" / "main.py"
sys.path.insert(0, str(ROOT / "src"))

from rotator_library.provider_config import ProviderConfig, discover_api_keys_from_env
from rotator_library.client import RotatingClient


def load_proxy_main_env_helpers():
    source = MAIN_PATH.read_text(encoding="utf-8")
    start = source.index("def _set_env_default(")
    end = source.index("# Load main .env first")
    namespace = {
        "os": os,
        "Path": Path,
        "dotenv_values": dotenv_values,
        "load_dotenv": load_dotenv,
    }
    exec(source[start:end], namespace)
    return namespace


def load_proxy_main_env_normalizer():
    return load_proxy_main_env_helpers()["_normalize_provider_env_aliases"]


def load_proxy_main_credential_summary_builder():
    source = MAIN_PATH.read_text(encoding="utf-8")
    start = source.index("def build_credential_summary(")
    end = source.index('@app.get("/v1/credential-summary")')
    namespace = {"Any": object, "RotatingClient": RotatingClient, "json": json}
    exec(source[start:end], namespace)
    return namespace["build_credential_summary"]


def load_proxy_main_startup_credential_summary_printer():
    source = MAIN_PATH.read_text(encoding="utf-8")
    start = source.index("def build_credential_summary(")
    end = source.index('@app.get("/v1/credential-summary")')
    namespace = {"Any": object, "RotatingClient": RotatingClient, "json": json}
    exec(source[start:end], namespace)
    return namespace["print_startup_credential_summary"]


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


def test_proxy_main_env_normalizer_materializes_numbered_ollama_cloud_alias(
    monkeypatch,
):
    for key in (
        "OLLAMA_API_KEY",
        "OLLAMA_CLOUD_API_KEY",
        "OLLAMA_API_KEY_1",
        "OLLAMA_CLOUD_API_KEY_1",
        "OLLAMA_API_KEY_2",
        "OLLAMA_CLOUD_API_KEY_2",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OLLAMA_API_KEY", "fake-exact-canonical")
    monkeypatch.setenv("OLLAMA_API_KEY_1", "fake-numbered-canonical")
    monkeypatch.setenv("OLLAMA_API_KEY_2", "fake-numbered-canonical-with-explicit")
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY_2", "fake-numbered-runtime-explicit")

    normalize = load_proxy_main_env_normalizer()
    normalize()

    assert os.environ["OLLAMA_CLOUD_API_KEY"] == "fake-exact-canonical"
    assert os.environ["OLLAMA_CLOUD_API_KEY_1"] == "fake-numbered-canonical"
    assert os.environ["OLLAMA_CLOUD_API_KEY_2"] == "fake-numbered-runtime-explicit"
    relevant_env = {
        key: os.environ[key]
        for key in (
            "OLLAMA_API_KEY",
            "OLLAMA_CLOUD_API_KEY",
            "OLLAMA_API_KEY_1",
            "OLLAMA_CLOUD_API_KEY_1",
            "OLLAMA_API_KEY_2",
            "OLLAMA_CLOUD_API_KEY_2",
        )
        if key in os.environ
    }
    api_keys = discover_api_keys_from_env(relevant_env)
    client = RotatingClient(api_keys=api_keys, configure_logging=False)

    assert sorted(api_keys) == ["ollama_cloud"]
    assert sorted(client.all_credentials) == ["ollama_cloud"]
    assert len(client.all_credentials["ollama_cloud"]) == 4


def test_proxy_main_env_normalizer_clears_parent_disabled_when_durable_env_omits_it(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DISABLED_PROVIDERS", "ollama_cloud")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OLLAMA_API_KEY_1=fake-numbered-canonical\nFIREWORKS_API_V2_KEY=fake-fireworks\n",
        encoding="utf-8",
    )
    source = MAIN_PATH.read_text(encoding="utf-8")
    start = source.index("def _set_env_default(")
    end = source.index("# Load main .env first")
    namespace = {"os": os, "Path": Path, "dotenv_values": dotenv_values}
    exec(source[start:end], namespace)

    namespace["_apply_durable_disabled_providers"](namespace["_load_durable_env_values"](env_file))

    assert "DISABLED_PROVIDERS" not in os.environ


def test_proxy_main_env_normalizer_preserves_durable_disabled_when_explicitly_set(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DISABLED_PROVIDERS", "chutes")
    env_file = tmp_path / ".env"
    env_file.write_text("DISABLED_PROVIDERS=ollama_cloud\n", encoding="utf-8")
    source = MAIN_PATH.read_text(encoding="utf-8")
    start = source.index("def _set_env_default(")
    end = source.index("# Load main .env first")
    namespace = {"os": os, "Path": Path, "dotenv_values": dotenv_values}
    exec(source[start:end], namespace)

    namespace["_apply_durable_disabled_providers"](namespace["_load_durable_env_values"](env_file))

    assert os.environ["DISABLED_PROVIDERS"] == "ollama_cloud"


def test_proxy_main_durable_ollama_alias_overrides_inherited_cloud_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY", "fake-inherited-cloud")
    monkeypatch.setenv("OLLAMA_CLOUD_API_KEY_1", "fake-inherited-numbered-cloud")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OLLAMA_API_KEY=fake-durable-canonical\nOLLAMA_API_KEY_1=fake-durable-numbered\n",
        encoding="utf-8",
    )
    helpers = load_proxy_main_env_helpers()

    helpers["_load_router_env"](env_file)

    assert os.environ["OLLAMA_CLOUD_API_KEY"] == "fake-durable-canonical"
    assert os.environ["OLLAMA_CLOUD_API_KEY_1"] == "fake-durable-numbered"
    api_keys = discover_api_keys_from_env(
        {
            "OLLAMA_API_KEY": os.environ["OLLAMA_API_KEY"],
            "OLLAMA_CLOUD_API_KEY": os.environ["OLLAMA_CLOUD_API_KEY"],
            "OLLAMA_API_KEY_1": os.environ["OLLAMA_API_KEY_1"],
            "OLLAMA_CLOUD_API_KEY_1": os.environ["OLLAMA_CLOUD_API_KEY_1"],
        }
    )

    assert api_keys == {"ollama_cloud": ["fake-durable-canonical", "fake-durable-numbered"]}


def test_proxy_main_credential_summary_counts_only_and_omits_identifiers():
    client = RotatingClient(
        api_keys={
            "ollama_cloud": ["fake-ollama-secret", "fake-ollama-secret-2"],
            "openrouter_non_zdr": ["fake-openrouter-secret"],
        },
        oauth_credentials={"gemini_cli": ["env://gemini_cli/1", "/tmp/fake-oauth.json"]},
        configure_logging=False,
    )
    build_summary = load_proxy_main_credential_summary_builder()

    summary = build_summary(client)
    encoded = json.dumps(summary, sort_keys=True)

    assert summary == {
        "schema_version": "credential_summary.v1",
        "providers": {"gemini_cli": 2, "ollama_cloud": 2, "openrouter_non_zdr": 1},
        "api_key_providers": {"ollama_cloud": 2, "openrouter_non_zdr": 1},
        "oauth_providers": {"gemini_cli": 2},
        "total_providers": 3,
        "total_credentials": 5,
    }
    for forbidden in (
        "fake-ollama-secret",
        "fake-openrouter-secret",
        "env://gemini_cli/1",
        "/tmp/fake-oauth.json",
        "Authorization",
        "Bearer",
    ):
        assert forbidden not in encoded


def test_proxy_main_startup_credential_summary_is_count_only(capsys):
    client = RotatingClient(
        api_keys={
            "ollama_cloud": ["fake-ollama-secret", "fake-ollama-secret-2"],
            "openrouter_non_zdr": ["fake-openrouter-secret"],
        },
        oauth_credentials={
            "gemini_cli": ["env://gemini_cli/1", "/tmp/fake-oauth.json"],
            "qwen_code": ["/tmp/fake-qwen-oauth.json"],
        },
        configure_logging=False,
    )
    print_summary = load_proxy_main_startup_credential_summary_printer()

    print_summary(client, disabled_provider_count=1)

    captured = capsys.readouterr().out
    assert "Credential Summary: " in captured
    assert "credential_summary.v1" in captured
    assert "ollama_cloud" in captured
    assert "openrouter_non_zdr" in captured
    assert "gemini_cli" in captured
    assert "qwen_code" in captured
    assert '"disabled_provider_count": 1' in captured
    assert '"total_providers": 4' in captured
    assert '"total_credentials": 6' in captured
    for forbidden in (
        "fake-ollama-secret",
        "fake-ollama-secret-2",
        "fake-openrouter-secret",
        "env://gemini_cli/1",
        "/tmp/fake-oauth.json",
        "/tmp/fake-qwen-oauth.json",
        "Authorization",
        "Bearer",
        "fake-proxy-key",
        "cpk_fake_proxy_key",
        "sk-fake-provider-key",
    ):
        assert forbidden not in captured


def test_proxy_main_startup_display_does_not_include_raw_proxy_key():
    source = MAIN_PATH.read_text(encoding="utf-8")

    assert 'key_display = "✓ Set"' in source
    assert 'key_display = f"✓ {proxy_api_key}"' not in source
    assert 'print(f"Proxy API Key: {key_display}")' in source
