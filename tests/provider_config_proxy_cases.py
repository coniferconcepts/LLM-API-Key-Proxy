import os

from provider_config_case_support import (
    load_proxy_main_env_helpers,
    load_proxy_main_env_normalizer,
)
from rotator_library.client import RotatingClient
from rotator_library.provider_config import discover_api_keys_from_env


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
    namespace = load_proxy_main_env_helpers()

    namespace["_apply_durable_disabled_providers"](namespace["_load_durable_env_values"](env_file))

    assert "DISABLED_PROVIDERS" not in os.environ


def test_proxy_main_env_normalizer_preserves_durable_disabled_when_explicitly_set(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DISABLED_PROVIDERS", "chutes")
    env_file = tmp_path / ".env"
    env_file.write_text("DISABLED_PROVIDERS=ollama_cloud\n", encoding="utf-8")
    namespace = load_proxy_main_env_helpers()

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
