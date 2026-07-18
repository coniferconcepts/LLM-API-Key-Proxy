import json

from provider_config_case_support import (
    MAIN_PATH,
    load_proxy_main_credential_summary_builder,
    load_proxy_main_startup_credential_summary_printer,
)
from rotator_library.client import RotatingClient


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
