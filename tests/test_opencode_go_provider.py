from __future__ import annotations

import pytest

from rotator_library.go_usage.refresh import go_usage_job_config
from rotator_library.providers.opencode_go_messages_provider import OpenCodeGoMessagesProvider
from rotator_library.providers.opencode_go_provider import OpenCodeGoProvider


@pytest.fixture
def go_bases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCODE_GO_API_BASE", "https://opencode.ai/zen/go/v1")
    monkeypatch.setenv("OPENCODE_GO_MESSAGES_API_BASE", "https://opencode.ai/zen/go")


def test_job_config_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GO_QUOTA_REFRESH_ENABLED", raising=False)
    assert go_usage_job_config() is None


def test_job_config_enabled_has_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GO_QUOTA_REFRESH_ENABLED", "1")
    monkeypatch.setenv("GO_QUOTA_REFRESH_INTERVAL", "10")
    cfg = go_usage_job_config()
    assert cfg is not None
    assert cfg["interval"] == 60
    assert cfg["name"] == "opencode_go_quota_refresh"


def test_zero_arg_subclass_preserves_openai_name(go_bases: None) -> None:
    provider = OpenCodeGoProvider()
    assert provider.provider_name == "opencode_go"
    assert provider.get_background_job_config() is None


def test_messages_provider_name(go_bases: None) -> None:
    provider = OpenCodeGoMessagesProvider()
    assert provider.provider_name == "opencode_go_messages"


def test_plugins_register_zero_arg(go_bases: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GO_QUOTA_REFRESH_ENABLED", raising=False)
    from rotator_library.providers import PROVIDER_PLUGINS

    go_cls = PROVIDER_PLUGINS["opencode_go"]
    messages_cls = PROVIDER_PLUGINS["opencode_go_messages"]
    go = go_cls()
    messages = messages_cls()
    assert go.provider_name == "opencode_go"
    assert messages.provider_name == "opencode_go_messages"
    assert go.get_background_job_config() is None
    assert messages.get_background_job_config() is None
