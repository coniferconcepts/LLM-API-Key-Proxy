from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
TUI_MODULE_NAMES = ("proxy_app", "proxy_app.launcher_tui")


@pytest.fixture
def launcher_tui_module() -> Iterator[ModuleType]:
    original_modules = {name: sys.modules[name] for name in TUI_MODULE_NAMES if name in sys.modules}
    sys.path.insert(0, str(ROOT / "src"))
    try:
        for name in TUI_MODULE_NAMES:
            sys.modules.pop(name, None)
        yield importlib.import_module("proxy_app.launcher_tui")
    finally:
        sys.path.remove(str(ROOT / "src"))
        for name in TUI_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(original_modules)


def _configured_tui(
    launcher_tui: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    proxy_key: str,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROXY_API_KEY", proxy_key)
    tui = launcher_tui.LauncherTUI()
    tui.env_file.touch()
    tui.console = Console(record=True, force_terminal=False, width=100)
    monkeypatch.setattr(launcher_tui, "clear_screen", lambda: None)
    return tui


def test_main_menu_reports_key_status_without_rendering_secret(
    launcher_tui_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given a configured proxy whose key is a console-detection sentinel.
    secret = "main-menu-secret-sentinel"
    tui = _configured_tui(launcher_tui_module, monkeypatch, tmp_path, secret)
    monkeypatch.setattr(
        launcher_tui_module.SettingsDetector,
        "get_basic_settings",
        lambda: {
            "credentials": {},
            "custom_bases": {},
            "model_definitions": {},
            "concurrency_limits": {},
            "model_filters": {},
        },
    )
    monkeypatch.setattr(launcher_tui_module.Prompt, "ask", lambda *_args, **_kwargs: "8")

    # When the main menu renders.
    with pytest.raises(SystemExit):
        tui.show_main_menu()

    # Then it reports only presence and never writes the key to the console.
    rendered = tui.console.export_text()
    assert "Proxy API Key:       Set" in rendered
    assert secret not in rendered


def test_key_change_uses_hidden_empty_input_and_never_echoes_secrets(
    launcher_tui_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given an existing key and a replacement key that must remain secret.
    current_secret = "current-key-console-sentinel"
    replacement_secret = "replacement-key-console-sentinel"
    tui = _configured_tui(launcher_tui_module, monkeypatch, tmp_path, current_secret)
    monkeypatch.setattr(tui, "confirm_setting_change", lambda *_args: True)
    updated_keys: list[str] = []
    monkeypatch.setattr(
        launcher_tui_module.LauncherConfig,
        "update_proxy_api_key",
        lambda new_key: updated_keys.append(new_key),
    )
    key_prompt_kwargs: list[dict[str, bool]] = []

    def answer_prompt(prompt: str, **kwargs):
        if prompt == "Select option":
            return "3" if not updated_keys else "7"
        key_prompt_kwargs.append(kwargs)
        return replacement_secret

    monkeypatch.setattr(launcher_tui_module.Prompt, "ask", answer_prompt)

    # When the operator replaces the proxy key.
    tui.show_config_menu()

    # Then input is hidden, has no secret-bearing default, and output is redacted.
    assert updated_keys == [replacement_secret]
    assert key_prompt_kwargs == [{"password": True, "show_default": False}]
    rendered = tui.console.export_text()
    assert current_secret not in rendered
    assert replacement_secret not in rendered


def test_reset_generates_strong_random_key_without_rendering_fragments(
    launcher_tui_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given a reset flow with sentinel values for both old and generated keys.
    current_secret = "reset-current-key-console-sentinel"
    generated_secret = "reset-generated-key-console-sentinel"
    tui = _configured_tui(launcher_tui_module, monkeypatch, tmp_path, current_secret)
    warning_lines: list[str] = []
    updated_keys: list[str] = []
    token_sizes: list[int] = []

    def confirm_reset(_setting_name: str, lines: list[str]) -> bool:
        warning_lines.extend(lines)
        return True

    def generate_token(size: int) -> str:
        token_sizes.append(size)
        return generated_secret

    monkeypatch.setattr(tui, "confirm_setting_change", confirm_reset)
    monkeypatch.setattr(launcher_tui_module.secrets, "token_urlsafe", generate_token)
    monkeypatch.setattr(
        launcher_tui_module.LauncherConfig,
        "update_proxy_api_key",
        lambda new_key: updated_keys.append(new_key),
    )
    choices = iter(("6", "7"))
    monkeypatch.setattr(
        launcher_tui_module.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(choices),
    )

    # When all proxy settings are reset.
    tui.show_config_menu()

    # Then reset uses 256 bits of randomness and exposes only key status.
    assert token_sizes == [32]
    assert updated_keys == [generated_secret]
    rendered = "\n".join(warning_lines) + tui.console.export_text()
    assert "Proxy API Key        Set" in rendered
    assert current_secret not in rendered
    assert generated_secret not in rendered
