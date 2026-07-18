from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from deployment_surface_helpers import BOOTSTRAP_ENV, COMPOSE_FILES, DOCKERFILE, ROOT, read as _read

TUI_MODULE_NAMES = ("proxy_app", "proxy_app.launcher_tui")


@pytest.fixture
def _tui_process_state_is_restored() -> Iterator[None]:
    original_argv = sys.argv
    original_argv_values = original_argv.copy()
    original_modules = {name: sys.modules[name] for name in TUI_MODULE_NAMES if name in sys.modules}
    original_modules = {name: sys.modules[name] for name in TUI_MODULE_NAMES if name in sys.modules}
    try:
        yield
        assert sys.argv is original_argv
        assert sys.argv == original_argv_values
        assert {name for name in TUI_MODULE_NAMES if name in sys.modules} == set(original_modules)
        assert all(sys.modules[name] is module for name, module in original_modules.items())
    finally:
        sys.argv = original_argv
        original_argv[:] = original_argv_values
        for name in TUI_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(original_modules)


def test_docker_published_port_has_explicit_network_bind_approval() -> None:
    source = _read(DOCKERFILE)

    assert "ENV MIRROWEL_ALLOW_NETWORK_BIND=true" in source
    assert (
        'CMD ["python", "src/proxy_app/main.py", "--host", "0.0.0.0", "--port", "8000"]' in source
    )
    assert "ENV MIRROWEL_ALLOWED_HOSTS=" not in source
    for compose_file in COMPOSE_FILES:
        compose_source = _read(compose_file)
        assert "MIRROWEL_ALLOWED_HOSTS=${MIRROWEL_ALLOWED_HOSTS:?" in compose_source
        assert "127.0.0.1:${PORT:-8000}:8000" in compose_source


def test_tui_network_bind_records_approval_in_launch_environment(
    _tui_process_state_is_restored: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROXY_API_KEY", "test-token")
    monkeypatch.delenv("MIRROWEL_ALLOW_NETWORK_BIND", raising=False)
    monkeypatch.delenv("MIRROWEL_ALLOWED_HOSTS", raising=False)
    original_argv = sys.argv
    original_argv_values = original_argv.copy()
    original_modules = {name: sys.modules[name] for name in TUI_MODULE_NAMES if name in sys.modules}
    sys.path.insert(0, str(ROOT / "src"))
    try:
        sys.modules.pop("proxy_app.launcher_tui", None)
        launcher_tui = importlib.import_module("proxy_app.launcher_tui")
        tui = launcher_tui.LauncherTUI()
        tui.env_file.touch()
        tui.config.config.update(
            {
                "host": "0.0.0.0",
                "port": 8000,
                "enable_request_logging": False,
                "enable_raw_logging": False,
            }
        )
        monkeypatch.setattr(tui, "confirm_setting_change", lambda *_args: True)
        monkeypatch.setattr(
            launcher_tui.Prompt,
            "ask",
            lambda *_args, **_kwargs: "proxy.example.com,192.0.2.10",
        )
        monkeypatch.setattr(launcher_tui, "clear_screen", lambda: None)
        monkeypatch.setattr("time.sleep", lambda _seconds: None)

        tui.run_proxy()

        assert os.environ["MIRROWEL_ALLOW_NETWORK_BIND"] == "true"
        assert os.environ["MIRROWEL_ALLOWED_HOSTS"] == ("proxy.example.com,192.0.2.10")
        assert sys.argv == ["main.py", "--host", "0.0.0.0", "--port", "8000"]
    finally:
        sys.path.remove(str(ROOT / "src"))
        sys.argv = original_argv
        original_argv[:] = original_argv_values
        for name in TUI_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(original_modules)


def test_tui_network_bind_approval_survives_conflicting_dotenv_reload(
    _tui_process_state_is_restored: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: the operator approves a public bind before stale dotenv values appear.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROXY_API_KEY", "test-token")
    monkeypatch.delenv("MIRROWEL_ALLOW_NETWORK_BIND", raising=False)
    monkeypatch.delenv("MIRROWEL_ALLOWED_HOSTS", raising=False)
    (tmp_path / ".env").touch()
    original_argv = sys.argv
    original_argv_values = original_argv.copy()
    original_modules = {name: sys.modules[name] for name in TUI_MODULE_NAMES if name in sys.modules}
    sys.path.insert(0, str(ROOT / "src"))
    try:
        sys.modules.pop("proxy_app.launcher_tui", None)
        launcher_tui = importlib.import_module("proxy_app.launcher_tui")
        tui = launcher_tui.LauncherTUI()
        tui.config.config.update(
            {
                "host": "0.0.0.0",
                "port": 8000,
                "enable_request_logging": False,
                "enable_raw_logging": False,
            }
        )
        monkeypatch.setattr(tui, "confirm_setting_change", lambda *_args: True)
        monkeypatch.setattr(
            launcher_tui.Prompt,
            "ask",
            lambda *_args, **_kwargs: "approved.example",
        )
        monkeypatch.setattr(launcher_tui, "clear_screen", lambda: None)
        monkeypatch.setattr("time.sleep", lambda _seconds: None)

        # When: a later dotenv load conflicts and the typed TUI approval is reapplied.
        tui.run_proxy()
        tui.env_file.write_text(
            "MIRROWEL_ALLOW_NETWORK_BIND=false\nMIRROWEL_ALLOWED_HOSTS=stale.example\n",
            encoding="utf-8",
        )
        launcher_tui.load_dotenv(tui.env_file, override=True)
        assert tui.network_bind_approval is not None
        tui.network_bind_approval.apply_to_environment()

        # Then: the operator-approved values, not stale dotenv values, form runtime policy.
        assert os.environ["MIRROWEL_ALLOW_NETWORK_BIND"] == "true"
        assert os.environ["MIRROWEL_ALLOWED_HOSTS"] == "approved.example"
    finally:
        sys.path.remove(str(ROOT / "src"))
        sys.argv = original_argv
        original_argv[:] = original_argv_values
        for name in TUI_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(original_modules)


def test_main_reapplies_tui_network_bind_approval_after_dotenv_loading() -> None:
    # Given: main's ordered startup source.
    source = (ROOT / "src" / "proxy_app" / "main.py").read_text(encoding="utf-8")
    bootstrap_source = BOOTSTRAP_ENV.read_text(encoding="utf-8")

    # When: the final dotenv load, typed override, and policy freeze are located.
    loader_start = bootstrap_source.index("def load_router_env(")
    loader_source = bootstrap_source[loader_start:]
    dotenv_load = loader_source.index("load_dotenv(env_file, override=True)")
    approval_reapply = loader_source.index("network_bind_approval.apply_to_environment()")
    security_freeze = source.index("_runtime_security_config = _build_runtime_security_config()")

    # Then: approval overrides dotenv before the immutable runtime policy is built.
    assert dotenv_load < approval_reapply
    assert (
        source.index("_load_router_env(_main_env_file, _tui_network_bind_approval)")
        < security_freeze
    )


def test_tui_network_bind_cancellation_does_not_form_launch_argv(
    _tui_process_state_is_restored: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROXY_API_KEY", "test-token")
    monkeypatch.delenv("MIRROWEL_ALLOW_NETWORK_BIND", raising=False)
    monkeypatch.delenv("MIRROWEL_ALLOWED_HOSTS", raising=False)
    original_argv = sys.argv
    original_argv_values = original_argv.copy()
    original_modules = {name: sys.modules[name] for name in TUI_MODULE_NAMES if name in sys.modules}
    sys.path.insert(0, str(ROOT / "src"))
    try:
        sys.modules.pop("proxy_app.launcher_tui", None)
        launcher_tui = importlib.import_module("proxy_app.launcher_tui")
        tui = launcher_tui.LauncherTUI()
        tui.env_file.touch()
        tui.config.config["host"] = "0.0.0.0"
        monkeypatch.setattr(tui, "confirm_setting_change", lambda *_args: False)

        tui.run_proxy()

        assert "MIRROWEL_ALLOW_NETWORK_BIND" not in os.environ
        assert "MIRROWEL_ALLOWED_HOSTS" not in os.environ
        assert sys.argv == original_argv_values
    finally:
        sys.path.remove(str(ROOT / "src"))
        sys.argv = original_argv
        original_argv[:] = original_argv_values
        for name in TUI_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(original_modules)
