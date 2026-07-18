from __future__ import annotations

import logging
import os
from pathlib import Path
import stat
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.secure_logging import OwnerOnlyRotatingFileHandler  # noqa: E402


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_owner_only_rotating_handler_forces_rollover_retention_and_modes(
    tmp_path: Path,
) -> None:
    # Given: a legacy log and a permissive process umask.
    log_path = tmp_path / "proxy.log"
    log_path.write_text("legacy\n", encoding="utf-8")
    log_path.chmod(0o644)
    previous_umask = os.umask(0o000)
    handler = OwnerOnlyRotatingFileHandler(
        log_path,
        maxBytes=96,
        backupCount=2,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("runtime-log-rotation-test")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # When: enough bounded diagnostic records are emitted to force repeated rollover.
    try:
        for sequence in range(20):
            logger.info("sequence=%02d category=bounded-runtime-diagnostic", sequence)
        handler.flush()
    finally:
        handler.close()
        logger.handlers = []
        os.umask(previous_umask)

    # Then: the base plus two retained backups are owner-only and no older files remain.
    log_files = sorted(tmp_path.glob("proxy.log*"))
    assert [path.name for path in log_files] == ["proxy.log", "proxy.log.1", "proxy.log.2"]
    assert all(_mode(path) == 0o600 for path in log_files)
    assert all(path.stat().st_size <= 96 for path in log_files)


def test_proxy_runtime_wires_bounded_owner_only_application_logs() -> None:
    # Given: Mirrowel's runtime logging configuration.
    source = (SRC / "proxy_app" / "main.py").read_text(encoding="utf-8")

    # When/Then: both application sinks use fixed size and retention caps.
    assert "from rotator_library.secure_logging import OwnerOnlyRotatingFileHandler" in source
    assert source.count("OwnerOnlyRotatingFileHandler(") == 2
    assert source.count("maxBytes=5 * 1024 * 1024") == 2
    assert "backupCount=3" in source
    assert "backupCount=2" in source
    assert "OwnerOnlyFileHandler(LOG_DIR" not in source


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode contract")
def test_rollover_replaces_backup_symlink_without_touching_target(tmp_path: Path) -> None:
    # Given: the next backup name redirects to an external sentinel file.
    sentinel = tmp_path / "external.log"
    sentinel.write_text("sentinel", encoding="utf-8")
    sentinel.chmod(0o644)
    log_path = tmp_path / "proxy.log"
    log_path.write_text("active\n", encoding="utf-8")
    backup_path = tmp_path / "proxy.log.1"
    backup_path.symlink_to(sentinel)
    handler = OwnerOnlyRotatingFileHandler(
        log_path,
        maxBytes=1,
        backupCount=1,
        encoding="utf-8",
    )

    # When: rollover replaces the hostile backup entry.
    try:
        handler.doRollover()
    finally:
        handler.close()

    # Then: the external file is unchanged and retained logs are regular owner-only files.
    assert _mode(sentinel) == 0o644
    assert sentinel.read_text(encoding="utf-8") == "sentinel"
    assert not backup_path.is_symlink()
    assert backup_path.is_file()
    assert _mode(log_path) == 0o600
    assert _mode(backup_path) == 0o600
