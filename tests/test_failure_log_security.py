from __future__ import annotations

import json
import logging
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library import failure_logger  # noqa: E402
from rotator_library.secure_logging import OwnerOnlyFileHandler  # noqa: E402
from rotator_library.utils.paths import get_logs_dir  # noqa: E402

SECRET = "synthetic-secret-never-persist"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_failure_log_persists_only_allowlisted_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o755)
    log_path = logs_dir / "failures.log"
    log_path.write_text("", encoding="utf-8")
    log_path.chmod(0o644)
    monkeypatch.setattr(failure_logger, "FAILURE_LOG_MAX_SIZE", 220)
    failure_logger.configure_failure_logger(logs_dir)

    try:
        for attempt in range(1, 8):
            error = RuntimeError(f"{SECRET}-exception-{attempt}")
            error.diagnostics = {"token": f"{SECRET}-diagnostic"}
            failure_logger.log_failure(
                api_key=f"{SECRET}-key",
                model=f"{SECRET}-model",
                attempt=attempt,
                error=error,
                request_headers={"Authorization": f"Bearer {SECRET}"},
                raw_response_text=f"{SECRET}-response-body",
            )
    finally:
        for handler in failure_logger.get_failure_logger().handlers:
            handler.close()
        failure_logger.configure_failure_logger(None)

    log_files = sorted(logs_dir.glob("failures.log*"))
    persisted = "".join(path.read_text(encoding="utf-8") for path in log_files)
    records = [json.loads(line) for path in log_files for line in path.read_text().splitlines()]

    assert SECRET not in persisted
    assert records
    assert all(
        set(record)
        <= {"schema_version", "timestamp", "attempt_number", "error_type", "status_code"}
        for record in records
    )
    if os.name == "posix":
        assert _mode(logs_dir) == 0o700
        assert all(_mode(path) == 0o600 for path in log_files)


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_log_directory_repairs_existing_files_under_permissive_umask(tmp_path: Path) -> None:
    previous_umask = os.umask(0o022)
    try:
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(mode=0o755)
        existing = logs_dir / "proxy.log"
        rotated = logs_dir / "proxy.log.1"
        unrelated = logs_dir / "notes.txt"
        for path in (existing, rotated, unrelated):
            path.write_text("fixture", encoding="utf-8")
            path.chmod(0o644)

        result = get_logs_dir(tmp_path)
    finally:
        os.umask(previous_umask)

    assert result == logs_dir
    assert _mode(logs_dir) == 0o700
    assert _mode(existing) == 0o600
    assert _mode(rotated) == 0o600
    assert _mode(unrelated) == 0o644


def test_owner_only_handler_opens_when_fchmod_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: a platform without os.fchmod, such as native Windows.
    monkeypatch.delattr(os, "fchmod", raising=False)
    log_path = tmp_path / "portable.log"

    # When: the secure handler creates and writes its log file.
    handler = OwnerOnlyFileHandler(log_path, encoding="utf-8")
    try:
        handler.emit(logging.makeLogRecord({"msg": "portable"}))
        handler.flush()
    finally:
        handler.close()

    # Then: logging succeeds without relying on the unavailable API.
    assert log_path.read_text(encoding="utf-8").strip() == "portable"


def test_configure_failure_logger_closes_old_handler_and_target(tmp_path: Path) -> None:
    # Given: the singleton logger is already writing to one target.
    first_logs_dir = tmp_path / "first"
    second_logs_dir = tmp_path / "second"
    failure_logger.configure_failure_logger(first_logs_dir)
    old_logger = failure_logger.get_failure_logger()
    old_handler = old_logger.handlers[0]
    failure_logger.log_failure("ignored", "ignored", 1, RuntimeError("safe"), {})

    # When: the singleton is reconfigured to another target.
    failure_logger.configure_failure_logger(second_logs_dir)
    failure_logger.log_failure("ignored", "ignored", 2, RuntimeError("safe"), {})

    # Then: the old descriptor is closed and subsequent records reach only the new target.
    try:
        assert old_handler.stream is None
        first_records = (first_logs_dir / "failures.log").read_text(encoding="utf-8")
        second_records = (second_logs_dir / "failures.log").read_text(encoding="utf-8")
        assert '"attempt_number": 1' in first_records
        assert '"attempt_number": 2' not in first_records
        assert '"attempt_number": 2' in second_records
    finally:
        failure_logger.configure_failure_logger(None)


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode contract")
def test_log_directory_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    # Given: the configured logs path redirects to an external sentinel directory.
    sentinel_dir = tmp_path / "sentinel"
    sentinel_dir.mkdir(mode=0o755)
    sentinel_file = sentinel_dir / "external.log"
    sentinel_file.write_text("sentinel", encoding="utf-8")
    sentinel_file.chmod(0o644)
    (tmp_path / "logs").symlink_to(sentinel_dir, target_is_directory=True)

    # When: path setup is asked to secure the redirected logs directory.
    with pytest.raises(OSError):
        get_logs_dir(tmp_path)

    # Then: neither the external directory nor its content or mode is changed.
    assert _mode(sentinel_dir) == 0o755
    assert _mode(sentinel_file) == 0o644
    assert sentinel_file.read_text(encoding="utf-8") == "sentinel"


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode contract")
def test_owner_only_handler_rejects_symlink_without_touching_target(tmp_path: Path) -> None:
    # Given: the requested active log redirects to an external sentinel file.
    sentinel = tmp_path / "external.log"
    sentinel.write_text("sentinel", encoding="utf-8")
    sentinel.chmod(0o644)
    log_path = tmp_path / "runtime.log"
    log_path.symlink_to(sentinel)

    # When: the secure handler attempts to open the redirected log path.
    with pytest.raises(OSError):
        OwnerOnlyFileHandler(log_path, encoding="utf-8")

    # Then: the external file's content and mode remain unchanged.
    assert _mode(sentinel) == 0o644
    assert sentinel.read_text(encoding="utf-8") == "sentinel"


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode contract")
def test_owner_only_handler_rejects_symlinked_ancestor_without_touching_target(
    tmp_path: Path,
) -> None:
    # Given: an ancestor of the requested log path redirects to an external directory.
    sentinel_dir = tmp_path / "external"
    sentinel_dir.mkdir(mode=0o755)
    sentinel_logs = sentinel_dir / "logs"
    sentinel_logs.mkdir(mode=0o755)
    sentinel_file = sentinel_logs / "runtime.log"
    sentinel_file.write_text("sentinel", encoding="utf-8")
    sentinel_file.chmod(0o644)
    redirected_parent = tmp_path / "redirected"
    redirected_parent.symlink_to(sentinel_dir, target_is_directory=True)

    # When: the secure handler walks through the redirected ancestor.
    with pytest.raises(OSError):
        OwnerOnlyFileHandler(redirected_parent / "logs" / "runtime.log", encoding="utf-8")

    # Then: the external file is neither opened nor permission-mutated.
    assert _mode(sentinel_file) == 0o644
    assert sentinel_file.read_text(encoding="utf-8") == "sentinel"
