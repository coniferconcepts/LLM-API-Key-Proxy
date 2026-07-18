from __future__ import annotations

import os
from pathlib import Path
import stat
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proxy_app import detailed_logger  # noqa: E402
from rotator_library import secure_log_domain  # noqa: E402


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.parametrize(
    ("body", "expected"),
    (
        ({"reasoning": "direct"}, "direct"),
        ({"choices": [{"message": {"reasoning": "nested"}}]}, "nested"),
        ({"choices": [{"message": {"reasoning_content": "content"}}]}, "content"),
        ({"reasoning": {"unexpected": "shape"}}, None),
        ({"choices": [{"message": {"reasoning": 7}}]}, None),
        ({"choices": "unexpected"}, None),
    ),
)
def test_raw_logger_extracts_only_string_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: dict[str, object],
    expected: str | None,
) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)

    assert detailed_logger.RawIOLogger()._extract_reasoning(body) == expected


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode contract")
def test_raw_logger_rejects_symlinked_domain_without_touching_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: raw_io redirects outside the verified owner-only logs directory.
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    external = tmp_path / "external"
    external.mkdir(mode=0o755)
    (logs_dir / "raw_io").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)

    # When: raw logging is enabled for a request.
    logger = detailed_logger.RawIOLogger()
    logger.log_request({"authorization": "synthetic"}, {"stream": False})

    # Then: no raw artifact escapes into the symlink target.
    assert list(external.iterdir()) == []
    assert logger._dir_available is False


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode contract")
def test_raw_logger_creates_owner_only_domain_and_artifacts_under_umask_022(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: a normal logs root and the conventional process umask.
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    previous_umask = os.umask(0o022)

    # When: every raw artifact type is written.
    try:
        logger = detailed_logger.RawIOLogger()
        logger.log_request({}, {"stream": True})
        logger.log_stream_chunk({"delta": "synthetic"})
        logger.log_final_response(200, {}, {"model": "synthetic"})
    finally:
        os.umask(previous_umask)

    # Then: directories are 0700 and files are 0600 regardless of umask.
    assert _mode(logs_dir / "raw_io") == 0o700
    assert _mode(logger.log_dir) == 0o700
    assert {path.name for path in logger.log_dir.iterdir()} == {
        "final_response.json",
        "metadata.json",
        "request.json",
        "streaming_chunks.jsonl",
    }
    assert all(_mode(path) == 0o600 for path in logger.log_dir.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode contract")
def test_raw_logger_rejects_replaced_request_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: a valid logger whose request directory is replaced by a symlink.
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    external = tmp_path / "external"
    external.mkdir(mode=0o755)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    logger = detailed_logger.RawIOLogger()
    logger.log_dir.rmdir()
    logger.log_dir.symlink_to(external, target_is_directory=True)

    # When: a request artifact is written through the stale logger instance.
    logger.log_request({}, {"stream": False})

    # Then: the replacement is rejected and the external target stays untouched.
    assert list(external.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode contract")
def test_raw_logger_rejects_symlinked_artifact_without_touching_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: a raw artifact redirects to a permissive external sentinel.
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    logger = detailed_logger.RawIOLogger()
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("sentinel", encoding="utf-8")
    sentinel.chmod(0o644)
    (logger.log_dir / "request.json").symlink_to(sentinel)

    # When: the logger writes the request artifact.
    logger.log_request({}, {"stream": False})

    # Then: neither content nor permissions escape through the symlink.
    assert sentinel.read_text(encoding="utf-8") == "sentinel"
    assert _mode(sentinel) == 0o644


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode contract")
def test_raw_logger_repairs_preexisting_permissive_regular_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: a regular artifact left with legacy permissive permissions.
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    logger = detailed_logger.RawIOLogger()
    request_path = logger.log_dir / "request.json"
    request_path.write_text("legacy", encoding="utf-8")
    request_path.chmod(0o644)

    # When: the raw logger replaces its content.
    logger.log_request({}, {"stream": False})

    # Then: the artifact is a regular owner-only file.
    assert request_path.is_file()
    assert not request_path.is_symlink()
    assert _mode(request_path) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode contract")
def test_raw_logger_recovers_after_unsafe_domain_is_removed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: initial raw_io setup is blocked by a hostile symlink.
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    external = tmp_path / "external"
    external.mkdir(mode=0o755)
    raw_io_path = logs_dir / "raw_io"
    raw_io_path.symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    logger = detailed_logger.RawIOLogger()
    raw_io_path.unlink()

    # When: the next request write retries after the unsafe entry is removed.
    logger.log_request({}, {"stream": False})

    # Then: logging recovers inside a newly secured owner-only domain.
    assert logger._dir_available is True
    assert (logger.log_dir / "request.json").is_file()
    assert _mode(raw_io_path) == 0o700
    assert _mode(logger.log_dir / "request.json") == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink and mode contract")
def test_raw_logger_rejects_artifact_replacement_during_descriptor_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: an attacker replaces request.json at the final descriptor-open boundary.
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    logger = detailed_logger.RawIOLogger()
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("sentinel", encoding="utf-8")
    original_open = secure_log_domain.os.open
    replacement_done = False

    def replacing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replacement_done
        if path == "request.json" and dir_fd is not None and not replacement_done:
            replacement_done = True
            (logger.log_dir / "request.json").symlink_to(sentinel)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(secure_log_domain, "_require_descriptor_security", lambda: None)
    monkeypatch.setattr(secure_log_domain.os, "open", replacing_open)

    # When: the logger reaches the raced artifact open.
    logger.log_request({}, {"stream": False})

    # Then: O_NOFOLLOW rejects the replacement without touching its target.
    assert replacement_done is True
    assert sentinel.read_text(encoding="utf-8") == "sentinel"
    assert logger._dir_available is False


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_raw_logger_repairs_permissive_domain_modes_before_each_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: an existing raw domain is made permissive after initialization.
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    logger = detailed_logger.RawIOLogger()
    raw_io_path = logs_dir / "raw_io"
    raw_io_path.chmod(0o755)
    logger.log_dir.chmod(0o755)

    # When: a streaming append reopens the descriptor-relative domain.
    logger.log_stream_chunk({"delta": "synthetic"})

    # Then: every descendant directory is restored to owner-only permissions.
    assert _mode(raw_io_path) == 0o700
    assert _mode(logger.log_dir) == 0o700
    assert _mode(logger.log_dir / "streaming_chunks.jsonl") == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_raw_logger_fails_closed_when_descriptor_chmod_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: owner-only descriptor permissions cannot be enforced on this platform.
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    logger = detailed_logger.RawIOLogger()
    raw_io_path = logs_dir / "raw_io"
    request_path = logger.log_dir / "request.json"
    stream_path = logger.log_dir / "streaming_chunks.jsonl"
    request_path.write_text("request-sentinel", encoding="utf-8")
    stream_path.write_text("stream-sentinel\n", encoding="utf-8")
    raw_io_path.chmod(0o755)
    logger.log_dir.chmod(0o755)
    request_path.chmod(0o644)
    stream_path.chmod(0o644)
    monkeypatch.delattr(secure_log_domain.os, "fchmod")

    # When: raw logging tries to create its storage domain.
    logger.log_request({}, {"stream": False})
    logger.log_stream_chunk({"delta": "replacement"})

    # Then: it writes nothing instead of accepting permissive fallback modes.
    assert logger._dir_available is False
    assert request_path.read_text(encoding="utf-8") == "request-sentinel"
    assert stream_path.read_text(encoding="utf-8") == "stream-sentinel\n"
    assert (_mode(raw_io_path), _mode(logger.log_dir)) == (0o755, 0o755)
    assert (_mode(request_path), _mode(stream_path)) == (0o644, 0o644)


@pytest.mark.skipif(os.name != "posix", reason="POSIX owner contract")
def test_raw_logger_rejects_domain_not_owned_by_effective_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: ownership checks observe a domain owned by a different effective user.
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    monkeypatch.setattr(os, "geteuid", lambda: logs_dir.stat().st_uid + 1)

    # When: raw logging tries to establish its verified domain.
    logger = detailed_logger.RawIOLogger()

    # Then: initialization fails closed without creating descendants.
    assert logger._dir_available is False
    assert not (logs_dir / "raw_io").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX special-file contract")
def test_raw_logger_rejects_fifo_artifact_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: the streaming artifact name is occupied by a FIFO.
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    logger = detailed_logger.RawIOLogger()
    stream_path = logger.log_dir / "streaming_chunks.jsonl"
    os.mkfifo(stream_path, mode=0o600)

    # When: a streaming append opens the artifact with nonblocking safeguards.
    logger.log_stream_chunk({"delta": "synthetic"})

    # Then: the special file is rejected and remains a FIFO.
    assert logger._dir_available is False
    assert stat.S_ISFIFO(stream_path.lstat().st_mode)
