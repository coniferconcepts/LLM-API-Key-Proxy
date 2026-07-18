from __future__ import annotations

import logging
import os
import stat
from io import BufferedWriter, FileIO, TextIOWrapper
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE_MODE = 0o600
LOG_DIRECTORY_MODE = 0o700


class UnsafeLogPathError(OSError):
    pass


def _optional_open_flags(*names: str) -> int:
    return sum(getattr(os, name, 0) for name in names)


def _supports_directory_descriptors() -> bool:
    return (
        os.open in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
        and hasattr(os, "O_DIRECTORY")
    )


def _verify_owned_directory_path(path: Path) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(path)
    get_effective_uid = getattr(os, "geteuid", None)
    if get_effective_uid is not None and metadata.st_uid != get_effective_uid():
        raise PermissionError(f"log directory is not owned by this process: {path}")


def _verify_owned_directory(descriptor: int, path: Path) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(path)
    get_effective_uid = getattr(os, "geteuid", None)
    if get_effective_uid is not None and metadata.st_uid != get_effective_uid():
        raise PermissionError(f"log directory is not owned by this process: {path}")


def _open_verified_directory(path: Path, *, create: bool = False) -> int:
    absolute_path = Path(os.path.abspath(path))
    flags = os.O_RDONLY | _optional_open_flags("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    descriptor = os.open(absolute_path.anchor, flags)
    try:
        for component in absolute_path.parts[1:]:
            if create:
                try:
                    os.mkdir(component, LOG_DIRECTORY_MODE, dir_fd=descriptor)
                except FileExistsError:
                    pass
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        _verify_owned_directory(descriptor, path)
    except OSError:
        os.close(descriptor)
        raise
    return descriptor


def secure_log_directory(path: Path) -> None:
    if _supports_directory_descriptors():
        descriptor = _open_verified_directory(path, create=True)
        try:
            _secure_descriptor(descriptor, LOG_DIRECTORY_MODE)
        finally:
            os.close(descriptor)
    else:
        path.mkdir(parents=True, mode=LOG_DIRECTORY_MODE, exist_ok=True)
        _verify_owned_directory_path(path)
        path.chmod(LOG_DIRECTORY_MODE)


def secure_log_file(path: Path) -> None:
    try:
        descriptor = _open_existing_regular_file(path)
    except (FileNotFoundError, IsADirectoryError, UnsafeLogPathError):
        return
    try:
        _secure_descriptor(descriptor, LOG_FILE_MODE)
    finally:
        os.close(descriptor)


def _secure_descriptor(descriptor: int, mode: int) -> None:
    try:
        os.fchmod(descriptor, mode)
    except AttributeError:
        return


def _verify_regular_file(descriptor: int, path: Path) -> None:
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise UnsafeLogPathError(f"log path is not a regular file: {path}")


def _open_existing_regular_file(path: Path) -> int:
    if path.is_symlink():
        raise UnsafeLogPathError(f"log path must not be a symlink: {path}")
    flags = os.O_RDONLY | _optional_open_flags("O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
    descriptor = os.open(path, flags)
    try:
        _verify_regular_file(descriptor, path)
    except OSError:
        os.close(descriptor)
        raise
    return descriptor


def _open_log_descriptor(path: Path) -> int:
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    flags |= _optional_open_flags("O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
    if _supports_directory_descriptors():
        directory_descriptor = _open_verified_directory(path.parent)
        try:
            descriptor = os.open(path.name, flags, LOG_FILE_MODE, dir_fd=directory_descriptor)
        finally:
            os.close(directory_descriptor)
    else:
        _verify_owned_directory_path(path.parent)
        if path.is_symlink():
            raise UnsafeLogPathError(f"log path must not be a symlink: {path}")
        descriptor = os.open(path, flags, LOG_FILE_MODE)
    try:
        _verify_regular_file(descriptor, path)
        _secure_descriptor(descriptor, LOG_FILE_MODE)
    except OSError:
        os.close(descriptor)
        raise
    return descriptor


def _open_log_stream(path: Path, encoding: str | None, errors: str | None) -> TextIOWrapper:
    descriptor = _open_log_descriptor(path)
    return TextIOWrapper(
        BufferedWriter(FileIO(descriptor, mode="w", closefd=True)),
        encoding=encoding,
        errors=errors,
    )


class OwnerOnlyFileHandler(logging.FileHandler):
    def _open(self) -> TextIOWrapper:
        return _open_log_stream(Path(self.baseFilename), self.encoding, self.errors)


class OwnerOnlyRotatingFileHandler(RotatingFileHandler):
    def _open(self) -> TextIOWrapper:
        return _open_log_stream(Path(self.baseFilename), self.encoding, self.errors)

    def doRollover(self) -> None:
        super().doRollover()
        base_path = Path(self.baseFilename)
        for path in base_path.parent.glob(f"{base_path.name}*"):
            secure_log_file(path)
