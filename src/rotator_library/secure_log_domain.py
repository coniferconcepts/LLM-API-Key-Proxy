from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat

DIRECTORY_MODE = 0o700
FILE_MODE = 0o600


class UnsafeLogDomainError(OSError):
    pass


def _open_flags(*names: str) -> int:
    return sum(getattr(os, name, 0) for name in names)


def _require_descriptor_security() -> None:
    if (
        os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "fchmod")
    ):
        raise UnsafeLogDomainError("secure raw logging is unsupported on this platform")


def _validate_component(component: str) -> None:
    if component in {"", ".", ".."} or Path(component).name != component:
        raise UnsafeLogDomainError("raw log path contains an invalid component")


def _verify_directory(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError("raw log path component is not a directory")
    get_effective_uid = getattr(os, "geteuid", None)
    if get_effective_uid is not None and metadata.st_uid != get_effective_uid():
        raise PermissionError("raw log directory is not owned by this process")


def _verify_file(descriptor: int) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise UnsafeLogDomainError("raw log artifact is not a regular file")
    get_effective_uid = getattr(os, "geteuid", None)
    if get_effective_uid is not None and metadata.st_uid != get_effective_uid():
        raise PermissionError("raw log artifact is not owned by this process")


def _secure_mode(descriptor: int, mode: int) -> None:
    os.fchmod(descriptor, mode)


def _open_root(path: Path) -> int:
    _require_descriptor_security()
    absolute_path = Path(os.path.abspath(path))
    flags = os.O_RDONLY | _open_flags("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    descriptor = os.open(absolute_path.anchor, flags)
    try:
        for component in absolute_path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        _verify_directory(descriptor)
    except OSError:
        os.close(descriptor)
        raise
    return descriptor


def _open_child_directory(parent_descriptor: int, component: str) -> int:
    _validate_component(component)
    try:
        os.mkdir(component, DIRECTORY_MODE, dir_fd=parent_descriptor)
    except FileExistsError:
        pass
    flags = os.O_RDONLY | _open_flags("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    descriptor = os.open(component, flags, dir_fd=parent_descriptor)
    try:
        _verify_directory(descriptor)
        _secure_mode(descriptor, DIRECTORY_MODE)
    except OSError:
        os.close(descriptor)
        raise
    return descriptor


@dataclass(frozen=True, slots=True)
class OwnerOnlyLogDomain:
    root: Path
    components: tuple[str, ...]

    def ensure(self) -> None:
        descriptor = self._open_directory()
        os.close(descriptor)

    def _open_directory(self) -> int:
        descriptor = _open_root(self.root)
        try:
            for component in self.components:
                next_descriptor = _open_child_directory(descriptor, component)
                os.close(descriptor)
                descriptor = next_descriptor
        except OSError:
            os.close(descriptor)
            raise
        return descriptor

    def write_text(self, filename: str, content: str, *, append: bool) -> None:
        _validate_component(filename)
        directory_descriptor = self._open_directory()
        flags = os.O_WRONLY | os.O_CREAT
        flags |= _open_flags("O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
        if append:
            flags |= os.O_APPEND
        try:
            descriptor = os.open(
                filename,
                flags,
                FILE_MODE,
                dir_fd=directory_descriptor,
            )
        finally:
            os.close(directory_descriptor)
        try:
            _verify_file(descriptor)
            _secure_mode(descriptor, FILE_MODE)
            if not append:
                os.ftruncate(descriptor, 0)
            stream = os.fdopen(descriptor, "a" if append else "w", encoding="utf-8")
            descriptor = -1
            with stream:
                stream.write(content)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
