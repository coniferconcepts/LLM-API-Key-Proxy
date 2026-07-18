from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Final, Literal

DisconnectCheck = Callable[[], Awaitable[bool]]
TerminalField = tuple[str, str]


@dataclass(frozen=True, slots=True)
class SSEProtocolTerminalMissing(Exception):
    protocol: Literal["openai", "anthropic"]

    def __str__(self) -> str:
        return f"{self.protocol} response stream ended before its protocol terminal"


class _SSERecordTerminalTracker:
    __slots__ = (
        "_after_cr",
        "_exact_field_cardinality",
        "_line",
        "_lines",
        "_protocol",
        "_record_size",
        "_target_names",
        "_terminals",
    )

    _MAX_RECORD_LENGTH: Final = 16 * 1024 * 1024

    def __init__(
        self,
        terminals: frozenset[TerminalField],
        *,
        exact_field_cardinality: bool,
        protocol: Literal["openai", "anthropic"],
    ) -> None:
        self._after_cr = False
        self._exact_field_cardinality = exact_field_cardinality
        self._line: list[str] = []
        self._lines: list[str] = []
        self._protocol = protocol
        self._record_size = 0
        self._target_names = frozenset(field for field, _ in terminals)
        self._terminals = terminals

    def feed(self, chunk: str) -> Iterator[tuple[str, bool]]:
        for character in chunk:
            if character == "\n" and self._after_cr:
                self._after_cr = False
                continue
            if character in "\r\n":
                self._after_cr = character == "\r"
                if self._line:
                    self._lines.append("".join(self._line))
                    self._line.clear()
                    continue
                record = "\n".join(self._lines) + "\n\n"
                terminal = self._is_terminal_record()
                self._lines.clear()
                self._record_size = 0
                if record != "\n\n":
                    yield record, terminal
                if terminal:
                    return
                continue
            self._after_cr = False
            self._line.append(character)
            self._record_size += 1
            if self._record_size > self._MAX_RECORD_LENGTH:
                raise SSEProtocolTerminalMissing(self._protocol)

    def _is_terminal_record(self) -> bool:
        record_match = False
        target_count = 0
        for line in self._lines:
            field, value = _sse_field(line)
            if field in self._target_names:
                target_count += 1
                record_match = (field, value) in self._terminals
        return record_match and (not self._exact_field_cardinality or target_count == 1)


_OPENAI_TERMINALS: Final = frozenset({("data", "[DONE]")})
_ANTHROPIC_TERMINALS: Final = frozenset({("event", "error"), ("event", "message_stop")})


def _sse_field(line: str) -> TerminalField:
    field, separator, value = line.partition(":")
    if not separator:
        return field, ""
    return field, value[1:] if value.startswith(" ") else value


def sse_data_content(record: str) -> str | None:
    values: list[str] = []
    for line in record.splitlines():
        field, value = _sse_field(line)
        if field == "data":
            values.append(value)
    return "\n".join(values) if values else None


async def require_openai_terminal(
    source: AsyncGenerator[str, None],
    is_disconnected: DisconnectCheck | None = None,
) -> AsyncGenerator[str, None]:
    tracker = _SSERecordTerminalTracker(
        _OPENAI_TERMINALS, exact_field_cardinality=True, protocol="openai"
    )
    async for chunk in _require_terminal(source, "openai", tracker, is_disconnected):
        yield chunk


async def require_anthropic_terminal(
    source: AsyncGenerator[str, None],
    is_disconnected: DisconnectCheck | None = None,
) -> AsyncGenerator[str, None]:
    tracker = _SSERecordTerminalTracker(
        _ANTHROPIC_TERMINALS, exact_field_cardinality=False, protocol="anthropic"
    )
    async for chunk in _require_terminal(source, "anthropic", tracker, is_disconnected):
        yield chunk


async def _require_terminal(
    source: AsyncGenerator[str, None],
    protocol: Literal["openai", "anthropic"],
    tracker: _SSERecordTerminalTracker,
    is_disconnected: DisconnectCheck | None,
) -> AsyncGenerator[str, None]:
    iterator = source.__aiter__()
    while True:
        try:
            chunk = await iterator.__anext__()
        except StopAsyncIteration:
            break
        except Exception:
            if is_disconnected is not None and await is_disconnected():
                return
            raise
        if is_disconnected is not None and await is_disconnected():
            return
        for record, terminal in tracker.feed(chunk):
            yield record
            if terminal:
                return
    if is_disconnected is not None and await is_disconnected():
        return
    raise SSEProtocolTerminalMissing(protocol)
