from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Final, Literal

import anyio

from rotator_library.stream_cleanup import close_stream_with_timeout


@dataclass(frozen=True, slots=True)
class SSEStreamPolicy:
    max_bytes: int
    max_events: int
    idle_timeout_seconds: float
    total_timeout_seconds: float
    cleanup_timeout_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class SSEStreamLimitExceeded(Exception):
    boundary: Literal["bytes", "events"]

    def __str__(self) -> str:
        return f"response stream exceeded the {self.boundary} boundary"


@dataclass(frozen=True, slots=True)
class SSEStreamTimedOut(Exception):
    boundary: Literal["idle", "total"]

    def __str__(self) -> str:
        return f"response stream exceeded the {self.boundary} timeout"


class SSEStreamCapacityExceeded(Exception):
    def __str__(self) -> str:
        return "response stream capacity is exhausted"


DEFAULT_SSE_STREAM_POLICY: Final = SSEStreamPolicy(
    max_bytes=16 * 1024 * 1024,
    max_events=10_000,
    idle_timeout_seconds=30.0,
    total_timeout_seconds=900.0,
)
DEFAULT_SSE_STREAM_CAPACITY: Final = anyio.CapacityLimiter(64)
SSE_TERMINAL_RESYNC: Final = "\n\n"

OpenAIStreamPolicy = SSEStreamPolicy
OpenAIStreamLimitExceeded = SSEStreamLimitExceeded
OpenAIStreamTimedOut = SSEStreamTimedOut
OpenAIStreamCapacityExceeded = SSEStreamCapacityExceeded
DEFAULT_OPENAI_STREAM_POLICY = DEFAULT_SSE_STREAM_POLICY
DEFAULT_OPENAI_STREAM_CAPACITY = DEFAULT_SSE_STREAM_CAPACITY


@dataclass(slots=True)
class _ProducerState:
    error: Exception | None = None


class _SSEDataFieldCounter:
    __slots__ = ("_line_state",)

    _FIELD_NAME: Final = "data"
    _NOT_DATA: Final = -1
    _COUNTED: Final = -2

    def __init__(self) -> None:
        self._line_state = 0

    def feed(self, chunk: str) -> int:
        fields = 0
        for character in chunk:
            if character in "\r\n":
                if self._line_state == len(self._FIELD_NAME):
                    fields += 1
                self._line_state = 0
            elif self._line_state < 0:
                continue
            elif self._line_state < len(self._FIELD_NAME):
                expected = self._FIELD_NAME[self._line_state]
                self._line_state = self._line_state + 1 if character == expected else self._NOT_DATA
            elif character == ":":
                fields += 1
                self._line_state = self._COUNTED
            else:
                self._line_state = self._NOT_DATA
        return fields


async def bounded_sse_stream(
    source: AsyncGenerator[str, None],
    policy: SSEStreamPolicy,
    capacity: anyio.CapacityLimiter,
) -> AsyncGenerator[str, None]:
    state = _ProducerState()
    send_stream, receive_stream = anyio.create_memory_object_stream[str](1)

    async def produce() -> None:
        acquired = False
        try:
            try:
                capacity.acquire_on_behalf_of_nowait(source)
                acquired = True
            except anyio.WouldBlock:
                state.error = SSEStreamCapacityExceeded()
                return
            byte_count = 0
            data_field_count = 0
            data_fields = _SSEDataFieldCounter()
            with anyio.fail_after(policy.total_timeout_seconds):
                iterator = source.__aiter__()
                while True:
                    try:
                        with anyio.fail_after(policy.idle_timeout_seconds):
                            chunk = await iterator.__anext__()
                    except StopAsyncIteration:
                        return
                    except TimeoutError:
                        raise SSEStreamTimedOut("idle") from None
                    byte_count += len(chunk.encode("utf-8"))
                    if byte_count > policy.max_bytes:
                        raise SSEStreamLimitExceeded("bytes")
                    data_field_count += data_fields.feed(chunk)
                    if data_field_count > policy.max_events:
                        raise SSEStreamLimitExceeded("events")
                    await send_stream.send(chunk)
        except TimeoutError:
            state.error = SSEStreamTimedOut("total")
        except Exception as exc:
            state.error = exc
        finally:
            try:
                await close_stream_with_timeout(source, policy.cleanup_timeout_seconds)
            except Exception as exc:
                if state.error is None:
                    state.error = exc
            finally:
                if acquired:
                    capacity.release_on_behalf_of(source)
                send_stream.close()

    task_group = anyio.create_task_group()
    await task_group.__aenter__()
    async with receive_stream:
        task_group.start_soon(produce)
        try:
            while True:
                try:
                    chunk = await receive_stream.receive()
                except anyio.EndOfStream:
                    if state.error is not None:
                        raise state.error
                    return
                yield chunk
        finally:
            task_group.cancel_scope.cancel()
            await task_group.__aexit__(None, None, None)


bounded_openai_sse_stream = bounded_sse_stream
