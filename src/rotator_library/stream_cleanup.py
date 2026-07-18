from __future__ import annotations

from typing import Protocol

import anyio


class AsyncClosable(Protocol):
    async def aclose(self) -> None: ...


async def close_stream_with_timeout(
    source: AsyncClosable,
    timeout_seconds: float,
) -> bool:
    with anyio.CancelScope(shield=True):
        with anyio.move_on_after(timeout_seconds) as cleanup_scope:
            await source.aclose()
    return not cleanup_scope.cancel_called
