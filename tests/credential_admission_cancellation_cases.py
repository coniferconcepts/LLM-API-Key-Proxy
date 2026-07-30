from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from credential_admission_contract_support import (
    _CREDENTIAL,
    _MODEL,
    active_count,
    make_client,
    observe_real_admission_wait,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("iteration", range(5))
async def test_client_cancellation_releases_permit_and_next_request_recovers(
    tmp_path: Path,
    iteration: int,
) -> None:
    client, manager = make_client(tmp_path, acquire_timeout=2.0)
    holder_entered_api = asyncio.Event()
    hold_request_open = asyncio.Event()
    holder: asyncio.Task[object] | None = None
    waiter: asyncio.Task[object] | None = None

    async def blocking_api(**_kwargs: object) -> object:
        holder_entered_api.set()
        await hold_request_open.wait()
        return object()

    async def unexpected_api(**_kwargs: object) -> object:
        pytest.fail("cancelled waiter unexpectedly reached the provider API")

    try:
        holder = asyncio.create_task(
            client._execute_with_retry(
                blocking_api,
                request=None,
                model=_MODEL,
                messages=[{"role": "user", "content": "hold"}],
            )
        )
        await asyncio.wait_for(holder_entered_api.wait(), timeout=1.0)
        waiter_blocked = observe_real_admission_wait(manager)
        waiter = asyncio.create_task(
            client._execute_with_retry(
                unexpected_api,
                request=None,
                model=_MODEL,
                messages=[{"role": "user", "content": "wait"}],
            )
        )
        await asyncio.wait_for(waiter_blocked.wait(), timeout=1.0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert active_count(manager) == 1

        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder
        assert active_count(manager) == 0

        async def successful_api(**_kwargs: object) -> dict[str, int]:
            return {"iteration": iteration}

        response = await asyncio.wait_for(
            client._execute_with_retry(
                successful_api,
                request=None,
                model=_MODEL,
                messages=[{"role": "user", "content": "recover"}],
            ),
            timeout=1.0,
        )
        assert response == {"iteration": iteration}
        assert active_count(manager) == 0
    finally:
        pending = [task for task in (waiter, holder) if task is not None and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_interrupt_permit_release(tmp_path: Path) -> None:
    client, manager = make_client(tmp_path, acquire_timeout=1.0)
    holder_entered_api = asyncio.Event()
    hold_request_open = asyncio.Event()
    release_started = asyncio.Event()

    async def blocking_api(**_kwargs: object) -> object:
        holder_entered_api.set()
        await hold_request_open.wait()
        return object()

    original_release = manager.release_key

    async def observed_release(key: str, model: str) -> None:
        release_started.set()
        await original_release(key, model)

    manager.release_key = observed_release  # type: ignore[method-assign]
    holder = asyncio.create_task(
        client._execute_with_retry(
            blocking_api,
            request=None,
            model=_MODEL,
            messages=[{"role": "user", "content": "hold"}],
        )
    )
    await asyncio.wait_for(holder_entered_api.wait(), timeout=1.0)
    state_lock = manager.key_states[_CREDENTIAL]["lock"]
    await state_lock.acquire()

    try:
        holder.cancel()
        await asyncio.wait_for(release_started.wait(), timeout=1.0)
        holder.cancel()
    finally:
        state_lock.release()

    with pytest.raises(asyncio.CancelledError):
        await holder
    assert active_count(manager) == 0


@pytest.mark.asyncio
async def test_upstream_exception_releases_permit(tmp_path: Path) -> None:
    client, manager = make_client(tmp_path, acquire_timeout=1.0)

    async def failing_api(**_kwargs: object) -> object:
        raise ConnectionError("synthetic upstream disconnect")

    assert active_count(manager) == 0
    failure_response = await client._execute_with_retry(
        failing_api,
        request=None,
        model=_MODEL,
        messages=[{"role": "user", "content": "fail"}],
    )
    assert failure_response["error"]["status"] == 503
    assert failure_response["error"]["code"] == "all_credentials_exhausted"
    assert active_count(manager) == 0
