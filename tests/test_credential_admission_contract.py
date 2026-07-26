from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rotator_library.client import RotatingClient
from rotator_library.error_handler import NoAvailableKeysError, build_public_stream_error
from rotator_library.usage_manager import UsageManager, lib_logger

_CREDENTIAL = "synthetic-credential"
_MODEL = "synthetic/model"


class _AvailableCooldownManager:
    async def is_cooling_down(self, _provider: str) -> bool:
        return False


class _IdentityProviderConfig:
    def convert_for_litellm(self, **kwargs: object) -> dict[str, object]:
        return dict(kwargs)


class _PriorityProvider:
    def get_credential_priority(self, _credential: str) -> int:
        return 1

    def get_credential_tier_name(self, _credential: str) -> str:
        return "synthetic"

    def has_custom_logic(self) -> bool:
        return False


def _active_count(manager: UsageManager) -> int:
    state = manager.key_states.get(_CREDENTIAL)
    return 0 if state is None else int(state["models_in_use"].get(_MODEL, 0))


def _make_client(tmp_path: Path, *, acquire_timeout: float) -> tuple[RotatingClient, UsageManager]:
    manager = UsageManager(str(tmp_path / "usage.json"))
    manager._usage_data = {}  # noqa: SLF001
    manager._initialized.set()  # noqa: SLF001

    async def record_success(
        _credential: str,
        _model: str,
        _usage: dict[str, object] | None = None,
    ) -> None:
        return None

    manager.record_success = record_success  # type: ignore[method-assign]

    client = object.__new__(RotatingClient)
    client.all_credentials = {"synthetic": [_CREDENTIAL]}
    client.global_timeout = 10.0
    client.acquire_timeout = acquire_timeout
    client.enable_request_logging = False
    client.max_concurrent_requests_per_key = {"synthetic": 1}
    client.cooldown_manager = _AvailableCooldownManager()
    client.usage_manager = manager
    client._apply_routing_policy = lambda model: (model, None)
    client._log_route_decision = lambda _decision: None
    client._get_provider_instance = lambda _provider: _PriorityProvider()
    client._resolve_model_id = lambda model, _provider: model
    client._apply_default_safety_settings = lambda _provider, _kwargs: None
    client.litellm_provider_params = {}
    client.max_retries = 1
    client.oauth_providers = set()
    client.provider_config = _IdentityProviderConfig()
    client.abort_on_callback_error = True
    client._litellm_logger_callback = lambda *_args, **_kwargs: None
    return client, manager


def _observe_real_admission_wait(manager: UsageManager) -> asyncio.Event:
    condition = manager.key_states[_CREDENTIAL]["condition"]
    original_wait = condition.wait
    blocked = asyncio.Event()

    async def observed_wait() -> bool:
        blocked.set()
        return await original_wait()

    condition.wait = observed_wait
    return blocked


@pytest.mark.asyncio
async def test_busy_admission_has_a_bounded_proxy_busy_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, manager = _make_client(tmp_path, acquire_timeout=0.05)
    holder_entered_api = asyncio.Event()
    hold_request_open = asyncio.Event()
    original_log_level = lib_logger.level
    lib_logger.addHandler(caplog.handler)
    lib_logger.setLevel("INFO")

    async def blocking_api(**_kwargs: object) -> object:
        holder_entered_api.set()
        await hold_request_open.wait()
        return object()

    async def unexpected_api(**_kwargs: object) -> object:
        pytest.fail("busy waiter unexpectedly reached the provider API")

    # Given a real client request owns the only credential permit.
    holder = asyncio.create_task(
        client._execute_with_retry(
            blocking_api,
            request=None,
            model=_MODEL,
            messages=[{"role": "user", "content": "hold"}],
        )
    )
    try:
        await asyncio.wait_for(holder_entered_api.wait(), timeout=1.0)
        waiter_blocked = _observe_real_admission_wait(manager)

        # When a second request reaches the real admission condition.
        started = asyncio.get_running_loop().time()
        with pytest.raises(NoAvailableKeysError) as captured:
            await asyncio.wait_for(
                client._execute_with_retry(
                    unexpected_api,
                    request=None,
                    model=_MODEL,
                    messages=[{"role": "user", "content": "wait"}],
                ),
                timeout=0.5,
            )
        elapsed = asyncio.get_running_loop().time() - started

        # Then the configured admission deadline wins before the outer safety bound.
        assert waiter_blocked.is_set()
        assert captured.value.code == "acquisition_timeout_exhausted"
        assert elapsed < 0.5
        assert _active_count(manager) == 1
        admission_record = next(
            record
            for record in reversed(caplog.records)
            if record.msg == "Credential admission state"
        )
        assert admission_record.active == 1
        assert admission_record.waiting == 1
        assert admission_record.capacity == 1
        assert _CREDENTIAL not in admission_record.getMessage()
    finally:
        lib_logger.removeHandler(caplog.handler)
        lib_logger.setLevel(original_log_level)
        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder

    assert _active_count(manager) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("iteration", range(5))
async def test_client_cancellation_releases_permit_and_next_request_recovers(
    tmp_path: Path,
    iteration: int,
) -> None:
    client, manager = _make_client(tmp_path, acquire_timeout=2.0)
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
        # Given a real request owns the permit and another request is blocked on admission.
        holder = asyncio.create_task(
            client._execute_with_retry(
                blocking_api,
                request=None,
                model=_MODEL,
                messages=[{"role": "user", "content": "hold"}],
            )
        )
        await asyncio.wait_for(holder_entered_api.wait(), timeout=1.0)
        waiter_blocked = _observe_real_admission_wait(manager)
        waiter = asyncio.create_task(
            client._execute_with_retry(
                unexpected_api,
                request=None,
                model=_MODEL,
                messages=[{"role": "user", "content": "wait"}],
            )
        )
        await asyncio.wait_for(waiter_blocked.wait(), timeout=1.0)

        # When both request lifecycles are cancelled at their production boundaries.
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert _active_count(manager) == 1

        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder

        # Then the production finally block releases the permit and a real next request works.
        assert _active_count(manager) == 0

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
        assert _active_count(manager) == 0
    finally:
        pending = [task for task in (waiter, holder) if task is not None and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_interrupt_permit_release(tmp_path: Path) -> None:
    client, manager = _make_client(tmp_path, acquire_timeout=1.0)
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

    # Given a request owns the only permit and its release lock is briefly contended.
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
        # When cancellation is repeated while the production release is awaiting the lock.
        holder.cancel()
        await asyncio.wait_for(release_started.wait(), timeout=1.0)
        holder.cancel()
    finally:
        state_lock.release()

    with pytest.raises(asyncio.CancelledError):
        await holder

    # Then cancellation still propagates, but only after the permit has been returned.
    assert _active_count(manager) == 0


@pytest.mark.asyncio
async def test_upstream_exception_releases_permit(tmp_path: Path) -> None:
    client, manager = _make_client(tmp_path, acquire_timeout=1.0)

    async def failing_api(**_kwargs: object) -> object:
        raise ConnectionError("synthetic upstream disconnect")

    # Given the only permit is available to a provider request.
    assert _active_count(manager) == 0

    # When the provider disconnects after admission.
    failure_response = await client._execute_with_retry(
        failing_api,
        request=None,
        model=_MODEL,
        messages=[{"role": "user", "content": "fail"}],
    )
    assert failure_response["error"]["status"] == 503
    assert failure_response["error"]["code"] == "all_credentials_exhausted"

    # Then the permit is released even though provider failure policy quarantines the credential.
    assert _active_count(manager) == 0


def test_proxy_busy_public_contract_is_finite_and_sanitized() -> None:
    # Given an internal admission-exhaustion detail contains a credential value.
    internal_detail = "credential synthetic-secret-token is busy"

    # When the error crosses the public streaming boundary.
    body = build_public_stream_error("proxy_busy")

    # Then clients receive a finite sanitized proxy-busy contract.
    assert body == {
        "error": {
            "type": "proxy_busy",
            "code": "acquisition_timeout",
            "status": 503,
            "message": "No upstream credential is currently available. Retry later.",
        }
    }
    assert internal_detail not in str(body)
    assert "synthetic-secret-token" not in str(body)
