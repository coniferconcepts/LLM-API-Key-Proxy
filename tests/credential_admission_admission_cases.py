from __future__ import annotations

import asyncio
from pathlib import Path
import time

import pytest

from credential_admission_contract_support import (
    _CREDENTIAL,
    _MODEL,
    active_count,
    make_client,
    observe_real_admission_wait,
)
from rotator_library.error_handler import NoAvailableKeysError
from rotator_library.usage_manager import UsageManager, lib_logger


@pytest.mark.asyncio
async def test_busy_admission_has_a_bounded_proxy_busy_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, manager = make_client(tmp_path, acquire_timeout=0.05)
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
        waiter_blocked = observe_real_admission_wait(manager)
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
        assert waiter_blocked.is_set()
        assert captured.value.code == "acquisition_timeout_exhausted"
        assert captured.value.category == "proxy_busy"
        assert elapsed < 0.5
        assert active_count(manager) == 1
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

    assert active_count(manager) == 0


@pytest.mark.asyncio
async def test_cooldown_only_admission_is_classified_as_credential_exhaustion(
    tmp_path: Path,
) -> None:
    manager = UsageManager(str(tmp_path / "usage.json"))
    manager._usage_data = {_CREDENTIAL: {"key_cooldown_until": time.time() + 60.0}}  # noqa: SLF001
    manager._initialized.set()  # noqa: SLF001
    deadline = time.time() + 0.05

    with pytest.raises(NoAvailableKeysError) as captured:
        await manager.acquire_key(
            [_CREDENTIAL],
            _MODEL,
            deadline,
            acquire_deadline=deadline,
            all_provider_credentials=[_CREDENTIAL],
        )

    assert captured.value.code == "acquisition_timeout_exhausted"
    assert captured.value.category == "proxy_all_credentials_exhausted"


@pytest.mark.asyncio
async def test_expired_admission_budget_does_not_claim_unobserved_exhaustion(
    tmp_path: Path,
) -> None:
    manager = UsageManager(str(tmp_path / "usage.json"))
    manager._usage_data = {}  # noqa: SLF001
    manager._initialized.set()  # noqa: SLF001
    expired_deadline = time.time() - 1.0

    with pytest.raises(NoAvailableKeysError) as captured:
        await manager.acquire_key(
            [_CREDENTIAL],
            _MODEL,
            expired_deadline,
            acquire_deadline=expired_deadline,
            all_provider_credentials=[_CREDENTIAL],
        )

    assert captured.value.category == "proxy_busy"


@pytest.mark.asyncio
async def test_ollama_cloud_provider_pool_is_shared_across_models_and_releases(
    tmp_path: Path,
) -> None:
    manager = UsageManager(str(tmp_path / "usage.json"))
    manager._usage_data = {}  # noqa: SLF001
    manager._initialized.set()  # noqa: SLF001
    credentials = ["synthetic-cloud-a", "synthetic-cloud-b"]
    models = [
        "ollama_cloud/glm-5.3-flash:cloud",
        "ollama_cloud/kimi-k2.7-code:cloud",
    ]
    acquired: list[tuple[str, str]] = []

    for index in range(6):
        model = models[index % len(models)]
        key = await manager.acquire_key(
            credentials,
            model,
            time.time() + 1,
            acquire_deadline=time.time() + 1,
            max_concurrent=3,
            all_provider_credentials=credentials,
        )
        acquired.append((key, model))

    started = asyncio.get_running_loop().time()
    with pytest.raises(NoAvailableKeysError) as captured:
        await manager.acquire_key(
            credentials,
            "ollama_cloud/minimax-m3:cloud",
            time.time() + 10,
            acquire_deadline=time.time() + 10,
            max_concurrent=3,
            all_provider_credentials=credentials,
        )
    assert captured.value.category == "proxy_busy"
    assert asyncio.get_running_loop().time() - started < 0.2

    released_key, released_model = acquired.pop()
    await manager.release_key(released_key, released_model)
    replacement_model = "ollama_cloud/qwen3.5:397b"
    leaked_key, leaked_model = acquired.pop()
    await manager.release_key(leaked_key, leaked_model)
    replacement_key = await manager.acquire_key(
        credentials,
        replacement_model,
        time.time() + 1,
        acquire_deadline=time.time() + 1,
        max_concurrent=3,
        all_provider_credentials=credentials,
    )
    acquired.append((replacement_key, replacement_model))

    for key, held_model in acquired:
        await manager.release_key(key, held_model)
    assert manager._provider_pool.in_use("ollama_cloud") == 0  # noqa: SLF001


@pytest.mark.asyncio
async def test_ollama_cloud_provider_pool_stays_at_six_with_extra_credentials(
    tmp_path: Path,
) -> None:
    manager = UsageManager(str(tmp_path / "usage.json"))
    manager._usage_data = {}  # noqa: SLF001
    manager._initialized.set()  # noqa: SLF001
    credentials = ["synthetic-cloud-a", "synthetic-cloud-b", "synthetic-cloud-c"]
    acquired: list[tuple[str, str]] = []
    model = "ollama_cloud/glm-5.3-flash:cloud"

    for _ in range(6):
        key = await manager.acquire_key(
            credentials,
            model,
            time.time() + 1,
            acquire_deadline=time.time() + 1,
            max_concurrent=3,
            all_provider_credentials=credentials,
        )
        acquired.append((key, model))

    started = asyncio.get_running_loop().time()
    with pytest.raises(NoAvailableKeysError) as captured:
        await manager.acquire_key(
            credentials,
            "ollama_cloud/minimax-m3:cloud",
            time.time() + 10,
            acquire_deadline=time.time() + 10,
            max_concurrent=3,
            all_provider_credentials=credentials,
        )
    assert captured.value.category == "proxy_busy"
    assert captured.value.diagnostics["provider_pool_capacity"] == 6
    assert asyncio.get_running_loop().time() - started < 0.2

    for key, held_model in acquired:
        await manager.release_key(key, held_model)
    assert manager._provider_pool.in_use("ollama_cloud") == 0  # noqa: SLF001
