from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.error_handler import NoAvailableKeysError  # noqa: E402
from rotator_library.usage_manager import UsageManager  # noqa: E402


def _manager(tmp_path: Path, name: str = "usage.json") -> UsageManager:
    # Tests pin admission classification, not the 03:00 UTC daily-reset path.
    # Leaving the default reset enabled can spend the short acquire budget on
    # _save_usage and skip the loop, so cooling-only looks like proxy_busy.
    manager = UsageManager(str(tmp_path / name), daily_reset_time_utc=None)
    manager._usage_data = {}  # noqa: SLF001
    manager._initialized.set()  # noqa: SLF001
    return manager


async def _acquire(
    manager: UsageManager,
    credentials: list[str],
    model: str,
    *,
    max_concurrent: int = 1,
) -> str:
    return await manager.acquire_key(
        credentials,
        model,
        time.time() + 1,
        acquire_deadline=time.time() + 1,
        max_concurrent=max_concurrent,
        all_provider_credentials=credentials,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["fireworks", "openai"])
async def test_saturation_fails_fast_for_fireworks_and_openai(
    tmp_path: Path,
    provider: str,
) -> None:
    manager = _manager(tmp_path, f"{provider}.json")
    credentials = [f"{provider}-a", f"{provider}-b"]
    model = f"{provider}/model-a"
    acquired = [await _acquire(manager, credentials, model) for _ in credentials]

    started = asyncio.get_running_loop().time()
    with pytest.raises(NoAvailableKeysError) as captured:
        await manager.acquire_key(
            credentials,
            model,
            time.time() + 10,
            acquire_deadline=time.time() + 10,
            all_provider_credentials=credentials,
        )

    assert captured.value.category == "proxy_busy"
    assert asyncio.get_running_loop().time() - started < 0.2

    await manager.release_key(acquired.pop(), model)
    replacement = await _acquire(manager, credentials, model)
    await manager.release_key(replacement, model)
    for key in acquired:
        await manager.release_key(key, model)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["chutes", "opencode_go", "xai_oauth"])
async def test_other_providers_keep_the_existing_acquire_wait(
    tmp_path: Path,
    provider: str,
) -> None:
    manager = _manager(tmp_path, f"{provider}.json")
    credentials = [f"{provider}-only"]
    model = f"{provider}/model-a"
    acquired = await _acquire(manager, credentials, model)
    deadline = time.time() + 0.05

    started = asyncio.get_running_loop().time()
    with pytest.raises(NoAvailableKeysError):
        await manager.acquire_key(
            credentials,
            model,
            deadline,
            acquire_deadline=deadline,
            all_provider_credentials=credentials,
        )

    assert asyncio.get_running_loop().time() - started >= 0.04
    await manager.release_key(acquired, model)


@pytest.mark.asyncio
async def test_saturated_model_does_not_block_another_model_on_same_key(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    credentials = ["fireworks-only"]
    model_a_key = await _acquire(manager, credentials, "fireworks/model-a")
    model_b_key = await _acquire(manager, credentials, "fireworks/model-b")

    assert model_a_key == model_b_key == credentials[0]
    await manager.release_key(model_b_key, "fireworks/model-b")
    await manager.release_key(model_a_key, "fireworks/model-a")


@pytest.mark.asyncio
async def test_saturation_with_cooling_peer_is_busy(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager._usage_data = {  # noqa: SLF001
        "fireworks-cooling": {"key_cooldown_until": time.time() + 60.0},
    }
    credentials = ["fireworks-busy", "fireworks-cooling"]
    model = "fireworks/model-a"
    busy_key = await _acquire(manager, [credentials[0]], model)

    started = asyncio.get_running_loop().time()
    with pytest.raises(NoAvailableKeysError) as captured:
        await manager.acquire_key(
            credentials,
            model,
            time.time() + 10,
            acquire_deadline=time.time() + 10,
            all_provider_credentials=credentials,
        )

    assert captured.value.category == "proxy_busy"
    assert asyncio.get_running_loop().time() - started < 0.2
    await manager.release_key(busy_key, model)


@pytest.mark.asyncio
async def test_cooling_only_is_credential_exhaustion(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager._usage_data = {  # noqa: SLF001
        "fireworks-cooling": {"key_cooldown_until": time.time() + 60.0},
    }

    # Budget stays far below the 60s cooldown so fail-fast still classifies
    # exhaustion, but is long enough that pre-loop setup cannot skip the loop.
    deadline = time.time() + 1.0
    with pytest.raises(NoAvailableKeysError) as captured:
        await manager.acquire_key(
            ["fireworks-cooling"],
            "fireworks/model-a",
            deadline,
            acquire_deadline=deadline,
            all_provider_credentials=["fireworks-cooling"],
        )

    assert captured.value.category == "proxy_all_credentials_exhausted"
