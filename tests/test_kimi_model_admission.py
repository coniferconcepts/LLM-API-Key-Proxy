from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proxy_app.safe_errors import handle_credential_failure  # noqa: E402
from rotator_library.cooldown_manager import ModelAdmissionLimiter  # noqa: E402
from rotator_library.error_handler import NoAvailableKeysError  # noqa: E402


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, duration: float) -> None:
        self.now += duration


@pytest.mark.asyncio
async def test_kimi_admission_timeout_expires_and_propagates_retry_after() -> None:
    clock = FakeClock()
    limiter = ModelAdmissionLimiter(clock=clock, sleeper=clock.sleep)
    model = limiter.MODEL
    permits = [await limiter.acquire(model) for _ in range(4)]

    with pytest.raises(NoAvailableKeysError) as captured:
        await limiter.acquire(model)

    assert captured.value.code == "model_admission_timeout"
    assert captured.value.retry_after_seconds == 5
    response = handle_credential_failure(captured.value, None)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert json.loads(response.body)["error"]["retry_after_seconds"] == 5

    for permit in permits:
        await limiter.release(permit)
    assert await limiter.acquire(model)
    await limiter.release(True)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["complete", "exception", "disconnect"])
async def test_stream_completion_exception_and_disconnect_recover_permit(finish: str) -> None:
    limiter = ModelAdmissionLimiter(limit=1)
    model = limiter.MODEL
    acquired = await limiter.acquire(model)

    async def source():
        yield "first"
        if finish == "exception":
            raise RuntimeError("stream failed")
        if finish == "complete":
            yield "last"

    wrapped = limiter.release_after_stream(source(), acquired)
    assert await anext(wrapped) == "first"
    if finish == "complete":
        assert await anext(wrapped) == "last"
        with pytest.raises(StopAsyncIteration):
            await anext(wrapped)
    elif finish == "exception":
        with pytest.raises(RuntimeError):
            await anext(wrapped)
    else:
        await wrapped.aclose()

    assert await limiter.acquire(model)
    await limiter.release(True)


@pytest.mark.asyncio
async def test_limiter_only_applies_to_exact_kimi_model() -> None:
    limiter = ModelAdmissionLimiter(limit=1)
    assert not await limiter.acquire("chutes/deepseek-ai/DeepSeek-V4-Flash-0731-TEE")
    assert not await limiter.acquire("chutes/moonshotai/Kimi-K3")
    assert await limiter.acquire(limiter.MODEL)
    await limiter.release(True)
