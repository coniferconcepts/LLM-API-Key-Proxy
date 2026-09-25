# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import asyncio
import logging
import math
import time
from typing import Awaitable, Callable, Collection, Dict, Iterable, Optional


def has_untried_peer_credentials(
    credentials_for_provider: Iterable[str],
    tried_creds: Collection[str],
) -> bool:
    """
    Return True when at least one provider credential has not yet been tried.

    Callers must add the current credential to ``tried_creds`` *before* the
    upstream attempt (as ``RotatingClient`` already does). After a rate-limit
    failure on the current key, any remaining untried credentials are peers
    that can still serve the request without a provider-wide pause.
    """
    for cred in credentials_for_provider:
        if cred not in tried_creds:
            return True
    return False


def should_apply_provider_cooldown_for_rate_limit_error(
    *,
    status_code: Optional[int],
    error_type: Optional[str],
) -> bool:
    """
    Whether a classified failure is a short rate-limit (not long quota empty).

    Matches the historical client conditions used before peer-aware cooldowns:
    ``error_type == "rate_limit"`` or HTTP 429 excluding ``quota_exceeded``.
    """
    if error_type == "rate_limit":
        return True
    if status_code == 429 and error_type != "quota_exceeded":
        return True
    return False


COOLDOWN_BUDGET_EXCEEDED_MESSAGE = (
    "No credentials were available within the remaining request budget."
)


def remaining_budget_seconds(deadline: float, *, now: Optional[float] = None) -> float:
    """Return non-negative seconds left before ``deadline``."""
    current = time.time() if now is None else now
    return max(0.0, deadline - current)


def raise_if_cooldown_exceeds_budget(remaining_cooldown: float, remaining_budget: float) -> None:
    from .error_handler import NoAvailableKeysError

    if remaining_cooldown <= remaining_budget:
        return
    logging.getLogger("rotator_library").warning(
        "Provider cooldown exceeds remaining request budget. Failing early."
    )
    raise NoAvailableKeysError(
        COOLDOWN_BUDGET_EXCEEDED_MESSAGE,
        code="acquisition_timeout_exhausted",
        category="proxy_all_credentials_exhausted",
        soonest_end=time.time() + remaining_cooldown,
    )


class ModelAdmissionLimiter:
    """Bound concurrency for the Kimi Chutes model at the proxy boundary."""

    MODEL = "chutes/moonshotai/Kimi-K3-TEE"
    # DeepSeek Flash shares the Chutes account bucket and is outside this limiter.
    DEFAULT_LIMIT = 4
    DEFAULT_WAIT_SECONDS = 5.0

    def __init__(
        self,
        *,
        limit: int = DEFAULT_LIMIT,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if limit < 1 or wait_seconds < 0:
            raise ValueError("limit must be positive and wait_seconds non-negative")
        self.limit = limit
        self.wait_seconds = wait_seconds
        self._clock = clock
        self._sleep = sleeper
        self._active = 0
        self._lock = asyncio.Lock()

    async def acquire(self, model: str) -> bool:
        """Acquire a permit for the target model; other models pass through."""
        if model != self.MODEL:
            return False

        deadline = self._clock() + self.wait_seconds
        while True:
            async with self._lock:
                if self._active < self.limit:
                    self._active += 1
                    return True
            remaining = deadline - self._clock()
            if remaining <= 0:
                from .error_handler import NoAvailableKeysError

                error = NoAvailableKeysError(
                    "Kimi model admission limit reached.",
                    code="model_admission_timeout",
                    category="proxy_busy",
                )
                error.retry_after_seconds = max(1, math.ceil(self.wait_seconds))
                raise error
            await self._sleep(min(0.05, remaining))

    async def release(self, acquired: bool) -> None:
        """Release a previously acquired target-model permit exactly once."""
        if not acquired:
            return
        async with self._lock:
            self._active = max(0, self._active - 1)

    async def release_after_stream(self, stream, acquired: bool):
        """Yield a response stream, close it, and release its permit."""
        try:
            async for chunk in stream:
                yield chunk
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
            await self.release(acquired)


class CooldownManager:
    """
    Manages optional provider-wide cooldown periods.

    Provider-wide cooldowns are appropriate when:
    - the limit is shared across the provider (e.g. IP / edge rate limit), or
    - every credential for the provider has already been tried and rate-limited.

    Multi-account setups (e.g. two OpenCode GO keys) must **not** freeze the
    whole provider on the first key's 429: cool the failing credential via
    UsageManager and rotate to an untried peer instead. See
    ``has_untried_peer_credentials`` and RotatingClient's
    ``_maybe_start_provider_cooldown_on_rate_limit``.
    """

    def __init__(self):
        self._cooldowns: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def is_cooling_down(self, provider: str) -> bool:
        """Checks if a provider is currently in a cooldown period."""
        async with self._lock:
            return provider in self._cooldowns and time.time() < self._cooldowns[provider]

    async def start_cooldown(self, provider: str, duration: int):
        """
        Initiates or extends a cooldown period for a provider.
        The cooldown is set to the current time plus the specified duration.
        """
        async with self._lock:
            self._cooldowns[provider] = time.time() + duration

    async def get_cooldown_remaining(self, provider: str) -> float:
        """
        Returns the remaining cooldown time in seconds for a provider.
        Returns 0 if the provider is not in a cooldown period.
        """
        async with self._lock:
            if provider in self._cooldowns:
                remaining = self._cooldowns[provider] - time.time()
                return max(0, remaining)
            return 0
