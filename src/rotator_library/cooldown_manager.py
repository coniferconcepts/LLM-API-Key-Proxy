# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import asyncio
import random
import time
from typing import Dict


class ExponentialBackoff:
    """Exponential backoff with optional jitter for provider re-enable."""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        jitter: bool = True,
        jitter_range: float = 0.5,  # +/-50% jitter
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.jitter_range = jitter_range
        self._rng = random.Random()  # Seeded for deterministic tests
        self._attempt_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    def seed(self, seed_value: int) -> None:
        """Seed the RNG for deterministic testing."""
        self._rng.seed(seed_value)

    async def calculate(self, provider: str) -> float:
        """Calculate backoff delay for a provider."""
        async with self._lock:
            attempt = self._attempt_counts.get(provider, 0)
            delay = min(self.base_delay * (2**attempt), self.max_delay)

            if self.jitter:
                jitter_factor = self._rng.uniform(
                    1 - self.jitter_range,
                    1 + self.jitter_range,
                )
                delay *= jitter_factor

            self._attempt_counts[provider] = attempt + 1
            return delay

    async def reset(self, provider: str) -> None:
        """Reset backoff for a provider (on successful recovery)."""
        async with self._lock:
            self._attempt_counts.pop(provider, None)


class CooldownManager:
    """
    Manages global cooldown periods for API providers to handle IP-based rate limiting.
    This ensures that once a 429 error is received for a provider, all subsequent
    requests to that provider are paused for a specified duration.
    """

    def __init__(self):
        self._cooldowns: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._backoff = ExponentialBackoff(
            base_delay=1.0,
            max_delay=300.0,
            jitter=True,
        )

    async def is_cooling_down(self, provider: str) -> bool:
        """Checks if a provider is currently in a cooldown period."""
        async with self._lock:
            return (
                provider in self._cooldowns and time.time() < self._cooldowns[provider]
            )

    async def start_cooldown(self, provider: str, duration: int | None = None):
        """
        Initiates or extends a cooldown period for a provider.
        The cooldown is set using exponential backoff delay with jitter.
        """
        delay = await self._backoff.calculate(provider)
        expiry = time.time() + delay
        async with self._lock:
            self._cooldowns[provider] = expiry

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
