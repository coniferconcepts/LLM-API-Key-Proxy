# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import asyncio
import time
from typing import Collection, Dict, Iterable, Optional


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
