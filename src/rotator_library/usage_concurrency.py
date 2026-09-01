from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .error_handler import NoAvailableKeysError

OLLAMA_CLOUD_PROVIDER = "ollama_cloud"
OLLAMA_CLOUD_PROVIDER_POOL_CAPACITY = 6
PER_MODEL_FAIL_FAST_PROVIDERS = frozenset({"fireworks", "openai"})

_PriorityMultiplier = Callable[[str, int, str], int | float]
_ActiveKeyBlock = Callable[..., float]


def provider_from_model(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else ""


def provider_pool_capacity(provider: str, credential_count: int, max_concurrent: int) -> int:
    if provider == OLLAMA_CLOUD_PROVIDER:
        return OLLAMA_CLOUD_PROVIDER_POOL_CAPACITY
    return credential_count * max_concurrent


class ProviderPoolTracker:
    """Provider-wide in-flight slot counter. Ollama Cloud uses a fixed cap of 6."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._in_use: dict[str, int] = {}

    def in_use(self, provider: str) -> int:
        return self._in_use.get(provider, 0)

    async def is_full(self, provider: str, capacity: int) -> bool:
        async with self._lock:
            return self._in_use.get(provider, 0) >= capacity

    async def try_reserve(self, provider: str, capacity: int) -> bool:
        async with self._lock:
            active = self._in_use.get(provider, 0)
            if active >= capacity:
                return False
            self._in_use[provider] = active + 1
            return True

    async def release(self, provider: str) -> None:
        async with self._lock:
            active = self._in_use.get(provider, 0)
            if active <= 1:
                self._in_use.pop(provider, None)
            else:
                self._in_use[provider] = active - 1


def pool_busy_error(provider: str, model: str, capacity: int) -> NoAvailableKeysError:
    return NoAvailableKeysError(
        f"Provider concurrency pool is full for model {model}.",
        code="acquisition_timeout_exhausted",
        diagnostics={
            "model": model,
            "provider": provider,
            "provider_pool_capacity": capacity,
        },
        category="proxy_busy",
    )


def per_model_busy_error(provider: str, model: str, capacity: int) -> NoAvailableKeysError:
    return NoAvailableKeysError(
        f"All credentials are at per-model concurrency capacity for {model}.",
        code="acquisition_timeout_exhausted",
        diagnostics={
            "model": model,
            "provider": provider,
            "per_model_capacity": capacity,
        },
        category="proxy_busy",
    )


async def reserve_ollama_cloud_slot(
    tracker: ProviderPoolTracker,
    provider: str,
    model: str,
    capacity: int,
) -> None:
    if provider != OLLAMA_CLOUD_PROVIDER:
        return
    if not await tracker.try_reserve(provider, capacity):
        raise pool_busy_error(provider, model, capacity)


def _key_is_cooling(
    key_data: Mapping[str, Any],
    normalized_model: str,
    now: float,
    active_key_block: _ActiveKeyBlock,
) -> bool:
    cooldown = key_data.get("model_cooldowns", {})
    model_until = 0
    if isinstance(cooldown, Mapping):
        model_until = cooldown.get(normalized_model, 0) or 0
    return active_key_block(key_data, now) > now or model_until > now


def _key_data(usage_data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = usage_data.get(key, {})
    return value if isinstance(value, Mapping) else {}


def has_non_cooling_key(
    available_keys: Sequence[str],
    usage_data: Mapping[str, Any],
    normalized_model: str,
    now: float,
    active_key_block: _ActiveKeyBlock,
) -> bool:
    return any(
        not _key_is_cooling(_key_data(usage_data, key), normalized_model, now, active_key_block)
        for key in available_keys
    )


async def enforce_admission_limits(
    *,
    provider: str,
    model: str,
    available_keys: Sequence[str],
    usage_data: Mapping[str, Any],
    key_states: Mapping[str, Any],
    data_lock: asyncio.Lock,
    tracker: ProviderPoolTracker,
    normalized_model: str,
    now: float,
    max_concurrent: int,
    credential_priorities: Mapping[str, int] | None,
    rotation_mode: str,
    priority_multiplier: _PriorityMultiplier,
    active_key_block: _ActiveKeyBlock,
    pool_capacity: int,
) -> None:
    if provider in PER_MODEL_FAIL_FAST_PROVIDERS:
        async with data_lock:
            saturated = per_model_is_saturated(
                available_keys=available_keys,
                usage_data=usage_data,
                key_states=key_states,
                model=model,
                normalized_model=normalized_model,
                now=now,
                max_concurrent=max_concurrent,
                provider=provider,
                credential_priorities=credential_priorities,
                rotation_mode=rotation_mode,
                priority_multiplier=priority_multiplier,
                active_key_block=active_key_block,
            )
        if saturated:
            raise per_model_busy_error(provider, model, max_concurrent)

    if provider == OLLAMA_CLOUD_PROVIDER and pool_capacity > 0:
        async with data_lock:
            cooling_idle = has_non_cooling_key(
                available_keys,
                usage_data,
                normalized_model,
                now,
                active_key_block,
            )
        if cooling_idle and await tracker.is_full(provider, pool_capacity):
            raise pool_busy_error(provider, model, pool_capacity)


def per_model_is_saturated(
    *,
    available_keys: Sequence[str],
    usage_data: Mapping[str, Any],
    key_states: Mapping[str, Any],
    model: str,
    normalized_model: str,
    now: float,
    max_concurrent: int,
    provider: str,
    credential_priorities: Mapping[str, int] | None,
    rotation_mode: str,
    priority_multiplier: _PriorityMultiplier,
    active_key_block: _ActiveKeyBlock,
) -> bool:
    non_cooling_keys = [
        key
        for key in available_keys
        if not _key_is_cooling(
            _key_data(usage_data, key),
            normalized_model,
            now,
            active_key_block,
        )
    ]
    if not non_cooling_keys:
        return False
    return all(
        key_states[key]["models_in_use"].get(model, 0)
        >= max_concurrent
        * priority_multiplier(
            provider,
            credential_priorities.get(key, 999) if credential_priorities else 999,
            rotation_mode,
        )
        for key in non_cooling_keys
    )
