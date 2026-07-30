from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Final

_LEGACY_FIREWORKS_KEY_ENV_VARS: Final = frozenset({"FIREWORKS_API_KEY", "FIREWORKS_AI_API_KEY"})
_CREDENTIAL_PROVIDER_ALIASES: Final = {
    "ollama": "ollama_cloud",
    "openrouter_free": "openrouter_non_zdr",
}
_EXPLICIT_CREDENTIAL_PROVIDER_ENV_ALIASES: Final = {
    "OPENROUTER_FREE_KEY": "openrouter_non_zdr",
    "OPENROUTER_NON_ZDR_KEY": "openrouter_non_zdr",
}


def normalize_credential_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    return _CREDENTIAL_PROVIDER_ALIASES.get(normalized, normalized)


def _provider_from_credential_env_key(key: str) -> str | None:
    if key == "FIREWORKS_API_V2_KEY":
        return "fireworks"
    if key in _EXPLICIT_CREDENTIAL_PROVIDER_ENV_ALIASES:
        return _EXPLICIT_CREDENTIAL_PROVIDER_ENV_ALIASES[key]
    if "_API_KEY" in key:
        return normalize_credential_provider(key.split("_API_KEY")[0])
    return None


def discover_api_keys_from_env(env: Mapping[str, str] | None = None) -> dict[str, list[str]]:
    """Discover canonical provider API keys while preserving subscription boundaries."""
    source = os.environ if env is None else env
    api_keys: dict[str, list[str]] = {}

    for key, value in source.items():
        if not value or key == "PROXY_API_KEY":
            continue
        if key in _LEGACY_FIREWORKS_KEY_ENV_VARS:
            continue
        provider = _provider_from_credential_env_key(key)
        if provider is None:
            continue

        provider_keys = api_keys.setdefault(provider, [])
        if value not in provider_keys:
            provider_keys.append(value)

    return api_keys
