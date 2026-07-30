from __future__ import annotations

import logging
import re
from typing import TypeAlias

_UsageScalar: TypeAlias = str | int | float | bool | None
_UsageValue: TypeAlias = _UsageScalar | list["_UsageValue"] | dict[str, "_UsageValue"]

_API_KEY_PROVIDERS = {
    "sk-nano-": "nanogpt",
    "sk-or-": "openrouter",
    "sk-ant-": "anthropic",
}


def _provider_from_models(value: _UsageValue) -> str | None:
    if not isinstance(value, dict):
        return None
    first_model = next(iter(value), None)
    if first_model and "/" in first_model:
        return first_model.split("/", maxsplit=1)[0].lower()
    return None


def provider_from_credential(
    credential: str,
    usage_data: dict[str, _UsageValue] | None,
) -> str | None:
    if credential.startswith("env://"):
        provider = credential[6:].split("/", maxsplit=1)[0]
        if provider:
            return provider.lower()
        logging.getLogger("rotator_library").warning(
            "Malformed env:// credential URI: %s", credential
        )
        return None

    normalized = credential.replace("\\", "/")
    for pattern in (
        r"/([a-z_]+)_oauth_\d+\.json$",
        r"oauth_creds/([a-z_]+)_",
        r"^([a-z_]+)_oauth_\d+\.json$",
    ):
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    for prefix, provider in _API_KEY_PROVIDERS.items():
        if credential.startswith(prefix):
            return provider

    if not usage_data:
        return None
    credential_data = usage_data.get(credential)
    if not isinstance(credential_data, dict):
        return None

    stored_provider = _provider_from_models(credential_data.get("models"))
    if stored_provider is not None:
        return stored_provider
    daily_data = credential_data.get("daily")
    if not isinstance(daily_data, dict):
        return None
    return _provider_from_models(daily_data.get("models"))
