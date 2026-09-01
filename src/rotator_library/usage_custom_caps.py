from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

lib_logger = logging.getLogger("rotator_library")


def model_quota_group_for_provider(plugin_instance: object | None, model: str) -> str | None:
    if plugin_instance is not None and hasattr(plugin_instance, "get_model_quota_group"):
        group = plugin_instance.get_model_quota_group(model)
        return group if isinstance(group, str) else None
    return None


def resolve_custom_cap_config(
    provider_caps: Mapping[Any, Any] | None,
    *,
    tier_priority: int,
    clean_model: str,
    group: str | None,
) -> dict[str, Any] | None:
    if not provider_caps:
        return None

    tier_config = None
    default_config = None
    for tier_key, models_config in provider_caps.items():
        if tier_key == "default":
            default_config = models_config
            continue
        if isinstance(tier_key, int) and tier_key == tier_priority:
            tier_config = models_config
            break
        if isinstance(tier_key, tuple) and tier_priority in tier_key:
            tier_config = models_config
            break

    if isinstance(tier_config, Mapping):
        if clean_model in tier_config:
            match = tier_config[clean_model]
            return match if isinstance(match, dict) else None
        if group and group in tier_config:
            match = tier_config[group]
            return match if isinstance(match, dict) else None

    if isinstance(default_config, Mapping):
        if clean_model in default_config:
            match = default_config[clean_model]
            return match if isinstance(match, dict) else None
        if group and group in default_config:
            match = default_config[group]
            return match if isinstance(match, dict) else None
    return None


def resolve_custom_cap_max(
    provider: str,
    model: str,
    cap_config: Mapping[str, Any],
    actual_max: int | None,
) -> int | None:
    max_requests = cap_config.get("max_requests")
    if max_requests is None:
        return None

    if isinstance(max_requests, str) and max_requests.endswith("%"):
        if actual_max is None:
            lib_logger.warning(
                "Custom cap '%s' for %s/%s requires known max_requests. "
                "Skipping until quota baseline is fetched. Use absolute value "
                "for immediate enforcement.",
                max_requests,
                provider,
                model,
            )
            return None
        try:
            percentage = float(max_requests.rstrip("%")) / 100.0
            calculated = int(actual_max * percentage)
        except ValueError:
            lib_logger.warning(
                "Invalid percentage cap '%s' for %s/%s",
                max_requests,
                provider,
                model,
            )
            return None
    else:
        try:
            calculated = int(max_requests)
        except (ValueError, TypeError):
            lib_logger.warning("Invalid cap value '%s' for %s/%s", max_requests, provider, model)
            return None

    if actual_max is not None:
        return min(calculated, actual_max)
    return calculated
