from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from proxy_app.local_transport_policy import normalize_local_xai_base

_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def _flag_enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in _ENABLED_VALUES


@dataclass(frozen=True, slots=True)
class LocalTransportRuntimePolicy:
    enabled: bool
    xai_api_base: str | None

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> LocalTransportRuntimePolicy:
        enabled = _flag_enabled(environment.get("MIRROWEL_LOCAL_TRANSPORT_SAFE_MODE"))
        if not enabled:
            return cls(enabled=False, xai_api_base=None)
        configured_base = environment.get("XAI_OAUTH_API_BASE")
        if configured_base is None:
            raise ValueError("XAI_OAUTH_API_BASE is required in local transport safe mode")
        return cls(enabled=True, xai_api_base=normalize_local_xai_base(configured_base))

    def is_current(self, environment: Mapping[str, str]) -> bool:
        current_enabled = _flag_enabled(environment.get("MIRROWEL_LOCAL_TRANSPORT_SAFE_MODE"))
        if current_enabled != self.enabled:
            return False
        if not self.enabled:
            return True
        configured_base = environment.get("XAI_OAUTH_API_BASE")
        if configured_base is None:
            return False
        try:
            current_base = normalize_local_xai_base(configured_base)
        except ValueError:
            return False
        return current_base == self.xai_api_base
