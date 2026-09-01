# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from __future__ import annotations

from collections.abc import Callable, MutableMapping
import os
from typing import Any, Final

FIREWORKS_PROVIDER: Final = "fireworks"
APPROVED_FIREWORKS_MODEL: Final = "fireworks/accounts/fireworks/models/glm-5p3-flash"
_NOT_ADMITTED = "Fireworks model is not admitted by the exact-ID policy."

_ApplyRouting = Callable[[str], tuple[str, Any]]
_LogDecision = Callable[[Any], None]


def is_model_admitted(model: str) -> bool:
    if not model.startswith(f"{FIREWORKS_PROVIDER}/"):
        return True

    configured = os.environ.get("WHITELIST_MODELS_FIREWORKS")
    if configured is not None and configured not in {"", APPROVED_FIREWORKS_MODEL}:
        raise ValueError(
            "WHITELIST_MODELS_FIREWORKS must be unset, empty, or exactly "
            f"'{APPROVED_FIREWORKS_MODEL}'."
        )
    if configured == "":
        return False
    return model == APPROVED_FIREWORKS_MODEL


def require_admitted_model(model: str) -> None:
    if not is_model_admitted(model):
        raise ValueError(_NOT_ADMITTED)


def validate_fireworks_admission_config() -> None:
    is_model_admitted(APPROVED_FIREWORKS_MODEL)


def fireworks_catalog_unavailable(provider: str) -> bool:
    return provider == FIREWORKS_PROVIDER and not is_model_admitted(APPROVED_FIREWORKS_MODEL)


def admit(
    kwargs: MutableMapping[str, Any],
    apply_routing: _ApplyRouting,
    log_decision: _LogDecision,
) -> str:
    model = kwargs.get("model", "")
    routed, decision = apply_routing(model) if model else (model, None)
    if routed != model:
        kwargs["model"] = routed
        log_decision(decision)
    require_admitted_model(routed)
    return routed
