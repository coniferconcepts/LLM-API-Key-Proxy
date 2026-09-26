# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel
"""Pinned model prices used by LiteLLM cost calculation."""

from __future__ import annotations

import litellm

CHUTES_CATALOG_DATE = "2026-09-24"

# Rates are dollars per token. These entries are pinned from the Chutes catalog;
# the application must not fetch a catalog at runtime.
PINNED_MODEL_PRICING = {
    "chutes/moonshotai/Kimi-K3-TEE": {
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 15e-6,
        "cache_read_input_token_cost": 0.3e-6,
        "output_cost_per_reasoning_token": 15e-6,
        "litellm_provider": "chutes",
        "mode": "chat",
    },
    "chutes/deepseek-ai/DeepSeek-V4-Flash-0731-TEE": {
        "input_cost_per_token": 0.44e-6,
        "output_cost_per_token": 1.32e-6,
        "cache_read_input_token_cost": 0.044e-6,
        "output_cost_per_reasoning_token": 1.32e-6,
        "litellm_provider": "chutes",
        "mode": "chat",
    },
}


def register_pinned_model_pricing() -> None:
    """Register pinned entries under their exact completion lookup keys."""
    litellm.register_model(PINNED_MODEL_PRICING)


def is_unpriced_chutes_model(model: str) -> bool:
    """Return whether a Chutes key is absent from our pinned price map."""
    return model.startswith("chutes/") and model not in PINNED_MODEL_PRICING


def completion_cost_or_unknown(response: object, model: str) -> float | None:
    """Calculate cost while preserving unknown Chutes prices as ``None``."""
    if is_unpriced_chutes_model(model):
        return None
    cost: float = litellm.completion_cost(completion_response=response, model=model)
    return cost


register_pinned_model_pricing()
