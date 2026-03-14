import sys
from pathlib import Path
import random

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotator_library.routing_policy import RoutingPolicy, RoutingPolicyError


def make_policy(overrides=None, provider_models=None, providers=None):
    return RoutingPolicy(
        model_overrides=overrides
        or {
            "nemotron-3-super": {
                "strategy": "single",
                "primary": "ollama",
                "allowed_providers": ["ollama"],
                "fallback_providers": [],
                "strict": True,
                "allow_global_fallback": False,
                "reason": "Only available on Ollama Cloud",
            }
        },
        available_providers=providers or {"ollama", "chutes"},
        provider_models=provider_models
        or {
            "ollama": {"nemotron-3-super", "qwen3.5"},
            "chutes": {"qwen3.5"},
        },
    )


def test_single_override_rewrites_weighted_router_model():
    decision = make_policy().resolve("weighted-router/nemotron-3-super")

    assert decision.selected_provider == "ollama"
    assert decision.rewritten_model == "ollama/nemotron-3-super"
    assert decision.selection_source == "model_override"
    assert decision.override_applied is True


def test_non_weighted_router_model_passes_through():
    decision = make_policy().resolve("ollama/nemotron-3-super")

    assert decision.rewritten_model == "ollama/nemotron-3-super"
    assert decision.override_applied is False
    assert decision.selection_source == "passthrough"


def test_missing_override_for_weighted_router_model_fails_closed():
    with pytest.raises(RoutingPolicyError, match="No routing override configured"):
        make_policy().resolve("weighted-router/qwen3.5")


def test_unknown_provider_fails_validation():
    with pytest.raises(RoutingPolicyError, match="unknown provider 'go'"):
        make_policy(
            overrides={
                "nemotron-3-super": {
                    "strategy": "single",
                    "primary": "go",
                    "allowed_providers": ["go"],
                    "fallback_providers": [],
                }
            }
        )


@pytest.mark.parametrize(
    "override, expected_error",
    [
        (
            {"nemotron-3-super": {"primary": "ollama", "allowed_providers": ["ollama"]}},
            "strategy 'single'",
        ),
        (
            {
                "nemotron-3-super": {
                    "strategy": "single",
                    "primary": "ollama",
                    "allowed_providers": ["ollama", "chutes"],
                }
            },
            "restrict 'allowed_providers'",
        ),
        (
            {
                "nemotron-3-super": {
                    "strategy": "single",
                    "primary": "ollama",
                    "allowed_providers": ["ollama"],
                    "fallback_providers": ["chutes"],
                }
            },
            "cannot define 'fallback_providers'",
        ),
    ],
)
def test_invalid_single_override_shapes_fail_validation(override, expected_error):
    with pytest.raises(RoutingPolicyError, match=expected_error):
        make_policy(overrides=override)


def test_provider_model_mismatch_fails_validation_when_models_are_known():
    with pytest.raises(RoutingPolicyError, match="does not expose model 'nemotron-3-super'"):
        make_policy(
            overrides={
                "nemotron-3-super": {
                    "strategy": "single",
                    "primary": "chutes",
                    "allowed_providers": ["chutes"],
                    "fallback_providers": [],
                }
            }
        )


def test_provider_in_available_providers_remains_known_when_plugins_differ():
    policy = RoutingPolicy(
        model_overrides={},
        available_providers={"ollama", "chutes"},
        known_providers={"opencode_go"},
    )

    assert policy.known_providers == {"ollama", "chutes", "opencode_go"}


def test_weighted_override_rewrites_qwen3_5_to_allowed_provider():
    policy = RoutingPolicy(
        model_overrides={
            "qwen3.5": {
                "strategy": "weighted",
                "allowed_providers": ["ollama", "chutes"],
                "weights": {"ollama": 80, "chutes": 20},
                "excluded_providers": ["opencode_go"],
                "fallback_providers": [],
                "strict": True,
                "allow_global_fallback": False,
                "reason": "Exclude opencode_go for qwen3.5",
            }
        },
        available_providers={"ollama", "chutes"},
        provider_models={
            "ollama": {"qwen3.5"},
            "chutes": {"qwen3.5"},
        },
        known_providers={"ollama", "chutes", "opencode_go"},
        rng=random.Random(1),
    )

    decision = policy.resolve("weighted-router/qwen3.5")

    assert decision.selected_provider in {"ollama", "chutes"}
    assert decision.selected_provider != "opencode_go"
    assert decision.rewritten_model == f"{decision.selected_provider}/qwen3.5"
    assert decision.strategy == "weighted"
    assert decision.candidate_providers == ["ollama", "chutes"]
    assert decision.excluded_providers == ["opencode_go"]


def test_weighted_override_with_zero_roll_selects_first_provider():
    policy = RoutingPolicy(
        model_overrides={
            "qwen3.5": {
                "strategy": "weighted",
                "allowed_providers": ["ollama", "chutes"],
                "weights": {"ollama": 80, "chutes": 20},
                "excluded_providers": ["opencode_go"],
                "fallback_providers": [],
            }
        },
        available_providers={"ollama", "chutes"},
        provider_models={"ollama": {"qwen3.5"}, "chutes": {"qwen3.5"}},
        known_providers={"ollama", "chutes", "opencode_go"},
        rng=random.Random(0),
    )
    policy.rng.uniform = lambda start, end: 0.0

    decision = policy.resolve("weighted-router/qwen3.5")

    assert decision.selected_provider == "ollama"


@pytest.mark.parametrize(
    "override, expected_error",
    [
        (
            {
                "qwen3.5": {
                    "strategy": "weighted",
                    "allowed_providers": ["ollama", "chutes"],
                    "weights": {"ollama": 80, "chutes": 20},
                    "excluded_providers": ["chutes"],
                    "fallback_providers": [],
                }
            },
            "both 'allowed_providers' and 'excluded_providers'",
        ),
        (
            {
                "qwen3.5": {
                    "strategy": "weighted",
                    "allowed_providers": ["ollama", "chutes"],
                    "weights": {"ollama": 80},
                    "excluded_providers": ["opencode_go"],
                    "fallback_providers": [],
                }
            },
            "matching providers in 'weights' and 'allowed_providers'",
        ),
        (
            {
                "qwen3.5": {
                    "strategy": "weighted",
                    "allowed_providers": ["ollama", "chutes"],
                    "weights": {"ollama": 80, "chutes": -20},
                    "excluded_providers": ["opencode_go"],
                    "fallback_providers": [],
                }
            },
            "negative weights",
        ),
        (
            {
                "qwen3.5": {
                    "strategy": "weighted",
                    "allowed_providers": ["ollama", "chutes"],
                    "weights": {"ollama": 0, "chutes": 0},
                    "excluded_providers": ["opencode_go"],
                    "fallback_providers": [],
                }
            },
            "total greater than zero",
        ),
        (
            {
                "qwen3.5": {
                    "strategy": "weighted",
                    "allowed_providers": ["ollama", "chutes"],
                    "weights": {"ollama": 80, "chutes": 20},
                    "excluded_providers": ["opencode_go"],
                    "fallback_providers": [],
                    "allow_global_fallback": True,
                }
            },
            "cannot enable 'allow_global_fallback'",
        ),
        (
            {
                "qwen3.5": {
                    "strategy": "weighted",
                    "allowed_providers": ["ollama", "chutes"],
                    "weights": {"ollama": 80, "chutes": 20},
                    "excluded_providers": ["opencode_go"],
                    "fallback_providers": ["opencode_go"],
                }
            },
            "cannot define 'fallback_providers' in v2",
        ),
    ],
)
def test_invalid_weighted_override_shapes_fail_validation(override, expected_error):
    with pytest.raises(RoutingPolicyError, match=expected_error):
        RoutingPolicy(
            model_overrides=override,
            available_providers={"ollama", "chutes"},
            provider_models={"ollama": {"qwen3.5"}, "chutes": {"qwen3.5"}},
            known_providers={"ollama", "chutes", "opencode_go"},
        )
