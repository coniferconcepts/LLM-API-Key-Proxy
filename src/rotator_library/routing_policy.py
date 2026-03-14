from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, Iterable, Optional, Set


class RoutingPolicyError(ValueError):
    """Raised when routing override configuration or resolution is invalid."""


@dataclass(frozen=True)
class RouteDecision:
    requested_model: str
    clean_model: str
    selected_provider: Optional[str]
    rewritten_model: Optional[str]
    strategy: str
    selection_source: str
    override_applied: bool
    candidate_providers: list[str]
    strict: bool
    allow_global_fallback: bool
    excluded_providers: list[str]
    reason: Optional[str] = None


class RoutingPolicy:
    """Resolve weighted-router models into concrete provider-prefixed models.

    Weighted-router aliases are rewritten before provider lock-in so the
    existing retry and credential machinery can continue unchanged.
    """

    def __init__(
        self,
        model_overrides: Dict[str, Any],
        available_providers: Iterable[str],
        provider_models: Optional[Dict[str, Set[str]]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        if not isinstance(model_overrides, dict):
            raise RoutingPolicyError("MODEL_ROUTING_OVERRIDES must decode to an object")

        self.model_overrides = model_overrides
        self.available_providers = set(available_providers)
        self.provider_models = provider_models or {}
        self.rng = rng or random.Random()
        self._validate()

    def _validate_provider_model(self, provider: str, clean_model: str) -> None:
        provider_models = self.provider_models.get(provider)
        if provider_models and clean_model not in provider_models:
            raise RoutingPolicyError(
                f"provider '{provider}' does not expose model '{clean_model}' in configured model definitions"
            )

    def _validate_provider_name(self, provider: str, clean_model: str, field_name: str) -> None:
        if provider not in self.available_providers:
            raise RoutingPolicyError(
                f"routing override for '{clean_model}' references unknown provider '{provider}' in '{field_name}'"
            )

    def _validate_weighted_override(self, clean_model: str, override: Dict[str, Any]) -> None:
        weights = override.get("weights")
        if not isinstance(weights, dict) or not weights:
            raise RoutingPolicyError(
                f"routing override for '{clean_model}' must define a non-empty 'weights' object in v2"
            )

        allowed_providers = override.get("allowed_providers")
        if not isinstance(allowed_providers, list) or not allowed_providers:
            raise RoutingPolicyError(
                f"routing override for '{clean_model}' must define a non-empty 'allowed_providers' list in v2"
            )
        if len(set(allowed_providers)) != len(allowed_providers):
            raise RoutingPolicyError(
                f"routing override for '{clean_model}' cannot repeat providers in 'allowed_providers'"
            )

        excluded_providers = override.get("excluded_providers", [])
        if not isinstance(excluded_providers, list):
            raise RoutingPolicyError(
                f"routing override for '{clean_model}' must use a list for 'excluded_providers'"
            )

        if override.get("allow_global_fallback", False):
            raise RoutingPolicyError(
                f"routing override for '{clean_model}' cannot enable 'allow_global_fallback' in v2"
            )

        fallback_providers = override.get("fallback_providers", [])
        if fallback_providers not in (None, []):
            raise RoutingPolicyError(
                f"routing override for '{clean_model}' cannot define 'fallback_providers' in v2"
            )

        for provider in allowed_providers:
            if not isinstance(provider, str) or not provider:
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' must use string providers in 'allowed_providers'"
                )
            self._validate_provider_name(provider, clean_model, "allowed_providers")

        for provider in excluded_providers:
            if not isinstance(provider, str) or not provider:
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' must use string providers in 'excluded_providers'"
                )
            self._validate_provider_name(provider, clean_model, "excluded_providers")

        if set(allowed_providers) & set(excluded_providers):
            raise RoutingPolicyError(
                f"routing override for '{clean_model}' cannot include the same provider in both 'allowed_providers' and 'excluded_providers'"
            )

        if set(weights.keys()) != set(allowed_providers):
            raise RoutingPolicyError(
                f"routing override for '{clean_model}' must use matching providers in 'weights' and 'allowed_providers'"
            )

        total_weight = 0.0
        for provider, weight in weights.items():
            if not isinstance(provider, str) or not provider:
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' must use string providers in 'weights'"
                )
            self._validate_provider_name(provider, clean_model, "weights")
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' must use numeric weights"
                )
            if weight < 0:
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' cannot use negative weights"
                )
            total_weight += float(weight)
            self._validate_provider_model(provider, clean_model)

        if total_weight <= 0:
            raise RoutingPolicyError(
                f"routing override for '{clean_model}' must define weights with a total greater than zero"
            )

    def _validate(self) -> None:
        for clean_model, override in self.model_overrides.items():
            if not isinstance(clean_model, str) or not clean_model:
                raise RoutingPolicyError("routing override keys must be non-empty model names")
            if not isinstance(override, dict):
                raise RoutingPolicyError(f"routing override for '{clean_model}' must be an object")

            strategy = override.get("strategy")
            if strategy not in {"single", "weighted"}:
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' must use strategy 'single' or 'weighted'"
                )

            if strategy == "weighted":
                self._validate_weighted_override(clean_model, override)
                continue

            primary = override.get("primary")
            if not isinstance(primary, str) or not primary:
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' requires a non-empty 'primary' provider"
                )
            self._validate_provider_name(primary, clean_model, "primary")

            allowed_providers = override.get("allowed_providers", [primary])
            if not isinstance(allowed_providers, list) or not all(
                isinstance(provider, str) and provider for provider in allowed_providers
            ):
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' must use a string list for 'allowed_providers'"
                )
            if allowed_providers != [primary]:
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' must restrict 'allowed_providers' to ['{primary}'] in v1"
                )

            fallback_providers = override.get("fallback_providers", [])
            if fallback_providers not in (None, []):
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' cannot define 'fallback_providers' in v1"
                )

            self._validate_provider_model(primary, clean_model)

    def _select_weighted_provider(self, clean_model: str, weights: Dict[str, Any]) -> str:
        total_weight = sum(float(weight) for weight in weights.values())
        if total_weight <= 0:
            raise RoutingPolicyError(
                f"routing override for '{clean_model}' must define weights with a total greater than zero"
            )

        target = self.rng.uniform(0, total_weight)
        running_total = 0.0
        last_provider = None
        for provider, weight in weights.items():
            running_total += float(weight)
            last_provider = provider
            if target <= running_total:
                return provider

        if last_provider is None:
            raise RoutingPolicyError(
                f"routing override for '{clean_model}' produced no selectable providers"
            )
        return last_provider

    def resolve(self, model: str) -> RouteDecision:
        if "/" not in model:
            return RouteDecision(
                requested_model=model,
                clean_model=model,
                selected_provider=None,
                rewritten_model=model,
                strategy="passthrough",
                selection_source="passthrough",
                override_applied=False,
                candidate_providers=[],
                strict=False,
                allow_global_fallback=True,
                excluded_providers=[],
            )

        provider, clean_model = model.split("/", 1)
        if provider != "weighted-router":
            return RouteDecision(
                requested_model=model,
                clean_model=clean_model,
                selected_provider=provider,
                rewritten_model=model,
                strategy="passthrough",
                selection_source="passthrough",
                override_applied=False,
                candidate_providers=[provider],
                strict=False,
                allow_global_fallback=True,
                excluded_providers=[],
            )

        override = self.model_overrides.get(clean_model)
        if override is None:
            raise RoutingPolicyError(
                f"No routing override configured for weighted-router model '{clean_model}'"
            )

        strategy = override["strategy"]
        if strategy == "weighted":
            selected_provider = self._select_weighted_provider(clean_model, override["weights"])
            candidate_providers = list(override["allowed_providers"])
            excluded_providers = list(override.get("excluded_providers", []))
            selection_source = "model_override_weighted"
        else:
            selected_provider = override["primary"]
            candidate_providers = [selected_provider]
            excluded_providers = []
            selection_source = "model_override"

        return RouteDecision(
            requested_model=model,
            clean_model=clean_model,
            selected_provider=selected_provider,
            rewritten_model=f"{selected_provider}/{clean_model}",
            strategy=strategy,
            selection_source=selection_source,
            override_applied=True,
            candidate_providers=candidate_providers,
            strict=bool(override.get("strict", True)),
            allow_global_fallback=bool(override.get("allow_global_fallback", False)),
            excluded_providers=excluded_providers,
            reason=override.get("reason"),
        )
