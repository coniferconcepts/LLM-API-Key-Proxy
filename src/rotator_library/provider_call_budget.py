from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_PROVIDER_CALLS_HEADER = b"x-opencode-max-provider-calls"
MAX_PROVIDER_CALLS_CEILING = 1000


class InvalidProviderCallBudget(ValueError):
    pass


class ProviderCallBudgetExhausted(Exception):
    def __init__(self, first_failure: Exception | None) -> None:
        self.first_failure = first_failure
        super().__init__("provider call budget exhausted")


@dataclass(slots=True)
class ProviderCallBudget:
    maximum: int | None
    admitted: int = 0

    @property
    def exhausted(self) -> bool:
        return self.maximum is not None and self.admitted >= self.maximum

    def admit(self, first_failure: Exception | None = None) -> None:
        if self.maximum is not None and self.admitted >= self.maximum:
            raise ProviderCallBudgetExhausted(first_failure)
        self.admitted += 1


def parse_provider_call_budget(request: Any) -> ProviderCallBudget:
    scope = getattr(request, "scope", {})
    raw_headers = scope.get("headers", ()) if isinstance(scope, dict) else ()
    values = [value for name, value in raw_headers if name.lower() == MAX_PROVIDER_CALLS_HEADER]
    if not values:
        return ProviderCallBudget(None)
    if len(values) != 1:
        raise InvalidProviderCallBudget("provider call ceiling header must appear exactly once")
    raw_value = values[0]
    if not raw_value.isascii() or not raw_value.isdigit():
        raise InvalidProviderCallBudget("provider call ceiling must be a positive decimal integer")
    if len(raw_value) > 1 and raw_value.startswith(b"0"):
        raise InvalidProviderCallBudget("provider call ceiling must use canonical decimal form")
    maximum = int(raw_value)
    if maximum < 1 or maximum > MAX_PROVIDER_CALLS_CEILING:
        raise InvalidProviderCallBudget(
            f"provider call ceiling must be between 1 and {MAX_PROVIDER_CALLS_CEILING}"
        )
    return ProviderCallBudget(maximum)
