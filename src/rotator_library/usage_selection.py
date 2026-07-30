from __future__ import annotations

import random


def select_weighted_random(candidates: list[tuple[str, int]], tolerance: float) -> str:
    if not candidates:
        raise ValueError("Cannot select from empty candidate list")
    if len(candidates) == 1:
        return candidates[0][0]
    max_usage = max(usage for _, usage in candidates)
    weights = [(max_usage - usage) + tolerance + 1 for _, usage in candidates]
    selected: str = random.choices(
        [credential for credential, _ in candidates], weights=weights, k=1
    )[0]
    return selected
