from __future__ import annotations

import importlib
from pathlib import Path

import litellm
import pytest

from rotator_library import pricing
from rotator_library.usage_manager import UsageManager


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("model", "expected_cost"),
    [
        ("chutes/moonshotai/Kimi-K3-TEE", 49.2e-6),
        ("chutes/deepseek-ai/DeepSeek-V4-Flash-0731-TEE", 5.456e-6),
    ],
)
@pytest.mark.asyncio
async def test_pinned_pricing_records_streaming_and_non_streaming_costs(
    tmp_path: Path, streaming: bool, model: str, expected_cost: float
) -> None:
    response = litellm.ModelResponse(
        model=model,
        stream=streaming,
        usage=litellm.Usage(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            prompt_tokens_details={"cached_tokens": 4},
        ),
    )
    manager = UsageManager(tmp_path / "usage.json")

    await manager.record_success("opaque-credential", model, response)

    model_data = manager._usage_data["opaque-credential"]["daily"]["models"][model]  # noqa: SLF001
    assert model_data["prompt_tokens"] == 6
    assert model_data["prompt_tokens_cached"] == 4
    assert model_data["completion_tokens"] == 2
    assert model_data["approx_cost"] == pytest.approx(expected_cost)


def test_double_prefixed_chutes_model_cost_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_cost_call(**_kwargs: object) -> float:
        raise AssertionError("unknown prices must not be passed to LiteLLM")

    monkeypatch.setattr(litellm, "completion_cost", unexpected_cost_call)

    assert (
        pricing.completion_cost_or_unknown(object(), "chutes/chutes/moonshotai/Kimi-K3-TEE") is None
    )


def test_pinned_registration_does_not_fetch_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    def network_fetch_is_forbidden(*_args: object, **_kwargs: object) -> dict:
        raise AssertionError("pricing registration must not fetch a catalog")

    monkeypatch.setattr(litellm, "get_model_cost_map", network_fetch_is_forbidden)

    reloaded = importlib.reload(pricing)

    assert reloaded.CHUTES_CATALOG_DATE == "2026-09-24"
    assert set(reloaded.PINNED_MODEL_PRICING) == {
        "chutes/moonshotai/Kimi-K3-TEE",
        "chutes/deepseek-ai/DeepSeek-V4-Flash-0731-TEE",
    }
