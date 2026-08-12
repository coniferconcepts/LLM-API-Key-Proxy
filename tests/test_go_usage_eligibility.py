# SPDX-License-Identifier: LGPL-3.0-only

from rotator_library.go_usage.eligibility import (
    collect_go_credentials,
    summarize_go_eligibility,
    unique_credentials,
)


def test_unique_and_collect_dedup_same_key_across_namespaces() -> None:
    keys = {
        "opencode_go": ["alpha", "beta"],
        "opencode_go_messages": ["alpha", "gamma"],
        "chutes": ["ignored"],
    }
    assert unique_credentials(["alpha", "alpha", "", "beta"]) == ["alpha", "beta"]
    assert collect_go_credentials(keys) == ["alpha", "beta", "gamma"]


def test_eligibility_uses_only_key_and_go_cooldowns() -> None:
    now = 1_000_000.0
    records = {
        "a": {
            "go_usage_cooldown_until": now + 100,
            "model_cooldowns": {"glm-5.2": now + 999},
        },
        "b": {"key_cooldown_until": 0, "go_usage_cooldown_until": None},
    }
    summary = summarize_go_eligibility(records, ["a", "b"], now=now)
    assert summary["schema_version"] == "go_usage_eligibility.v1"
    assert summary["unique_keys"] == 2
    assert summary["eligible"] == 1
    assert summary["blocked"] == 1
    assert summary["soonest_reset_unix"] == int(now + 100)


def test_model_cooldown_alone_does_not_block_family() -> None:
    now = 1_000_000.0
    records = {"a": {"model_cooldowns": {"glm-5.2": now + 999}}}
    summary = summarize_go_eligibility(records, ["a"], now=now)
    assert summary["eligible"] == 1
    assert summary["blocked"] == 0
    assert summary["soonest_reset_unix"] is None


def test_all_blocked_reports_soonest() -> None:
    now = 1_000_000.0
    records = {
        "a": {"go_usage_cooldown_until": now + 50},
        "b": {"go_usage_cooldown_until": now + 10},
    }
    summary = summarize_go_eligibility(records, ["a", "b"], now=now)
    assert summary["eligible"] == 0
    assert summary["blocked"] == 2
    assert summary["soonest_reset_unix"] == int(now + 10)
