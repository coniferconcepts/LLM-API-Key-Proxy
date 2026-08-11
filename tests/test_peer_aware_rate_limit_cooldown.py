# SPDX-License-Identifier: LGPL-3.0-only
"""Peer-aware provider cooldown: multi-key 429 must not freeze the whole provider."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rotator_library.client import RotatingClient
from rotator_library.cooldown_manager import (
    CooldownManager,
    has_untried_peer_credentials,
    should_apply_provider_cooldown_for_rate_limit_error,
)


def test_has_untried_peer_when_second_key_not_tried():
    creds = ["key-a", "key-b"]
    tried = {"key-a"}  # current already counted after acquire
    assert has_untried_peer_credentials(creds, tried) is True


def test_no_untried_peer_when_all_tried():
    creds = ["key-a", "key-b"]
    tried = {"key-a", "key-b"}
    assert has_untried_peer_credentials(creds, tried) is False


def test_no_untried_peer_single_key():
    assert has_untried_peer_credentials(["key-a"], {"key-a"}) is False


@pytest.mark.parametrize(
    "status_code,error_type,expected",
    [
        (429, "rate_limit", True),
        (429, "quota_exceeded", False),
        (429, "unknown", True),
        (500, "rate_limit", True),
        (500, "server_error", False),
        (None, "rate_limit", True),
        (None, "auth_error", False),
    ],
)
def test_should_apply_provider_cooldown_for_rate_limit_error(status_code, error_type, expected):
    assert (
        should_apply_provider_cooldown_for_rate_limit_error(
            status_code=status_code, error_type=error_type
        )
        is expected
    )


def _bare_client() -> RotatingClient:
    client = object.__new__(RotatingClient)
    client.cooldown_manager = CooldownManager()
    return client


@pytest.mark.asyncio
async def test_maybe_start_skips_provider_cool_when_peer_remains():
    client = _bare_client()
    classified = SimpleNamespace(status_code=429, error_type="rate_limit", retry_after=60)
    # key-a already tried (current); key-b still available
    did = await client._maybe_start_provider_cooldown_on_rate_limit(
        provider="opencode_go",
        credentials_for_provider=["key-a", "key-b"],
        tried_creds={"key-a"},
        classified_error=classified,
    )
    assert did is False
    assert await client.cooldown_manager.is_cooling_down("opencode_go") is False


@pytest.mark.asyncio
async def test_maybe_start_provider_cools_when_no_peers():
    client = _bare_client()
    classified = SimpleNamespace(status_code=429, error_type="rate_limit", retry_after=45)
    did = await client._maybe_start_provider_cooldown_on_rate_limit(
        provider="opencode_go",
        credentials_for_provider=["key-a", "key-b"],
        tried_creds={"key-a", "key-b"},
        classified_error=classified,
    )
    assert did is True
    assert await client.cooldown_manager.is_cooling_down("opencode_go") is True
    remaining = await client.cooldown_manager.get_cooldown_remaining("opencode_go")
    assert remaining > 0
    assert remaining <= 45


@pytest.mark.asyncio
async def test_maybe_start_single_key_still_provider_cools():
    client = _bare_client()
    classified = SimpleNamespace(status_code=429, error_type="rate_limit", retry_after=30)
    did = await client._maybe_start_provider_cooldown_on_rate_limit(
        provider="chutes",
        credentials_for_provider=["only-key"],
        tried_creds={"only-key"},
        classified_error=classified,
    )
    assert did is True
    assert await client.cooldown_manager.is_cooling_down("chutes") is True


@pytest.mark.asyncio
async def test_maybe_start_skips_non_rate_limit_errors():
    client = _bare_client()
    classified = SimpleNamespace(status_code=401, error_type="auth_error", retry_after=None)
    did = await client._maybe_start_provider_cooldown_on_rate_limit(
        provider="opencode_go",
        credentials_for_provider=["key-a"],
        tried_creds={"key-a"},
        classified_error=classified,
    )
    assert did is False
    assert await client.cooldown_manager.is_cooling_down("opencode_go") is False


@pytest.mark.asyncio
async def test_quota_exceeded_does_not_provider_cool_via_helper():
    """quota_exceeded stays credential/exhaustion path — not short provider cool."""
    client = _bare_client()
    classified = SimpleNamespace(status_code=429, error_type="quota_exceeded", retry_after=3600)
    did = await client._maybe_start_provider_cooldown_on_rate_limit(
        provider="opencode_go",
        credentials_for_provider=["key-a"],
        tried_creds={"key-a"},
        classified_error=classified,
    )
    assert did is False
    assert await client.cooldown_manager.is_cooling_down("opencode_go") is False
