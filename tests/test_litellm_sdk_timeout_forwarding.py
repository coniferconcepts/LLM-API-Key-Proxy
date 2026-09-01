"""LiteLLM SDK timeout forwarding: streaming read default 300s."""

from __future__ import annotations

import inspect

import litellm
import pytest

from rotator_library.timeout_config import TimeoutConfig


@pytest.fixture
def default_read_timeouts(monkeypatch):
    monkeypatch.delenv("TIMEOUT_READ_STREAMING", raising=False)
    monkeypatch.delenv("TIMEOUT_READ_NON_STREAMING", raising=False)


def test_streaming_read_default_is_300_seconds(default_read_timeouts):
    assert TimeoutConfig._READ_STREAMING == 300.0
    assert TimeoutConfig.read_streaming() == 300.0
    assert TimeoutConfig.litellm_timeout_seconds(stream=True) == 300.0
    assert TimeoutConfig.litellm_timeout_seconds(stream=False) == 600.0


def test_apply_litellm_timeout_forwards_streaming_read_default(default_read_timeouts):
    streamed = TimeoutConfig.apply_litellm_timeout({"model": "x", "stream": True})
    assert streamed["timeout"] == 300.0
    complete = TimeoutConfig.apply_litellm_timeout({"model": "x", "stream": False})
    assert complete["timeout"] == 600.0
    preserved = TimeoutConfig.apply_litellm_timeout({"timeout": 12, "stream": True})
    assert preserved["timeout"] == 12


def test_sdk_acompletion_still_accepts_timeout_kwarg(default_read_timeouts):
    assert "timeout" in inspect.signature(litellm.acompletion).parameters
    forwarded = TimeoutConfig.apply_litellm_timeout({"stream": True})
    assert forwarded["timeout"] == TimeoutConfig.litellm_timeout_seconds(stream=True)
