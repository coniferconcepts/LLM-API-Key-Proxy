"""LiteLLM SDK Anthropic /v1/messages path stays Python SDK, not LITELLM_RUST."""

from __future__ import annotations

import inspect
import os

import litellm

from proxy_app.litellm_loader import load_litellm
from rotator_library.client import RotatingClient


def test_anthropic_messages_uses_sdk_acompletion_translator():
    source = inspect.getsource(RotatingClient.anthropic_messages)
    assert "translate_anthropic_request" in source
    assert "self.acompletion" in source
    assert "LITELLM_RUST" not in source


def test_litellm_loader_does_not_enable_rust_gateway():
    source = inspect.getsource(load_litellm)
    assert "LITELLM_RUST" not in source
    assert os.environ.get("LITELLM_RUST") in {None, "", "0", "false", "False"}


def test_sdk_exposes_acompletion_not_proxy_listener():
    assert hasattr(litellm, "acompletion")
    assert not hasattr(litellm, "is_request_body_safe")


def test_go_messages_runtime_base_omits_v1():
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from router_runtime.provider_endpoint_contracts import PROVIDER_ENDPOINT_CONTRACTS

    contract = PROVIDER_ENDPOINT_CONTRACTS["opencode_go_messages"]
    assert contract.runtime_base_default == "https://opencode.ai/zen/go"
    assert contract.runtime_base_must_end_with_v1 is False
    assert contract.expected_endpoint_suffix == "/v1/messages"
