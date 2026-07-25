# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from __future__ import annotations

import os
from typing import Any, Dict, Optional

# ChatGPT/Codex-backed OpenAI-compatible sidecars (e.g. codex-lb on OPENAI_API_BASE)
# reject several OpenAI Platform fields that AI SDK / OpenCode still emit.
_CODEX_LB_UNSUPPORTED_CHAT_PARAMS = frozenset(
    {
        "user",
    }
)
_CODEX_LB_UNSUPPORTED_SERVICE_TIERS = frozenset({"auto", "default"})


def _requires_reasoning_content_placeholder(message: Dict[str, Any]) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("role") != "assistant":
        return False
    if not message.get("tool_calls") and not message.get("function_call"):
        return False
    return "reasoning_content" not in message or message.get("reasoning_content") is None


def _ensure_reasoning_content_for_tool_history(payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload

    for message in messages:
        if _requires_reasoning_content_placeholder(message):
            message["reasoning_content"] = ""
    return payload


def _payload_uses_tooling(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        return True
    if payload.get("tool_choice") not in (None, "none", "auto"):
        return True
    functions = payload.get("functions")
    if isinstance(functions, list) and functions:
        return True
    if payload.get("function_call") not in (None, "none", "auto"):
        return True
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("tool_calls") or message.get("function_call"):
                return True
            if message.get("role") == "tool":
                return True
    return False


def _disable_thinking_for_opencode_go_tooling(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "thinking" in payload and _payload_uses_tooling(payload):
        del payload["thinking"]
    return payload


def _is_openai_family_model(model: str) -> bool:
    if not isinstance(model, str) or not model:
        return False
    lowered = model.lower()
    if lowered.startswith("openai/"):
        return True
    bare = lowered.split("/", 1)[-1]
    return (
        bare.startswith("gpt-")
        or bare.startswith("o1")
        or bare.startswith("o3")
        or bare.startswith("o4")
        or bare.startswith("chatgpt-")
    )


def _openai_api_base_is_local_override(api_base: Optional[str] = None) -> bool:
    """True when OPENAI_API_BASE points at a local OpenAI-compatible sidecar (codex-lb)."""
    base = (
        (api_base if api_base is not None else os.getenv("OPENAI_API_BASE") or "").strip().lower()
    )
    if not base:
        return False
    return any(
        token in base
        for token in (
            "127.0.0.1",
            "localhost",
            "0.0.0.0",
            "[::1]",
            "2455",
        )
    )


def _strip_codex_lb_incompatible_chat_params(
    payload: Dict[str, Any], model: str, *, api_base: Optional[str] = None
) -> Dict[str, Any]:
    """Drop fields ChatGPT/Codex sidecars reject that Platform OpenAI still accepts.

    OpenCode / AI SDK routinely send ``user`` (and sometimes ``service_tier=auto``).
    codex-lb returns invalid_request_error for those, which Mirrowel surfaces as a
    public stream error and can open the openai circuit.
    """
    if not _is_openai_family_model(model):
        return payload
    if not _openai_api_base_is_local_override(api_base):
        return payload

    for key in _CODEX_LB_UNSUPPORTED_CHAT_PARAMS:
        payload.pop(key, None)

    service_tier = payload.get("service_tier")
    if (
        isinstance(service_tier, str)
        and service_tier.strip().lower() in _CODEX_LB_UNSUPPORTED_SERVICE_TIERS
    ):
        payload.pop("service_tier", None)

    return payload


def sanitize_request_payload(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Removes unsupported parameters from the request payload based on the model.
    """
    if "dimensions" in payload and not model.startswith("openai/text-embedding-3"):
        del payload["dimensions"]

    if payload.get("thinking") == {"type": "enabled", "budget_tokens": -1}:
        if model not in ["gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash"]:
            del payload["thinking"]

    if model.startswith("opencode_go/"):
        payload = _ensure_reasoning_content_for_tool_history(payload)
        payload = _disable_thinking_for_opencode_go_tooling(payload)

    payload = _strip_codex_lb_incompatible_chat_params(payload, model)

    return payload
