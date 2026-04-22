# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from typing import Dict, Any


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
             
    return payload
