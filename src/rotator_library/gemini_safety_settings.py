"""Default safety settings for Gemini requests."""

from typing import Any, Dict


def apply_default_safety_settings(litellm_kwargs: Dict[str, Any], provider: str):
    """
    Ensure default Gemini safety settings are present when calling the Gemini provider.
    This will not override any explicit settings provided by the request. It accepts
    either OpenAI-compatible generic `safety_settings` (dict) or direct Gemini-style
    `safetySettings` (list of dicts). Missing categories will be added with safe defaults.
    """
    if provider != "gemini":
        return

    # Generic defaults (openai-compatible style)
    default_generic = {
        "harassment": "OFF",
        "hate_speech": "OFF",
        "sexually_explicit": "OFF",
        "dangerous_content": "OFF",
        "civic_integrity": "BLOCK_NONE",
    }

    # Gemini defaults (direct Gemini format)
    default_gemini = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
    ]

    # If generic form is present, ensure missing generic keys are filled in
    if "safety_settings" in litellm_kwargs and isinstance(
        litellm_kwargs["safety_settings"], dict
    ):
        for k, v in default_generic.items():
            if k not in litellm_kwargs["safety_settings"]:
                litellm_kwargs["safety_settings"][k] = v
        return

    # If Gemini form is present, ensure missing gemini categories are appended
    if "safetySettings" in litellm_kwargs and isinstance(
        litellm_kwargs["safetySettings"], list
    ):
        present = {
            item.get("category")
            for item in litellm_kwargs["safetySettings"]
            if isinstance(item, dict)
        }
        for d in default_gemini:
            if d["category"] not in present:
                litellm_kwargs["safetySettings"].append(d)
        return

    # Neither present: set generic defaults so provider conversion will translate them
    if "safety_settings" not in litellm_kwargs and "safetySettings" not in litellm_kwargs:
        litellm_kwargs["safety_settings"] = default_generic.copy()
