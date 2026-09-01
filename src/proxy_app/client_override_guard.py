"""Reject client-controlled provider routing and credential overrides."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

# Identity, endpoint, and credential fields LiteLLM Proxy bans in
# is_request_body_safe. Observability callback hosts are out of scope here.
_BLOCKED_FIELDS = frozenset(
    {
        "api_base",
        "base_url",
        "user_config",
        "api_key",
        "model_list",
        "fallbacks",
        "aws_sts_endpoint",
        "aws_web_identity_token",
        "aws_role_name",
        "aws_bedrock_runtime_endpoint",
        "aws_bedrock_project_id",
        "bedrock_tags",
        "vertex_credentials",
        "azure_ad_token",
        "s3_endpoint_url",
        "sagemaker_base_url",
        "deployment_url",
        "nvcf_function_id",
        "use_ssl",
    }
)

# Single-level containers. Do not recurse; Proxy walks these by name only.
_NESTED_CONTAINERS = (
    "extra_body",
    "metadata",
    "litellm_metadata",
    "litellm_params",
    "litellm_embedding_config",
)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        stripped = value.lstrip()
        if not stripped.startswith("{"):
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return parsed
    return None


def _first_blocked_field(container: Mapping[str, Any], prefix: str = "") -> str | None:
    for field in sorted(_BLOCKED_FIELDS):
        if field in container:
            return f"{prefix}{field}" if prefix else field
    return None


def find_client_override(payload: Any) -> str | None:
    """Return the first forbidden client-controlled field, if present."""
    if not isinstance(payload, Mapping):
        return None

    blocked = _first_blocked_field(payload)
    if blocked is not None:
        return blocked

    for key in _NESTED_CONTAINERS:
        nested = _as_mapping(payload.get(key))
        if nested is None:
            continue
        blocked = _first_blocked_field(nested, f"{key}.")
        if blocked is not None:
            return blocked
        if key == "litellm_params":
            nested_metadata = _as_mapping(nested.get("metadata"))
            if nested_metadata is not None:
                blocked = _first_blocked_field(
                    nested_metadata,
                    "litellm_params.metadata.",
                )
                if blocked is not None:
                    return blocked

    return None
