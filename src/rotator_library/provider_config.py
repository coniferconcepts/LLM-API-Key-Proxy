# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/provider_config.py
"""
Centralized provider configuration for the rotator library.

This module handl- Known LiteLLM provider definitions (from scraped data)
- UI configuration (categories, notes, extra vars) for credential tool
- API base overrides for known providers
- Custom OpenAI-compatible provider detection and routing
"""

import os
import logging
from typing import Dict, Any, Set, Optional

from .litellm_providers import (
    SCRAPED_PROVIDERS,
)
from .credential_discovery import (
    discover_api_keys_from_env as discover_api_keys_from_env,
    normalize_credential_provider as normalize_credential_provider,
)

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# LiteLLM Provider UI Configuration
#
# Keys are route-based provider identifiers (e.g., "openai", "anthropic").
# Provider data (display_name, api_key_env_vars, etc.) comes from SCRAPED_PROVIDERS.
#
# This dict only contains UI-specific configuration:
#   - category: Provider category for display grouping
#   - note: (optional) Configuration notes shown to user
#   - extra_vars: (optional) Additional env vars to prompt for [(name, label, default), ...]
# =============================================================================

LITELLM_PROVIDERS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # POPULAR - Most commonly used providers
    # =========================================================================
    "openai": {
        "category": "popular",
    },
    "anthropic": {
        "category": "popular",
    },
    "gemini": {
        "category": "popular",
    },
    "xai": {
        "category": "popular",
    },
    "deepseek": {
        "category": "popular",
    },
    "mistral": {
        "category": "popular",
    },
    "codestral": {
        "category": "popular",
    },
    "openrouter": {
        "category": "popular",
        "extra_vars": [
            ("OPENROUTER_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "groq": {
        "category": "popular",
    },
    "chutes": {
        "category": "popular",
    },
    "nvidia_nim": {
        "category": "popular",
        "extra_vars": [
            ("NVIDIA_NIM_API_BASE", "NIM API Base (optional)", None),
        ],
    },
    "perplexity": {
        "category": "popular",
    },
    "moonshot": {
        "category": "popular",
        "extra_vars": [
            ("MOONSHOT_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "zai": {
        "category": "popular",
    },
    "minimax": {
        "category": "popular",
        "extra_vars": [
            ("MINIMAX_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "xiaomi_mimo": {
        "category": "popular",
    },
    "nano-gpt": {
        "category": "popular",
    },
    "synthetic": {
        "category": "popular",
    },
    # =========================================================================
    # CLOUD PLATFORMS - Aggregators & cloud inference platforms
    # =========================================================================
    "together_ai": {
        "category": "cloud",
    },
    "fireworks": {
        "category": "cloud",
        "extra_vars": [
            ("FIREWORKS_V2_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "replicate": {
        "category": "cloud",
    },
    "deepinfra": {
        "category": "cloud",
    },
    "anyscale": {
        "category": "cloud",
    },
    "baseten": {
        "category": "cloud",
    },
    "predibase": {
        "category": "cloud",
    },
    "novita": {
        "category": "cloud",
    },
    "featherless_ai": {
        "category": "cloud",
    },
    "hyperbolic": {
        "category": "cloud",
    },
    "lambda_ai": {
        "category": "cloud",
        "extra_vars": [
            ("LAMBDA_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "nebius": {
        "category": "cloud",
    },
    "galadriel": {
        "category": "cloud",
    },
    "friendliai": {
        "category": "cloud",
    },
    "sambanova": {
        "category": "cloud",
    },
    "cerebras": {
        "category": "cloud",
    },
    "meta_llama": {
        "category": "cloud",
    },
    "ai21": {
        "category": "cloud",
    },
    "cohere_chat": {
        "category": "cloud",
    },
    "alephalpha": {
        "category": "cloud",
    },
    "huggingface": {
        "category": "cloud",
    },
    "github": {
        "category": "cloud",
    },
    "helicone": {
        "category": "cloud",
        "note": "LLM gateway/proxy with analytics.",
    },
    "heroku": {
        "category": "cloud",
        "extra_vars": [
            (
                "HEROKU_API_BASE",
                "Heroku Inference URL",
                "https://us.inference.heroku.com",
            ),
        ],
    },
    "morph": {
        "category": "cloud",
    },
    "poe": {
        "category": "cloud",
    },
    "llamagate": {
        "category": "cloud",
    },
    "manus": {
        "category": "cloud",
    },
    # =========================================================================
    # ENTERPRISE / COMPLEX AUTH - Major cloud providers (may need extra config)
    # =========================================================================
    "azure": {
        "category": "enterprise",
        "note": "Requires Azure endpoint and API version.",
        "extra_vars": [
            ("AZURE_API_BASE", "Azure endpoint URL", None),
            ("AZURE_API_VERSION", "API version", "2024-02-15-preview"),
        ],
    },
    "azure_ai": {
        "category": "enterprise",
        "extra_vars": [
            ("AZURE_AI_API_BASE", "Azure AI endpoint URL", None),
        ],
    },
    "vertex_ai": {
        "category": "enterprise",
        "note": "Uses Google Cloud service account. Enter path to credentials JSON file.",
        "extra_vars": [
            ("VERTEXAI_PROJECT", "GCP Project ID", None),
            ("VERTEXAI_LOCATION", "GCP Location", "us-central1"),
        ],
    },
    "bedrock": {
        "category": "enterprise",
        "note": "Requires all three AWS credentials.",
        "extra_vars": [
            ("AWS_SECRET_ACCESS_KEY", "AWS Secret Access Key", None),
            ("AWS_REGION_NAME", "AWS Region", "us-east-1"),
        ],
    },
    "sagemaker": {
        "category": "enterprise",
        "note": "Requires all three AWS credentials.",
        "extra_vars": [
            ("AWS_SECRET_ACCESS_KEY", "AWS Secret Access Key", None),
            ("AWS_REGION_NAME", "AWS Region", "us-east-1"),
        ],
    },
    "databricks": {
        "category": "enterprise",
        "extra_vars": [
            ("DATABRICKS_API_BASE", "Databricks workspace URL", None),
        ],
    },
    "snowflake": {
        "category": "enterprise",
        "note": "Uses JWT authentication.",
        "extra_vars": [
            ("SNOWFLAKE_ACCOUNT_ID", "Snowflake Account ID", None),
        ],
    },
    "watsonx": {
        "category": "enterprise",
        "extra_vars": [
            ("WATSONX_URL", "watsonx.ai URL (optional)", None),
        ],
    },
    "cloudflare": {
        "category": "enterprise",
        "extra_vars": [
            ("CLOUDFLARE_ACCOUNT_ID", "Cloudflare Account ID", None),
        ],
    },
    "oci": {
        "category": "enterprise",
        "note": "Oracle Cloud Infrastructure. Requires OCI SDK configuration.",
    },
    "sap": {
        "category": "enterprise",
        "note": "SAP Generative AI Hub. Requires AI Core configuration.",
    },
    # =========================================================================
    # SPECIALIZED - Image, audio, embeddings, rerank providers
    # =========================================================================
    "stability": {
        "category": "specialized",
        "note": "Image generation provider.",
    },
    "fal_ai": {
        "category": "specialized",
        "note": "Image generation provider.",
    },
    "runwayml": {
        "category": "specialized",
        "note": "Image generation provider.",
    },
    "recraft": {
        "category": "specialized",
        "note": "Image generation and editing.",
        "extra_vars": [
            ("RECRAFT_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "topaz": {
        "category": "specialized",
        "note": "Image enhancement provider.",
    },
    "elevenlabs": {
        "category": "specialized",
        "note": "Text-to-speech and audio transcription.",
    },
    "deepgram": {
        "category": "specialized",
        "note": "Audio transcription provider.",
    },
    "voyage": {
        "category": "specialized",
        "note": "Embeddings and rerank provider.",
    },
    "jina_ai": {
        "category": "specialized",
        "note": "Embeddings and rerank provider.",
    },
    "clarifai": {
        "category": "specialized",
    },
    "nlp_cloud": {
        "category": "specialized",
    },
    "milvus": {
        "category": "specialized",
        "note": "Vector database provider.",
        "extra_vars": [
            ("MILVUS_API_BASE", "Milvus Server URL", None),
        ],
    },
    "infinity": {
        "category": "specialized",
        "note": "Self-hosted embeddings/rerank server. API key is optional.",
        "extra_vars": [
            ("INFINITY_API_BASE", "Infinity Server URL", "http://localhost:8080"),
        ],
    },
    # =========================================================================
    # REGIONAL - Region-specific or specialized regional providers
    # =========================================================================
    "dashscope": {
        "category": "regional",
        "note": "Alibaba Cloud Qwen models.",
    },
    "volcengine": {
        "category": "regional",
        "note": "ByteDance cloud platform.",
    },
    "ovhcloud": {
        "category": "regional",
        "note": "European cloud provider.",
    },
    "nscale": {
        "category": "regional",
        "note": "EU sovereign cloud.",
    },
    # =========================================================================
    # LOCAL / SELF-HOSTED - Run locally or on your own infrastructure
    # =========================================================================
    "lm_studio": {
        "category": "local",
        "note": "Local provider. API key is optional. Start LM Studio server first.",
        "extra_vars": [
            ("LM_STUDIO_API_BASE", "API Base URL", "http://localhost:1234/v1"),
        ],
    },
    "hosted_vllm": {
        "category": "local",
        "note": "Self-hosted vLLM server. API key is optional.",
        "extra_vars": [
            ("HOSTED_VLLM_API_BASE", "vLLM Server URL", None),
        ],
    },
    "xinference": {
        "category": "local",
        "note": "Local Xinference server. API key is optional.",
        "extra_vars": [
            ("XINFERENCE_API_BASE", "Xinference URL", "http://127.0.0.1:9997/v1"),
        ],
    },
    "litellm_proxy": {
        "category": "local",
        "note": "Self-hosted LiteLLM Proxy gateway.",
        "extra_vars": [
            ("LITELLM_PROXY_API_BASE", "LiteLLM Proxy URL", "http://localhost:4000"),
        ],
    },
    "langgraph": {
        "category": "local",
        "note": "Self-hosted LangGraph server.",
        "extra_vars": [
            ("LANGGRAPH_API_BASE", "LangGraph URL", "http://localhost:2024"),
        ],
    },
    "ragflow": {
        "category": "local",
        "note": "Self-hosted RAGFlow server.",
        "extra_vars": [
            ("RAGFLOW_API_BASE", "RAGFlow URL", "http://localhost:9380"),
        ],
    },
    "docker_model_runner": {
        "category": "local",
        "note": "Local Docker Model Runner. API key is optional.",
        "extra_vars": [
            (
                "DOCKER_MODEL_RUNNER_API_BASE",
                "Docker Model Runner URL",
                "http://localhost:22088",
            ),
        ],
    },
    "lemonade": {
        "category": "local",
        "note": "Local proxy. API key is optional.",
        "extra_vars": [
            ("LEMONADE_API_BASE", "Lemonade URL", "http://localhost:8000/api/v1"),
        ],
    },
    # NOTE: ollama, llamafile, petals, triton are in PROVIDER_BLACKLIST
    # because they don't use standard API key authentication.
    # Use "Add Custom OpenAI-Compatible Provider" for these.
    # =========================================================================
    # OTHER - Miscellaneous providers
    # =========================================================================
    "aiml": {
        "category": "other",
        "extra_vars": [
            ("AIML_API_BASE", "API Base URL (optional)", None),
        ],
    },
    "abliteration": {
        "category": "other",
    },
    "amazon_nova": {
        "category": "other",
    },
    "apertis": {
        "category": "other",
    },
    "bytez": {
        "category": "other",
    },
    "cometapi": {
        "category": "other",
    },
    "compactifai": {
        "category": "other",
    },
    "datarobot": {
        "category": "other",
        "extra_vars": [
            ("DATAROBOT_API_BASE", "DataRobot URL", "https://app.datarobot.com"),
        ],
    },
    "gradient_ai": {
        "category": "other",
        "extra_vars": [
            ("GRADIENT_AI_AGENT_ENDPOINT", "Gradient AI Endpoint (optional)", None),
        ],
    },
    "publicai": {
        "category": "other",
        "extra_vars": [
            ("PUBLICAI_API_BASE", "PublicAI URL", "https://platform.publicai.co/"),
        ],
    },
    "v0": {
        "category": "other",
    },
    "vercel_ai_gateway": {
        "category": "other",
    },
    "wandb": {
        "category": "other",
    },
}

# Category display order and labels
PROVIDER_CATEGORIES = [
    ("custom", "Custom (First-Party)"),
    ("custom_openai", "Custom OpenAI-Compatible"),
    ("popular", "Popular"),
    ("cloud", "Cloud Platforms"),
    ("enterprise", "Enterprise / Complex Auth"),
    ("specialized", "Specialized (Image/Audio/Embeddings)"),
    ("regional", "Regional"),
    ("local", "Local / Self-Hosted"),
    ("other", "Other"),
]

# =============================================================================
# Provider Blacklist
# =============================================================================
# Providers that are in LiteLLM but should be excluded from:
# - KNOWN_PROVIDERS (so _API_BASE for them creates a "custom" provider)
# - Credential tool UI (won't show up in provider selection)
#
# Reasons for blacklisting:
# - No standard API key authentication (requires OAuth, token files, etc.)
# - Not actual LLM providers (protocols, templates, etc.)
# - Legacy/deprecated APIs
# - Complex auth requiring multiple credentials
# - Non-standard API key patterns (proxy only supports *_API_KEY)
# =============================================================================

PROVIDER_BLACKLIST: Set[str] = {
    # Not standard LLM providers / protocols
    "a2a",  # Pydantic AI agent-to-agent protocol
    "my-custom-llm",  # Template, not a real provider
    "text-completion-openai",  # Legacy text completion API
    # Require special auth (token files, OAuth, etc.)
    "github_copilot",  # Requires token file configuration
    "vercel_ai_gateway",  # Requires OIDC token
    # No API key authentication (use custom provider instead)
    "ollama",  # Local, no API key
    "llamafile",  # Local, no API key
    "petals",  # Distributed network, no API key
    "triton",  # NVIDIA Triton server, no API key
    "lemonade",  # Local, no API key
    "oci",  # OCI SDK auth only
    # Complex multi-credential auth (proxy only supports API_KEY + API_BASE)
    "azure",  # Requires API key + endpoint + API version
    "bedrock",  # Requires AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + region
    "sagemaker",  # Same as bedrock
    "vertex_ai",  # File path credential + project + location
    "sap",  # Multiple service credentials required
    "cloudflare",  # Requires API key + account ID
    "snowflake",  # Requires JWT + account ID
    "watsonx",  # Requires API key + special URL parameter
    # Non-standard API key patterns (uses *_TOKEN or *_KEY, not *_API_KEY)
    "cometapi",  # Uses COMETAPI_KEY
    "friendliai",  # Uses FRIENDLI_TOKEN
    "huggingface",  # Uses HF_TOKEN
}


def _build_known_providers_set() -> Set[str]:
    """
    Build set of known provider routes from scraped LiteLLM data.

    Uses routes as the primary identifier (authoritative from LiteLLM docs).
    Only uses API key prefix as fallback when provider has no route.

    Excludes providers in PROVIDER_BLACKLIST.

    Returns:
        Set of lowercase provider route identifiers known to LiteLLM.
    """
    known = set()

    for provider_key, info in SCRAPED_PROVIDERS.items():
        # Skip blacklisted providers
        if provider_key in PROVIDER_BLACKLIST:
            continue

        route = info.get("route", "").rstrip("/").lower()

        if route:
            # Provider has a route - use it as the canonical key
            known.add(route)
        else:
            # No route - fall back to API key prefix
            for api_key_var in info.get("api_key_env_vars", []):
                prefix = _extract_api_key_prefix(api_key_var)
                if prefix:
                    known.add(prefix)
                    break  # Only need one fallback

    return known


def _extract_api_key_prefix(api_key_var: str) -> Optional[str]:
    """Extract provider prefix from an API key environment variable name.

    Examples:
        OPENAI_API_KEY -> openai
        HF_TOKEN -> hf
        WATSONX_APIKEY -> watsonx
    """
    if not api_key_var:
        return None

    api_key_var = api_key_var.upper()

    if api_key_var.endswith("_API_KEY"):
        return api_key_var[:-8].lower()
    elif api_key_var.endswith("_TOKEN"):
        return api_key_var[:-6].lower()
    elif api_key_var.endswith("_APIKEY"):
        return api_key_var[:-7].lower()
    elif api_key_var.endswith("_KEY"):
        return api_key_var[:-4].lower()
    elif api_key_var.endswith("_JWT"):
        return api_key_var[:-4].lower()

    return None


# Pre-computed set of known provider names
KNOWN_PROVIDERS: Set[str] = _build_known_providers_set()


# Provider-specific env aliases whose shell-facing names intentionally differ
# from the model prefix used at the Mirrowel/LiteLLM boundary.
#
# Fireworks Kimi K2.6 uses a subscription-only key (`FIREWORKS_API_V2_KEY`)
# and API base (`FIREWORKS_V2_API_BASE`) while requests retain the model
# prefix `fireworks/...`. Normalize the API-base provider name here so the
# custom OpenAI-compatible rewrite in `convert_for_litellm()` can apply.
API_BASE_PROVIDER_ALIASES: Dict[str, str] = {
    "fireworks_v2": "fireworks",
}

ANTHROPIC_COMPATIBLE_CUSTOM_PROVIDERS: Set[str] = {
    "opencode_go_messages",
}


def get_provider_ui_config(provider_key: str) -> Dict[str, Any]:
    """Get UI configuration for a provider.

    Returns the UI-specific config (category, note, extra_vars) if defined,
    otherwise returns a default config.
    """
    return LITELLM_PROVIDERS.get(provider_key, {"category": "other"})


def get_full_provider_config(provider_key: str) -> Dict[str, Any]:
    """Get combined provider config (scraped data + UI config).

    Merges scraped provider data with UI configuration.
    """
    scraped = SCRAPED_PROVIDERS.get(provider_key, {})
    ui_config = LITELLM_PROVIDERS.get(provider_key, {"category": "other"})
    return {**scraped, **ui_config}


class ProviderConfig:
    """
    Centralized provider configuration handling.

    Handles:
    - API base overrides for known LiteLLM providers
    - Custom OpenAI-compatible providers (unknown provider names)

    Usage patterns:

    1. Override existing provider's API base:
       Set OPENAI_API_BASE=http://my-local-llm/v1
       Request: openai/gpt-4 → LiteLLM gets model="openai/gpt-4", api_base="http://..."

    2. Custom OpenAI-compatible provider:
       Set MYSERVER_API_BASE=http://myserver:8000/v1
       Request: myserver/llama-3 → LiteLLM gets model="openai/llama-3",
                api_base="http://...", custom_llm_provider="openai"
    """

    def __init__(self):
        self._api_bases: Dict[str, str] = {}
        self._custom_providers: Set[str] = set()
        self._load_api_bases()

    def _load_api_bases(self) -> None:
        """
        Load all <PROVIDER>_API_BASE environment variables.

        Detects whether each is an override for a known provider
        or defines a new custom provider.
        """
        for key, value in os.environ.items():
            if key.endswith("_API_BASE") and value:
                provider = key[:-9].lower()  # Remove _API_BASE
                provider = API_BASE_PROVIDER_ALIASES.get(provider, provider)
                if provider in ANTHROPIC_COMPATIBLE_CUSTOM_PROVIDERS and value.rstrip("/").endswith(
                    "/v1"
                ):
                    value = value.rstrip("/")[:-3]
                self._api_bases[provider] = value.rstrip("/")

                # Track if this is a custom provider (not known to LiteLLM)
                if provider not in KNOWN_PROVIDERS:
                    self._custom_providers.add(provider)
                    lib_logger.info(
                        f"Detected custom OpenAI-compatible provider: {provider} "
                        f"(api_base: {value})"
                    )
                else:
                    lib_logger.info(f"Detected API base override for {provider}: {value}")

    def is_known_provider(self, provider: str) -> bool:
        """Check if provider is known to LiteLLM."""
        return provider.lower() in KNOWN_PROVIDERS

    def is_custom_provider(self, provider: str) -> bool:
        """Check if provider is a custom OpenAI-compatible provider."""
        return provider.lower() in self._custom_providers

    def get_api_base(self, provider: str) -> Optional[str]:
        """Get configured API base for a provider, if any."""
        return self._api_bases.get(provider.lower())

    def get_custom_providers(self) -> Set[str]:
        """Get the set of detected custom provider names."""
        return self._custom_providers.copy()

    @staticmethod
    def openai_catalog_model_id(model: str) -> str:
        """Return the bare OpenAI catalog id for route forms like ``openai/gpt-5.6-sol``.

        Router/Mirrowel route identity stays provider-qualified. Local OpenAI-compatible
        bases (codex-lb) allowlist bare catalog ids only; the wire JSON ``model`` field
        must not include the ``openai/`` route prefix.
        """
        if not isinstance(model, str) or not model:
            return model
        if "/" not in model:
            return model
        provider, remainder = model.split("/", 1)
        if provider.lower() == "openai" and remainder:
            return remainder
        return model

    def convert_for_litellm(self, **kwargs) -> Dict[str, Any]:
        """
        Convert model params for LiteLLM call.

        Handles:
        - Known provider with _API_BASE: pass api_base as override
        - OpenAI + local/custom OPENAI_API_BASE: bare catalog model + custom_llm_provider
        - Unknown provider with _API_BASE: convert to openai/, set custom_llm_provider
        - Bare OpenAI-family model ids (gpt-*, o1/o3/o4-*) when OPENAI_API_BASE is set
        - No _API_BASE configured: pass through unchanged

        Args:
            **kwargs: LiteLLM call kwargs including 'model'

        Returns:
            Modified kwargs dict ready for LiteLLM
        """
        model = kwargs.get("model")
        if not model:
            return kwargs

        # Extract provider from model string (e.g., "openai/gpt-4" → "openai")
        if "/" in model:
            provider = model.split("/", 1)[0].lower()
        else:
            # Bare model ids (common after internal resolution): apply openai base when
            # OPENAI_API_BASE is configured and the id looks like an OpenAI family surface.
            # Without this, LiteLLM routes sk-clb/* sidecar keys to api.openai.com (403).
            bare = model.lower()
            openai_base = self._api_bases.get("openai")
            if openai_base and (
                bare.startswith("gpt-")
                or bare.startswith("o1")
                or bare.startswith("o3")
                or bare.startswith("o4")
                or bare.startswith("chatgpt-")
            ):
                kwargs = kwargs.copy()
                # Wire bare catalog id; explicit openai provider + api_base for routing.
                kwargs["model"] = model
                kwargs["custom_llm_provider"] = "openai"
                kwargs["api_base"] = openai_base
                kwargs["base_url"] = openai_base
                lib_logger.info(
                    "Applying openai api_base to bare model %s -> %s", model, openai_base
                )
                return kwargs
            return kwargs

        api_base = self._api_bases.get(provider)

        if not api_base:
            # No override configured for this provider
            return kwargs

        # Create a copy to avoid modifying the original
        kwargs = kwargs.copy()

        if provider == "openai":
            # codex-lb and other OpenAI-compatible loopback servers reject
            # "openai/<catalog-id>" as the JSON model field. Keep LiteLLM on the
            # openai provider via custom_llm_provider while sending the bare id.
            bare = self.openai_catalog_model_id(model)
            kwargs["model"] = bare
            kwargs["custom_llm_provider"] = "openai"
            kwargs["api_base"] = api_base
            kwargs["base_url"] = api_base
            lib_logger.info(
                "Applying openai api_base with bare wire model %s (route %s) -> %s",
                bare,
                model,
                api_base,
            )
        elif provider in KNOWN_PROVIDERS:
            # Known provider - just add api_base override
            kwargs["api_base"] = api_base
            # Some LiteLLM versions honor base_url over api_base for openai/*
            kwargs["base_url"] = api_base
            lib_logger.info(
                "Applying api_base override for known provider %s: %s", provider, api_base
            )
        elif provider in ANTHROPIC_COMPATIBLE_CUSTOM_PROVIDERS:
            # Custom provider - route through Anthropic-compatible /messages endpoint.
            model_name = model.split("/", 1)[1] if "/" in model else model
            kwargs["model"] = f"anthropic/{model_name}"
            kwargs["api_base"] = api_base
            kwargs["custom_llm_provider"] = "anthropic"
            lib_logger.debug(
                f"Routing custom provider {provider} through anthropic: "
                f"model={kwargs['model']}, api_base={api_base}"
            )
        else:
            # Custom provider - route through OpenAI-compatible endpoint
            model_name = model.split("/", 1)[1] if "/" in model else model
            kwargs["model"] = f"openai/{model_name}"
            kwargs["api_base"] = api_base
            kwargs["custom_llm_provider"] = "openai"
            lib_logger.debug(
                f"Routing custom provider {provider} through openai: "
                f"model={kwargs['model']}, api_base={api_base}"
            )

        return kwargs
