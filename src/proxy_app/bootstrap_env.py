"""Lightweight environment normalization used before proxy dependency loading."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import sys
from typing import Protocol

from dotenv import dotenv_values, load_dotenv


class NetworkBindApproval(Protocol):
    def apply_to_environment(self) -> None: ...


def set_env_default(target: str, *sources: str, default: str | None = None) -> None:
    if os.getenv(target):
        return
    for source in sources:
        value = os.getenv(source)
        if value:
            os.environ[target] = value
            return
    if default is not None:
        os.environ[target] = default


def set_numbered_env_defaults(target_base: str, *source_bases: str) -> None:
    for source_base in source_bases:
        prefix = f"{source_base}_"
        for key, value in list(os.environ.items()):
            if not value or not key.startswith(prefix):
                continue
            suffix = key.removeprefix(prefix)
            if not suffix.isdigit():
                continue
            target = f"{target_base}_{suffix}"
            if not os.getenv(target):
                os.environ[target] = value


def load_durable_env_values(env_file: Path) -> dict[str, str | None]:
    return dict(dotenv_values(env_file)) if env_file.exists() else {}


def apply_durable_disabled_providers(durable_values: Mapping[str, str | None]) -> None:
    if "DISABLED_PROVIDERS" not in durable_values:
        os.environ.pop("DISABLED_PROVIDERS", None)
        return
    value = durable_values.get("DISABLED_PROVIDERS") or ""
    if value:
        os.environ["DISABLED_PROVIDERS"] = value
    else:
        os.environ.pop("DISABLED_PROVIDERS", None)


def apply_durable_ollama_aliases(durable_values: Mapping[str, str | None]) -> None:
    if "OLLAMA_CLOUD_API_KEY" not in durable_values and durable_values.get("OLLAMA_API_KEY"):
        os.environ["OLLAMA_CLOUD_API_KEY"] = durable_values["OLLAMA_API_KEY"] or ""
    for key, value in durable_values.items():
        if not value or not key.startswith("OLLAMA_API_KEY_"):
            continue
        suffix = key.removeprefix("OLLAMA_API_KEY_")
        if not suffix.isdigit():
            continue
        target = f"OLLAMA_CLOUD_API_KEY_{suffix}"
        if target not in durable_values:
            os.environ[target] = value


def normalize_provider_env_aliases() -> None:
    set_env_default("PROXY_API_KEY", "MIRROWEL_PROXY_KEY")
    set_env_default("OLLAMA_CLOUD_API_KEY", "OLLAMA_API_KEY")
    set_numbered_env_defaults("OLLAMA_CLOUD_API_KEY", "OLLAMA_API_KEY")
    set_env_default("OPENCODE_GO_API_KEY", "OPENCODE_GO_KEY")
    set_env_default("OPENCODE_GO_MESSAGES_API_KEY", "OPENCODE_GO_API_KEY", "OPENCODE_GO_KEY")
    set_env_default("OPENROUTER_ZDR_API_KEY", "OPENROUTER_ZDR_KEY")
    set_env_default(
        "OPENROUTER_FREE_API_KEY",
        "OPENROUTER_FREE_KEY",
        "OPENROUTER_NON_ZDR_KEY",
        "OPENROUTER_NON_ZDR_API_KEY",
    )
    set_env_default(
        "OPENROUTER_NON_ZDR_API_KEY",
        "OPENROUTER_FREE_KEY",
        "OPENROUTER_FREE_API_KEY",
        "OPENROUTER_NON_ZDR_KEY",
    )
    set_env_default("OPENROUTER_NON_ZDR_KEY", "OPENROUTER_FREE_KEY")
    set_env_default("OPENROUTER_FREE_KEY", "OPENROUTER_NON_ZDR_KEY")
    set_env_default("OLLAMA_CLOUD_API_BASE", "OLLAMA_API_BASE", default="https://ollama.com/v1")
    set_env_default("OPENCODE_GO_API_BASE", default="https://opencode.ai/zen/go/v1")
    set_env_default("OPENCODE_GO_MESSAGES_API_BASE", default="https://opencode.ai/zen/go")
    set_env_default("XAI_OAUTH_API_BASE", default="http://127.0.0.1:2465/v1")
    set_env_default(
        "FIREWORKS_V2_API_BASE",
        "FIREWORKS_API_BASE",
        default="https://api.fireworks.ai/inference/v1",
    )
    set_env_default("FIREWORKS_API_BASE", default="https://api.fireworks.ai/inference/v1")
    set_env_default("OPENROUTER_ZDR_API_BASE", default="https://openrouter.ai/api/v1")
    set_env_default(
        "OPENROUTER_NON_ZDR_API_BASE",
        "OPENROUTER_FREE_API_BASE",
        default="https://openrouter.ai/api/v1",
    )
    set_env_default(
        "OPENROUTER_FREE_API_BASE",
        "OPENROUTER_NON_ZDR_API_BASE",
        default="https://openrouter.ai/api/v1",
    )


def load_router_env(
    env_file: Path,
    network_bind_approval: NetworkBindApproval | None = None,
) -> None:
    if os.getenv("OPENCODE_ROUTER_PROVIDER_ENV_PATH", "").strip():
        normalize_provider_env_aliases()
    else:
        load_dotenv(env_file, override=True)
        durable_values = load_durable_env_values(env_file)
        apply_durable_disabled_providers(durable_values)
        apply_durable_ollama_aliases(durable_values)
        normalize_provider_env_aliases()
    if network_bind_approval is not None:
        network_bind_approval.apply_to_environment()
    loaded_provider_registry = sys.modules.get("rotator_library.providers")
    if loaded_provider_registry is not None:
        loaded_provider_registry._register_providers()
