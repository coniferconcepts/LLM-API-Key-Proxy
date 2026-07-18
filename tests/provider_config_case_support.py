import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "src" / "proxy_app" / "main.py"
sys.path.insert(0, str(ROOT / "src"))

from proxy_app.bootstrap_env import (  # noqa: E402
    apply_durable_disabled_providers,
    apply_durable_ollama_aliases,
    load_durable_env_values,
    load_router_env,
    normalize_provider_env_aliases,
    set_env_default,
    set_numbered_env_defaults,
)
from rotator_library.client import RotatingClient  # noqa: E402


def load_proxy_main_env_helpers():
    return {
        "_set_env_default": set_env_default,
        "_set_numbered_env_defaults": set_numbered_env_defaults,
        "_load_durable_env_values": load_durable_env_values,
        "_apply_durable_disabled_providers": apply_durable_disabled_providers,
        "_apply_durable_ollama_aliases": apply_durable_ollama_aliases,
        "_load_router_env": load_router_env,
        "_normalize_provider_env_aliases": normalize_provider_env_aliases,
    }


def load_proxy_main_env_normalizer():
    return load_proxy_main_env_helpers()["_normalize_provider_env_aliases"]


def load_proxy_main_credential_summary_builder():
    source = MAIN_PATH.read_text(encoding="utf-8")
    start = source.index("def build_credential_summary(")
    end = source.index('@app.get("/v1/credential-summary")')
    namespace = {"Any": object, "RotatingClient": RotatingClient, "json": json}
    exec(source[start:end], namespace)
    return namespace["build_credential_summary"]


def load_proxy_main_startup_credential_summary_printer():
    source = MAIN_PATH.read_text(encoding="utf-8")
    start = source.index("def build_credential_summary(")
    end = source.index('@app.get("/v1/credential-summary")')
    namespace = {"Any": object, "RotatingClient": RotatingClient, "json": json}
    exec(source[start:end], namespace)
    return namespace["print_startup_credential_summary"]
