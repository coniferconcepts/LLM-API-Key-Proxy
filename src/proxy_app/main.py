# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

import time

# Phase 1: Minimal imports for arg parsing and TUI
import asyncio
import os
from pathlib import Path
import sys
import argparse
import logging
import secrets

_LOCAL_SRC_PATH = str(Path(__file__).resolve().parent.parent)
if _LOCAL_SRC_PATH in sys.path:
    sys.path.remove(_LOCAL_SRC_PATH)
sys.path.insert(0, _LOCAL_SRC_PATH)

from proxy_app.runtime_security import (
    RuntimeSecurityConfig,  # noqa: F401 - compatibility export
    build_allowed_hosts as _build_allowed_hosts,  # noqa: F401 - compatibility export
    build_cors_allowed_origins as _build_cors_allowed_origins,
    build_runtime_security_config as _build_runtime_security_config,
    env_flag_enabled as _env_flag_enabled,
    is_loopback_bind_host as _is_loopback_bind_host,  # noqa: F401 - compatibility export
    network_bind_configuration_error as _network_bind_configuration_error,  # noqa: F401 - compatibility export
    runtime_security_source as _runtime_security_source,  # noqa: F401 - compatibility export
    validate_bind_host as _validate_bind_host,
)

# --- Argument Parsing (BEFORE heavy imports) ---
parser = argparse.ArgumentParser(description="API Key Proxy Server")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to.")
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
parser.add_argument(
    "--enable-request-logging",
    action="store_true",
    help="Enable transaction logging in the library (logs request/response with provider correlation).",
)
parser.add_argument(
    "--enable-raw-logging",
    action="store_true",
    help="Enable raw I/O logging at proxy boundary (captures unmodified HTTP data, disabled by default).",
)
parser.add_argument(
    "--add-credential",
    action="store_true",
    help="Launch the interactive tool to add a new OAuth credential.",
)
args, _ = parser.parse_known_args()


_frozen_local_transport_safe_mode: bool | None = None


def _local_transport_safe_mode_enabled() -> bool:
    if _frozen_local_transport_safe_mode is not None:
        return _frozen_local_transport_safe_mode
    return _env_flag_enabled("MIRROWEL_LOCAL_TRANSPORT_SAFE_MODE")


_tui_network_bind_approval = None

# Check if we should launch TUI (no arguments = TUI mode)
if len(sys.argv) == 1:
    # TUI MODE - Load ONLY what's needed for the launcher (fast path!)
    from proxy_app.launcher_tui import run_launcher_tui

    _tui_network_bind_approval = run_launcher_tui()
    # Launcher modifies sys.argv and returns, or exits if user chose Exit
    # If we get here, user chose "Run Proxy" and sys.argv is modified
    # Re-parse arguments with modified sys.argv
    args = parser.parse_args()

# Check if credential tool mode (also doesn't need heavy proxy imports)
if args.add_credential:
    from rotator_library.credential_tool import run_credential_tool

    run_credential_tool()
    sys.exit(0)

# If we get here, we're ACTUALLY running the proxy - NOW show startup messages and start timer
_start_time = time.time()

# Load all .env files from root folder (main .env first, then any additional *.env files)
from dotenv import load_dotenv
from proxy_app.bootstrap_env import load_router_env as _load_router_env

# Get the application root directory (EXE dir if frozen, else CWD)
# Inlined here to avoid triggering heavy rotator_library imports before loading screen
if getattr(sys, "frozen", False):
    _root_dir = Path(sys.executable).parent
else:
    _root_dir = Path.cwd()


_provider_environment_is_authoritative = bool(
    os.getenv("OPENCODE_ROUTER_PROVIDER_ENV_PATH", "").strip()
)


# Load main .env first
_main_env_file = _root_dir / ".env"
_load_router_env(_main_env_file, _tui_network_bind_approval)

# Load any additional .env files (e.g., antigravity_all_combined.env, gemini_cli_all_combined.env)
_env_files_found = [] if _provider_environment_is_authoritative else list(_root_dir.glob("*.env"))
if not _provider_environment_is_authoritative:
    for _env_file in sorted(_env_files_found):
        if _env_file.name != ".env":  # Skip main .env (already loaded)
            load_dotenv(_env_file, override=False)  # Don't override existing values

_load_router_env(_main_env_file, _tui_network_bind_approval)
if _local_transport_safe_mode_enabled():
    from proxy_app.local_transport_policy import normalize_local_xai_base

    try:
        os.environ["XAI_OAUTH_API_BASE"] = normalize_local_xai_base(
            os.environ["XAI_OAUTH_API_BASE"]
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
from proxy_app.local_transport_runtime import LocalTransportRuntimePolicy

try:
    _local_transport_runtime_policy = LocalTransportRuntimePolicy.from_environment(os.environ)
except ValueError as exc:
    raise SystemExit(str(exc)) from exc
_frozen_local_transport_safe_mode = _local_transport_runtime_policy.enabled
_runtime_security_config = _build_runtime_security_config()
args.host = _validate_bind_host(args.host, _runtime_security_config)

# Log discovered .env files for deployment verification
if _env_files_found:
    _env_names = [_ef.name for _ef in _env_files_found]
    print(f"📁 Loaded {len(_env_files_found)} .env file(s): {', '.join(_env_names)}")

# Get proxy API key for display
proxy_api_key = os.getenv("PROXY_API_KEY")
if proxy_api_key:
    key_display = "✓ Set"
else:
    key_display = "✗ Not Set (INSECURE - anyone can access!)"

print("━" * 70)
print(f"Starting proxy on {args.host}:{args.port}")
print(f"Proxy API Key: {key_display}")
print("GitHub: https://github.com/Mirrowel/LLM-API-Key-Proxy")
print("━" * 70)
print("Loading server components...")


# Phase 2: Load Rich for loading spinner (lightweight)
from rich.console import Console

_console = Console()

# Phase 3: Heavy dependencies with granular loading messages
print("  → Loading FastAPI framework...")
with _console.status("[dim]Loading FastAPI framework...", spinner="dots"):
    from contextlib import aclosing, asynccontextmanager
    from fastapi import FastAPI, Request, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse

print("  → Loading core dependencies...")
with _console.status("[dim]Loading core dependencies...", spinner="dots"):
    from dotenv import load_dotenv
    import colorlog
    import json
    from typing import AsyncGenerator, Any, List, Optional, Union
    from pydantic import BaseModel, ConfigDict, Field

    # --- Early Log Level Configuration ---
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

print("  → Loading LiteLLM library...")
with _console.status("[dim]Loading LiteLLM library...", spinner="dots"):
    from proxy_app.litellm_loader import load_litellm

    litellm = load_litellm(
        local_transport_safe_mode=_local_transport_safe_mode_enabled(),
    )

# Phase 4: Application imports with granular loading messages
print("  → Initializing proxy core...")
with _console.status("[dim]Initializing proxy core...", spinner="dots"):
    from rotator_library import RotatingClient
    from rotator_library.credential_manager import CredentialManager
    from rotator_library.provider_config import (
        discover_api_keys_from_env,
        normalize_credential_provider,
    )
    from rotator_library.model_info_service import init_model_info_service
    from proxy_app.request_logger import log_request_to_console
    from proxy_app.batch_manager import EmbeddingBatcher
    from proxy_app.request_boundary import BoundedJSONBodyMiddleware
    from proxy_app.safe_errors import (
        SafeUnhandledErrorMiddleware,
        anthropic_error_content,
        log_safe_exception,
        public_error_detail,
    )
    from proxy_app.anthropic_stream import (
        bounded_anthropic_sse_response as anthropic_streaming_response_wrapper,
    )
    from proxy_app.stream_bounds import (
        DEFAULT_OPENAI_STREAM_CAPACITY,
        DEFAULT_OPENAI_STREAM_POLICY,
        OpenAIStreamPolicy,
        bounded_sse_stream,
    )
    from rotator_library.stream_terminal import require_openai_terminal, sse_data_content
    from proxy_app.detailed_logger import RawIOLogger

OPENAI_STREAM_MAX_BYTES = DEFAULT_OPENAI_STREAM_POLICY.max_bytes
OPENAI_STREAM_MAX_EVENTS = DEFAULT_OPENAI_STREAM_POLICY.max_events
OPENAI_STREAM_IDLE_TIMEOUT_SECONDS = DEFAULT_OPENAI_STREAM_POLICY.idle_timeout_seconds
OPENAI_STREAM_TOTAL_TIMEOUT_SECONDS = DEFAULT_OPENAI_STREAM_POLICY.total_timeout_seconds
OPENAI_STREAM_CAPACITY = DEFAULT_OPENAI_STREAM_CAPACITY

print("  → Discovering provider plugins...")
# Provider lazy loading happens during import, so time it here
_provider_start = time.time()
with _console.status("[dim]Discovering provider plugins...", spinner="dots"):
    from rotator_library import (
        PROVIDER_PLUGINS,
    )  # This triggers lazy load via __getattr__
    from rotator_library.error_handler import NoAvailableKeysError
_provider_time = time.time() - _provider_start

# Get count after import (without timing to avoid double-counting)
_plugin_count = len(PROVIDER_PLUGINS)


# --- Pydantic Models ---
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    input_type: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None


class ModelCard(BaseModel):
    """Basic model card for minimal response."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "Mirro-Proxy"


class ModelCapabilities(BaseModel):
    """Model capability flags."""

    tool_choice: bool = False
    function_calling: bool = False
    reasoning: bool = False
    vision: bool = False
    system_messages: bool = True
    prompt_caching: bool = False
    assistant_prefill: bool = False


class EnrichedModelCard(BaseModel):
    """Extended model card with pricing and capabilities."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "unknown"
    # Pricing (optional - may not be available for all models)
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    cache_read_input_token_cost: Optional[float] = None
    cache_creation_input_token_cost: Optional[float] = None
    # Limits (optional)
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    context_window: Optional[int] = None
    # Capabilities
    mode: str = "chat"
    supported_modalities: List[str] = Field(default_factory=lambda: ["text"])
    supported_output_modalities: List[str] = Field(default_factory=lambda: ["text"])
    capabilities: Optional[ModelCapabilities] = None
    # Debug info (optional)
    _sources: Optional[List[str]] = None
    _match_type: Optional[str] = None

    model_config = ConfigDict(extra="allow")  # Allow extra fields from the service


class ModelList(BaseModel):
    """List of models response."""

    object: str = "list"
    data: List[ModelCard]


class EnrichedModelList(BaseModel):
    """List of enriched models with pricing and capabilities."""

    object: str = "list"
    data: List[EnrichedModelCard]


# --- Anthropic API Models (imported from library) ---
from rotator_library.anthropic_compat import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
)

# Calculate total loading time
_elapsed = time.time() - _start_time
print(
    f"✓ Server ready in {_elapsed:.2f}s ({_plugin_count} providers discovered in {_provider_time:.2f}s)"
)

# Clear screen and reprint header for clean startup view
# This pushes loading messages up (still in scroll history) but shows a clean final screen
import os as _os_module

_os_module.system("cls" if _os_module.name == "nt" else "clear")

# Reprint header
print("━" * 70)
print(f"Starting proxy on {args.host}:{args.port}")
print(f"Proxy API Key: {key_display}")
print("GitHub: https://github.com/Mirrowel/LLM-API-Key-Proxy")
print("━" * 70)
print(
    f"✓ Server ready in {_elapsed:.2f}s ({_plugin_count} providers discovered in {_provider_time:.2f}s)"
)


# Note: Debug logging will be added after logging configuration below

# --- Logging Configuration ---
# Import path utilities here (after loading screen) to avoid triggering heavy imports early
from rotator_library.secure_logging import OwnerOnlyRotatingFileHandler
from rotator_library.utils.paths import get_logs_dir, get_data_file

LOG_DIR = get_logs_dir(_root_dir)

# Configure a console handler with color (INFO and above only, no DEBUG)
console_handler = colorlog.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)
console_handler.setFormatter(formatter)

# Configure a file handler for INFO-level logs and higher
info_file_handler = OwnerOnlyRotatingFileHandler(
    LOG_DIR / "proxy.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
info_file_handler.setLevel(logging.INFO)
info_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Configure a dedicated file handler for all DEBUG-level logs
debug_file_handler = OwnerOnlyRotatingFileHandler(
    LOG_DIR / "proxy_debug.log", maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
)
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)


# Create a filter to ensure the debug handler ONLY gets DEBUG messages from the rotator_library
class RotatorDebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG and record.name.startswith("rotator_library")


debug_file_handler.addFilter(RotatorDebugFilter())

# Configure a console handler with color
console_handler = colorlog.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)
console_handler.setFormatter(formatter)


# Add a filter to prevent any LiteLLM logs from cluttering the console
class NoLiteLLMLogFilter(logging.Filter):
    def filter(self, record):
        return not record.name.startswith("LiteLLM")


console_handler.addFilter(NoLiteLLMLogFilter())

# Get the root logger and set it to DEBUG to capture all messages
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Add all handlers to the root logger
root_logger.addHandler(info_file_handler)
root_logger.addHandler(console_handler)
root_logger.addHandler(debug_file_handler)

# Silence other noisy loggers by setting their level higher than root
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Isolate LiteLLM's logger to prevent it from reaching the console.
# We will capture its logs via the logger_fn callback in the client instead.
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.handlers = []
litellm_logger.propagate = False

# Now that logging is configured, log the module load time to debug file only
logging.debug(f"Modules loaded in {_elapsed:.2f}s")

# Load environment variables from .env file
_load_router_env(_main_env_file, _tui_network_bind_approval)

# --- Configuration ---
USE_EMBEDDING_BATCHER = False
ENABLE_REQUEST_LOGGING = args.enable_request_logging
ENABLE_RAW_LOGGING = args.enable_raw_logging
if ENABLE_REQUEST_LOGGING:
    logging.info("Transaction logging is enabled (library-level with provider correlation).")
if ENABLE_RAW_LOGGING:
    logging.info("Raw I/O logging is enabled (proxy boundary, unmodified HTTP data).")
# Discover API keys from environment variables.
# Fireworks V2 is intentionally handled by the shared helper because
# `FIREWORKS_API_V2_KEY` does not match the generic `_API_KEY` pattern.
api_keys = discover_api_keys_from_env()
if _local_transport_safe_mode_enabled():
    api_keys = {"xai_oauth": api_keys["xai_oauth"]} if api_keys.get("xai_oauth") else {}

disabled_providers = {
    normalize_credential_provider(provider)
    for provider in os.getenv("DISABLED_PROVIDERS", "").split(",")
    if provider.strip()
}
if disabled_providers:
    api_keys = {
        provider: keys for provider, keys in api_keys.items() if provider not in disabled_providers
    }

# Load model ignore lists from environment variables
ignore_models = {}
for key, value in os.environ.items():
    if key.startswith("IGNORE_MODELS_"):
        provider = key.replace("IGNORE_MODELS_", "").lower()
        models_to_ignore = [model.strip() for model in value.split(",") if model.strip()]
        ignore_models[provider] = models_to_ignore
        logging.debug(f"Loaded ignore list for provider '{provider}': {models_to_ignore}")

# Load model whitelist from environment variables
whitelist_models = {}
for key, value in os.environ.items():
    if key.startswith("WHITELIST_MODELS_"):
        provider = key.replace("WHITELIST_MODELS_", "").lower()
        models_to_whitelist = [model.strip() for model in value.split(",") if model.strip()]
        whitelist_models[provider] = models_to_whitelist
        logging.debug(f"Loaded whitelist for provider '{provider}': {models_to_whitelist}")

# Load max concurrent requests per key from environment variables
max_concurrent_requests_per_key = {}
for key, value in os.environ.items():
    if key.startswith("MAX_CONCURRENT_REQUESTS_PER_KEY_"):
        provider = key.replace("MAX_CONCURRENT_REQUESTS_PER_KEY_", "").lower()
        try:
            max_concurrent = int(value)
            if max_concurrent < 1:
                logging.warning(
                    f"Invalid max_concurrent value for provider '{provider}': {value}. Must be >= 1. Using default (1)."
                )
                max_concurrent = 1
            max_concurrent_requests_per_key[provider] = max_concurrent
            logging.debug(
                f"Loaded max concurrent requests for provider '{provider}': {max_concurrent}"
            )
        except ValueError:
            logging.warning(
                f"Invalid max_concurrent value for provider '{provider}': {value}. Using default (1)."
            )


# --- Lifespan Management ---
from proxy_app.app_lifecycle import LifecycleDependencies, application_lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the RotatingClient's lifecycle with the app's lifespan."""
    dependencies = LifecycleDependencies(
        credential_manager_factory=CredentialManager,
        rotating_client_factory=RotatingClient,
        provider_plugins=PROVIDER_PLUGINS,
        init_model_info_service=init_model_info_service,
        embedding_batcher_factory=EmbeddingBatcher,
        litellm=litellm,
        log_safe_exception=log_safe_exception,
        print_startup_credential_summary=print_startup_credential_summary,
        local_transport_safe_mode_enabled=_local_transport_safe_mode_enabled,
        api_keys=api_keys,
        disabled_providers=frozenset(disabled_providers),
        ignore_models=ignore_models,
        whitelist_models=whitelist_models,
        max_concurrent_requests_per_key=max_concurrent_requests_per_key,
        enable_request_logging=ENABLE_REQUEST_LOGGING,
        use_embedding_batcher=USE_EMBEDDING_BATCHER,
    )
    async with application_lifespan(app, dependencies):
        yield


# --- FastAPI App Setup ---
app = FastAPI(lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, error: HTTPException) -> JSONResponse:
    if request.url.path in {"/v1/messages", "/v1/messages/count_tokens"}:
        return JSONResponse(
            status_code=error.status_code,
            content=anthropic_error_content(error.status_code),
            headers=error.headers,
        )
    return JSONResponse(
        status_code=error.status_code,
        content={"detail": error.detail},
        headers=error.headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, error: Exception) -> JSONResponse:
    log_safe_exception("Unhandled request", error, 500)
    return JSONResponse(status_code=500, content={"detail": public_error_detail(500)})


from proxy_app.security_middleware import (
    BindApprovalMiddleware,
    TrustedHostMiddleware,
)

_cors_allowed_origins = _build_cors_allowed_origins()
_allowed_hosts = list(_runtime_security_config.allowed_hosts)


app.add_middleware(
    BoundedJSONBodyMiddleware,
    credential_config_getter=lambda: (
        _runtime_security_config.proxy_api_key,
        _runtime_security_config.is_current(),
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=_allowed_hosts)
app.add_middleware(
    BindApprovalMiddleware,
    runtime_config_getter=lambda: _runtime_security_config,
)
app.add_middleware(SafeUnhandledErrorMiddleware)


def get_rotating_client(request: Request) -> RotatingClient:
    """Dependency to get the rotating client instance from the app state."""
    return request.app.state.rotating_client


def get_embedding_batcher(request: Request) -> EmbeddingBatcher:
    """Dependency to get the embedding batcher instance from the app state."""
    return request.app.state.embedding_batcher


def _credential_header_values(request: Request) -> tuple[list[str], list[str]]:
    authorization_values: list[str] = []
    api_key_values: list[str] = []
    for name, value in request.scope.get("headers", []):
        normalized_name = name.lower()
        if normalized_name == b"authorization":
            authorization_values.append(value.decode("latin-1"))
        elif normalized_name == b"x-api-key":
            api_key_values.append(value.decode("latin-1"))
    return authorization_values, api_key_values


def _validated_credential_headers(request: Request) -> tuple[str | None, str | None]:
    if not _runtime_security_config.is_current():
        raise HTTPException(
            status_code=503,
            detail="Runtime security configuration changed after initialization",
        )
    authorization_values, api_key_values = _credential_header_values(request)
    if (
        len(authorization_values) > 1
        or len(api_key_values) > 1
        or (authorization_values and api_key_values)
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    authorization = authorization_values[0] if authorization_values else None
    api_key = api_key_values[0] if api_key_values else None
    return authorization, api_key


def _constant_time_equal(candidate: str, expected: str) -> bool:
    return secrets.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


async def verify_api_key(request: Request) -> str | None:
    authorization, _api_key = _validated_credential_headers(request)
    proxy_api_key = _runtime_security_config.proxy_api_key
    if not proxy_api_key:
        return None
    expected_authorization = f"Bearer {proxy_api_key}"
    if authorization is None or not _constant_time_equal(
        authorization,
        expected_authorization,
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return authorization


async def verify_anthropic_api_key(
    request: Request,
) -> str:
    """
    Dependency to verify API key for Anthropic endpoints.
    Accepts either x-api-key header (Anthropic style) or Authorization Bearer (OpenAI style).
    """
    authorization, api_key = _validated_credential_headers(request)
    proxy_api_key = _runtime_security_config.proxy_api_key
    if not proxy_api_key:
        return ""
    if api_key and _constant_time_equal(api_key, proxy_api_key):
        return api_key
    expected_authorization = f"Bearer {proxy_api_key}"
    if authorization and _constant_time_equal(authorization, expected_authorization):
        return authorization
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")


async def streaming_response_wrapper(
    request: Request,
    request_data: dict,
    response_stream: AsyncGenerator[str, None],
    logger: Optional[RawIOLogger] = None,
) -> AsyncGenerator[str, None]:
    """
    Wraps a streaming response to log the full response after completion
    and ensures any errors during the stream are sent to the client.
    """
    response_chunks = []
    full_response = {}
    stream_failed = False
    try:
        policy = OpenAIStreamPolicy(
            max_bytes=OPENAI_STREAM_MAX_BYTES,
            max_events=OPENAI_STREAM_MAX_EVENTS,
            idle_timeout_seconds=OPENAI_STREAM_IDLE_TIMEOUT_SECONDS,
            total_timeout_seconds=OPENAI_STREAM_TOTAL_TIMEOUT_SECONDS,
        )
        bounded_stream = bounded_sse_stream(
            response_stream,
            policy,
            OPENAI_STREAM_CAPACITY,
        )
        async with aclosing(bounded_stream):
            disconnect_check = request.is_disconnected
            async for chunk_str in require_openai_terminal(bounded_stream, disconnect_check):
                yield chunk_str
                content = sse_data_content(chunk_str)
                if content is not None and content != "[DONE]":
                    try:
                        chunk_data = json.loads(content)
                        response_chunks.append(chunk_data)
                        if logger:
                            logger.log_stream_chunk(chunk_data)
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        stream_failed = True
        log_safe_exception("OpenAI response stream", e, 500)
        # Yield a final error message to the client to ensure they are not left hanging.
        error_payload = {"error": public_error_detail(500, code="stream_error")}
        yield f"\n\ndata: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
        # Also log this as a failed request
        if logger:
            logger.log_final_response(status_code=500, headers=None, body=error_payload)
        return  # Stop further processing
    finally:
        if response_chunks:
            # --- Aggregation Logic ---
            final_message = {"role": "assistant"}
            aggregated_tool_calls = {}
            usage_data = None
            finish_reason = None

            for chunk in response_chunks:
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})

                    # Dynamically aggregate all fields from the delta
                    for key, value in delta.items():
                        if value is None:
                            continue

                        if key == "content":
                            if "content" not in final_message:
                                final_message["content"] = ""
                            if value:
                                final_message["content"] += value

                        elif key == "tool_calls":
                            for tc_chunk in value:
                                index = tc_chunk["index"]
                                if index not in aggregated_tool_calls:
                                    aggregated_tool_calls[index] = {
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                # Ensure 'function' key exists for this index before accessing its sub-keys
                                if "function" not in aggregated_tool_calls[index]:
                                    aggregated_tool_calls[index]["function"] = {
                                        "name": "",
                                        "arguments": "",
                                    }
                                if tc_chunk.get("id"):
                                    aggregated_tool_calls[index]["id"] = tc_chunk["id"]
                                if "function" in tc_chunk:
                                    if "name" in tc_chunk["function"]:
                                        if tc_chunk["function"]["name"] is not None:
                                            aggregated_tool_calls[index]["function"][
                                                "name"
                                            ] += tc_chunk["function"]["name"]
                                    if "arguments" in tc_chunk["function"]:
                                        if tc_chunk["function"]["arguments"] is not None:
                                            aggregated_tool_calls[index]["function"][
                                                "arguments"
                                            ] += tc_chunk["function"]["arguments"]

                        elif key == "function_call":
                            if "function_call" not in final_message:
                                final_message["function_call"] = {
                                    "name": "",
                                    "arguments": "",
                                }
                            if "name" in value:
                                if value["name"] is not None:
                                    final_message["function_call"]["name"] += value["name"]
                            if "arguments" in value:
                                if value["arguments"] is not None:
                                    final_message["function_call"]["arguments"] += value[
                                        "arguments"
                                    ]

                        else:
                            if key == "role":
                                final_message[key] = value
                            elif key not in final_message:
                                final_message[key] = value
                            elif isinstance(final_message.get(key), str):
                                final_message[key] += value
                            else:
                                final_message[key] = value

                    if "finish_reason" in choice and choice["finish_reason"]:
                        finish_reason = choice["finish_reason"]

                if "usage" in chunk and chunk["usage"]:
                    usage_data = chunk["usage"]

            # --- Final Response Construction ---
            if aggregated_tool_calls:
                final_message["tool_calls"] = list(aggregated_tool_calls.values())
                finish_reason = "tool_calls"

            # Ensure standard fields are present for consistent logging
            for field in ["content", "tool_calls", "function_call"]:
                if field not in final_message:
                    final_message[field] = None

            first_chunk = response_chunks[0]
            final_choice = {
                "index": 0,
                "message": final_message,
                "finish_reason": finish_reason,
            }

            full_response = {
                "id": first_chunk.get("id"),
                "object": "chat.completion",
                "created": first_chunk.get("created"),
                "model": first_chunk.get("model"),
                "choices": [final_choice],
                "usage": usage_data,
            }

        if logger and not stream_failed:
            logger.log_final_response(
                status_code=200,
                headers=None,  # Headers are not available at this stage
                body=full_response,
            )


def _safe_http_exception(
    status: int,
    context: str,
    error: BaseException,
    *,
    raw_logger: Optional[RawIOLogger] = None,
    code: str | None = None,
) -> HTTPException:
    log_safe_exception(context, error, status)
    detail = public_error_detail(status, code=code)
    if raw_logger:
        raw_logger.log_final_response(status_code=status, headers=None, body={"error": detail})
    return HTTPException(status_code=status, detail=detail)


def reject_non_chat_inference_in_safe_mode() -> None:
    _ensure_local_transport_configuration_current()
    if _local_transport_runtime_policy.enabled:
        raise HTTPException(
            status_code=409,
            detail=public_error_detail(
                409,
                code="local_transport_endpoint_disabled",
            ),
        )


def _ensure_local_transport_configuration_current() -> None:
    if not _local_transport_runtime_policy.is_current(os.environ):
        raise HTTPException(
            status_code=503,
            detail=public_error_detail(
                503,
                code="local_transport_configuration_changed",
            ),
        )


def build_credential_summary(
    client: RotatingClient,
    disabled_provider_count: int | None = None,
) -> dict[str, Any]:
    api_key_provider_counts = {
        provider: len(credentials)
        for provider, credentials in sorted(client.api_keys.items())
        if credentials
    }
    oauth_provider_counts = {
        provider: len(credentials)
        for provider, credentials in sorted(client.oauth_credentials.items())
        if credentials
    }
    provider_counts = {
        provider: len(credentials)
        for provider, credentials in sorted(client.all_credentials.items())
        if credentials
    }
    summary = {
        "schema_version": "credential_summary.v1",
        "providers": provider_counts,
        "api_key_providers": api_key_provider_counts,
        "oauth_providers": oauth_provider_counts,
        "total_providers": len(provider_counts),
        "total_credentials": sum(provider_counts.values()),
    }
    if disabled_provider_count is not None:
        summary["disabled_provider_count"] = disabled_provider_count
    return summary


def print_startup_credential_summary(
    client: RotatingClient,
    disabled_provider_count: int = 0,
) -> None:
    summary = build_credential_summary(
        client,
        disabled_provider_count=disabled_provider_count,
    )
    print(f"Credential Summary: {json.dumps(summary, sort_keys=True)}")


@app.get("/v1/credential-summary")
async def credential_summary(
    _=Depends(verify_api_key),
    client: RotatingClient = Depends(get_rotating_client),
):
    return build_credential_summary(client)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint powered by the RotatingClient.
    Handles both streaming and non-streaming responses and logs them.
    """
    # Raw I/O logger captures unmodified HTTP data at proxy boundary (disabled by default)
    raw_logger = RawIOLogger() if ENABLE_RAW_LOGGING else None
    try:
        # Read and parse the request body only once at the beginning.
        try:
            request_data = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

        _ensure_local_transport_configuration_current()
        if _local_transport_runtime_policy.enabled:
            model = request_data.get("model")
            if (
                not isinstance(model, str)
                or not model.startswith("xai_oauth/")
                or model == "xai_oauth/"
            ):
                raise HTTPException(
                    status_code=400,
                    detail=public_error_detail(
                        400,
                        code="local_transport_xai_only",
                    ),
                )
        # Global temperature=0 override (controlled by .env variable, default: OFF)
        # Low temperature makes models deterministic and prone to following training data
        # instead of actual schemas, which can cause tool hallucination
        # Modes: "remove" = delete temperature key, "set" = change to 1.0, "false" = disabled
        override_temp_zero = os.getenv("OVERRIDE_TEMPERATURE_ZERO", "false").lower()

        if (
            override_temp_zero in ("remove", "set", "true", "1", "yes")
            and "temperature" in request_data
            and request_data["temperature"] == 0
        ):
            if override_temp_zero == "remove":
                # Remove temperature key entirely
                del request_data["temperature"]
                logging.debug(
                    "OVERRIDE_TEMPERATURE_ZERO=remove: Removed temperature=0 from request"
                )
            else:
                # Set to 1.0 (for "set", "true", "1", "yes")
                request_data["temperature"] = 1.0
                logging.debug(
                    "OVERRIDE_TEMPERATURE_ZERO=set: Converting temperature=0 to temperature=1.0"
                )

        # If raw logging is enabled, capture the unmodified request data.
        if raw_logger:
            raw_logger.log_request(headers=request.headers, body=request_data)

        # Extract and log specific reasoning parameters for monitoring.
        model = request_data.get("model")
        generation_cfg = (
            request_data.get("generationConfig", {})
            or request_data.get("generation_config", {})
            or {}
        )
        reasoning_effort = request_data.get("reasoning_effort") or generation_cfg.get(
            "reasoning_effort"
        )

        logging.getLogger("rotator_library").debug(
            f"Handling reasoning parameters: model={model}, reasoning_effort={reasoning_effort}"
        )

        if isinstance(model, str) and "/" not in model:
            raise ValueError(
                f"Plain alias model '{model}' is not supported on upstream port 8000. "
                f"Use the weighted router on port 8001 for clean aliases, or send a provider-prefixed model "
                f"such as 'ollama_cloud/{model}' or 'chutes/...'."
            )

        # Log basic request info to console (this is a separate, simpler logger).
        log_request_to_console(
            url=str(request.url),
            headers=dict(request.headers),
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )
        is_streaming = request_data.get("stream", False)

        if is_streaming:
            response_generator = client.acompletion(request=request, **request_data)
            return StreamingResponse(
                streaming_response_wrapper(request, request_data, response_generator, raw_logger),
                media_type="text/event-stream",
            )
        else:
            response = await client.acompletion(request=request, **request_data)
            if raw_logger:
                # Assuming response has status_code and headers attributes
                # This might need adjustment based on the actual response object
                response_headers = response.headers if hasattr(response, "headers") else None
                status_code = response.status_code if hasattr(response, "status_code") else 200
                raw_logger.log_final_response(
                    status_code=status_code,
                    headers=response_headers,
                    body=response.model_dump(),
                )
            return response

    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        raise _safe_http_exception(400, "OpenAI request", e, raw_logger=raw_logger)
    except HTTPException:
        raise
    except NoAvailableKeysError as e:
        raise _safe_http_exception(
            503,
            "OpenAI credential acquisition",
            e,
            raw_logger=raw_logger,
            code="proxy_busy",
        )
    except litellm.AuthenticationError as e:
        raise _safe_http_exception(401, "OpenAI authentication", e, raw_logger=raw_logger)
    except litellm.RateLimitError as e:
        raise _safe_http_exception(429, "OpenAI rate limit", e, raw_logger=raw_logger)
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        raise _safe_http_exception(503, "OpenAI service", e, raw_logger=raw_logger)
    except litellm.Timeout as e:
        raise _safe_http_exception(504, "OpenAI timeout", e, raw_logger=raw_logger)
    except (litellm.InternalServerError, litellm.OpenAIError) as e:
        raise _safe_http_exception(502, "OpenAI upstream", e, raw_logger=raw_logger)
    except Exception as e:
        raise _safe_http_exception(500, "OpenAI request", e, raw_logger=raw_logger)


# --- Anthropic Messages API Endpoint ---
@app.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    body: AnthropicMessagesRequest,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_anthropic_api_key),
    _local_transport_guard=Depends(reject_non_chat_inference_in_safe_mode),
):
    """
    Anthropic-compatible Messages API endpoint.

    Accepts requests in Anthropic's format and returns responses in Anthropic's format.
    Internally translates to OpenAI format for processing via LiteLLM.

    This endpoint is compatible with Claude Code and other Anthropic API clients.
    """
    # Initialize raw I/O logger if enabled (for debugging proxy boundary)
    logger = RawIOLogger() if ENABLE_RAW_LOGGING else None

    # Log raw Anthropic request if raw logging is enabled
    if logger:
        logger.log_request(
            headers=dict(request.headers),
            body=body.model_dump(exclude_none=True),
        )

    try:
        # Log the request to console
        log_request_to_console(
            url=str(request.url),
            headers=dict(request.headers),
            client_info=(
                request.client.host if request.client else "unknown",
                request.client.port if request.client else 0,
            ),
            request_data=body.model_dump(exclude_none=True),
        )

        # Use the library method to handle the request
        result = await client.anthropic_messages(body, raw_request=request)

        if body.stream:
            # Streaming response
            return StreamingResponse(
                anthropic_streaming_response_wrapper(result, logger, request=request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming response
            if logger:
                logger.log_final_response(
                    status_code=200,
                    headers=None,
                    body=result,
                )
            return JSONResponse(content=result)

    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        raise _safe_http_exception(400, "Anthropic messages", e, raw_logger=logger)
    except litellm.AuthenticationError as e:
        raise _safe_http_exception(401, "Anthropic messages", e, raw_logger=logger)
    except litellm.RateLimitError as e:
        raise _safe_http_exception(429, "Anthropic messages", e, raw_logger=logger)
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        raise _safe_http_exception(503, "Anthropic messages", e, raw_logger=logger)
    except litellm.Timeout as e:
        raise _safe_http_exception(504, "Anthropic messages", e, raw_logger=logger)
    except Exception as e:
        raise _safe_http_exception(500, "Anthropic messages", e, raw_logger=logger)


# --- Anthropic Count Tokens Endpoint ---
@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(
    request: Request,
    body: AnthropicCountTokensRequest,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_anthropic_api_key),
    _local_transport_guard=Depends(reject_non_chat_inference_in_safe_mode),
):
    """
    Anthropic-compatible count_tokens endpoint.

    Counts the number of tokens that would be used by a Messages API request.
    This is useful for estimating costs and managing context windows.

    Accepts requests in Anthropic's format and returns token count in Anthropic's format.
    """
    try:
        # Use the library method to handle the request
        result = await client.anthropic_count_tokens(body)
        return JSONResponse(content=result)

    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        raise _safe_http_exception(400, "Anthropic token count", e)
    except litellm.AuthenticationError as e:
        raise _safe_http_exception(401, "Anthropic token count", e)
    except litellm.RateLimitError as e:
        raise _safe_http_exception(429, "Anthropic token count", e)
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        raise _safe_http_exception(503, "Anthropic token count", e)
    except litellm.Timeout as e:
        raise _safe_http_exception(504, "Anthropic token count", e)
    except Exception as e:
        raise _safe_http_exception(500, "Anthropic token count", e)


@app.post("/v1/embeddings")
async def embeddings(
    request: Request,
    body: EmbeddingRequest,
    client: RotatingClient = Depends(get_rotating_client),
    batcher: Optional[EmbeddingBatcher] = Depends(get_embedding_batcher),
    _=Depends(verify_api_key),
    _local_transport_guard=Depends(reject_non_chat_inference_in_safe_mode),
):
    """
    OpenAI-compatible endpoint for creating embeddings.
    Supports two modes based on the USE_EMBEDDING_BATCHER flag:
    - True: Uses a server-side batcher for high throughput.
    - False: Passes requests directly to the provider.
    """
    try:
        request_data = body.model_dump(exclude_none=True)
        log_request_to_console(
            url=str(request.url),
            headers=dict(request.headers),
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )
        if USE_EMBEDDING_BATCHER and batcher:
            # --- Server-Side Batching Logic ---
            request_data = body.model_dump(exclude_none=True)
            inputs = request_data.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]

            tasks = []
            for single_input in inputs:
                individual_request = request_data.copy()
                individual_request["input"] = single_input
                tasks.append(batcher.add_request(individual_request))

            results = await asyncio.gather(*tasks)

            all_data = []
            total_prompt_tokens = 0
            total_tokens = 0
            for i, result in enumerate(results):
                result["data"][0]["index"] = i
                all_data.extend(result["data"])
                total_prompt_tokens += result["usage"]["prompt_tokens"]
                total_tokens += result["usage"]["total_tokens"]

            final_response_data = {
                "object": "list",
                "model": results[0]["model"],
                "data": all_data,
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "total_tokens": total_tokens,
                },
            }
            response = litellm.EmbeddingResponse(**final_response_data)

        else:
            # --- Direct Pass-Through Logic ---
            request_data = body.model_dump(exclude_none=True)
            if isinstance(request_data.get("input"), str):
                request_data["input"] = [request_data["input"]]

            response = await client.aembedding(request=request, **request_data)

        return response

    except HTTPException as e:
        # Re-raise HTTPException to ensure it's not caught by the generic Exception handler
        raise e
    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        raise _safe_http_exception(400, "Embedding request", e)
    except litellm.AuthenticationError as e:
        raise _safe_http_exception(401, "Embedding request", e)
    except litellm.RateLimitError as e:
        raise _safe_http_exception(429, "Embedding request", e)
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        raise _safe_http_exception(503, "Embedding request", e)
    except litellm.Timeout as e:
        raise _safe_http_exception(504, "Embedding request", e)
    except (litellm.InternalServerError, litellm.OpenAIError) as e:
        raise _safe_http_exception(502, "Embedding request", e)
    except Exception as e:
        raise _safe_http_exception(500, "Embedding request", e)


@app.get("/")
def read_root():
    return {"Status": "API Key Proxy is running"}


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "mirrowel-upstream"}


@app.get("/v1/models")
async def list_models(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
    enriched: bool = True,
):
    """
    Returns a list of available models in the OpenAI-compatible format.

    Query Parameters:
        enriched: If True (default), returns detailed model info with pricing and capabilities.
                  If False, returns minimal OpenAI-compatible response.
    """
    _ensure_local_transport_configuration_current()
    if _local_transport_runtime_policy.enabled:
        model_ids = await client.get_available_models("xai_oauth")
    else:
        model_ids = await client.get_all_available_models(grouped=False)

    if enriched and hasattr(request.app.state, "model_info_service"):
        model_info_service = request.app.state.model_info_service
        if model_info_service.is_ready:
            # Return enriched model data
            enriched_data = model_info_service.enrich_model_list(model_ids)
            return {"object": "list", "data": enriched_data}

    # Fallback to basic model cards
    model_cards = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "Mirro-Proxy",
        }
        for model_id in model_ids
    ]
    return {"object": "list", "data": model_cards}


@app.get("/v1/models/{model_id:path}")
async def get_model(
    model_id: str,
    request: Request,
    _=Depends(verify_api_key),
):
    """
    Returns detailed information about a specific model.

    Path Parameters:
        model_id: The model ID (e.g., "anthropic/claude-3-opus", "openrouter/openai/gpt-4")
    """
    if hasattr(request.app.state, "model_info_service"):
        model_info_service = request.app.state.model_info_service
        if model_info_service.is_ready:
            info = model_info_service.get_model_info(model_id)
            if info:
                return info.to_dict()

    # Return basic info if service not ready or model not found
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": model_id.split("/")[0] if "/" in model_id else "unknown",
    }


@app.get("/v1/model-info/stats")
async def model_info_stats(
    request: Request,
    _=Depends(verify_api_key),
):
    """
    Returns statistics about the model info service (for monitoring/debugging).
    """
    if hasattr(request.app.state, "model_info_service"):
        return request.app.state.model_info_service.get_stats()
    return {"error": "Model info service not initialized"}


@app.get("/v1/providers")
async def list_providers(_=Depends(verify_api_key)):
    """
    Returns a list of all available providers.
    """
    return list(PROVIDER_PLUGINS.keys())


@app.get("/v1/quota-stats")
async def get_quota_stats(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
    provider: str = None,
):
    """
    Returns quota and usage statistics for all credentials.

    This returns cached data from the proxy without making external API calls.
    Use POST to reload from disk or force refresh from external APIs.

    Query Parameters:
        provider: Optional filter to return stats for a specific provider only

    Returns:
        {
            "providers": {
                "provider_name": {
                    "credential_count": int,
                    "active_count": int,
                    "on_cooldown_count": int,
                    "exhausted_count": int,
                    "total_requests": int,
                    "tokens": {...},
                    "approx_cost": float | null,
                    "quota_groups": {...},  // For Antigravity
                    "credentials": [...]
                }
            },
            "summary": {...},
            "data_source": "cache",
            "timestamp": float
        }
    """
    _ensure_local_transport_configuration_current()
    try:
        stats = await client.get_quota_stats(provider_filter=provider)
        return stats
    except Exception as e:
        raise _safe_http_exception(500, "Quota stats", e)


@app.post("/v1/quota-stats")
async def refresh_quota_stats(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    Refresh quota and usage statistics.

    Request body:
        {
            "action": "reload" | "force_refresh",
            "scope": "all" | "provider" | "credential",
            "provider": "antigravity",  // required if scope != "all"
            "credential": "antigravity_oauth_1.json"  // required if scope == "credential"
        }

    Actions:
        - reload: Re-read data from disk (no external API calls)
        - force_refresh: For Antigravity, fetch live quota from API.
                        For other providers, same as reload.

    Returns:
        Same as GET, plus a "refresh_result" field with operation details.
    """
    _ensure_local_transport_configuration_current()
    try:
        data = await request.json()
        action = data.get("action", "reload")
        scope = data.get("scope", "all")
        provider = data.get("provider")
        credential = data.get("credential")

        if action == "force_refresh" and _local_transport_runtime_policy.enabled:
            raise HTTPException(
                status_code=409,
                detail=public_error_detail(
                    409,
                    code="local_transport_live_refresh_disabled",
                ),
            )

        # Validate parameters
        if action not in ("reload", "force_refresh"):
            raise HTTPException(
                status_code=400,
                detail="action must be 'reload' or 'force_refresh'",
            )

        if scope not in ("all", "provider", "credential"):
            raise HTTPException(
                status_code=400,
                detail="scope must be 'all', 'provider', or 'credential'",
            )

        if scope in ("provider", "credential") and not provider:
            raise HTTPException(
                status_code=400,
                detail="'provider' is required when scope is 'provider' or 'credential'",
            )

        if scope == "credential" and not credential:
            raise HTTPException(
                status_code=400,
                detail="'credential' is required when scope is 'credential'",
            )

        refresh_result = {
            "action": action,
            "scope": scope,
        }

        if action == "reload":
            # Just reload from disk
            start_time = time.time()
            await client.reload_usage_from_disk()
            refresh_result["duration_ms"] = int((time.time() - start_time) * 1000)
            refresh_result["success"] = True
            refresh_result["message"] = "Reloaded usage data from disk"

        elif action == "force_refresh":
            # Force refresh from external API (for supported providers like Antigravity)
            result = await client.force_refresh_quota(
                provider=provider if scope in ("provider", "credential") else None,
                credential=credential if scope == "credential" else None,
            )
            refresh_result.update(result)
            refresh_result["success"] = result["failed_count"] == 0

        # Get updated stats
        stats = await client.get_quota_stats(provider_filter=provider)
        stats["refresh_result"] = refresh_result
        stats["data_source"] = "refreshed"

        return stats

    except HTTPException:
        raise
    except Exception as e:
        raise _safe_http_exception(500, "Quota refresh", e)


@app.post("/v1/token-count")
async def token_count(
    request: Request,
    _local_transport_guard=Depends(reject_non_chat_inference_in_safe_mode),
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    try:
        data = await request.json()
        model = data.get("model")
        messages = data.get("messages")

        if not model or not messages:
            raise HTTPException(status_code=400, detail="'model' and 'messages' are required.")

        count = client.token_count(**data)
        return {"token_count": count}

    except HTTPException:
        raise
    except Exception as e:
        raise _safe_http_exception(500, "Token count", e)


@app.post("/v1/cost-estimate")
async def cost_estimate(request: Request, _=Depends(verify_api_key)):
    """
    Estimates the cost for a request based on token counts and model pricing.

    Request body:
        {
            "model": "anthropic/claude-3-opus",
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "cache_read_tokens": 0,       # optional
            "cache_creation_tokens": 0    # optional
        }

    Returns:
        {
            "model": "anthropic/claude-3-opus",
            "cost": 0.0375,
            "currency": "USD",
            "pricing": {
                "input_cost_per_token": 0.000015,
                "output_cost_per_token": 0.000075
            },
            "source": "model_info_service"  # or "litellm_fallback"
        }
    """
    try:
        data = await request.json()
        model = data.get("model")
        prompt_tokens = data.get("prompt_tokens", 0)
        completion_tokens = data.get("completion_tokens", 0)
        cache_read_tokens = data.get("cache_read_tokens", 0)
        cache_creation_tokens = data.get("cache_creation_tokens", 0)

        if not model:
            raise HTTPException(status_code=400, detail="'model' is required.")

        result = {
            "model": model,
            "cost": None,
            "currency": "USD",
            "pricing": {},
            "source": None,
        }

        # Try model info service first
        if hasattr(request.app.state, "model_info_service"):
            model_info_service = request.app.state.model_info_service
            if model_info_service.is_ready:
                cost = model_info_service.calculate_cost(
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                )
                if cost is not None:
                    cost_info = model_info_service.get_cost_info(model)
                    result["cost"] = cost
                    result["pricing"] = cost_info or {}
                    result["source"] = "model_info_service"
                    return result

        # Fallback to litellm
        try:
            import litellm

            # Create a mock response for cost calculation
            model_info = litellm.get_model_info(model)
            input_cost = model_info.get("input_cost_per_token", 0)
            output_cost = model_info.get("output_cost_per_token", 0)

            if input_cost or output_cost:
                cost = (prompt_tokens * input_cost) + (completion_tokens * output_cost)
                result["cost"] = cost
                result["pricing"] = {
                    "input_cost_per_token": input_cost,
                    "output_cost_per_token": output_cost,
                }
                result["source"] = "litellm_fallback"
                return result
        except Exception:
            pass

        result["source"] = "unknown"
        result["error"] = "Pricing data not available for this model"
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise _safe_http_exception(500, "Cost estimate", e)


if __name__ == "__main__":
    # Define ENV_FILE for onboarding checks using centralized path
    ENV_FILE = get_data_file(".env")

    # Check if launcher TUI should be shown (no arguments provided)
    if len(sys.argv) == 1:
        # No arguments - show launcher TUI (lazy import)
        from proxy_app.launcher_tui import run_launcher_tui

        run_launcher_tui()
        # Launcher modifies sys.argv and returns, or exits if user chose Exit
        # If we get here, user chose "Run Proxy" and sys.argv is modified
        # Re-parse arguments with modified sys.argv
        args = parser.parse_args()

    def needs_onboarding() -> bool:
        """
        Check if the proxy needs onboarding (first-time setup).
        Returns True if onboarding is needed, False otherwise.
        """
        # Only check if .env file exists
        # PROXY_API_KEY is optional (will show warning if not set)
        if not ENV_FILE.is_file():
            return True

        return False

    def show_onboarding_message():
        """Display clear explanatory message for why onboarding is needed."""
        os.system("cls" if os.name == "nt" else "clear")  # Clear terminal for clean presentation
        console.print(
            Panel.fit(
                "[bold cyan]🚀 LLM API Key Proxy - First Time Setup[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print("[bold yellow]⚠️  Configuration Required[/bold yellow]\n")

        console.print("The proxy needs initial configuration:")
        console.print("  [red]❌ No .env file found[/red]")

        console.print("\n[bold]Why this matters:[/bold]")
        console.print("  • The .env file stores your credentials and settings")
        console.print("  • PROXY_API_KEY protects your proxy from unauthorized access")
        console.print("  • Provider API keys enable LLM access")

        console.print("\n[bold]What happens next:[/bold]")
        console.print("  1. We'll create a .env file with PROXY_API_KEY")
        console.print("  2. You can add LLM provider credentials (API keys or OAuth)")
        console.print("  3. The proxy will then start normally")

        console.print(
            "\n[bold yellow]⚠️  Note:[/bold yellow] The credential tool adds PROXY_API_KEY by default."
        )
        console.print("   You can remove it later if you want an unsecured proxy.\n")

        console.input("[bold green]Press Enter to launch the credential setup tool...[/bold green]")

    # Check if user explicitly wants to add credentials
    if args.add_credential:
        # Import and call ensure_env_defaults to create .env and PROXY_API_KEY if needed
        from rotator_library.credential_tool import ensure_env_defaults

        ensure_env_defaults()
        # Reload environment variables after ensure_env_defaults creates/updates .env
        _load_router_env(ENV_FILE, _tui_network_bind_approval)
        run_credential_tool()
    else:
        # Check if onboarding is needed
        if needs_onboarding():
            # Import console from rich for better messaging
            from rich.console import Console
            from rich.panel import Panel

            console = Console()

            # Show clear explanatory message
            show_onboarding_message()

            # Launch credential tool automatically
            from rotator_library.credential_tool import ensure_env_defaults

            ensure_env_defaults()
            _load_router_env(ENV_FILE, _tui_network_bind_approval)
            run_credential_tool()

            # After credential tool exits, reload and re-check
            _load_router_env(ENV_FILE, _tui_network_bind_approval)
            _runtime_security_config = _build_runtime_security_config()

            # Verify onboarding is complete
            if needs_onboarding():
                console.print("\n[bold red]❌ Configuration incomplete.[/bold red]")
                console.print(
                    "The proxy still cannot start. Please ensure PROXY_API_KEY is set in .env\n"
                )
                sys.exit(1)
            else:
                console.print("\n[bold green]✅ Configuration complete![/bold green]")
                console.print("\nStarting proxy server...\n")

        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
