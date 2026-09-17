"""Application startup and exactly-once shutdown orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from typing import Any, AsyncIterator, Protocol

import anyio
from fastapi import FastAPI
import aiohttp

from proxy_app.oauth_bootstrap import (
    ProviderFactory,
    SafeExceptionLogger,
    initialize_oauth_credentials,
)


class BackgroundRefresher(Protocol):
    def start(self) -> None: ...

    async def stop(self) -> None: ...


class RotatingClientRuntime(Protocol):
    all_credentials: Mapping[str, list[str]]
    http_client: Any
    background_refresher: BackgroundRefresher
    litellm_shared_session: Any
    trust_env: bool

    async def close(self) -> None: ...


class CredentialManagerRuntime(Protocol):
    def discover_and_prepare(self) -> dict[str, list[str]]: ...


class EmbeddingBatcherRuntime(Protocol):
    async def stop(self) -> None: ...


class ModelInfoRuntime(Protocol):
    async def stop(self) -> None: ...


class LiteLLMRuntime(Protocol):
    aclient_session: Any
    set_verbose: bool
    drop_params: bool


CredentialManagerFactory = Callable[[dict[str, str]], CredentialManagerRuntime]
RotatingClientFactory = Callable[..., RotatingClientRuntime]
EmbeddingBatcherFactory = Callable[..., EmbeddingBatcherRuntime]
ModelInfoFactory = Callable[[], Awaitable[ModelInfoRuntime]]
CredentialSummaryPrinter = Callable[..., None]
AsyncCleanup = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class LifecycleDependencies:
    credential_manager_factory: CredentialManagerFactory
    rotating_client_factory: RotatingClientFactory
    provider_plugins: Mapping[str, ProviderFactory]
    init_model_info_service: ModelInfoFactory
    embedding_batcher_factory: EmbeddingBatcherFactory
    litellm: LiteLLMRuntime
    log_safe_exception: SafeExceptionLogger
    print_startup_credential_summary: CredentialSummaryPrinter
    local_transport_safe_mode_enabled: Callable[[], bool]
    api_keys: Mapping[str, list[str]]
    disabled_providers: frozenset[str]
    ignore_models: Mapping[str, list[str]]
    whitelist_models: Mapping[str, list[str]]
    max_concurrent_requests_per_key: Mapping[str, int]
    enable_request_logging: bool
    use_embedding_batcher: bool


def _local_only_credentials(
    credentials: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    return {"xai_oauth": credentials["xai_oauth"]} if credentials.get("xai_oauth") else {}


def _without_disabled_providers(
    credentials: Mapping[str, list[str]],
    disabled_providers: frozenset[str],
) -> dict[str, list[str]]:
    return {
        provider: list(values)
        for provider, values in credentials.items()
        if provider not in disabled_providers
    }


async def _run_cleanup(
    operation: AsyncCleanup,
    label: str,
    log_safe_exception: SafeExceptionLogger,
    errors: list[BaseException],
) -> None:
    try:
        await operation()
    except BaseException as exc:
        errors.append(exc)
        log_safe_exception(f"Lifecycle cleanup: {label}", exc, 500)


def _restore_litellm_session(
    litellm: LiteLLMRuntime,
    owned_session: Any,
    previous_session: Any,
) -> None:
    if litellm.aclient_session is owned_session:
        litellm.aclient_session = previous_session


async def _drop_litellm_async_httpx_cache(litellm_module: Any) -> None:
    closer = getattr(litellm_module, "close_litellm_async_clients", None)
    if callable(closer):
        try:
            await closer()
        except Exception:
            pass
    cache = getattr(litellm_module, "in_memory_llm_clients_cache", None)
    cache_dict = getattr(cache, "cache_dict", None) if cache is not None else None
    if isinstance(cache_dict, dict):
        cache_dict.clear()


@asynccontextmanager
async def application_lifespan(
    app: FastAPI,
    dependencies: LifecycleDependencies,
) -> AsyncIterator[None]:
    cleanup_errors: list[BaseException] = []
    lifecycle_failed = False
    client: RotatingClientRuntime | None = None
    refresher_started = False
    batcher: EmbeddingBatcherRuntime | None = None
    model_info: ModelInfoRuntime | None = None
    previous_session: Any = None
    owned_session: Any = None
    session_replaced = False
    aiohttp_session: Any = None
    local_transport_safe_mode = dependencies.local_transport_safe_mode_enabled()
    app.state.embedding_batcher = None
    app.state.model_info_service = None
    try:
        skip_oauth_init = (
            local_transport_safe_mode
            or os.getenv("SKIP_OAUTH_INIT_CHECK", "false").lower() == "true"
        )
        credential_manager = dependencies.credential_manager_factory(dict(os.environ))
        oauth_credentials = credential_manager.discover_and_prepare()
        if local_transport_safe_mode:
            oauth_credentials = _local_only_credentials(oauth_credentials)
        if not skip_oauth_init and oauth_credentials:
            oauth_credentials = await initialize_oauth_credentials(
                oauth_credentials,
                dependencies.provider_plugins,
                dependencies.log_safe_exception,
            )
        if dependencies.disabled_providers:
            oauth_credentials = _without_disabled_providers(
                oauth_credentials, dependencies.disabled_providers
            )

        runtime_api_keys = dict(dependencies.api_keys)
        if local_transport_safe_mode:
            runtime_api_keys = _local_only_credentials(runtime_api_keys)
        provider_params = {"gemini_cli": {"project_id": os.getenv("GEMINI_CLI_PROJECT_ID")}}
        if local_transport_safe_mode:
            provider_params = {}
        client = dependencies.rotating_client_factory(
            api_keys=runtime_api_keys,
            oauth_credentials=oauth_credentials,
            configure_logging=True,
            global_timeout=int(os.getenv("GLOBAL_TIMEOUT", "30")),
            acquire_timeout=int(os.getenv("ACQUIRE_TIMEOUT", "10")),
            litellm_provider_params=provider_params,
            ignore_models=dependencies.ignore_models,
            whitelist_models=dependencies.whitelist_models,
            enable_request_logging=dependencies.enable_request_logging,
            max_concurrent_requests_per_key=dependencies.max_concurrent_requests_per_key,
            trust_env=not local_transport_safe_mode,
        )
        dependencies.print_startup_credential_summary(
            client, disabled_provider_count=len(dependencies.disabled_providers)
        )
        if not local_transport_safe_mode:
            client.background_refresher.start()
            refresher_started = True
        app.state.rotating_client = client

        os.environ["LITELLM_LOG"] = "ERROR"
        dependencies.litellm.set_verbose = False
        dependencies.litellm.drop_params = True
        if dependencies.use_embedding_batcher:
            batcher = dependencies.embedding_batcher_factory(client=client)
            app.state.embedding_batcher = batcher
        if not local_transport_safe_mode:
            model_info = await dependencies.init_model_info_service()
            app.state.model_info_service = model_info
            trust_env = bool(getattr(client, "trust_env", True))
            aiohttp_session = aiohttp.ClientSession(trust_env=trust_env)
            client.litellm_shared_session = aiohttp_session
        else:
            previous_session = dependencies.litellm.aclient_session
            owned_session = client.http_client
            dependencies.litellm.aclient_session = owned_session
            session_replaced = True
        yield
    except BaseException:
        lifecycle_failed = True
        raise
    finally:
        with anyio.CancelScope(shield=True):
            if session_replaced:
                _restore_litellm_session(dependencies.litellm, owned_session, previous_session)
            if client is not None and refresher_started:
                await _run_cleanup(
                    client.background_refresher.stop,
                    "background refresher",
                    dependencies.log_safe_exception,
                    cleanup_errors,
                )
            if batcher is not None:
                await _run_cleanup(
                    batcher.stop,
                    "embedding batcher",
                    dependencies.log_safe_exception,
                    cleanup_errors,
                )
            if client is not None:
                await _run_cleanup(
                    client.close,
                    "rotating client",
                    dependencies.log_safe_exception,
                    cleanup_errors,
                )
            if aiohttp_session is not None:

                async def _close_owned_aiohttp() -> None:
                    await aiohttp_session.close()
                    if client is not None:
                        client.litellm_shared_session = None
                    await _drop_litellm_async_httpx_cache(dependencies.litellm)

                await _run_cleanup(
                    _close_owned_aiohttp,
                    "litellm shared aiohttp session",
                    dependencies.log_safe_exception,
                    cleanup_errors,
                )
            if model_info is not None:
                await _run_cleanup(
                    model_info.stop,
                    "model info service",
                    dependencies.log_safe_exception,
                    cleanup_errors,
                )
        if cleanup_errors and not lifecycle_failed:
            raise cleanup_errors[0]
