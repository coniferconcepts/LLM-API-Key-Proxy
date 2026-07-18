from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anyio
from fastapi import FastAPI
import pytest

from proxy_app.app_lifecycle import LifecycleDependencies, application_lifespan


@dataclass
class FakeLiteLLM:
    aclient_session: Any
    set_verbose: bool = True
    drop_params: bool = False


def _dependencies(
    events: list[str],
    *,
    safe_mode: bool,
    fail_at: str | None = None,
    cleanup_failure: str | None = None,
) -> tuple[LifecycleDependencies, FakeLiteLLM]:
    class CredentialManager:
        def __init__(self, _environment: dict[str, str]) -> None:
            events.append("credentials.construct")

        def discover_and_prepare(self) -> dict[str, list[str]]:
            events.append("credentials.discover")
            return {}

    class Refresher:
        def start(self) -> None:
            events.append("refresher.start")

        async def stop(self) -> None:
            events.append("refresher.stop")
            if cleanup_failure == "refresher":
                raise RuntimeError("refresher cleanup failed")

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            events.append("client.construct")
            self.all_credentials: dict[str, list[str]] = {}
            self.http_client = object()
            self.background_refresher = Refresher()

        async def close(self) -> None:
            events.append("client.close")
            if cleanup_failure == "client":
                raise RuntimeError("client cleanup failed")

    class Batcher:
        async def stop(self) -> None:
            events.append("batcher.stop")
            if cleanup_failure == "batcher":
                raise RuntimeError("batcher cleanup failed")

    class ModelInfo:
        async def stop(self) -> None:
            events.append("model.stop")
            if cleanup_failure == "model":
                raise RuntimeError("model cleanup failed")

    def print_summary(*_args: Any, **_kwargs: Any) -> None:
        events.append("summary")
        if fail_at == "after_client":
            raise RuntimeError("startup failed after client")

    def make_batcher(**_kwargs: Any) -> Batcher:
        if fail_at == "after_refresher":
            raise RuntimeError("startup failed after refresher")
        events.append("batcher.construct")
        return Batcher()

    async def make_model() -> ModelInfo:
        if fail_at == "after_batcher":
            raise RuntimeError("startup failed after batcher")
        events.append("model.construct")
        return ModelInfo()

    litellm = FakeLiteLLM(aclient_session=object())
    dependencies = LifecycleDependencies(
        credential_manager_factory=CredentialManager,
        rotating_client_factory=Client,
        provider_plugins={},
        init_model_info_service=make_model,
        embedding_batcher_factory=make_batcher,
        litellm=litellm,
        log_safe_exception=lambda label, _error, _status: events.append(f"logged:{label}"),
        print_startup_credential_summary=print_summary,
        local_transport_safe_mode_enabled=lambda: safe_mode,
        api_keys={},
        disabled_providers=frozenset(),
        ignore_models={},
        whitelist_models={},
        max_concurrent_requests_per_key={},
        enable_request_logging=False,
        use_embedding_batcher=not safe_mode,
    )
    return dependencies, litellm


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fail_at", "expected_cleanup"),
    (
        ("after_client", ["client.close"]),
        ("after_refresher", ["refresher.stop", "client.close"]),
        ("after_batcher", ["refresher.stop", "batcher.stop", "client.close"]),
    ),
)
async def test_partial_startup_cleans_each_owned_resource_once(
    fail_at: str,
    expected_cleanup: list[str],
) -> None:
    events: list[str] = []
    dependencies, _litellm = _dependencies(events, safe_mode=False, fail_at=fail_at)

    with pytest.raises(RuntimeError, match="startup failed"):
        async with application_lifespan(FastAPI(), dependencies):
            pytest.fail("startup unexpectedly reached serving")

    cleanup = [event for event in events if event.endswith((".stop", ".close"))]
    assert cleanup == expected_cleanup


@pytest.mark.asyncio
async def test_serving_exception_preserves_established_cleanup_order() -> None:
    events: list[str] = []
    dependencies, _litellm = _dependencies(events, safe_mode=False)

    with pytest.raises(RuntimeError, match="serving failed"):
        async with application_lifespan(FastAPI(), dependencies):
            events.append("serving")
            raise RuntimeError("serving failed")

    assert events[-4:] == ["refresher.stop", "batcher.stop", "client.close", "model.stop"]


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_skip_remaining_resources() -> None:
    events: list[str] = []
    dependencies, _litellm = _dependencies(events, safe_mode=False, cleanup_failure="batcher")

    with pytest.raises(RuntimeError, match="batcher cleanup failed"):
        async with application_lifespan(FastAPI(), dependencies):
            events.append("serving")

    assert events[-5:] == [
        "refresher.stop",
        "batcher.stop",
        "logged:Lifecycle cleanup: embedding batcher",
        "client.close",
        "model.stop",
    ]


@pytest.mark.asyncio
async def test_cancellation_restores_session_and_closes_client_once() -> None:
    events: list[str] = []
    dependencies, litellm = _dependencies(events, safe_mode=True)
    original_session = litellm.aclient_session
    serving = anyio.Event()

    async def run_lifespan() -> None:
        async with application_lifespan(FastAPI(), dependencies):
            serving.set()
            await anyio.sleep_forever()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_lifespan)
        await serving.wait()
        task_group.cancel_scope.cancel()

    assert litellm.aclient_session is original_session
    assert events.count("client.close") == 1
