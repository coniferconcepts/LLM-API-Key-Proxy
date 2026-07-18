from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_environment_load_refreshes_an_already_imported_provider_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from proxy_app.bootstrap_env import load_router_env

    registrations: list[str] = []
    provider_module = SimpleNamespace(_register_providers=lambda: registrations.append("refreshed"))
    monkeypatch.setitem(sys.modules, "rotator_library.providers", provider_module)
    env_file = tmp_path / ".env"
    env_file.write_text("SYNTHETIC_API_BASE=http://127.0.0.1:1/v1\n", encoding="utf-8")

    load_router_env(env_file)

    assert registrations == ["refreshed"]


def test_environment_load_registers_dynamic_provider_after_registry_preload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from proxy_app.bootstrap_env import load_router_env
    from rotator_library import providers

    provider_name = "synthetic_import_order"
    providers.PROVIDER_PLUGINS.pop(provider_name, None)
    monkeypatch.setenv(
        "SYNTHETIC_IMPORT_ORDER_API_BASE",
        "http://127.0.0.1:1/v1",
    )
    try:
        load_router_env(tmp_path / ".env")

        assert provider_name in providers.PROVIDER_PLUGINS
    finally:
        providers.PROVIDER_PLUGINS.pop(provider_name, None)


def _import_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for key in list(os.environ):
        if key != "PROXY_API_KEY" and (
            "_API_KEY" in key or key.endswith("_API_BASE") or key.endswith("_KEY")
        ):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SKIP_OAUTH_INIT_CHECK", "true")
    monkeypatch.setenv("MIRROWEL_LOCAL_TRANSPORT_SAFE_MODE", "true")
    monkeypatch.setenv("MIRROWEL_ALLOWED_HOSTS", "127.0.0.1,localhost,::1,testserver")
    monkeypatch.delenv("PROXY_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest", "--host", "127.0.0.1", "--port", "8000"])
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    sys.modules.pop("proxy_app.main", None)
    return importlib.import_module("proxy_app.main")


async def _invoke(app: Any, *, host: bytes) -> tuple[int, bytes]:
    sent: list[dict[str, Any]] = []
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/health",
            "raw_path": b"/health",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", host)],
            "client": ("127.0.0.1", 42000),
            "server": ("127.0.0.1", 8000),
        },
        receive,
        send,
    )
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return status, body


@pytest.mark.asyncio
async def test_real_app_preserves_middleware_order_and_health_request_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given the production app assembled under its loopback-safe configuration.
    module = _import_main(monkeypatch, tmp_path)
    module.app.middleware_stack = module.app.build_middleware_stack()
    chain: list[str] = []
    current = module.app.middleware_stack
    while hasattr(current, "app"):
        chain.append(type(current).__name__)
        current = current.app

    # When a valid request and a malformed-Host request traverse that same stack.
    valid_status, valid_body = await _invoke(module.app, host=b"127.0.0.1:8000")
    rejected_status, rejected_body = await _invoke(module.app, host=b"bad host")

    # Then the security boundary remains outside CORS/body parsing and routing.
    assert chain[1:6] == [
        "SafeUnhandledErrorMiddleware",
        "BindApprovalMiddleware",
        "TrustedHostMiddleware",
        "CORSMiddleware",
        "BoundedJSONBodyMiddleware",
    ]
    assert (valid_status, valid_body) == (
        200,
        b'{"status":"ok","service":"mirrowel-upstream"}',
    )
    assert rejected_status == 400
    assert rejected_body == b"Invalid host header"


@pytest.mark.asyncio
async def test_lifespan_constructs_and_tears_down_client_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given isolated lifecycle collaborators and no live provider credentials.
    module = _import_main(monkeypatch, tmp_path)
    events: list[str] = []
    original_session = object()
    module.litellm.aclient_session = original_session

    class FakeCredentialManager:
        def __init__(self, _environment: dict[str, str]) -> None:
            events.append("credentials.construct")

        def discover_and_prepare(self) -> dict[str, list[str]]:
            events.append("credentials.discover")
            return {}

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            events.append("client.construct")
            self.all_credentials: list[str] = []
            self.http_client = object()

        async def close(self) -> None:
            events.append("client.close")

    monkeypatch.setattr(module, "CredentialManager", FakeCredentialManager)
    monkeypatch.setattr(module, "RotatingClient", FakeClient)
    monkeypatch.setattr(module, "print_startup_credential_summary", lambda *_a, **_k: None)
    app = FastAPI()

    # When the lifespan context starts and exits normally.
    async with module.lifespan(app):
        events.append("serving")
        assert app.state.rotating_client.http_client is module.litellm.aclient_session
        assert app.state.embedding_batcher is None

    # Then construction and teardown occur once, in order, and global state is restored.
    assert events == [
        "credentials.construct",
        "credentials.discover",
        "client.construct",
        "serving",
        "client.close",
    ]
    assert module.litellm.aclient_session is original_session


@pytest.mark.asyncio
async def test_lifespan_restores_and_closes_when_serving_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_main(monkeypatch, tmp_path)
    events: list[str] = []
    original_session = object()
    module.litellm.aclient_session = original_session

    class FakeCredentialManager:
        def __init__(self, _environment: dict[str, str]) -> None:
            return

        def discover_and_prepare(self) -> dict[str, list[str]]:
            return {}

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.all_credentials: list[str] = []
            self.http_client = object()

        async def close(self) -> None:
            events.append("client.close")

    monkeypatch.setattr(module, "CredentialManager", FakeCredentialManager)
    monkeypatch.setattr(module, "RotatingClient", FakeClient)
    monkeypatch.setattr(module, "print_startup_credential_summary", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="serving failed"):
        async with module.lifespan(FastAPI()):
            raise RuntimeError("serving failed")

    assert events == ["client.close"]
    assert module.litellm.aclient_session is original_session
