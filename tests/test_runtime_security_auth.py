from __future__ import annotations

import asyncio
import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import Depends

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key != "PROXY_API_KEY" and (
            "_API_KEY" in key or key.endswith("_API_BASE") or key.endswith("_KEY")
        ):
            monkeypatch.delenv(key, raising=False)


def _import_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    proxy_api_key: str | None,
) -> ModuleType:
    _clear_provider_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SKIP_OAUTH_INIT_CHECK", "true")
    monkeypatch.setenv("MIRROWEL_LOCAL_TRANSPORT_SAFE_MODE", "true")
    monkeypatch.setenv("MIRROWEL_ALLOW_NETWORK_BIND", "true")
    monkeypatch.setenv("MIRROWEL_ALLOWED_HOSTS", "proxy.example.com")
    monkeypatch.setattr(sys, "argv", ["pytest", "--port", "8000"])
    if proxy_api_key is None:
        monkeypatch.delenv("PROXY_API_KEY", raising=False)
    else:
        monkeypatch.setenv("PROXY_API_KEY", proxy_api_key)

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    sys.modules.pop("proxy_app.main", None)
    return importlib.import_module("proxy_app.main")


def _invoke_app(
    app: Callable,
    path: str,
    headers: list[tuple[bytes, bytes]],
) -> tuple[int, bytes]:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"proxy.example.com"), *headers],
        "client": ("192.0.2.20", 54321),
        "server": ("0.0.0.0", 8000),
    }
    sent: list[dict[str, object]] = []
    request_delivered = False

    async def receive() -> dict[str, object]:
        nonlocal request_delivered
        if request_delivered:
            return {"type": "http.disconnect"}
        request_delivered = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(
        int(message["status"]) for message in sent if message["type"] == "http.response.start"
    )
    body = b"".join(
        bytes(message.get("body", b""))
        for message in sent
        if message["type"] == "http.response.body"
    )
    return status, body


def test_import_before_key_then_public_environment_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_main(monkeypatch, tmp_path, proxy_api_key=None)

    @module.app.get("/_runtime-security-import-order-probe")
    async def probe(_credential=Depends(module.verify_api_key)) -> dict[str, bool]:
        return {"accepted": True}

    monkeypatch.setenv("PROXY_API_KEY", "late-public-key")

    status, body = _invoke_app(
        module.app,
        "/_runtime-security-import-order-probe",
        [],
    )

    assert status == 503
    assert b"runtime security configuration changed" in body.lower()


@pytest.mark.parametrize(
    ("dependency_name", "headers"),
    [
        (
            "verify_api_key",
            [(b"authorization", b"Bearer proxy-token"), (b"authorization", b"Bearer wrong")],
        ),
        (
            "verify_api_key",
            [(b"authorization", b"Bearer wrong"), (b"authorization", b"Bearer proxy-token")],
        ),
        (
            "verify_anthropic_api_key",
            [(b"x-api-key", b"proxy-token"), (b"x-api-key", b"wrong")],
        ),
        (
            "verify_anthropic_api_key",
            [(b"x-api-key", b"wrong"), (b"x-api-key", b"proxy-token")],
        ),
        (
            "verify_anthropic_api_key",
            [(b"authorization", b"Bearer proxy-token"), (b"x-api-key", b"proxy-token")],
        ),
        (
            "verify_anthropic_api_key",
            [(b"x-api-key", b"proxy-token"), (b"authorization", b"Bearer proxy-token")],
        ),
    ],
)
def test_raw_asgi_ambiguous_credentials_are_rejected_in_both_orders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dependency_name: str,
    headers: list[tuple[bytes, bytes]],
) -> None:
    module = _import_main(monkeypatch, tmp_path, proxy_api_key="proxy-token")
    dependency = getattr(module, dependency_name)
    path = f"/_raw-auth-probe/{dependency_name}"

    @module.app.get(path)
    async def probe(_credential=Depends(dependency)) -> dict[str, bool]:
        return {"accepted": True}

    status, _body = _invoke_app(module.app, path, headers)

    assert status == 401


@pytest.mark.parametrize(
    ("dependency_name", "headers", "expected_comparison"),
    [
        (
            "verify_api_key",
            [(b"authorization", b"Bearer proxy-token")],
            (b"Bearer proxy-token", b"Bearer proxy-token"),
        ),
        (
            "verify_anthropic_api_key",
            [(b"authorization", b"Bearer proxy-token")],
            (b"Bearer proxy-token", b"Bearer proxy-token"),
        ),
        (
            "verify_anthropic_api_key",
            [(b"x-api-key", b"proxy-token")],
            (b"proxy-token", b"proxy-token"),
        ),
    ],
)
def test_configured_key_comparisons_use_constant_time_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dependency_name: str,
    headers: list[tuple[bytes, bytes]],
    expected_comparison: tuple[bytes, bytes],
) -> None:
    module = _import_main(monkeypatch, tmp_path, proxy_api_key="proxy-token")
    compared: list[tuple[bytes, bytes]] = []

    def record_compare(left: bytes, right: bytes) -> bool:
        compared.append((left, right))
        return left == right

    monkeypatch.setattr(module.secrets, "compare_digest", record_compare)
    dependency = getattr(module, dependency_name)
    path = f"/_constant-time-auth-probe/{dependency_name}"

    @module.app.get(path)
    async def probe(_credential=Depends(dependency)) -> dict[str, bool]:
        return {"accepted": True}

    status, _body = _invoke_app(module.app, path, headers)

    assert status == 200
    assert compared == [expected_comparison]


def test_non_ascii_raw_credential_is_rejected_without_compare_digest_type_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_main(monkeypatch, tmp_path, proxy_api_key="proxy-token")

    @module.app.get("/_non-ascii-auth-probe")
    async def probe(_credential=Depends(module.verify_api_key)) -> dict[str, bool]:
        return {"accepted": True}

    status, _body = _invoke_app(
        module.app,
        "/_non-ascii-auth-probe",
        [(b"authorization", b"Bearer \xff")],
    )

    assert status == 401


def test_loopback_anthropic_no_key_needs_no_authorization_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_main(monkeypatch, tmp_path, proxy_api_key=None)
    request = module.Request({"type": "http", "headers": []})

    credential = asyncio.run(module.verify_anthropic_api_key(request))

    assert credential == ""


def test_loopback_anthropic_no_key_does_not_treat_bearer_none_as_a_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_main(monkeypatch, tmp_path, proxy_api_key=None)
    request = module.Request({"type": "http", "headers": [(b"authorization", b"Bearer None")]})

    credential = asyncio.run(module.verify_anthropic_api_key(request))

    assert credential == ""
