from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from deployment_surface_helpers import load_main_symbols as _load_main_symbols


def test_credentialed_cors_rejects_wildcard_even_with_legacy_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a wildcard browser origin and the former wildcard approval switch.
    namespace = _load_main_symbols("_build_cors_allowed_origins")
    build = namespace["_build_cors_allowed_origins"]
    monkeypatch.setenv("MIRROWEL_CORS_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("MIRROWEL_ALLOW_WILDCARD_CORS_CREDENTIALS", "true")

    # When/Then: credentialed CORS refuses the wildcard unconditionally.
    with pytest.raises(SystemExit, match="must use explicit origins"):
        build()


def test_direct_asgi_public_server_scope_requires_network_bind_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load_main_symbols("BindApprovalMiddleware")
    middleware_type = namespace["BindApprovalMiddleware"]

    async def ok(_request):
        return PlainTextResponse("ok")

    monkeypatch.delenv("MIRROWEL_ALLOW_NETWORK_BIND", raising=False)
    test_app = Starlette(routes=[Route("/", ok)])
    guarded = middleware_type(test_app, config=namespace["_build_runtime_security_config"]())

    with TestClient(guarded, base_url="http://0.0.0.0") as client:
        response = client.get("/", headers={"host": "127.0.0.1"})

    assert response.status_code == 503
    assert (
        response.text == "Non-loopback ASGI server bind requires MIRROWEL_ALLOW_NETWORK_BIND=true"
    )


def test_cli_public_bind_rejects_approval_without_allowed_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load_main_symbols("_validate_bind_host")
    validate = namespace["_validate_bind_host"]
    monkeypatch.setenv("MIRROWEL_ALLOW_NETWORK_BIND", "true")
    monkeypatch.setenv("PROXY_API_KEY", "test-token")
    monkeypatch.delenv("MIRROWEL_ALLOWED_HOSTS", raising=False)

    with pytest.raises(SystemExit, match="MIRROWEL_ALLOWED_HOSTS"):
        validate("0.0.0.0")


@pytest.mark.parametrize(
    "configured_hosts",
    [
        "proxy.example.com:443",
        "https://proxy.example.com",
        "user@proxy.example.com",
        "proxy.example.com/path",
        "[::1]",
    ],
)
def test_allowed_host_configuration_rejects_request_authority_syntax(
    monkeypatch: pytest.MonkeyPatch,
    configured_hosts: str,
) -> None:
    namespace = _load_main_symbols("_build_allowed_hosts")
    build = namespace["_build_allowed_hosts"]
    monkeypatch.setenv("MIRROWEL_ALLOWED_HOSTS", configured_hosts)

    with pytest.raises(SystemExit, match="without schemes or ports"):
        build()


def test_wildcard_allowed_host_requires_separate_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load_main_symbols("_build_allowed_hosts")
    build = namespace["_build_allowed_hosts"]
    monkeypatch.setenv("MIRROWEL_ALLOWED_HOSTS", "*.example.com")
    monkeypatch.delenv("MIRROWEL_ALLOW_WILDCARD_HOSTS", raising=False)

    with pytest.raises(SystemExit, match="Wildcard Host acceptance"):
        build()


def test_approved_wildcard_matches_only_subdomains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load_main_symbols("_build_allowed_hosts", "TrustedHostMiddleware")
    build = namespace["_build_allowed_hosts"]
    middleware_type = namespace["TrustedHostMiddleware"]
    monkeypatch.setenv("MIRROWEL_ALLOWED_HOSTS", "*.example.com")
    monkeypatch.setenv("MIRROWEL_ALLOW_WILDCARD_HOSTS", "true")

    async def ok(_request):
        return PlainTextResponse("ok")

    test_app = Starlette(routes=[Route("/", ok)])
    guarded = middleware_type(test_app, allowed_hosts=build())

    with TestClient(guarded) as client:
        accepted = client.get("/", headers={"host": "api.example.com"})
        deceptive_suffix = client.get("/", headers={"host": "badexample.com"})
        apex = client.get("/", headers={"host": "example.com"})
        empty_label = client.get("/", headers={"host": ".example.com"})
        repeated_separator = client.get("/", headers={"host": "a..example.com"})
        invalid_label = client.get("/", headers={"host": "-a.example.com"})

    assert accepted.status_code == 200
    assert deceptive_suffix.status_code == 400
    assert apex.status_code == 400
    assert empty_label.status_code == 400
    assert repeated_separator.status_code == 400
    assert invalid_label.status_code == 400


def test_direct_asgi_public_server_scope_rejects_approval_without_allowed_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load_main_symbols("BindApprovalMiddleware")
    middleware_type = namespace["BindApprovalMiddleware"]

    async def ok(_request):
        return PlainTextResponse("ok")

    monkeypatch.setenv("MIRROWEL_ALLOW_NETWORK_BIND", "true")
    monkeypatch.setenv("PROXY_API_KEY", "test-token")
    monkeypatch.delenv("MIRROWEL_ALLOWED_HOSTS", raising=False)
    test_app = Starlette(routes=[Route("/", ok)])
    guarded = middleware_type(test_app, config=namespace["_build_runtime_security_config"]())

    with TestClient(guarded, base_url="http://0.0.0.0") as client:
        response = client.get("/", headers={"host": "127.0.0.1"})

    assert response.status_code == 503
    assert "MIRROWEL_ALLOWED_HOSTS" in response.text


@pytest.mark.parametrize("configured_hosts", ["proxy.example.com", "*.example.com"])
def test_public_bind_rejects_allowed_hosts_without_inbound_key(
    monkeypatch: pytest.MonkeyPatch,
    configured_hosts: str,
) -> None:
    namespace = _load_main_symbols("_validate_bind_host")
    validate = namespace["_validate_bind_host"]
    monkeypatch.setenv("MIRROWEL_ALLOW_NETWORK_BIND", "true")
    monkeypatch.setenv("MIRROWEL_ALLOWED_HOSTS", configured_hosts)
    if configured_hosts.startswith("*"):
        monkeypatch.setenv("MIRROWEL_ALLOW_WILDCARD_HOSTS", "true")
    monkeypatch.setenv("PROXY_API_KEY", "   ")

    with pytest.raises(SystemExit, match="nonempty PROXY_API_KEY"):
        validate("0.0.0.0")


def test_public_asgi_scope_rejects_allowed_hosts_without_inbound_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load_main_symbols("BindApprovalMiddleware")
    middleware_type = namespace["BindApprovalMiddleware"]

    async def ok(_request):
        return PlainTextResponse("ok")

    monkeypatch.setenv("MIRROWEL_ALLOW_NETWORK_BIND", "true")
    monkeypatch.setenv("MIRROWEL_ALLOWED_HOSTS", "proxy.example.com")
    monkeypatch.delenv("PROXY_API_KEY", raising=False)
    test_app = Starlette(routes=[Route("/", ok)])
    guarded = middleware_type(test_app, config=namespace["_build_runtime_security_config"]())

    with TestClient(guarded, base_url="http://0.0.0.0") as client:
        response = client.get("/", headers={"host": "proxy.example.com"})

    assert response.status_code == 503
    assert "nonempty PROXY_API_KEY" in response.text


def test_public_asgi_stack_accepts_only_explicit_allowed_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load_main_symbols("BindApprovalMiddleware", "TrustedHostMiddleware")
    bind_middleware_type = namespace["BindApprovalMiddleware"]
    host_middleware_type = namespace["TrustedHostMiddleware"]

    async def ok(_request):
        return PlainTextResponse("ok")

    monkeypatch.setenv("MIRROWEL_ALLOW_NETWORK_BIND", "true")
    monkeypatch.setenv("MIRROWEL_ALLOWED_HOSTS", "proxy.example.com")
    monkeypatch.setenv("PROXY_API_KEY", "test-token")
    test_app = Starlette(routes=[Route("/", ok)])
    host_guarded = host_middleware_type(test_app, allowed_hosts=["proxy.example.com"])
    guarded = bind_middleware_type(
        host_guarded,
        config=namespace["_build_runtime_security_config"](),
    )

    with TestClient(guarded, base_url="http://0.0.0.0") as client:
        accepted = client.get("/", headers={"host": "proxy.example.com"})
        rejected = client.get("/", headers={"host": "attacker.example"})

    assert accepted.status_code == 200
    assert accepted.text == "ok"
    assert rejected.status_code == 400
    assert rejected.text == "Invalid host header"
