from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from deployment_surface_helpers import load_main_symbols as _load_main_symbols


@pytest.mark.parametrize("host", ["[::1]", "[::1]:8000"])
def test_trusted_host_accepts_bracketed_ipv6_loopback(host: str) -> None:
    namespace = _load_main_symbols("TrustedHostMiddleware")
    middleware_type = namespace["TrustedHostMiddleware"]

    async def ok(_request):
        return PlainTextResponse("ok")

    test_app = Starlette(routes=[Route("/", ok)])
    guarded = middleware_type(test_app, allowed_hosts=["127.0.0.1", "localhost", "::1"])

    with TestClient(guarded) as client:
        response = client.get("/", headers={"host": host})

    assert response.status_code == 200
    assert response.text == "ok"


@pytest.mark.parametrize(
    "host",
    [
        "localhost:1",
        "localhost:00001",
        "localhost:00080",
        "localhost:65535",
        "[::1]:1",
        "[::1]:65535",
    ],
)
def test_trusted_host_accepts_valid_ascii_port_boundaries(host: str) -> None:
    namespace = _load_main_symbols("TrustedHostMiddleware")
    middleware_type = namespace["TrustedHostMiddleware"]

    async def ok(_request):
        return PlainTextResponse("ok")

    test_app = Starlette(routes=[Route("/", ok)])
    guarded = middleware_type(test_app, allowed_hosts=["localhost", "::1"])

    with TestClient(guarded) as client:
        response = client.get("/", headers={"host": host})

    assert response.status_code == 200
    assert response.text == "ok"


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1:abc",
        "127.0.0.1:80:evil",
        "localhost:",
        "localhost:0",
        "localhost:65536",
        "localhost:000001",
        "localhost:" + ("9" * 256),
        "localhost:+80",
        "user@localhost",
        "[::1",
        "[::1]extra",
        "[::1]:",
        "[::1]:0",
        "[::1]:65536",
        "[127.0.0.1]",
        "::1",
    ],
)
def test_trusted_host_rejects_malformed_or_partially_consumed_authority(
    host: str,
) -> None:
    namespace = _load_main_symbols("TrustedHostMiddleware")
    middleware_type = namespace["TrustedHostMiddleware"]

    async def ok(_request):
        return PlainTextResponse("ok")

    test_app = Starlette(routes=[Route("/", ok)])
    guarded = middleware_type(test_app, allowed_hosts=["127.0.0.1", "localhost", "::1"])

    with TestClient(guarded) as client:
        response = client.get("/", headers={"host": host})

    assert response.status_code == 400
    assert response.text == "Invalid host header"


def test_authority_parser_rejects_non_ascii_decimal_port() -> None:
    namespace = _load_main_symbols("_normalize_host_authority")
    normalize = namespace["_normalize_host_authority"]

    assert normalize("localhost:１２") == ""


def test_authority_parser_rejects_huge_port() -> None:
    namespace = _load_main_symbols("_normalize_host_authority")
    normalize = namespace["_normalize_host_authority"]

    assert normalize("localhost:" + ("9" * 4301)) == ""


@pytest.mark.parametrize(
    ("port", "expected"),
    [
        ("1", True),
        ("00001", True),
        ("00080", True),
        ("65535", True),
        ("", False),
        ("0", False),
        ("00000", False),
        ("65536", False),
        ("000001", False),
        ("+80", False),
        ("１２", False),
        ("9" * 4301, False),
    ],
)
def test_authority_port_grammar(port: str, expected: bool) -> None:
    namespace = _load_main_symbols("_is_valid_authority_port")
    validate = namespace["_is_valid_authority_port"]

    assert validate(port) is expected


def test_oversized_authority_port_is_rejected_before_integer_conversion() -> None:
    namespace = _load_main_symbols("_is_valid_authority_port")
    validate = namespace["_is_valid_authority_port"]
    conversions: list[str] = []

    def record_conversion(value: str) -> int:
        conversions.append(value)
        raise AssertionError("oversized port reached integer conversion")

    namespace["int"] = record_conversion

    assert validate("9" * 4301) is False
    assert conversions == []
