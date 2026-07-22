from __future__ import annotations

import pytest

from deployment_surface_helpers import load_main_symbols as _load_main_symbols


def test_credentialed_cors_rejects_wildcard_even_with_legacy_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a wildcard browser origin and the former wildcard approval switch.
    build = _load_main_symbols("_build_cors_allowed_origins")["_build_cors_allowed_origins"]
    monkeypatch.setenv("MIRROWEL_CORS_ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("MIRROWEL_ALLOW_WILDCARD_CORS_CREDENTIALS", "true")

    # When/Then: credentialed CORS refuses the wildcard unconditionally.
    with pytest.raises(SystemExit, match="must use explicit origins"):
        build()


@pytest.mark.parametrize(
    "configured_origin",
    [
        "null",
        "https://*.example.com",
        "file://localhost",
        "https://user@example.com",
        "https://example.com/path",
        "https://example.com?query=1",
        "https://example.com#fragment",
        "https://example.com:not-a-port",
        "https://example.com:+80",
        "https://example.com:70000",
        "https://example.com:",
        "https://[::1]:",
        "https://",
        "https://exa mple.com",
        "https://example.com\\evil",
        "https://example.com%00.evil",
        "http://[fe80::1%25eth0]",
        "https://[::ffff:192.0.2.128]x",
        " https://example.com",
        "https://example.com ",
        "\u00a0https://example.com",
        "https://example.com\u2003",
        "https://example.com\u200b",
        "https://example.com\n.evil",
    ],
)
def test_credentialed_cors_rejects_noncanonical_origins(
    monkeypatch: pytest.MonkeyPatch,
    configured_origin: str,
) -> None:
    # Given: an origin that is not an absolute canonical HTTP(S) origin.
    build = _load_main_symbols("_build_cors_allowed_origins")["_build_cors_allowed_origins"]
    monkeypatch.setenv("MIRROWEL_CORS_ALLOWED_ORIGINS", configured_origin)

    # When/Then: startup fails closed before the value reaches CORS middleware.
    with pytest.raises(SystemExit, match="absolute HTTP\(S\) origins"):
        build()


@pytest.mark.parametrize(
    "legacy_host",
    [
        "2130706433",
        "017700000001",
        "0x7f000001",
        "127.1",
        "0x7f.1",
        "0177.1",
        "1.2.3",
        "127.0.1",
        "0x7f.0x0.1",
        "0177.0.1",
        "127.000.000.001",
        "0x7f.0.0.1",
        "0x7f.0x0.0x0.0x1",
        "127.0x0.0.1",
        "0177.000.000.001",
    ],
)
def test_credentialed_cors_rejects_legacy_numeric_host_grammars(
    monkeypatch: pytest.MonkeyPatch,
    legacy_host: str,
) -> None:
    # Given: a hostname composed entirely of legacy decimal, octal-looking, or hex tokens.
    build = _load_main_symbols("_build_cors_allowed_origins")["_build_cors_allowed_origins"]
    monkeypatch.setenv("MIRROWEL_CORS_ALLOWED_ORIGINS", f"https://{legacy_host}")

    # When/Then: startup rejects resolver-dependent IPv4 grammar.
    with pytest.raises(SystemExit, match="absolute HTTP\(S\) origins"):
        build()


@pytest.mark.parametrize("dns_host", ["localhost", "api.internal", "123.example", "deadbeef"])
def test_credentialed_cors_preserves_alphabetic_dns_hostnames(
    monkeypatch: pytest.MonkeyPatch,
    dns_host: str,
) -> None:
    # Given: a valid DNS hostname with at least one nonnumeric label.
    build = _load_main_symbols("_build_cors_allowed_origins")["_build_cors_allowed_origins"]
    monkeypatch.setenv("MIRROWEL_CORS_ALLOWED_ORIGINS", f"https://{dns_host}")

    # When/Then: the strict origin boundary preserves the existing DNS contract.
    assert build() == [f"https://{dns_host}"]


def test_credentialed_cors_canonicalizes_and_deduplicates_valid_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: valid domain, localhost, IPv4, and IPv6 origins with equivalent spellings.
    build = _load_main_symbols("_build_cors_allowed_origins")["_build_cors_allowed_origins"]
    monkeypatch.setenv(
        "MIRROWEL_CORS_ALLOWED_ORIGINS",
        ",".join(
            (
                "HTTPS://EXAMPLE.COM:443/",
                "https://example.com",
                "http://localhost:8000/",
                "http://127.0.0.1:8080",
                "http://[0:0:0:0:0:0:0:1]:8000",
            )
        ),
    )

    # When: the startup boundary parses the configured list.
    origins = build()

    # Then: each semantic origin has one deterministic canonical spelling.
    assert origins == [
        "https://example.com",
        "http://localhost:8000",
        "http://127.0.0.1:8080",
        "http://[::1]:8000",
    ]
