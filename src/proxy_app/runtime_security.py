"""Immutable inbound network-security policy for the Mirrowel proxy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from ipaddress import ip_address
from unicodedata import category
from urllib.parse import urlsplit


def env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class RuntimeSecurityConfig:
    """Security values frozen after startup environment normalization."""

    proxy_api_key: str | None
    network_bind_approved: bool
    allowed_hosts: tuple[str, ...]
    allowed_hosts_configured: bool
    source: tuple[str | None, str | None, str | None, str | None]

    def is_current(self) -> bool:
        return self.source == runtime_security_source()


def runtime_security_source() -> tuple[str | None, str | None, str | None, str | None]:
    return (
        os.getenv("PROXY_API_KEY"),
        os.getenv("MIRROWEL_ALLOW_NETWORK_BIND"),
        os.getenv("MIRROWEL_ALLOWED_HOSTS"),
        os.getenv("MIRROWEL_ALLOW_WILDCARD_HOSTS"),
    )


def is_loopback_bind_host(host: str) -> bool:
    normalized_host = host.strip().lower()
    if normalized_host == "localhost":
        return True
    try:
        return ip_address(normalized_host).is_loopback
    except ValueError:
        return False


def network_bind_configuration_error(
    config: RuntimeSecurityConfig | None = None,
) -> str | None:
    runtime_config = config or build_runtime_security_config()
    if not runtime_config.network_bind_approved:
        return "requires MIRROWEL_ALLOW_NETWORK_BIND=true"
    if not runtime_config.allowed_hosts_configured:
        return "requires MIRROWEL_ALLOWED_HOSTS with explicit hostnames or IP addresses"
    if not runtime_config.proxy_api_key or not runtime_config.proxy_api_key.strip():
        return "requires a nonempty PROXY_API_KEY for inbound authentication"
    return None


def validate_bind_host(host: str, config: RuntimeSecurityConfig | None = None) -> str:
    normalized_host = host.strip()
    if not normalized_host:
        raise SystemExit("Host must not be empty")
    if is_loopback_bind_host(normalized_host):
        return normalized_host
    configuration_error = network_bind_configuration_error(config)
    if configuration_error:
        raise SystemExit(f"Non-loopback Mirrowel bind {configuration_error}")
    return normalized_host


def build_cors_allowed_origins() -> list[str]:
    configured_origins = os.getenv("MIRROWEL_CORS_ALLOWED_ORIGINS", "")
    if not configured_origins:
        return [
            "http://127.0.0.1",
            "http://127.0.0.1:8000",
            "http://localhost",
            "http://localhost:8000",
        ]

    origins = configured_origins.split(",")
    if any(not origin or origin != origin.strip() for origin in origins):
        raise SystemExit("MIRROWEL_CORS_ALLOWED_ORIGINS entries must be absolute HTTP(S) origins")
    if "*" in origins:
        raise SystemExit("Credentialed CORS must use explicit origins; wildcard '*' is forbidden")
    return list(dict.fromkeys(_canonical_cors_origin(origin) for origin in origins))


def _canonical_cors_origin(origin: str) -> str:
    error = "MIRROWEL_CORS_ALLOWED_ORIGINS entries must be absolute HTTP(S) origins"
    if any(character.isspace() or category(character).startswith("C") for character in origin):
        raise SystemExit(error)
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError:
        raise SystemExit(error) from None
    hostname = parsed.hostname
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or (port is None and parsed.netloc.endswith(":"))
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or not _valid_cors_hostname(hostname)
        or not _has_canonical_authority_syntax(parsed.netloc, hostname, port)
    ):
        raise SystemExit(error)

    try:
        normalized_host = str(ip_address(hostname))
    except ValueError:
        normalized_host = hostname.lower()
    authority = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    scheme = parsed.scheme.lower()
    if port is not None and port != {"http": 80, "https": 443}[scheme]:
        authority = f"{authority}:{port}"
    return f"{scheme}://{authority}"


def _valid_cors_hostname(hostname: str) -> bool:
    if "*" in hostname or "%" in hostname:
        return False
    try:
        ip_address(hostname)
        return True
    except ValueError:
        return not _looks_like_noncanonical_ipv4(hostname) and _valid_allowed_host_pattern(
            hostname.lower()
        )


def _looks_like_noncanonical_ipv4(hostname: str) -> bool:
    return all(_is_legacy_numeric_host_token(label) for label in hostname.lower().split("."))


def _is_legacy_numeric_host_token(label: str) -> bool:
    return label.isdigit() or (
        label.startswith("0x")
        and len(label) > 2
        and all(character in "0123456789abcdef" for character in label[2:])
    )


def _has_canonical_authority_syntax(netloc: str, hostname: str, port: int | None) -> bool:
    raw_host = f"[{hostname}]" if ":" in hostname else hostname
    expected_authority = raw_host if port is None else f"{raw_host}:{port}"
    return netloc.lower() == expected_authority.lower()


def _valid_allowed_host_pattern(pattern: str) -> bool:
    if pattern == "*":
        return True
    candidate = pattern[2:] if pattern.startswith("*.") else pattern
    try:
        ip_address(candidate)
        return not pattern.startswith("*.")
    except ValueError:
        labels = candidate.split(".")
        return (
            candidate.isascii()
            and len(candidate) <= 253
            and all(
                label
                and len(label) <= 63
                and label[0] != "-"
                and label[-1] != "-"
                and all(character.isalnum() or character == "-" for character in label)
                for label in labels
            )
        )


def build_allowed_hosts(
    configured_hosts: str | None = None,
    wildcard_hosts_approved: bool | None = None,
) -> list[str]:
    raw_configured_hosts = (
        os.getenv("MIRROWEL_ALLOWED_HOSTS", "") if configured_hosts is None else configured_hosts
    )
    normalized_configured_hosts = raw_configured_hosts.strip()
    if not normalized_configured_hosts:
        return ["127.0.0.1", "localhost", "::1", "[::1]"]

    hosts = [
        host.strip().lower() for host in normalized_configured_hosts.split(",") if host.strip()
    ]
    if not hosts:
        raise SystemExit("MIRROWEL_ALLOWED_HOSTS must name at least one hostname or IP address")
    if not all(_valid_allowed_host_pattern(pattern) for pattern in hosts):
        raise SystemExit(
            "MIRROWEL_ALLOWED_HOSTS entries must be hostnames or IP addresses without schemes or ports"
        )
    wildcard_approved = (
        env_flag_enabled("MIRROWEL_ALLOW_WILDCARD_HOSTS")
        if wildcard_hosts_approved is None
        else wildcard_hosts_approved
    )
    if any(host.startswith("*") for host in hosts) and not wildcard_approved:
        raise SystemExit("Wildcard Host acceptance requires MIRROWEL_ALLOW_WILDCARD_HOSTS=true")
    return hosts


def build_runtime_security_config() -> RuntimeSecurityConfig:
    source = runtime_security_source()
    proxy_api_key, network_bind_value, allowed_hosts_value, wildcard_hosts_value = source
    return RuntimeSecurityConfig(
        proxy_api_key=proxy_api_key,
        network_bind_approved=(network_bind_value or "").strip().lower()
        in {"1", "true", "yes", "on"},
        allowed_hosts=tuple(
            build_allowed_hosts(
                allowed_hosts_value,
                (wildcard_hosts_value or "").strip().lower() in {"1", "true", "yes", "on"},
            )
        ),
        allowed_hosts_configured=bool((allowed_hosts_value or "").strip()),
        source=source,
    )
