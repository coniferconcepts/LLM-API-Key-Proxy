"""ASGI Host and bind-approval enforcement for Mirrowel."""

from __future__ import annotations

from collections.abc import Callable
from ipaddress import ip_address

from starlette.datastructures import URL
from starlette.middleware.trustedhost import TrustedHostMiddleware as StarletteTrustedHostMiddleware
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from proxy_app.runtime_security import (
    RuntimeSecurityConfig,
    build_runtime_security_config,
    is_loopback_bind_host,
    network_bind_configuration_error,
)


def normalize_host_authority(authority: str) -> str:
    normalized = authority.strip().lower()
    if not normalized or "@" in normalized:
        return ""
    if normalized.startswith("["):
        closing_bracket = normalized.find("]")
        if closing_bracket < 2:
            return ""
        address = normalized[1:closing_bracket]
        try:
            parsed_address = ip_address(address)
        except ValueError:
            return ""
        if parsed_address.version != 6:
            return ""
        suffix = normalized[closing_bracket + 1 :]
        if suffix and (not suffix.startswith(":") or not is_valid_authority_port(suffix[1:])):
            return ""
        return parsed_address.compressed

    if any(character.isspace() or character in "[]/?#\\" for character in normalized):
        return ""
    separator_count = normalized.count(":")
    if separator_count == 0:
        host = normalized
    elif separator_count == 1:
        host, port = normalized.split(":", 1)
        if not is_valid_authority_port(port):
            return ""
    else:
        return ""
    return host if is_valid_host_name(host) else ""


def is_valid_host_name(host: str) -> bool:
    if not host or not host.isascii() or len(host) > 253:
        return False
    try:
        return ip_address(host).version == 4
    except ValueError:
        pass
    return all(
        label
        and len(label) <= 63
        and label[0] != "-"
        and label[-1] != "-"
        and all(character.isalnum() or character == "-" for character in label)
        for label in host.split(".")
    )


def is_valid_authority_port(port: str) -> bool:
    return port.isascii() and port.isdecimal() and 1 <= len(port) <= 5 and 1 <= int(port) <= 65535


async def send_scope_rejection(
    scope: Scope,
    receive: Receive,
    send: Send,
    message: str,
    status_code: int,
) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})
        return
    response = PlainTextResponse(message, status_code=status_code)
    await response(scope, receive, send)


class TrustedHostMiddleware(StarletteTrustedHostMiddleware):
    """Validate Host headers without treating IPv6 address colons as port separators."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        host_headers = [
            value for name, value in scope.get("headers", []) if name.lower() == b"host"
        ]
        host = (
            normalize_host_authority(host_headers[0].decode("latin-1"))
            if len(host_headers) == 1 and host_headers[0].strip()
            else ""
        )
        if not host:
            await send_scope_rejection(scope, receive, send, "Invalid host header", 400)
            return
        if self.allow_any:
            await self.app(scope, receive, send)
            return

        valid_host = False
        found_www_redirect = False
        for pattern in self.allowed_hosts:
            wildcard_match = _matches_wildcard_host(host, pattern)
            if host == pattern or wildcard_match:
                valid_host = True
                break
            if "www." + host == pattern:
                found_www_redirect = True

        if valid_host:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send_scope_rejection(scope, receive, send, "Invalid host header", 400)
            return
        if found_www_redirect and self.www_redirect:
            url = URL(scope=scope)
            response: Response = RedirectResponse(url=str(url.replace(netloc="www." + url.netloc)))
        else:
            response = PlainTextResponse("Invalid host header", status_code=400)
        await response(scope, receive, send)


def _matches_wildcard_host(host: str, pattern: str) -> bool:
    if pattern == "*":
        return True
    if not pattern.startswith("*.") or not host.endswith(pattern[1:]):
        return False
    subdomain = host[: -len(pattern[1:])]
    return all(
        label
        and len(label) <= 63
        and label[0] != "-"
        and label[-1] != "-"
        and all(character.isalnum() or character == "-" for character in label)
        for label in subdomain.split(".")
    )


class BindApprovalMiddleware:
    """Keep ASGI launches inside the same bind-approval boundary as the CLI."""

    def __init__(
        self,
        app: ASGIApp,
        config: RuntimeSecurityConfig | None = None,
        runtime_config_getter: Callable[[], RuntimeSecurityConfig] = build_runtime_security_config,
    ) -> None:
        self.app = app
        self.config = config
        self.runtime_config_getter = runtime_config_getter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        runtime_config = self.config or self.runtime_config_getter()
        if not runtime_config.is_current():
            await send_scope_rejection(
                scope,
                receive,
                send,
                "Runtime security configuration changed after initialization",
                503,
            )
            return

        server = scope.get("server")
        server_host = server[0] if server else ""
        if is_loopback_bind_host(server_host):
            await self.app(scope, receive, send)
            return

        configuration_error = network_bind_configuration_error(runtime_config)
        if not configuration_error:
            await self.app(scope, receive, send)
            return

        await send_scope_rejection(
            scope,
            receive,
            send,
            f"Non-loopback ASGI server bind {configuration_error}",
            503,
        )
