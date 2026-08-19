from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit, urlunsplit

LOCAL_XAI_PORT = 2465


def _invalid_local_xai_base(reason: str) -> ValueError:
    return ValueError(f"Invalid local xAI base: {reason}")


def normalize_local_xai_base(configured: str) -> str:
    if (
        not configured
        or not configured.isascii()
        or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in configured)
    ):
        raise _invalid_local_xai_base("use an ASCII URL without controls or spaces")

    try:
        parsed = urlsplit(configured)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise _invalid_local_xai_base("malformed URL authority") from exc

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise _invalid_local_xai_base("scheme must be http or https")
    if parsed.username is not None or parsed.password is not None:
        raise _invalid_local_xai_base("userinfo is forbidden")
    if not hostname or "%" in hostname:
        raise _invalid_local_xai_base("host must be an unambiguous loopback address")
    if parsed.query or parsed.fragment:
        raise _invalid_local_xai_base("query strings and fragments are forbidden")
    if parsed.path not in {"/v1", "/v1/"}:
        raise _invalid_local_xai_base("path must be /v1")
    if port != LOCAL_XAI_PORT:
        raise _invalid_local_xai_base(f"port must be {LOCAL_XAI_PORT}")

    normalized_host = hostname.lower()
    if normalized_host == "localhost":
        canonical_input_host = "localhost"
        authority_host = "127.0.0.1"
    else:
        try:
            address = ip_address(normalized_host)
        except ValueError as exc:
            raise _invalid_local_xai_base("host must be localhost or a loopback IP") from exc
        if not address.is_loopback or getattr(address, "ipv4_mapped", None) is not None:
            raise _invalid_local_xai_base("host must be loopback")
        canonical_address = str(address)
        authority_host = f"[{canonical_address}]" if address.version == 6 else canonical_address
        canonical_input_host = authority_host

    canonical_input = urlunsplit(
        (scheme, f"{canonical_input_host}:{LOCAL_XAI_PORT}", "/v1", "", "")
    )
    if configured != canonical_input:
        raise _invalid_local_xai_base("URL must use its canonical representation")

    return urlunsplit((scheme, f"{authority_host}:{LOCAL_XAI_PORT}", "/v1", "", ""))
