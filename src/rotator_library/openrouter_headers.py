from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol, TypeVar
import uuid

OPENROUTER_ATTRIBUTION_HEADER_KEYS: Final = (
    "HTTP-Referer",
    "X-OpenRouter-Title",
    "X-Title",
    "X-OpenRouter-Categories",
)

_SESSION_HEADER_KEYS: Final = (
    "x-opencode-session",
    "x-session-id",
    "x-conversation-id",
    "x-grok-conv-id",
    "x-request-id",
)
_GO_PROVIDERS: Final = frozenset({"opencode_go", "opencode_go_messages"})
_XAI_PROVIDERS: Final = frozenset({"xai", "xai_oauth"})
_FIREWORKS_PROVIDERS: Final = frozenset({"fireworks"})
_OPENROUTER_PROVIDERS: Final = frozenset({"openrouter", "openrouter_zdr", "openrouter_non_zdr"})


class _RequestWithHeaders(Protocol):
    @property
    def headers(self) -> Mapping[str, str]: ...


class _LiteLLMKwargs(Protocol):
    def get(self, key: str) -> Mapping[str, str] | None: ...

    def __setitem__(self, key: str, value: dict[str, str]) -> None: ...


_LiteLLMKwargsT = TypeVar("_LiteLLMKwargsT", bound=_LiteLLMKwargs)


def _get_request_header(request: _RequestWithHeaders | None, key: str) -> str | None:
    if request is None:
        return None
    for header_name, value in request.headers.items():
        if header_name.casefold() != key.casefold() or any(char in value for char in "\r\n\x00"):
            continue
        candidate = value.strip()
        if candidate:
            return candidate
    return None


def _resolve_session(request: _RequestWithHeaders | None) -> str:
    """Resolve boundary aliases before the validated request ID, then its raw alias."""

    for key in _SESSION_HEADER_KEYS[:-1]:
        value = _get_request_header(request, key)
        if value:
            return value
    request_id = getattr(request, "request_id", None)
    if isinstance(request_id, str) and not any(char in request_id for char in "\r\n\x00"):
        validated_request_id = request_id.strip()
        if validated_request_id:
            return validated_request_id
    raw_request_id = _get_request_header(request, _SESSION_HEADER_KEYS[-1])
    if raw_request_id:
        return raw_request_id
    return str(uuid.uuid4())


def merge_provider_extra_headers(
    litellm_kwargs: _LiteLLMKwargsT,
    request: _RequestWithHeaders | None,
    provider: str,
) -> _LiteLLMKwargsT:
    extra_headers = dict(litellm_kwargs.get("extra_headers") or {})
    session = _resolve_session(request)
    mapped_headers: dict[str, str] = {}
    provider_name = provider.casefold()
    if provider_name in _GO_PROVIDERS:
        mapped_headers = {
            "x-opencode-session": session,
            "x-opencode-client": "opencode-router",
            "User-Agent": "opencode-router-mirrowel/1",
        }
    elif provider_name in _XAI_PROVIDERS:
        mapped_headers = {"x-grok-conv-id": session}
    elif provider_name in _FIREWORKS_PROVIDERS:
        mapped_headers = {"x-session-affinity": session}
    elif provider_name in _OPENROUTER_PROVIDERS:
        mapped_headers = {"x-session-id": session}
        for key in OPENROUTER_ATTRIBUTION_HEADER_KEYS:
            value = _get_request_header(request, key)
            if value:
                mapped_headers[key] = value

    for key, value in mapped_headers.items():
        if all(name.casefold() != key.casefold() for name in extra_headers):
            extra_headers[key] = value
    if extra_headers:
        litellm_kwargs["extra_headers"] = extra_headers
    return litellm_kwargs


def _merge_openrouter_extra_headers(
    litellm_kwargs: _LiteLLMKwargsT,
    request: _RequestWithHeaders | None,
) -> _LiteLLMKwargsT:
    return merge_provider_extra_headers(litellm_kwargs, request, "openrouter")
