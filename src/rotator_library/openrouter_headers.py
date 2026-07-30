from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol, TypeVar

OPENROUTER_ATTRIBUTION_HEADER_KEYS: Final = (
    "HTTP-Referer",
    "X-OpenRouter-Title",
    "X-Title",
    "X-OpenRouter-Categories",
)


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
    return request.headers.get(key) or request.headers.get(key.lower())


def _merge_openrouter_extra_headers(
    litellm_kwargs: _LiteLLMKwargsT,
    request: _RequestWithHeaders | None,
) -> _LiteLLMKwargsT:
    extra_headers = dict(litellm_kwargs.get("extra_headers") or {})
    for key in OPENROUTER_ATTRIBUTION_HEADER_KEYS:
        value = _get_request_header(request, key)
        if value and key not in extra_headers:
            extra_headers[key] = value
    if extra_headers:
        litellm_kwargs["extra_headers"] = extra_headers
    return litellm_kwargs
