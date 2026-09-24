"""None-upstream-stream guard extracted from rotator_library.client."""

from __future__ import annotations

from typing import Any

from .error_handler import UpstreamStreamUnavailableError


class StreamedAPIError(Exception):
    """Custom exception to signal an API error received over a stream."""

    def __init__(self, message, data=None):
        super().__init__(message)
        self.data = data


def _safe_request_headers(request: Any) -> dict[str, str]:
    if request is None:
        return {}
    blocked = {
        "x-opencode-bounded-capability",
        "x-opencode-internal-bounded-capability",
        "x-opencode-internal-bounded-entry",
        "x-mirrowel-single-dispatch",
        "x-mirrowel-single-dispatch-token",
    }
    return {name: value for name, value in request.headers.items() if name.lower() not in blocked}


def require_async_stream_iterator(stream: Any) -> Any:
    """Return ``stream.__aiter__()``, or raise StreamedAPIError for None/non-iterable streams.

    Guard before any chunk is yielded so a None/non-iterable upstream
    stream becomes StreamedAPIError (retryable) instead of AttributeError.
    """
    if stream is None or not callable(getattr(stream, "__aiter__", None)):
        raise StreamedAPIError(
            "upstream_stream_unavailable",
            data=UpstreamStreamUnavailableError(),
        )
    return stream.__aiter__()
