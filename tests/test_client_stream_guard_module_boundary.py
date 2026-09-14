from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.client import StreamedAPIError, _safe_request_headers  # noqa: E402
from rotator_library.client_stream_guard import (  # noqa: E402
    StreamedAPIError as GuardStreamedAPIError,
    require_async_stream_iterator,
)
from rotator_library.error_handler import UpstreamStreamUnavailableError  # noqa: E402


def test_streamed_api_error_remains_importable_from_client() -> None:
    assert StreamedAPIError is GuardStreamedAPIError
    assert StreamedAPIError.__module__ == "rotator_library.client_stream_guard"


def test_safe_request_headers_remains_importable_from_client() -> None:
    assert _safe_request_headers.__module__ == "rotator_library.client_stream_guard"


def test_require_async_stream_iterator_rejects_none() -> None:
    with pytest.raises(StreamedAPIError) as exc_info:
        require_async_stream_iterator(None)
    inner = exc_info.value.data
    assert isinstance(inner, UpstreamStreamUnavailableError)
    assert inner.status_code == 502
    assert str(exc_info.value) == "upstream_stream_unavailable"


def test_require_async_stream_iterator_rejects_non_iterable() -> None:
    with pytest.raises(StreamedAPIError) as exc_info:
        require_async_stream_iterator(object())
    assert isinstance(exc_info.value.data, UpstreamStreamUnavailableError)
