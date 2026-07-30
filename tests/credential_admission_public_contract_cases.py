from __future__ import annotations

from pathlib import Path

import pytest

from credential_admission_contract_support import _CREDENTIAL, _MODEL, make_client
from rotator_library.error_handler import (
    ClassifiedError,
    NoAvailableKeysError,
    RequestErrorAccumulator,
    build_public_stream_error,
    classify_credential_failure,
)


def test_proxy_busy_public_contract_is_finite_and_sanitized() -> None:
    internal_detail = "credential synthetic-secret-token is busy"
    body = build_public_stream_error("proxy_busy")
    assert body == {
        "error": {
            "type": "proxy_busy",
            "code": "acquisition_timeout",
            "status": 503,
            "message": "No upstream credential is currently available. Retry later.",
        }
    }
    assert internal_detail not in str(body)
    assert "synthetic-secret-token" not in str(body)


def test_exhausted_http_and_sse_contracts_are_distinct_and_sanitized() -> None:
    accumulator = RequestErrorAccumulator()
    accumulator.record_error(
        _CREDENTIAL,
        ClassifiedError(
            error_type="quota_exceeded",
            original_exception=RuntimeError("synthetic-secret-token"),
            status_code=429,
        ),
        "credential synthetic-secret-token exhausted",
    )
    http_body = accumulator.build_client_error_response()
    sse_body = build_public_stream_error("proxy_all_credentials_exhausted")
    expected_public = {
        "message": "No upstream credential is currently available. Retry later.",
        "type": "proxy_all_credentials_exhausted",
        "code": "all_credentials_exhausted",
        "status": 503,
    }
    assert {key: http_body["error"][key] for key in expected_public} == expected_public
    assert sse_body == {"error": expected_public}
    assert "synthetic-secret-token" not in str(http_body)
    assert "synthetic-secret-token" not in str(sse_body)
    assert _CREDENTIAL not in str(http_body)


def test_credential_metric_and_monitor_classifications_remain_distinct() -> None:
    busy = classify_credential_failure("proxy_busy")
    exhausted = classify_credential_failure("proxy_all_credentials_exhausted")
    classifications = {
        (busy.metric_code, busy.monitor_code),
        (exhausted.metric_code, exhausted.monitor_code),
    }
    assert classifications == {
        ("proxy_busy", "acquisition_timeout"),
        ("proxy_all_credentials_exhausted", "all_credentials_exhausted"),
    }
    assert busy.retryable is True
    assert exhausted.retryable is False


@pytest.mark.asyncio
async def test_exhausted_dispatch_does_not_retry_the_failed_credential(tmp_path: Path) -> None:
    client, _manager = make_client(tmp_path, acquire_timeout=1.0)
    provider_calls = 0

    async def failing_api(**_kwargs: object) -> object:
        nonlocal provider_calls
        provider_calls += 1
        raise ConnectionError("synthetic upstream disconnect")

    assert client.max_retries == 1
    response = await client._execute_with_retry(
        failing_api,
        request=None,
        model=_MODEL,
        messages=[{"role": "user", "content": "fail"}],
    )
    assert response["error"]["code"] == "all_credentials_exhausted"
    assert provider_calls == 1


def test_untrusted_stream_category_fails_closed_without_reflection() -> None:
    category = "proxy_busy synthetic-secret-token\nretry"
    body = build_public_stream_error(category)
    assert body["error"]["type"] == "proxy_internal_error"
    assert body["error"]["code"] == "stream_error"
    assert "synthetic-secret-token" not in str(body)


def test_untrusted_exception_category_fails_closed_before_logging_or_lookup() -> None:
    error = NoAvailableKeysError(
        "synthetic-secret-token",
        category="proxy_busy synthetic-secret-token",
    )
    body = build_public_stream_error(error.category)
    assert error.category == "proxy_busy"
    assert body == build_public_stream_error("proxy_busy")
    assert "synthetic-secret-token" not in str(body)
