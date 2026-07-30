from __future__ import annotations

import json
import logging
from typing import Any, Literal

import pytest

from error_boundary_regression_support import SECRET_CREDENTIAL, SECRET_MESSAGE
from rotator_library.error_handler import (
    ClassifiedError,
    NoAvailableKeysError,
    RequestErrorAccumulator,
)


def test_rotation_error_response_and_log_keep_only_categories_and_counts() -> None:
    accumulator = RequestErrorAccumulator()
    accumulator.provider = "synthetic"
    accumulator.model = "synthetic/model"
    accumulator.record_error(
        SECRET_CREDENTIAL,
        ClassifiedError("authentication", RuntimeError(SECRET_MESSAGE), status_code=401),
        SECRET_MESSAGE,
    )
    accumulator.record_error(
        f"second-{SECRET_CREDENTIAL}",
        ClassifiedError(SECRET_MESSAGE, RuntimeError(SECRET_MESSAGE), status_code=599),
        SECRET_MESSAGE,
    )
    response = accumulator.build_client_error_response()
    log_message = accumulator.build_log_message()
    serialized = json.dumps(response)
    assert SECRET_MESSAGE not in serialized
    assert SECRET_CREDENTIAL not in serialized
    assert "STABLE-OAUTH-FILENAME-SENTINEL" not in serialized
    assert SECRET_MESSAGE not in log_message
    assert "STABLE-OAUTH-FILENAME-SENTINEL" not in log_message
    assert response["error"]["details"]["credentials_tried"] == 2
    assert response["error"]["details"]["failure_categories"] == {"authentication": 1, "unknown": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_kind", "category", "expected_type"),
    (
        ("acquisition", "proxy_busy", "proxy_busy"),
        ("acquisition", "proxy_all_credentials_exhausted", "proxy_all_credentials_exhausted"),
        ("unexpected", None, "proxy_internal_error"),
    ),
)
async def test_terminal_sse_omits_exception_diagnostics_and_credential_identity(
    failure_kind: str,
    category: Literal["proxy_busy", "proxy_all_credentials_exhausted"] | None,
    expected_type: str,
    caplog,
    monkeypatch,
) -> None:
    import rotator_library.client as client_module

    class AvailableCooldown:
        async def is_cooling_down(self, _provider: str) -> bool:
            return False

    class FailingUsageManager:
        async def get_credential_availability_stats(self, *_args: Any) -> dict[str, int]:
            return {"available": 1, "on_cooldown": 0, "fair_cycle_excluded": 0}

        async def acquire_key(self, **_kwargs: Any) -> str:
            if failure_kind == "acquisition":
                assert category is not None
                raise NoAvailableKeysError(
                    SECRET_MESSAGE,
                    code=SECRET_MESSAGE,
                    diagnostics={"provider_payload": SECRET_MESSAGE},
                    category=category,
                )
            raise RuntimeError(SECRET_MESSAGE)

    rotating_client = object.__new__(client_module.RotatingClient)
    rotating_client.all_credentials = {"synthetic": [SECRET_CREDENTIAL]}
    rotating_client.global_timeout = 10.0
    rotating_client.acquire_timeout = 1.0
    rotating_client.enable_request_logging = False
    rotating_client.max_concurrent_requests_per_key = {"synthetic": 1}
    rotating_client.cooldown_manager = AvailableCooldown()
    rotating_client.usage_manager = FailingUsageManager()
    rotating_client._apply_routing_policy = lambda model: (model, None)
    rotating_client._get_provider_instance = lambda _provider: None
    rotating_client._resolve_model_id = lambda model, _provider: model
    monkeypatch.setattr(client_module.lib_logger, "propagate", True)
    with caplog.at_level(logging.ERROR, logger="rotator_library"):
        chunks = [
            chunk
            async for chunk in rotating_client._streaming_acompletion_with_retry(
                None, model="synthetic/model", messages=[]
            )
        ]
    body = "".join(chunks)
    assert SECRET_MESSAGE not in body
    assert SECRET_CREDENTIAL not in body
    assert SECRET_MESSAGE not in caplog.text
    assert SECRET_CREDENTIAL not in caplog.text
    expected_status = 503 if failure_kind == "acquisition" else 500
    assert f'"status": {expected_status}' in body
    assert f'"type": "{expected_type}"' in body
    assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_low_level_stream_buffering_logs_omit_payload_and_credential_identity(
    caplog, monkeypatch
) -> None:
    import rotator_library.client as client_module

    class UsageManager:
        async def release_key(self, _key: str, _model: str) -> None:
            return

    class FailingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError(f'Received chunk: {{"error":"{SECRET_MESSAGE}"}}')

    rotating_client = object.__new__(client_module.RotatingClient)
    rotating_client.usage_manager = UsageManager()
    monkeypatch.setattr(client_module.lib_logger, "propagate", True)
    with caplog.at_level(logging.INFO, logger="rotator_library"):
        with pytest.raises(client_module.StreamedAPIError):
            _ = [
                chunk
                async for chunk in rotating_client._safe_streaming_wrapper(
                    FailingStream(), SECRET_CREDENTIAL, "synthetic/model"
                )
            ]
    assert SECRET_MESSAGE not in caplog.text
    assert SECRET_CREDENTIAL not in caplog.text
    assert "STABLE-OAUTH-FILENAME-SENTINEL" not in caplog.text
