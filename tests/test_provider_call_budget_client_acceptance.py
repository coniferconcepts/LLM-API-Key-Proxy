from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Callable

import httpx
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rotator_library.client as client_module  # noqa: E402
from rotator_library.client import RotatingClient  # noqa: E402
from rotator_library.provider_call_budget import (  # noqa: E402
    ProviderCallBudget,
    ProviderCallBudgetExhausted,
)

UPSTREAM_SENTINEL = "UPSTREAM-PAYLOAD-SENTINEL"


class DeterministicUsageManager:
    def __init__(self) -> None:
        self.acquired: list[str] = []

    async def get_credential_availability_stats(
        self, available_keys: list[str], *_args: Any
    ) -> dict[str, int]:
        return {
            "available": len(available_keys),
            "on_cooldown": 0,
            "fair_cycle_excluded": 0,
        }

    async def acquire_key(self, *, available_keys: list[str], **_kwargs: Any) -> str:
        credential = available_keys[0]
        self.acquired.append(credential)
        return credential

    async def record_failure(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def record_success(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def release_key(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class AvailableCooldownManager:
    async def is_cooling_down(self, _provider: str) -> bool:
        return False

    async def get_cooldown_remaining(self, _provider: str) -> float:
        return 0.0

    async def start_cooldown(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class PassthroughProviderConfig:
    def is_custom_provider(self, _provider: str) -> bool:
        return False

    def convert_for_litellm(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs


def configured_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[RotatingClient, DeterministicUsageManager]:
    client = RotatingClient(
        api_keys={"synthetic": ["credential-a", "credential-b"]},
        oauth_credentials={},
        max_retries=2,
        configure_logging=False,
        global_timeout=10,
        acquire_timeout=1,
        data_dir=tmp_path,
    )
    usage_manager = DeterministicUsageManager()
    client.usage_manager = usage_manager
    client.cooldown_manager = AvailableCooldownManager()
    client.provider_config = PassthroughProviderConfig()
    client._provider_plugins = {}
    client._provider_instances = {}
    monkeypatch.setattr(client_module.random, "shuffle", lambda _items: None)
    monkeypatch.setattr(client_module.asyncio, "sleep", immediate_sleep)
    return client, usage_manager


async def immediate_sleep(_delay: float) -> None:
    return None


def status_failure(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://upstream.invalid/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(UPSTREAM_SENTINEL, request=request, response=response)


def timeout_failure() -> httpx.ReadTimeout:
    request = httpx.Request("POST", "https://upstream.invalid/v1/chat/completions")
    return httpx.ReadTimeout(UPSTREAM_SENTINEL, request=request)


def malformed_failure() -> json.JSONDecodeError:
    return json.JSONDecodeError(UPSTREAM_SENTINEL, "not-json", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_factory",
    [
        pytest.param(lambda: status_failure(429), id="rate-limit-429"),
        pytest.param(lambda: status_failure(500), id="server-500"),
        pytest.param(timeout_failure, id="timeout"),
        pytest.param(malformed_failure, id="malformed"),
        pytest.param(lambda: status_failure(401), id="authentication-401"),
        pytest.param(lambda: status_failure(403), id="authorization-403"),
    ],
)
async def test_litellm_json_path_never_dispatches_a_second_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_factory: Callable[[], Exception],
) -> None:
    # Given: two credentials and a request-local ceiling of one provider call.
    client, usage_manager = configured_client(monkeypatch, tmp_path)
    dispatch_credentials: list[str] = []

    async def fail_upstream(**kwargs: Any) -> Any:
        dispatch_credentials.append(kwargs["api_key"])
        raise failure_factory()

    # When: the real JSON retry/rotation loop handles the admitted failure.
    with pytest.raises(Exception) as exc_info:
        await client._execute_with_retry(
            fail_upstream,
            request=None,
            model="synthetic/model",
            messages=[],
            _provider_call_budget=ProviderCallBudget(1),
        )

    # Then: no same-key retry or credential rotation reaches the provider seam.
    assert dispatch_credentials == ["credential-a"]
    assert "credential-b" not in dispatch_credentials
    assert usage_manager.acquired[0] == "credential-a"
    assert isinstance(exc_info.value, ProviderCallBudgetExhausted)
    assert exc_info.value.first_failure is not None
    assert UPSTREAM_SENTINEL not in str(exc_info.value)


@pytest.mark.asyncio
async def test_exhausted_budget_stops_before_acquiring_a_second_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: one retry per credential, two credentials, and one admitted provider call.
    client, usage_manager = configured_client(monkeypatch, tmp_path)
    client.max_retries = 1

    async def fail_upstream(**_kwargs: Any) -> Any:
        raise status_failure(500)

    # When: the first credential fails and the rotation loop considers credential B.
    with pytest.raises(ProviderCallBudgetExhausted):
        await client._execute_with_retry(
            fail_upstream,
            request=None,
            model="synthetic/model",
            messages=[],
            _provider_call_budget=ProviderCallBudget(1),
        )

    # Then: terminal budget state is observed before credential B is reserved.
    assert usage_manager.acquired == ["credential-a"]


@pytest.mark.asyncio
async def test_litellm_stream_path_propagates_typed_terminal_denial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: a streaming request whose first upstream connection fails before a byte.
    client, usage_manager = configured_client(monkeypatch, tmp_path)
    dispatch_credentials: list[str] = []

    async def fail_upstream(**kwargs: Any) -> Any:
        dispatch_credentials.append(kwargs["api_key"])
        raise status_failure(500)

    monkeypatch.setattr(client_module.litellm, "acompletion", fail_upstream)

    # When: the private retry loop reaches the typed proxy-boundary signal.
    with pytest.raises(ProviderCallBudgetExhausted) as exc_info:
        async for _chunk in client._streaming_acompletion_with_retry(
            None,
            model="synthetic/model",
            messages=[],
            stream=True,
            _provider_call_budget=ProviderCallBudget(1),
        ):
            pass

    # Then: one dispatch occurs and the public proxy wrapper receives the typed
    # first failure without leaking provider-controlled diagnostics in its message.
    assert dispatch_credentials == ["credential-a"]
    assert "credential-b" not in dispatch_credentials
    assert usage_manager.acquired == ["credential-a"]
    assert isinstance(exc_info.value.first_failure, httpx.HTTPStatusError)
    assert exc_info.value.first_failure.response.status_code == 500
    assert UPSTREAM_SENTINEL not in str(exc_info.value)
