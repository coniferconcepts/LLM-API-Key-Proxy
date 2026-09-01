from __future__ import annotations

import asyncio
from pathlib import Path

from rotator_library.client import RotatingClient
from rotator_library.usage_manager import UsageManager

_CREDENTIAL = "synthetic-credential"
_MODEL = "synthetic/model"


class _AvailableCooldownManager:
    async def is_cooling_down(self, _provider: str) -> bool:
        return False


class _IdentityProviderConfig:
    def convert_for_litellm(self, **kwargs: object) -> dict[str, object]:
        return dict(kwargs)


class _PriorityProvider:
    def get_credential_priority(self, _credential: str) -> int:
        return 1

    def get_credential_tier_name(self, _credential: str) -> str:
        return "synthetic"

    def has_custom_logic(self) -> bool:
        return False


def active_count(manager: UsageManager) -> int:
    state = manager.key_states.get(_CREDENTIAL)
    return 0 if state is None else int(state["models_in_use"].get(_MODEL, 0))


def make_usage_manager(tmp_path: Path, *, name: str = "usage.json") -> UsageManager:
    # Admission tests are not about the 03:00 UTC daily-reset save path.
    manager = UsageManager(str(tmp_path / name), daily_reset_time_utc=None)
    manager._usage_data = {}  # noqa: SLF001
    manager._initialized.set()  # noqa: SLF001
    return manager


def make_client(tmp_path: Path, *, acquire_timeout: float) -> tuple[RotatingClient, UsageManager]:
    manager = make_usage_manager(tmp_path)

    async def record_success(
        _credential: str,
        _model: str,
        _usage: dict[str, object] | None = None,
    ) -> None:
        return None

    manager.record_success = record_success  # type: ignore[method-assign]
    client = object.__new__(RotatingClient)
    client.all_credentials = {"synthetic": [_CREDENTIAL]}
    client.global_timeout = 10.0
    client.acquire_timeout = acquire_timeout
    client.enable_request_logging = False
    client.max_concurrent_requests_per_key = {"synthetic": 1}
    client.cooldown_manager = _AvailableCooldownManager()
    client.usage_manager = manager
    client._apply_routing_policy = lambda model: (model, None)
    client._log_route_decision = lambda _decision: None
    client._get_provider_instance = lambda _provider: _PriorityProvider()
    client._resolve_model_id = lambda model, _provider: model
    client._apply_default_safety_settings = lambda _provider, _kwargs: None
    client.litellm_provider_params = {}
    client.max_retries = 1
    client.oauth_providers = set()
    client.provider_config = _IdentityProviderConfig()
    client.abort_on_callback_error = True
    client._litellm_logger_callback = lambda *_args, **_kwargs: None
    return client, manager


def observe_real_admission_wait(manager: UsageManager) -> asyncio.Event:
    condition = manager.key_states[_CREDENTIAL]["condition"]
    original_wait = condition.wait
    blocked = asyncio.Event()

    async def observed_wait() -> bool:
        blocked.set()
        return await original_wait()

    condition.wait = observed_wait
    return blocked
