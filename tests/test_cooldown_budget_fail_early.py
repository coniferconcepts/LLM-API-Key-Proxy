from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credential_admission_contract_support import _MODEL, make_client  # noqa: E402
from rotator_library.error_handler import NoAvailableKeysError  # noqa: E402
from rotator_library.client import _COOLDOWN_BUDGET_EXCEEDED_MESSAGE  # noqa: E402


class _LongCooldownManager:
    async def is_cooling_down(self, _provider: str) -> bool:
        return True

    async def get_cooldown_remaining(self, _provider: str) -> float:
        return 60.0


@pytest.mark.asyncio
async def test_nonstream_fail_early_raises_exhausted_without_calling_upstream(
    tmp_path: Path,
) -> None:
    client, _manager = make_client(tmp_path, acquire_timeout=0.05)
    client.global_timeout = 1.0
    client.cooldown_manager = _LongCooldownManager()
    called = False

    async def unexpected_api(**_kwargs: object) -> object:
        nonlocal called
        called = True
        pytest.fail("upstream must not be called when cooldown exceeds budget")

    with pytest.raises(NoAvailableKeysError) as captured:
        await client._execute_with_retry(
            unexpected_api,
            request=None,
            model=_MODEL,
            messages=[{"role": "user", "content": "wait"}],
        )
    assert called is False
    assert captured.value.category == "proxy_all_credentials_exhausted"
    assert captured.value.code == "acquisition_timeout_exhausted"
    assert captured.value.message == _COOLDOWN_BUDGET_EXCEEDED_MESSAGE
    assert "synthetic" not in captured.value.message
    assert captured.value.soonest_end is not None

    from proxy_app.safe_errors import handle_credential_failure

    http = handle_credential_failure(captured.value, None)
    assert http.status_code == 503
    body = json.loads(http.body)
    assert body["error"]["code"] == "all_credentials_exhausted"
    assert body["error"] is not None


@pytest.mark.asyncio
async def test_stream_fail_early_emits_terminal_exhausted_then_done(tmp_path: Path) -> None:
    client, _manager = make_client(tmp_path, acquire_timeout=0.05)
    client.global_timeout = 1.0
    client.cooldown_manager = _LongCooldownManager()

    chunks: list[str] = []
    async for chunk in client._streaming_acompletion_with_retry(
        request=None,
        model=_MODEL,
        messages=[{"role": "user", "content": "wait"}],
        stream=True,
    ):
        chunks.append(chunk)

    assert chunks[-1] == "data: [DONE]\n\n"
    payload_line = chunks[0]
    assert payload_line.startswith("data: ")
    body = json.loads(payload_line[len("data: ") :].strip())
    assert body["error"]["type"] == "proxy_all_credentials_exhausted"
    assert "synthetic-credential" not in json.dumps(body)
