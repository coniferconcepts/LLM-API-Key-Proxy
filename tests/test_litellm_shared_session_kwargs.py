from __future__ import annotations

import inspect
from pathlib import Path
import sys

import litellm
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credential_admission_contract_support import _MODEL, make_client  # noqa: E402


def test_litellm_acompletion_still_accepts_shared_session() -> None:
    assert "shared_session" in inspect.signature(litellm.acompletion).parameters


@pytest.mark.asyncio
async def test_execute_with_retry_forwards_owned_shared_session(tmp_path: Path) -> None:
    client, _manager = make_client(tmp_path, acquire_timeout=1.0)
    sentinel = object()
    client.litellm_shared_session = sentinel
    captured: dict[str, object] = {}

    async def api_call(**kwargs: object) -> dict[str, bool]:
        captured.update(kwargs)
        return {"ok": True}

    await client._execute_with_retry(
        api_call,
        request=None,
        model=_MODEL,
        messages=[{"role": "user", "content": "session"}],
    )

    assert captured["shared_session"] is sentinel
    assert captured.get("model") == _MODEL
