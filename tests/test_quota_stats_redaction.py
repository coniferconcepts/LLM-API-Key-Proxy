from __future__ import annotations

import json
from pathlib import Path

import pytest

from rotator_library.client import RotatingClient
from rotator_library.usage_persistence import _credential_fingerprint


@pytest.mark.asyncio
async def test_get_quota_stats_redacts_public_credential_fields(tmp_path: Path) -> None:
    raw_credential = "sk-test-quota-stats-secret"
    env_credential = "env://provider/1"
    client = RotatingClient(
        api_keys={},
        oauth_credentials={},
        usage_file_path=tmp_path / "usage.json",
        configure_logging=False,
    )
    client.usage_manager._usage_data = {  # noqa: SLF001
        raw_credential: {"models": {"openai/model": {}}},
        env_credential: {"models": {"provider/model": {}}},
    }
    client.usage_manager._initialized.set()  # noqa: SLF001

    stats = await client.get_quota_stats()
    serialized = json.dumps(stats)
    credentials = [
        credential
        for provider in stats["providers"].values()
        for credential in provider["credentials"]
    ]

    assert "sk-" not in serialized
    assert all("full_path" not in credential for credential in credentials)
    assert {credential["identifier"] for credential in credentials} == {
        _credential_fingerprint(raw_credential),
        env_credential,
    }
