from __future__ import annotations

import json
from pathlib import Path

import pytest

from proxy_app import detailed_logger


@pytest.mark.parametrize(
    "body",
    (
        {"choices": [None]},
        {"choices": ["unexpected"]},
        {"choices": {"finish_reason": "stop"}},
        {"usage": 7},
        {"usage": "unexpected"},
        {"usage": [1, 2, 3]},
    ),
)
def test_raw_logger_drops_malformed_metadata_shapes_without_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: dict[str, object],
) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(mode=0o700)
    monkeypatch.setattr(detailed_logger, "get_logs_dir", lambda: logs_dir)
    logger = detailed_logger.RawIOLogger()

    logger.log_final_response(200, {}, body)

    metadata = json.loads((logger.log_dir / "metadata.json").read_text())
    assert metadata["finish_reason"] == "N/A"
    assert metadata["usage"] == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
