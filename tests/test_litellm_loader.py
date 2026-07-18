from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMPORT_MARKER = "__LITELLM_IMPORT_POLICY__"
REMOTE_COST_MAP = "https://fixture.invalid/model_prices_and_context_window.json"


def _run_loader_probe(tmp_path: Path, *, safe_mode: bool) -> list[str]:
    probe = "\n".join(
        (
            "import json",
            "import os",
            "from proxy_app import litellm_loader",
            "attempts = []",
            "sentinel = object()",
            "def fake_import(name):",
            "    assert name == 'litellm'",
            "    if os.getenv('LITELLM_LOCAL_MODEL_COST_MAP', '').lower() != 'true':",
            f"        attempts.append({REMOTE_COST_MAP!r})",
            "    return sentinel",
            "litellm_loader.importlib.import_module = fake_import",
            "loaded = litellm_loader.load_litellm(",
            "    local_transport_safe_mode=os.environ['SAFE_MODE'] == 'true'",
            ")",
            "assert loaded is sentinel",
            f"print({IMPORT_MARKER!r} + json.dumps(attempts))",
        )
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["SAFE_MODE"] = str(safe_mode).lower()
    environment.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    record = next(line for line in result.stdout.splitlines() if line.startswith(IMPORT_MARKER))
    return json.loads(record.removeprefix(IMPORT_MARKER))


@pytest.mark.parametrize(
    "safe_modes",
    (
        (True, False, True, False),
        (False, True, False, True),
    ),
)
def test_fresh_process_import_policy_is_order_independent(
    tmp_path: Path,
    safe_modes: tuple[bool, ...],
) -> None:
    attempts = [_run_loader_probe(tmp_path, safe_mode=safe_mode) for safe_mode in safe_modes]

    assert attempts == [[] if safe_mode else [REMOTE_COST_MAP] for safe_mode in safe_modes]
