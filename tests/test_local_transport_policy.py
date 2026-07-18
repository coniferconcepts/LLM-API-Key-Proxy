from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

normalize_local_xai_base = importlib.import_module(
    "proxy_app.local_transport_policy"
).normalize_local_xai_base


@pytest.mark.parametrize(
    ("configured", "expected"),
    (
        ("http://127.0.0.1:2465/v1", "http://127.0.0.1:2465/v1"),
        ("https://127.0.0.2:2465/v1", "https://127.0.0.2:2465/v1"),
        ("http://localhost:2465/v1", "http://127.0.0.1:2465/v1"),
        ("http://[::1]:2465/v1", "http://[::1]:2465/v1"),
    ),
)
def test_local_xai_base_accepts_only_canonical_loopback_variants(
    configured: str,
    expected: str,
) -> None:
    assert normalize_local_xai_base(configured) == expected


@pytest.mark.parametrize(
    "configured",
    (
        "http://198.51.100.7:2465/v1",
        "http://example.com:2465/v1",
        "http://localhost.example:2465/v1",
        "http://localhost.:2465/v1",
        "http://2130706433:2465/v1",
        "http://0x7f000001:2465/v1",
        "http://[::ffff:127.0.0.1]:2465/v1",
        "http://%31%32%37.0.0.1:2465/v1",
        "http://127.0.0.1:2465/%76%31",
        "http://user@127.0.0.1:2465/v1",
        "http://user:secret@127.0.0.1:2465/v1",
        "ftp://127.0.0.1:2465/v1",
        "http://127.0.0.1/v1",
        "http://127.0.0.1:80/v1",
        "https://127.0.0.1:443/v1",
        "http://127.0.0.1:2466/v1",
        "http://127.0.0.1:invalid/v1",
        "http://127.0.0.1:2465/v1?target=evil",
        "http://127.0.0.1:2465/v1#fragment",
        "http://127.0.0.1:2465/v1/models",
        "http://127.0.0.1:02465/v1",
        "http://127.0.0.1:002465/v1",
        "http://127.0.0.1:+2465/v1",
        "http://127.0.0.1:2465/v1/",
        "http://127.0.0.1:2465/v1?",
        "http://127.0.0.1:2465/v1#",
        "http://127.0.0.1:2465/v1?#",
        "HTTP://127.0.0.1:2465/v1",
        "http://LOCALHOST:2465/v1",
        "http://[0:0:0:0:0:0:0:1]:2465/v1",
        " http://127.0.0.1:2465/v1",
        "",
    ),
)
def test_local_xai_base_rejects_ambiguous_or_non_reserved_destinations(
    configured: str,
) -> None:
    with pytest.raises(ValueError, match="local xAI base"):
        normalize_local_xai_base(configured)


@pytest.mark.parametrize("character", tuple(chr(value) for value in range(0x21)) + ("\x7f",))
@pytest.mark.parametrize(
    "template",
    (
        "http://127.0.0.1{character}:2465/v1",
        "http://127.0.0.1:{character}2465/v1",
        "http://127.0.0.1:24{character}65/v1",
        "http://127.0.0.1:2465/v{character}1",
    ),
)
def test_local_xai_base_rejects_ascii_controls_and_space_everywhere(
    character: str,
    template: str,
) -> None:
    with pytest.raises(ValueError, match="local xAI base"):
        normalize_local_xai_base(template.format(character=character))


def test_safe_mode_rejects_non_loopback_base_before_litellm_import(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    environment["MIRROWEL_LOCAL_TRANSPORT_SAFE_MODE"] = "true"
    environment["XAI_OAUTH_API_BASE"] = "http://198.51.100.7:2465/v1"
    environment["PROXY_API_KEY"] = "proxy-token"
    environment["MIRROWEL_ALLOWED_HOSTS"] = "127.0.0.1,localhost"

    result = subprocess.run(
        [sys.executable, "-c", "import proxy_app.main", "--port", "8000"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Invalid local xAI base: host must be loopback" in combined_output
    assert "Loading LiteLLM library" not in combined_output


def test_safe_mode_rejects_noncanonical_port_before_litellm_import(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    environment["MIRROWEL_LOCAL_TRANSPORT_SAFE_MODE"] = "true"
    environment["XAI_OAUTH_API_BASE"] = "http://127.0.0.1:02465/v1"
    environment["PROXY_API_KEY"] = "proxy-token"
    environment["MIRROWEL_ALLOWED_HOSTS"] = "127.0.0.1,localhost"

    result = subprocess.run(
        [sys.executable, "-c", "import proxy_app.main", "--port", "8000"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Invalid local xAI base" in combined_output
    assert "Loading LiteLLM library" not in combined_output
