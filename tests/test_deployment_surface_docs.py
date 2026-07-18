from __future__ import annotations

from deployment_surface_helpers import (
    DEPLOYMENT_GUIDE,
    README,
    read as _read,
)


def test_readme_public_bind_commands_include_the_approval_contract() -> None:
    source = _read(README)
    deployment_guide = _read(DEPLOYMENT_GUIDE)

    assert "MIRROWEL_ALLOW_NETWORK_BIND=true" in source
    assert "MIRROWEL_ALLOWED_HOSTS=proxy.example.com" in source
    assert "Direct ASGI launch is supported only when" in source
    assert "--host 0.0.0.0 --port 8000" in source
    assert "default: 127.0.0.1" in source
    assert "MIRROWEL_ALLOW_NETWORK_BIND=true" in deployment_guide
    assert "MIRROWEL_ALLOWED_HOSTS=proxy.example.com" in deployment_guide

    for line in deployment_guide.splitlines():
        if "uvicorn" in line and "--host 0.0.0.0" in line:
            assert (
                "MIRROWEL_ALLOW_NETWORK_BIND=true" in line
                and "MIRROWEL_ALLOWED_HOSTS=proxy.example.com" in line
            ) or line.startswith("ExecStart=")
