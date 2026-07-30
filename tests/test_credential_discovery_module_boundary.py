from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.provider_config import discover_api_keys_from_env  # noqa: E402


def test_credential_discovery_has_dedicated_module_boundary() -> None:
    # Given the compatibility export used by provider configuration consumers.
    discover = discover_api_keys_from_env

    # When its implementation owner is inspected.
    owner = discover.__module__

    # Then environment credential discovery remains outside provider conversion policy.
    assert owner == "rotator_library.credential_discovery"
