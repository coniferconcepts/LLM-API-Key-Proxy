from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

headers_module = importlib.import_module("rotator_library.openrouter_headers")
merge_provider_extra_headers = getattr(headers_module, "merge_provider_extra_headers")


@dataclass(frozen=True, slots=True)
class Request:
    headers: Mapping[str, str]


def test_http_session_overwrites_mixed_case_body_session() -> None:
    request = Request({"x-opencode-session": "derived-session"})

    result = merge_provider_extra_headers(
        {"extra_headers": {"X-OpenCode-Session": "attacker"}},
        request,
        "opencode_go",
    )

    assert result["extra_headers"]["x-opencode-session"] == "derived-session"
    assert sum(key.casefold() == "x-opencode-session" for key in result["extra_headers"]) == 1


@pytest.mark.parametrize(
    "protected_name",
    ["Authorization", "Proxy-Authorization", "Cookie", "X-Api-Key"],
)
def test_protected_body_header_is_never_copied(protected_name: str) -> None:
    request = Request({"x-opencode-session": "derived-session"})

    result = merge_provider_extra_headers(
        {"extra_headers": {protected_name: "attacker"}},
        request,
        "opencode_go",
    )

    assert not any(key.casefold() == protected_name.casefold() for key in result["extra_headers"])


def test_bounded_attempt_body_header_is_preserved() -> None:
    request = Request({"x-opencode-session": "derived-session"})

    result = merge_provider_extra_headers(
        {"extra_headers": {"X-OpenCode-Bounded-Attempt": "attempt-2"}},
        request,
        "opencode_go",
    )

    assert result["extra_headers"]["X-OpenCode-Bounded-Attempt"] == "attempt-2"


def test_openrouter_free_receives_session_header() -> None:
    request = Request({"x-opencode-session": "derived-session"})

    result = merge_provider_extra_headers({"messages": []}, request, "openrouter_free")

    assert result["extra_headers"]["x-session-id"] == "derived-session"


@pytest.mark.parametrize(
    "protected_name",
    ["Authorization", "Proxy-Authorization", "Cookie", "X-Api-Key"],
)
def test_protected_only_body_headers_are_dropped_for_unmapped_provider(
    protected_name: str,
) -> None:
    result = merge_provider_extra_headers(
        {"extra_headers": {protected_name: "attacker"}},
        Request({}),
        "ollama_cloud",
    )

    assert result.get("extra_headers", {}) == {}
