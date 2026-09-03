import importlib
import sys
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

headers_module = importlib.import_module("rotator_library.openrouter_headers")
go_usage_module = importlib.import_module("rotator_library.go_usage.client")
merge_provider_extra_headers = getattr(headers_module, "merge_provider_extra_headers")


class Request:
    def __init__(self, headers: dict[str, str], request_id: str | None = None) -> None:
        self.headers = headers
        self.request_id = request_id


@pytest.mark.parametrize(
    ("provider", "expected_key"),
    [
        ("opencode_go", "x-opencode-session"),
        ("opencode_go_messages", "x-opencode-session"),
        ("xai_oauth", "x-grok-conv-id"),
        ("xai", "x-grok-conv-id"),
        ("fireworks", "x-session-affinity"),
        ("openrouter", "x-session-id"),
    ],
)
def test_provider_headers_use_conversation_session(provider: str, expected_key: str) -> None:
    # Given an inbound conversation identifier and provider-specific mapping.
    request = Request({"X-Grok-Conv-Id": "conversation-42"})

    # When provider headers are merged.
    result = merge_provider_extra_headers({"messages": []}, request, provider)

    # Then the provider receives the same resolved session value.
    assert result["extra_headers"][expected_key] == "conversation-42"


def test_go_headers_add_router_identity() -> None:
    # Given a GO request with no inbound session id.
    request = Request({}, request_id="request-42")

    # When GO headers are merged.
    result = merge_provider_extra_headers({"messages": []}, request, "opencode_go")

    # Then GO receives a stable session plus the router identity headers.
    assert result["extra_headers"] == {
        "x-opencode-session": "request-42",
        "x-opencode-client": "opencode-router",
        "User-Agent": "opencode-router-mirrowel/1",
    }


@pytest.mark.parametrize(
    ("provider", "expected_additions"),
    [
        (
            "opencode_go",
            {
                "x-opencode-client": "opencode-router",
                "User-Agent": "opencode-router-mirrowel/1",
            },
        ),
        ("xai_oauth", {}),
        ("fireworks", {}),
        ("openrouter", {}),
    ],
)
def test_existing_provider_header_is_overwritten_by_derived(
    provider: str, expected_additions: dict[str, str]
) -> None:
    # Given a caller-supplied provider header using mixed casing.
    existing_key = {
        "opencode_go": "X-OpenCode-Session",
        "xai_oauth": "X-Grok-Conv-Id",
        "fireworks": "X-Session-Affinity",
        "openrouter": "X-Session-Id",
    }[provider]
    request = Request({"x-opencode-session": "inbound-session"})

    # When provider headers are merged.
    result = merge_provider_extra_headers(
        {"extra_headers": {existing_key: "caller-value"}}, request, provider
    )

    # Then the request-derived value replaces the mixed-case body value.
    expected_key = {
        "opencode_go": "x-opencode-session",
        "xai_oauth": "x-grok-conv-id",
        "fireworks": "x-session-affinity",
        "openrouter": "x-session-id",
    }[provider]
    assert result["extra_headers"] == {
        expected_key: "inbound-session",
        **expected_additions,
    }


@pytest.mark.parametrize(
    ("headers", "request_id", "expected"),
    [
        (
            {
                "X-OpenCode-Session": "open-code",
                "X-Session-Id": "session",
                "X-Grok-Conv-Id": "grok",
                "X-Request-Id": "inbound-request",
            },
            "request-attribute",
            "open-code",
        ),
        (
            {
                "X-OpenCode-Session": " ",
                "X-Session-Id": "session",
                "X-Grok-Conv-Id": "grok",
                "X-Request-Id": "inbound-request",
            },
            "request-attribute",
            "session",
        ),
        (
            {"X-Session-Id": " ", "X-Grok-Conv-Id": "grok", "X-Request-Id": "inbound-request"},
            "request-attribute",
            "grok",
        ),
        ({"X-Request-Id": "inbound-request"}, None, "inbound-request"),
        ({}, "request-attribute", "request-attribute"),
    ],
)
def test_session_resolution_uses_first_nonempty_candidate(
    headers: dict[str, str], request_id: str | None, expected: str
) -> None:
    # Given candidate inbound identifiers with distinct values.
    request = Request(headers, request_id=request_id)

    # When headers are merged for GO.
    result = merge_provider_extra_headers({"messages": []}, request, "opencode_go")

    # Then the first non-empty identifier in contract order wins.
    assert result["extra_headers"]["x-opencode-session"] == expected


def test_go_session_uses_uuid_as_last_resort() -> None:
    # Given a GO request without any session or request identifier.
    request = Request({})

    # When provider headers are merged.
    result = merge_provider_extra_headers({"messages": []}, request, "opencode_go")

    # Then GO still receives a valid session value.
    generated = result["extra_headers"]["x-opencode-session"]
    assert str(uuid.UUID(generated)) == generated


@pytest.mark.asyncio
async def test_go_usage_sends_session_header() -> None:
    # Given a fake GO usage endpoint capturing the request headers.
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "usage": {
                    "rolling": {"status": "ok", "percent": 1, "resetsAt": "2026-09-03T00:00:00Z"},
                    "weekly": {"status": "ok", "percent": 1, "resetsAt": "2026-09-03T00:00:00Z"},
                    "monthly": {"status": "ok", "percent": 1, "resetsAt": "2026-09-03T00:00:00Z"},
                }
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await go_usage_module.fetch_usage(
            client, base="https://opencode.ai/zen/go/v1", api_key="secret"
        )

    # Then usage traffic carries the stable router session marker.
    assert result["ok"] is True
    assert captured["x-opencode-session"] == "opencode-router-go-usage"
