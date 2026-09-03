from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from email.message import Message
import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credential_admission_contract_support import make_client  # noqa: E402

client_module = importlib.import_module("rotator_library.client")
headers_module = importlib.import_module("rotator_library.openrouter_headers")
merge_provider_extra_headers = getattr(headers_module, "merge_provider_extra_headers")


@dataclass(frozen=True, slots=True)
class Request:
    headers: Mapping[str, str]
    request_id: str | None = None

    async def is_disconnected(self) -> bool:
        return False


def make_headers(values: list[tuple[str, str]]) -> Message:
    headers = Message()
    for name, value in values:
        headers[name] = value
    return headers


@pytest.mark.parametrize(
    "provider", ["openrouter", "openrouter_zdr", "openrouter_non_zdr", "openrouter_free"]
)
def test_openrouter_family_receives_session_and_attribution(provider: str) -> None:
    request = Request(
        {
            "X-Conversation-ID": "  conversation-42  ",
            "HTTP-Referer": "https://example.invalid/router",
            "X-OpenRouter-Title": "Synthetic Router",
        }
    )

    result = merge_provider_extra_headers({"messages": []}, request, provider)

    assert result["extra_headers"]["x-session-id"] == "conversation-42"
    assert result["extra_headers"]["HTTP-Referer"] == "https://example.invalid/router"
    assert result["extra_headers"]["X-OpenRouter-Title"] == "Synthetic Router"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("x-opencode-session", "opencode"),
        ("x-session-id", "session"),
        ("x-conversation-id", "conversation"),
        ("x-grok-conv-id", "grok"),
        ("x-request-id", "request"),
    ],
)
def test_session_resolution_accepts_every_alias(name: str, value: str) -> None:
    request = Request(make_headers([(name.upper(), f"  {value}  ")]))

    result = merge_provider_extra_headers({"messages": []}, request, "opencode_go")

    assert result["extra_headers"]["x-opencode-session"] == value


def test_session_resolution_uses_first_nonempty_repeated_case_variant() -> None:
    request = Request(
        make_headers(
            [
                ("X-OpenCode-Session", "  "),
                ("x-opencode-session", "  first  "),
                ("X-OpenCode-Session", "later"),
                ("X-Session-ID", "lower-priority"),
            ]
        )
    )

    result = merge_provider_extra_headers({"messages": []}, request, "opencode_go")

    assert result["extra_headers"]["x-opencode-session"] == "first"


@pytest.mark.parametrize("invalid", ["bad\rvalue", "bad\nvalue", "bad\x00value"])
def test_session_resolution_rejects_control_characters(invalid: str) -> None:
    request = Request(
        {
            "X-OpenCode-Session": invalid,
            "X-Session-ID": "safe-session",
        }
    )

    result = merge_provider_extra_headers({"messages": []}, request, "opencode_go")

    assert result["extra_headers"]["x-opencode-session"] == "safe-session"


def test_request_attribute_outranks_raw_request_id_header() -> None:
    request = Request(
        {"X-Request-ID": "raw-request"},
        request_id="validated-request",
    )

    result = merge_provider_extra_headers({"messages": []}, request, "opencode_go")

    assert result["extra_headers"]["x-opencode-session"] == "validated-request"


@pytest.mark.parametrize("stream", [False, True], ids=["execute", "streaming"])
@pytest.mark.parametrize(
    ("provider", "expected_session_header"),
    [
        ("opencode_go", "x-opencode-session"),
        ("openrouter_zdr", "x-session-id"),
        ("openrouter_non_zdr", "x-session-id"),
    ],
)
@pytest.mark.asyncio
async def test_final_litellm_kwargs_include_provider_headers_at_both_dispatch_seams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stream: bool,
    provider: str,
    expected_session_header: str,
) -> None:
    client, _manager = make_client(tmp_path, acquire_timeout=1.0)
    client.all_credentials = {provider: ["synthetic-credential"]}
    client.max_concurrent_requests_per_key = {provider: 1}
    request = Request(
        {
            "Authorization": "Bearer inbound-secret",
            "X-OpenCode-Session": "  inbound-session  ",
            "x-opencode-session": "later-session",
            "HTTP-Referer": "https://example.invalid/router",
        },
        request_id="validated-request",
    )
    captured_headers: list[dict[str, str]] = []
    events: list[str] = []
    original_sanitize = client_module.sanitize_request_payload
    original_merge = client_module.merge_provider_extra_headers

    def tracked_sanitize(kwargs, model):
        events.append("sanitize")
        return original_sanitize(kwargs, model)

    def tracked_merge(kwargs, inbound_request, inbound_provider):
        events.append("merge")
        return original_merge(kwargs, inbound_request, inbound_provider)

    async def empty_stream():
        if False:
            yield "unreachable"

    async def capturing_acompletion(**kwargs):
        captured_headers.append(dict(kwargs["extra_headers"]))
        return empty_stream() if stream else {"ok": True}

    monkeypatch.setattr(client_module, "sanitize_request_payload", tracked_sanitize)
    monkeypatch.setattr(client_module, "merge_provider_extra_headers", tracked_merge)
    monkeypatch.setattr(client_module.litellm, "acompletion", capturing_acompletion)

    dispatched = client.acompletion(
        request=request,
        model=f"{provider}/synthetic-model",
        messages=[{"role": "user", "content": "headers"}],
        stream=stream,
    )
    if stream:
        async for _chunk in dispatched:
            pass
    else:
        await dispatched

    assert events == ["sanitize", "merge"]
    assert len(captured_headers) == 1
    final_headers = captured_headers[0]
    assert final_headers[expected_session_header] == "inbound-session"
    assert sum(key.casefold() == expected_session_header for key in final_headers) == 1
    assert not any(key.casefold() == "authorization" for key in final_headers)
