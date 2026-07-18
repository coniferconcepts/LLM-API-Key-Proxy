from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import pytest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proxy_app.safe_errors import SafeUnhandledErrorMiddleware  # noqa: E402

HTTP_SCOPE: Scope = {"type": "http"}


class SyntheticAppError(RuntimeError):
    pass


async def _empty_receive() -> Message:
    return {"type": "http.disconnect"}


def _run_middleware(app: ASGIApp, *, path: str | None = None) -> list[Message]:
    sent: list[Message] = []

    async def record(message: Message) -> None:
        sent.append(message)

    async def invoke() -> None:
        scope = {**HTTP_SCOPE, **({"path": path} if path is not None else {})}
        await SafeUnhandledErrorMiddleware(app)(scope, _empty_receive, record)

    anyio.run(invoke)
    return sent


def test_exception_before_response_start_returns_safe_500() -> None:
    # Given a downstream app that fails before committing an HTTP response.
    async def fail_before_start(_scope: Scope, _receive: Receive, _send: Send) -> None:
        raise SyntheticAppError("private upstream detail")

    # When the safe-error middleware handles the failure.
    messages = _run_middleware(fail_before_start)

    # Then it emits one complete safe 500 response.
    assert [message["type"] for message in messages] == [
        "http.response.start",
        "http.response.body",
    ]
    assert messages[0]["status"] == 500
    assert messages[1].get("more_body", False) is False
    assert b"private upstream detail" not in messages[1]["body"]


@pytest.mark.parametrize("path", ("/v1/messages", "/v1/messages/count_tokens"))
def test_exception_before_anthropic_response_uses_compatible_safe_500(path: str) -> None:
    # Given an Anthropic-compatible request that fails outside endpoint handlers.
    async def fail_before_start(_scope: Scope, _receive: Receive, _send: Send) -> None:
        raise SyntheticAppError("private upstream detail")

    # When the safe-error middleware handles the failure.
    messages = _run_middleware(fail_before_start, path=path)

    # Then the fallback retains Anthropic's envelope without provider details.
    payload = json.loads(messages[1]["body"])
    assert messages[0]["status"] == 500
    assert payload == {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": "The proxy could not complete the request.",
        },
    }
    assert "private upstream detail" not in messages[1]["body"].decode()


def test_exception_after_incomplete_response_terminates_response_once() -> None:
    # Given a downstream app that starts a response and leaves its body incomplete.
    async def fail_during_body(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        raise SyntheticAppError("private stream detail")

    # When the safe-error middleware handles the failure.
    messages = _run_middleware(fail_during_body)

    # Then it preserves the partial response and adds exactly one terminal body.
    assert messages == [
        {"type": "http.response.start", "status": 200, "headers": []},
        {"type": "http.response.body", "body": b"partial", "more_body": True},
        {"type": "http.response.body", "body": b"", "more_body": False},
    ]


@pytest.mark.parametrize(
    "final_body",
    [
        {"type": "http.response.body", "body": b"complete"},
        {"type": "http.response.body", "body": b"complete", "more_body": False},
    ],
    ids=["implicit-final-body", "explicit-final-body"],
)
def test_exception_after_completed_response_does_not_send_second_terminal_body(
    final_body: Message,
) -> None:
    # Given a downstream app that completes its response and then raises.
    async def fail_after_completion(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(final_body)
        raise SyntheticAppError("late private detail")

    # When the safe-error middleware observes the late failure.
    messages = _run_middleware(fail_after_completion)

    # Then it logs only; the already-complete ASGI response remains unchanged.
    assert messages == [
        {"type": "http.response.start", "status": 200, "headers": []},
        final_body,
    ]
