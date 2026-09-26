from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest
from starlette.datastructures import Headers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import rotator_library.client as client_module
from rotator_library.single_dispatch import (
    SingleDispatchRejected,
    single_dispatch_requested,
)


@pytest.fixture(autouse=True)
def isolate_litellm_session(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "aclient_session", None)


def test_client_import_uses_requested_mirrowel_tree() -> None:
    tree = os.environ.get("MIRROWEL_TEST_TREE")
    if tree is None:
        pytest.skip("MIRROWEL_TEST_TREE is not set")
    expected = Path(tree).resolve() / "src/rotator_library/client.py"
    assert Path(client_module.__file__).resolve() == expected


def test_single_dispatch_rejection_reason_omits_token(monkeypatch: pytest.MonkeyPatch) -> None:
    canary = "canary-token-value-0123456789abcdef"
    monkeypatch.setenv("MIRROWEL_SINGLE_DISPATCH_TOKEN", "expected-token-value-0123456789abcd")
    headers = Headers(
        {
            "x-mirrowel-single-dispatch": "1",
            "x-mirrowel-single-dispatch-token": canary,
            "x-opencode-alias": "kimi-k3-advisor",
        }
    )
    request = SimpleNamespace(headers=headers, client=SimpleNamespace(host="127.0.0.1"))
    with pytest.raises(SingleDispatchRejected) as raised:
        single_dispatch_requested(request, "chutes/moonshotai/Kimi-K3-TEE", authenticated=True)
    assert raised.value.reason_class == "token_missing_or_invalid"
    assert canary not in str(raised.value)


def test_single_dispatch_requires_loopback_proxy_auth_alias_and_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "synthetic-token-of-at-least-32-characters"
    monkeypatch.setenv("MIRROWEL_SINGLE_DISPATCH_TOKEN", token)
    headers = Headers(
        {
            "x-mirrowel-single-dispatch": "1",
            "x-mirrowel-single-dispatch-token": token,
            "x-opencode-alias": "kimi-k3-advisor",
        }
    )
    request = SimpleNamespace(headers=headers, client=SimpleNamespace(host="127.0.0.1"))
    model = "chutes/moonshotai/Kimi-K3-TEE"
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=False)
    assert single_dispatch_requested(request, model, authenticated=True)
    request.client.host = "192.0.2.1"
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    request.client.host = "127.0.0.1"
    request.headers = Headers(
        {
            "x-mirrowel-single-dispatch": "1",
            "x-mirrowel-single-dispatch-token": token,
            "x-opencode-alias": "kimi-k3",
        }
    )
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    request.headers = Headers(
        {
            "x-mirrowel-single-dispatch": "1",
            "x-mirrowel-single-dispatch-token": "forged",
            "x-opencode-alias": "kimi-k3-advisor",
        }
    )
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    request.headers = Headers(
        raw=[
            (b"x-mirrowel-single-dispatch", b"1"),
            (b"x-mirrowel-single-dispatch-token", b"\xe9" * 40),
            (b"x-opencode-alias", b"kimi-k3-advisor"),
        ]
    )
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    request.headers = headers
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, "chutes/moonshotai/Kimi-K3", authenticated=True)
    request.headers = Headers(
        raw=[
            (b"x-mirrowel-single-dispatch", b"1"),
            (b"x-mirrowel-single-dispatch", b"1"),
            (b"x-mirrowel-single-dispatch-token", token.encode()),
            (b"x-opencode-alias", b"kimi-k3-advisor"),
        ]
    )
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    request.headers = headers
    monkeypatch.delenv("MIRROWEL_SINGLE_DISPATCH_TOKEN")
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    monkeypatch.setenv("MIRROWEL_SINGLE_DISPATCH_TOKEN", token)
    request.client.host = "127.0.0.1"
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, "chutes/other-model", authenticated=True)
    request.headers = Headers({"x-mirrowel-single-dispatch": "yes"})
    with pytest.raises(SingleDispatchRejected):
        single_dispatch_requested(request, model, authenticated=True)
    request.headers = Headers({})
    assert not single_dispatch_requested(request, model, authenticated=False)
