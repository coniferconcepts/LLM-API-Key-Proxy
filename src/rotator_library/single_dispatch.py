from __future__ import annotations

import hmac
import ipaddress
import os
from typing import Any

SINGLE_DISPATCH_HEADER = "x-mirrowel-single-dispatch"
SINGLE_DISPATCH_TOKEN_HEADER = "x-mirrowel-single-dispatch-token"
SINGLE_DISPATCH_TOKEN_ENV = "MIRROWEL_SINGLE_DISPATCH_TOKEN"
SINGLE_DISPATCH_ALIAS_HEADER = "x-opencode-alias"


class SingleDispatchRejected(ValueError):
    def __init__(self, message: str, reason_class: str = "rejected") -> None:
        super().__init__(message)
        self.reason_class = reason_class


def single_dispatch_requested(request: Any, model: Any, *, authenticated: bool) -> bool:
    values = request.headers.getlist(SINGLE_DISPATCH_HEADER)
    tokens = request.headers.getlist(SINGLE_DISPATCH_TOKEN_HEADER)
    if not values and not tokens:
        return False
    try:
        peer_is_loopback = ipaddress.ip_address(request.client.host).is_loopback
    except (AttributeError, ValueError):
        peer_is_loopback = False
    expected = os.environ.get(SINGLE_DISPATCH_TOKEN_ENV, "")
    token = tokens[0] if len(tokens) == 1 else ""
    token_ok = (
        len(tokens) == 1
        and len(expected) >= 32
        and expected.isascii()
        and token.isascii()
        and hmac.compare_digest(token.encode("utf-8"), expected.encode("ascii"))
    )
    if values != ["1"] or len(tokens) != 1:
        raise SingleDispatchRejected("single dispatch authorization rejected", "header_shape")
    if not token_ok:
        raise SingleDispatchRejected(
            "single dispatch authorization rejected", "token_missing_or_invalid"
        )
    if request.headers.getlist(SINGLE_DISPATCH_ALIAS_HEADER) != ["kimi-k3-advisor"]:
        raise SingleDispatchRejected("single dispatch authorization rejected", "alias_mismatch")
    if not authenticated:
        raise SingleDispatchRejected("single dispatch authorization rejected", "not_authenticated")
    if not peer_is_loopback:
        raise SingleDispatchRejected("single dispatch authorization rejected", "not_loopback")
    if not isinstance(model, str) or model != "chutes/moonshotai/Kimi-K3-TEE":
        raise SingleDispatchRejected("single dispatch authorization rejected", "model_mismatch")
    return True


class SingleDispatchGuard:
    def __init__(self) -> None:
        self.dispatched = False

    async def __call__(self, _request: Any, prepared: dict[str, Any]) -> None:
        if prepared.get("model") != "chutes/moonshotai/Kimi-K3-TEE":
            raise SingleDispatchRejected("single dispatch target rejected", "model_mismatch")
        if self.dispatched:
            raise SingleDispatchRejected("single dispatch budget exhausted", "budget_exhausted")
        self.dispatched = True
        prepared["num_retries"] = 0
        prepared["max_retries"] = 0
        prepared.pop("retry", None)
