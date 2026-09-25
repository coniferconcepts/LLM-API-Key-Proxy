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
    pass


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
    if (
        values != ["1"]
        or len(tokens) != 1
        or len(expected) < 32
        or not expected.isascii()
        or not hmac.compare_digest(tokens[0].encode("utf-8"), expected.encode("ascii"))
        or request.headers.getlist(SINGLE_DISPATCH_ALIAS_HEADER) != ["kimi-k3-advisor"]
        or not authenticated
        or not peer_is_loopback
        or not isinstance(model, str)
        or model != "chutes/moonshotai/Kimi-K3-TEE"
    ):
        raise SingleDispatchRejected("single dispatch authorization rejected")
    return True


class SingleDispatchGuard:
    def __init__(self) -> None:
        self.dispatched = False

    async def __call__(self, _request: Any, prepared: dict[str, Any]) -> None:
        if prepared.get("model") != "chutes/moonshotai/Kimi-K3-TEE":
            raise SingleDispatchRejected("single dispatch target rejected")
        if self.dispatched:
            raise SingleDispatchRejected("single dispatch budget exhausted")
        self.dispatched = True
        prepared["num_retries"] = 0
        prepared["max_retries"] = 0
        prepared.pop("retry", None)
