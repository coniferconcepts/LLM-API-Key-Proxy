# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenCode GO Messages provider. Shares GO usage refresh (deduped by key)."""

from __future__ import annotations

from . import opencode_go_provider as _go


class OpenCodeGoMessagesProvider(_go.OpenCodeGoProvider):
    def __init__(self) -> None:
        super().__init__("opencode_go_messages")
