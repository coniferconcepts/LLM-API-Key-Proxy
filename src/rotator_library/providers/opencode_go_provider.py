# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenCode GO OpenAI-compatible provider with optional usage-refresh job."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..go_usage.refresh import go_usage_job_config, run_go_usage_refresh
from . import openai_compatible_provider as _oai

if TYPE_CHECKING:
    from ..usage_manager import UsageManager


class OpenCodeGoProvider(_oai.OpenAICompatibleProvider):
    """File-registered plugin for opencode_go. Job is default-OFF."""

    def __init__(self, provider_name: str = "opencode_go") -> None:
        super().__init__(provider_name)

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        return go_usage_job_config()

    async def run_background_job(
        self,
        usage_manager: UsageManager,
        credentials: List[str],
    ) -> None:
        await run_go_usage_refresh(usage_manager, credentials)
