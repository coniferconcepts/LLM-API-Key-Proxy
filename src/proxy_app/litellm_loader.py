from __future__ import annotations

import importlib
import os
from types import ModuleType


def load_litellm(*, local_transport_safe_mode: bool) -> ModuleType:
    if local_transport_safe_mode:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    return importlib.import_module("litellm")
