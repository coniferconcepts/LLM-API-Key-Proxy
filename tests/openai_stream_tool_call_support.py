from __future__ import annotations

from collections.abc import AsyncGenerator

from rotator_library.openai_stream_normalize import JsonObject


class UsageManager:
    def __init__(self) -> None:
        self.released: list[tuple[str, str]] = []

    async def record_success(self, _key: str, _model: str) -> None:
        return

    async def release_key(self, key: str, model: str) -> None:
        self.released.append((key, model))


async def upstream_chunks() -> AsyncGenerator[JsonObject, None]:
    yield {"id": None, "choices": [{"index": 0, "delta": {"content": "planning"}}]}
    yield {
        "id": None,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": None,
                            "type": "function",
                            "function": {"name": None, "arguments": ""},
                        }
                    ]
                },
            }
        ],
    }
    yield {
        "id": None,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": None, "function": {"name": "edit", "arguments": ""}}
                    ]
                },
            }
        ],
    }
    yield {
        "id": None,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": None, "function": {"name": None, "arguments": "{"}}
                    ]
                },
            }
        ],
    }
    yield {
        "id": None,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": None,
                            "type": "function",
                            "function": {"name": "read", "arguments": ""},
                        }
                    ]
                },
            }
        ],
    }
    yield {
        "id": "chatcmpl_existing",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 2,
                            "id": "call_existing",
                            "type": "function",
                            "function": {"name": "write", "arguments": ""},
                        }
                    ]
                },
            }
        ],
    }
