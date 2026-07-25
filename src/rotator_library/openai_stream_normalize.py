from __future__ import annotations

from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class OpenAIStreamNormalizer:
    """Normalize OpenAI chunk identifiers while retaining per-stream tool state."""

    def __init__(self) -> None:
        self._chunk_id: str | None = None
        self._tool_call_ids: dict[tuple[int, int], str] = {}

    def normalize(self, chunk: JsonObject) -> JsonObject:
        if "id" in chunk:
            chunk_id = chunk["id"]
            if chunk_id is None:
                if self._chunk_id is None:
                    self._chunk_id = "chatcmpl_stream"
                chunk["id"] = self._chunk_id
            elif isinstance(chunk_id, str) and chunk_id:
                self._chunk_id = chunk_id

        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return chunk

        for choice_position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_index = tool_call.get("index")
                if not isinstance(tool_index, int):
                    continue

                state_key = (choice_position, tool_index)
                stable_id = self._tool_call_ids.get(state_key)
                if stable_id is None:
                    provided_id = tool_call.get("id")
                    if isinstance(provided_id, str) and provided_id:
                        stable_id = provided_id
                    elif choice_position == 0:
                        stable_id = f"call_{tool_index}"
                    else:
                        stable_id = f"call_{choice_position}_{tool_index}"
                    self._tool_call_ids[state_key] = stable_id

                tool_call["id"] = stable_id

        return chunk
