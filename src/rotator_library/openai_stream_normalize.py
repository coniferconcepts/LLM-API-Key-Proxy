from __future__ import annotations

from typing import Any, TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def _as_plain_dict(value: Any) -> JsonObject | None:
    """Coerce pydantic/LiteLLM objects to plain dicts for SSE JSON."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else None
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else None
    return None


def _coerce_tool_index(raw: Any, fallback: int) -> int:
    if isinstance(raw, bool):
        return fallback
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
    return fallback


class OpenAIStreamNormalizer:
    """Normalize OpenAI stream chunks for strict OpenAI-compatible clients.

    OpenCode / AI SDK reject tool_call deltas where:
    - ``id`` is null/missing on open (Expected 'id' to be a string)
    - ``function.name`` is null (Expected 'function.name' to be a string)

    This normalizer keeps per-stream sticky ids and names so every delta for a
    tool index is client-safe, and strips null keys that would fail typeof checks.
    """

    def __init__(self) -> None:
        self._chunk_id: str | None = None
        self._tool_call_ids: dict[tuple[int, int], str] = {}
        self._tool_call_names: dict[tuple[int, int], str] = {}

    def normalize(self, chunk: Any) -> Any:
        plain = _as_plain_dict(chunk)
        if plain is None:
            return chunk
        return self._normalize_object(plain)

    def _normalize_object(self, chunk: JsonObject) -> JsonObject:
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
            plain_choice = _as_plain_dict(choice)
            if plain_choice is None:
                continue
            choices[choice_position] = plain_choice

            for container_key in ("delta", "message"):
                container = plain_choice.get(container_key)
                plain_container = _as_plain_dict(container)
                if plain_container is None:
                    continue
                plain_choice[container_key] = plain_container
                self._normalize_tool_calls(plain_container, choice_position)

        return chunk

    def _normalize_tool_calls(self, container: JsonObject, choice_position: int) -> None:
        tool_calls = container.get("tool_calls")
        if not isinstance(tool_calls, list):
            return

        for list_position, tool_call in enumerate(tool_calls):
            plain_tc = _as_plain_dict(tool_call)
            if plain_tc is None:
                continue
            tool_calls[list_position] = plain_tc

            tool_index = _coerce_tool_index(plain_tc.get("index"), list_position)
            plain_tc["index"] = tool_index
            state_key = (choice_position, tool_index)

            # --- id: always a non-empty string, stable per index ---
            stable_id = self._tool_call_ids.get(state_key)
            if stable_id is None:
                provided_id = plain_tc.get("id")
                if isinstance(provided_id, str) and provided_id:
                    stable_id = provided_id
                elif choice_position == 0:
                    stable_id = f"call_{tool_index}"
                else:
                    stable_id = f"call_{choice_position}_{tool_index}"
                self._tool_call_ids[state_key] = stable_id
            plain_tc["id"] = stable_id

            # --- type: only emit when a non-empty string ---
            tc_type = plain_tc.get("type")
            if tc_type is None:
                plain_tc.pop("type", None)
            elif not isinstance(tc_type, str) or not tc_type:
                plain_tc.pop("type", None)
            # if missing entirely on first open, default is optional; OpenAI uses "function"
            # only set default when function payload is present and type absent
            if "type" not in plain_tc and (
                "function" in plain_tc or self._tool_call_names.get(state_key)
            ):
                plain_tc["type"] = "function"

            # --- function.name / arguments ---
            function = plain_tc.get("function")
            plain_fn = _as_plain_dict(function) if function is not None else None
            if plain_fn is None and function is not None and not isinstance(function, dict):
                # unusable function payload; drop nullish function
                plain_tc.pop("function", None)
                continue
            if plain_fn is None:
                continue
            plain_tc["function"] = plain_fn

            provided_name = plain_fn.get("name")
            if isinstance(provided_name, str) and provided_name:
                self._tool_call_names[state_key] = provided_name
                plain_fn["name"] = provided_name
            else:
                # null / empty / non-string name: never emit null (OpenCode typeof check)
                sticky = self._tool_call_names.get(state_key)
                if sticky:
                    plain_fn["name"] = sticky
                else:
                    plain_fn.pop("name", None)

            # arguments: prefer string; coerce null to empty string for partial deltas
            if "arguments" in plain_fn:
                args = plain_fn["arguments"]
                if args is None:
                    plain_fn["arguments"] = ""
                elif not isinstance(args, str):
                    # leave non-string alone only if JSON-serializable primitive handled upstream
                    plain_fn["arguments"] = str(args)
