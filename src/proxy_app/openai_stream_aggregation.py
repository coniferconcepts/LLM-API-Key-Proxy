# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

"""Aggregate OpenAI streaming chunks for the final response log entry."""

from typing import Any


def aggregate_openai_chunks(
    response_chunks: list[dict], final_message: dict
) -> tuple[dict, dict | None, str | None]:
    aggregated_tool_calls: dict[int, dict[str, Any]] = {}
    usage_data = None
    finish_reason = None

    for chunk in response_chunks:
        if "choices" in chunk and chunk["choices"]:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            # Dynamically aggregate all fields from the delta
            for key, value in delta.items():
                if value is None:
                    continue

                if key == "content":
                    if "content" not in final_message:
                        final_message["content"] = ""
                    if value:
                        final_message["content"] += value

                elif key == "tool_calls":
                    for tc_chunk in value:
                        index = tc_chunk["index"]
                        if index not in aggregated_tool_calls:
                            aggregated_tool_calls[index] = {
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        # Ensure 'function' key exists for this index before accessing its sub-keys
                        if "function" not in aggregated_tool_calls[index]:
                            aggregated_tool_calls[index]["function"] = {
                                "name": "",
                                "arguments": "",
                            }
                        if tc_chunk.get("id"):
                            aggregated_tool_calls[index]["id"] = tc_chunk["id"]
                        if "function" in tc_chunk:
                            if "name" in tc_chunk["function"]:
                                if tc_chunk["function"]["name"] is not None:
                                    aggregated_tool_calls[index]["function"]["name"] += tc_chunk[
                                        "function"
                                    ]["name"]
                            if "arguments" in tc_chunk["function"]:
                                if tc_chunk["function"]["arguments"] is not None:
                                    aggregated_tool_calls[index]["function"][
                                        "arguments"
                                    ] += tc_chunk["function"]["arguments"]

                elif key == "function_call":
                    if "function_call" not in final_message:
                        final_message["function_call"] = {
                            "name": "",
                            "arguments": "",
                        }
                    if "name" in value:
                        if value["name"] is not None:
                            final_message["function_call"]["name"] += value["name"]
                    if "arguments" in value:
                        if value["arguments"] is not None:
                            final_message["function_call"]["arguments"] += value["arguments"]

                else:
                    if key == "role":
                        final_message[key] = value
                    elif key not in final_message:
                        final_message[key] = value
                    elif isinstance(final_message.get(key), str):
                        final_message[key] += value
                    else:
                        final_message[key] = value

            if "finish_reason" in choice and choice["finish_reason"]:
                finish_reason = choice["finish_reason"]

        if "usage" in chunk and chunk["usage"]:
            usage_data = chunk["usage"]

    return aggregated_tool_calls, usage_data, finish_reason
