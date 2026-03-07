# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/openai_chatgpt_provider.py

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import litellm
from litellm.exceptions import AuthenticationError, RateLimitError

from .openai_chatgpt_auth_base import OpenAIChatGPTAuthBase
from .provider_interface import ProviderInterface
from ..model_definitions import ModelDefinitions
from ..timeout_config import TimeoutConfig
from ..transaction_logger import ProviderLogger

AVAILABLE_MODELS = [
    "gpt-5.4",  # Only gpt-5.4 for now
]

CODEX_RESPONSES_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"


class OpenAIChatGPTProvider(OpenAIChatGPTAuthBase, ProviderInterface):
    skip_cost_calculation = True

    # Sequential rotation: stick to one account until exhausted/failed.
    default_rotation_mode: str = "sequential"

    provider_env_name: str = "openai_chatgpt"

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        models = []
        env_var_ids = set()

        static_models = self.model_definitions.get_all_provider_models("openai_chatgpt")
        if static_models:
            for model in static_models:
                model_name = model.split("/")[-1] if "/" in model else model
                model_id = self.model_definitions.get_model_id(
                    "openai_chatgpt", model_name
                )
                if model_id not in AVAILABLE_MODELS:
                    continue
                models.append(model)
                if model_id:
                    env_var_ids.add(model_id)

        for model_id in AVAILABLE_MODELS:
            if model_id not in env_var_ids:
                models.append(f"openai_chatgpt/{model_id}")
                env_var_ids.add(model_id)

        return models

    def _to_codex_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI chat-format messages to a simple Codex responses input format.
        """
        codex_input: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                codex_input.append(
                    {
                        "role": role,
                        "content": [{"type": "input_text", "text": content}],
                    }
                )
            elif isinstance(content, list):
                parts: List[Dict[str, Any]] = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(
                                {
                                    "type": "input_text",
                                    "text": item.get("text", ""),
                                }
                            )
                        elif item.get("type") == "input_text":
                            parts.append(item)

                if parts:
                    codex_input.append({"role": role, "content": parts})

        if not codex_input:
            codex_input.append(
                {"role": "user", "content": [{"type": "input_text", "text": ""}]}
            )

        return codex_input

    def _build_codex_payload(
        self, model: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        model_name = model.split("/")[-1]
        model_id = self.model_definitions.get_model_id("openai_chatgpt", model_name)
        resolved_model = model_id or model_name
        if resolved_model not in AVAILABLE_MODELS:
            supported_models = ", ".join(AVAILABLE_MODELS)
            raise ValueError(
                f"Unsupported openai_chatgpt model '{resolved_model}'. Supported models: {supported_models}"
            )
        messages = kwargs.get("messages", []) or []

        payload: Dict[str, Any] = {
            "model": resolved_model,
            "input": self._to_codex_input(messages),
            "stream": True,
        }

        if "temperature" in kwargs and kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs and kwargs.get("max_tokens") is not None:
            payload["max_output_tokens"] = kwargs["max_tokens"]
        if "tools" in kwargs and kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs and kwargs.get("tool_choice") is not None:
            payload["tool_choice"] = kwargs["tool_choice"]
        if "response_format" in kwargs and kwargs.get("response_format") is not None:
            payload["response_format"] = kwargs["response_format"]

        return payload

    def _normalize_response_text(self, raw_chunk: Dict[str, Any]) -> Optional[str]:
        """
        Extract best-effort content text from non-standard codex chunks.
        """
        if not isinstance(raw_chunk, dict):
            return None

        if isinstance(raw_chunk.get("output_text"), str):
            return raw_chunk["output_text"]

        delta = raw_chunk.get("delta")
        if isinstance(delta, str):
            return delta

        output = raw_chunk.get("output")
        if isinstance(output, list):
            text_parts: List[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str):
                                text_parts.append(text)
            if text_parts:
                return "".join(text_parts)

        return None

    def _convert_chunk_to_openai(
        self,
        raw_chunk: Dict[str, Any],
        model: str,
        chunk_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert ChatGPT Codex stream chunk into OpenAI-compatible chat chunk.
        """
        text = self._normalize_response_text(raw_chunk)
        finish_reason = raw_chunk.get("finish_reason")

        if text is None and finish_reason is None:
            return None

        return {
            "id": raw_chunk.get("id", chunk_id or f"chatcmpl-chatgpt-{time.time()}"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text} if text is not None else {},
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _stream_to_completion_response(
        self, chunks: List[litellm.ModelResponse]
    ) -> litellm.ModelResponse:
        if not chunks:
            raise ValueError("No chunks provided for reassembly")

        first_chunk = chunks[0]
        content_parts: List[str] = []
        finish_reason = None
        usage_data = None

        for chunk in chunks:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.get("delta", {})
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    content_parts.append(content)

            if choice.get("finish_reason"):
                finish_reason = choice.get("finish_reason")

            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage

        final_response = {
            "id": first_chunk.id,
            "object": "chat.completion",
            "created": first_chunk.created,
            "model": first_chunk.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "".join(content_parts) if content_parts else None,
                    },
                    "finish_reason": finish_reason or "stop",
                }
            ],
            "usage": usage_data,
        }

        return litellm.ModelResponse(**final_response)

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential_identifier = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)
        model = kwargs["model"]

        file_logger = ProviderLogger(transaction_context)

        async def make_request():
            creds = await self._load_credentials(credential_identifier)
            if self._is_token_expired(creds):
                creds = await self._refresh_token(credential_identifier)

            access_token = creds.get("access_token")
            account_id = creds.get("account_id") or await self.get_account_id(
                credential_identifier
            )

            if not access_token:
                raise AuthenticationError(
                    message="Missing ChatGPT access token",
                    llm_provider="openai_chatgpt",
                    model=model,
                )
            if not account_id:
                raise AuthenticationError(
                    message="Missing ChatGPT account_id",
                    llm_provider="openai_chatgpt",
                    model=model,
                )

            payload = self._build_codex_payload(model, kwargs)
            file_logger.log_request(payload)

            headers = {
                "Authorization": f"Bearer {access_token}",
                "chatgpt-account-id": str(account_id),
                "OpenAI-Beta": "responses=experimental",
                "originator": "pi",
                "User-Agent": "pi (darwin 24.0.0; arm64)",
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            }

            return client.stream(
                "POST",
                CODEX_RESPONSES_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=TimeoutConfig.streaming(),
            )

        async def stream_handler(response_stream, attempt=1):
            async with response_stream as response:
                if response.status_code >= 400:
                    error_text_bytes = await response.aread()
                    error_text = (
                        error_text_bytes.decode("utf-8", errors="replace")
                        if isinstance(error_text_bytes, (bytes, bytearray))
                        else str(error_text_bytes)
                    )

                    if response.status_code == 401 and attempt == 1:
                        await self._refresh_token(credential_identifier, force=True)
                        retry_stream = await make_request()
                        async for chunk in stream_handler(retry_stream, attempt=2):
                            yield chunk
                        return

                    if response.status_code == 429:
                        raise RateLimitError(
                            f"ChatGPT rate limit exceeded: {error_text}",
                            llm_provider="openai_chatgpt",
                            model=model,
                            response=response,
                        )

                    if response.status_code in (401, 403):
                        raise AuthenticationError(
                            message=f"ChatGPT auth failed: {error_text}",
                            llm_provider="openai_chatgpt",
                            model=model,
                            response=response,
                        )

                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {error_text}",
                        request=response.request,
                        response=response,
                    )

                async for line in response.aiter_lines():
                    file_logger.log_response_chunk(line)

                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            raw_chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        openai_chunk = self._convert_chunk_to_openai(raw_chunk, model)
                        if openai_chunk:
                            yield litellm.ModelResponse(**openai_chunk)

                    elif line.startswith("{"):
                        try:
                            raw_chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        openai_chunk = self._convert_chunk_to_openai(raw_chunk, model)
                        if openai_chunk:
                            yield litellm.ModelResponse(**openai_chunk)

        async def logging_stream_wrapper():
            openai_chunks = []
            try:
                async for chunk in stream_handler(await make_request()):
                    openai_chunks.append(chunk)
                    yield chunk
            finally:
                if openai_chunks:
                    final_response = self._stream_to_completion_response(openai_chunks)
                    file_logger.log_final_response(final_response.dict())

        if kwargs.get("stream"):
            return logging_stream_wrapper()

        chunks = [chunk async for chunk in logging_stream_wrapper()]
        return self._stream_to_completion_response(chunks)
