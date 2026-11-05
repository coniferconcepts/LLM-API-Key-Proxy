"""Fail closed after a streaming response has yielded a chunk."""

import json
import logging
from contextlib import aclosing

import litellm
from litellm.exceptions import APIConnectionError

from .error_handler import build_public_stream_error, classify_error

lib_logger = logging.getLogger("rotator_library")


async def fail_closed_stream(stream, credential, model, provider, usage_manager):
    yielded_chunk = False
    try:
        async with aclosing(stream):
            async for chunk in stream:
                yielded_chunk = True
                yield chunk
    except Exception as e:
        if not yielded_chunk:
            raise
        original_exc = getattr(e, "data", e)
        classified_error = classify_error(original_exc, provider=provider)
        await usage_manager.record_failure(
            credential,
            model,
            classified_error,
            increment_consecutive_failures=not isinstance(
                original_exc,
                (
                    APIConnectionError,
                    litellm.InternalServerError,
                    litellm.ServiceUnavailableError,
                ),
            ),
        )
        lib_logger.warning(
            "Streaming request failed after a chunk category=%s status=%s; terminating.",
            classified_error.error_type,
            classified_error.status_code,
        )
        yield f"data: {json.dumps(build_public_stream_error('internal_error'))}\n\n"
        yield "data: [DONE]\n\n"
