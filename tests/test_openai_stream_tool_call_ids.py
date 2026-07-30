# ruff: noqa: F401

from openai_stream_tool_call_wrapper_identity_cases import (
    test_safe_streaming_wrapper_assigns_stable_ids_when_tool_call_ids_are_null,
)
from openai_stream_tool_call_wrapper_identity_cases import (
    test_safe_streaming_wrapper_serializes_multi_index_tool_state_without_null_names,
)
from openai_stream_tool_call_wrapper_lifecycle_cases import (
    test_safe_streaming_wrapper_releases_key_before_done,
)
from openai_stream_tool_call_wrapper_lifecycle_cases import (
    test_safe_streaming_wrapper_releases_key_after_upstream_exception,
)
from openai_stream_tool_call_wrapper_lifecycle_cases import (
    test_safe_streaming_wrapper_releases_key_after_consumer_cancellation,
)
from openai_stream_tool_call_normalizer_basic_cases import (
    test_normalizer_reuses_synthetic_id_for_continuation_delta,
)
from openai_stream_tool_call_normalizer_basic_cases import (
    test_normalizer_assigns_distinct_ids_to_distinct_tool_indexes,
)
from openai_stream_tool_call_normalizer_basic_cases import (
    test_normalizer_preserves_existing_tool_call_id,
)
from openai_stream_tool_call_normalizer_basic_cases import (
    test_normalizer_leaves_chunk_without_tool_calls_unchanged,
)
from openai_stream_tool_call_normalizer_basic_cases import (
    test_normalizer_replaces_null_top_level_chunk_id,
)
from openai_stream_tool_call_normalizer_fields_cases import (
    test_normalizer_drops_null_function_name_on_open,
)
from openai_stream_tool_call_normalizer_fields_cases import (
    test_normalizer_sticky_function_name_across_null_deltas,
)
from openai_stream_tool_call_normalizer_fields_cases import (
    test_normalizer_coerces_string_tool_index,
)
from openai_stream_tool_call_normalizer_fields_cases import (
    test_normalizer_null_arguments_become_empty_string,
)
