"""Transaction logs preserve provider reasoning values without coercion."""

import pytest

from rotator_library.transaction_logger import TransactionLogger


@pytest.mark.parametrize("value", ["plain reasoning", {"summary": "structured"}, ["step"], None])
@pytest.mark.parametrize("location", ["top", "message", "reasoning_content"])
def test_transaction_reasoning_preserves_provider_value(value, location):
    if location == "top":
        response = {"reasoning": value}
    else:
        field = "reasoning" if location == "message" else "reasoning_content"
        response = {"choices": [{"message": {field: value}}]}
    logger = object.__new__(TransactionLogger)
    assert logger._extract_reasoning(response) is value
