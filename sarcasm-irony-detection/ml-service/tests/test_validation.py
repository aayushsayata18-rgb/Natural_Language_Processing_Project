"""
Input validation and error handling tests for Phase 10 inference.
"""

import pytest
from src.inference import (
    SarcasmIronyPredictor,
    InferenceTextPreprocessor,
    InvalidInputError,
)


@pytest.fixture(scope="module")
def predictor():
    return SarcasmIronyPredictor()


@pytest.fixture(scope="module")
def preprocessor():
    return InferenceTextPreprocessor(max_length_chars=5000)


def test_empty_string_rejected(predictor):
    """Test that empty string raises InvalidInputError with descriptive message."""
    with pytest.raises(InvalidInputError) as exc_info:
        predictor.predict("")
    assert "Text input cannot be empty." in str(exc_info.value)


def test_whitespace_only_rejected(predictor):
    """Test that whitespace-only string raises InvalidInputError."""
    with pytest.raises(InvalidInputError) as exc_info:
        predictor.predict("     \t \n   ")
    assert "Text input cannot be empty." in str(exc_info.value)


def test_none_input_rejected(predictor):
    """Test that None input raises InvalidInputError."""
    with pytest.raises(InvalidInputError) as exc_info:
        predictor.predict(None)
    assert "Text input cannot be None." in str(exc_info.value)


@pytest.mark.parametrize("invalid_val", [123, 45.67, ["a", "b"], {"text": "hello"}, True, b"binary"])
def test_invalid_types_rejected(predictor, invalid_val):
    """Test that non-string data types are safely rejected."""
    with pytest.raises(InvalidInputError) as exc_info:
        predictor.predict(invalid_val)
    assert "Text input must be a string" in str(exc_info.value)


def test_empty_batch_rejected(predictor):
    """Test that an empty batch list raises InvalidInputError."""
    with pytest.raises(InvalidInputError) as exc_info:
        predictor.predict_batch([])
    assert "Batch input list cannot be empty." in str(exc_info.value)


def test_batch_with_invalid_item_rejected(predictor):
    """Test that batch containing invalid item raises with index specified."""
    with pytest.raises(InvalidInputError) as exc_info:
        predictor.predict_batch(["Valid text 1", "", "Valid text 3"])
    assert "Validation failed at batch index 1" in str(exc_info.value)

    with pytest.raises(InvalidInputError) as exc_info:
        predictor.predict_batch(["Valid text 1", None])
    assert "Validation failed at batch index 1" in str(exc_info.value)


def test_excessively_long_text_truncated_safely(predictor):
    """Test that excessively long input is truncated safely without crashing."""
    long_text = "This is a very long text repeated. " * 300  # > 10,000 chars
    res = predictor.predict(long_text)
    assert res is not None
    assert res.predicted_class in {"Regular", "Sarcasm", "Irony", "Figurative"}
    assert 0.0 <= res.confidence <= 1.0


def test_short_text_handled_safely(predictor):
    """Test that minimal single-word or short texts are processed cleanly."""
    for short in ["Ok.", "Fine.", "Sure!", "Whatever..."]:
        res = predictor.predict(short)
        assert res.predicted_class in {"Regular", "Sarcasm", "Irony", "Figurative"}
