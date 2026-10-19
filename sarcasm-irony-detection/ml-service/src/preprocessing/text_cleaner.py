"""
Phase 3 - Text Validation & Cleaning
============================================
Sarcasm & Irony Detection System using NLP

Handles text validation, safe type coercion,
and basic text cleaning before NLP preprocessing.

Since Phase 1 already removed invalid records,
unexpected invalid input is logged rather than silently ignored.
"""

import logging
import math

logger = logging.getLogger(__name__)

# Counter for tracking unexpected invalid inputs
_invalid_input_count = 0


def validate_text(text) -> str:
    """
    Validate and safely normalize text input.
    
    Handles:
        - None
        - NaN (float)
        - Empty strings
        - Whitespace-only strings
        - Non-string input (int, float, list, etc.)
    
    Returns:
        str: A safe string representation. Empty string for invalid inputs.
    
    Note:
        Since Phase 1 should have removed invalid records, any unexpected
        invalid input is logged as a warning.
    """
    global _invalid_input_count
    
    # Handle None
    if text is None:
        _invalid_input_count += 1
        logger.warning(f"Unexpected None input (count: {_invalid_input_count})")
        return ""
    
    # Handle NaN (float NaN)
    if isinstance(text, float) and math.isnan(text):
        _invalid_input_count += 1
        logger.warning(f"Unexpected NaN input (count: {_invalid_input_count})")
        return ""
    
    # Handle non-string types
    if not isinstance(text, str):
        _invalid_input_count += 1
        logger.warning(
            f"Unexpected non-string input type {type(text).__name__}: "
            f"{repr(text)[:100]} (count: {_invalid_input_count})"
        )
        return str(text)
    
    # Handle empty or whitespace-only strings
    if not text.strip():
        _invalid_input_count += 1
        logger.warning(f"Unexpected empty/whitespace-only input (count: {_invalid_input_count})")
        return ""
    
    return text


def get_invalid_input_count() -> int:
    """Return the count of unexpected invalid inputs encountered."""
    return _invalid_input_count


def reset_invalid_input_count() -> None:
    """Reset the invalid input counter."""
    global _invalid_input_count
    _invalid_input_count = 0
