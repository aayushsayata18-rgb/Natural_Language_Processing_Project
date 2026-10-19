"""
Inference Preprocessing Pipeline for Sarcasm & Irony Detection.
Strictly implements the exact text normalization used during training in Phase 3/4:
- Input validation (rejects None, non-string, and empty input)
- URL normalization -> ' url '
- User mention normalization -> ' user_mention '
- Lowercasing
- Whitespace stripping and multi-space collapsing
- Configurable maximum length truncation
"""

import re
from typing import List
from .exceptions import InvalidInputError


class InferenceTextPreprocessor:
    """
    Lightweight, stateless inference text preprocessor.
    Ensures 100% fidelity to the feature representation learned in Phase 4.
    """

    URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    MENTION_REGEX = re.compile(r"@\w+", re.UNICODE)
    MULTI_SPACE_REGEX = re.compile(r"\s+")

    def __init__(self, max_length_chars: int = 5000):
        self.max_length_chars = max_length_chars

    def validate_and_clean(self, text: str) -> str:
        """
        Validates input type and non-emptiness, then normalizes text.

        Raises:
            InvalidInputError: If text is None, not a string, or empty.
        """
        if text is None:
            raise InvalidInputError("Text input cannot be None.")
        if not isinstance(text, str):
            raise InvalidInputError(f"Text input must be a string, got {type(text).__name__}.")

        stripped = text.strip()
        if not stripped:
            raise InvalidInputError("Text input cannot be empty.")

        # Apply maximum length truncation policy
        if len(stripped) > self.max_length_chars:
            stripped = stripped[: self.max_length_chars]

        # Normalization steps
        t = self.URL_REGEX.sub(" url ", stripped)
        t = self.MENTION_REGEX.sub(" user_mention ", t)
        t = t.lower()
        t = self.MULTI_SPACE_REGEX.sub(" ", t).strip()

        if not t:
            raise InvalidInputError("Text input became empty after normalization.")

        return t

    def validate_and_clean_batch(self, texts: List[str]) -> List[str]:
        """
        Validates and cleans a batch of input texts.

        Raises:
            InvalidInputError: If texts is not a list, is empty, or any item is invalid.
        """
        if texts is None:
            raise InvalidInputError("Batch input cannot be None.")
        if not isinstance(texts, list):
            raise InvalidInputError(f"Batch input must be a list, got {type(texts).__name__}.")
        if len(texts) == 0:
            raise InvalidInputError("Batch input list cannot be empty.")

        cleaned = []
        for i, item in enumerate(texts):
            try:
                cleaned.append(self.validate_and_clean(item))
            except InvalidInputError as e:
                raise InvalidInputError(f"Validation failed at batch index {i}: {str(e)}")

        return cleaned
