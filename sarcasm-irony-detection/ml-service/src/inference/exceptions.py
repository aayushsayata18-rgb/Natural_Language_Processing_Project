"""
Custom exceptions for the Sarcasm & Irony Detection Inference Service.
"""

class InferenceError(Exception):
    """Base exception for all inference errors."""
    pass


class ModelNotFoundError(InferenceError):
    """Raised when the serialized model artifact cannot be found."""
    pass


class TokenizerNotFoundError(InferenceError):
    """Raised when the serialized vectorizer or tokenizer artifact cannot be found."""
    pass


class ConfigurationError(InferenceError):
    """Raised when configuration or label mapping files are missing or malformed."""
    pass


class ArtifactValidationError(InferenceError):
    """Raised when artifact integrity verification or schema check fails."""
    pass


class InvalidInputError(InferenceError):
    """Raised when an inference input is malformed, empty, or of invalid type."""
    pass
