"""
Preprocessing Package
============================================
Sarcasm & Irony Detection System using NLP

Phase 1: Dataset validation and cleaning
Phase 3: NLP text preprocessing pipeline
"""

# Phase 3 - Preprocessing pipeline API
from .config import (
    PreprocessingConfig,
    get_all_variants,
    get_variant,
    variant_a_minimal,
    variant_b_lowercase,
    variant_c_lemmatized,
    variant_d_stopword,
    variant_e_stemmed,
    variant_f_classical,
)

from .text_cleaner import validate_text
from .social_media import (
    normalize_urls,
    normalize_mentions,
    handle_hashtags,
    handle_emojis,
    extract_social_media_features,
)
from .normalization import (
    normalize_whitespace,
    normalize_case,
    normalize_punctuation,
    normalize_repeated_chars,
    remove_stopwords_from_tokens,
    extract_text_features,
    NEGATION_WORDS,
)
from .tokenizer import tokenize_text, tokens_to_string
from .lemmatization import lemmatize_tokens
from .stemming import stem_tokens
from .pipeline import PreprocessingPipeline

__all__ = [
    # Config
    "PreprocessingConfig",
    "get_all_variants",
    "get_variant",
    "variant_a_minimal",
    "variant_b_lowercase",
    "variant_c_lemmatized",
    "variant_d_stopword",
    "variant_e_stemmed",
    "variant_f_classical",
    # Text cleaning
    "validate_text",
    # Social media
    "normalize_urls",
    "normalize_mentions",
    "handle_hashtags",
    "handle_emojis",
    "extract_social_media_features",
    # Normalization
    "normalize_whitespace",
    "normalize_case",
    "normalize_punctuation",
    "normalize_repeated_chars",
    "remove_stopwords_from_tokens",
    "extract_text_features",
    "NEGATION_WORDS",
    # Tokenization
    "tokenize_text",
    "tokens_to_string",
    # Morphological
    "lemmatize_tokens",
    "stem_tokens",
    # Pipeline
    "PreprocessingPipeline",
]
