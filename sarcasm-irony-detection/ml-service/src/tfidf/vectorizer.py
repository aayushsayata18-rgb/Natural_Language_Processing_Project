"""
Phase 5 - TF-IDF Vectorizer
============================================
Sarcasm & Irony Detection System using NLP

TF-IDF vectorizer with configurable n-gram range,
sublinear TF scaling, and leakage-safe fit/transform.

IMPORTANT: The vectorizer must be fitted ONLY on training data.
"""

import logging
from typing import Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

logger = logging.getLogger(__name__)


def create_tfidf_vectorizer(
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: Optional[int] = 50000,
    sublinear_tf: bool = True,
    norm: str = "l2",
) -> TfidfVectorizer:
    """
    Create a TfidfVectorizer.

    Args:
        ngram_range: Range of n-grams.
        min_df: Minimum document frequency.
        max_df: Maximum document frequency fraction.
        max_features: Maximum vocabulary size (safety cap).
        sublinear_tf: If True, apply sublinear tf scaling (1 + log(tf)).
        norm: Normalization method ('l1', 'l2', or None).

    Returns:
        TfidfVectorizer instance (not yet fitted).
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        norm=norm,
    )

    logger.info(
        f"Created TfidfVectorizer: ngram={ngram_range}, "
        f"min_df={min_df}, max_df={max_df}, max_features={max_features}, "
        f"sublinear_tf={sublinear_tf}"
    )
    return vectorizer


def fit_transform_tfidf(vectorizer: TfidfVectorizer, texts) -> spmatrix:
    """
    Fit the vectorizer on training texts and transform.
    MUST be called only on training data.
    """
    X = vectorizer.fit_transform(texts)
    vocab_size = len(vectorizer.vocabulary_)
    sparsity = 1.0 - X.nnz / (X.shape[0] * X.shape[1]) if X.shape[1] > 0 else 0

    logger.info(
        f"TF-IDF fitted: vocab={vocab_size}, shape={X.shape}, sparsity={sparsity:.4f}"
    )
    return X


def transform_tfidf(vectorizer: TfidfVectorizer, texts) -> spmatrix:
    """Transform texts using an already-fitted vectorizer (no fitting)."""
    X = vectorizer.transform(texts)
    logger.info(f"TF-IDF transformed: shape={X.shape}")
    return X


def get_tfidf_info(vectorizer: TfidfVectorizer) -> dict:
    """Get info about the fitted vectorizer."""
    return {
        "vocab_size": len(vectorizer.vocabulary_),
        "ngram_range": vectorizer.ngram_range,
        "n_features": len(vectorizer.get_feature_names_out()),
        "sublinear_tf": vectorizer.sublinear_tf,
    }
