"""
Phase 4 - Bag of Words Vectorizer
============================================
Sarcasm & Irony Detection System using NLP

Creates and manages CountVectorizer for Bag of Words
feature extraction.

IMPORTANT: The vectorizer must be fitted ONLY on training data.
Validation and test data must use transform() only.
"""

import logging
from typing import Optional, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import spmatrix

logger = logging.getLogger(__name__)


def create_bow_vectorizer(
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: Optional[int] = 50000,
) -> CountVectorizer:
    """
    Create a CountVectorizer for Bag of Words representation.
    
    Args:
        ngram_range: Range of n-grams. (1,1)=unigram, (1,2)=uni+bi, (1,3)=uni+bi+tri
        min_df: Minimum document frequency. Ignore terms appearing in fewer documents.
        max_df: Maximum document frequency. Ignore terms appearing in more than this
                fraction of documents.
        max_features: Maximum vocabulary size. Safety cap to control dimensionality.
    
    Returns:
        CountVectorizer instance (not yet fitted).
    """
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    
    logger.info(
        f"Created CountVectorizer: ngram_range={ngram_range}, "
        f"min_df={min_df}, max_df={max_df}, max_features={max_features}"
    )
    
    return vectorizer


def fit_transform_bow(
    vectorizer: CountVectorizer,
    texts,
) -> spmatrix:
    """
    Fit the vectorizer on training texts and transform to sparse matrix.
    
    MUST be called only on training data to prevent leakage.
    
    Args:
        vectorizer: CountVectorizer instance.
        texts: Training text data (Series or list of strings).
    
    Returns:
        Sparse matrix of BoW features.
    """
    X = vectorizer.fit_transform(texts)
    vocab_size = len(vectorizer.vocabulary_)
    
    logger.info(
        f"BoW fitted: vocab_size={vocab_size}, "
        f"matrix_shape={X.shape}, "
        f"sparsity={1.0 - X.nnz / (X.shape[0] * X.shape[1]):.4f}"
    )
    
    return X


def transform_bow(
    vectorizer: CountVectorizer,
    texts,
) -> spmatrix:
    """
    Transform texts using an already-fitted vectorizer.
    
    For validation and test data — no fitting occurs.
    
    Args:
        vectorizer: Fitted CountVectorizer.
        texts: Text data to transform.
    
    Returns:
        Sparse matrix of BoW features.
    """
    X = vectorizer.transform(texts)
    logger.info(f"BoW transformed: matrix_shape={X.shape}")
    return X


def get_feature_info(vectorizer: CountVectorizer) -> dict:
    """
    Get information about the fitted vectorizer's feature space.
    
    Returns:
        dict with vocab_size, ngram_range, feature_names_count
    """
    return {
        "vocab_size": len(vectorizer.vocabulary_),
        "ngram_range": vectorizer.ngram_range,
        "n_features": len(vectorizer.get_feature_names_out()),
    }
