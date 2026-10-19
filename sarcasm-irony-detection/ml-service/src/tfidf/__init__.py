"""
TF-IDF Package
============================================
Sarcasm & Irony Detection System using NLP

Phase 5: TF-IDF + N-grams + Feature Engineering
"""

from .vectorizer import create_tfidf_vectorizer, fit_transform_tfidf, transform_tfidf
from .feature_engineering import build_feature_set, combine_tfidf_features

__all__ = [
    "create_tfidf_vectorizer",
    "fit_transform_tfidf",
    "transform_tfidf",
    "build_feature_set",
    "combine_tfidf_features",
]
