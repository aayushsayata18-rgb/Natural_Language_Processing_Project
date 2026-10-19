"""
Classical ML Package
============================================
Sarcasm & Irony Detection System using NLP

Phase 4: Bag of Words + Classical Machine Learning baselines
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
"""

from .bow import create_bow_vectorizer, fit_transform_bow, transform_bow
from .evaluate import evaluate_model, generate_classification_report, collect_error_examples

__all__ = [
    "create_bow_vectorizer",
    "fit_transform_bow",
    "transform_bow",
    "evaluate_model",
    "generate_classification_report",
    "collect_error_examples",
]
