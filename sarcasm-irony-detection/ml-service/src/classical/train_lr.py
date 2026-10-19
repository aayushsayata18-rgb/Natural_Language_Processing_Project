"""
Phase 4 - Logistic Regression Trainer
============================================
Sarcasm & Irony Detection System using NLP

Multiclass Logistic Regression for four-class classification.
"""

import logging
import time

from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train,
    y_train,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> tuple:
    """
    Train a multiclass Logistic Regression classifier.
    
    Args:
        X_train: Training feature matrix (sparse BoW).
        y_train: Training labels.
        C: Regularization strength (inverse). Smaller = stronger regularization.
        max_iter: Maximum iterations for convergence.
        random_state: Random seed for reproducibility.
    
    Returns:
        Tuple of (trained_model, training_time_seconds).
    """
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    
    logger.info(f"Training Logistic Regression: C={C}, max_iter={max_iter}")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logger.info(f"LR training complete in {training_time:.2f}s")
    
    return model, training_time
