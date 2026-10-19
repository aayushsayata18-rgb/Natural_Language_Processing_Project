"""
Phase 4 - Multinomial Naive Bayes Trainer
============================================
Sarcasm & Irony Detection System using NLP

Multinomial Naive Bayes for four-class text classification.
"""

import logging
import time

from sklearn.naive_bayes import MultinomialNB

logger = logging.getLogger(__name__)


def train_naive_bayes(
    X_train,
    y_train,
    alpha: float = 1.0,
) -> tuple:
    """
    Train a Multinomial Naive Bayes classifier.
    
    Args:
        X_train: Training feature matrix (sparse BoW).
        y_train: Training labels.
        alpha: Additive (Laplace) smoothing parameter.
    
    Returns:
        Tuple of (trained_model, training_time_seconds).
    """
    model = MultinomialNB(alpha=alpha)
    
    logger.info(f"Training Multinomial NB: alpha={alpha}")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logger.info(f"NB training complete in {training_time:.2f}s")
    
    return model, training_time
