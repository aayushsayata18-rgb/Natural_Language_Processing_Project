"""
Phase 4 - Linear SVM Trainer
============================================
Sarcasm & Irony Detection System using NLP

Linear Support Vector Machine for four-class text classification.

IMPORTANT: LinearSVC does not provide standard probability outputs.
Do NOT display fake probability/confidence scores for this model.
"""

import logging
import time

from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


def train_linear_svm(
    X_train,
    y_train,
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> tuple:
    """
    Train a Linear SVM classifier.
    
    Args:
        X_train: Training feature matrix (sparse BoW).
        y_train: Training labels.
        C: Regularization parameter.
        max_iter: Maximum iterations for convergence.
        random_state: Random seed for reproducibility.
    
    Returns:
        Tuple of (trained_model, training_time_seconds).
    
    Note:
        LinearSVC does not support predict_proba().
        Do not treat decision_function output as probabilities.
    """
    model = LinearSVC(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        dual="auto",
    )
    
    logger.info(f"Training Linear SVM: C={C}, max_iter={max_iter}")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logger.info(f"SVM training complete in {training_time:.2f}s")
    
    return model, training_time
