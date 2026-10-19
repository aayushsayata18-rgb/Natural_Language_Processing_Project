"""
Phase 4 - Model Evaluation
============================================
Sarcasm & Irony Detection System using NLP

Evaluation utilities for classical ML models:
- Accuracy, Precision, Recall, F1 (macro and per-class)
- Confusion matrices
- Classification reports
- Error analysis
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Regular", "Sarcasm", "Irony", "Figurative"]


def evaluate_model(
    model,
    X,
    y,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate a trained model and return comprehensive metrics.
    
    Computes:
        - Accuracy
        - Macro Precision, Recall, F1
        - Per-class Precision, Recall, F1, Support
        - Confusion matrix
    
    Args:
        model: Trained classifier.
        X: Feature matrix.
        y: True labels.
        class_names: List of class names for labeling.
    
    Returns:
        Dict containing all metrics and confusion matrix.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    macro_precision = precision_score(y, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    
    # Per-class metrics
    per_class_precision = precision_score(y, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Build per-class dict
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(np.sum(y == i)),
        }
    
    results = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
    }
    
    return results


def generate_classification_report(
    model,
    X,
    y,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Generate a full sklearn classification report as a dict.
    
    Args:
        model: Trained classifier.
        X: Feature matrix.
        y: True labels.
        class_names: Class label names.
    
    Returns:
        Dict from sklearn.metrics.classification_report with output_dict=True.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    y_pred = model.predict(X)
    
    report = classification_report(
        y, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    
    return report


def collect_error_examples(
    model,
    X,
    y,
    texts,
    class_names: Optional[List[str]] = None,
    n: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Collect misclassified examples for error analysis.
    
    Focuses on important confusion pairs:
        Sarcasm <-> Irony
        Sarcasm <-> Figurative
        Irony <-> Figurative
        Regular <-> Figurative
    
    Args:
        model: Trained classifier.
        X: Feature matrix.
        y: True labels (array-like).
        texts: Original text data (Series or list).
        class_names: Class label names.
        n: Maximum number of error examples to collect.
        random_state: Seed for reproducible sampling.
    
    Returns:
        DataFrame with columns: text, actual_label, predicted_label
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    y_pred = model.predict(X)
    y_arr = np.array(y)
    texts_arr = np.array(texts)
    
    # Find misclassified indices
    misclassified_mask = y_arr != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) == 0:
        logger.info("No misclassified examples found.")
        return pd.DataFrame(columns=["text", "actual_label", "predicted_label"])
    
    # Sample up to n examples
    rng = np.random.RandomState(random_state)
    if len(misclassified_indices) > n:
        sampled = rng.choice(misclassified_indices, size=n, replace=False)
    else:
        sampled = misclassified_indices
    
    errors = []
    for idx in sampled:
        errors.append({
            "text": texts_arr[idx],
            "actual_label": class_names[y_arr[idx]],
            "predicted_label": class_names[y_pred[idx]],
        })
    
    error_df = pd.DataFrame(errors)
    
    # Log confusion pair distribution
    if len(error_df) > 0:
        pairs = error_df.groupby(["actual_label", "predicted_label"]).size()
        logger.info(f"Error examples collected: {len(error_df)}")
        for (actual, predicted), count in pairs.items():
            logger.info(f"  {actual} -> {predicted}: {count}")
    
    return error_df


def format_results_row(
    experiment_id: str,
    preprocessing: str,
    vectorizer_type: str,
    ngram_range: tuple,
    model_name: str,
    hyperparams: dict,
    metrics: dict,
    training_time: float,
    vocab_size: int,
    split: str = "validation",
) -> Dict:
    """
    Format a single experiment result as a flat dict for CSV storage.
    """
    return {
        "experiment_id": experiment_id,
        "split": split,
        "preprocessing": preprocessing,
        "vectorizer": vectorizer_type,
        "ngram_range": str(ngram_range),
        "model": model_name,
        "hyperparameters": str(hyperparams),
        "accuracy": round(metrics["accuracy"], 4),
        "macro_precision": round(metrics["macro_precision"], 4),
        "macro_recall": round(metrics["macro_recall"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "regular_f1": round(metrics["per_class"]["Regular"]["f1"], 4),
        "sarcasm_f1": round(metrics["per_class"]["Sarcasm"]["f1"], 4),
        "irony_f1": round(metrics["per_class"]["Irony"]["f1"], 4),
        "figurative_f1": round(metrics["per_class"]["Figurative"]["f1"], 4),
        "training_time": round(training_time, 2),
        "vocab_size": vocab_size,
    }
