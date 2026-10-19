"""
Phase 5 - Experiment Definitions
============================================
Sarcasm & Irony Detection System using NLP

Defines and runs controlled experiments for TF-IDF + Feature Engineering.
Reuses Phase 4 training and evaluation code.

Three stages:
    Stage 1: Core TF-IDF experiments (TFIDF-01 to TFIDF-08)
    Stage 2: Feature set experiments (FEAT-01 to FEAT-05)
    Stage 3: Preprocessing comparison (PREP-01 to PREP-04)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from .vectorizer import create_tfidf_vectorizer, fit_transform_tfidf, transform_tfidf
from .feature_engineering import build_feature_set, combine_tfidf_features

# Reuse Phase 4 training functions
from src.classical.train_lr import train_logistic_regression
from src.classical.train_nb import train_naive_bayes
from src.classical.train_svm import train_linear_svm
from src.classical.evaluate import evaluate_model, CLASS_NAMES

logger = logging.getLogger(__name__)


# ============================================================
# STAGE 1: CORE TF-IDF EXPERIMENTS
# ============================================================

CORE_EXPERIMENTS = [
    {
        "id": "TFIDF-01", "preprocessing": "classical",
        "ngram_range": (1, 1), "model": "LogisticRegression",
        "hp_key": "C", "hp_values": [0.1, 1.0, 10.0],
        "sublinear_tf": True, "feature_set": "A",
    },
    {
        "id": "TFIDF-02", "preprocessing": "classical",
        "ngram_range": (1, 2), "model": "LogisticRegression",
        "hp_key": "C", "hp_values": [0.1, 1.0, 10.0],
        "sublinear_tf": True, "feature_set": "A",
    },
    {
        "id": "TFIDF-03", "preprocessing": "classical",
        "ngram_range": (1, 3), "model": "LogisticRegression",
        "hp_key": "C", "hp_values": [0.1, 1.0, 10.0],
        "sublinear_tf": True, "feature_set": "A",
    },
    {
        "id": "TFIDF-04", "preprocessing": "classical",
        "ngram_range": (1, 1), "model": "MultinomialNB",
        "hp_key": "alpha", "hp_values": [0.1, 0.5, 1.0],
        "sublinear_tf": True, "feature_set": "A",
    },
    {
        "id": "TFIDF-05", "preprocessing": "classical",
        "ngram_range": (1, 2), "model": "MultinomialNB",
        "hp_key": "alpha", "hp_values": [0.1, 0.5, 1.0],
        "sublinear_tf": True, "feature_set": "A",
    },
    {
        "id": "TFIDF-06", "preprocessing": "classical",
        "ngram_range": (1, 1), "model": "LinearSVM",
        "hp_key": "C", "hp_values": [0.1, 1.0, 10.0],
        "sublinear_tf": True, "feature_set": "A",
    },
    {
        "id": "TFIDF-07", "preprocessing": "classical",
        "ngram_range": (1, 2), "model": "LinearSVM",
        "hp_key": "C", "hp_values": [0.1, 1.0, 10.0],
        "sublinear_tf": True, "feature_set": "A",
    },
    {
        "id": "TFIDF-08", "preprocessing": "classical",
        "ngram_range": (1, 3), "model": "LinearSVM",
        "hp_key": "C", "hp_values": [0.1, 1.0, 10.0],
        "sublinear_tf": True, "feature_set": "A",
    },
]


def _train_model(model_name, X_train, y_train, hp_key, hp_val):
    """Train a model with specific hyperparameters."""
    if model_name == "LogisticRegression":
        return train_logistic_regression(X_train, y_train, C=hp_val)
    elif model_name == "MultinomialNB":
        return train_naive_bayes(X_train, y_train, alpha=hp_val)
    elif model_name == "LinearSVM":
        return train_linear_svm(X_train, y_train, C=hp_val)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_single_experiment(
    exp_config: dict,
    X_train, y_train,
    X_val, y_val,
    vocab_size: int,
    total_features: int,
) -> Tuple[Dict, object, float]:
    """
    Run a single experiment with hyperparameter selection on validation Macro F1.

    Returns:
        Tuple of (result_row, best_model, best_hp_value)
    """
    exp_id = exp_config["id"]
    model_name = exp_config["model"]
    hp_key = exp_config["hp_key"]
    hp_values = exp_config["hp_values"]

    logger.info(f"\n--- Experiment {exp_id}: {model_name} ---")

    best_f1 = -1
    best_model = None
    best_result = None
    best_hp = None
    best_time = 0

    for hp_val in hp_values:
        logger.info(f"  Trying {hp_key}={hp_val}...")
        model, train_time = _train_model(model_name, X_train, y_train, hp_key, hp_val)
        metrics = evaluate_model(model, X_val, y_val)

        logger.info(
            f"  {hp_key}={hp_val}: acc={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_model = model
            best_result = metrics
            best_hp = hp_val
            best_time = train_time

    logger.info(f"  Best: {hp_key}={best_hp}, macro_f1={best_f1:.4f}")

    result_row = _format_result(
        exp_config, best_result, best_hp, best_time, vocab_size, total_features
    )

    return result_row, best_model, best_hp


def _format_result(
    exp_config, metrics, best_hp, train_time, vocab_size, total_features
):
    """Format a single experiment result for CSV."""
    return {
        "experiment_id": exp_config["id"],
        "split": "validation",
        "preprocessing_variant": exp_config["preprocessing"],
        "feature_set": exp_config.get("feature_set", "A"),
        "ngram_range": str(exp_config["ngram_range"]),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": exp_config.get("sublinear_tf", True),
        "model": exp_config["model"],
        "hyperparameters": str({exp_config["hp_key"]: best_hp}),
        "vocabulary_size": vocab_size,
        "total_features": total_features,
        "accuracy": round(metrics["accuracy"], 4),
        "macro_precision": round(metrics["macro_precision"], 4),
        "macro_recall": round(metrics["macro_recall"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "regular_f1": round(metrics["per_class"]["Regular"]["f1"], 4),
        "sarcasm_f1": round(metrics["per_class"]["Sarcasm"]["f1"], 4),
        "irony_f1": round(metrics["per_class"]["Irony"]["f1"], 4),
        "figurative_f1": round(metrics["per_class"]["Figurative"]["f1"], 4),
        "training_time": round(train_time, 2),
        "notes": "",
    }


def determine_best_ngram(results: List[Dict]) -> tuple:
    """Determine best n-gram from Stage 1 results (LR experiments)."""
    lr_results = [r for r in results if r["model"] == "LogisticRegression"]
    if not lr_results:
        return (1, 2)
    best = max(lr_results, key=lambda r: r["macro_f1"])
    parts = best["ngram_range"].strip("()").split(",")
    return (int(parts[0].strip()), int(parts[1].strip()))


def determine_best_model(results: List[Dict]) -> str:
    """Determine best model from Stage 1 results."""
    best = max(results, key=lambda r: r["macro_f1"])
    return best["model"]
