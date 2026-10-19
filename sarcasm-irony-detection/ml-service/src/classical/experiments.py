"""
Phase 4 - Experiment Runner
============================================
Sarcasm & Irony Detection System using NLP

Defines and runs the controlled experiment matrix for
BoW + Classical ML baselines.

Experiment Matrix:
    Core (7): BOW-01 to BOW-07 (minimal preprocessing, varying n-grams and models)
    Preprocessing comparison (9): BOW-08 to BOW-16 (varying preprocessing, best n-gram)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from .bow import create_bow_vectorizer, fit_transform_bow, transform_bow, get_feature_info
from .train_lr import train_logistic_regression
from .train_nb import train_naive_bayes
from .train_svm import train_linear_svm
from .evaluate import evaluate_model, format_results_row

logger = logging.getLogger(__name__)


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

CORE_EXPERIMENTS = [
    {
        "id": "BOW-01",
        "preprocessing": "minimal",
        "ngram_range": (1, 1),
        "model": "LogisticRegression",
        "hyperparam_key": "C",
        "hyperparam_values": [0.1, 1.0, 10.0],
    },
    {
        "id": "BOW-02",
        "preprocessing": "minimal",
        "ngram_range": (1, 2),
        "model": "LogisticRegression",
        "hyperparam_key": "C",
        "hyperparam_values": [0.1, 1.0, 10.0],
    },
    {
        "id": "BOW-03",
        "preprocessing": "minimal",
        "ngram_range": (1, 3),
        "model": "LogisticRegression",
        "hyperparam_key": "C",
        "hyperparam_values": [0.1, 1.0, 10.0],
    },
    {
        "id": "BOW-04",
        "preprocessing": "minimal",
        "ngram_range": (1, 1),
        "model": "MultinomialNB",
        "hyperparam_key": "alpha",
        "hyperparam_values": [0.1, 0.5, 1.0],
    },
    {
        "id": "BOW-05",
        "preprocessing": "minimal",
        "ngram_range": (1, 2),
        "model": "MultinomialNB",
        "hyperparam_key": "alpha",
        "hyperparam_values": [0.1, 0.5, 1.0],
    },
    {
        "id": "BOW-06",
        "preprocessing": "minimal",
        "ngram_range": (1, 1),
        "model": "LinearSVM",
        "hyperparam_key": "C",
        "hyperparam_values": [0.1, 1.0, 10.0],
    },
    {
        "id": "BOW-07",
        "preprocessing": "minimal",
        "ngram_range": (1, 2),
        "model": "LinearSVM",
        "hyperparam_key": "C",
        "hyperparam_values": [0.1, 1.0, 10.0],
    },
]

PREPROCESSING_VARIANTS_TO_TEST = ["lowercase", "lemmatized", "classical"]

MODEL_CONFIGS_FOR_PREPROC = [
    {"model": "LogisticRegression", "hyperparam_key": "C", "hyperparam_values": [0.1, 1.0, 10.0]},
    {"model": "MultinomialNB", "hyperparam_key": "alpha", "hyperparam_values": [0.1, 0.5, 1.0]},
    {"model": "LinearSVM", "hyperparam_key": "C", "hyperparam_values": [0.1, 1.0, 10.0]},
]


def _train_model(model_name, X_train, y_train, hyperparam_key, hyperparam_value):
    """Train a model with specific hyperparameters."""
    if model_name == "LogisticRegression":
        return train_logistic_regression(X_train, y_train, C=hyperparam_value)
    elif model_name == "MultinomialNB":
        return train_naive_bayes(X_train, y_train, alpha=hyperparam_value)
    elif model_name == "LinearSVM":
        return train_linear_svm(X_train, y_train, C=hyperparam_value)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_single_experiment(
    experiment_config: dict,
    X_train,
    y_train,
    X_val,
    y_val,
    vocab_size: int,
    preprocessing_name: str,
) -> Tuple[Dict, object, float]:
    """
    Run a single experiment with hyperparameter selection on validation set.
    
    Tries all hyperparameter values, selects the best on validation Macro F1.
    
    Returns:
        Tuple of (best_result_row, best_model, best_training_time)
    """
    exp_id = experiment_config["id"]
    model_name = experiment_config["model"]
    ngram_range = experiment_config["ngram_range"]
    hp_key = experiment_config["hyperparam_key"]
    hp_values = experiment_config["hyperparam_values"]
    
    logger.info(f"\n--- Experiment {exp_id}: {model_name} ngram={ngram_range} ---")
    
    best_f1 = -1
    best_model = None
    best_result = None
    best_hp = None
    best_time = 0
    
    for hp_val in hp_values:
        logger.info(f"  Trying {hp_key}={hp_val}...")
        
        model, train_time = _train_model(
            model_name, X_train, y_train, hp_key, hp_val
        )
        
        metrics = evaluate_model(model, X_val, y_val)
        
        logger.info(
            f"  {hp_key}={hp_val}: "
            f"acc={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
        
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_model = model
            best_result = metrics
            best_hp = hp_val
            best_time = train_time
    
    logger.info(f"  Best: {hp_key}={best_hp}, macro_f1={best_f1:.4f}")
    
    result_row = format_results_row(
        experiment_id=exp_id,
        preprocessing=preprocessing_name,
        vectorizer_type="CountVectorizer",
        ngram_range=ngram_range,
        model_name=model_name,
        hyperparams={hp_key: best_hp},
        metrics=best_result,
        training_time=best_time,
        vocab_size=vocab_size,
        split="validation",
    )
    
    return result_row, best_model, best_time


def determine_best_ngram(results: List[Dict]) -> tuple:
    """
    Determine the best n-gram range from core LR experiments.
    
    Compares BOW-01, BOW-02, BOW-03 on validation Macro F1.
    """
    lr_results = [r for r in results if r["model"] == "LogisticRegression"]
    if not lr_results:
        return (1, 2)  # Default fallback
    
    best = max(lr_results, key=lambda r: r["macro_f1"])
    
    # Parse ngram_range string back to tuple
    ngram_str = best["ngram_range"]
    # Format is "(1, 2)" — parse it
    parts = ngram_str.strip("()").split(",")
    best_ngram = (int(parts[0].strip()), int(parts[1].strip()))
    
    logger.info(f"Best n-gram range from core experiments: {best_ngram}")
    return best_ngram


def generate_preprocessing_experiments(
    best_ngram: tuple,
    start_id: int = 8,
) -> List[Dict]:
    """
    Generate preprocessing comparison experiments using the best n-gram range.
    """
    experiments = []
    exp_id = start_id
    
    for preproc in PREPROCESSING_VARIANTS_TO_TEST:
        for model_cfg in MODEL_CONFIGS_FOR_PREPROC:
            experiments.append({
                "id": f"BOW-{exp_id:02d}",
                "preprocessing": preproc,
                "ngram_range": best_ngram,
                "model": model_cfg["model"],
                "hyperparam_key": model_cfg["hyperparam_key"],
                "hyperparam_values": model_cfg["hyperparam_values"],
            })
            exp_id += 1
    
    return experiments
