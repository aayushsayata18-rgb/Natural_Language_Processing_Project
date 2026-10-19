"""
Phase 5 - Feature Analysis
============================================
Sarcasm & Irony Detection System using NLP

Extract and analyze feature importance from trained
LR/SVM models. Identify top features per class.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_feature_importance(
    model,
    feature_names: list,
    class_names: list,
    top_n: int = 30,
) -> dict:
    """
    Extract top features per class from model coefficients.

    Works for LogisticRegression and LinearSVC which expose .coef_.

    Args:
        model: Trained classifier with .coef_ attribute.
        feature_names: List of feature names from vectorizer.
        class_names: List of class names.
        top_n: Number of top features per class.

    Returns:
        Dict mapping class_name -> DataFrame with (feature, coefficient).
    """
    if not hasattr(model, "coef_"):
        logger.warning(f"Model {type(model).__name__} has no coef_ attribute.")
        return {}

    coefs = model.coef_  # shape: (n_classes, n_features)
    feature_names = np.array(feature_names)

    results = {}
    for i, class_name in enumerate(class_names):
        class_coefs = coefs[i]
        top_indices = np.argsort(class_coefs)[-top_n:][::-1]

        df = pd.DataFrame({
            "feature": feature_names[top_indices],
            "coefficient": class_coefs[top_indices],
        })
        results[class_name] = df

    logger.info(f"Extracted top-{top_n} features for {len(class_names)} classes")
    return results


def save_feature_importance(
    importance_dict: dict,
    output_dir: str,
) -> None:
    """Save feature importance CSVs per class."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for class_name, df in importance_dict.items():
        safe_name = class_name.lower().replace(" ", "_")
        filepath = output_path / f"{safe_name}_features.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Feature importance saved: {filepath}")


def create_feature_analysis_row(
    experiment_id: str,
    feature_type: str,
    feature_count: int,
    vocab_size: int,
    sparsity: float,
    training_time: float,
) -> dict:
    """Create a single row for the feature_analysis.csv."""
    return {
        "experiment_id": experiment_id,
        "feature_type": feature_type,
        "feature_count": feature_count,
        "vocabulary_size": vocab_size,
        "sparsity": round(sparsity, 6),
        "training_time": round(training_time, 2),
    }
