"""
Phase 4 - Artifact Saving
============================================
Sarcasm & Irony Detection System using NLP

Save and load trained model artifacts:
- Trained models (joblib)
- Vectorizers (joblib)
- Configurations (JSON)
- Label mapping
"""

import json
import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


def save_model_artifacts(
    model,
    vectorizer,
    config: dict,
    output_dir: str,
    experiment_id: str,
) -> None:
    """
    Save a trained model and its associated artifacts.
    
    Saves:
        - model.joblib: Trained classifier
        - vectorizer.joblib: Fitted CountVectorizer
        - config.json: Experiment configuration and metadata
    
    Args:
        model: Trained sklearn classifier.
        vectorizer: Fitted CountVectorizer.
        config: Dict with experiment configuration.
        output_dir: Directory to save artifacts.
        experiment_id: Unique experiment identifier.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")
    
    # Save vectorizer
    vectorizer_path = output_path / "vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"Vectorizer saved: {vectorizer_path}")
    
    # Save config
    config["experiment_id"] = experiment_id
    config_path = output_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"Config saved: {config_path}")


def load_model_artifacts(model_dir: str) -> tuple:
    """
    Load a saved model and its artifacts.
    
    Returns:
        Tuple of (model, vectorizer, config).
    """
    model_path = Path(model_dir)
    
    model = joblib.load(model_path / "model.joblib")
    vectorizer = joblib.load(model_path / "vectorizer.joblib")
    
    with open(model_path / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    logger.info(f"Loaded model artifacts from {model_dir}")
    return model, vectorizer, config


def save_confusion_matrix(cm, class_names, filepath: str) -> None:
    """Save confusion matrix as CSV."""
    import pandas as pd
    
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"
    cm_df.to_csv(filepath)
    logger.info(f"Confusion matrix saved: {filepath}")


def save_classification_report(report: dict, filepath: str) -> None:
    """Save classification report as JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Classification report saved: {filepath}")
