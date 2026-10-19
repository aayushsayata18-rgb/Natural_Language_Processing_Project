"""
Evaluation metrics, confusion matrices, prediction probabilities, and error analysis for Bi-LSTM.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CLASS_NAMES = ["Regular", "Sarcasm", "Irony", "Figurative"]


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, Union[float, Dict[str, Dict[str, float]], np.ndarray]]:
    """
    Evaluates Bi-LSTM model on a DataLoader and computes full classification metrics.
    """
    device_obj = torch.device(device)
    model.eval()
    model.to(device_obj)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device_obj)
            logits = model(batch_x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0
    )

    per_class = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        per_class[cls_name] = {
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "f1": float(f1_class[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    return {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def generate_predictions_df(
    model: nn.Module,
    data_loader: DataLoader,
    texts: Union[List[str], np.ndarray],
    true_labels: Union[List[int], np.ndarray],
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Generates detailed prediction table with confidence and class probabilities.
    """
    device_obj = torch.device(device)
    model.eval()
    model.to(device_obj)

    all_probs = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device_obj)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probs_arr = np.vstack(all_probs)
    preds = probs_arr.argmax(axis=1)
    confs = probs_arr.max(axis=1)

    y_true = np.array(true_labels)
    correct = preds == y_true

    df = pd.DataFrame(
        {
            "text": list(texts),
            "true_class": [CLASS_NAMES[i] for i in y_true],
            "predicted_class": [CLASS_NAMES[i] for i in preds],
            "correct": correct,
            "confidence": confs,
            "prob_Regular": probs_arr[:, 0],
            "prob_Sarcasm": probs_arr[:, 1],
            "prob_Irony": probs_arr[:, 2],
            "prob_Figurative": probs_arr[:, 3],
        }
    )
    return df


def collect_errors(predictions_df: pd.DataFrame, n_samples: int = 50) -> pd.DataFrame:
    """
    Filters and samples prediction errors for qualitative error analysis.
    """
    errors_df = predictions_df[~predictions_df["correct"]]
    if len(errors_df) > n_samples:
        return errors_df.head(n_samples)
    return errors_df


def generate_classification_report_str(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    target_names: List[str] = CLASS_NAMES,
) -> str:
    """
    Generates formatted classification report text.
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
