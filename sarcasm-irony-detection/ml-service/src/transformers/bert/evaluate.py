"""
Phase 8 — BERT Evaluation Metrics & Error Analysis
==================================================
Sarcasm & Irony Detection System using NLP

Computes:
1. Accuracy
2. Macro Precision
3. Macro Recall
4. Macro F1 (Primary Model Selection Metric)
5. Weighted F1
6. Per-Class Precision, Recall, and F1
7. 4x4 Confusion Matrix (Regular, Sarcasm, Irony, Figurative)
8. Prediction probabilities and confidence scores
9. Misclassification analysis focused on subtle boundary confusion:
   - Sarcasm ↔ Irony
   - Sarcasm ↔ Figurative
   - Irony ↔ Figurative
   - Regular ↔ others
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Regular", "Sarcasm", "Irony", "Figurative"]
ID_TO_LABEL = {0: "Regular", 1: "Sarcasm", 2: "Irony", 3: "Figurative"}
LABEL_TO_ID = {"Regular": 0, "Sarcasm": 1, "Irony": 2, "Figurative": 3}


def calculate_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes all standard multiclass classification metrics with focus on Macro F1.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    macro_p = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    macro_r = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    # Per-class metrics
    per_class_p = precision_score(y_t, y_p, average=None, zero_division=0)
    per_class_r = recall_score(y_t, y_p, average=None, zero_division=0)
    per_class_f1 = f1_score(y_t, y_p, average=None, zero_division=0)

    per_class_metrics = {}
    for idx, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(per_class_p[idx]) if idx < len(per_class_p) else 0.0,
            "recall": float(per_class_r[idx]) if idx < len(per_class_r) else 0.0,
            "f1": float(per_class_f1[idx]) if idx < len(per_class_f1) else 0.0,
            "support": int(np.sum(y_t == idx)),
        }

    cm = confusion_matrix(y_t, y_p, labels=list(range(len(class_names))))

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(y_t)),
    }


def evaluate_bert_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    max_steps: Optional[int] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates BERT on a DataLoader and returns metrics, logits, probabilities, and labels.
    """
    model.eval()
    total_loss = 0.0
    all_logits: List[np.ndarray] = []
    all_labels: List[int] = []

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for step_idx, batch in enumerate(dataloader, start=1):
            if max_steps and step_idx > max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits

            if labels is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * len(labels)
                all_labels.extend(labels.cpu().numpy().tolist())

            all_logits.append(logits.cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    # Convert logits to probabilities via softmax
    exp_logits = np.exp(logits_arr - np.max(logits_arr, axis=-1, keepdims=True))
    probs_arr = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    preds_arr = np.argmax(probs_arr, axis=-1)

    labels_arr = np.array(all_labels) if len(all_labels) > 0 else np.array([])
    metrics: Dict[str, Any] = {}

    if len(labels_arr) > 0:
        avg_loss = total_loss / len(labels_arr)
        metrics = calculate_classification_metrics(labels_arr, preds_arr)
        metrics["loss"] = round(float(avg_loss), 4)

    return metrics, logits_arr, probs_arr, labels_arr


def generate_predictions_dataframe(
    texts: List[str],
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    probabilities: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Builds a detailed predictions DataFrame containing text, actual label,
    predicted label, per-class probabilities, confidence, and correctness flag.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    probs = np.array(probabilities)
    conf = np.max(probs, axis=-1)

    df = pd.DataFrame({
        "text": texts,
        "actual_label_id": y_t,
        "actual_class": [class_names[i] if 0 <= i < len(class_names) else "Unknown" for i in y_t],
        "predicted_label_id": y_p,
        "predicted_class": [class_names[i] if 0 <= i < len(class_names) else "Unknown" for i in y_p],
        "confidence": np.round(conf, 4),
        "is_correct": (y_t == y_p),
    })

    for idx, name in enumerate(class_names):
        col_name = f"prob_{name.lower()}"
        df[col_name] = np.round(probs[:, idx], 4)

    return df


def analyze_misclassifications(
    pred_df: pd.DataFrame,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyzes model errors with particular attention to:
    1. Sarcasm ↔ Irony confusion
    2. Sarcasm ↔ Figurative confusion
    3. Irony ↔ Figurative confusion
    4. Regular ↔ Others confusion
    """
    if class_names is None:
        class_names = CLASS_NAMES

    errors_df = pred_df[~pred_df["is_correct"]].copy()
    total_samples = len(pred_df)
    total_errors = len(errors_df)
    error_rate = (total_errors / total_samples) * 100.0 if total_samples > 0 else 0.0

    # Pairwise confusion counts
    pairwise_errors: Dict[str, int] = {}
    for actual in class_names:
        for pred in class_names:
            if actual != pred:
                key = f"{actual} -> {pred}"
                cnt = int(len(errors_df[(errors_df["actual_class"] == actual) & (errors_df["predicted_class"] == pred)]))
                pairwise_errors[key] = cnt

    # Key focal confusion pairs
    sarcasm_as_irony = pairwise_errors.get("Sarcasm -> Irony", 0)
    irony_as_sarcasm = pairwise_errors.get("Irony -> Sarcasm", 0)
    sarcasm_as_figurative = pairwise_errors.get("Sarcasm -> Figurative", 0)
    irony_as_figurative = pairwise_errors.get("Irony -> Figurative", 0)
    regular_as_other = sum(v for k, v in pairwise_errors.items() if k.startswith("Regular ->"))
    other_as_regular = sum(v for k, v in pairwise_errors.items() if k.endswith("-> Regular"))

    # Top high-confidence errors
    top_confident_errors = (
        errors_df.sort_values(by="confidence", ascending=False)
        .head(10)[["text", "actual_class", "predicted_class", "confidence"]]
        .to_dict(orient="records")
    )

    return {
        "total_samples": total_samples,
        "total_errors": total_errors,
        "error_rate_pct": round(error_rate, 2),
        "pairwise_errors": pairwise_errors,
        "sarcasm_irony_mutual_confusion": sarcasm_as_irony + irony_as_sarcasm,
        "sarcasm_as_irony": sarcasm_as_irony,
        "irony_as_sarcasm": irony_as_sarcasm,
        "sarcasm_as_figurative": sarcasm_as_figurative,
        "irony_as_figurative": irony_as_figurative,
        "regular_misclassified_as_other": regular_as_other,
        "other_misclassified_as_regular": other_as_regular,
        "top_high_confidence_errors": top_confident_errors,
    }
