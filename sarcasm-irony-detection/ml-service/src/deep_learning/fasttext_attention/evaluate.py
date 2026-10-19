"""
Phase 7 — Evaluation Metrics, Probability Generation, Error Analysis, and Attention Extraction
==============================================================================================
Sarcasm & Irony Detection System using NLP

Computes:
- Classification accuracy, Macro F1, Macro Precision, Macro Recall, Weighted F1
- Per-class precision, recall, F1, support
- Confusion matrix in standard class order: [Regular, Sarcasm, Irony, Figurative]
- Detailed prediction records with softmax probabilities and confidence
- Focused error analysis (Sarcasm ↔ Irony, Sarcasm ↔ Figurative, Irony ↔ Figurative)
- Token-level attention weights extraction for explainability
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
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
) -> Dict[str, Any]:
    """
    Evaluates model on a DataLoader and computes full classification metrics.
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
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0
    )

    per_class = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        per_class[cls_name] = {
            "precision": round(float(p_class[i]), 4),
            "recall": round(float(r_class[i]), 4),
            "f1": round(float(f1_class[i]), 4),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(weighted_f1, 4),
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
    Generates detailed prediction records with class softmax probabilities and confidence score.
    """
    device_obj = torch.device(device)
    model.eval()
    model.to(device_obj)

    all_probs = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device_obj)
            logits = model(batch_x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probs_arr = np.vstack(all_probs)
    preds = probs_arr.argmax(axis=1)
    confs = probs_arr.max(axis=1)

    y_true = np.array(true_labels)
    correct = (preds == y_true)

    df = pd.DataFrame(
        {
            "text": list(texts),
            "actual_label": [CLASS_NAMES[i] for i in y_true],
            "actual_id": y_true,
            "predicted_label": [CLASS_NAMES[i] for i in preds],
            "predicted_id": preds,
            "correct": correct,
            "confidence": np.round(confs, 4),
            "probability_regular": np.round(probs_arr[:, 0], 4),
            "probability_sarcasm": np.round(probs_arr[:, 1], 4),
            "probability_irony": np.round(probs_arr[:, 2], 4),
            "probability_figurative": np.round(probs_arr[:, 3], 4),
        }
    )
    return df


def collect_error_breakdown(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Categorizes errors with special focus on Sarcasm <-> Irony semantic overlap.
    """
    errors_df = predictions_df[~predictions_df["correct"]]
    sarcasm_to_irony = len(errors_df[(errors_df["actual_label"] == "Sarcasm") & (errors_df["predicted_label"] == "Irony")])
    irony_to_sarcasm = len(errors_df[(errors_df["actual_label"] == "Irony") & (errors_df["predicted_label"] == "Sarcasm")])
    sarcasm_to_fig = len(errors_df[(errors_df["actual_label"] == "Sarcasm") & (errors_df["predicted_label"] == "Figurative")])
    irony_to_fig = len(errors_df[(errors_df["actual_label"] == "Irony") & (errors_df["predicted_label"] == "Figurative")])
    regular_confused = len(errors_df[errors_df["actual_label"] == "Regular"])
    fig_confused = len(errors_df[errors_df["actual_label"] == "Figurative"])

    summary = {
        "total_errors": len(errors_df),
        "sarcasm_as_irony": sarcasm_to_irony,
        "irony_as_sarcasm": irony_to_sarcasm,
        "sarcasm_as_figurative": sarcasm_to_fig,
        "irony_as_figurative": irony_to_fig,
        "regular_misclassified": regular_confused,
        "figurative_misclassified": fig_confused,
    }
    return summary


def extract_attention_examples(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    labels: List[int],
    max_len: int = 30,
    device: str = "cpu",
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Extracts token-level attention weights for representative test instances.
    Returns structured records containing tokens and their corresponding attention weights.
    """
    if not hasattr(model, "has_attention") or not model.has_attention:
        return []

    device_obj = torch.device(device)
    model.eval()
    model.to(device_obj)

    sequences = tokenizer.texts_to_sequences(texts)
    padded = tokenizer.pad_sequences(sequences, max_len=max_len, padding="pre", truncating="post")
    tensor_x = torch.tensor(padded, dtype=torch.long).to(device_obj)

    results = []
    with torch.no_grad():
        logits, weights = model(tensor_x, return_attention=True)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        attn_matrix = weights.cpu().numpy()

    for i, text in enumerate(texts):
        actual_cls = CLASS_NAMES[labels[i]]
        pred_cls = CLASS_NAMES[preds[i]]
        conf = float(probs[i, preds[i]])
        seq = padded[i]
        attn_weights = attn_matrix[i]

        # Filter out padding tokens (index 0)
        valid_pairs = []
        for tok_idx, w in zip(seq, attn_weights):
            if tok_idx != 0:
                word = tokenizer.idx2word.get(int(tok_idx), "<UNK>")
                valid_pairs.append((word, float(w)))

        # Normalize valid pairs so their displayed weights sum to 1.0 (for intuitive viewing)
        total_w = sum(w for _, w in valid_pairs)
        if total_w > 1e-9:
            norm_pairs = [(word, round(w / total_w, 4)) for word, w in valid_pairs]
        else:
            norm_pairs = valid_pairs

        results.append(
            {
                "text": text,
                "actual_label": actual_cls,
                "predicted_label": pred_cls,
                "correct": (actual_cls == pred_cls),
                "confidence": round(conf, 4),
                "token_attention": norm_pairs,
                "raw_weight_sum": round(float(np.sum(attn_weights)), 4),
            }
        )

    return results


def generate_classification_report_str(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    target_names: List[str] = CLASS_NAMES,
) -> str:
    """
    Generates standard formatted classification report text.
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
