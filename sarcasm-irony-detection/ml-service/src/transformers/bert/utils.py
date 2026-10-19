"""
Phase 8 — BERT Utilities & Visualization
========================================
Sarcasm & Irony Detection System using NLP

Includes:
- Reproducibility seeding (Python, NumPy, PyTorch, CUDA)
- Environment and computational hardware metadata capture (environment.json)
- Visualization plots:
  1. Training and validation loss curves
  2. Validation Accuracy and Macro F1 curves
  3. 4x4 Confusion Matrix heatmap
  4. Cross-phase model progression comparison chart
"""

import json
import logging
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import transformers

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic random seeds across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set global random seed to {seed}")


def get_environment_info(seed: int = 42) -> Dict[str, Any]:
    """
    Captures complete system, hardware, and dependency environment details.
    """
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    device_count = torch.cuda.device_count() if cuda_available else 0

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name": device_name,
        "random_seed": seed,
    }


def plot_training_curves(
    history: List[Dict[str, Any]],
    save_path: Union[str, Path],
) -> None:
    """
    Plots training loss, validation loss, validation accuracy, and Macro F1.
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curves
    ax1.plot(epochs, train_loss, marker="o", color="#2563eb", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, marker="s", color="#dc2626", label="Val Loss", linewidth=2)
    ax1.set_title("BERT Fine-Tuning Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # Accuracy & Macro F1
    ax2.plot(epochs, val_acc, marker="^", color="#16a34a", label="Val Accuracy", linewidth=2)
    ax2.plot(epochs, val_f1, marker="D", color="#9333ea", label="Val Macro F1 (Selection)", linewidth=2)
    ax2.set_title("BERT Validation Performance", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300)
    plt.close()
    logger.info(f"Saved training curves plot to {save_path}")


def plot_confusion_matrix_heatmap(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: List[str],
    save_path: Union[str, Path],
    title: str = "BERT Confusion Matrix (Phase 8)",
) -> None:
    """
    Generates and saves a 4x4 annotated confusion matrix heatmap.
    """
    cm_arr = np.array(cm)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_norm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr, dtype=float), where=row_sums != 0)

    # Create annotation labels (count + percentage)
    labels = np.empty_like(cm_arr, dtype=object)
    for r in range(cm_arr.shape[0]):
        for c in range(cm_arr.shape[1]):
            cnt = cm_arr[r, c]
            pct = row_norm[r, c] * 100.0
            labels[r, c] = f"{cnt:,}\n({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        row_norm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("True Class", fontsize=12, labelpad=10)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix heatmap to {save_path}")


def plot_cross_phase_progression(
    progression_df: pd.DataFrame,
    save_path: Union[str, Path],
) -> None:
    """
    Creates comparative bar charts across Phase 4, Phase 5, Phase 6, Phase 7, and Phase 8.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    models = progression_df["model"].tolist()
    macro_f1 = progression_df["macro_f1"].tolist()
    accuracy = progression_df["accuracy"].tolist()

    y_pos = np.arange(len(models))

    # Macro F1 (Primary Metric)
    bars1 = ax1.barh(y_pos, macro_f1, color="#6366f1", edgecolor="#312e81")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Macro F1 Score (Primary Selection Metric)", fontsize=11, fontweight="bold")
    ax1.set_title("Model Progression — Macro F1", fontsize=13, fontweight="bold")
    ax1.set_xlim(0.60, 0.85)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="x")

    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.003, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9, fontweight="bold")

    # Accuracy (Secondary Metric)
    bars2 = ax2.barh(y_pos, accuracy, color="#10b981", edgecolor="#064e3b")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Accuracy (Secondary Metric)", fontsize=11, fontweight="bold")
    ax2.set_title("Model Progression — Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlim(0.80, 0.90)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")

    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300)
    plt.close()
    logger.info(f"Saved cross-phase model progression plot to {save_path}")
