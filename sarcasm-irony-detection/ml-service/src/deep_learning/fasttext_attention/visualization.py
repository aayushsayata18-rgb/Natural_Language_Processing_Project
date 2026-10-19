"""
Phase 7 — Visualizations: Training History, Confusion Matrices, and Attention Heatmaps
======================================================================================
Sarcasm & Irony Detection System using NLP

Generates:
- Training vs Validation Loss and Accuracy curves
- Normalized and Raw Confusion Matrix heatmaps
- Token-level Attention heatmaps and bar plots
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CLASS_NAMES = ["Regular", "Sarcasm", "Irony", "Figurative"]


def plot_training_history(
    history: Dict[str, List[float]],
    title: str,
    output_path: str,
) -> None:
    """
    Plots training and validation loss and accuracy side-by-side.
    """
    epochs = history["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", color="#2b5c8f")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="s", color="#d95f02")
    ax1.set_title(f"{title} — Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker="o", color="#2b5c8f")
    ax2.plot(epochs, history["val_acc"], label="Val Acc", marker="s", color="#2ca02c")
    ax2.set_title(f"{title} — Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    output_path: str,
    normalize: bool = False,
) -> None:
    """
    Plots confusion matrix with annotations.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    if normalize:
        cm_display = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-9)
        fmt = ".2%"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = None

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True,
        ax=ax,
        vmax=vmax,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_attention_heatmap(
    token_weights_pairs: List[tuple],
    title: str,
    output_path: str,
) -> None:
    """
    Visualizes token attention weights as a clear horizontal heatmap.
    """
    if not token_weights_pairs:
        return

    tokens = [p[0] for p in token_weights_pairs]
    weights = np.array([p[1] for p in token_weights_pairs]).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.8), 2.5))
    sns.heatmap(
        weights,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=tokens,
        yticklabels=["Attn"],
        cbar=True,
        ax=ax,
        cbar_kws={"orientation": "horizontal", "pad": 0.3},
    )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
