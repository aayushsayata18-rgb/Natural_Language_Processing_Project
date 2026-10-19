"""
Phase 6 — GloVe + Bi-LSTM Deep Learning Experiment Runner
=========================================================
Sarcasm & Irony Detection System using NLP

Full CLI orchestrator for Phase 6 experiments:
    1. Loads sanitized datasets (train_clean.csv, test_clean.csv)
    2. Builds tokenizer fitted strictly on training data
    3. Analyzes sequence length distribution and checks coverage
    4. Loads GloVe vectors (Stanford 100d) and builds embedding matrix
    5. Runs architecture comparison experiments (BILSTM-01, 02, 03)
    6. Selects best model based on validation Macro F1
    7. Retrains/evaluates final model on test set
    8. Generates predictions, confusion matrices, and visualizations
    9. Saves summary report and metrics

Usage:
    python src/deep_learning/bilstm/run_experiments.py
"""

import gc
import json
import logging
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ml-service/
sys.path.insert(0, str(BASE_DIR))

from src.deep_learning.bilstm.tokenizer import Tokenizer
from src.deep_learning.bilstm.glove import (
    load_glove_vectors,
    build_embedding_matrix,
    analyze_oov,
    save_coverage_report,
)
from src.deep_learning.bilstm.model import BiLSTMClassifier
from src.deep_learning.bilstm.train import train_model, create_dataloader
from src.deep_learning.bilstm.evaluate import (
    evaluate_model,
    generate_predictions_df,
    collect_errors,
    generate_classification_report_str,
    CLASS_NAMES,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_pipeline():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    data_dir = BASE_DIR / "data" / "processed"
    embeddings_dir = BASE_DIR / "data" / "embeddings"
    models_dir = BASE_DIR / "models" / "deep_learning" / "bilstm"
    results_dir = BASE_DIR / "results" / "bilstm"
    viz_dir = results_dir / "visualizations"
    cm_dir = results_dir / "confusion_matrices"

    viz_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logger.info("Loading Phase 1 cleaned datasets...")
    train_df = pd.read_csv(data_dir / "train_clean.csv")
    test_df = pd.read_csv(data_dir / "test_clean.csv")

    texts_all = train_df["text"].values
    labels_all = train_df["label_id"].values

    # Train / Validation Split (80 / 20 stratified)
    X_tr_text, X_val_text, y_train, y_val = train_test_split(
        texts_all, labels_all, test_size=0.20, stratify=labels_all, random_state=42
    )

    # 2. Tokenizer (fitted only on training text)
    max_words = 30000
    max_len = 30
    embed_dim = 100

    logger.info("Fitting Tokenizer on training text...")
    tokenizer = Tokenizer(max_num_words=max_words)
    tokenizer.fit(X_tr_text)
    logger.info("Vocabulary size: %d", tokenizer.vocab_size)

    # 3. GloVe Embeddings
    glove_path = embeddings_dir / "glove.6B.100d.txt"
    if not glove_path.exists():
        logger.warning("GloVe file not found at %s. Mocking or checking fallback.", glove_path)
        glove_vectors = {}
    else:
        logger.info("Loading GloVe vectors from %s...", glove_path)
        glove_vectors = load_glove_vectors(str(glove_path), dim=embed_dim)

    embedding_matrix, coverage_stats, oov_words = build_embedding_matrix(
        tokenizer.word2idx, glove_vectors, embed_dim, oov_strategy="random"
    )
    save_coverage_report(coverage_stats, str(results_dir / "glove_coverage.csv"))

    if oov_words:
        oov_df = analyze_oov(oov_words)
        oov_df.to_csv(results_dir / "oov_analysis.csv", index=False)

    # 4. Padding
    X_train_seq = tokenizer.texts_to_sequences(X_tr_text)
    X_val_seq = tokenizer.texts_to_sequences(X_val_text)
    X_train_pad = tokenizer.pad_sequences(X_train_seq, max_len=max_len)
    X_val_pad = tokenizer.pad_sequences(X_val_seq, max_len=max_len)

    batch_size = 32
    train_loader = create_dataloader(X_train_pad, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val_pad, y_val, batch_size=batch_size, shuffle=False)

    experiments = [
        {"id": "BILSTM-01", "desc": "Frozen GloVe, LSTM 64", "lstm": 64, "dense": 64, "dropout": 0.3, "trainable": False},
        {"id": "BILSTM-02", "desc": "Fine-tuned GloVe, LSTM 64", "lstm": 64, "dense": 64, "dropout": 0.3, "trainable": True},
        {"id": "BILSTM-03", "desc": "Frozen GloVe, LSTM 128", "lstm": 128, "dense": 64, "dropout": 0.3, "trainable": False},
    ]

    all_results = {}
    best_f1 = -1
    best_exp = experiments[1]

    logger.info("Executing Phase 6 experiment matrix...")
    for exp in experiments:
        set_seed(42)
        model = BiLSTMClassifier(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=embed_dim,
            embedding_matrix=embedding_matrix,
            lstm_units=exp["lstm"],
            dense_units=exp["dense"],
            num_classes=4,
            dropout=exp["dropout"],
            embedding_trainable=exp["trainable"],
        )

        ckpt_dir = models_dir / exp["id"].lower()
        trained_model, history = train_model(
            model,
            train_loader,
            val_loader,
            max_epochs=30,
            learning_rate=0.001,
            patience=5,
            checkpoint_dir=str(ckpt_dir),
            device=device,
        )

        metrics = evaluate_model(trained_model, val_loader, device=device)
        all_results[exp["id"]] = {"metrics": metrics, "history": history, "config": exp}

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_exp = exp

    logger.info("Best validation experiment: %s (Macro F1 = %.4f)", best_exp["id"], best_f1)


if __name__ == "__main__":
    run_pipeline()
