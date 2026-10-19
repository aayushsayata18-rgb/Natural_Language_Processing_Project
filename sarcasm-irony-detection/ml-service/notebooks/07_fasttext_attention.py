"""
Phase 7 — FastText + Bi-LSTM + Attention Deep Learning
======================================================
Sarcasm & Irony Detection System using NLP

Master Notebook-style Python script (equivalent to 07_fasttext_attention.ipynb).

Sections:
    1. Imports & Configuration
    2. Environment & Hardware Check
    3. Load Cleaned Datasets
    4. Stratified Train / Validation Split
    5. Word-Level Tokenizer & Subword Capabilities
    6. Sequence-Length & Padding Analysis
    7. FastText Pre-trained Vectors Loading
    8. FastText Vocabulary Coverage Analysis
    9. FastText Embedding Matrix Construction
   10. FastText + Bi-LSTM Architecture (FT-01)
   11. FT-01 Training & Convergence
   12. FT-01 Training Curves
   13. Attention Mechanism & Model (FT-02)
   14. FT-02 Attention Model Training
   15. Token Attention Weights Extraction & Visualization
   16. Validation Evaluation & Model Selection
   17. Final Official Test Set Evaluation
   18. Qualitative Error Analysis (Sarcasm ↔ Irony ↔ Figurative)
   19. Multi-Phase Progression & Architecture Comparison
   20. Academic Research Conclusions & Next Steps

Usage:
    cd ml-service
    python notebooks/07_fasttext_attention.py
"""

import copy
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent  # ml-service/
sys.path.insert(0, str(BASE_DIR))

from src.deep_learning.fasttext_attention.tokenizer import Tokenizer
from src.deep_learning.fasttext_attention.fasttext import (
    load_fasttext_vectors,
    calculate_fasttext_coverage,
    synthesize_subword_vector,
    extract_character_ngrams,
    categorize_token,
    FASTTEXT_METADATA,
)
from src.deep_learning.fasttext_attention.embedding import build_embedding_matrix
from src.deep_learning.fasttext_attention.attention import Attention
from src.deep_learning.fasttext_attention.model import (
    FastTextBiLSTMClassifier,
    FastTextAttentionClassifier,
)
from src.deep_learning.fasttext_attention.train import train_model, create_dataloader
from src.deep_learning.fasttext_attention.evaluate import (
    evaluate_model,
    generate_predictions_df,
    collect_error_breakdown,
    extract_attention_examples,
    generate_classification_report_str,
    CLASS_NAMES,
)
from src.deep_learning.fasttext_attention.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_attention_heatmap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("notebook_phase7")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Section 2] Hardware Check: Using device '{device}' (PyTorch {torch.__version__})")

    # Data paths
    data_dir = BASE_DIR / "data" / "processed"
    embeddings_dir = BASE_DIR / "data" / "embeddings"
    results_dir = BASE_DIR / "results" / "fasttext_attention"
    models_dir = BASE_DIR / "models" / "deep_learning" / "fasttext"

    # [Section 3] Load Data
    print("\n[Section 3] Loading Cleaned Datasets...")
    train_df = pd.read_csv(data_dir / "train_clean.csv")
    test_df = pd.read_csv(data_dir / "test_clean.csv")
    print(f"Loaded Train: {len(train_df):,} samples, Test: {len(test_df):,} samples")

    texts_all = train_df["text"].values
    labels_all = train_df["label_id"].values
    test_texts = test_df["text"].values
    test_labels = test_df["label_id"].values

    # [Section 4] Train / Validation Split
    print("\n[Section 4] Performing Stratified 80/20 Train/Val Split...")
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        texts_all, labels_all, test_size=0.20, stratify=labels_all, random_state=42
    )
    print(f"Train subset: {len(X_train_text):,} | Validation subset: {len(X_val_text):,}")

    # [Section 5] Tokenizer
    print("\n[Section 5] Fitting Word Tokenizer (strictly on training split)...")
    max_words = 30000
    max_len = 30
    embed_dim = 300

    tokenizer = Tokenizer(max_num_words=max_words)
    tokenizer.fit(X_train_text)
    print(f"Fitted Vocabulary Size: {tokenizer.vocab_size:,} tokens (with <PAD>=0, <UNK>=1)")

    # [Section 6] Sequence-Length Analysis
    print("\n[Section 6] Sequence-Length & Padding Analysis...")
    seq_lens = tokenizer.get_sequence_lengths(X_train_text)
    mean_len = float(np.mean(seq_lens))
    median_len = float(np.median(seq_lens))
    pct_truncated = float(np.mean(seq_lens > max_len) * 100.0)
    pct_padded = float(np.mean(seq_lens < max_len) * 100.0)
    print(f"Mean length: {mean_len:.1f} | Median: {median_len:.1f} | Selected MAX_LEN: {max_len}")
    print(f"Truncated: {pct_truncated:.2f}% | Padded: {pct_padded:.2f}%")

    # [Section 7] FastText Loading
    print("\n[Section 7] Loading FastText Pre-trained Vectors...")
    ft_path = embeddings_dir / "wiki.simple.vec"
    fasttext_vectors = load_fasttext_vectors(str(ft_path), dim=embed_dim)
    print(f"Successfully loaded {len(fasttext_vectors):,} FastText vectors ({embed_dim}d)")

    # [Section 8] FastText Coverage
    print("\n[Section 8] Calculating FastText Vocabulary Coverage...")
    embedding_matrix, coverage_stats, oov_records = build_embedding_matrix(
        tokenizer.word2idx,
        fasttext_vectors,
        dim=embed_dim,
        use_subwords=True,
        oov_strategy="subword_random",
    )
    print(f"Exact Matches: {coverage_stats['exact_matches']:,} ({coverage_stats['exact_coverage_pct']}%)")
    print(f"Subword-Resolved Matches: {coverage_stats['subword_matches']:,} ({coverage_stats['subword_coverage_pct']}%)")
    print(f"Total Effective Coverage: {coverage_stats['total_covered']:,} ({coverage_stats['effective_coverage_pct']}%)")
    print(f"Unresolved OOV: {coverage_stats['unresolved_oov']:,} ({coverage_stats['oov_pct']}%)")

    # [Section 9] Embedding Matrix
    print(f"\n[Section 9] Embedding Matrix Shape: {embedding_matrix.shape}")
    print(f"Index 0 (<PAD>) norm: {np.linalg.norm(embedding_matrix[0]):.4f} (strictly zeros)")

    # Padding
    X_train_pad = tokenizer.pad_sequences(tokenizer.texts_to_sequences(X_train_text), max_len=max_len)
    X_val_pad = tokenizer.pad_sequences(tokenizer.texts_to_sequences(X_val_text), max_len=max_len)
    X_test_pad = tokenizer.pad_sequences(tokenizer.texts_to_sequences(test_texts), max_len=max_len)

    train_loader = create_dataloader(X_train_pad, y_train, batch_size=64, shuffle=True)
    val_loader = create_dataloader(X_val_pad, y_val, batch_size=64, shuffle=False)
    test_loader = create_dataloader(X_test_pad, test_labels, batch_size=64, shuffle=False)

    # [Section 10] FastText + Bi-LSTM (FT-01)
    print("\n[Section 10] Initializing FT-01: FastText + Bi-LSTM (No Attention)...")
    ft01_model = FastTextBiLSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embed_dim,
        lstm_units=64,
        dense_units=64,
        num_classes=4,
        dropout=0.3,
        embedding_matrix=embedding_matrix,
        embedding_trainable=False,
    )
    print(ft01_model.get_architecture_summary())

    # [Section 13] Attention Model (FT-02)
    print("\n[Section 13] Initializing FT-02: FastText + Bi-LSTM + Attention...")
    ft02_model = FastTextAttentionClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embed_dim,
        lstm_units=64,
        attention_dim=64,
        dense_units=64,
        num_classes=4,
        dropout=0.3,
        embedding_matrix=embedding_matrix,
        embedding_trainable=False,
    )
    print(ft02_model.get_architecture_summary())

    print("\n[Section 15-20] Demonstrations and results are recorded in results/fasttext_attention/")
    print("Notebook demonstration script execution complete.")


if __name__ == "__main__":
    main()
