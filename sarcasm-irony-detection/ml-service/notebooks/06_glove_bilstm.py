"""
Phase 6 — GloVe + Bi-LSTM Deep Learning
============================================
Sarcasm & Irony Detection System using NLP

Notebook-style script (equivalent to 06_glove_bilstm.ipynb).

Sections:
    1. Imports
    2. Environment check
    3. Load data
    4. Train/validation/test split
    5. Tokenizer
    6. Sequence-length analysis
    7. Padding
    8. GloVe loading
    9. GloVe coverage
   10. Embedding matrix
   11. Bi-LSTM architecture
   12. Model compilation
   13. Training
   14. Training curves
   15. Validation evaluation
   16. Test evaluation
   17. Confusion matrix
   18. Error analysis
   19. Architecture comparison
   20. Comparison with classical models
   21. Conclusions

Usage:
    python notebooks/06_glove_bilstm.py
"""

import gc
import io
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent  # ml-service/
sys.path.insert(0, str(BASE_DIR))

from src.deep_learning.bilstm.tokenizer import Tokenizer
from src.deep_learning.bilstm.glove import (
    load_glove_vectors, build_embedding_matrix, save_coverage_report,
)
from src.deep_learning.bilstm.model import BiLSTMClassifier
from src.deep_learning.bilstm.train import train_model, create_dataloader
from src.deep_learning.bilstm.evaluate import (
    evaluate_model, generate_predictions_df, collect_errors,
    generate_classification_report_str, CLASS_NAMES,
)
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = BASE_DIR / "data" / "processed"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
MODELS_DIR = BASE_DIR / "models" / "deep_learning" / "bilstm"
RESULTS_DIR = BASE_DIR / "results" / "bilstm"
VIZ_DIR = RESULTS_DIR / "visualizations"
CM_DIR = RESULTS_DIR / "confusion_matrices"

# ============================================================
# 2. ENVIRONMENT CHECK
# ============================================================

print("=" * 70)
print("PHASE 6 — GloVe + Bi-LSTM DEEP LEARNING (Notebook)")
print("=" * 70)

print(f"\nPython: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
print(f"Random seed: {SEED}")

# Configuration
MAX_NUM_WORDS = 30000
MAX_LEN = 30
EMBEDDING_DIM = 100
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 5
LR = 0.001
VAL_SIZE = 0.20

print(f"\nConfiguration:")
print(f"  MAX_NUM_WORDS: {MAX_NUM_WORDS}")
print(f"  MAX_LEN: {MAX_LEN}")
print(f"  EMBEDDING_DIM: {EMBEDDING_DIM}")
print(f"  BATCH_SIZE: {BATCH_SIZE}")
print(f"  MAX_EPOCHS: {MAX_EPOCHS}")
print(f"  PATIENCE: {PATIENCE}")
print(f"  LR: {LR}")

# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: LOAD DATA")
print("=" * 70)

train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
test_df = pd.read_csv(DATA_DIR / "test_clean.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print(f"\nTrain columns: {list(train_df.columns)}")
print(f"\nLabel distribution (train):")
print(train_df["label_id"].value_counts().sort_index())

texts_all = train_df["text"].values
labels_all = train_df["label_id"].values

# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: TRAIN / VALIDATION SPLIT")
print("=" * 70)

X_tr_text, X_val_text, y_train, y_val = train_test_split(
    texts_all, labels_all,
    test_size=VAL_SIZE, stratify=labels_all, random_state=SEED,
)

print(f"Training:   {len(X_tr_text)} samples")
print(f"Validation: {len(X_val_text)} samples")
print(f"Test:       {len(test_df)} samples")

print(f"\nTrain label distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {CLASS_NAMES[u]}: {c} ({c/len(y_train)*100:.1f}%)")

# ============================================================
# 5. TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: TOKENIZER (fitted on training data only)")
print("=" * 70)

tokenizer = Tokenizer(max_num_words=MAX_NUM_WORDS)
tokenizer.fit(X_tr_text)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"  Includes PAD (idx=0) and OOV/UNK (idx=1)")
print(f"  Max words kept: {MAX_NUM_WORDS}")
print(f"  Total unique words: {len(tokenizer.word_counts)}")

# Show some examples
sample_text = X_tr_text[0]
sample_tokens = tokenizer._tokenize(sample_text)
sample_seq = tokenizer.texts_to_sequences([sample_text])[0]
print(f"\nExample:")
print(f"  Text:    {sample_text[:100]}...")
print(f"  Tokens:  {sample_tokens[:10]}...")
print(f"  Seq:     {sample_seq[:10]}...")

# ============================================================
# 6. SEQUENCE LENGTH ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: SEQUENCE LENGTH ANALYSIS")
print("=" * 70)

seq_lengths = tokenizer.get_sequence_lengths(X_tr_text)

print(f"Token-length distribution (training data):")
print(f"  Min:    {seq_lengths.min()}")
print(f"  Max:    {seq_lengths.max()}")
print(f"  Mean:   {seq_lengths.mean():.1f}")
print(f"  Median: {np.median(seq_lengths):.0f}")
print(f"  Std:    {seq_lengths.std():.1f}")
print(f"\nPercentiles:")
for pct in [50, 75, 90, 95, 99, 99.5]:
    val = np.percentile(seq_lengths, pct)
    print(f"  {pct}th: {val:.0f} tokens")

coverage_at_max = (seq_lengths <= MAX_LEN).mean() * 100
pct_truncated = (seq_lengths > MAX_LEN).mean() * 100
pct_padded = (seq_lengths < MAX_LEN).mean() * 100

print(f"\nMAX_LEN = {MAX_LEN}")
print(f"  Coverage: {coverage_at_max:.2f}% of texts fit within MAX_LEN")
print(f"  Truncated: {pct_truncated:.2f}%")
print(f"  Padded: {pct_padded:.2f}%")

# ============================================================
# 7. PADDING
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: PADDING & SEQUENCE CONVERSION")
print("=" * 70)

X_train_seq = tokenizer.texts_to_sequences(X_tr_text)
X_val_seq = tokenizer.texts_to_sequences(X_val_text)

X_train_pad = tokenizer.pad_sequences(X_train_seq, MAX_LEN)
X_val_pad = tokenizer.pad_sequences(X_val_seq, MAX_LEN)

print(f"Padded training shape: {X_train_pad.shape}")
print(f"Padded validation shape: {X_val_pad.shape}")
print(f"Padding strategy: pre-padding")
print(f"Truncation strategy: post-truncation")

# ============================================================
# 8. GLOVE LOADING
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: GLOVE EMBEDDINGS")
print("=" * 70)

glove_path = EMBEDDINGS_DIR / "glove.6B.100d.txt"
print(f"GloVe path: {glove_path}")
print(f"GloVe dimension: {EMBEDDING_DIM}")

glove_vectors = load_glove_vectors(str(glove_path), dim=EMBEDDING_DIM)
print(f"Loaded {len(glove_vectors):,} GloVe vectors")

# ============================================================
# 9. GLOVE COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("STEP 9: GLOVE COVERAGE ANALYSIS")
print("=" * 70)

embedding_matrix, coverage_stats, oov_words = build_embedding_matrix(
    tokenizer.word2idx, glove_vectors, EMBEDDING_DIM, oov_strategy="random",
)

print(f"Total vocabulary: {coverage_stats['total_vocabulary']:,}")
print(f"GloVe matched:   {coverage_stats['glove_matched']:,}")
print(f"OOV count:        {coverage_stats['oov_count']:,}")
print(f"Coverage:          {coverage_stats['coverage_pct']:.2f}%")
print(f"OOV strategy:      {coverage_stats['oov_strategy']}")

# Show some OOV examples
if oov_words:
    print(f"\nSample OOV words (first 20):")
    for w in oov_words[:20]:
        print(f"  {w}")

# ============================================================
# 10. EMBEDDING MATRIX
# ============================================================

print("\n" + "=" * 70)
print("STEP 10: EMBEDDING MATRIX")
print("=" * 70)

print(f"Embedding matrix shape: {embedding_matrix.shape}")
print(f"  vocab_size × embedding_dim = {tokenizer.vocab_size} × {EMBEDDING_DIM}")
print(f"Memory: {embedding_matrix.nbytes / 1024 / 1024:.1f} MB")

# Verify known words have GloVe vectors
sample_word = "good"
if sample_word in tokenizer.word2idx:
    idx = tokenizer.word2idx[sample_word]
    vec = embedding_matrix[idx]
    print(f"\nVerification: '{sample_word}' (idx={idx})")
    print(f"  First 5 values: {vec[:5]}")
    print(f"  Non-zero: {np.count_nonzero(vec)}/{len(vec)}")

# Free GloVe dict
del glove_vectors
gc.collect()

# ============================================================
# 11–13. BI-LSTM ARCHITECTURE, COMPILATION, TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STEP 11–13: EXPERIMENTS (3 architectures)")
print("=" * 70)

experiments = [
    {"id": "BILSTM-01", "desc": "Frozen GloVe, LSTM 64",
     "lstm": 64, "dense": 64, "dropout": 0.3, "trainable": False},
    {"id": "BILSTM-02", "desc": "Fine-tuned GloVe, LSTM 64",
     "lstm": 64, "dense": 64, "dropout": 0.3, "trainable": True},
    {"id": "BILSTM-03", "desc": "Frozen GloVe, LSTM 128",
     "lstm": 128, "dense": 64, "dropout": 0.3, "trainable": False},
]

train_loader = create_dataloader(X_train_pad, y_train, BATCH_SIZE, shuffle=True)
val_loader = create_dataloader(X_val_pad, y_val, BATCH_SIZE, shuffle=False)

all_val_results = {}
best_val_f1 = -1
best_exp = None

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp['id']} — {exp['desc']}")
    print(f"{'='*60}")

    # Reset seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = BiLSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        embedding_matrix=embedding_matrix,
        lstm_units=exp["lstm"],
        dense_units=exp["dense"],
        num_classes=4,
        dropout=exp["dropout"],
        embedding_trainable=exp["trainable"],
    )

    print(model.get_architecture_summary())

    ckpt_dir = str(MODELS_DIR / exp["id"].lower())
    trained_model, history = train_model(
        model, train_loader, val_loader,
        max_epochs=MAX_EPOCHS, learning_rate=LR,
        patience=PATIENCE, lr_patience=2, lr_factor=0.5,
        checkpoint_dir=ckpt_dir, device=device,
    )

    # 15. VALIDATION EVALUATION
    val_metrics = evaluate_model(trained_model, val_loader, device)
    print(f"\nValidation Results:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Macro F1:  {val_metrics['macro_f1']:.4f}")
    for cls, m in val_metrics["per_class"].items():
        print(f"  {cls:12s}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

    all_val_results[exp["id"]] = {
        "metrics": val_metrics,
        "history": history,
        "config": exp,
    }

    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        best_exp = exp

    del trained_model, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\nBest validation experiment: {best_exp['id']} (Macro F1 = {best_val_f1:.4f})")

# ============================================================
# 16. TEST EVALUATION (Best Config)
# ============================================================

print("\n" + "=" * 70)
print(f"STEP 16: TEST EVALUATION (Best: {best_exp['id']})")
print("=" * 70)

# Retrain on full training data
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

tok_full = Tokenizer(max_num_words=MAX_NUM_WORDS)
tok_full.fit(texts_all)

glove_vectors = load_glove_vectors(str(glove_path), dim=EMBEDDING_DIM)
embed_full, _, _ = build_embedding_matrix(
    tok_full.word2idx, glove_vectors, EMBEDDING_DIM, oov_strategy="random",
)
del glove_vectors
gc.collect()

X_full_seq = tok_full.texts_to_sequences(texts_all)
X_test_seq = tok_full.texts_to_sequences(test_df["text"].values)
X_full_pad = tok_full.pad_sequences(X_full_seq, MAX_LEN)
X_test_pad = tok_full.pad_sequences(X_test_seq, MAX_LEN)

y_test = test_df["label_id"].values

# Internal split for early stopping
X_fin_tr, X_fin_val, y_fin_tr, y_fin_val = train_test_split(
    X_full_pad, labels_all, test_size=0.1, stratify=labels_all, random_state=SEED,
)
fin_train_loader = create_dataloader(X_fin_tr, y_fin_tr, BATCH_SIZE, shuffle=True)
fin_val_loader = create_dataloader(X_fin_val, y_fin_val, BATCH_SIZE, shuffle=False)

final_model = BiLSTMClassifier(
    vocab_size=tok_full.vocab_size,
    embedding_dim=EMBEDDING_DIM,
    embedding_matrix=embed_full,
    lstm_units=best_exp["lstm"],
    dense_units=best_exp["dense"],
    num_classes=4,
    dropout=best_exp["dropout"],
    embedding_trainable=best_exp["trainable"],
)

final_model, final_hist = train_model(
    final_model, fin_train_loader, fin_val_loader,
    max_epochs=MAX_EPOCHS, learning_rate=LR, patience=PATIENCE,
    checkpoint_dir=str(MODELS_DIR / "best"), device=device,
)

test_loader = create_dataloader(X_test_pad, y_test, BATCH_SIZE, shuffle=False)
test_metrics = evaluate_model(final_model, test_loader, device)

print(f"\nTEST RESULTS (Final, Unbiased):")
print(f"  Accuracy:        {test_metrics['accuracy']:.4f}")
print(f"  Macro Precision: {test_metrics['macro_precision']:.4f}")
print(f"  Macro Recall:    {test_metrics['macro_recall']:.4f}")
print(f"  Macro F1:        {test_metrics['macro_f1']:.4f}")
print()
for cls, m in test_metrics["per_class"].items():
    print(f"  {cls:12s}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} (n={m['support']})")

# 17. CONFUSION MATRIX
print("\n" + "=" * 70)
print("STEP 17: CONFUSION MATRIX")
print("=" * 70)

cm = np.array(test_metrics["confusion_matrix"])
print(f"\n{'':>12s} {'Regular':>10s} {'Sarcasm':>10s} {'Irony':>10s} {'Figurative':>10s}")
for i, name in enumerate(CLASS_NAMES):
    row = "  ".join(f"{cm[i,j]:>8d}" for j in range(4))
    print(f"{name:>12s}   {row}")

# Classification report
report = generate_classification_report_str(test_metrics["labels"], test_metrics["predictions"])
print(f"\nClassification Report:\n{report}")

# 18. ERROR ANALYSIS
print("\n" + "=" * 70)
print("STEP 18: ERROR ANALYSIS")
print("=" * 70)

errors_df = collect_errors(
    test_df["text"].values, test_metrics["labels"],
    test_metrics["predictions"], test_metrics["probabilities"],
)
print(f"Collected {len(errors_df)} error examples")
if len(errors_df) > 0:
    print("\nSample errors:")
    for _, row in errors_df.head(5).iterrows():
        print(f"  [{row['actual']} → {row['predicted']}] (conf={row['confidence']:.2f})")
        print(f"    {row['text'][:80]}...")

# ============================================================
# 19. ARCHITECTURE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("STEP 19: ARCHITECTURE COMPARISON")
print("=" * 70)

print(f"\n{'Experiment':<15s} {'Embedding':<20s} {'LSTM':<6s} {'Val Acc':<10s} {'Val Macro F1':<12s}")
print("-" * 65)
for exp_id, data in all_val_results.items():
    m = data["metrics"]
    c = data["config"]
    embed_str = "Fine-tuned" if c["trainable"] else "Frozen"
    print(f"{exp_id:<15s} {embed_str:<20s} {c['lstm']:<6d} {m['accuracy']:<10.4f} {m['macro_f1']:<12.4f}")

# ============================================================
# 20. COMPARISON WITH CLASSICAL MODELS
# ============================================================

print("\n" + "=" * 70)
print("STEP 20: COMPARISON WITH CLASSICAL MODELS")
print("=" * 70)

comparison = []

p4_path = BASE_DIR / "results" / "classical" / "model_results.csv"
if p4_path.exists():
    p4 = pd.read_csv(str(p4_path))
    p4_test = p4[p4["split"] == "test"]
    best_p4 = p4_test.loc[p4_test["macro_f1"].idxmax()]
    comparison.append({
        "Phase": "Phase 4",
        "Model": f"BoW + {best_p4['model']}",
        "Accuracy": best_p4["accuracy"],
        "Macro_F1": best_p4["macro_f1"],
    })

p5_path = BASE_DIR / "results" / "tfidf" / "model_results.csv"
if p5_path.exists():
    p5 = pd.read_csv(str(p5_path))
    p5_test = p5[p5["split"] == "test"]
    best_p5 = p5_test.loc[p5_test["macro_f1"].idxmax()]
    comparison.append({
        "Phase": "Phase 5",
        "Model": f"TF-IDF + {best_p5['model']}",
        "Accuracy": best_p5["accuracy"],
        "Macro_F1": best_p5["macro_f1"],
    })

comparison.append({
    "Phase": "Phase 6",
    "Model": f"GloVe + Bi-LSTM ({best_exp['id']})",
    "Accuracy": test_metrics["accuracy"],
    "Macro_F1": test_metrics["macro_f1"],
})

comp_df = pd.DataFrame(comparison)
print(f"\n{comp_df.to_string(index=False)}")

# ============================================================
# 21. CONCLUSIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 21: CONCLUSIONS")
print("=" * 70)

print("""
PHASE 6 KEY FINDINGS:

1. BEST BI-LSTM CONFIGURATION:
   {best_id} ({best_desc})
   Validation Macro F1: {val_f1:.4f}
   Test Macro F1:       {test_f1:.4f}
   Test Accuracy:       {test_acc:.4f}

2. RESEARCH QUESTION:
   "Does sequence modeling with GloVe + Bi-LSTM provide an advantage
   over traditional sparse text representations?"

   ANSWER: No. The Bi-LSTM achieves higher accuracy but LOWER Macro F1
   than both BoW ({p4_f1}) and TF-IDF ({p5_f1}) approaches.

3. FIGURATIVE CLASS:
   Catastrophic failure (F1 ≈ 0.00). The model essentially ignores
   the Figurative class, assigning nearly all samples to Sarcasm or Irony.

4. GLOVE COVERAGE:
   Only {coverage:.1f}% of vocabulary matched. Social media language
   is poorly represented in GloVe embeddings.

5. NEXT STEPS:
   Phase 7 (FastText/Attention) may address the low vocabulary coverage
   issue through subword embeddings.
""".format(
    best_id=best_exp["id"],
    best_desc=best_exp["desc"],
    val_f1=best_val_f1,
    test_f1=test_metrics["macro_f1"],
    test_acc=test_metrics["accuracy"],
    p4_f1=comparison[0]["Macro_F1"] if len(comparison) > 0 else "N/A",
    p5_f1=comparison[1]["Macro_F1"] if len(comparison) > 1 else "N/A",
    coverage=coverage_stats["coverage_pct"],
))

print("=" * 70)
print("PHASE 6 COMPLETE")
print("=" * 70)
