"""
Phase 4 — Bag of Words + Classical ML Notebook
============================================
Sarcasm & Irony Detection System using NLP

This script demonstrates the complete Phase 4 pipeline
including BoW vectorization, model training, evaluation,
and comparison.

Sections:
    1. Imports
    2. Load clean data
    3. Load preprocessing configuration
    4. Train/validation split
    5. BoW setup
    6. Logistic Regression
    7. Naive Bayes
    8. Linear SVM
    9. Experiment comparison
    10. Validation metrics
    11. Confusion matrices
    12. Error examples
    13. Final selected baseline
    14. Test evaluation
    15. Conclusions

Usage:
    python notebooks/04_bow_classical_ml.py
"""

import sys
from pathlib import Path

# Setup path
NOTEBOOK_DIR = Path(__file__).resolve().parent
ML_SERVICE_DIR = NOTEBOOK_DIR.parent
sys.path.insert(0, str(ML_SERVICE_DIR))

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing.config import get_variant
from src.classical.bow import create_bow_vectorizer, fit_transform_bow, transform_bow, get_feature_info
from src.classical.train_lr import train_logistic_regression
from src.classical.train_nb import train_naive_bayes
from src.classical.train_svm import train_linear_svm
from src.classical.evaluate import evaluate_model, generate_classification_report, collect_error_examples, CLASS_NAMES


RANDOM_STATE = 42
VAL_SIZE = 0.20

# Paths
DATA_DIR = ML_SERVICE_DIR / "data" / "processed"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
RESULTS_DIR = ML_SERVICE_DIR / "results" / "classical"


def section(title, num):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {num}. {title}")
    print(f"{'='*70}\n")


def main():
    # ============================================================
    # 1. IMPORTS
    # ============================================================
    section("IMPORTS", 1)
    print("Libraries: sklearn, pandas, numpy, nltk")
    print("Phase 3 preprocessing modules loaded")
    print("Phase 4 classical ML modules loaded")

    # ============================================================
    # 2. LOAD CLEAN DATA
    # ============================================================
    section("LOAD CLEAN DATA", 2)

    train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
    test_df = pd.read_csv(DATA_DIR / "test_clean.csv")

    print(f"Training set: {train_df.shape}")
    print(f"Test set:     {test_df.shape}")
    print(f"\nLabel distribution (train):")
    print(train_df["label"].value_counts().to_string())
    print(f"\nLabel mapping: Regular=0, Sarcasm=1, Irony=2, Figurative=3")

    # ============================================================
    # 3. LOAD PREPROCESSING CONFIGURATION
    # ============================================================
    section("LOAD PREPROCESSING CONFIGURATION", 3)

    # Load preprocessed minimal variant
    minimal_train = pd.read_csv(PREPROCESSED_DIR / "minimal" / "train.csv")
    print(f"Preprocessed minimal train: {minimal_train.shape}")
    print(f"Columns: {minimal_train.columns.tolist()}")
    print(f"\nSample preprocessed text:")
    for i in range(3):
        print(f"  Original:    {minimal_train['text'].iloc[i][:80]}...")
        print(f"  Preprocessed: {minimal_train['processed_text'].iloc[i][:80]}...")
        print()

    # ============================================================
    # 4. TRAIN/VALIDATION SPLIT
    # ============================================================
    section("TRAIN/VALIDATION SPLIT", 4)

    texts = minimal_train["processed_text"].values
    labels = train_df["label_id"].values

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        texts, labels, test_size=VAL_SIZE, stratify=labels, random_state=RANDOM_STATE
    )

    print(f"Training:   {len(X_train_text)} samples")
    print(f"Validation: {len(X_val_text)} samples")
    print(f"\nTrain label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Val label distribution:   {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"\nStratification confirmed: proportions match")

    # ============================================================
    # 5. BAG OF WORDS SETUP
    # ============================================================
    section("BAG OF WORDS SETUP", 5)

    # Unigram
    vec_uni = create_bow_vectorizer(ngram_range=(1, 1))
    X_train_uni = fit_transform_bow(vec_uni, X_train_text)
    X_val_uni = transform_bow(vec_uni, X_val_text)

    print(f"Unigram BoW:")
    print(f"  Vocabulary: {len(vec_uni.vocabulary_):,}")
    print(f"  Train matrix: {X_train_uni.shape}")
    print(f"  Sparsity: {1.0 - X_train_uni.nnz / (X_train_uni.shape[0] * X_train_uni.shape[1]):.4f}")

    # Bigram
    vec_bi = create_bow_vectorizer(ngram_range=(1, 2))
    X_train_bi = fit_transform_bow(vec_bi, X_train_text)
    X_val_bi = transform_bow(vec_bi, X_val_text)

    print(f"\nUnigram + Bigram BoW:")
    print(f"  Vocabulary: {len(vec_bi.vocabulary_):,}")
    print(f"  Train matrix: {X_train_bi.shape}")

    print(f"\nNote: max_features=50,000 caps vocabulary for bi/trigram")

    # ============================================================
    # 6. LOGISTIC REGRESSION
    # ============================================================
    section("LOGISTIC REGRESSION", 6)

    print("Training LR with C=0.1 (best from hyperparameter search)...")
    lr_model, lr_time = train_logistic_regression(X_train_uni, y_train, C=0.1)
    lr_metrics = evaluate_model(lr_model, X_val_uni, y_val)

    print(f"\nLogistic Regression (Unigram, C=0.1):")
    print(f"  Accuracy:        {lr_metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {lr_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {lr_metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {lr_metrics['macro_f1']:.4f}")
    print(f"  Training time:   {lr_time:.2f}s")
    print(f"\n  Per-class F1:")
    for cls, m in lr_metrics["per_class"].items():
        print(f"    {cls:12s}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")

    # ============================================================
    # 7. NAIVE BAYES
    # ============================================================
    section("NAIVE BAYES", 7)

    print("Training MultinomialNB with alpha=1.0 (best from search)...")
    nb_model, nb_time = train_naive_bayes(X_train_uni, y_train, alpha=1.0)
    nb_metrics = evaluate_model(nb_model, X_val_uni, y_val)

    print(f"\nMultinomial NB (Unigram, alpha=1.0):")
    print(f"  Accuracy:        {nb_metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {nb_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {nb_metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {nb_metrics['macro_f1']:.4f}")
    print(f"  Training time:   {nb_time:.2f}s")
    print(f"\n  Per-class F1:")
    for cls, m in nb_metrics["per_class"].items():
        print(f"    {cls:12s}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")

    # ============================================================
    # 8. LINEAR SVM
    # ============================================================
    section("LINEAR SVM", 8)

    print("Training LinearSVM with C=0.1 (best from search)...")
    svm_model, svm_time = train_linear_svm(X_train_uni, y_train, C=0.1)
    svm_metrics = evaluate_model(svm_model, X_val_uni, y_val)

    print(f"\nLinear SVM (Unigram, C=0.1):")
    print(f"  Accuracy:        {svm_metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {svm_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {svm_metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {svm_metrics['macro_f1']:.4f}")
    print(f"  Training time:   {svm_time:.2f}s")
    print(f"\n  Per-class F1:")
    for cls, m in svm_metrics["per_class"].items():
        print(f"    {cls:12s}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")
    print(f"\n  Note: LinearSVC does not support predict_proba()")

    # ============================================================
    # 9. EXPERIMENT COMPARISON
    # ============================================================
    section("EXPERIMENT COMPARISON", 9)

    # Load saved results
    results_path = RESULTS_DIR / "model_results.csv"
    if results_path.exists():
        all_results = pd.read_csv(results_path)
        val_results = all_results[all_results["split"] == "validation"]
        print(f"Total experiments: {len(val_results)}")
        print(f"\nValidation Results (sorted by Macro F1):")
        sorted_results = val_results.sort_values("macro_f1", ascending=False)
        cols = ["experiment_id", "preprocessing", "ngram_range", "model", "accuracy", "macro_f1"]
        print(sorted_results[cols].to_string(index=False))
    else:
        print("Run src/classical/run_experiments.py first to generate results.")

    # ============================================================
    # 10. VALIDATION METRICS
    # ============================================================
    section("VALIDATION METRICS", 10)

    models_data = [
        ("Logistic Regression", lr_metrics),
        ("Multinomial NB", nb_metrics),
        ("Linear SVM", svm_metrics),
    ]

    print(f"{'Model':<22s} {'Accuracy':>10s} {'Macro P':>10s} {'Macro R':>10s} {'Macro F1':>10s}")
    print("-" * 65)
    for name, m in models_data:
        print(f"{name:<22s} {m['accuracy']:>10.4f} {m['macro_precision']:>10.4f} "
              f"{m['macro_recall']:>10.4f} {m['macro_f1']:>10.4f}")

    # ============================================================
    # 11. CONFUSION MATRICES
    # ============================================================
    section("CONFUSION MATRICES", 11)

    for name, model in [("LR", lr_model), ("NB", nb_model), ("SVM", svm_model)]:
        y_pred = model.predict(X_val_uni)
        cm = confusion_matrix(y_val, y_pred)
        print(f"\n  {name} Confusion Matrix (Validation):")
        print(f"  {'':12s}", end="")
        for c in CLASS_NAMES:
            print(f" {c[:6]:>8s}", end="")
        print()
        for i, row_name in enumerate(CLASS_NAMES):
            print(f"  {row_name:12s}", end="")
            for j in range(len(CLASS_NAMES)):
                print(f" {cm[i][j]:>8d}", end="")
            print()

    # ============================================================
    # 12. ERROR EXAMPLES
    # ============================================================
    section("ERROR EXAMPLES", 12)

    print("Collecting misclassified examples from LR model...")
    errors = collect_error_examples(lr_model, X_val_uni, y_val, X_val_text, n=10)
    
    if len(errors) > 0:
        print(f"\nSample misclassifications ({len(errors)} examples):")
        for _, row in errors.head(10).iterrows():
            print(f"  Actual: {row['actual_label']:12s} -> Predicted: {row['predicted_label']:12s}")
            print(f"  Text: {row['text'][:80]}...")
            print()

    # ============================================================
    # 13. FINAL SELECTED BASELINE
    # ============================================================
    section("FINAL SELECTED BASELINE", 13)

    if results_path.exists():
        val_results = pd.read_csv(results_path)
        val_only = val_results[val_results["split"] == "validation"]
        best = val_only.loc[val_only["macro_f1"].idxmax()]

        print(f"Best current classical baseline (validation):")
        print(f"  Experiment:     {best['experiment_id']}")
        print(f"  Model:          {best['model']}")
        print(f"  Preprocessing:  {best['preprocessing']}")
        print(f"  N-gram range:   {best['ngram_range']}")
        print(f"  Macro F1:       {best['macro_f1']:.4f}")
        print(f"  Accuracy:       {best['accuracy']:.4f}")
        print(f"\n  NOTE: This is NOT the final project model.")
        print(f"  It is the best current classical baseline.")

    # ============================================================
    # 14. TEST EVALUATION
    # ============================================================
    section("TEST EVALUATION", 14)

    if results_path.exists():
        all_results = pd.read_csv(results_path)
        test_results = all_results[all_results["split"] == "test"]
        
        if len(test_results) > 0:
            print(f"Test Set Results (final models retrained on full training data):")
            cols = ["experiment_id", "model", "accuracy", "macro_f1",
                    "regular_f1", "sarcasm_f1", "irony_f1", "figurative_f1"]
            print(test_results[cols].to_string(index=False))
        else:
            print("No test results found. Run run_experiments.py first.")

    # ============================================================
    # 15. CONCLUSIONS
    # ============================================================
    section("CONCLUSIONS", 15)

    print("""
  Phase 4 — BoW + Classical ML Conclusions:

  1. 16 controlled experiments were run across 3 models and 4
     preprocessing variants.

  2. Unigram BoW consistently outperformed bigram and trigram
     representations. Adding n-grams increased dimensionality
     without improving classification.

  3. The Figurative class is extremely difficult for all classical
     models — near-zero F1 in most configurations. This class
     may require richer representations (TF-IDF, embeddings).

  4. Regular class achieves near-perfect F1 (~1.0) across all
     models, likely due to distinctive URL/link patterns.

  5. Classical preprocessing (lowercase, stopword removal, 
     lemmatization) slightly improved LR performance over
     minimal preprocessing.

  6. LinearSVM achieved the highest test Macro F1, but all
     three models perform similarly.

  7. These baselines will be compared against:
     - Phase 5: TF-IDF + N-grams
     - Phase 6: Bi-LSTM
     - Phase 8: BERT

  IMPORTANT: No classical model is declared the final project model.
  These are baselines for comparison.
    """)


if __name__ == "__main__":
    main()
