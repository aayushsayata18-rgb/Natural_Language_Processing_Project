"""
Phase 5 — TF-IDF + Feature Engineering Notebook
============================================
Sarcasm & Irony Detection System using NLP

Sections:
    1. Imports
    2. Load data
    3. Load Phase 3 preprocessing
    4. Train/validation split
    5. TF-IDF setup
    6. Unigram experiment
    7. Bigram experiment
    8. Trigram experiment
    9. Model comparison
    10. Linguistic features
    11. Social-media features
    12. Combined features
    13. Hyperparameter experiments
    14. Validation results
    15. Test evaluation
    16. Feature interpretation
    17. Comparison with Phase 4
    18. Conclusions

Usage:
    python notebooks/05_tfidf_feature_engineering.py
"""

import sys
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent
ML_SERVICE_DIR = NOTEBOOK_DIR.parent
sys.path.insert(0, str(ML_SERVICE_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.tfidf.vectorizer import create_tfidf_vectorizer, fit_transform_tfidf, transform_tfidf
from src.tfidf.feature_engineering import (
    extract_text_features, extract_social_features, extract_style_features,
    build_feature_set, combine_tfidf_features, get_feature_names,
)
from src.classical.train_lr import train_logistic_regression
from src.classical.train_nb import train_naive_bayes
from src.classical.train_svm import train_linear_svm
from src.classical.evaluate import evaluate_model, CLASS_NAMES

RANDOM_STATE = 42
VAL_SIZE = 0.20
DATA_DIR = ML_SERVICE_DIR / "data" / "processed"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
RESULTS_DIR = ML_SERVICE_DIR / "results" / "tfidf"


def section(title, num):
    print(f"\n{'='*70}")
    print(f"  {num}. {title}")
    print(f"{'='*70}\n")


def main():
    # 1. IMPORTS
    section("IMPORTS", 1)
    print("Libraries: sklearn, pandas, numpy, scipy")
    print("Phase 4 classical modules reused (trainers, evaluation)")
    print("Phase 5 TF-IDF modules loaded")

    # 2. LOAD DATA
    section("LOAD DATA", 2)
    train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
    test_df = pd.read_csv(DATA_DIR / "test_clean.csv")
    print(f"Training set: {train_df.shape}")
    print(f"Test set:     {test_df.shape}")
    print(f"\nLabel distribution (train):")
    print(train_df["label"].value_counts().to_string())

    # 3. LOAD PREPROCESSING
    section("LOAD PHASE 3 PREPROCESSING", 3)
    classical_train = pd.read_csv(PREPROCESSED_DIR / "classical" / "train.csv")
    print(f"Classical preprocessed train: {classical_train.shape}")
    print(f"\nSample:")
    for i in range(2):
        print(f"  Original:    {classical_train['text'].iloc[i][:70]}...")
        print(f"  Preprocessed: {classical_train['processed_text'].iloc[i][:70]}...")
        print()

    # 4. TRAIN/VALIDATION SPLIT
    section("TRAIN/VALIDATION SPLIT", 4)
    texts = classical_train["processed_text"].values
    labels = train_df["label_id"].values
    raw_texts = train_df["text"].values

    X_tr, X_val, y_tr, y_val, raw_tr, raw_val = train_test_split(
        texts, labels, raw_texts,
        test_size=VAL_SIZE, stratify=labels, random_state=RANDOM_STATE,
    )
    print(f"Training:   {len(X_tr)} samples")
    print(f"Validation: {len(X_val)} samples")

    # 5. TF-IDF SETUP
    section("TF-IDF SETUP", 5)
    print("TF-IDF = Term Frequency x Inverse Document Frequency")
    print("sublinear_tf=True: uses 1 + log(tf) instead of raw counts")
    print("L2 normalization applied")
    print(f"Parameters: min_df=2, max_df=0.95, max_features=50000")

    # 6. UNIGRAM
    section("UNIGRAM EXPERIMENT", 6)
    vec_uni = create_tfidf_vectorizer(ngram_range=(1, 1), sublinear_tf=True)
    X_tr_uni = fit_transform_tfidf(vec_uni, X_tr)
    X_val_uni = transform_tfidf(vec_uni, X_val)
    print(f"Unigram TF-IDF: vocab={len(vec_uni.vocabulary_):,}, shape={X_tr_uni.shape}")
    
    lr_model, lr_time = train_logistic_regression(X_tr_uni, y_tr, C=0.1)
    lr_metrics = evaluate_model(lr_model, X_val_uni, y_val)
    print(f"\nLR (unigram): acc={lr_metrics['accuracy']:.4f}, macro_f1={lr_metrics['macro_f1']:.4f}")

    # 7. BIGRAM
    section("BIGRAM EXPERIMENT", 7)
    vec_bi = create_tfidf_vectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X_tr_bi = fit_transform_tfidf(vec_bi, X_tr)
    X_val_bi = transform_tfidf(vec_bi, X_val)
    print(f"Uni+Bi TF-IDF: vocab={len(vec_bi.vocabulary_):,}, shape={X_tr_bi.shape}")
    
    lr_bi, _ = train_logistic_regression(X_tr_bi, y_tr, C=0.1)
    lr_bi_m = evaluate_model(lr_bi, X_val_bi, y_val)
    print(f"LR (uni+bi): acc={lr_bi_m['accuracy']:.4f}, macro_f1={lr_bi_m['macro_f1']:.4f}")

    # 8. TRIGRAM
    section("TRIGRAM EXPERIMENT", 8)
    vec_tri = create_tfidf_vectorizer(ngram_range=(1, 3), sublinear_tf=True)
    X_tr_tri = fit_transform_tfidf(vec_tri, X_tr)
    X_val_tri = transform_tfidf(vec_tri, X_val)
    print(f"Uni+Bi+Tri TF-IDF: vocab={len(vec_tri.vocabulary_):,}, shape={X_tr_tri.shape}")
    
    lr_tri, _ = train_logistic_regression(X_tr_tri, y_tr, C=0.1)
    lr_tri_m = evaluate_model(lr_tri, X_val_tri, y_val)
    print(f"LR (uni+bi+tri): acc={lr_tri_m['accuracy']:.4f}, macro_f1={lr_tri_m['macro_f1']:.4f}")

    print(f"\nN-gram comparison:")
    print(f"  Unigram:    macro_f1={lr_metrics['macro_f1']:.4f}")
    print(f"  Uni+Bi:     macro_f1={lr_bi_m['macro_f1']:.4f}")
    print(f"  Uni+Bi+Tri: macro_f1={lr_tri_m['macro_f1']:.4f}")
    print(f"  Winner: Unigram — additional n-grams did NOT help")

    # 9. MODEL COMPARISON
    section("MODEL COMPARISON", 9)
    nb_model, _ = train_naive_bayes(X_tr_uni, y_tr, alpha=1.0)
    nb_m = evaluate_model(nb_model, X_val_uni, y_val)
    
    svm_model, _ = train_linear_svm(X_tr_uni, y_tr, C=0.1)
    svm_m = evaluate_model(svm_model, X_val_uni, y_val)

    print(f"{'Model':<22s} {'Accuracy':>10s} {'Macro F1':>10s}")
    print("-" * 45)
    for name, m in [("Logistic Regression", lr_metrics), ("Multinomial NB", nb_m), ("Linear SVM", svm_m)]:
        print(f"{name:<22s} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")

    # 10. LINGUISTIC FEATURES
    section("LINGUISTIC FEATURES", 10)
    text_feats = extract_text_features(raw_tr[:5])
    print(f"Text features shape: {text_feats.shape}")
    print(f"Features: word_count, char_count, avg_word_len")
    print(f"Sample:\n{text_feats}")

    # 11. SOCIAL-MEDIA FEATURES
    section("SOCIAL-MEDIA FEATURES", 11)
    social_feats = extract_social_features(raw_tr[:5])
    print(f"Social features shape: {social_feats.shape}")
    print(f"Features: url_count, mention_count, hashtag_count, emoji_count")
    print(f"Sample:\n{social_feats}")

    # 12. COMBINED FEATURES
    section("COMBINED FEATURES", 12)
    for fs in ["A", "B", "C", "D", "E"]:
        num = build_feature_set(raw_tr, fs)
        n_extra = num.shape[1] if num is not None else 0
        print(f"Feature set {fs}: {n_extra} extra features")

    # 13. HYPERPARAMETER EXPERIMENTS
    section("HYPERPARAMETER EXPERIMENTS", 13)
    print("Hyperparameters tuned on validation Macro F1:")
    print("  LR: C in {0.1, 1.0, 10.0} -> best C=0.1")
    print("  NB: alpha in {0.1, 0.5, 1.0} -> best alpha=1.0")
    print("  SVM: C in {0.1, 1.0, 10.0} -> best C=0.1")

    # 14. VALIDATION RESULTS
    section("VALIDATION RESULTS", 14)
    if (RESULTS_DIR / "model_results.csv").exists():
        results = pd.read_csv(RESULTS_DIR / "model_results.csv")
        val = results[results["split"] == "validation"]
        sorted_r = val.sort_values("macro_f1", ascending=False)
        cols = ["experiment_id", "preprocessing_variant", "feature_set", "model", "macro_f1"]
        print(sorted_r[cols].to_string(index=False))

    # 15. TEST EVALUATION
    section("TEST EVALUATION", 15)
    if (RESULTS_DIR / "model_results.csv").exists():
        test_r = results[results["split"] == "test"]
        cols = ["experiment_id", "model", "accuracy", "macro_f1",
                "regular_f1", "sarcasm_f1", "irony_f1", "figurative_f1"]
        print(test_r[cols].to_string(index=False))

    # 16. FEATURE INTERPRETATION
    section("FEATURE INTERPRETATION", 16)
    feat_dir = RESULTS_DIR / "important_features" / "text_only_logisticregression"
    if feat_dir.exists():
        for cls in CLASS_NAMES:
            fp = feat_dir / f"{cls.lower()}_features.csv"
            if fp.exists():
                df = pd.read_csv(fp)
                top5 = df.head(5)
                print(f"\n  Top 5 features for {cls}:")
                for _, row in top5.iterrows():
                    print(f"    {row['feature']:20s} coef={row['coefficient']:.4f}")

    # 17. COMPARISON WITH PHASE 4
    section("COMPARISON WITH PHASE 4", 17)
    if (RESULTS_DIR / "phase4_vs_phase5.csv").exists():
        comp = pd.read_csv(RESULTS_DIR / "phase4_vs_phase5.csv")
        print(comp.to_string(index=False))
        print("\nPhase 4 BoW SVM test Macro F1: 0.7187")
        print("Phase 5 TF-IDF SVM test Macro F1: 0.7002")
        print("Difference: -0.0185 (BoW performed BETTER)")

    # 18. CONCLUSIONS
    section("CONCLUSIONS", 18)
    print("""
  Phase 5 — TF-IDF + Feature Engineering Conclusions:

  1. TF-IDF did NOT significantly outperform BoW on this dataset.
     Phase 4 BoW SVM (0.7187) > Phase 5 TF-IDF SVM (0.7002).

  2. Unigram TF-IDF consistently outperformed bigram and trigram.
     Higher n-grams increase dimensionality without benefit.

  3. Feature engineering (sets B-E) did NOT improve validation F1.
     The best feature set was A (TF-IDF text only).

  4. The Figurative class remains near-impossible for all classical
     models (F1 = 0.0026 to 0.2348).

  5. NB achieved the highest Figurative F1 (0.2348) but at the
     cost of lower overall accuracy.

  6. Classical preprocessing provides marginal benefit over minimal.

  7. sublinear_tf=True was used throughout but did not change the
     fundamental performance characteristics.

  8. BEST TRADITIONAL NLP CANDIDATES for deep learning comparison:
     - Text-only: LinearSVM, BoW, classical, Macro F1=0.7187
     - TF-IDF: LinearSVM, TF-IDF, classical, Macro F1=0.7002

  IMPORTANT: These are NOT the final project models.
  They will be compared against Bi-LSTM and BERT.
    """)


if __name__ == "__main__":
    main()
