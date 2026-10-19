"""
Phase 5 - Run All TF-IDF Experiments
============================================
Sarcasm & Irony Detection System using NLP

Main entry point for Phase 5.
Orchestrates:
    Stage 1: Core TF-IDF experiments (TFIDF-01 to TFIDF-08)
    Stage 2: Feature set experiments (FEAT-01 to FEAT-05)
    Stage 3: Preprocessing comparison (PREP-01 to PREP-04)
    Test evaluation for top candidates
    Visualizations and Phase 4 comparison

Usage:
    python src/tfidf/run_experiments.py
"""

import io
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import sparse
from sklearn.model_selection import train_test_split

# --- Path Setup ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ml-service/
sys.path.insert(0, str(BASE_DIR))

from src.tfidf.vectorizer import (
    create_tfidf_vectorizer, fit_transform_tfidf, transform_tfidf,
)
from src.tfidf.feature_engineering import (
    build_feature_set, combine_tfidf_features, get_feature_names,
)
from src.tfidf.experiments import (
    CORE_EXPERIMENTS, run_single_experiment,
    determine_best_ngram, determine_best_model, _train_model,
)
from src.tfidf.feature_analysis import (
    extract_feature_importance, save_feature_importance,
    create_feature_analysis_row,
)
from src.classical.evaluate import (
    evaluate_model, generate_classification_report, collect_error_examples,
    format_results_row, CLASS_NAMES,
)
from src.classical.save_artifacts import (
    save_model_artifacts, save_confusion_matrix, save_classification_report,
)

# --- Logging ---
_utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(_utf8_stdout)],
)
logger = logging.getLogger(__name__)

# --- Paths ---
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PREPROCESSED_DIR = PROCESSED_DIR / "preprocessed"
MODELS_DIR = BASE_DIR / "models" / "classical" / "tfidf"
RESULTS_DIR = BASE_DIR / "results" / "tfidf"
REPORTS_DIR = RESULTS_DIR / "classification_reports"
CM_DIR = RESULTS_DIR / "confusion_matrices"
VIZ_DIR = RESULTS_DIR / "visualizations"
FEATURES_DIR = RESULTS_DIR / "important_features"

RANDOM_STATE = 42
VAL_SIZE = 0.20


def load_data():
    """Load Phase 1 clean datasets."""
    train_df = pd.read_csv(PROCESSED_DIR / "train_clean.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_clean.csv")
    logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df


def load_preprocessed_texts(variant_name, split="train"):
    """Load pre-computed preprocessed texts."""
    path = PREPROCESSED_DIR / variant_name / f"{split}.csv"
    df = pd.read_csv(path)
    logger.info(f"Loaded preprocessed '{variant_name}/{split}': {df.shape}")
    return df["processed_text"]


def create_split(texts, labels, raw_texts=None):
    """Create stratified train/validation split."""
    if raw_texts is not None:
        X_tr, X_val, y_tr, y_val, raw_tr, raw_val = train_test_split(
            texts, labels, raw_texts,
            test_size=VAL_SIZE, stratify=labels, random_state=RANDOM_STATE,
        )
        return X_tr, X_val, y_tr, y_val, raw_tr, raw_val
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            texts, labels,
            test_size=VAL_SIZE, stratify=labels, random_state=RANDOM_STATE,
        )
        return X_tr, X_val, y_tr, y_val


def plot_confusion_matrix(cm, class_names, title, filepath):
    """Generate and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        ylabel="Actual", xlabel="Predicted", title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results_df, metric, title, filepath):
    """Bar chart comparing a metric across experiments."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(results_df))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))
    bars = ax.bar(x, results_df[metric], color=colors)
    ax.set_xlabel("Experiment")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["experiment_id"], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, min(1.0, results_df[metric].max() + 0.05))
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    logger.info("=" * 70)
    logger.info("PHASE 5 — TF-IDF + N-GRAMS + FEATURE ENGINEERING")
    logger.info("=" * 70)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    for d in [MODELS_DIR, RESULTS_DIR, REPORTS_DIR, CM_DIR, VIZ_DIR, FEATURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: LOAD DATA")
    logger.info("=" * 70)

    train_df, test_df = load_data()
    labels = train_df["label_id"].values
    raw_train_texts = train_df["text"].values  # Original texts for feature extraction

    # ============================================================
    # STEP 2: STAGE 1 — CORE TF-IDF EXPERIMENTS
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: STAGE 1 — CORE TF-IDF EXPERIMENTS (TFIDF-01 to TFIDF-08)")
    logger.info("=" * 70)

    all_results = []
    feature_analysis = []
    best_models_stage1 = {}  # model_name -> (model, vectorizer, result, exp_id, hp)

    # Load classical preprocessing
    texts_classical = load_preprocessed_texts("classical", "train")
    X_tr_text, X_val_text, y_train, y_val, raw_tr, raw_val = create_split(
        texts_classical.values, labels, raw_train_texts,
    )
    logger.info(f"Split: train={len(X_tr_text)}, val={len(X_val_text)}")

    # Cache vectorizers
    vec_cache = {}
    X_tr_cache = {}
    X_val_cache = {}

    for exp in CORE_EXPERIMENTS:
        ngram_key = str(exp["ngram_range"])

        if ngram_key not in vec_cache:
            vec = create_tfidf_vectorizer(
                ngram_range=exp["ngram_range"],
                sublinear_tf=exp.get("sublinear_tf", True),
            )
            X_tr = fit_transform_tfidf(vec, X_tr_text)
            X_val = transform_tfidf(vec, X_val_text)
            vec_cache[ngram_key] = vec
            X_tr_cache[ngram_key] = X_tr
            X_val_cache[ngram_key] = X_val
        else:
            vec = vec_cache[ngram_key]
            X_tr = X_tr_cache[ngram_key]
            X_val = X_val_cache[ngram_key]

        vocab_size = len(vec.vocabulary_)
        sparsity = 1.0 - X_tr.nnz / (X_tr.shape[0] * X_tr.shape[1])

        result_row, best_model, best_hp = run_single_experiment(
            exp, X_tr, y_train, X_val, y_val,
            vocab_size=vocab_size, total_features=X_tr.shape[1],
        )
        all_results.append(result_row)

        feature_analysis.append(create_feature_analysis_row(
            exp["id"], f"TF-IDF {ngram_key}", X_tr.shape[1],
            vocab_size, sparsity, result_row["training_time"],
        ))

        model_name = exp["model"]
        if model_name not in best_models_stage1 or result_row["macro_f1"] > best_models_stage1[model_name][2]["macro_f1"]:
            best_models_stage1[model_name] = (best_model, vec, result_row, exp["id"], best_hp)

    # Print Stage 1 summary
    logger.info("\nStage 1 Results:")
    for r in all_results:
        logger.info(f"  {r['experiment_id']}: {r['model']}, ngram={r['ngram_range']}, "
                     f"macro_f1={r['macro_f1']:.4f}")

    # ============================================================
    # STEP 3: DETERMINE BEST CONFIG FOR STAGE 2
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: DETERMINE BEST N-GRAM AND MODEL")
    logger.info("=" * 70)

    best_ngram = determine_best_ngram(all_results)
    best_model_name = determine_best_model(all_results)
    logger.info(f"Best n-gram: {best_ngram}")
    logger.info(f"Best model: {best_model_name}")

    # Get best hp for this model
    best_stage1 = best_models_stage1[best_model_name]
    best_hp_val = best_stage1[4]
    hp_key = "C" if best_model_name != "MultinomialNB" else "alpha"

    # ============================================================
    # STEP 4: STAGE 2 — FEATURE SET EXPERIMENTS
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: STAGE 2 — FEATURE SET EXPERIMENTS (FEAT-01 to FEAT-05)")
    logger.info("=" * 70)

    # Get the TF-IDF matrices for best n-gram
    ngram_key = str(best_ngram)
    if ngram_key in vec_cache:
        vec_best = vec_cache[ngram_key]
        X_tr_tfidf = X_tr_cache[ngram_key]
        X_val_tfidf = X_val_cache[ngram_key]
    else:
        vec_best = create_tfidf_vectorizer(ngram_range=best_ngram, sublinear_tf=True)
        X_tr_tfidf = fit_transform_tfidf(vec_best, X_tr_text)
        X_val_tfidf = transform_tfidf(vec_best, X_val_text)

    vocab_size_best = len(vec_best.vocabulary_)
    best_feature_model = None
    best_feature_result = None
    best_feature_scaler = None
    best_feature_set_name = "A"

    for fs_name in ["A", "B", "C", "D", "E"]:
        feat_id = f"FEAT-{['A','B','C','D','E'].index(fs_name)+1:02d}"
        logger.info(f"\n--- Feature Set {fs_name}: {feat_id} ---")

        # Build numeric features from RAW text
        num_train = build_feature_set(raw_tr, fs_name)
        num_val = build_feature_set(raw_val, fs_name)

        # Combine
        if num_train is not None:
            X_tr_combined, scaler = combine_tfidf_features(
                X_tr_tfidf, num_train, fit_scaler=True
            )
            X_val_combined, _ = combine_tfidf_features(
                X_val_tfidf, num_val, fit_scaler=False, scaler=scaler
            )
        else:
            X_tr_combined = X_tr_tfidf
            X_val_combined = X_val_tfidf
            scaler = None

        total_features = X_tr_combined.shape[1]
        sparsity = 1.0 - X_tr_combined.nnz / (X_tr_combined.shape[0] * X_tr_combined.shape[1])

        # Check if NB can handle this (needs non-negative features)
        actual_model = best_model_name
        if best_model_name == "MultinomialNB" and num_train is not None:
            # MultinomialNB requires non-negative features; scaled features may be negative
            # Fall back to LinearSVM for feature experiments
            logger.info("  NB incompatible with scaled features, using LinearSVM")
            actual_model = "LinearSVM"
            hp_key_feat = "C"
            hp_val_feat = 0.1
        else:
            hp_key_feat = hp_key
            hp_val_feat = best_hp_val

        exp_config = {
            "id": feat_id, "preprocessing": "classical",
            "ngram_range": best_ngram, "model": actual_model,
            "hp_key": hp_key_feat, "hp_values": [hp_val_feat],
            "sublinear_tf": True, "feature_set": fs_name,
        }

        result_row, model, _ = run_single_experiment(
            exp_config, X_tr_combined, y_train, X_val_combined, y_val,
            vocab_size=vocab_size_best, total_features=total_features,
        )
        all_results.append(result_row)

        feat_desc = f"TF-IDF + FeatureSet-{fs_name}" if fs_name != "A" else f"TF-IDF only"
        n_numeric = num_train.shape[1] if num_train is not None else 0
        feature_analysis.append(create_feature_analysis_row(
            feat_id, feat_desc, total_features, vocab_size_best, sparsity,
            result_row["training_time"],
        ))

        if best_feature_result is None or result_row["macro_f1"] > best_feature_result["macro_f1"]:
            best_feature_model = model
            best_feature_result = result_row
            best_feature_scaler = scaler
            best_feature_set_name = fs_name

    logger.info(f"\nBest feature set: {best_feature_set_name}, macro_f1={best_feature_result['macro_f1']:.4f}")

    # ============================================================
    # STEP 5: STAGE 3 — PREPROCESSING COMPARISON
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: STAGE 3 — PREPROCESSING COMPARISON (PREP-01 to PREP-04)")
    logger.info("=" * 70)

    preproc_variants = ["minimal", "lowercase", "lemmatized", "classical"]
    best_preproc_model = None
    best_preproc_result = None
    best_preproc_name = "classical"
    best_preproc_vec = None
    best_preproc_scaler = None
    best_preproc_hp = None

    # Determine the model and hp for preprocessing experiments
    prep_model_name = best_feature_result["model"]
    prep_hp_key = "C" if prep_model_name != "MultinomialNB" else "alpha"
    prep_hp_val = eval(best_feature_result["hyperparameters"])[prep_hp_key]

    for i, preproc in enumerate(preproc_variants):
        prep_id = f"PREP-{i+1:02d}"
        logger.info(f"\n--- Preprocessing: {preproc} ({prep_id}) ---")

        texts_var = load_preprocessed_texts(preproc, "train")
        raw_texts_var = train_df["text"].values  # Raw text is always the same

        X_tr_txt, X_val_txt, y_tr, y_val_p, raw_tr_p, raw_val_p = create_split(
            texts_var.values, labels, raw_texts_var,
        )

        # TF-IDF
        vec_p = create_tfidf_vectorizer(ngram_range=best_ngram, sublinear_tf=True)
        X_tr_tfidf_p = fit_transform_tfidf(vec_p, X_tr_txt)
        X_val_tfidf_p = transform_tfidf(vec_p, X_val_txt)

        # Features
        num_tr = build_feature_set(raw_tr_p, best_feature_set_name)
        num_val_p = build_feature_set(raw_val_p, best_feature_set_name)

        if num_tr is not None:
            X_tr_comb, scaler_p = combine_tfidf_features(X_tr_tfidf_p, num_tr, fit_scaler=True)
            X_val_comb, _ = combine_tfidf_features(X_val_tfidf_p, num_val_p, fit_scaler=False, scaler=scaler_p)
        else:
            X_tr_comb = X_tr_tfidf_p
            X_val_comb = X_val_tfidf_p
            scaler_p = None

        vocab_p = len(vec_p.vocabulary_)
        total_f = X_tr_comb.shape[1]
        sparsity_p = 1.0 - X_tr_comb.nnz / (X_tr_comb.shape[0] * X_tr_comb.shape[1])

        exp_config = {
            "id": prep_id, "preprocessing": preproc,
            "ngram_range": best_ngram, "model": prep_model_name,
            "hp_key": prep_hp_key, "hp_values": [prep_hp_val],
            "sublinear_tf": True, "feature_set": best_feature_set_name,
        }

        result_row, model, _ = run_single_experiment(
            exp_config, X_tr_comb, y_tr, X_val_comb, y_val_p,
            vocab_size=vocab_p, total_features=total_f,
        )
        all_results.append(result_row)

        feature_analysis.append(create_feature_analysis_row(
            prep_id, f"TF-IDF+{best_feature_set_name} ({preproc})",
            total_f, vocab_p, sparsity_p, result_row["training_time"],
        ))

        if best_preproc_result is None or result_row["macro_f1"] > best_preproc_result["macro_f1"]:
            best_preproc_model = model
            best_preproc_result = result_row
            best_preproc_name = preproc
            best_preproc_vec = vec_p
            best_preproc_scaler = scaler_p
            best_preproc_hp = prep_hp_val

    logger.info(f"\nBest preprocessing: {best_preproc_name}, macro_f1={best_preproc_result['macro_f1']:.4f}")

    # ============================================================
    # STEP 6: SAVE VALIDATION RESULTS
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: SAVE VALIDATION RESULTS")
    logger.info("=" * 70)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "model_results.csv", index=False)
    logger.info(f"Saved {len(results_df)} results")

    fa_df = pd.DataFrame(feature_analysis)
    fa_df.to_csv(RESULTS_DIR / "feature_analysis.csv", index=False)

    logger.info("\nAll Validation Results:")
    display_cols = ["experiment_id", "preprocessing_variant", "feature_set",
                    "ngram_range", "model", "accuracy", "macro_f1"]
    logger.info("\n" + results_df[display_cols].to_string(index=False))

    # ============================================================
    # STEP 7: TEST EVALUATION
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: TEST EVALUATION")
    logger.info("=" * 70)

    test_results = []

    # Find the overall best text-only (feature_set A) and best with features
    val_only = results_df[results_df["split"] == "validation"]
    best_text_only = val_only[val_only["feature_set"] == "A"].sort_values("macro_f1", ascending=False).iloc[0]
    best_with_feat = val_only.sort_values("macro_f1", ascending=False).iloc[0]

    # Candidates for test: best per model type from stage 1, plus best feature model
    test_candidates = []

    # 1) Best text-only per model
    for model_name, (model, vec, result, exp_id, hp_val) in best_models_stage1.items():
        test_candidates.append({
            "label": f"text_only_{model_name.lower()}",
            "model_name": model_name,
            "preprocessing": result["preprocessing_variant"],
            "ngram_range": eval(result["ngram_range"]),
            "hp_key": "C" if model_name != "MultinomialNB" else "alpha",
            "hp_val": hp_val,
            "feature_set": "A",
            "exp_id": exp_id,
        })

    # 2) Best with features
    if best_feature_set_name != "A":
        test_candidates.append({
            "label": f"features_{best_feature_result['model'].lower()}",
            "model_name": best_feature_result["model"],
            "preprocessing": best_preproc_name,
            "ngram_range": best_ngram,
            "hp_key": prep_hp_key,
            "hp_val": best_preproc_hp if best_preproc_hp else prep_hp_val,
            "feature_set": best_feature_set_name,
            "exp_id": best_feature_result["experiment_id"] + "_feat",
        })

    for cand in test_candidates:
        logger.info(f"\n--- Test: {cand['label']} ---")

        preproc = cand["preprocessing"]
        train_texts_full = load_preprocessed_texts(preproc, "train")
        test_texts = load_preprocessed_texts(preproc, "test")
        raw_test = test_df["text"].values

        # Fit on full training
        vec_final = create_tfidf_vectorizer(ngram_range=cand["ngram_range"], sublinear_tf=True)
        X_train_full = fit_transform_tfidf(vec_final, train_texts_full)
        X_test = transform_tfidf(vec_final, test_texts)

        y_train_full = train_df["label_id"].values
        y_test = test_df["label_id"].values

        # Features
        fs = cand["feature_set"]
        num_tr_full = build_feature_set(train_df["text"].values, fs)
        num_test = build_feature_set(raw_test, fs)

        if num_tr_full is not None:
            X_train_full, scaler_final = combine_tfidf_features(X_train_full, num_tr_full, fit_scaler=True)
            X_test, _ = combine_tfidf_features(X_test, num_test, fit_scaler=False, scaler=scaler_final)
        else:
            scaler_final = None

        # Train
        final_model, train_time = _train_model(
            cand["model_name"], X_train_full, y_train_full,
            cand["hp_key"], cand["hp_val"],
        )

        # Evaluate
        test_metrics = evaluate_model(final_model, X_test, y_test)
        vocab_final = len(vec_final.vocabulary_)

        logger.info(f"  TEST acc={test_metrics['accuracy']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}")
        for cls, m in test_metrics["per_class"].items():
            logger.info(f"    {cls}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")

        test_row = {
            "experiment_id": cand["exp_id"],
            "split": "test",
            "preprocessing_variant": preproc,
            "feature_set": fs,
            "ngram_range": str(cand["ngram_range"]),
            "min_df": 2, "max_df": 0.95, "sublinear_tf": True,
            "model": cand["model_name"],
            "hyperparameters": str({cand["hp_key"]: cand["hp_val"]}),
            "vocabulary_size": vocab_final,
            "total_features": X_train_full.shape[1],
            "accuracy": round(test_metrics["accuracy"], 4),
            "macro_precision": round(test_metrics["macro_precision"], 4),
            "macro_recall": round(test_metrics["macro_recall"], 4),
            "macro_f1": round(test_metrics["macro_f1"], 4),
            "regular_f1": round(test_metrics["per_class"]["Regular"]["f1"], 4),
            "sarcasm_f1": round(test_metrics["per_class"]["Sarcasm"]["f1"], 4),
            "irony_f1": round(test_metrics["per_class"]["Irony"]["f1"], 4),
            "figurative_f1": round(test_metrics["per_class"]["Figurative"]["f1"], 4),
            "training_time": round(train_time, 2),
            "notes": cand["label"],
        }
        test_results.append(test_row)

        # Confusion matrix
        test_cm = np.array(test_metrics["confusion_matrix"])
        safe_label = cand["label"].replace(" ", "_")
        plot_confusion_matrix(test_cm, CLASS_NAMES,
                              f"Test: {cand['label']}", str(CM_DIR / f"{safe_label}_test_cm.png"))
        save_confusion_matrix(test_cm, CLASS_NAMES, str(CM_DIR / f"{safe_label}_test_cm.csv"))

        # Classification report
        test_report = generate_classification_report(final_model, X_test, y_test)
        save_classification_report(test_report, str(REPORTS_DIR / f"{safe_label}_test_report.json"))

        # Save model artifacts
        model_dir = MODELS_DIR / safe_label
        save_model_artifacts(
            final_model, vec_final,
            config={
                "model": cand["model_name"],
                "preprocessing": preproc,
                "ngram_range": str(cand["ngram_range"]),
                "feature_set": fs,
                "hyperparameters": {cand["hp_key"]: cand["hp_val"]},
                "vocab_size": vocab_final,
                "test_macro_f1": test_metrics["macro_f1"],
                "label_mapping": {"Regular": 0, "Sarcasm": 1, "Irony": 2, "Figurative": 3},
            },
            output_dir=str(model_dir),
            experiment_id=cand["exp_id"],
        )

        # Feature importance (LR/SVM only)
        if hasattr(final_model, "coef_"):
            feat_names = list(vec_final.get_feature_names_out())
            if fs != "A":
                feat_names += get_feature_names(fs)
            importance = extract_feature_importance(
                final_model, feat_names, CLASS_NAMES, top_n=30,
            )
            save_feature_importance(importance, str(FEATURES_DIR / safe_label))

    # Append test results
    test_df_results = pd.DataFrame(test_results)
    combined = pd.concat([results_df, test_df_results], ignore_index=True)
    combined.to_csv(RESULTS_DIR / "model_results.csv", index=False)

    # ============================================================
    # STEP 8: VISUALIZATIONS
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: VISUALIZATIONS")
    logger.info("=" * 70)

    # 1. Macro F1 comparison (all validation)
    plot_comparison(results_df, "macro_f1", "Macro F1 — All Experiments (Validation)",
                    str(VIZ_DIR / "macro_f1_comparison.png"))

    # 2. Accuracy comparison
    plot_comparison(results_df, "accuracy", "Accuracy — All Experiments (Validation)",
                    str(VIZ_DIR / "accuracy_comparison.png"))

    # 3. BoW vs TF-IDF comparison
    p4_results = pd.read_csv(BASE_DIR / "results" / "classical" / "model_results.csv")
    p4_test = p4_results[p4_results["split"] == "test"]
    p5_test = pd.DataFrame(test_results)

    comparison_rows = []
    for _, r in p4_test.iterrows():
        comparison_rows.append({
            "Phase": "Phase 4 (BoW)",
            "Model": r["model"],
            "Accuracy": r["accuracy"],
            "Macro F1": r["macro_f1"],
        })
    for _, r in p5_test.iterrows():
        comparison_rows.append({
            "Phase": "Phase 5 (TF-IDF)",
            "Model": f"{r['model']} ({r.get('notes','')})",
            "Accuracy": r["accuracy"],
            "Macro F1": r["macro_f1"],
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(RESULTS_DIR / "phase4_vs_phase5.csv", index=False)

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comp_df))
    colors = ["#2196F3" if "Phase 4" in p else "#4CAF50" for p in comp_df["Phase"]]
    bars = ax.bar(x, comp_df["Macro F1"], color=colors)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Macro F1")
    ax.set_title("Phase 4 (BoW) vs Phase 5 (TF-IDF) — Test Macro F1")
    ax.set_xticks(x)
    labels_short = [f"{r['Phase'].split('(')[1].rstrip(')')}\n{r['Model'][:15]}" for _, r in comp_df.iterrows()]
    ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=7)
    for bar, val in zip(bars, comp_df["Macro F1"]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(VIZ_DIR / "bow_vs_tfidf_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. N-gram comparison (Stage 1 LR only)
    stage1_lr = results_df[
        (results_df["experiment_id"].str.startswith("TFIDF")) &
        (results_df["model"] == "LogisticRegression")
    ]
    if len(stage1_lr) > 0:
        plot_comparison(stage1_lr.reset_index(drop=True), "macro_f1",
                        "N-gram Comparison (LR, classical, validation)",
                        str(VIZ_DIR / "ngram_comparison.png"))

    # 5. Feature set comparison (Stage 2)
    stage2 = results_df[results_df["experiment_id"].str.startswith("FEAT")]
    if len(stage2) > 0:
        plot_comparison(stage2.reset_index(drop=True), "macro_f1",
                        "Feature Set Comparison (Validation)",
                        str(VIZ_DIR / "feature_set_comparison.png"))

    # 6. Vocab size comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(fa_df))
    ax.bar(x, fa_df["vocabulary_size"], color=plt.cm.Set3(np.linspace(0, 1, len(fa_df))))
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Vocabulary Size")
    ax.set_title("Vocabulary Size by Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(fa_df["experiment_id"], rotation=45, ha="right")
    for i, v in enumerate(fa_df["vocabulary_size"]):
        ax.text(i, v+200, f"{v:,}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(str(VIZ_DIR / "vocab_size_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # STEP 9: SUMMARY
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5 COMPLETE")
    logger.info("=" * 70)

    best_val = results_df.loc[results_df["macro_f1"].idxmax()]
    logger.info(f"\nBest validation result:")
    logger.info(f"  Experiment: {best_val['experiment_id']}")
    logger.info(f"  Model: {best_val['model']}")
    logger.info(f"  Feature set: {best_val['feature_set']}")
    logger.info(f"  Preprocessing: {best_val['preprocessing_variant']}")
    logger.info(f"  Macro F1: {best_val['macro_f1']:.4f}")

    logger.info(f"\nTest results:")
    for r in test_results:
        logger.info(f"  {r['notes']}: acc={r['accuracy']:.4f}, macro_f1={r['macro_f1']:.4f}")

    # Save summary
    summary = {
        "phase": "Phase 5 — TF-IDF + Feature Engineering",
        "total_experiments": len(all_results),
        "best_ngram": str(best_ngram),
        "best_feature_set": best_feature_set_name,
        "best_preprocessing": best_preproc_name,
        "best_validation": {
            "experiment_id": best_val["experiment_id"],
            "model": best_val["model"],
            "macro_f1": float(best_val["macro_f1"]),
        },
        "test_results": test_results,
        "phase4_comparison": comparison_rows,
    }

    with open(RESULTS_DIR / "phase5_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved: {RESULTS_DIR / 'phase5_summary.json'}")


if __name__ == "__main__":
    main()
