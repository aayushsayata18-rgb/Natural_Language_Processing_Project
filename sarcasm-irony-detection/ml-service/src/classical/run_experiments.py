"""
Phase 4 - Run All Experiments
============================================
Sarcasm & Irony Detection System using NLP

Main entry point for Phase 4 experiments.
Orchestrates the full pipeline:
    1. Load data
    2. Apply Phase 3 preprocessing
    3. Create train/validation split
    4. Run core BoW experiments (BOW-01 to BOW-07)
    5. Run preprocessing comparison experiments (BOW-08 to BOW-16)
    6. Evaluate on test set
    7. Generate confusion matrices and classification reports
    8. Save model artifacts
    9. Generate visualizations
    10. Save all results

Usage:
    python src/classical/run_experiments.py
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
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split

# --- Path Setup ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ml-service/
sys.path.insert(0, str(BASE_DIR))

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.config import get_variant

from src.classical.bow import create_bow_vectorizer, fit_transform_bow, transform_bow, get_feature_info
from src.classical.train_lr import train_logistic_regression
from src.classical.train_nb import train_naive_bayes
from src.classical.train_svm import train_linear_svm
from src.classical.evaluate import (
    evaluate_model,
    generate_classification_report,
    collect_error_examples,
    format_results_row,
    CLASS_NAMES,
)
from src.classical.experiments import (
    CORE_EXPERIMENTS,
    run_single_experiment,
    determine_best_ngram,
    generate_preprocessing_experiments,
)
from src.classical.save_artifacts import (
    save_model_artifacts,
    save_confusion_matrix,
    save_classification_report,
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
MODELS_DIR = BASE_DIR / "models" / "classical"
RESULTS_DIR = BASE_DIR / "results" / "classical"
REPORTS_DIR = RESULTS_DIR / "classification_reports"
CM_DIR = RESULTS_DIR / "confusion_matrices"
VIZ_DIR = RESULTS_DIR / "visualizations"

# Class names and constants
RANDOM_STATE = 42
VAL_SIZE = 0.20


def load_data():
    """Load Phase 1 clean datasets."""
    train_df = pd.read_csv(PROCESSED_DIR / "train_clean.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_clean.csv")
    
    logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    logger.info(f"Columns: {train_df.columns.tolist()}")
    logger.info(f"Label distribution (train):\n{train_df['label'].value_counts().to_string()}")
    
    return train_df, test_df


def apply_preprocessing(df, variant_name):
    """Apply Phase 3 preprocessing to a dataframe."""
    # Check if preprocessed variant already exists
    preproc_path = PREPROCESSED_DIR / variant_name / "train.csv"
    if preproc_path.exists():
        preproc_df = pd.read_csv(preproc_path)
        logger.info(f"Loaded preprocessed '{variant_name}' from cache: {preproc_df.shape}")
        return preproc_df["processed_text"]
    
    # Otherwise, run preprocessing
    config = get_variant(variant_name)
    pipeline = PreprocessingPipeline(config)
    processed = pipeline.process_dataframe(df)
    return processed["processed_text"]


def load_preprocessed_texts(variant_name, split="train"):
    """Load pre-computed preprocessed texts."""
    preproc_path = PREPROCESSED_DIR / variant_name / f"{split}.csv"
    if preproc_path.exists():
        df = pd.read_csv(preproc_path)
        logger.info(f"Loaded preprocessed '{variant_name}/{split}': {df.shape}")
        return df["processed_text"]
    else:
        raise FileNotFoundError(f"Preprocessed data not found: {preproc_path}")


def create_split(texts, labels, val_size=VAL_SIZE, random_state=RANDOM_STATE):
    """Create stratified train/validation split."""
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=val_size,
        stratify=labels,
        random_state=random_state,
    )
    
    logger.info(f"Split: train={len(X_train)}, val={len(X_val)}")
    logger.info(f"Train label distribution: {pd.Series(y_train).value_counts().to_dict()}")
    logger.info(f"Val label distribution: {pd.Series(y_val).value_counts().to_dict()}")
    
    return X_train, X_val, y_train, y_val


def plot_confusion_matrix(cm, class_names, title, filepath):
    """Generate and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Actual",
        xlabel="Predicted",
        title=title,
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix plot saved: {filepath}")


def plot_metric_comparison(results_df, metric, title, filepath):
    """Generate a bar chart comparing a metric across experiments."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(results_df))
    bars = ax.bar(x, results_df[metric], color=plt.cm.Set2(np.linspace(0, 1, len(results_df))))
    
    ax.set_xlabel("Experiment")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["experiment_id"], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    
    # Add value labels on bars
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    
    ax.set_ylim(0, min(1.0, results_df[metric].max() + 0.05))
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Metric comparison plot saved: {filepath}")


def main():
    """Run all Phase 4 experiments."""
    logger.info("=" * 70)
    logger.info("PHASE 4 — BAG OF WORDS + CLASSICAL MACHINE LEARNING")
    logger.info("=" * 70)
    
    # Suppress convergence warnings during hyperparameter search
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Create output directories
    for d in [MODELS_DIR, RESULTS_DIR, REPORTS_DIR, CM_DIR, VIZ_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # STEP 1: Load Data
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: LOAD DATA")
    logger.info("=" * 70)
    
    train_df, test_df = load_data()
    labels = train_df["label_id"].values
    
    # ============================================================
    # STEP 2: Run Core Experiments (BOW-01 to BOW-07)
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: CORE EXPERIMENTS (BOW-01 to BOW-07)")
    logger.info("=" * 70)
    
    all_results = []
    best_models = {}  # model_name -> (model, vectorizer, result, exp_id)
    
    # Load minimal preprocessing
    texts_minimal = load_preprocessed_texts("minimal", "train")
    
    # Create train/val split
    X_train_text, X_val_text, y_train, y_val = create_split(
        texts_minimal.values, labels
    )
    
    # Cache vectorizers to avoid re-fitting
    vectorizer_cache = {}
    X_train_cache = {}
    X_val_cache = {}
    
    for exp in CORE_EXPERIMENTS:
        ngram_key = str(exp["ngram_range"])
        
        # Create and fit vectorizer if not cached
        if ngram_key not in vectorizer_cache:
            vectorizer = create_bow_vectorizer(ngram_range=exp["ngram_range"])
            X_train = fit_transform_bow(vectorizer, X_train_text)
            X_val = transform_bow(vectorizer, X_val_text)
            
            vectorizer_cache[ngram_key] = vectorizer
            X_train_cache[ngram_key] = X_train
            X_val_cache[ngram_key] = X_val
        else:
            vectorizer = vectorizer_cache[ngram_key]
            X_train = X_train_cache[ngram_key]
            X_val = X_val_cache[ngram_key]
        
        vocab_size = len(vectorizer.vocabulary_)
        
        result_row, best_model, _ = run_single_experiment(
            exp, X_train, y_train, X_val, y_val,
            vocab_size=vocab_size,
            preprocessing_name="minimal",
        )
        
        all_results.append(result_row)
        
        # Track best model per model type
        model_name = exp["model"]
        if model_name not in best_models or result_row["macro_f1"] > best_models[model_name][2]["macro_f1"]:
            best_models[model_name] = (best_model, vectorizer, result_row, exp["id"])
    
    # ============================================================
    # STEP 3: Determine Best N-gram
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: DETERMINE BEST N-GRAM RANGE")
    logger.info("=" * 70)
    
    best_ngram = determine_best_ngram(all_results)
    logger.info(f"Best n-gram range: {best_ngram}")
    
    # ============================================================
    # STEP 4: Preprocessing Comparison Experiments (BOW-08 to BOW-16)
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: PREPROCESSING COMPARISON EXPERIMENTS")
    logger.info("=" * 70)
    
    preproc_experiments = generate_preprocessing_experiments(best_ngram)
    
    for exp in preproc_experiments:
        preproc_name = exp["preprocessing"]
        
        # Load preprocessed texts for this variant
        texts_variant = load_preprocessed_texts(preproc_name, "train")
        
        # Create split with same indices
        X_train_text_p, X_val_text_p, y_train_p, y_val_p = create_split(
            texts_variant.values, labels
        )
        
        # Create and fit vectorizer
        vectorizer = create_bow_vectorizer(ngram_range=exp["ngram_range"])
        X_train_p = fit_transform_bow(vectorizer, X_train_text_p)
        X_val_p = transform_bow(vectorizer, X_val_text_p)
        
        vocab_size = len(vectorizer.vocabulary_)
        
        result_row, best_model, _ = run_single_experiment(
            exp, X_train_p, y_train_p, X_val_p, y_val_p,
            vocab_size=vocab_size,
            preprocessing_name=preproc_name,
        )
        
        all_results.append(result_row)
        
        # Update best models if this preproc variant is better
        model_name = exp["model"]
        if result_row["macro_f1"] > best_models[model_name][2]["macro_f1"]:
            best_models[model_name] = (best_model, vectorizer, result_row, exp["id"])
    
    # ============================================================
    # STEP 5: Save Validation Results
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: SAVE VALIDATION RESULTS")
    logger.info("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / "model_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"All validation results saved: {results_path}")
    
    # Print results table
    logger.info("\nValidation Results Summary:")
    display_cols = ["experiment_id", "preprocessing", "ngram_range", "model", "accuracy", "macro_f1"]
    logger.info("\n" + results_df[display_cols].to_string(index=False))
    
    # ============================================================
    # STEP 6: Generate Confusion Matrices & Reports (Validation)
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: CONFUSION MATRICES & CLASSIFICATION REPORTS")
    logger.info("=" * 70)
    
    for model_name, (model, vectorizer, result, exp_id) in best_models.items():
        preproc_name = result["preprocessing"]
        
        # Re-load the correct preprocessing variant
        texts_variant = load_preprocessed_texts(preproc_name, "train")
        _, X_val_text_eval, _, y_val_eval = create_split(texts_variant.values, labels)
        X_val_eval = transform_bow(vectorizer, X_val_text_eval)
        
        # Evaluate
        metrics = evaluate_model(model, X_val_eval, y_val_eval)
        
        # Confusion matrix
        cm = np.array(metrics["confusion_matrix"])
        cm_csv_path = CM_DIR / f"{exp_id}_confusion_matrix.csv"
        save_confusion_matrix(cm, CLASS_NAMES, str(cm_csv_path))
        
        cm_png_path = CM_DIR / f"{exp_id}_confusion_matrix.png"
        plot_confusion_matrix(cm, CLASS_NAMES, f"Confusion Matrix: {exp_id} ({model_name})", str(cm_png_path))
        
        # Classification report
        report = generate_classification_report(model, X_val_eval, y_val_eval)
        report_path = REPORTS_DIR / f"{exp_id}_classification_report.json"
        save_classification_report(report, str(report_path))
        
        # Error examples
        texts_for_errors = load_preprocessed_texts(preproc_name, "train")
        _, val_texts, _, _ = create_split(texts_for_errors.values, labels)
        error_df = collect_error_examples(model, X_val_eval, y_val_eval, val_texts)
        error_path = RESULTS_DIR / f"{exp_id}_error_examples.csv"
        error_df.to_csv(error_path, index=False)
        logger.info(f"Error examples saved: {error_path}")
    
    # ============================================================
    # STEP 7: Test Set Evaluation
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: TEST SET EVALUATION")
    logger.info("=" * 70)
    
    test_results = []
    
    for model_name, (_, _, best_result, exp_id) in best_models.items():
        preproc_name = best_result["preprocessing"]
        ngram_range = eval(best_result["ngram_range"])  # Parse string back
        hp = eval(best_result["hyperparameters"])  # Parse dict string
        
        logger.info(f"\nFinal test evaluation: {exp_id} ({model_name})")
        logger.info(f"  Preprocessing: {preproc_name}")
        logger.info(f"  N-gram range: {ngram_range}")
        logger.info(f"  Hyperparameters: {hp}")
        
        # Load full training preprocessed data
        train_texts_full = load_preprocessed_texts(preproc_name, "train")
        test_texts = load_preprocessed_texts(preproc_name, "test")
        
        # Fit vectorizer on FULL training data
        vectorizer_final = create_bow_vectorizer(ngram_range=ngram_range)
        X_train_full = fit_transform_bow(vectorizer_final, train_texts_full)
        X_test = transform_bow(vectorizer_final, test_texts)
        
        y_train_full = train_df["label_id"].values
        y_test = test_df["label_id"].values
        
        vocab_size_final = len(vectorizer_final.vocabulary_)
        
        # Train final model on full training data
        hp_key = list(hp.keys())[0]
        hp_val = hp[hp_key]
        
        if model_name == "LogisticRegression":
            final_model, train_time = train_logistic_regression(X_train_full, y_train_full, C=hp_val)
        elif model_name == "MultinomialNB":
            final_model, train_time = train_naive_bayes(X_train_full, y_train_full, alpha=hp_val)
        elif model_name == "LinearSVM":
            final_model, train_time = train_linear_svm(X_train_full, y_train_full, C=hp_val)
        
        # Evaluate on test set
        test_metrics = evaluate_model(final_model, X_test, y_test)
        
        logger.info(f"  TEST accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  TEST macro_f1: {test_metrics['macro_f1']:.4f}")
        for cls, cls_metrics in test_metrics["per_class"].items():
            logger.info(f"    {cls}: P={cls_metrics['precision']:.4f}, R={cls_metrics['recall']:.4f}, F1={cls_metrics['f1']:.4f}")
        
        # Save test result row
        test_row = format_results_row(
            experiment_id=exp_id,
            preprocessing=preproc_name,
            vectorizer_type="CountVectorizer",
            ngram_range=ngram_range,
            model_name=model_name,
            hyperparams=hp,
            metrics=test_metrics,
            training_time=train_time,
            vocab_size=vocab_size_final,
            split="test",
        )
        test_results.append(test_row)
        
        # Test confusion matrix
        test_cm = np.array(test_metrics["confusion_matrix"])
        test_cm_path = CM_DIR / f"{exp_id}_test_confusion_matrix.png"
        plot_confusion_matrix(test_cm, CLASS_NAMES, f"Test Confusion: {exp_id} ({model_name})", str(test_cm_path))
        save_confusion_matrix(test_cm, CLASS_NAMES, str(CM_DIR / f"{exp_id}_test_confusion_matrix.csv"))
        
        # Test classification report
        test_report = generate_classification_report(final_model, X_test, y_test)
        save_classification_report(test_report, str(REPORTS_DIR / f"{exp_id}_test_classification_report.json"))
        
        # Save final model artifacts
        model_dir = MODELS_DIR / f"bow_{model_name.lower()}"
        save_model_artifacts(
            final_model, vectorizer_final,
            config={
                "model": model_name,
                "preprocessing": preproc_name,
                "ngram_range": str(ngram_range),
                "hyperparameters": hp,
                "vocab_size": vocab_size_final,
                "validation_macro_f1": best_result["macro_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
                "random_state": RANDOM_STATE,
                "label_mapping": {"Regular": 0, "Sarcasm": 1, "Irony": 2, "Figurative": 3},
            },
            output_dir=str(model_dir),
            experiment_id=exp_id,
        )
    
    # Append test results to CSV
    test_results_df = pd.DataFrame(test_results)
    all_combined = pd.concat([results_df, test_results_df], ignore_index=True)
    all_combined.to_csv(results_path, index=False)
    logger.info(f"Combined results (val + test) saved: {results_path}")
    
    # ============================================================
    # STEP 8: Visualizations
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: VISUALIZATIONS")
    logger.info("=" * 70)
    
    val_results = results_df.copy()
    
    # Macro F1 comparison
    plot_metric_comparison(
        val_results, "macro_f1",
        "Macro F1 Comparison (Validation)",
        str(VIZ_DIR / "macro_f1_comparison.png"),
    )
    
    # Accuracy comparison
    plot_metric_comparison(
        val_results, "accuracy",
        "Accuracy Comparison (Validation)",
        str(VIZ_DIR / "accuracy_comparison.png"),
    )
    
    # Precision comparison
    plot_metric_comparison(
        val_results, "macro_precision",
        "Macro Precision Comparison (Validation)",
        str(VIZ_DIR / "precision_comparison.png"),
    )
    
    # Recall comparison
    plot_metric_comparison(
        val_results, "macro_recall",
        "Macro Recall Comparison (Validation)",
        str(VIZ_DIR / "recall_comparison.png"),
    )
    
    # Vocab size comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(val_results))
    ax.bar(x, val_results["vocab_size"], color=plt.cm.Set3(np.linspace(0, 1, len(val_results))))
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Vocabulary Size")
    ax.set_title("Vocabulary Size by Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(val_results["experiment_id"], rotation=45, ha="right")
    for i, v in enumerate(val_results["vocab_size"]):
        ax.text(i, v + 200, f"{v:,}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(str(VIZ_DIR / "vocab_size_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # ============================================================
    # STEP 9: Summary
    # ============================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4 COMPLETE")
    logger.info("=" * 70)
    
    # Find overall best on validation
    best_val = results_df.loc[results_df["macro_f1"].idxmax()]
    logger.info(f"\nBest classical baseline (validation):")
    logger.info(f"  Experiment: {best_val['experiment_id']}")
    logger.info(f"  Model: {best_val['model']}")
    logger.info(f"  Preprocessing: {best_val['preprocessing']}")
    logger.info(f"  N-gram: {best_val['ngram_range']}")
    logger.info(f"  Macro F1: {best_val['macro_f1']:.4f}")
    logger.info(f"  Accuracy: {best_val['accuracy']:.4f}")
    
    logger.info(f"\nTest results:")
    for _, row in test_results_df.iterrows():
        logger.info(f"  {row['experiment_id']} ({row['model']}): acc={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}")
    
    logger.info(f"\nOutput directories:")
    logger.info(f"  Models:  {MODELS_DIR}")
    logger.info(f"  Results: {RESULTS_DIR}")
    
    # Save summary JSON
    summary = {
        "phase": "Phase 4 — BoW + Classical ML",
        "total_experiments": len(all_results),
        "best_validation": {
            "experiment_id": best_val["experiment_id"],
            "model": best_val["model"],
            "preprocessing": best_val["preprocessing"],
            "ngram_range": best_val["ngram_range"],
            "macro_f1": float(best_val["macro_f1"]),
            "accuracy": float(best_val["accuracy"]),
        },
        "test_results": test_results,
        "random_state": RANDOM_STATE,
    }
    
    summary_path = RESULTS_DIR / "phase4_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
