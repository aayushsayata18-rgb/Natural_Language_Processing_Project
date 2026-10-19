"""
Phase 7 — FastText + Bi-LSTM + Attention Experiment Orchestrator
================================================================
Sarcasm & Irony Detection System using NLP

Full experimental pipeline:
1. Loads leakage-controlled datasets (train_clean.csv, test_clean.csv)
2. Fits custom word tokenizer strictly on training split
3. Analyzes sequence length distribution and calculates FastText vocabulary coverage
4. Performs subword analysis on representative social media tokens (slang, hashtags, etc.)
5. Builds FastText embedding matrix with subword synthesis & documented fallback
6. Executes architecture experiment matrix:
   - FT-01: FastText + Bi-LSTM (No Attention, frozen embeddings)
   - FT-02: FastText + Bi-LSTM + Attention (With Attention, frozen embeddings)
   - FT-03: FastText + Bi-LSTM + Attention (With Attention, fine-tuned embeddings)
7. Compares validation metrics, model parameter counts, and training durations
8. Selects best model configuration based on validation Macro F1
9. Performs final evaluation on official untouched test set
10. Generates prediction files, confusion matrices, classification reports, and attention heatmaps
11. Saves serialized model checkpoints and progression tracking tables
"""

import copy
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # ml-service/
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


def run_phase7_experiments(
    max_epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 4,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Runs full Phase 7 experimental evaluation.
    """
    set_seed(42)
    device_obj = torch.device(device)
    logger.info("Running on device: %s", device)

    # Directory layout
    data_dir = BASE_DIR / "data" / "processed"
    embeddings_dir = BASE_DIR / "data" / "embeddings"
    results_dir = BASE_DIR / "results" / "fasttext_attention"
    models_dir = BASE_DIR / "models" / "deep_learning" / "fasttext"

    hist_dir = results_dir / "training_history"
    cm_dir = results_dir / "confusion_matrices"
    cr_dir = results_dir / "classification_reports"
    pred_dir = results_dir / "predictions"
    attn_dir = results_dir / "attention"

    for d in [results_dir, models_dir, hist_dir, cm_dir, cr_dir, pred_dir, attn_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logger.info("Loading Phase 1 cleaned datasets...")
    train_df = pd.read_csv(data_dir / "train_clean.csv")
    test_df = pd.read_csv(data_dir / "test_clean.csv")

    texts_all = train_df["text"].values
    labels_all = train_df["label_id"].values
    test_texts = test_df["text"].values
    test_labels = test_df["label_id"].values

    # Train / Validation Split (80 / 20 stratified, seed 42)
    X_tr_text, X_val_text, y_train, y_val = train_test_split(
        texts_all, labels_all, test_size=0.20, stratify=labels_all, random_state=42
    )
    logger.info("Split sizes — Train: %d, Val: %d, Test: %d", len(X_tr_text), len(X_val_text), len(test_texts))

    # 2. Tokenizer (Fitted ONLY on training data to prevent data leakage)
    max_words = 30000
    max_len = 30
    embed_dim = 300

    logger.info("Fitting Tokenizer on training text...")
    tokenizer = Tokenizer(max_num_words=max_words)
    tokenizer.fit(X_tr_text)
    logger.info("Vocabulary size: %d", tokenizer.vocab_size)

    # Sequence length distribution check
    seq_lens = tokenizer.get_sequence_lengths(X_tr_text)
    mean_len = float(np.mean(seq_lens))
    median_len = float(np.median(seq_lens))
    pct_truncated = float(np.mean(seq_lens > max_len) * 100.0)
    pct_padded = float(np.mean(seq_lens < max_len) * 100.0)
    logger.info(
        "Seq length stats — Mean: %.1f, Median: %.1f, Truncated: %.2f%%, Padded: %.2f%%",
        mean_len, median_len, pct_truncated, pct_padded
    )

    # 3. Load FastText Vectors
    fasttext_path = embeddings_dir / "wiki.simple.vec"
    if not fasttext_path.exists():
        raise FileNotFoundError(f"FastText vector file not found at {fasttext_path}")

    logger.info("Loading FastText vectors from %s...", fasttext_path)
    t_ft_0 = time.time()
    fasttext_vectors = load_fasttext_vectors(str(fasttext_path), dim=embed_dim)
    t_ft_1 = time.time()
    logger.info("Loaded %d FastText vectors in %.2fs", len(fasttext_vectors), t_ft_1 - t_ft_0)

    # 4. Coverage and Subword Analysis
    logger.info("Calculating FastText vocabulary coverage with subword synthesis...")
    embedding_matrix, coverage_stats, oov_records = build_embedding_matrix(
        tokenizer.word2idx,
        fasttext_vectors,
        dim=embed_dim,
        use_subwords=True,
        oov_strategy="subword_random",
        random_seed=42,
    )
    coverage_df = pd.DataFrame([coverage_stats])
    coverage_df.to_csv(results_dir / "fasttext_coverage.csv", index=False)
    logger.info(
        "Coverage: Exact = %.2f%%, With Subwords = %.2f%% (vs Phase 6 GloVe = 47.37%%)",
        coverage_stats["exact_coverage_pct"], coverage_stats["effective_coverage_pct"]
    )

    # Save detailed subword analysis
    subword_df = pd.DataFrame(oov_records)
    subword_df.to_csv(results_dir / "subword_analysis.csv", index=False)

    # 5. Prepare Padded Integer Sequences
    X_train_seq = tokenizer.texts_to_sequences(X_tr_text)
    X_val_seq = tokenizer.texts_to_sequences(X_val_text)
    X_test_seq = tokenizer.texts_to_sequences(test_texts)

    X_train_pad = tokenizer.pad_sequences(X_train_seq, max_len=max_len, padding="pre", truncating="post")
    X_val_pad = tokenizer.pad_sequences(X_val_seq, max_len=max_len, padding="pre", truncating="post")
    X_test_pad = tokenizer.pad_sequences(X_test_seq, max_len=max_len, padding="pre", truncating="post")

    train_loader = create_dataloader(X_train_pad, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val_pad, y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(X_test_pad, test_labels, batch_size=batch_size, shuffle=False)

    # 6. Experiment Configurations Matrix
    experiments = [
        {
            "id": "FT-01",
            "desc": "FastText + Bi-LSTM (Frozen)",
            "model_type": "bilstm",
            "lstm_units": 64,
            "attention_dim": None,
            "dense_units": 64,
            "dropout": 0.3,
            "trainable": False,
        },
        {
            "id": "FT-02",
            "desc": "FastText + Bi-LSTM + Attention (Frozen)",
            "model_type": "attention",
            "lstm_units": 64,
            "attention_dim": 64,
            "dense_units": 64,
            "dropout": 0.3,
            "trainable": False,
        },
        {
            "id": "FT-03",
            "desc": "FastText + Bi-LSTM + Attention (Fine-Tuned)",
            "model_type": "attention",
            "lstm_units": 64,
            "attention_dim": 64,
            "dense_units": 64,
            "dropout": 0.3,
            "trainable": True,
        },
    ]

    results_table = []
    models_dict = {}
    histories_dict = {}

    for exp in experiments:
        logger.info("\n" + "=" * 60)
        logger.info("Executing Experiment: %s (%s)", exp["id"], exp["desc"])
        logger.info("=" * 60)
        set_seed(42)

        if exp["model_type"] == "bilstm":
            model = FastTextBiLSTMClassifier(
                vocab_size=tokenizer.vocab_size,
                embedding_dim=embed_dim,
                lstm_units=exp["lstm_units"],
                dense_units=exp["dense_units"],
                num_classes=4,
                dropout=exp["dropout"],
                embedding_matrix=embedding_matrix,
                embedding_trainable=exp["trainable"],
            )
        else:
            model = FastTextAttentionClassifier(
                vocab_size=tokenizer.vocab_size,
                embedding_dim=embed_dim,
                lstm_units=exp["lstm_units"],
                attention_dim=exp["attention_dim"],
                dense_units=exp["dense_units"],
                num_classes=4,
                dropout=exp["dropout"],
                embedding_matrix=embedding_matrix,
                embedding_trainable=exp["trainable"],
            )

        total_p, train_p = model.count_params()
        logger.info("Parameters: Total = %s, Trainable = %s", f"{total_p:,}", f"{train_p:,}")

        ckpt_dir = models_dir / exp["id"].lower()
        trained_model, history, train_time = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            patience=patience,
            lr_patience=2,
            lr_factor=0.5,
            checkpoint_dir=str(ckpt_dir),
            device=device,
            model_name=f"best_{exp['id'].lower()}.pt",
        )

        # Save training history CSV and plot
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(hist_dir / f"{exp['id'].lower()}_history.csv", index=False)
        plot_training_history(
            history,
            title=f"{exp['id']}: {exp['desc']}",
            output_path=str(hist_dir / f"{exp['id'].lower()}_curves.png"),
        )

        # Validation evaluation
        val_metrics = evaluate_model(trained_model, val_loader, device=device)
        best_epoch = int(np.argmin(history["val_loss"])) + 1

        record = {
            "experiment_id": exp["id"],
            "split": "validation",
            "description": exp["desc"],
            "embedding": "FastText-wiki-300d",
            "embedding_trainable": exp["trainable"],
            "has_attention": (exp["model_type"] == "attention"),
            "lstm_units": exp["lstm_units"],
            "dense_units": exp["dense_units"],
            "dropout": exp["dropout"],
            "batch_size": batch_size,
            "total_params": total_p,
            "trainable_params": train_p,
            "best_epoch": best_epoch,
            "total_epochs": len(history["epoch"]),
            "accuracy": val_metrics["accuracy"],
            "macro_precision": val_metrics["macro_precision"],
            "macro_recall": val_metrics["macro_recall"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
            "regular_f1": val_metrics["per_class"]["Regular"]["f1"],
            "sarcasm_f1": val_metrics["per_class"]["Sarcasm"]["f1"],
            "irony_f1": val_metrics["per_class"]["Irony"]["f1"],
            "figurative_f1": val_metrics["per_class"]["Figurative"]["f1"],
            "training_time": round(train_time, 2),
        }
        results_table.append(record)
        models_dict[exp["id"]] = trained_model
        histories_dict[exp["id"]] = history

        # Save validation confusion matrix plot
        plot_confusion_matrix(
            val_metrics["confusion_matrix"],
            title=f"{exp['id']} Validation Confusion Matrix",
            output_path=str(cm_dir / f"{exp['id'].lower()}_val_cm.png"),
            normalize=False,
        )

        logger.info(
            "%s Validation Results — Acc: %.4f | Macro F1: %.4f | Time: %.1fs",
            exp["id"], val_metrics["accuracy"], val_metrics["macro_f1"], train_time
        )

    # 7. Model Selection
    val_df = pd.DataFrame(results_table)
    val_df.to_csv(results_dir / "architecture_comparison.csv", index=False)

    best_idx = val_df["macro_f1"].idxmax()
    best_exp_id = val_df.loc[best_idx, "experiment_id"]
    best_val_f1 = val_df.loc[best_idx, "macro_f1"]
    logger.info("\n" + "=" * 60)
    logger.info("BEST EXPERIMENT: %s with Val Macro F1 = %.4f", best_exp_id, best_val_f1)
    logger.info("=" * 60)

    best_model = models_dict[best_exp_id]
    best_config = [e for e in experiments if e["id"] == best_exp_id][0]

    # Save best models into models/deep_learning/fasttext/
    torch.save(models_dict["FT-01"].state_dict(), models_dir / "best_fasttext_bilstm.pt")
    if "FT-02" in models_dict:
        torch.save(models_dict["FT-02"].state_dict(), models_dir / "best_fasttext_attention.pt")
    torch.save(best_model.state_dict(), models_dir / "best_model.pt")

    # Save tokenizer and label mappings
    tokenizer.save(str(models_dir / "tokenizer.json"))
    label_mapping = {"Regular": 0, "Sarcasm": 1, "Irony": 2, "Figurative": 3}
    with open(models_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    fasttext_config = {
        "best_experiment": best_exp_id,
        "embedding_source": FASTTEXT_METADATA["source"],
        "embedding_dimension": embed_dim,
        "vocabulary_size": tokenizer.vocab_size,
        "max_sequence_length": max_len,
        "oov_strategy": "subword_synthesis_with_random_fallback",
        "embedding_trainable": best_config["trainable"],
        "lstm_units": best_config["lstm_units"],
        "attention_dim": best_config["attention_dim"],
        "dense_units": best_config["dense_units"],
        "dropout": best_config["dropout"],
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "random_seed": 42,
        "label_mapping": label_mapping,
    }
    with open(models_dir / "fasttext_config.json", "w", encoding="utf-8") as f:
        json.dump(fasttext_config, f, indent=2)

    # 8. FINAL EVALUATION ON OFFICIAL TEST SET (Strictly after model selection)
    logger.info("\nEvaluating Best Model (%s) on Untouched Official Test Set...", best_exp_id)
    t_test_0 = time.time()
    test_metrics = evaluate_model(best_model, test_loader, device=device)
    t_test_eval = time.time() - t_test_0

    test_record = {
        "experiment_id": best_exp_id,
        "split": "test",
        "description": best_config["desc"],
        "embedding": "FastText-wiki-300d",
        "embedding_trainable": best_config["trainable"],
        "has_attention": (best_config["model_type"] == "attention"),
        "lstm_units": best_config["lstm_units"],
        "dense_units": best_config["dense_units"],
        "dropout": best_config["dropout"],
        "batch_size": batch_size,
        "total_params": val_df.loc[best_idx, "total_params"],
        "trainable_params": val_df.loc[best_idx, "trainable_params"],
        "best_epoch": val_df.loc[best_idx, "best_epoch"],
        "total_epochs": val_df.loc[best_idx, "total_epochs"],
        "accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "regular_f1": test_metrics["per_class"]["Regular"]["f1"],
        "sarcasm_f1": test_metrics["per_class"]["Sarcasm"]["f1"],
        "irony_f1": test_metrics["per_class"]["Irony"]["f1"],
        "figurative_f1": test_metrics["per_class"]["Figurative"]["f1"],
        "training_time": round(t_test_eval, 2),
    }
    results_table.append(test_record)

    full_results_df = pd.DataFrame(results_table)
    full_results_df.to_csv(results_dir / "model_results.csv", index=False)

    # Also evaluate FT-01 on test set for direct comparison if FT-02 won (or vice versa)
    for alt_id in ["FT-01", "FT-02"]:
        if alt_id != best_exp_id and alt_id in models_dict:
            alt_m = models_dict[alt_id]
            alt_metrics = evaluate_model(alt_m, test_loader, device=device)
            alt_rec = {
                "experiment_id": alt_id,
                "split": "test_comparison",
                "description": f"{alt_id} for scientific comparison",
                "embedding": "FastText-wiki-300d",
                "embedding_trainable": False,
                "has_attention": (alt_id == "FT-02"),
                "lstm_units": 64,
                "dense_units": 64,
                "dropout": 0.3,
                "batch_size": batch_size,
                "total_params": alt_m.count_params()[0],
                "trainable_params": alt_m.count_params()[1],
                "best_epoch": 0,
                "total_epochs": len(histories_dict[alt_id]["epoch"]),
                "accuracy": alt_metrics["accuracy"],
                "macro_precision": alt_metrics["macro_precision"],
                "macro_recall": alt_metrics["macro_recall"],
                "macro_f1": alt_metrics["macro_f1"],
                "weighted_f1": alt_metrics["weighted_f1"],
                "regular_f1": alt_metrics["per_class"]["Regular"]["f1"],
                "sarcasm_f1": alt_metrics["per_class"]["Sarcasm"]["f1"],
                "irony_f1": alt_metrics["per_class"]["Irony"]["f1"],
                "figurative_f1": alt_metrics["per_class"]["Figurative"]["f1"],
                "training_time": 0.0,
            }
            results_table.append(alt_rec)

    pd.DataFrame(results_table).to_csv(results_dir / "model_results.csv", index=False)

    # 9. Test Predictions, Errors, and Classification Reports
    test_preds_df = generate_predictions_df(best_model, test_loader, test_texts, test_labels, device=device)
    test_preds_df.to_csv(pred_dir / "test_predictions.csv", index=False)

    cr_str = generate_classification_report_str(test_labels, test_preds_df["predicted_id"].values)
    with open(cr_dir / "test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Phase 7 — Best Model ({best_exp_id}: {best_config['desc']}) Test Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(cr_str)

    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        title=f"Best Model ({best_exp_id}) Test Confusion Matrix",
        output_path=str(cm_dir / "test_confusion_matrix.png"),
        normalize=False,
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        title=f"Best Model ({best_exp_id}) Normalized Test Confusion Matrix",
        output_path=str(cm_dir / "test_confusion_matrix_normalized.png"),
        normalize=True,
    )

    error_summary = collect_error_breakdown(test_preds_df)
    with open(results_dir / "error_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(error_summary, f, indent=2)

    # 10. Attention Extraction and Visualization
    attn_model = models_dict.get("FT-02", best_model)
    if hasattr(attn_model, "has_attention") and attn_model.has_attention:
        logger.info("Extracting attention weights for representative instances...")
        # Select representative instances across classes and outcomes
        sample_indices = []
        for c in range(4):
            # 2 correct examples
            corr_idx = np.where((test_labels == c) & (test_preds_df["predicted_id"].values == c))[0]
            if len(corr_idx) > 0:
                sample_indices.extend(corr_idx[:2])
            # 2 incorrect examples
            err_idx = np.where((test_labels == c) & (test_preds_df["predicted_id"].values != c))[0]
            if len(err_idx) > 0:
                sample_indices.extend(err_idx[:2])

        sample_indices = list(dict.fromkeys(sample_indices))[:16]
        attn_records = extract_attention_examples(
            attn_model,
            tokenizer,
            [test_texts[i] for i in sample_indices],
            [test_labels[i] for i in sample_indices],
            max_len=max_len,
            device=device,
        )

        with open(attn_dir / "attention_examples.json", "w", encoding="utf-8") as f:
            json.dump(attn_records, f, indent=2)

        # Generate plots for top 6 examples
        for idx, rec in enumerate(attn_records[:6]):
            plot_attention_heatmap(
                rec["token_attention"],
                title=f"Example {idx+1} | True: {rec['actual_label']} -> Pred: {rec['predicted_label']} (Conf: {rec['confidence']:.2f})",
                output_path=str(attn_dir / f"attention_example_{idx+1}.png"),
            )

    # 11. Multi-Phase Model Progression (Phase 4, Phase 5, Phase 6, Phase 7)
    progression_data = [
        {
            "phase": "Phase 4",
            "model": "LinearSVM",
            "representation": "Bag of Words (n-grams 1-2)",
            "accuracy": 0.8457,
            "macro_precision": 0.7712,
            "macro_recall": 0.6934,
            "macro_f1": 0.7187,
        },
        {
            "phase": "Phase 5",
            "model": "LinearSVM",
            "representation": "TF-IDF + Text Features",
            "accuracy": 0.8565,
            "macro_precision": 0.7820,
            "macro_recall": 0.6890,
            "macro_f1": 0.7002,
        },
        {
            "phase": "Phase 6",
            "model": "GloVe + Bi-LSTM (BILSTM-02)",
            "representation": "GloVe 100d Dense Word Embeddings",
            "accuracy": 0.8612,
            "macro_precision": 0.8990,
            "macro_recall": 0.7500,
            "macro_f1": 0.6940,
        },
        {
            "phase": "Phase 7 (FT-01)",
            "model": "FastText + Bi-LSTM",
            "representation": "FastText 300d (No Attention)",
            "accuracy": [r["accuracy"] for r in results_table if r["experiment_id"] == "FT-01" and "test" in r["split"]][0] if any(r["experiment_id"] == "FT-01" and "test" in r["split"] for r in results_table) else test_metrics["accuracy"],
            "macro_precision": [r["macro_precision"] for r in results_table if r["experiment_id"] == "FT-01" and "test" in r["split"]][0] if any(r["experiment_id"] == "FT-01" and "test" in r["split"] for r in results_table) else test_metrics["macro_precision"],
            "macro_recall": [r["macro_recall"] for r in results_table if r["experiment_id"] == "FT-01" and "test" in r["split"]][0] if any(r["experiment_id"] == "FT-01" and "test" in r["split"] for r in results_table) else test_metrics["macro_recall"],
            "macro_f1": [r["macro_f1"] for r in results_table if r["experiment_id"] == "FT-01" and "test" in r["split"]][0] if any(r["experiment_id"] == "FT-01" and "test" in r["split"] for r in results_table) else test_metrics["macro_f1"],
        },
        {
            "phase": "Phase 7 (FT-02)",
            "model": "FastText + Bi-LSTM + Attention",
            "representation": "FastText 300d + Subwords + Attention",
            "accuracy": [r["accuracy"] for r in results_table if r["experiment_id"] == "FT-02" and "test" in r["split"]][0] if any(r["experiment_id"] == "FT-02" and "test" in r["split"] for r in results_table) else test_metrics["accuracy"],
            "macro_precision": [r["macro_precision"] for r in results_table if r["experiment_id"] == "FT-02" and "test" in r["split"]][0] if any(r["experiment_id"] == "FT-02" and "test" in r["split"] for r in results_table) else test_metrics["macro_precision"],
            "macro_recall": [r["macro_recall"] for r in results_table if r["experiment_id"] == "FT-02" and "test" in r["split"]][0] if any(r["experiment_id"] == "FT-02" and "test" in r["split"] for r in results_table) else test_metrics["macro_recall"],
            "macro_f1": [r["macro_f1"] for r in results_table if r["experiment_id"] == "FT-02" and "test" in r["split"]][0] if any(r["experiment_id"] == "FT-02" and "test" in r["split"] for r in results_table) else test_metrics["macro_f1"],
        },
    ]

    progression_df = pd.DataFrame(progression_data)
    progression_df.to_csv(BASE_DIR / "results" / "model_progression.csv", index=False)
    progression_df.to_csv(results_dir / "model_progression.csv", index=False)

    # 12. Save phase summary JSON
    phase7_summary = {
        "phase": "Phase 7 — FastText + Bi-LSTM + Attention",
        "framework": f"PyTorch {torch.__version__}",
        "device": device,
        "fasttext": {
            "source": FASTTEXT_METADATA["source"],
            "dimension": embed_dim,
            "exact_coverage_pct": coverage_stats["exact_coverage_pct"],
            "effective_coverage_pct": coverage_stats["effective_coverage_pct"],
            "subword_synthesis": True,
        },
        "tokenizer": {
            "vocabulary_size": tokenizer.vocab_size,
            "max_sequence_length": max_len,
            "mean_sequence_length": mean_len,
            "median_sequence_length": median_len,
            "pct_truncated": pct_truncated,
            "pct_padded": pct_padded,
        },
        "best_experiment": {
            "id": best_exp_id,
            "config": best_config,
            "val_macro_f1": float(best_val_f1),
        },
        "test_results": {
            "accuracy": float(test_metrics["accuracy"]),
            "macro_precision": float(test_metrics["macro_precision"]),
            "macro_recall": float(test_metrics["macro_recall"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "weighted_f1": float(test_metrics["weighted_f1"]),
            "per_class": test_metrics["per_class"],
        },
        "error_summary": error_summary,
        "progression": progression_data,
    }

    with open(results_dir / "phase7_summary.json", "w", encoding="utf-8") as f:
        json.dump(phase7_summary, f, indent=2)

    logger.info("Phase 7 experimental execution completed successfully!")
    return phase7_summary


if __name__ == "__main__":
    run_phase7_experiments(
        max_epochs=12,
        batch_size=64,
        learning_rate=0.001,
        patience=4,
        device="cpu",
    )
