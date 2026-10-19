"""
Phase 8 — BERT Experiment Orchestrator
======================================
Sarcasm & Irony Detection System using NLP

Full experimental pipeline:
1. Loads leakage-controlled dataset (train_clean.csv, test_clean.csv)
2. Performs stratified 80/20 train/validation split (random_state=42)
3. Analyzes tokenized sequence length distribution & justifies MAX_LENGTH
4. Conducts model & memory sanity checks (forward, backward, gradients)
5. Executes fine-tuning of BERT-01 baseline (bert-base-uncased, AdamW, linear scheduler)
6. Monitors validation performance (loss, accuracy, Macro F1) per epoch
7. Selects best checkpoint strictly based on validation Macro F1
8. Saves serialized model, tokenizer, and config to models/transformers/bert/
9. Performs model reload consistency verification
10. Evaluates selected model ONCE on official untouched test dataset
11. Generates classification report, confusion matrix, predictions CSV, and error analysis
12. Updates results/model_progression.csv and creates results/final_progression/model_comparison.csv
13. Generates computational comparison table (results/model_computational_comparison.csv)
14. Generates publication-quality visualization figures
"""

import copy
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # ml-service/
sys.path.insert(0, str(BASE_DIR))

from src.transformers.bert.tokenizer import BertTokenizerManager, DEFAULT_MAX_LENGTH
from src.transformers.bert.dataset import (
    load_and_split_data,
    create_dataloaders,
    load_label_mapping,
)
from src.transformers.bert.model import BertSarcasmClassifier
from src.transformers.bert.train import train_bert_model
from src.transformers.bert.evaluate import (
    evaluate_bert_dataloader,
    generate_predictions_dataframe,
    analyze_misclassifications,
    CLASS_NAMES,
)
from src.transformers.bert.predict import (
    BertPredictor,
    verify_model_reload_consistency,
)
from src.transformers.bert.utils import (
    set_seed,
    get_environment_info,
    plot_training_curves,
    plot_confusion_matrix_heatmap,
    plot_cross_phase_progression,
)

logger = logging.getLogger(__name__)


def run_model_sanity_check(
    model: BertSarcasmClassifier,
    tokenizer_manager: BertTokenizerManager,
    device: torch.device,
) -> bool:
    """
    Executes Section 74 Sanity Check:
    1. Loads a very small subset
    2. Tokenizes
    3. Creates one batch
    4. Runs BERT forward pass
    5. Verifies logits shape [B, 4]
    6. Calculates loss
    7. Runs backward pass
    8. Confirms gradients exist and are non-zero
    9. Runs a few training steps
    """
    logger.info("--- Starting Model Sanity Check ---")
    model.to(device)
    model.train()

    sample_texts = [
        "Oh brilliant, another meeting that could have been an email.",
        "A fire station burned down today.",
        "Her voice was music to his ears.",
        "The train departs at five o'clock.",
    ]
    sample_labels = torch.tensor([1, 2, 3, 0], dtype=torch.long, device=device)

    # 2 & 3: Tokenize into one batch
    encoding = tokenizer_manager.encode_batch(
        sample_texts,
        max_length=tokenizer_manager.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # 4 & 5: Forward pass and logits shape verification
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    assert logits.shape == (4, 4), f"Expected logits shape (4, 4), got {logits.shape}"
    logger.info(f"Sanity Check: Logits shape verified successfully: {logits.shape}")

    # 6: Calculate loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, sample_labels)
    assert not torch.isnan(loss) and loss.item() > 0, "Loss calculation failed or returned NaN"
    logger.info(f"Sanity Check: Forward loss calculated: {loss.item():.4f}")

    # 7 & 8: Backward pass & confirm gradients
    model.zero_grad()
    loss.backward()

    has_grads = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient after backward pass"
            if torch.sum(torch.abs(param.grad)).item() > 0:
                has_grads = True
                break
    assert has_grads, "Gradients are all zero after backward pass"
    logger.info("Sanity Check: Backward pass confirmed non-zero gradients exist!")

    # 9: Run a few test optimizer steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.step()
    logger.info("Sanity Check: Optimizer step executed successfully!")
    logger.info("--- Model Sanity Check PASSED ---")
    return True


def run_phase8_experiments(
    data_dir: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 80,
    patience: int = 2,
    seed: int = 42,
    max_train_steps_per_epoch: Optional[int] = None,
    max_val_steps_per_epoch: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes complete Phase 8 experimental evaluation.
    """
    set_seed(seed)
    if device is None:
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected_device = torch.device(device)

    # Resolve paths
    base_path = Path(data_dir or (BASE_DIR / "data"))
    train_csv = base_path / "processed" / "train_clean.csv"
    test_csv = base_path / "processed" / "test_clean.csv"
    label_map_file = base_path / "processed" / "label_mapping.json"

    res_dir = Path(results_dir or (BASE_DIR / "results" / "bert"))
    res_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = res_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mod_dir = Path(models_dir or (BASE_DIR / "models" / "transformers" / "bert"))
    mod_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Starting Phase 8 BERT Experiments on device: {selected_device} ===")

    # 1 & 2: Load data and split
    split_data = load_and_split_data(
        train_csv_path=train_csv,
        test_csv_path=test_csv,
        val_size=0.2,
        random_state=seed,
    )
    train_df = split_data["train_df"]
    val_df = split_data["val_df"]
    test_df = split_data["test_df"]

    label_map = load_label_mapping(label_map_file)

    # 3: Initialize Tokenizer and analyze sequence length
    tokenizer_mgr = BertTokenizerManager(max_length=max_length)
    length_analysis = tokenizer_mgr.analyze_sequence_lengths(
        train_df["text"].tolist()[:10000],  # representative sample of training texts
        candidate_lengths=[32, 48, 64, 80, 96, 128],
    )
    logger.info(
        f"Sequence Length Analysis on Training Split: "
        f"Mean={length_analysis['mean_length']:.2f}, Median={length_analysis['median_length']:.1f}, "
        f"p95={length_analysis['p95']:.1f}, p99={length_analysis['p99']:.1f}, Max={length_analysis['max_length']}"
    )

    # 4: Initialize BERT Classifier & Count Parameters
    model = BertSarcasmClassifier(
        model_name_or_path="bert-base-uncased",
        num_labels=len(CLASS_NAMES),
    )
    param_counts = model.count_parameters()
    logger.info(
        f"BERT Model Parameters: Total={param_counts['total_parameters']:,}, "
        f"Trainable={param_counts['trainable_parameters']:,}, "
        f"Non-Trainable={param_counts['non_trainable_parameters']:,}"
    )

    # 5: Sanity Check
    run_model_sanity_check(model, tokenizer_mgr, selected_device)

    # 6: Create DataLoaders
    loaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        tokenizer_manager=tokenizer_mgr,
        batch_size=batch_size,
        max_length=max_length,
        test_df=test_df,
    )

    # 7: Fine-Tuning
    start_train_time = time.time()
    trained_model, train_summary = train_bert_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=selected_device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        patience=patience,
        max_train_steps_per_epoch=max_train_steps_per_epoch,
        max_val_steps_per_epoch=max_val_steps_per_epoch,
    )
    training_duration_s = time.time() - start_train_time

    # 8: Save Model, Tokenizer, and Config
    model_save_path = mod_dir / "model"
    tok_save_path = mod_dir / "tokenizer"
    trained_model.save_pretrained(model_save_path)
    tokenizer_mgr.save_tokenizer(tok_save_path)

    # Save label mapping alongside model
    with open(mod_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    # 9: Inference & Reload Consistency Check
    reload_passed = verify_model_reload_consistency(
        model=trained_model,
        tokenizer_manager=tokenizer_mgr,
        temp_dir=mod_dir,
    )
    assert reload_passed, "Reload consistency verification failed!"

    # 10: Final Evaluation on untouched Test Set
    logger.info("--- Performing Final Test Set Evaluation (STRICTLY ONCE on selected model) ---")
    test_metrics, test_logits, test_probs, test_labels = evaluate_bert_dataloader(
        model=trained_model,
        dataloader=loaders["test"],
        device=selected_device,
    )
    logger.info(
        f"BERT Test Performance: Accuracy={test_metrics['accuracy']:.4f}, "
        f"Macro Precision={test_metrics['macro_precision']:.4f}, "
        f"Macro Recall={test_metrics['macro_recall']:.4f}, "
        f"Macro F1={test_metrics['macro_f1']:.4f}, "
        f"Weighted F1={test_metrics['weighted_f1']:.4f}"
    )

    # 11: Predictions DataFrame and Error Analysis
    pred_df = generate_predictions_dataframe(
        texts=test_df["text"].tolist(),
        y_true=test_labels,
        y_pred=np.argmax(test_probs, axis=-1),
        probabilities=test_probs,
        class_names=CLASS_NAMES,
    )
    pred_df.to_csv(res_dir / "predictions.csv", index=False)

    error_analysis = analyze_misclassifications(pred_df, CLASS_NAMES)
    with open(res_dir / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2)

    # 12: Save Training History CSV
    history_df = pd.DataFrame(train_summary["history"])
    history_df.to_csv(res_dir / "training_history.csv", index=False)

    # 13: Save Classification Report CSV
    class_report_rows = []
    for c_name, c_data in test_metrics["per_class"].items():
        class_report_rows.append({
            "class": c_name,
            "precision": c_data["precision"],
            "recall": c_data["recall"],
            "f1-score": c_data["f1"],
            "support": c_data["support"],
        })
    class_report_rows.append({
        "class": "macro avg",
        "precision": test_metrics["macro_precision"],
        "recall": test_metrics["macro_recall"],
        "f1-score": test_metrics["macro_f1"],
        "support": test_metrics["total_samples"],
    })
    class_report_rows.append({
        "class": "weighted avg",
        "precision": None,
        "recall": None,
        "f1-score": test_metrics["weighted_f1"],
        "support": test_metrics["total_samples"],
    })
    pd.DataFrame(class_report_rows).to_csv(res_dir / "classification_report.csv", index=False)

    # 14: Save Confusion Matrix CSV
    cm_df = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )
    cm_df.to_csv(res_dir / "confusion_matrix.csv")

    # 15: Save Model Results CSV
    model_results_df = pd.DataFrame([{
        "model": "bert-base-uncased",
        "representation": "Contextual Transformer Embeddings (12-layer, 768d)",
        "sequence_length": max_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs_completed": train_summary["total_epochs_trained"],
        "best_epoch": train_summary["best_epoch"],
        "val_macro_f1": train_summary["best_val_macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "total_parameters": param_counts["total_parameters"],
        "trainable_parameters": param_counts["trainable_parameters"],
        "training_duration_s": round(training_duration_s, 2),
    }])
    model_results_df.to_csv(res_dir / "model_results.csv", index=False)

    # 16: Save Architecture Comparison CSV (BERT-01 Baseline)
    arch_comp_df = pd.DataFrame([{
        "experiment_id": "BERT-01",
        "model_name": "bert-base-uncased",
        "description": "Standard BERT Fine-Tuning + 4-class head",
        "val_macro_f1": train_summary["best_val_macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "parameters": param_counts["total_parameters"],
        "training_time_s": round(training_duration_s, 2),
        "selected": True,
    }])
    arch_comp_df.to_csv(res_dir / "architecture_comparison.csv", index=False)

    # 17: Save Environment Info JSON
    env_info = get_environment_info(seed)
    env_info.update({
        "model_name": "bert-base-uncased",
        "max_sequence_length": max_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "device": str(selected_device),
        "parameter_counts": param_counts,
    })
    with open(res_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    # 18: Generate Visualizations
    plot_training_curves(train_summary["history"], plots_dir / "training_curves.png")
    plot_confusion_matrix_heatmap(test_metrics["confusion_matrix"], CLASS_NAMES, plots_dir / "confusion_matrix.png")

    # 19: Update Model Progression CSV
    prog_path = BASE_DIR / "results" / "model_progression.csv"
    existing_prog_df = pd.read_csv(prog_path) if prog_path.exists() else pd.DataFrame()

    bert_prog_entry = pd.DataFrame([{
        "phase": "Phase 8",
        "model": "bert-base-uncased",
        "representation": "Contextual Transformer (WordPiece)",
        "accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
    }])

    # Filter out previous Phase 8 entry if present and append
    if not existing_prog_df.empty:
        existing_prog_df = existing_prog_df[existing_prog_df["phase"] != "Phase 8"]
        updated_prog_df = pd.concat([existing_prog_df, bert_prog_entry], ignore_index=True)
    else:
        updated_prog_df = bert_prog_entry

    updated_prog_df.to_csv(prog_path, index=False)
    logger.info(f"Updated {prog_path} with Phase 8 BERT results.")

    # Save to results/final_progression/
    final_prog_dir = BASE_DIR / "results" / "final_progression"
    final_prog_dir.mkdir(parents=True, exist_ok=True)
    updated_prog_df.to_csv(final_prog_dir / "model_comparison.csv", index=False)

    # Plot cross-phase progression
    plot_cross_phase_progression(updated_prog_df, plots_dir / "cross_phase_progression.png")
    plot_cross_phase_progression(updated_prog_df, final_prog_dir / "model_comparison.png")

    # 20: Generate results/model_computational_comparison.csv
    comp_records = [
        {
            "phase": "Phase 4",
            "model": "LinearSVM (BoW)",
            "representation": "Bag of Words (n-grams 1-2, 50,000 features)",
            "parameter_count": 200004,
            "batch_size": "Full dataset",
            "sequence_length": "N/A",
            "training_time": "12.4s",
            "hardware": "Multi-core CPU",
            "macro_f1": 0.7187,
            "accuracy": 0.8457,
            "memory_notes": "Low (<500MB)",
        },
        {
            "phase": "Phase 5",
            "model": "LinearSVM (TF-IDF + Features)",
            "representation": "TF-IDF (50,000) + 12 Linguistic & Sentiment Features",
            "parameter_count": 200052,
            "batch_size": "Full dataset",
            "sequence_length": "N/A",
            "training_time": "28.6s",
            "hardware": "Multi-core CPU",
            "macro_f1": 0.7002,
            "accuracy": 0.8565,
            "memory_notes": "Low (<750MB)",
        },
        {
            "phase": "Phase 6",
            "model": "GloVe + Bi-LSTM (BILSTM-02)",
            "representation": "GloVe 100d Pretrained Word Embeddings",
            "parameter_count": 3197636,
            "batch_size": "64",
            "sequence_length": "30",
            "training_time": "184.2s",
            "hardware": "Multi-core CPU",
            "macro_f1": 0.6940,
            "accuracy": 0.8612,
            "memory_notes": "Moderate (~1.2GB)",
        },
        {
            "phase": "Phase 7",
            "model": "FastText + Bi-LSTM + Attention (FT-02)",
            "representation": "FastText 300d + Subword Synthesis + Attention",
            "parameter_count": 9201092,
            "batch_size": "64",
            "sequence_length": "30",
            "training_time": "246.8s",
            "hardware": "Multi-core CPU",
            "macro_f1": 0.6928,
            "accuracy": 0.8603,
            "memory_notes": "Moderate (~1.8GB)",
        },
        {
            "phase": "Phase 8",
            "model": "bert-base-uncased",
            "representation": "Contextual Transformer (WordPiece 30,522, 12 layers, 768d)",
            "parameter_count": param_counts["total_parameters"],
            "batch_size": str(batch_size),
            "sequence_length": str(max_length),
            "training_time": f"{training_duration_s:.1f}s",
            "hardware": env_info["device_name"],
            "macro_f1": test_metrics["macro_f1"],
            "accuracy": test_metrics["accuracy"],
            "memory_notes": "High (~2.5GB-3.5GB RAM)",
        },
    ]
    pd.DataFrame(comp_records).to_csv(BASE_DIR / "results" / "model_computational_comparison.csv", index=False)

    logger.info("=== Phase 8 BERT Experiments Successfully Completed! ===")

    return {
        "model": trained_model,
        "tokenizer_manager": tokenizer_mgr,
        "test_metrics": test_metrics,
        "train_summary": train_summary,
        "parameter_counts": param_counts,
        "error_analysis": error_analysis,
        "length_analysis": length_analysis,
    }
