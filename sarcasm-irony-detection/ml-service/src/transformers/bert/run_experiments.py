"""
Phase 8 — BERT CLI Experiment Execution Entrypoint
=================================================
Sarcasm & Irony Detection System using NLP

Usage:
    python -m src.transformers.bert.run_experiments
    python -m src.transformers.bert.run_experiments --epochs 3 --batch_size 16 --max_length 80
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # ml-service/
sys.path.insert(0, str(BASE_DIR))

from src.transformers.bert.experiments import run_phase8_experiments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Phase8_Runner")


def main():
    parser = argparse.ArgumentParser(description="Phase 8 BERT Experiment Runner")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for AdamW")
    parser.add_argument("--max_length", type=int, default=80, help="Maximum sequence length")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train_steps", type=int, default=None, help="Max train steps per epoch")
    parser.add_argument("--val_steps", type=int, default=None, help="Max val steps per epoch")
    parser.add_argument("--threads", type=int, default=12, help="Number of PyTorch CPU threads")

    args = parser.parse_args()

    # Optimize CPU threading for multi-core processors
    if not torch.cuda.is_available() and args.threads:
        num_threads = min(args.threads, os.cpu_count() or 4)
        torch.set_num_threads(num_threads)
        logger.info(f"Set PyTorch CPU threads to {num_threads} (available logical CPUs: {os.cpu_count()})")

    logger.info("Executing Phase 8 BERT Experiments...")
    results = run_phase8_experiments(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        patience=args.patience,
        seed=args.seed,
        max_train_steps_per_epoch=args.train_steps,
        max_val_steps_per_epoch=args.val_steps,
    )

    metrics = results["test_metrics"]
    logger.info("=" * 60)
    logger.info("PHASE 8 FINAL BERT TEST RESULTS (UNTOUCHED TEST SET)")
    logger.info("=" * 60)
    logger.info(f"Accuracy:        {metrics['accuracy']:.4f}")
    logger.info(f"Macro Precision: {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    logger.info(f"Macro F1:        {metrics['macro_f1']:.4f} (Primary Selection Metric)")
    logger.info(f"Weighted F1:     {metrics['weighted_f1']:.4f}")
    logger.info("=" * 60)
    logger.info("Per-Class Results:")
    for c_name, c_data in metrics["per_class"].items():
        logger.info(
            f"  {c_name:12s} - Precision: {c_data['precision']:.4f} | "
            f"Recall: {c_data['recall']:.4f} | F1: {c_data['f1']:.4f} | Support: {c_data['support']}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
