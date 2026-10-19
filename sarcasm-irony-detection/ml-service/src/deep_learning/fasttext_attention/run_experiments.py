"""
Phase 7 — Entry Point for CLI Execution
=======================================
Sarcasm & Irony Detection System using NLP

Usage:
    cd ml-service
    python src/deep_learning/fasttext_attention/run_experiments.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # ml-service/
sys.path.insert(0, str(BASE_DIR))

from src.deep_learning.fasttext_attention.experiments import run_phase7_experiments

if __name__ == "__main__":
    run_phase7_experiments(
        max_epochs=12,
        batch_size=64,
        learning_rate=0.001,
        patience=4,
        device="cpu",
    )
