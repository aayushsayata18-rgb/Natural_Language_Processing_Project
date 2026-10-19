"""
Phase 8 — BERT-Based Transformer Classification Package
======================================================
Sarcasm & Irony Detection System using NLP
MCA Final-Year Academic Project

Modules:
    - tokenizer: Pretrained BERT tokenizer management and sequence analysis
    - dataset: PyTorch Dataset, batching, and leakage-controlled dataloaders
    - model: Pretrained BERT sequence classification wrapper and architecture inspection
    - train: Fine-tuning training loop with AdamW, linear scheduler, and early stopping
    - evaluate: Evaluation metrics, confusion matrix, error analysis, classification reports
    - predict: Single-string and batch inference pipeline with probability output
    - utils: Reproducibility seeding, environment metadata, and visualization plotting
    - experiments: Full Phase 8 experimental execution orchestrator
"""

__version__ = "1.0.0"
__all__ = [
    "tokenizer",
    "dataset",
    "model",
    "train",
    "evaluate",
    "predict",
    "utils",
    "experiments",
]
