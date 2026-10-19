"""
Bi-LSTM with GloVe embeddings module for Sarcasm & Irony Detection.
"""

from src.deep_learning.bilstm.tokenizer import Tokenizer
from src.deep_learning.bilstm.glove import (
    load_glove_vectors,
    build_embedding_matrix,
    analyze_oov,
    save_coverage_report,
)
from src.deep_learning.bilstm.model import BiLSTMClassifier
from src.deep_learning.bilstm.train import train_model, create_dataloader, EarlyStopping
from src.deep_learning.bilstm.evaluate import (
    evaluate_model,
    generate_predictions_df,
    collect_errors,
    generate_classification_report_str,
    CLASS_NAMES,
)

__all__ = [
    "Tokenizer",
    "load_glove_vectors",
    "build_embedding_matrix",
    "analyze_oov",
    "save_coverage_report",
    "BiLSTMClassifier",
    "train_model",
    "create_dataloader",
    "EarlyStopping",
    "evaluate_model",
    "generate_predictions_df",
    "collect_errors",
    "generate_classification_report_str",
    "CLASS_NAMES",
]
