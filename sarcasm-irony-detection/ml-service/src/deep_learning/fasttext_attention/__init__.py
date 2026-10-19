"""
Phase 7 — FastText + Bi-LSTM + Attention Deep Learning Package
==============================================================
Sarcasm & Irony Detection System using NLP
"""

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
from src.deep_learning.fasttext_attention.attention import Attention
from src.deep_learning.fasttext_attention.model import (
    FastTextBiLSTMClassifier,
    FastTextAttentionClassifier,
)
from src.deep_learning.fasttext_attention.train import (
    train_model,
    create_dataloader,
    EarlyStopping,
)
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

__all__ = [
    "Tokenizer",
    "load_fasttext_vectors",
    "calculate_fasttext_coverage",
    "synthesize_subword_vector",
    "extract_character_ngrams",
    "categorize_token",
    "FASTTEXT_METADATA",
    "build_embedding_matrix",
    "Attention",
    "FastTextBiLSTMClassifier",
    "FastTextAttentionClassifier",
    "train_model",
    "create_dataloader",
    "EarlyStopping",
    "evaluate_model",
    "generate_predictions_df",
    "collect_error_breakdown",
    "extract_attention_examples",
    "generate_classification_report_str",
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_attention_heatmap",
    "CLASS_NAMES",
]
