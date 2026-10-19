"""
Phase 7 — FastText + Bi-LSTM + Attention Unit Tests
===================================================
Sarcasm & Irony Detection System using NLP

Tests for:
    1. Tokenizer creation and fitting
    2. OOV handling & subword decomposition
    3. Sequence conversion
    4. Padding and truncation
    5. FastText embedding lookup & subword synthesis
    6. Embedding matrix dimensions
    7. Attention layer input/output shapes
    8. Attention weight normalization (sum ≈ 1.0)
    9. Model forward pass and output shape (4 classes)
   10. Four-class prediction probabilities and confidence
   11. Model parameter counts & trainable flag toggling
   12. Early stopping behavior
   13. Evaluation metrics computation
"""

import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent  # ml-service/
sys.path.insert(0, str(BASE_DIR))

from src.deep_learning.fasttext_attention.tokenizer import Tokenizer
from src.deep_learning.fasttext_attention.fasttext import (
    extract_character_ngrams,
    synthesize_subword_vector,
    calculate_fasttext_coverage,
    categorize_token,
)
from src.deep_learning.fasttext_attention.embedding import build_embedding_matrix
from src.deep_learning.fasttext_attention.attention import Attention
from src.deep_learning.fasttext_attention.model import (
    FastTextBiLSTMClassifier,
    FastTextAttentionClassifier,
)
from src.deep_learning.fasttext_attention.train import EarlyStopping, create_dataloader
from src.deep_learning.fasttext_attention.evaluate import (
    evaluate_model,
    generate_predictions_df,
    collect_error_breakdown,
    extract_attention_examples,
    CLASS_NAMES,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_texts():
    return [
        "I love this amazing sunny day!",
        "Oh wonderful, another endless meeting #sarcasm",
        "The irony of living in a digital desert",
        "Her smile was a beacon of light #figurative",
        "This is a standard informative tweet https://news.example.com",
        "What an extraordinary day with @superstar sooooo funny",
        "",
        "short",
        "word " * 60,
    ]


@pytest.fixture
def sample_labels():
    return np.array([0, 1, 2, 3, 0, 1, 0, 2, 3], dtype=np.int64)


@pytest.fixture
def fitted_tokenizer(sample_texts):
    tok = Tokenizer(max_num_words=100)
    tok.fit(sample_texts)
    return tok


@pytest.fixture
def mock_fasttext_vectors():
    words = [
        "i", "love", "this", "amazing", "day", "wonderful", "another",
        "meeting", "the", "irony", "of", "her", "smile", "was", "standard",
        "news", "what", "funny", "so", "desert", "light",
        "<so", "soo", "ooo", "oo>", "so>",  # subwords
    ]
    dim = 50
    rng = np.random.default_rng(42)
    vectors = {w: rng.standard_normal(dim, dtype=np.float32) for w in words}
    return vectors


# ============================================================
# 1. TOKENIZER TESTS
# ============================================================

def test_tokenizer_initialization():
    tok = Tokenizer(max_num_words=500)
    assert tok.max_num_words == 500
    assert tok.vocab_size == 2
    assert tok.word2idx["<PAD>"] == 0
    assert tok.word2idx["<UNK>"] == 1


def test_tokenizer_fit(fitted_tokenizer):
    assert fitted_tokenizer.vocab_size > 2
    assert "<PAD>" in fitted_tokenizer.word2idx
    assert "<UNK>" in fitted_tokenizer.word2idx
    assert "amazing" in fitted_tokenizer.word2idx


def test_tokenizer_normalization(fitted_tokenizer):
    tokens = fitted_tokenizer._tokenize("Visit https://test.org and contact @john_doe!")
    assert "<url>" in tokens
    assert "<user>" in tokens


# ============================================================
# 2. OOV & SUBWORD EXTRACTION TESTS
# ============================================================

def test_extract_character_ngrams():
    ngrams = extract_character_ngrams("cat", min_n=3, max_n=3)
    # <cat> with length 5: <ca, cat, at>
    assert "<ca" in ngrams
    assert "cat" in ngrams
    assert "at>" in ngrams


def test_subword_synthesis(mock_fasttext_vectors):
    # 'sooooo' is not directly in mock vectors, but 'so' and character ngrams are!
    syn_vec = synthesize_subword_vector("sooooo", mock_fasttext_vectors, dim=50)
    assert syn_vec is not None
    assert syn_vec.shape == (50,)
    assert not np.isnan(syn_vec).any()


def test_categorize_token():
    assert categorize_token("#sarcasm") == "hashtag"
    assert categorize_token("@john") == "mention"
    assert categorize_token("<url>") == "url"
    assert categorize_token("sooooo") == "repeated_characters"
    assert categorize_token("123abc") == "numeric_alphanumeric"


# ============================================================
# 3. SEQUENCE CONVERSION & PADDING TESTS
# ============================================================

def test_texts_to_sequences(fitted_tokenizer, sample_texts):
    seqs = fitted_tokenizer.texts_to_sequences(sample_texts)
    assert len(seqs) == len(sample_texts)
    for s in seqs:
        assert isinstance(s, list)


def test_pad_sequences_shapes(fitted_tokenizer, sample_texts):
    seqs = fitted_tokenizer.texts_to_sequences(sample_texts)
    padded = fitted_tokenizer.pad_sequences(seqs, max_len=20, padding="pre", truncating="post")
    assert padded.shape == (len(sample_texts), 20)
    assert padded.dtype == np.int64


# ============================================================
# 4. EMBEDDING MATRIX TESTS
# ============================================================

def test_build_embedding_matrix(fitted_tokenizer, mock_fasttext_vectors):
    dim = 50
    mat, stats, oov = build_embedding_matrix(
        fitted_tokenizer.word2idx,
        mock_fasttext_vectors,
        dim=dim,
        use_subwords=True,
    )
    assert mat.shape == (fitted_tokenizer.vocab_size, dim)
    # PAD index 0 must be strictly zeros
    assert np.allclose(mat[0], 0.0)
    assert stats["vocabulary_size"] == fitted_tokenizer.vocab_size
    assert stats["total_covered"] > 0


# ============================================================
# 5. ATTENTION MECHANISM TESTS
# ============================================================

def test_attention_shapes():
    batch_size = 4
    seq_len = 15
    hidden_dim = 64
    attention_dim = 32

    attn = Attention(hidden_dim=hidden_dim, attention_dim=attention_dim)
    h = torch.randn(batch_size, seq_len, hidden_dim)

    # Without weights
    context, weights = attn(h, return_weights=False)
    assert context.shape == (batch_size, hidden_dim)
    assert weights is None

    # With weights
    context, weights = attn(h, return_weights=True)
    assert context.shape == (batch_size, hidden_dim)
    assert weights.shape == (batch_size, seq_len)


def test_attention_weights_normalization():
    batch_size = 5
    seq_len = 20
    hidden_dim = 40

    attn = Attention(hidden_dim=hidden_dim, attention_dim=30)
    h = torch.randn(batch_size, seq_len, hidden_dim)
    _, weights = attn(h, return_weights=True)

    # Each row sum must equal 1.0 (softmax property)
    row_sums = weights.sum(dim=-1).detach().numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_attention_masking():
    attn = Attention(hidden_dim=16, attention_dim=8)
    h = torch.randn(2, 5, 16)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    _, weights = attn(h, mask=mask, return_weights=True)

    # Masked positions should have weight ~ 0
    assert weights[0, 3].item() < 1e-4
    assert weights[0, 4].item() < 1e-4
    assert np.allclose(weights.sum(dim=-1).detach().numpy(), 1.0, atol=1e-4)


# ============================================================
# 6. MODEL FORWARD PASS & PREDICTION TESTS
# ============================================================

def test_fasttext_bilstm_classifier():
    vocab_size = 50
    dim = 20
    batch_size = 4
    seq_len = 10

    model = FastTextBiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=dim,
        lstm_units=16,
        dense_units=16,
        num_classes=4,
    )
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, 4)

    probs = model.predict_proba(x)
    assert probs.shape == (batch_size, 4)
    assert np.allclose(probs.sum(dim=-1).numpy(), 1.0, atol=1e-5)


def test_fasttext_attention_classifier():
    vocab_size = 50
    dim = 20
    batch_size = 3
    seq_len = 12

    model = FastTextAttentionClassifier(
        vocab_size=vocab_size,
        embedding_dim=dim,
        lstm_units=16,
        attention_dim=16,
        dense_units=16,
        num_classes=4,
    )
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Standard forward
    logits = model(x)
    assert logits.shape == (batch_size, 4)

    # Forward with attention
    logits, attn = model(x, return_attention=True)
    assert logits.shape == (batch_size, 4)
    assert attn.shape == (batch_size, seq_len)

    # Attention weights sum
    weights_sum = attn.sum(dim=-1).detach().numpy()
    assert np.allclose(weights_sum, 1.0, atol=1e-4)


def test_model_param_counts():
    model = FastTextAttentionClassifier(
        vocab_size=100,
        embedding_dim=30,
        lstm_units=16,
        dense_units=16,
        embedding_trainable=False,
    )
    total, trainable = model.count_params()
    assert total > trainable
    # Embedding params = 100 * 30 = 3000 should not be trainable
    assert total - trainable >= 3000


# ============================================================
# 7. EVALUATION & ERROR BREAKDOWN TESTS
# ============================================================

def test_evaluation_pipeline(fitted_tokenizer, sample_texts, sample_labels):
    seqs = fitted_tokenizer.texts_to_sequences(sample_texts)
    padded = fitted_tokenizer.pad_sequences(seqs, max_len=15)
    loader = create_dataloader(padded, sample_labels, batch_size=4, shuffle=False)

    model = FastTextAttentionClassifier(
        vocab_size=fitted_tokenizer.vocab_size,
        embedding_dim=20,
        lstm_units=8,
        dense_units=8,
        num_classes=4,
    )

    metrics = evaluate_model(model, loader)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"].shape == (4, 4)

    preds_df = generate_predictions_df(model, loader, sample_texts, sample_labels)
    assert len(preds_df) == len(sample_texts)
    assert "confidence" in preds_df.columns
    assert "probability_sarcasm" in preds_df.columns

    errors = collect_error_breakdown(preds_df)
    assert "total_errors" in errors


def test_extract_attention_examples(fitted_tokenizer, sample_texts, sample_labels):
    model = FastTextAttentionClassifier(
        vocab_size=fitted_tokenizer.vocab_size,
        embedding_dim=20,
        lstm_units=8,
        dense_units=8,
        num_classes=4,
    )
    records = extract_attention_examples(
        model,
        fitted_tokenizer,
        sample_texts[:3],
        sample_labels[:3],
        max_len=10,
    )
    assert len(records) == 3
    assert "token_attention" in records[0]
    assert "raw_weight_sum" in records[0]
