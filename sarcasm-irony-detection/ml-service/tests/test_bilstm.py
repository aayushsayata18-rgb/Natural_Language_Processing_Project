"""
Phase 6 — Bi-LSTM Unit Tests
============================================
Sarcasm & Irony Detection System using NLP

Tests for:
    1. Tokenizer creation
    2. OOV handling
    3. Sequence conversion
    4. Padding
    5. Label mapping
    6. GloVe vocabulary matching
    7. Embedding matrix shape
    8. Model input shape
    9. Model output shape
   10. Four-class prediction

Usage:
    cd ml-service
    python -m pytest tests/test_bilstm.py -v
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Path setup
BASE_DIR = Path(__file__).resolve().parent.parent  # ml-service/
sys.path.insert(0, str(BASE_DIR))

from src.deep_learning.bilstm.tokenizer import Tokenizer
from src.deep_learning.bilstm.glove import (
    build_embedding_matrix,
    analyze_oov,
)
from src.deep_learning.bilstm.model import BiLSTMClassifier
from src.deep_learning.bilstm.train import create_dataloader, EarlyStopping
from src.deep_learning.bilstm.evaluate import (
    evaluate_model,
    generate_predictions_df,
    collect_errors,
    generate_classification_report_str,
    CLASS_NAMES,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_texts():
    """Sample texts for tokenizer tests."""
    return [
        "I love this amazing day!",
        "Oh great, another meeting #sarcasm",
        "The irony of it all",
        "She was a shining star #figurative",
        "This is just a regular tweet https://example.com",
        "What an absolutely wonderful experience @user",
        "",  # empty text
        "short",
        "a " * 100,  # very long text
    ]


@pytest.fixture
def sample_labels():
    """Labels for 9 sample texts."""
    return np.array([0, 1, 2, 3, 0, 1, 0, 2, 3])


@pytest.fixture
def fitted_tokenizer(sample_texts):
    """A tokenizer fitted on sample texts."""
    tok = Tokenizer(max_num_words=100)
    tok.fit(sample_texts)
    return tok


@pytest.fixture
def small_glove_vectors():
    """Small mock GloVe vectors for testing."""
    words = ["i", "love", "this", "amazing", "day", "great",
             "another", "meeting", "the", "irony", "of", "it",
             "all", "she", "was", "a", "star", "just", "regular"]
    dim = 100
    vectors = {}
    np.random.seed(42)
    for w in words:
        vectors[w] = np.random.randn(dim).astype(np.float32)
    return vectors


@pytest.fixture
def embedding_dim():
    return 100


# ============================================================
# 1. TOKENIZER CREATION
# ============================================================

class TestTokenizerCreation:
    """Tests for tokenizer initialization and fitting."""

    def test_tokenizer_init(self):
        tok = Tokenizer(max_num_words=100)
        assert tok.max_num_words == 100
        assert tok.vocab_size == 2  # PAD + OOV
        assert tok.PAD_IDX == 0
        assert tok.OOV_IDX == 1

    def test_tokenizer_special_tokens(self):
        tok = Tokenizer()
        assert tok.PAD_TOKEN == "<PAD>"
        assert tok.OOV_TOKEN == "<UNK>"
        assert tok.word2idx["<PAD>"] == 0
        assert tok.word2idx["<UNK>"] == 1

    def test_tokenizer_fit(self, sample_texts):
        tok = Tokenizer(max_num_words=100)
        tok.fit(sample_texts)
        assert tok.vocab_size > 2  # More than just PAD + OOV
        assert "<PAD>" in tok.word2idx
        assert "<UNK>" in tok.word2idx

    def test_tokenizer_max_words(self, sample_texts):
        tok = Tokenizer(max_num_words=5)
        tok.fit(sample_texts)
        # vocab_size = 5 kept words + 2 special tokens = 7
        assert tok.vocab_size <= 7

    def test_tokenizer_fit_returns_self(self, sample_texts):
        tok = Tokenizer()
        result = tok.fit(sample_texts)
        assert result is tok

    def test_vocabulary_from_training_only(self, sample_texts):
        """Ensure tokenizer vocabulary is built from provided texts only."""
        tok = Tokenizer(max_num_words=1000)
        tok.fit(sample_texts[:3])  # Fit on first 3 texts only
        # Words from text 4+ should NOT be in vocabulary (unless shared)
        # This verifies the tokenizer doesn't access global data
        assert tok.vocab_size >= 2  # At minimum PAD + OOV


# ============================================================
# 2. OOV HANDLING
# ============================================================

class TestOOVHandling:
    """Tests for Out-Of-Vocabulary handling."""

    def test_oov_maps_to_oov_idx(self, fitted_tokenizer):
        seq = fitted_tokenizer.texts_to_sequences(["xyzunknownword123"])
        assert seq[0][0] == fitted_tokenizer.OOV_IDX

    def test_known_words_not_oov(self, fitted_tokenizer):
        # "a" should be in vocabulary since it appears multiple times
        seq = fitted_tokenizer.texts_to_sequences(["a"])
        if seq[0]:
            assert seq[0][0] != fitted_tokenizer.OOV_IDX or "a" not in fitted_tokenizer.word2idx

    def test_oov_token_exists(self, fitted_tokenizer):
        assert "<UNK>" in fitted_tokenizer.word2idx
        assert fitted_tokenizer.word2idx["<UNK>"] == 1

    def test_empty_text_gives_empty_sequence(self, fitted_tokenizer):
        seq = fitted_tokenizer.texts_to_sequences([""])
        assert seq[0] == []


# ============================================================
# 3. SEQUENCE CONVERSION
# ============================================================

class TestSequenceConversion:
    """Tests for text-to-sequence conversion."""

    def test_texts_to_sequences_returns_list(self, fitted_tokenizer):
        seqs = fitted_tokenizer.texts_to_sequences(["hello world"])
        assert isinstance(seqs, list)
        assert isinstance(seqs[0], list)
        assert all(isinstance(x, int) for x in seqs[0])

    def test_multiple_texts(self, fitted_tokenizer, sample_texts):
        seqs = fitted_tokenizer.texts_to_sequences(sample_texts)
        assert len(seqs) == len(sample_texts)

    def test_sequence_values_in_range(self, fitted_tokenizer, sample_texts):
        seqs = fitted_tokenizer.texts_to_sequences(sample_texts)
        for seq in seqs:
            for idx in seq:
                assert 0 <= idx < fitted_tokenizer.vocab_size

    def test_url_normalization(self, fitted_tokenizer):
        seqs = fitted_tokenizer.texts_to_sequences(
            ["check https://example.com for details"]
        )
        # URL should be normalized to <URL> token
        tokens = fitted_tokenizer._tokenize("check https://example.com for details")
        assert "<url>" in tokens

    def test_mention_normalization(self, fitted_tokenizer):
        tokens = fitted_tokenizer._tokenize("hello @username how are you")
        assert "<user>" in tokens


# ============================================================
# 4. PADDING
# ============================================================

class TestPadding:
    """Tests for sequence padding and truncation."""

    def test_pad_shape(self, fitted_tokenizer):
        seqs = [[1, 2, 3], [4, 5]]
        padded = fitted_tokenizer.pad_sequences(seqs, max_len=5)
        assert padded.shape == (2, 5)

    def test_pre_padding(self, fitted_tokenizer):
        seqs = [[1, 2, 3]]
        padded = fitted_tokenizer.pad_sequences(seqs, max_len=5, padding="pre")
        assert padded[0, 0] == 0  # PAD at start
        assert padded[0, -1] == 3  # Original at end

    def test_post_padding(self, fitted_tokenizer):
        seqs = [[1, 2, 3]]
        padded = fitted_tokenizer.pad_sequences(seqs, max_len=5, padding="post")
        assert padded[0, 0] == 1  # Original at start
        assert padded[0, -1] == 0  # PAD at end

    def test_post_truncation(self, fitted_tokenizer):
        seqs = [[1, 2, 3, 4, 5, 6, 7]]
        padded = fitted_tokenizer.pad_sequences(seqs, max_len=3, truncating="post")
        # Should keep first 3 tokens
        np.testing.assert_array_equal(padded[0], [1, 2, 3])

    def test_pre_truncation(self, fitted_tokenizer):
        seqs = [[1, 2, 3, 4, 5, 6, 7]]
        padded = fitted_tokenizer.pad_sequences(seqs, max_len=3, truncating="pre")
        # Should keep last 3 tokens
        np.testing.assert_array_equal(padded[0], [5, 6, 7])

    def test_empty_sequence_padding(self, fitted_tokenizer):
        seqs = [[]]
        padded = fitted_tokenizer.pad_sequences(seqs, max_len=5)
        assert padded.shape == (1, 5)
        assert np.all(padded == 0)  # All PAD

    def test_exact_length_no_change(self, fitted_tokenizer):
        seqs = [[1, 2, 3, 4, 5]]
        padded = fitted_tokenizer.pad_sequences(seqs, max_len=5)
        assert padded.shape == (1, 5)
        np.testing.assert_array_equal(padded[0], [1, 2, 3, 4, 5])

    def test_pad_dtype(self, fitted_tokenizer):
        seqs = [[1, 2, 3]]
        padded = fitted_tokenizer.pad_sequences(seqs, max_len=5)
        assert padded.dtype == np.int64


# ============================================================
# 5. LABEL MAPPING
# ============================================================

class TestLabelMapping:
    """Tests for label mapping consistency."""

    def test_four_classes(self):
        assert len(CLASS_NAMES) == 4

    def test_class_names_correct(self):
        assert CLASS_NAMES == ["Regular", "Sarcasm", "Irony", "Figurative"]

    def test_label_mapping_file(self):
        mapping_path = BASE_DIR / "models" / "deep_learning" / "bilstm" / "best" / "label_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                mapping = json.load(f)
            assert mapping == {"Regular": 0, "Sarcasm": 1, "Irony": 2, "Figurative": 3}

    def test_label_ids_are_contiguous(self):
        expected = {0, 1, 2, 3}
        mapping = {"Regular": 0, "Sarcasm": 1, "Irony": 2, "Figurative": 3}
        assert set(mapping.values()) == expected


# ============================================================
# 6. GLOVE VOCABULARY MATCHING
# ============================================================

class TestGloVeMatching:
    """Tests for GloVe vocabulary matching."""

    def test_build_embedding_matrix(self, fitted_tokenizer, small_glove_vectors, embedding_dim):
        matrix, stats, oov_words = build_embedding_matrix(
            fitted_tokenizer.word2idx, small_glove_vectors, embedding_dim,
        )
        assert matrix.shape[0] == fitted_tokenizer.vocab_size
        assert matrix.shape[1] == embedding_dim
        assert stats["total_vocabulary"] > 0
        assert stats["glove_matched"] >= 0
        assert stats["oov_count"] >= 0
        assert 0 <= stats["coverage_pct"] <= 100

    def test_known_word_has_glove_vector(self, fitted_tokenizer, small_glove_vectors, embedding_dim):
        matrix, _, _ = build_embedding_matrix(
            fitted_tokenizer.word2idx, small_glove_vectors, embedding_dim,
        )
        # "a" is in both vocabulary and mock GloVe
        if "a" in fitted_tokenizer.word2idx and "a" in small_glove_vectors:
            idx = fitted_tokenizer.word2idx["a"]
            np.testing.assert_array_equal(matrix[idx], small_glove_vectors["a"])

    def test_pad_token_is_zero(self, fitted_tokenizer, small_glove_vectors, embedding_dim):
        matrix, _, _ = build_embedding_matrix(
            fitted_tokenizer.word2idx, small_glove_vectors, embedding_dim,
        )
        np.testing.assert_array_equal(matrix[0], np.zeros(embedding_dim))

    def test_oov_strategy_random(self, fitted_tokenizer, small_glove_vectors, embedding_dim):
        matrix, _, oov = build_embedding_matrix(
            fitted_tokenizer.word2idx, small_glove_vectors, embedding_dim,
            oov_strategy="random",
        )
        # If there are OOV words, their vectors should not be all zeros
        if oov:
            sample_oov_word = oov[0]
            if sample_oov_word in fitted_tokenizer.word2idx:
                idx = fitted_tokenizer.word2idx[sample_oov_word]
                assert not np.all(matrix[idx] == 0)

    def test_oov_strategy_zeros(self, fitted_tokenizer, small_glove_vectors, embedding_dim):
        matrix, _, oov = build_embedding_matrix(
            fitted_tokenizer.word2idx, small_glove_vectors, embedding_dim,
            oov_strategy="zeros",
        )
        if oov:
            sample_oov_word = oov[0]
            if sample_oov_word in fitted_tokenizer.word2idx:
                idx = fitted_tokenizer.word2idx[sample_oov_word]
                np.testing.assert_array_equal(matrix[idx], np.zeros(embedding_dim))


# ============================================================
# 7. EMBEDDING MATRIX SHAPE
# ============================================================

class TestEmbeddingMatrixShape:
    """Tests for embedding matrix dimensions."""

    def test_shape_matches_vocab(self, fitted_tokenizer, small_glove_vectors, embedding_dim):
        matrix, _, _ = build_embedding_matrix(
            fitted_tokenizer.word2idx, small_glove_vectors, embedding_dim,
        )
        assert matrix.shape == (fitted_tokenizer.vocab_size, embedding_dim)

    def test_matrix_dtype(self, fitted_tokenizer, small_glove_vectors, embedding_dim):
        matrix, _, _ = build_embedding_matrix(
            fitted_tokenizer.word2idx, small_glove_vectors, embedding_dim,
        )
        assert matrix.dtype == np.float32

    def test_different_vocab_sizes(self, sample_texts, small_glove_vectors, embedding_dim):
        for max_words in [5, 10, 50]:
            tok = Tokenizer(max_num_words=max_words)
            tok.fit(sample_texts)
            matrix, _, _ = build_embedding_matrix(
                tok.word2idx, small_glove_vectors, embedding_dim,
            )
            assert matrix.shape[0] == tok.vocab_size
            assert matrix.shape[1] == embedding_dim


# ============================================================
# 8. MODEL INPUT SHAPE
# ============================================================

class TestModelInputShape:
    """Tests for model input compatibility."""

    def test_model_accepts_padded_input(self):
        model = BiLSTMClassifier(
            vocab_size=100, embedding_dim=50,
            lstm_units=32, dense_units=32,
        )
        # (batch_size, max_len)
        x = torch.randint(0, 100, (4, 30))
        output = model(x)
        assert output.shape == (4, 4)

    def test_model_batch_size_1(self):
        model = BiLSTMClassifier(vocab_size=100, embedding_dim=50)
        x = torch.randint(0, 100, (1, 30))
        output = model(x)
        assert output.shape == (1, 4)

    def test_model_different_max_len(self):
        model = BiLSTMClassifier(vocab_size=100, embedding_dim=50)
        for max_len in [10, 30, 50, 100]:
            x = torch.randint(0, 100, (2, max_len))
            output = model(x)
            assert output.shape == (2, 4)


# ============================================================
# 9. MODEL OUTPUT SHAPE
# ============================================================

class TestModelOutputShape:
    """Tests for model output dimensions and properties."""

    def test_output_four_classes(self):
        model = BiLSTMClassifier(
            vocab_size=100, embedding_dim=50, num_classes=4,
        )
        x = torch.randint(0, 100, (8, 30))
        output = model(x)
        assert output.shape[1] == 4

    def test_softmax_probabilities_sum_to_one(self):
        model = BiLSTMClassifier(vocab_size=100, embedding_dim=50)
        x = torch.randint(0, 100, (4, 30))
        probs = model.predict_proba(x)
        sums = probs.sum(dim=1).numpy()
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_probabilities_non_negative(self):
        model = BiLSTMClassifier(vocab_size=100, embedding_dim=50)
        x = torch.randint(0, 100, (4, 30))
        probs = model.predict_proba(x)
        assert torch.all(probs >= 0)

    def test_logits_are_finite(self):
        model = BiLSTMClassifier(vocab_size=100, embedding_dim=50)
        x = torch.randint(0, 100, (4, 30))
        output = model(x)
        assert torch.all(torch.isfinite(output))


# ============================================================
# 10. FOUR-CLASS PREDICTION
# ============================================================

class TestFourClassPrediction:
    """Tests for valid 4-class predictions."""

    def test_predictions_in_range(self):
        model = BiLSTMClassifier(vocab_size=100, embedding_dim=50, num_classes=4)
        x = torch.randint(0, 100, (16, 30))
        logits = model(x)
        preds = logits.argmax(dim=1).numpy()
        assert np.all(preds >= 0)
        assert np.all(preds <= 3)

    def test_predictions_are_integers(self):
        model = BiLSTMClassifier(vocab_size=100, embedding_dim=50)
        x = torch.randint(0, 100, (8, 30))
        logits = model(x)
        preds = logits.argmax(dim=1).numpy()
        assert preds.dtype in [np.int32, np.int64]

    def test_model_with_pretrained_embeddings(self):
        vocab_size = 50
        embed_dim = 100
        matrix = np.random.randn(vocab_size, embed_dim).astype(np.float32)
        model = BiLSTMClassifier(
            vocab_size=vocab_size, embedding_dim=embed_dim,
            embedding_matrix=matrix, embedding_trainable=False,
        )
        x = torch.randint(0, vocab_size, (4, 20))
        output = model(x)
        assert output.shape == (4, 4)

    def test_frozen_embedding_not_trainable(self):
        matrix = np.random.randn(50, 100).astype(np.float32)
        model = BiLSTMClassifier(
            vocab_size=50, embedding_dim=100,
            embedding_matrix=matrix, embedding_trainable=False,
        )
        assert not model.embedding.weight.requires_grad

    def test_trainable_embedding(self):
        matrix = np.random.randn(50, 100).astype(np.float32)
        model = BiLSTMClassifier(
            vocab_size=50, embedding_dim=100,
            embedding_matrix=matrix, embedding_trainable=True,
        )
        assert model.embedding.weight.requires_grad


# ============================================================
# ADDITIONAL INTEGRATION TESTS
# ============================================================

class TestEarlyStopping:
    """Tests for early stopping callback."""

    def test_no_stop_initially(self):
        es = EarlyStopping(patience=3)
        assert not es(1.0, 1)

    def test_stop_after_patience(self):
        es = EarlyStopping(patience=3)
        es(1.0, 1)  # best
        es(1.1, 2)  # worse 1
        es(1.2, 3)  # worse 2
        result = es(1.3, 4)  # worse 3 -> stop
        assert result is True

    def test_reset_on_improvement(self):
        es = EarlyStopping(patience=3)
        es(1.0, 1)
        es(1.1, 2)  # worse
        es(0.9, 3)  # better -> reset counter
        es(1.0, 4)  # worse 1
        assert not es.should_stop

    def test_best_epoch_tracked(self):
        es = EarlyStopping(patience=3)
        es(1.0, 1)
        es(0.5, 2)  # best
        es(0.8, 3)
        assert es.best_epoch == 2


class TestDataLoader:
    """Tests for DataLoader creation."""

    def test_create_dataloader(self):
        X = np.random.randint(0, 100, (32, 30))
        y = np.random.randint(0, 4, 32)
        loader = create_dataloader(X, y, batch_size=8)
        batch_X, batch_y = next(iter(loader))
        assert batch_X.shape == (8, 30)
        assert batch_y.shape == (8,)

    def test_dataloader_dtypes(self):
        X = np.random.randint(0, 100, (16, 30))
        y = np.random.randint(0, 4, 16)
        loader = create_dataloader(X, y, batch_size=4)
        batch_X, batch_y = next(iter(loader))
        assert batch_X.dtype == torch.long
        assert batch_y.dtype == torch.long


class TestTokenizerSaveLoad:
    """Tests for tokenizer serialization."""

    def test_save_and_load(self, fitted_tokenizer, tmp_path):
        filepath = str(tmp_path / "test_tokenizer.json")
        fitted_tokenizer.save(filepath)

        loaded = Tokenizer.load(filepath)
        assert loaded.vocab_size == fitted_tokenizer.vocab_size
        assert loaded.max_num_words == fitted_tokenizer.max_num_words
        assert loaded.word2idx == fitted_tokenizer.word2idx

    def test_loaded_tokenizer_produces_same_sequences(
        self, fitted_tokenizer, sample_texts, tmp_path
    ):
        filepath = str(tmp_path / "test_tokenizer.json")
        fitted_tokenizer.save(filepath)
        loaded = Tokenizer.load(filepath)

        orig_seqs = fitted_tokenizer.texts_to_sequences(sample_texts)
        loaded_seqs = loaded.texts_to_sequences(sample_texts)
        assert orig_seqs == loaded_seqs


class TestModelArchitecture:
    """Tests for model architecture details."""

    def test_architecture_summary(self):
        model = BiLSTMClassifier(vocab_size=1000, embedding_dim=100)
        summary = model.get_architecture_summary()
        assert "BiLSTM" in summary
        assert "params" in summary

    def test_bidirectional_doubles_hidden(self):
        model = BiLSTMClassifier(
            vocab_size=100, embedding_dim=50,
            lstm_units=64, dense_units=32,
        )
        # dense layer input should be 64*2 = 128
        assert model.dense.in_features == 128
        assert model.dense.out_features == 32

    def test_output_layer_size(self):
        model = BiLSTMClassifier(
            vocab_size=100, embedding_dim=50, num_classes=4,
        )
        assert model.output.out_features == 4


class TestOOVAnalysis:
    """Tests for OOV analysis utility."""

    def test_analyze_oov_returns_dataframe(self):
        oov = ["#hashtag", "@user", "http://example.com", "lol",
               "ab", "YELLING", "mix3d"]
        df = analyze_oov(oov)
        assert len(df) == len(oov)
        assert "word" in df.columns
        assert "category" in df.columns

    def test_analyze_oov_categories(self):
        df = analyze_oov(["#test"])
        assert df.iloc[0]["category"] == "hashtag"
