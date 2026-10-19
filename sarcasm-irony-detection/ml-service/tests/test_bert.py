"""
Phase 8 — BERT Transformer Classification Unit Tests
====================================================
Sarcasm & Irony Detection System using NLP

Tests for:
    1. BERT tokenizer loading & vocabulary properties
    2. WordPiece subword tokenization & decomposition
    3. Special tokens handling ([CLS], [SEP], [PAD], [UNK])
    4. Dataset encoding, input shapes, and attention mask
    5. Label mapping preservation (Regular: 0, Sarcasm: 1, Irony: 2, Figurative: 3)
    6. Model architecture & output logits shape [B, 4]
    7. Parameter counting (total, trainable, non-trainable)
    8. Cross-entropy loss computation on raw logits
    9. Probability normalization via Softmax (sum ≈ 1.0)
   10. Single-text inference & confidence scoring
   11. Batch inference consistency (N inputs -> N outputs)
   12. Saved model and tokenizer reload consistency
   13. Training loop sanity check (forward, loss, backward, gradients)
   14. Classification metrics & confusion matrix computation
   15. Error analysis & pairwise boundary confusion detection
"""

import json
from pathlib import Path
import sys
import tempfile
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent  # ml-service/
sys.path.insert(0, str(BASE_DIR))

from src.transformers.bert.tokenizer import BertTokenizerManager, DEFAULT_MAX_LENGTH
from src.transformers.bert.dataset import (
    SarcasmBertDataset,
    load_and_split_data,
    create_dataloaders,
    load_label_mapping,
)
from src.transformers.bert.model import BertSarcasmClassifier
from src.transformers.bert.evaluate import (
    calculate_classification_metrics,
    generate_predictions_dataframe,
    analyze_misclassifications,
    CLASS_NAMES,
)
from src.transformers.bert.predict import (
    BertPredictor,
    verify_model_reload_consistency,
)
from src.transformers.bert.experiments import run_model_sanity_check


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="session")
def tokenizer_manager():
    return BertTokenizerManager(model_name_or_path="bert-base-uncased", max_length=64)


@pytest.fixture(scope="session")
def bert_model():
    return BertSarcasmClassifier(model_name_or_path="bert-base-uncased", num_labels=4)


@pytest.fixture
def sample_texts():
    return [
        "What a wonderful rainy Monday for an outdoor wedding.",
        "A fire station burned down in the city center today.",
        "All the world is a stage and we are merely players.",
        "The train leaves at 5:30 PM from platform 2.",
    ]


@pytest.fixture
def sample_labels():
    return [1, 2, 3, 0]  # Sarcasm, Irony, Figurative, Regular


# ============================================================
# 1. TOKENIZER TESTS
# ============================================================

def test_bert_tokenizer_loading(tokenizer_manager):
    """Verifies tokenizer is properly loaded from bert-base-uncased with expected vocab."""
    assert tokenizer_manager is not None
    assert tokenizer_manager.vocab_size == 30522
    assert tokenizer_manager.cls_token_id == 101
    assert tokenizer_manager.sep_token_id == 102
    assert tokenizer_manager.pad_token_id == 0
    assert tokenizer_manager.unk_token_id == 100


def test_special_tokens_handling(tokenizer_manager):
    """Verifies [CLS] and [SEP] are automatically inserted at start and end."""
    encoding = tokenizer_manager.encode_text("Hello world", max_length=16, padding=False)
    input_ids = encoding["input_ids"][0].tolist()
    assert input_ids[0] == tokenizer_manager.cls_token_id
    assert input_ids[-1] == tokenizer_manager.sep_token_id


def test_subword_tokenization(tokenizer_manager):
    """Verifies WordPiece tokenization decomposes compound/slang words into subwords."""
    text = "sarcastically unbelievable sooooo"
    tokens = tokenizer_manager.tokenize_text(text)
    assert len(tokens) > 3  # WordPiece splits into multiple subwords
    assert any("##" in tok for tok in tokens)


def test_sequence_length_analysis(tokenizer_manager, sample_texts):
    """Verifies analyze_sequence_lengths calculates correct length statistics."""
    stats = tokenizer_manager.analyze_sequence_lengths(sample_texts, candidate_lengths=[16, 32, 64])
    assert stats["total_samples"] == 4
    assert stats["mean_length"] > 0
    assert stats["max_length"] >= stats["min_length"]
    assert "32" in stats["truncation_rates"]
    assert "percentage_preserved" in stats["truncation_rates"]["64"]


# ============================================================
# 2. DATASET & ENCODING TESTS
# ============================================================

def test_dataset_encoding_and_shapes(tokenizer_manager, sample_texts, sample_labels):
    """Verifies dataset item structure, shapes, and tensor datatypes."""
    dataset = SarcasmBertDataset(
        texts=sample_texts,
        labels=sample_labels,
        tokenizer_manager=tokenizer_manager,
        max_length=32,
    )
    assert len(dataset) == 4
    item = dataset[0]

    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item

    assert item["input_ids"].shape == (32,)
    assert item["attention_mask"].shape == (32,)
    assert item["labels"].dtype == torch.long
    assert item["labels"].item() == sample_labels[0]


def test_attention_mask_distinction(tokenizer_manager):
    """Verifies attention mask correctly contains 1s for tokens and 0s for padding."""
    short_text = "Short text."
    encoding = tokenizer_manager.encode_text(short_text, max_length=16, padding="max_length")
    mask = encoding["attention_mask"][0].tolist()
    ids = encoding["input_ids"][0].tolist()

    # Leading tokens (CLS, tokens, SEP) must have mask == 1
    # Trailing PAD tokens must have mask == 0
    real_token_count = ids.index(tokenizer_manager.sep_token_id) + 1
    assert all(m == 1 for m in mask[:real_token_count])
    assert all(m == 0 for m in mask[real_token_count:])


def test_label_mapping_preservation():
    """Verifies four-class project mapping is preserved."""
    mapping = load_label_mapping()
    assert mapping["Regular"] == 0
    assert mapping["Sarcasm"] == 1
    assert mapping["Irony"] == 2
    assert mapping["Figurative"] == 3


# ============================================================
# 3. MODEL ARCHITECTURE & FORWARD PASS TESTS
# ============================================================

def test_model_output_shape(bert_model, tokenizer_manager, sample_texts):
    """Verifies forward pass produces logits of shape [batch_size, 4]."""
    encoding = tokenizer_manager.encode_batch(sample_texts, max_length=32, padding="max_length")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    assert logits.shape == (4, 4), f"Expected shape (4, 4), got {logits.shape}"
    assert logits.shape[-1] == 4


def test_parameter_counts(bert_model):
    """Verifies total and trainable parameter counts."""
    counts = bert_model.count_parameters()
    assert counts["total_parameters"] > 100_000_000  # BERT-base is ~110M params
    assert counts["trainable_parameters"] > 0
    assert counts["total_parameters"] == counts["trainable_parameters"] + counts["non_trainable_parameters"]


def test_loss_computation(bert_model, tokenizer_manager, sample_texts, sample_labels):
    """Verifies CrossEntropyLoss computation on raw logits."""
    encoding = tokenizer_manager.encode_batch(sample_texts, max_length=32, padding="max_length")
    labels = torch.tensor(sample_labels, dtype=torch.long)

    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            labels=labels,
        )
        loss = outputs.loss

    assert loss is not None
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_probability_calculation(bert_model, tokenizer_manager, sample_texts):
    """Verifies Softmax probability outputs sum to 1.0 for all samples."""
    encoding = tokenizer_manager.encode_batch(sample_texts, max_length=32, padding="max_length")
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
        probs = bert_model.predict_probabilities(outputs.logits)

    sums = torch.sum(probs, dim=-1).cpu().numpy()
    np.testing.assert_allclose(sums, np.ones(len(sample_texts)), atol=1e-5)


# ============================================================
# 4. INFERENCE & RELOAD TESTS
# ============================================================

def test_single_text_inference(bert_model, tokenizer_manager):
    """Verifies single-string inference output structure."""
    predictor = BertPredictor(model=bert_model, tokenizer_manager=tokenizer_manager)
    res = predictor.predict_single("Oh great, another sunny day at the beach.")

    assert "text" in res
    assert "predicted_class" in res
    assert res["predicted_class"] in CLASS_NAMES
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res
    assert len(res["probabilities"]) == 4


def test_batch_inference_consistency(bert_model, tokenizer_manager, sample_texts):
    """Verifies that N inputs produce exactly N predictions."""
    predictor = BertPredictor(model=bert_model, tokenizer_manager=tokenizer_manager)
    batch_res = predictor.predict_batch(sample_texts, batch_size=2)

    assert len(batch_res) == len(sample_texts)
    for item in batch_res:
        assert item["predicted_class"] in CLASS_NAMES
        assert 0.0 <= item["confidence"] <= 1.0


def test_model_and_tokenizer_reload(bert_model, tokenizer_manager, sample_texts):
    """Verifies model and tokenizer serialization and reloading produce identical outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        success = verify_model_reload_consistency(
            model=bert_model,
            tokenizer_manager=tokenizer_manager,
            temp_dir=tmp_dir,
            sample_texts=sample_texts,
        )
        assert success is True


def test_model_sanity_check(bert_model, tokenizer_manager):
    """Verifies the Section 74 model sanity check function passes."""
    device = torch.device("cpu")
    passed = run_model_sanity_check(bert_model, tokenizer_manager, device)
    assert passed is True


# ============================================================
# 5. METRICS & ERROR ANALYSIS TESTS
# ============================================================

def test_classification_metrics_computation():
    """Verifies calculation of accuracy, macro precision, recall, and Macro F1."""
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 3, 0, 2, 2, 3]  # one mistake: 1 predicted as 2

    metrics = calculate_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics

    assert metrics["accuracy"] == 7 / 8
    assert len(metrics["confusion_matrix"]) == 4
    assert len(metrics["confusion_matrix"][0]) == 4


def test_error_analysis_breakdown():
    """Verifies misclassification analysis correctly tallies confusion pairs."""
    texts = ["t1", "t2", "t3", "t4"]
    y_true = [1, 2, 0, 3]
    y_pred = [2, 1, 0, 3]  # Sarcasm -> Irony, Irony -> Sarcasm
    probs = np.array([
        [0.1, 0.3, 0.5, 0.1],
        [0.1, 0.6, 0.2, 0.1],
        [0.8, 0.1, 0.05, 0.05],
        [0.05, 0.05, 0.1, 0.8],
    ])

    df = generate_predictions_dataframe(texts, y_true, y_pred, probs, CLASS_NAMES)
    analysis = analyze_misclassifications(df, CLASS_NAMES)

    assert analysis["total_samples"] == 4
    assert analysis["total_errors"] == 2
    assert analysis["sarcasm_as_irony"] == 1
    assert analysis["irony_as_sarcasm"] == 1
    assert analysis["sarcasm_irony_mutual_confusion"] == 2
