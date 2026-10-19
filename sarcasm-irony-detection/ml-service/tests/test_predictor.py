"""
Regression and functional tests for SarcasmIronyPredictor (Phase 10).
"""

import json
import os
import pytest
import numpy as np

from src.inference import (
    SarcasmIronyPredictor,
    predict_text,
    predict_batch,
    PredictionResult,
    BatchPredictionResult,
)


@pytest.fixture(scope="module")
def predictor():
    """Initializes and provides the production predictor instance."""
    return SarcasmIronyPredictor()


@pytest.fixture(scope="module")
def golden_cases():
    """Loads golden test cases from tests/data/golden_cases.json."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    golden_path = os.path.join(base_dir, "data", "golden_cases.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_predictor_initialization(predictor):
    """Test that predictor initializes properly and loads all artifacts."""
    assert predictor is not None
    assert predictor.bundle is not None
    assert predictor.bundle.model is not None
    assert predictor.bundle.vectorizer is not None
    assert len(predictor.classes) == 4
    assert set(predictor.classes) == {"Regular", "Sarcasm", "Irony", "Figurative"}


def test_single_prediction_schema(predictor):
    """Test that single prediction returns a valid PredictionResult object."""
    sample = "The weather is sunny and clear today in San Francisco."
    res = predictor.predict(sample)

    assert isinstance(res, PredictionResult)
    assert res.text == sample
    assert res.predicted_class in {"Regular", "Sarcasm", "Irony", "Figurative"}
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.probabilities) == 4
    assert res.model_name == "SarcasmIronyClassifier-LinearSVM"
    assert res.model_version == "SIC-v1.0"
    assert res.latency_ms is not None
    assert res.latency_ms > 0.0


def test_probability_distribution_properties(predictor):
    """Test that probabilities sum to 1.0 and each is within [0, 1]."""
    test_inputs = [
        "What a wonderful, pleasant flight with zero turbulence.",
        "Oh great, another flat tire right on my way to the interview! #sarcasm",
        "A towing company truck being towed away on the highway.",
        "Her voice was like sweet honey dripping in summer.",
    ]
    for text in test_inputs:
        res = predictor.predict(text)
        prob_sum = sum(res.probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-3, f"Sum {prob_sum} deviates from 1.0 for '{text}'"
        for cls_name, p in res.probabilities.items():
            assert 0.0 <= p <= 1.0, f"Probability {p} for {cls_name} out of bounds"


def test_confidence_matches_maximum_probability(predictor):
    """Test that confidence equals the maximum class probability."""
    res = predictor.predict("The meeting starts at 10:00 AM sharp.")
    max_prob = max(res.probabilities.values())
    assert abs(res.confidence - max_prob) < 1e-4


def test_batch_prediction_order_and_count(predictor):
    """Test that batch prediction preserves exact input count and ordering."""
    samples = [
        "Literal report of financial results.",
        "Just love waiting 45 minutes for a cold coffee.",
        "A pilot having a fear of heights.",
        "Time is a thief that steals our days.",
    ]
    batch_res = predictor.predict_batch(samples)

    assert isinstance(batch_res, BatchPredictionResult)
    assert batch_res.total_count == len(samples)
    assert len(batch_res.predictions) == len(samples)

    for i, (orig, pred) in enumerate(zip(samples, batch_res.predictions)):
        assert pred.text == orig
        single_res = predictor.predict(orig)
        assert pred.predicted_class == single_res.predicted_class
        for cls_name in predictor.classes:
            assert abs(pred.probabilities[cls_name] - single_res.probabilities[cls_name]) < 1e-4


def test_model_reload(predictor):
    """Test that predictor can reload from disk and continue functioning."""
    sample = "Everything is proceeding as scheduled."
    res_before = predictor.predict(sample)

    predictor.reload()

    res_after = predictor.predict(sample)
    assert res_before.predicted_class == res_after.predicted_class
    assert abs(res_before.confidence - res_after.confidence) < 1e-4


def test_convenience_functions():
    """Test module-level convenience functions predict_text and predict_batch."""
    text = "The bus arrives every fifteen minutes."
    single_dict = predict_text(text)
    assert isinstance(single_dict, dict)
    assert single_dict["predicted_class"] == "Regular"
    assert "probabilities" in single_dict

    batch_dict = predict_batch([text, "Oh brilliant, it is raining inside my room."])
    assert isinstance(batch_dict, dict)
    assert batch_dict["total_count"] == 2
    assert len(batch_dict["predictions"]) == 2


def test_golden_cases(predictor, golden_cases):
    """Test that all golden cases execute without crashing and stable cases match."""
    for case in golden_cases:
        res = predictor.predict(case["text"])
        assert res.predicted_class in {"Regular", "Sarcasm", "Irony", "Figurative"}
        if case.get("expected_class") and case["expected_class"] != "Not predefined":
            assert res.predicted_class == case["expected_class"], (
                f"Golden case {case['id']} expected {case['expected_class']} but got {res.predicted_class}"
            )
