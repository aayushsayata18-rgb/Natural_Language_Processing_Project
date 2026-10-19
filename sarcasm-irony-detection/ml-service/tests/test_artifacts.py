"""
Artifact validation and integrity tests for Phase 10 packaging.
"""

import json
import os
import joblib
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

from src.inference.utils import verify_artifact_checksums
from src.inference.loader import ModelLoader


@pytest.fixture(scope="module")
def model_dir():
    return ModelLoader.resolve_model_dir()


def test_required_artifacts_exist(model_dir):
    """Test that all required files and subdirectories exist in models/final/."""
    expected_files = [
        os.path.join("model", "model.joblib"),
        os.path.join("tokenizer", "vectorizer.joblib"),
        "config.json",
        "label_mapping.json",
        "preprocessing_config.json",
        "model_metadata.json",
        "manifest.json",
        "checksums.json",
        "environment.json",
        "README.md",
        "CHANGELOG.md",
    ]
    for rel_path in expected_files:
        full_path = os.path.join(model_dir, rel_path)
        assert os.path.exists(full_path), f"Artifact missing: {rel_path} ({full_path})"


def test_checksum_integrity_verification(model_dir):
    """Test that SHA-256 hashes of all artifacts match checksums.json."""
    ok, results = verify_artifact_checksums(model_dir)
    assert ok is True
    assert len(results) >= 6
    for rel_path, res in results.items():
        assert res["status"] == "verified"
        assert len(res["sha256"]) == 64


def test_model_metadata_structure(model_dir):
    """Test that model_metadata.json contains all required fields and valid metrics."""
    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_version"] == "SIC-v1.0"
    assert meta["model_name"] == "SarcasmIronyClassifier-LinearSVM"
    assert meta["num_classes"] == 4
    assert meta["classes"] == ["Regular", "Sarcasm", "Irony", "Figurative"]
    assert meta["macro_f1"] == 0.7187
    assert meta["accuracy"] == 0.8457
    assert "selected_reason" in meta


def test_label_mapping_integrity(model_dir):
    """Test that label_mapping.json matches the project's standard 4 classes."""
    lbl_path = os.path.join(model_dir, "label_mapping.json")
    with open(lbl_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    assert mapping["0"] == "Regular"
    assert mapping["1"] == "Sarcasm"
    assert mapping["2"] == "Irony"
    assert mapping["3"] == "Figurative"


def test_vectorizer_artifact_properties(model_dir):
    """Test that the serialized vectorizer is an intact CountVectorizer."""
    vec_path = os.path.join(model_dir, "tokenizer", "vectorizer.joblib")
    vec = joblib.load(vec_path)

    assert isinstance(vec, CountVectorizer)
    assert hasattr(vec, "vocabulary_")
    assert len(vec.vocabulary_) == 28077


def test_model_artifact_properties(model_dir):
    """Test that the serialized model is an intact LinearSVC classifier."""
    model_path = os.path.join(model_dir, "model", "model.joblib")
    model = joblib.load(model_path)

    assert isinstance(model, LinearSVC)
    assert hasattr(model, "coef_")
    assert model.coef_.shape[0] == 4  # 4 classes
    assert model.coef_.shape[1] == 28077  # 28,077 features
