"""
Core Inference Predictor for Sarcasm & Irony Detection.
Phase 10 — Production-Ready Service Implementation.
"""

import time
from typing import Dict, List, Union, Any, Optional
import numpy as np
import scipy.special

from .exceptions import InferenceError, InvalidInputError
from .loader import ModelLoader, ModelArtifactBundle
from .preprocessing import InferenceTextPreprocessor
from .schemas import PredictionResult, BatchPredictionResult


class SarcasmIronyPredictor:
    """
    Production-ready predictor implementing single and batch inference for the
    Phase 9 final selected model (LinearSVC One-vs-Rest on Bag of Words).
    """

    ALLOWED_CLASSES = {"Regular", "Sarcasm", "Irony", "Figurative"}

    def __init__(self, model_dir: Optional[str] = None, verify_checksums: bool = False):
        self.model_dir = model_dir
        self.verify_checksums = verify_checksums
        self.bundle: ModelArtifactBundle = ModelLoader.load(
            model_dir=model_dir, verify_checksums=verify_checksums
        )
        max_chars = self.bundle.preprocessing_config.get("input_validation", {}).get(
            "max_length_chars", 5000
        )
        self.preprocessor = InferenceTextPreprocessor(max_length_chars=max_chars)
        self.classes = self.bundle.classes
        self.model_name = self.bundle.metadata.get("model_name", "SarcasmIronyClassifier-LinearSVM")
        self.model_version = self.bundle.metadata.get("model_version", "SIC-v1.0")

    def predict(self, text: str) -> PredictionResult:
        """
        Executes inference for a single input text string.

        Args:
            text: Raw input string to classify.

        Returns:
            PredictionResult: Typed schema containing predicted class, confidence,
                              calibrated probabilities, and processing latency.

        Raises:
            InvalidInputError: If input is None, not a string, or empty.
            InferenceError: If prediction sanity checks fail.
        """
        start_time = time.perf_counter()

        # Step 1: Input validation and preprocessing
        clean_text = self.preprocessor.validate_and_clean(text)

        # Step 2: Vectorization (transform only, no fitting)
        X = self.bundle.vectorizer.transform([clean_text])

        # Step 3: Model decision scores
        decision_scores = self.bundle.model.decision_function(X)[0]

        # Step 4: Softmax probability conversion
        probs = scipy.special.softmax(decision_scores)

        # Step 5: Probability & Class Sanity Validation
        prob_sum = float(np.sum(probs))
        if abs(prob_sum - 1.0) > 1e-3:
            raise InferenceError(f"Probability distribution sum invalid: {prob_sum}")

        best_idx = int(np.argmax(probs))
        predicted_class = self.classes[best_idx]
        if predicted_class not in self.ALLOWED_CLASSES:
            raise InferenceError(f"Unknown predicted class: '{predicted_class}'")

        confidence = float(probs[best_idx])
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        prob_dict = {
            c: round(float(p), 4) for c, p in zip(self.classes, probs)
        }

        return PredictionResult(
            text=text,
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            model_name=self.model_name,
            model_version=self.model_version,
            latency_ms=latency_ms,
        )

    def predict_batch(self, texts: List[str]) -> BatchPredictionResult:
        """
        Executes vectorized batch inference for a list of input texts.

        Args:
            texts: List of raw input strings.

        Returns:
            BatchPredictionResult: Typed schema containing list of PredictionResults
                                  in exact order and total batch latency.

        Raises:
            InvalidInputError: If texts is not a list, is empty, or contains invalid items.
        """
        start_time = time.perf_counter()

        # Step 1: Batch validation and cleaning
        clean_texts = self.preprocessor.validate_and_clean_batch(texts)

        # Step 2: Batch vectorization
        X = self.bundle.vectorizer.transform(clean_texts)

        # Step 3: Batch decision scores
        decision_scores = self.bundle.model.decision_function(X)

        # Step 4: Batch softmax probabilities
        probs_all = scipy.special.softmax(decision_scores, axis=1)

        # Step 5: Format ordered results
        predictions = []
        for orig_text, probs in zip(texts, probs_all):
            best_idx = int(np.argmax(probs))
            predicted_class = self.classes[best_idx]
            if predicted_class not in self.ALLOWED_CLASSES:
                raise InferenceError(f"Unknown predicted class in batch: '{predicted_class}'")

            prob_dict = {c: round(float(p), 4) for c, p in zip(self.classes, probs)}
            predictions.append(
                PredictionResult(
                    text=orig_text,
                    predicted_class=predicted_class,
                    confidence=round(float(probs[best_idx]), 4),
                    probabilities=prob_dict,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    latency_ms=None,
                )
            )

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return BatchPredictionResult(
            predictions=predictions,
            total_count=len(predictions),
            batch_latency_ms=total_latency_ms,
            model_name=self.model_name,
            model_version=self.model_version,
        )

    def reload(self):
        """Forces an in-memory artifact unload and fresh reload from disk."""
        ModelLoader.clear_cache()
        self.bundle = ModelLoader.load(
            model_dir=self.model_dir,
            verify_checksums=self.verify_checksums,
            force_reload=True,
        )
        self.classes = self.bundle.classes

    def get_metadata(self) -> Dict[str, Any]:
        """Returns loaded model metadata."""
        return self.bundle.metadata

    def get_classes(self) -> List[str]:
        """Returns list of target classes in established project order."""
        return list(self.classes)
