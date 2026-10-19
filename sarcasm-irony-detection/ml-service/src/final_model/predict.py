"""
Final Model Inference Service for Sarcasm & Irony Detection.
Phase 9 / Phase 10 API-ready inference module.
"""

import json
import os
import re
from typing import Dict, List, Union, Any
import joblib
import numpy as np
import scipy.special


class SarcasmIronyPredictor:
    """
    Production-ready inference interface for the selected Final Model.
    Loads model, vectorizer, configuration, and label mappings from disk.
    """

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            # Default to models/final relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_dir = os.path.join(base_dir, "models", "final")

        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "model", "model.joblib")
        self.vectorizer_path = os.path.join(model_dir, "tokenizer", "vectorizer.joblib")
        self.config_path = os.path.join(model_dir, "config.json")
        self.label_map_path = os.path.join(model_dir, "label_mapping.json")

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model, vectorizer, and mappings from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer artifact not found at {self.vectorizer_path}")

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

        with open(self.label_map_path, "r", encoding="utf-8") as f:
            self.label_to_id = json.load(f)
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.classes = [self.id_to_label[i] for i in range(len(self.id_to_label))]

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def preprocess(self, text: str) -> str:
        """
        Lightweight classical preprocessing consistent with Phase 3/4 pipeline:
        - URL normalization
        - Mention normalization
        - Lowercasing
        - Whitespace stripping
        """
        if not isinstance(text, str):
            return ""
        t = re.sub(r"https?://\S+|www\.\S+", " url ", text)
        t = re.sub(r"@\w+", " user_mention ", t)
        t = t.lower().strip()
        return t

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Runs inference on a single text string.

        Returns:
            Dict containing:
                - text: original text
                - predicted_class: string label (Regular, Sarcasm, Irony, Figurative)
                - probabilities: dict of class -> normalized probability
                - confidence: highest probability float
        """
        clean_text = self.preprocess(text)
        if not clean_text:
            # Handle empty / whitespace input gracefully
            return {
                "text": text,
                "predicted_class": "Regular",
                "probabilities": {c: 0.25 for c in self.classes},
                "confidence": 0.25,
                "note": "Empty or non-text input defaulted to Regular with uniform uncertainty.",
            }

        X = self.vectorizer.transform([clean_text])
        decision_scores = self.model.decision_function(X)[0]
        probs = scipy.special.softmax(decision_scores)
        best_idx = int(np.argmax(probs))
        predicted_class = self.classes[best_idx]
        confidence = float(probs[best_idx])

        return {
            "text": text,
            "predicted_class": predicted_class,
            "probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
            "confidence": round(confidence, 4),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Runs vectorized inference on a batch of text strings.
        """
        if not texts:
            return []

        clean_texts = [self.preprocess(t) for t in texts]
        X = self.vectorizer.transform(clean_texts)
        decision_scores = self.model.decision_function(X)
        probs_all = scipy.special.softmax(decision_scores, axis=1)

        results = []
        for orig_text, probs in zip(texts, probs_all):
            best_idx = int(np.argmax(probs))
            results.append({
                "text": orig_text,
                "predicted_class": self.classes[best_idx],
                "probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, probs)},
                "confidence": round(float(probs[best_idx]), 4),
            })
        return results


# Module-level convenience function
_predictor_instance = None

def get_predictor() -> SarcasmIronyPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SarcasmIronyPredictor()
    return _predictor_instance

def predict(text: str) -> Dict[str, Any]:
    return get_predictor().predict(text)

def predict_batch(texts: List[str]) -> List[Dict[str, Any]]:
    return get_predictor().predict_batch(texts)


if __name__ == "__main__":
    print("Testing SarcasmIronyPredictor standalone...")
    predictor = SarcasmIronyPredictor()

    test_samples = [
        "The weather is sunny and clear today in San Francisco.",
        "Oh wonderful, another flat tire on Monday morning! Just what I needed. #sarcasm",
        "A fire station burning down while the firefighters are out on call.",
        "I have told you a million times, her smile was like sunshine in the dead of winter.",
        "",
        "   ",
        "a" * 1000,  # Very long text
    ]

    print("\n--- SINGLE PREDICTIONS ---")
    for s in test_samples[:4]:
        res = predictor.predict(s)
        print(f"\nText: {res['text']}")
        print(f"Prediction: {res['predicted_class']} (Confidence: {res['confidence']:.2%})")
        print(f"Probabilities: {res['probabilities']}")

    print("\n--- EDGE CASES & BATCH PREDICTION ---")
    batch_res = predictor.predict_batch(test_samples)
    print(f"Batch processed {len(batch_res)} items successfully.")
    for i, r in enumerate(batch_res):
        t_preview = (r['text'][:40] + '...') if len(r['text']) > 40 else r['text']
        print(f"[{i+1}] '{t_preview}' -> {r['predicted_class']} ({r['confidence']:.2%})")
