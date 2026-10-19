"""
Phase 8 — BERT Inference & Prediction Engine
============================================
Sarcasm & Irony Detection System using NLP

Provides:
- Single text inference with probability distribution & confidence
- Batch inference pipeline for N text inputs
- Reload verification to ensure saved model & tokenizer produce identical outputs
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from src.transformers.bert.model import BertSarcasmClassifier
from src.transformers.bert.tokenizer import BertTokenizerManager

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Regular", "Sarcasm", "Irony", "Figurative"]
ID_TO_LABEL = {0: "Regular", 1: "Sarcasm", 2: "Irony", 3: "Figurative"}


class BertPredictor:
    """
    High-level inference engine for four-class sarcasm and irony detection.
    """

    def __init__(
        self,
        model: BertSarcasmClassifier,
        tokenizer_manager: BertTokenizerManager,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer_manager = tokenizer_manager
        self.class_names = class_names or CLASS_NAMES

    @classmethod
    def load_from_directory(
        cls,
        model_dir: Union[str, Path],
        device: Optional[torch.device] = None,
        max_length: int = 80,
    ) -> "BertPredictor":
        """
        Loads saved model and tokenizer from an artifact directory.
        """
        p = Path(model_dir)
        tokenizer_path = p / "tokenizer" if (p / "tokenizer").exists() else p
        model_path = p / "model" if (p / "model").exists() else p

        tokenizer_mgr = BertTokenizerManager.load_saved_tokenizer(tokenizer_path, max_length=max_length)
        model = BertSarcasmClassifier.from_pretrained(model_path, num_labels=len(CLASS_NAMES))
        return cls(model=model, tokenizer_manager=tokenizer_mgr, device=device)

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predicts the class and probability distribution for a single text string.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        encoding = self.tokenizer_manager.encode_text(
            text,
            max_length=self.tokenizer_manager.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            probs = self.model.predict_probabilities(logits).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        pred_class = self.class_names[pred_id]
        confidence = float(probs[pred_id])

        prob_dict = {self.class_names[i]: round(float(probs[i]), 4) for i in range(len(self.class_names))}

        return {
            "text": text,
            "predicted_label_id": pred_id,
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
        }

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Predicts classes and probabilities for a batch of text strings.
        Guarantees that N inputs produce exactly N prediction outputs.
        """
        results: List[Dict[str, Any]] = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            encoding = self.tokenizer_manager.encode_batch(
                chunk,
                max_length=self.tokenizer_manager.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            token_type_ids = encoding.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs.logits
                probs = self.model.predict_probabilities(logits).cpu().numpy()

            for text, p in zip(chunk, probs):
                pred_id = int(np.argmax(p))
                pred_class = self.class_names[pred_id]
                conf = float(p[pred_id])
                prob_dict = {self.class_names[j]: round(float(p[j]), 4) for j in range(len(self.class_names))}

                results.append({
                    "text": text,
                    "predicted_label_id": pred_id,
                    "predicted_class": pred_class,
                    "confidence": round(conf, 4),
                    "probabilities": prob_dict,
                })

        return results


def verify_model_reload_consistency(
    model: BertSarcasmClassifier,
    tokenizer_manager: BertTokenizerManager,
    temp_dir: Union[str, Path],
    sample_texts: Optional[List[str]] = None,
) -> bool:
    """
    Saves model and tokenizer to temp_dir, reloads them, and asserts identical predictions.
    """
    if sample_texts is None:
        sample_texts = [
            "What a lovely rainy day for a picnic.",
            "I absolutely love sitting in traffic for two hours.",
            "The fire station burned down yesterday.",
            "Life is a highway and I want to ride it all night long.",
        ]

    p = Path(temp_dir)
    model_dir = p / "model"
    tok_dir = p / "tokenizer"

    # Save
    model.save_pretrained(model_dir)
    tokenizer_manager.save_tokenizer(tok_dir)

    # Reload
    reloaded_tok = BertTokenizerManager.load_saved_tokenizer(tok_dir, max_length=tokenizer_manager.max_length)
    reloaded_model = BertSarcasmClassifier.from_pretrained(model_dir, num_labels=model.num_labels)

    pred_original = BertPredictor(model, tokenizer_manager)
    pred_reloaded = BertPredictor(reloaded_model, reloaded_tok)

    orig_out = pred_original.predict_batch(sample_texts)
    reloaded_out = pred_reloaded.predict_batch(sample_texts)

    for o, r in zip(orig_out, reloaded_out):
        if o["predicted_label_id"] != r["predicted_label_id"]:
            logger.error(f"Prediction label mismatch on text '{o['text']}': orig={o['predicted_label_id']}, reload={r['predicted_label_id']}")
            return False
        for k in o["probabilities"]:
            diff = abs(o["probabilities"][k] - r["probabilities"][k])
            if diff > 1e-4:
                logger.error(f"Probability mismatch on '{k}' ({diff:.6f})")
                return False

    logger.info("Model reload consistency test PASSED with 0 discrepancies!")
    return True
