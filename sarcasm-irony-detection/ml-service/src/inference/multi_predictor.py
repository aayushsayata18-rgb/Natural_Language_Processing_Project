"""
Multi-Model Inference Engine for Sarcasm & Irony Detection.
Supports comparative inference across 5 NLP paradigms:
1. BoW (Phase 4): LinearSVM (BOW-16) [Champion Model]
2. TF-IDF (Phase 5): LinearSVM (TFIDF-06)
3. GloVe (Phase 6): Bi-LSTM (BILSTM-02)
4. FastText (Phase 7): Bi-LSTM + Attention (FT-02)
5. BERT (Phase 8): bert-base-uncased Contextual Transformer
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import scipy.special
import torch

from .exceptions import InferenceError, InvalidInputError, ModelNotFoundError
from .preprocessing import InferenceTextPreprocessor
from .schemas import BatchPredictionResult, PredictionResult

logger = logging.getLogger("ml-service.multi_predictor")

CLASSES = ["Regular", "Sarcasm", "Irony", "Figurative"]

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Metadata specifications for all 5 benchmark models
MODEL_METADATA_REGISTRY: Dict[str, Dict[str, Any]] = {
    "bow_linearsvm": {
        "id": "bow_linearsvm",
        "name": "LinearSVM (BoW)",
        "full_name": "Linear Support Vector Classifier with Bag of Words (One-vs-Rest)",
        "phase": "Phase 4",
        "paradigm": "Bag of Words (n-grams 1-2)",
        "parameter_count": 112312,
        "val_macro_f1": 0.7187,
        "val_accuracy": 0.8457,
        "latency_ms": 0.027,
        "is_champion": True,
        "description": "Selected champion model. Achieves top Macro F1 (0.7187), avoids minority class collapse, and executes ultra-low latency inference.",
    },
    "tfidf_linearsvm": {
        "id": "tfidf_linearsvm",
        "name": "LinearSVM (TF-IDF)",
        "full_name": "Linear Support Vector Classifier with Sublinear TF-IDF",
        "phase": "Phase 5",
        "paradigm": "TF-IDF + Statistical N-grams",
        "parameter_count": 112312,
        "val_macro_f1": 0.7002,
        "val_accuracy": 0.8565,
        "latency_ms": 0.014,
        "is_champion": False,
        "description": "High accuracy classical model leveraging inverse document frequency weighting across 28,077 unigram/bigram tokens.",
    },
    "glove_bilstm": {
        "id": "glove_bilstm",
        "name": "GloVe + Bi-LSTM",
        "full_name": "Bidirectional LSTM with 100d Pretrained GloVe Embeddings (BILSTM-02)",
        "phase": "Phase 6",
        "paradigm": "Dense Word Embeddings (Deep Learning)",
        "parameter_count": 3197636,
        "val_macro_f1": 0.6940,
        "val_accuracy": 0.8612,
        "latency_ms": 0.041,
        "is_champion": False,
        "description": "Recurrent neural sequence model capturing bidirectional token contexts using 100-dimensional GloVe word vectors.",
    },
    "fasttext_attention": {
        "id": "fasttext_attention",
        "name": "FastText + Bi-LSTM + Attention",
        "full_name": "FastText 300d + Bi-LSTM + Feed-Forward Attention Mechanism (FT-02)",
        "phase": "Phase 7",
        "paradigm": "Subword Embeddings & Feed-Forward Attention",
        "parameter_count": 9204828,
        "val_macro_f1": 0.6928,
        "val_accuracy": 0.8603,
        "latency_ms": 0.098,
        "is_champion": False,
        "description": "Resolves social media OOV slang via character n-grams and weights salient contrastive tokens through additive attention.",
    },
    "bert_transformer": {
        "id": "bert_transformer",
        "name": "BERT Contextual",
        "full_name": "bert-base-uncased (12-layer Bidirectional Transformer)",
        "phase": "Phase 8",
        "paradigm": "Pretrained Contextual Transformer (WordPiece)",
        "parameter_count": 109485316,
        "val_macro_f1": 0.6935,
        "val_accuracy": 0.8612,
        "latency_ms": 39.33,
        "is_champion": False,
        "description": "Deep contextual transformer fine-tuned end-to-end; completely eliminates mutual confusion between Sarcasm and Irony.",
    },
}

# Alias mapping for flexible model identification
MODEL_ALIASES: Dict[str, str] = {
    "bow": "bow_linearsvm",
    "bow_linearsvm": "bow_linearsvm",
    "linearsvm": "bow_linearsvm",
    "default": "bow_linearsvm",
    "phase4": "bow_linearsvm",
    "tfidf": "tfidf_linearsvm",
    "tfidf_linearsvm": "tfidf_linearsvm",
    "phase5": "tfidf_linearsvm",
    "bilstm": "glove_bilstm",
    "glove": "glove_bilstm",
    "glove_bilstm": "glove_bilstm",
    "phase6": "glove_bilstm",
    "fasttext": "fasttext_attention",
    "fasttext_attention": "fasttext_attention",
    "attention": "fasttext_attention",
    "phase7": "fasttext_attention",
    "bert": "bert_transformer",
    "bert_transformer": "bert_transformer",
    "transformer": "bert_transformer",
    "phase8": "bert_transformer",
}


class MultiModelPredictor:
    """
    Unified multi-model manager hosting and serving inference across all 5 NLP paradigms.
    Models are loaded and cached in memory with fast fallback.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None, preload_all: bool = False):
        self.base_dir = Path(base_dir) if base_dir else BASE_DIR
        self.preprocessor = InferenceTextPreprocessor(max_length_chars=5000)
        self.classes = CLASSES
        self.device = torch.device("cpu")

        # Cache of initialized model components
        self._loaded_models: Dict[str, Any] = {}

        if preload_all:
            self.preload_all_models()

    def normalize_model_id(self, model_id: Optional[str]) -> str:
        """Resolves raw model identifier or alias into canonical model key."""
        if not model_id:
            return "bow_linearsvm"
        key = str(model_id).strip().lower()
        if key in MODEL_ALIASES:
            return MODEL_ALIASES[key]
        if key in MODEL_METADATA_REGISTRY:
            return key
        raise ModelNotFoundError(
            f"Unknown model identifier '{model_id}'. Available models: {list(MODEL_METADATA_REGISTRY.keys())}"
        )

    def get_model_metadata(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns metadata for specified model or all models if None."""
        if model_id is not None:
            canonical_id = self.normalize_model_id(model_id)
            return MODEL_METADATA_REGISTRY[canonical_id]
        return MODEL_METADATA_REGISTRY

    def list_available_models(self) -> List[Dict[str, Any]]:
        """Returns structured list of all 5 available models."""
        return list(MODEL_METADATA_REGISTRY.values())

    # -------------------------------------------------------------------------
    # Model Loading Helpers
    # -------------------------------------------------------------------------

    def _get_bow_model(self) -> Tuple[Any, Any]:
        """Loads and returns (model, vectorizer) for BoW LinearSVM."""
        if "bow_linearsvm" not in self._loaded_models:
            bow_dir = self.base_dir / "models" / "classical" / "bow_linearsvm"
            if not bow_dir.exists():
                bow_dir = self.base_dir / "models" / "final" / "model"
            m = joblib.load(bow_dir / "model.joblib")
            v = joblib.load(bow_dir / "vectorizer.joblib")
            self._loaded_models["bow_linearsvm"] = (m, v)
            logger.info("Loaded BoW LinearSVM model.")
        return self._loaded_models["bow_linearsvm"]

    def _get_tfidf_model(self) -> Tuple[Any, Any]:
        """Loads and returns (model, vectorizer) for TF-IDF LinearSVM."""
        if "tfidf_linearsvm" not in self._loaded_models:
            tfidf_dir = self.base_dir / "models" / "classical" / "tfidf" / "text_only_linearsvm"
            m = joblib.load(tfidf_dir / "model.joblib")
            v = joblib.load(tfidf_dir / "vectorizer.joblib")
            self._loaded_models["tfidf_linearsvm"] = (m, v)
            logger.info("Loaded TF-IDF LinearSVM model.")
        return self._loaded_models["tfidf_linearsvm"]

    def _get_glove_bilstm_model(self) -> Tuple[Any, Any]:
        """Loads and returns (model, tokenizer) for GloVe Bi-LSTM."""
        if "glove_bilstm" not in self._loaded_models:
            from ..deep_learning.bilstm.model import BiLSTMClassifier
            from ..deep_learning.bilstm.tokenizer import Tokenizer

            bilstm_dir = self.base_dir / "models" / "deep_learning" / "bilstm" / "best"
            tok = Tokenizer.load(str(bilstm_dir / "tokenizer.json"))
            model = BiLSTMClassifier(
                vocab_size=30002,
                embedding_dim=100,
                lstm_units=64,
                dense_units=64,
                num_classes=4,
                dropout=0.3,
            )
            state_dict = torch.load(bilstm_dir / "best_model.pt", map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self._loaded_models["glove_bilstm"] = (model, tok)
            logger.info("Loaded GloVe Bi-LSTM model.")
        return self._loaded_models["glove_bilstm"]

    def _get_fasttext_attention_model(self) -> Tuple[Any, Any]:
        """Loads and returns (model, tokenizer) for FastText Bi-LSTM + Attention."""
        if "fasttext_attention" not in self._loaded_models:
            from ..deep_learning.bilstm.tokenizer import Tokenizer
            from ..deep_learning.fasttext_attention.model import FastTextAttentionClassifier

            ft_dir = self.base_dir / "models" / "deep_learning" / "fasttext"
            tok_path = ft_dir / "tokenizer.json"
            if not tok_path.exists():
                tok_path = self.base_dir / "models" / "deep_learning" / "bilstm" / "best" / "tokenizer.json"
            tok = Tokenizer.load(str(tok_path))

            model = FastTextAttentionClassifier(
                vocab_size=30002,
                embedding_dim=300,
                lstm_units=64,
                attention_dim=64,
                dense_units=64,
                num_classes=4,
                dropout=0.3,
            )
            ckpt_path = ft_dir / "best_fasttext_attention.pt"
            if not ckpt_path.exists():
                ckpt_path = ft_dir / "ft-02" / "best_ft-02.pt"
            state_dict = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self._loaded_models["fasttext_attention"] = (model, tok)
            logger.info("Loaded FastText + Bi-LSTM + Attention model.")
        return self._loaded_models["fasttext_attention"]

    def _get_bert_model(self) -> Tuple[Any, Any]:
        """Loads and returns (model, tokenizer_manager) for fine-tuned BERT."""
        if "bert_transformer" not in self._loaded_models:
            from ..transformers.bert.model import BertSarcasmClassifier
            from ..transformers.bert.tokenizer import BertTokenizerManager

            bert_dir = self.base_dir / "models" / "transformers" / "bert"
            tok_mgr = BertTokenizerManager.load_saved_tokenizer(
                str(bert_dir / "tokenizer"), max_length=80
            )
            model = BertSarcasmClassifier.from_pretrained(
                str(bert_dir / "model"), num_labels=4
            )
            model.to(self.device)
            model.eval()
            self._loaded_models["bert_transformer"] = (model, tok_mgr)
            logger.info("Loaded BERT Transformer model.")
        return self._loaded_models["bert_transformer"]

    def preload_all_models(self) -> None:
        """Pre-warms and caches all 5 models in memory."""
        logger.info("Preloading all 5 models into memory...")
        self._get_bow_model()
        self._get_tfidf_model()
        self._get_glove_bilstm_model()
        self._get_fasttext_attention_model()
        self._get_bert_model()
        logger.info("All 5 models preloaded successfully.")

    # -------------------------------------------------------------------------
    # Inference Implementation
    # -------------------------------------------------------------------------

    def predict(self, text: str, model_id: Optional[str] = "bow_linearsvm") -> PredictionResult:
        """
        Runs single text inference using the specified model.
        """
        canonical_id = self.normalize_model_id(model_id)
        clean_text = self.preprocessor.validate_and_clean(text)
        meta = MODEL_METADATA_REGISTRY[canonical_id]

        start_time = time.perf_counter()

        if canonical_id == "bow_linearsvm":
            model, vec = self._get_bow_model()
            X = vec.transform([clean_text])
            scores = model.decision_function(X)[0]
            probs = scipy.special.softmax(scores)

        elif canonical_id == "tfidf_linearsvm":
            model, vec = self._get_tfidf_model()
            X = vec.transform([clean_text])
            scores = model.decision_function(X)[0]
            probs = scipy.special.softmax(scores)

        elif canonical_id == "glove_bilstm":
            model, tok = self._get_glove_bilstm_model()
            seqs = tok.texts_to_sequences([clean_text])
            padded = tok.pad_sequences(seqs, max_len=30, padding="pre", truncating="post")
            inp = torch.tensor(padded, dtype=torch.long, device=self.device)
            with torch.no_grad():
                probs = model.predict_proba(inp)[0].cpu().numpy()

        elif canonical_id == "fasttext_attention":
            model, tok = self._get_fasttext_attention_model()
            seqs = tok.texts_to_sequences([clean_text])
            padded = tok.pad_sequences(seqs, max_len=30, padding="pre", truncating="post")
            inp = torch.tensor(padded, dtype=torch.long, device=self.device)
            with torch.no_grad():
                probs = model.predict_proba(inp, return_attention=False)[0].cpu().numpy()

        elif canonical_id == "bert_transformer":
            model, tok_mgr = self._get_bert_model()
            encoding = tok_mgr.encode_text(
                clean_text,
                max_length=80,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inp_ids = encoding["input_ids"].to(self.device)
            attn_mask = encoding["attention_mask"].to(self.device)
            token_type_ids = encoding.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=inp_ids,
                    attention_mask=attn_mask,
                    token_type_ids=token_type_ids,
                )
                probs = model.predict_probabilities(outputs.logits).cpu().numpy()[0]

        else:
            raise ModelNotFoundError(f"Model handler for '{canonical_id}' not implemented.")

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        # Sanity validation
        probs = np.clip(probs, 0.0, 1.0)
        p_sum = float(np.sum(probs))
        if p_sum > 0:
            probs = probs / p_sum
        else:
            probs = np.full(len(self.classes), 1.0 / len(self.classes))

        best_idx = int(np.argmax(probs))
        predicted_class = self.classes[best_idx]
        confidence = float(probs[best_idx])

        prob_dict = {
            c: round(float(p), 4) for c, p in zip(self.classes, probs)
        }

        return PredictionResult(
            text=text,
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            model_name=meta["name"],
            model_version=f"{meta['phase']} ({canonical_id})",
            latency_ms=latency_ms,
        )

    def predict_batch(
        self, texts: List[str], model_id: Optional[str] = "bow_linearsvm"
    ) -> BatchPredictionResult:
        """
        Runs batch prediction across a list of texts using the selected model.
        """
        canonical_id = self.normalize_model_id(model_id)
        meta = MODEL_METADATA_REGISTRY[canonical_id]
        start_time = time.perf_counter()

        predictions: List[PredictionResult] = []
        for text in texts:
            pred = self.predict(text, model_id=canonical_id)
            predictions.append(pred)

        batch_latency = round((time.perf_counter() - start_time) * 1000, 3)

        return BatchPredictionResult(
            predictions=predictions,
            total_count=len(texts),
            batch_latency_ms=batch_latency,
            model_name=meta["name"],
            model_version=f"{meta['phase']} ({canonical_id})",
        )


_global_multi_predictor: Optional[MultiModelPredictor] = None


def get_multi_predictor(base_dir: Optional[str] = None) -> MultiModelPredictor:
    """Returns singleton MultiModelPredictor instance."""
    global _global_multi_predictor
    if _global_multi_predictor is None:
        _global_multi_predictor = MultiModelPredictor(base_dir=base_dir)
    return _global_multi_predictor
