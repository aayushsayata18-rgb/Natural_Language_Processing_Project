"""
Sarcasm & Irony Detection Inference Package.
Phase 10 — Production-Ready Model Packaging & Service Interface.
"""

from typing import Dict, List, Any, Optional

from .exceptions import (
    InferenceError,
    ModelNotFoundError,
    TokenizerNotFoundError,
    ConfigurationError,
    ArtifactValidationError,
    InvalidInputError,
)
from .schemas import (
    PredictionInput,
    BatchPredictionInput,
    PredictionResult,
    BatchPredictionResult,
)
from .loader import ModelLoader, ModelArtifactBundle
from .preprocessing import InferenceTextPreprocessor
from .predictor import SarcasmIronyPredictor
from .utils import (
    verify_artifact_checksums,
    get_execution_device,
    get_memory_info,
)

from .multi_predictor import (
    MultiModelPredictor,
    get_multi_predictor,
    MODEL_METADATA_REGISTRY,
    MODEL_ALIASES,
)

__version__ = "1.0.0"

_global_predictor: Optional[SarcasmIronyPredictor] = None


def get_predictor(model_dir: Optional[str] = None) -> SarcasmIronyPredictor:
    """Returns singleton instance of SarcasmIronyPredictor."""
    global _global_predictor
    if _global_predictor is None or (model_dir and _global_predictor.model_dir != model_dir):
        _global_predictor = SarcasmIronyPredictor(model_dir=model_dir)
    return _global_predictor


def predict_text(text: str, model_id: Optional[str] = "bow_linearsvm", model_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for single text inference across models.
    Used by Python scripts and backend API integration.
    """
    return get_multi_predictor(base_dir=model_dir).predict(text, model_id=model_id).to_dict()


def predict_batch(texts: List[str], model_id: Optional[str] = "bow_linearsvm", model_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for vectorized batch inference across models.
    Used by Python scripts and backend API integration.
    """
    return get_multi_predictor(base_dir=model_dir).predict_batch(texts, model_id=model_id).to_dict()


def load_model(model_dir: Optional[str] = None) -> SarcasmIronyPredictor:
    """Pre-warms and returns the SarcasmIronyPredictor service instance."""
    return get_predictor(model_dir=model_dir)


__all__ = [
    "SarcasmIronyPredictor",
    "MultiModelPredictor",
    "get_multi_predictor",
    "MODEL_METADATA_REGISTRY",
    "MODEL_ALIASES",
    "predict_text",
    "predict_batch",
    "load_model",
    "get_predictor",
    "InferenceTextPreprocessor",
    "ModelLoader",
    "ModelArtifactBundle",
    "PredictionInput",
    "BatchPredictionInput",
    "PredictionResult",
    "BatchPredictionResult",
    "InferenceError",
    "ModelNotFoundError",
    "TokenizerNotFoundError",
    "ConfigurationError",
    "ArtifactValidationError",
    "InvalidInputError",
    "verify_artifact_checksums",
    "get_execution_device",
    "get_memory_info",
]

