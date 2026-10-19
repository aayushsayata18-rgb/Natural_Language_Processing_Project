"""
Model artifact loader with validation, integrity verification, and timing.
"""

import json
import os
import time
from typing import Dict, Any, Tuple
import joblib

from .exceptions import (
    ModelNotFoundError,
    TokenizerNotFoundError,
    ConfigurationError,
    ArtifactValidationError,
)
from .utils import verify_artifact_checksums


class ModelArtifactBundle:
    """Container holding in-memory instances of all loaded artifacts."""

    def __init__(
        self,
        model: Any,
        vectorizer: Any,
        config: Dict[str, Any],
        label_mapping: Dict[str, Any],
        preprocessing_config: Dict[str, Any],
        metadata: Dict[str, Any],
        load_time_ms: float,
        model_dir: str,
    ):
        self.model = model
        self.vectorizer = vectorizer
        self.config = config
        self.label_mapping = label_mapping
        self.preprocessing_config = preprocessing_config
        self.metadata = metadata
        self.load_time_ms = load_time_ms
        self.model_dir = model_dir

        # Extract normalized class lists
        # Support both {"0": "Regular"} and {"Regular": 0} formats
        self.id_to_class = {}
        for k, v in label_mapping.items():
            if str(k).isdigit():
                self.id_to_class[int(k)] = str(v)
            elif isinstance(v, int):
                self.id_to_class[v] = str(k)

        self.classes = [self.id_to_class[i] for i in range(len(self.id_to_class))]


class ModelLoader:
    """Manages artifact resolution, verification, loading, and lifecycle."""

    _cached_bundle: Any = None

    @classmethod
    def resolve_model_dir(cls, model_dir: str = None) -> str:
        """Resolves absolute path to models/final directory."""
        if model_dir:
            if os.path.exists(model_dir):
                return os.path.abspath(model_dir)
            raise ModelNotFoundError(f"Specified model directory not found: {model_dir}")

        # Check environment variable
        env_dir = os.environ.get("SARCASM_MODEL_DIR")
        if env_dir and os.path.exists(env_dir):
            return os.path.abspath(env_dir)

        # Standard relative lookup from src/inference/
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(curr_dir, "..", "..", "models", "final"),
            os.path.join(curr_dir, "..", "models", "final"),
            os.path.join(os.getcwd(), "models", "final"),
            os.path.join(os.getcwd(), "ml-service", "models", "final"),
        ]

        for cand in candidates:
            abs_cand = os.path.abspath(cand)
            if os.path.exists(abs_cand) and os.path.exists(os.path.join(abs_cand, "model")):
                return abs_cand

        raise ModelNotFoundError(
            f"Could not locate 'models/final' directory in candidates: {candidates}"
        )

    @classmethod
    def load(cls, model_dir: str = None, verify_checksums: bool = False, force_reload: bool = False) -> ModelArtifactBundle:
        """
        Loads all required model artifacts. Caches by default.
        """
        resolved_dir = cls.resolve_model_dir(model_dir)

        if cls._cached_bundle is not None and not force_reload and cls._cached_bundle.model_dir == resolved_dir:
            return cls._cached_bundle

        start_time = time.perf_counter()

        model_path = os.path.join(resolved_dir, "model", "model.joblib")
        vec_path = os.path.join(resolved_dir, "tokenizer", "vectorizer.joblib")
        cfg_path = os.path.join(resolved_dir, "config.json")
        lbl_path = os.path.join(resolved_dir, "label_mapping.json")
        prep_path = os.path.join(resolved_dir, "preprocessing_config.json")
        meta_path = os.path.join(resolved_dir, "model_metadata.json")

        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"Model weights file not found: {model_path}")
        if not os.path.exists(vec_path):
            raise TokenizerNotFoundError(f"Vectorizer file not found: {vec_path}")
        if not os.path.exists(cfg_path):
            raise ConfigurationError(f"Config file not found: {cfg_path}")
        if not os.path.exists(lbl_path):
            raise ConfigurationError(f"Label mapping file not found: {lbl_path}")
        if not os.path.exists(prep_path):
            raise ConfigurationError(f"Preprocessing config not found: {prep_path}")
        if not os.path.exists(meta_path):
            raise ConfigurationError(f"Model metadata file not found: {meta_path}")

        if verify_checksums:
            verify_artifact_checksums(resolved_dir)

        # Deserialize artifacts
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)

        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        with open(lbl_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)

        with open(prep_path, "r", encoding="utf-8") as f:
            preprocessing_config = json.load(f)

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        load_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        bundle = ModelArtifactBundle(
            model=model,
            vectorizer=vectorizer,
            config=config,
            label_mapping=label_mapping,
            preprocessing_config=preprocessing_config,
            metadata=metadata,
            load_time_ms=load_time_ms,
            model_dir=resolved_dir,
        )

        cls._cached_bundle = bundle
        return bundle

    @classmethod
    def clear_cache(cls):
        """Clears cached bundle to test reload mechanics."""
        cls._cached_bundle = None
