"""
FastAPI Pydantic Schemas for Sarcasm & Irony ML Service.
Phase 11 — Request and Response Contract Definitions.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Health status response."""
    status: str = Field(default="ok", description="Service health status")
    service: str = Field(default="ml-service", description="Service identifier")
    model_loaded: bool = Field(..., description="Whether the NLP model is currently loaded in memory")
    model_name: Optional[str] = Field(None, description="Name of the loaded model")
    model_version: Optional[str] = Field(None, description="Version of the loaded model")


class ReadinessResponse(BaseModel):
    """Readiness probe response."""
    ready: bool = Field(..., description="Whether service is ready to accept traffic")
    model_loaded: bool = Field(..., description="Whether model artifact is loaded and verified")


class ModelInfoResponse(BaseModel):
    """Safe model metadata response without exposing internal paths or secrets."""
    model_name: str = Field(..., description="Name of the deployed NLP model")
    model_version: str = Field(..., description="Version identifier")
    model_type: str = Field(..., description="Architecture and algorithm type")
    classes: List[str] = Field(..., description="Target classes recognized by the model")
    macro_f1: Optional[float] = Field(None, description="Validation Macro-averaged F1 score")
    accuracy: Optional[float] = Field(None, description="Validation accuracy score")


class ModelListItem(BaseModel):
    """Specification of an available model in the comparative registry."""
    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Short display name")
    full_name: str = Field(..., description="Full academic model title")
    phase: str = Field(..., description="Project phase")
    paradigm: str = Field(..., description="NLP representation paradigm")
    parameter_count: int = Field(..., description="Total parameter count")
    val_macro_f1: float = Field(..., description="Validation Macro F1 score")
    val_accuracy: float = Field(..., description="Validation accuracy score")
    latency_ms: float = Field(..., description="Expected per-text inference latency in milliseconds")
    is_champion: bool = Field(..., description="Whether this is the Phase 9 selected production model")
    description: str = Field(..., description="Architectural summary and trade-offs")


class ModelListResponse(BaseModel):
    """List of all available models for comparative inference."""
    models: List[ModelListItem] = Field(..., description="List of registered model specifications")
    default_model: str = Field(default="bow_linearsvm", description="Default model ID")
    total_models: int = Field(..., description="Total number of models available")


class PredictionRequest(BaseModel):
    """Single text inference request schema."""
    text: str = Field(
        ...,
        description="Input text snippet to classify for sarcasm and irony",
        examples=["Fantastic, another three-hour meeting."]
    )
    model: Optional[str] = Field(
        default="bow_linearsvm",
        description="Target model identifier (bow_linearsvm, tfidf_linearsvm, glove_bilstm, fasttext_attention, bert_transformer)",
        examples=["bow_linearsvm"]
    )

    @field_validator("text")
    def validate_text(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Input text must be a valid string.")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Input text cannot be empty or whitespace only.")
        if len(stripped) > 5000:
            raise ValueError(f"Input text exceeds maximum allowable length of 5000 characters (received {len(stripped)}).")
        return stripped


class PredictionResponse(BaseModel):
    """Standardized single prediction response schema."""
    text: str = Field(..., description="Original input text")
    predicted_class: str = Field(..., description="Predicted category: Regular, Sarcasm, Irony, or Figurative")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Highest softmax confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Calibrated class probability distribution")
    model_name: str = Field(..., description="Deployed model name")
    model_version: str = Field(..., description="Deployed model version")
    latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")


class BatchPredictionRequest(BaseModel):
    """Batch inference request schema."""
    texts: List[str] = Field(
        ...,
        description="List of input text snippets to classify",
        examples=[["Great job breaking the build.", "Have a nice weekend!"]]
    )
    model: Optional[str] = Field(
        default="bow_linearsvm",
        description="Target model identifier for batch classification",
        examples=["bow_linearsvm"]
    )

    @field_validator("texts")
    def validate_texts(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("Texts must be provided as a list of strings.")
        if len(v) == 0:
            raise ValueError("Batch list cannot be empty.")
        if len(v) > 100:
            raise ValueError(f"Batch size exceeds maximum limit of 100 texts (received {len(v)}).")
        cleaned = []
        for i, item in enumerate(v):
            if not isinstance(item, str):
                raise ValueError(f"Item at index {i} is not a string.")
            s = item.strip()
            if not s:
                raise ValueError(f"Item at index {i} cannot be empty.")
            if len(s) > 5000:
                raise ValueError(f"Item at index {i} exceeds max length of 5000 characters.")
            cleaned.append(s)
        return cleaned


class BatchPredictionResponse(BaseModel):
    """Batch inference response schema."""
    predictions: List[PredictionResponse] = Field(..., description="Ordered prediction results")
    total_count: int = Field(..., description="Total number of evaluated inputs")
    batch_latency_ms: float = Field(..., description="Total batch inference latency in milliseconds")
    model_name: str = Field(..., description="Deployed model name")
    model_version: str = Field(..., description="Deployed model version")


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    error: str = Field(..., description="Error category or code")
    message: str = Field(..., description="Human-readable explanation of error")
