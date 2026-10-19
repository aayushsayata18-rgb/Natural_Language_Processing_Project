"""
Data schemas and contracts for the Sarcasm & Irony Inference Service.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class PredictionInput(BaseModel):
    """Input payload for single text prediction."""
    text: str = Field(..., description="Raw text snippet to analyze")

    @field_validator("text")
    def validate_text(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Input must be a valid string.")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Text input cannot be empty.")
        return stripped


class BatchPredictionInput(BaseModel):
    """Input payload for batch text prediction."""
    texts: List[str] = Field(..., description="List of raw text snippets to analyze")

    @field_validator("texts")
    def validate_texts(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("Batch input must be a list of strings.")
        if len(v) == 0:
            raise ValueError("Batch list cannot be empty.")
        cleaned = []
        for i, item in enumerate(v):
            if not isinstance(item, str):
                raise ValueError(f"Item at index {i} is not a string: {type(item).__name__}")
            if not item.strip():
                raise ValueError(f"Item at index {i} cannot be empty.")
            cleaned.append(item)
        return cleaned


class PredictionResult(BaseModel):
    """Standardized prediction result for a single input text."""
    text: str = Field(..., description="Input text evaluated by the model")
    predicted_class: str = Field(..., description="Class with highest estimated probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Highest predicted class probability")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution across all 4 target classes")
    model_name: str = Field(..., description="Name of the deployed final model")
    model_version: str = Field(..., description="Semantic version of the deployed model")
    latency_ms: Optional[float] = Field(None, ge=0.0, description="Inference latency in milliseconds")

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return self.model_dump()


class BatchPredictionResult(BaseModel):
    """Standardized result for batch inference."""
    predictions: List[PredictionResult] = Field(..., description="Ordered list of prediction results")
    total_count: int = Field(..., ge=0, description="Total number of evaluated texts")
    batch_latency_ms: float = Field(..., ge=0.0, description="Total batch inference latency in milliseconds")
    model_name: str = Field(..., description="Name of the deployed final model")
    model_version: str = Field(..., description="Semantic version of the deployed model")

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch result to dictionary representation."""
        return self.model_dump()
