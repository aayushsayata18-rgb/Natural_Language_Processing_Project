"""
Configuration settings for FastAPI ML Service.
Phase 11 — ML API Service Configuration.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


# ML service base directory: d:/Natural_Language_Processing_Project/sarcasm-irony-detection/ml-service
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "final"


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # Service Bindings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    # ML Artifacts
    model_path: str = Field(default=str(DEFAULT_MODEL_DIR), alias="MODEL_PATH")
    model_version: str = Field(default="SIC-v1.0", alias="MODEL_VERSION")
    verify_checksums: bool = Field(default=False, alias="VERIFY_CHECKSUMS")

    # Request Limits
    max_text_length: int = Field(default=5000, alias="MAX_TEXT_LENGTH")
    max_batch_size: int = Field(default=100, alias="MAX_BATCH_SIZE")

    @property
    def resolved_model_dir(self) -> Path:
        """Resolve model directory path whether absolute or relative."""
        p = Path(self.model_path)
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        return p


settings = Settings()
