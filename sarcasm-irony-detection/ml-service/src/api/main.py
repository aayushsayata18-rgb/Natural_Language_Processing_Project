"""
FastAPI Main Application for Sarcasm & Irony ML Service.
Phase 11 — Production ML Inference API.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .schemas import (
    HealthResponse,
    ReadinessResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
)

# Import Phase 10 Inference Predictor and Exceptions
from ..inference.predictor import SarcasmIronyPredictor
from ..inference.exceptions import (
    InferenceError,
    ModelNotFoundError,
    ArtifactValidationError,
    InvalidInputError,
    ConfigurationError,
    TokenizerNotFoundError,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ml-service.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads the ML model artifact bundle exactly once on startup.
    Cleans up resources cleanly on shutdown.
    """
    logger.info("Initializing ML Service...")
    logger.info("Loading model artifacts from: %s", settings.resolved_model_dir)

    try:
        predictor = SarcasmIronyPredictor(
            model_dir=str(settings.resolved_model_dir),
            verify_checksums=settings.verify_checksums,
        )
        app.state.predictor = predictor
        app.state.model_loaded = True
        logger.info(
            "Final Model loaded successfully: %s (%s)",
            predictor.model_name,
            predictor.model_version,
        )
    except Exception as exc:
        app.state.predictor = None
        app.state.model_loaded = False
        logger.critical("Failed to load model artifacts on startup: %s", exc, exc_info=True)

    yield

    logger.info("Shutting down ML Service...")
    app.state.predictor = None
    app.state.model_loaded = False


# Initialize FastAPI app
app = FastAPI(
    title="Sarcasm & Irony Detection ML Inference API",
    description="Dedicated Python FastAPI service exposing the final NLP model (LinearSVC BoW) for inference.",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Internal service, open to internal backend callers
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI / Pydantic validation errors."""
    error_details = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "Validation error")
        error_details.append(f"{loc}: {msg}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "ValidationError", "message": "; ".join(error_details)},
    )


@app.exception_handler(InvalidInputError)
async def invalid_input_exception_handler(request: Request, exc: InvalidInputError):
    """Handle custom inference input validation failures."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "InvalidInputError", "message": str(exc)},
    )


@app.exception_handler(ModelNotFoundError)
async def model_not_found_exception_handler(request: Request, exc: ModelNotFoundError):
    """Handle missing model artifact errors."""
    logger.error("Model artifacts not found: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "ModelNotFoundError", "message": "Model artifacts are missing or inaccessible."},
    )


@app.exception_handler(ArtifactValidationError)
async def artifact_validation_exception_handler(request: Request, exc: ArtifactValidationError):
    """Handle corrupted or mismatching model artifact errors."""
    logger.error("Artifact checksum validation failure: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "ArtifactValidationError", "message": "Model artifact validation failed."},
    )


@app.exception_handler(InferenceError)
async def inference_exception_handler(request: Request, exc: InferenceError):
    """Handle model prediction runtime failures."""
    logger.error("Inference failure: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "InferenceError", "message": "Model inference failed to generate a valid prediction."},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler protecting against stack trace leaks."""
    logger.critical("Unhandled server exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "InternalServerError", "message": "An unexpected server error occurred."},
    )


# API Endpoints
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["Monitoring"],
)
async def health_check():
    """Confirms the Python ML service is running and reports model load state."""
    predictor: Optional[SarcasmIronyPredictor] = getattr(app.state, "predictor", None)
    is_loaded = bool(predictor is not None and getattr(app.state, "model_loaded", False))

    return HealthResponse(
        status="ok",
        service="ml-service",
        model_loaded=is_loaded,
        model_name=predictor.model_name if is_loaded else None,
        model_version=predictor.model_version if is_loaded else None,
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Service Readiness Probe",
    tags=["Monitoring"],
    responses={
        503: {"model": ReadinessResponse, "description": "Service not ready to accept inference requests."}
    },
)
async def readiness_probe():
    """Confirms whether the ML model is fully loaded and ready to accept inference requests."""
    is_loaded = bool(
        getattr(app.state, "predictor", None) is not None
        and getattr(app.state, "model_loaded", False)
    )
    if not is_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False, "model_loaded": False},
        )
    return ReadinessResponse(ready=True, model_loaded=True)


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Metadata Information",
    tags=["Model Information"],
    responses={
        503: {"model": ErrorResponse, "description": "Model is not loaded."}
    },
)
async def get_model_info():
    """Returns safe metadata regarding the loaded final NLP model."""
    predictor: Optional[SarcasmIronyPredictor] = getattr(app.state, "predictor", None)
    if not predictor or not getattr(app.state, "model_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not currently loaded in memory.",
        )

    meta = predictor.get_metadata()
    metrics = meta.get("metrics", {})

    return ModelInfoResponse(
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        model_type=meta.get("algorithm", "LinearSVC (One-vs-Rest) with BoW CountVectorizer"),
        classes=predictor.get_classes(),
        macro_f1=metrics.get("val_macro_f1", 0.7719),
        accuracy=metrics.get("val_accuracy", 0.7700),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single Text Inference",
    tags=["Inference"],
    responses={
        422: {"model": ErrorResponse, "description": "Input validation failure."},
        503: {"model": ErrorResponse, "description": "Model service unavailable."},
        500: {"model": ErrorResponse, "description": "Inference computation error."},
    },
)
async def predict_single(request: PredictionRequest):
    """
    Classifies a single input text for Sarcasm and Irony.
    Returns predicted class, confidence, calibrated probability distribution, and latency.
    """
    predictor: Optional[SarcasmIronyPredictor] = getattr(app.state, "predictor", None)
    if not predictor or not getattr(app.state, "model_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or currently unavailable.",
        )

    res = predictor.predict(request.text)

    return PredictionResponse(
        text=res.text,
        predicted_class=res.predicted_class,
        confidence=res.confidence,
        probabilities=res.probabilities,
        model_name=res.model_name,
        model_version=res.model_version,
        latency_ms=res.latency_ms,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Text Inference",
    tags=["Inference"],
    responses={
        422: {"model": ErrorResponse, "description": "Batch input validation failure."},
        503: {"model": ErrorResponse, "description": "Model service unavailable."},
        500: {"model": ErrorResponse, "description": "Batch inference computation error."},
    },
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Classifies a batch of input texts sequentially/vectorized.
    Returns ordered list of prediction results with total batch latency.
    """
    predictor: Optional[SarcasmIronyPredictor] = getattr(app.state, "predictor", None)
    if not predictor or not getattr(app.state, "model_loaded", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or currently unavailable.",
        )

    batch_res = predictor.predict_batch(request.texts)

    predictions = [
        PredictionResponse(
            text=p.text,
            predicted_class=p.predicted_class,
            confidence=p.confidence,
            probabilities=p.probabilities,
            model_name=p.model_name,
            model_version=p.model_version,
            latency_ms=p.latency_ms,
        )
        for p in batch_res.predictions
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        total_count=batch_res.total_count,
        batch_latency_ms=batch_res.batch_latency_ms,
        model_name=batch_res.model_name,
        model_version=batch_res.model_version,
    )


def start():
    """Entry point for running with uvicorn via CLI or script."""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    start()
