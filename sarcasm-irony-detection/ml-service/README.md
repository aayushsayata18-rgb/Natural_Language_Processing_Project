# Python ML Inference Service (FastAPI)

**Component:** Dedicated Machine Learning Inference Microservice  
**Project:** Sarcasm & Irony Detection System using NLP (Phase 11)  
**Port:** `8000` (configurable via `PORT`)  
**Framework:** FastAPI + Uvicorn  
**Model:** `SarcasmIronyClassifier-LinearSVM` (`SIC-v1.0`)

---

## Overview

The Python ML Service provides a dedicated HTTP microservice exposing the finalized NLP model for sarcasm and irony inference. It handles text validation, BoW CountVectorizer transformation, LinearSVC One-vs-Rest decision scoring, and softmax probability calibration.

### Core Architecture
- **Framework:** FastAPI with Uvicorn ASGI server
- **Lifespan Loader:** The model bundle (`models/final/`) is loaded into memory exactly once at startup.
- **Stateless Inference:** Pure functional prediction without session state or external dependencies.
- **Contract Schema:** Strongly typed Pydantic request and response contracts.

---

## Directory Structure

```
ml-service/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── config.py       # Service configuration via Pydantic settings
│   │   ├── schemas.py      # Request / response Pydantic models
│   │   └── main.py         # FastAPI application with lifespan & handlers
│   │
│   └── inference/
│       ├── __init__.py
│       ├── loader.py        # ModelLoader with checksum validation
│       ├── predictor.py     # SarcasmIronyPredictor
│       ├── preprocessing.py # Text preprocessing & normalization
│       ├── schemas.py       # Inference schemas
│       ├── exceptions.py    # Custom exception hierarchy
│       └── cli.py           # CLI evaluation utility
│
├── models/
│   └── final/               # Final packaged model artifacts (LinearSVC BoW)
│
├── tests/
│   ├── test_api.py          # FastAPI endpoint & validation tests
│   ├── test_artifacts.py    # Artifact bundle & checksum verification
│   ├── test_predictor.py    # Predictor logic & softmax tests
│   └── test_validation.py   # Input edge cases & boundary checks
│
├── requirements.txt         # Production runtime dependencies
├── requirements-inference.txt
├── .env.example
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status and model loaded state |
| `GET` | `/ready` | Kubernetes/Service readiness probe (200 when ready, 503 if unloaded) |
| `GET` | `/model-info` | Sanitized model metadata (macro F1, accuracy, classes) |
| `POST` | `/predict` | Single text prediction with calibrated probabilities |
| `POST` | `/predict/batch` | Vectorized batch text prediction |
| `GET` | `/docs` | Interactive Swagger UI API documentation |
| `GET` | `/openapi.json` | OpenAPI 3.1 specification schema |

---

## Getting Started

### Prerequisites
- Python 3.14 (or Python >= 3.11)
- Pip dependencies installed from `requirements.txt`

### Installation
```bash
pip install -r requirements.txt
```

### Environment Variables
Configure `.env` using `.env.example`:
- `HOST`: Host binding (default: `0.0.0.0`)
- `PORT`: Port number (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `info`)
- `MODEL_PATH`: Path to model directory (default: `models/final`)
- `MODEL_VERSION`: Expected model version (default: `SIC-v1.0`)
- `VERIFY_CHECKSUMS`: Validate artifact SHA256 hashes on load (default: `false`)
- `MAX_TEXT_LENGTH`: Max allowable character length (default: `5000`)

### Running the Service
```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
Interactive API documentation will be available at:
- `http://localhost:8000/docs`

### Running Tests
Execute the full test suite with pytest:
```bash
pytest tests/ -v
```
All 210 tests pass across data processing, model packaging, and API endpoints.
