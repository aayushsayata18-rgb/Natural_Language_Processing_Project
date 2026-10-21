# Python ML Inference Service (FastAPI)

**Component:** Dedicated Machine Learning Inference Microservice  
**Project:** Sarcasm & Irony Detection System using NLP (Phase 11 & 12)  
**Port:** `8000` (configurable via `PORT`)  
**Framework:** FastAPI + Uvicorn  
**Models:** 5 Cross-Paradigm NLP Models (Champion: `LinearSVM (BoW)`)  
**Dataset:** [Tweets with Sarcasm and Irony](https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony) (Kaggle)

---

## Overview

The Python ML Service provides a high-performance HTTP microservice exposing 5 benchmarked NLP architectures for sarcasm and irony inference. It handles text sanitization, tokenization/vectorization, decision scoring, and softmax probability distribution calibration across 4 classes: `Regular`, `Sarcasm`, `Irony`, and `Figurative`.

### Evaluated & Deployed 5 Models:
1. **LinearSVM (BoW)** (`bow_linearsvm`): CountVectorizer (n-grams 1-2) with LinearSVC. Selected Champion (Macro F1: `0.7187`, Accuracy: `84.57%`, Latency: `0.027 ms`).
2. **LinearSVM (TF-IDF)** (`tfidf_linearsvm`): Sublinear TF-IDF + statistical n-grams (Macro F1: `0.7002`, Accuracy: `85.65%`, Latency: `0.014 ms`).
3. **GloVe + Bi-LSTM** (`glove_bilstm`): 100d pretrained GloVe embeddings with 2-layer Bidirectional LSTM (Macro F1: `0.6940`, Accuracy: `86.12%`, Latency: `0.041 ms`).
4. **FastText + Bi-LSTM + Attention** (`fasttext_attention`): 300d subword embeddings & feed-forward additive self-attention (Macro F1: `0.6928`, Accuracy: `86.03%`, Latency: `0.098 ms`).
5. **BERT Contextual** (`bert_transformer`): Fine-tuned `bert-base-uncased` (12 layers, 768d, 12 heads) with AdamW (Macro F1: `0.6935`, Accuracy: `86.12%`, Latency: `39.33 ms`).

---

## Directory Structure

```
ml-service/
├── src/
│   ├── api/
│   │   ├── config.py            # Service configuration via Pydantic settings
│   │   ├── schemas.py           # Request / response Pydantic models
│   │   └── main.py              # FastAPI application with lifespan & handlers
│   │
│   ├── inference/
│   │   ├── loader.py            # ModelLoader with checksum validation
│   │   ├── predictor.py         # SarcasmIronyPredictor (single champion)
│   │   ├── multi_predictor.py   # MultiModelPredictor (all 5 models)
│   │   ├── preprocessing.py     # Text preprocessing & normalization
│   │   ├── schemas.py           # Inference schemas
│   │   └── exceptions.py        # Custom exception hierarchy
│   │
│   ├── classical/               # Phase 4 (BoW) & Phase 5 (TF-IDF) pipelines
│   ├── deep_learning/           # Phase 6 (GloVe) & Phase 7 (FastText) pipelines
│   └── transformers/            # Phase 8 (BERT) fine-tuning pipeline
│
├── models/
│   └── final/                   # Packaged model artifacts for all paradigms
│
├── tests/
│   ├── test_api.py              # FastAPI endpoint & validation tests
│   ├── test_artifacts.py        # Artifact bundle & checksum verification
│   ├── test_predictor.py        # Predictor logic & softmax tests
│   └── test_validation.py       # Input edge cases & boundary checks
│
├── requirements.txt             # Production runtime dependencies
├── .env.example
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status and loaded model states |
| `GET` | `/ready` | Service readiness probe (200 when ready, 503 if unloaded) |
| `GET` | `/model-info` | Metadata and metrics for the Champion model |
| `GET` | `/models` | Full registry of all 5 benchmarked models and parameters |
| `POST` | `/predict` | Single text prediction (supports specifying `"model": "bow_linearsvm"`) |
| `POST` | `/predict/batch` | Vectorized batch text prediction across any chosen model |
| `GET` | `/docs` | Interactive Swagger UI API documentation |
| `GET` | `/openapi.json` | OpenAPI 3.1 specification schema |

---

## Dataset Provenance

- **Name:** [Tweets with Sarcasm and Irony](https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony) (Kaggle)
- **Author:** Nikhil John K
- **Volume:** 89,536 tweets (decontaminated to 81,359 clean training tweets and 5,527 benchmark test tweets).
- **Classes:** `Regular` (33.6%), `Sarcasm` (29.0%), `Irony` (23.5%), `Figurative` (13.9%).

---

## Getting Started

### Prerequisites
- Python >= 3.11 (tested on Python 3.14)
- Pip dependencies installed from `requirements.txt`

### Installation
```bash
pip install -r requirements.txt
```

### Running the Service
```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
Interactive API documentation: `http://localhost:8000/docs`

### Running Tests
```bash
pytest tests/ -v
```
All 210 test cases covering preprocessing, artifacts, predictors, and API routes pass with 100% rate.
