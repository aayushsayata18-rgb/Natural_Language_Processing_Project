# API Reference Documentation

**Project:** Sarcasm & Irony Detection System using NLP  
**Phase:** Phase 11 — ML API + Node/Express Integration  
**Model:** `SarcasmIronyClassifier-LinearSVM` (`SIC-v1.0`)

---

## Architecture Overview

The system operates across two decoupled backend tiers:
1. **Application Backend (`Node.js + Express`)**: Exposes client-facing REST APIs (`/api/*`), manages application workflows, request validation, rate limiting, and persistence to MongoDB.
2. **ML Inference Service (`Python + FastAPI`)**: An internal microservice dedicated solely to NLP tokenization, BoW vectorization, and model inference.

```
React (Phase 12) ──► Node.js/Express (Port 5000) ──► Python FastAPI (Port 8000) ──► LinearSVC Model
                           │
                           ▼
                    MongoDB Database
```

---

## 1. Node.js Application Backend Endpoints (`http://localhost:5000`)

### 1.1 Health Check
- **Endpoint:** `/api/health`
- **Method:** `GET`
- **Purpose:** Checks health of the Node backend, Python ML service, and MongoDB database.
- **Request Headers:** None
- **Request Body:** None
- **Response `200 OK`:**
  ```json
  {
    "success": true,
    "service": "backend",
    "status": "healthy",
    "components": {
      "mlService": {
        "status": "available",
        "modelLoaded": true,
        "modelName": "SarcasmIronyClassifier-LinearSVM",
        "modelVersion": "SIC-v1.0"
      },
      "database": {
        "status": "connected"
      }
    },
    "timestamp": "2026-10-19T15:35:00.000Z"
  }
  ```
- **Error Responses:**
  - If ML service is unreachable, `status` becomes `"degraded"` and `components.mlService.status` becomes `"unavailable"`.

---

### 1.2 Single Text Prediction
- **Endpoint:** `/api/predict`
- **Method:** `POST`
- **Purpose:** Classifies text input for sarcasm and irony and persists the result to MongoDB.
- **Request Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "text": "Fantastic, another three-hour meeting on a Friday afternoon."
  }
  ```
- **Validation Rules:**
  - `text`: Required, string, trimmed, non-empty, max length 5,000 characters.
- **Response `200 OK`:**
  ```json
  {
    "success": true,
    "data": {
      "text": "Fantastic, another three-hour meeting on a Friday afternoon.",
      "predicted_class": "Regular",
      "predictedClass": "Regular",
      "confidence": 0.5803,
      "probabilities": {
        "Regular": 0.5803,
        "Sarcasm": 0.1578,
        "Irony": 0.1063,
        "Figurative": 0.1555
      },
      "model_name": "SarcasmIronyClassifier-LinearSVM",
      "modelName": "SarcasmIronyClassifier-LinearSVM",
      "model_version": "SIC-v1.0",
      "modelVersion": "SIC-v1.0",
      "latency_ms": 1.24,
      "history_saved": true,
      "id": "6713c230e527dca963df991b"
    }
  }
  ```
- **Status Codes & Error Responses:**
  - `400 Bad Request`: Validation failure (empty, non-string, or oversized text).
    ```json
    {
      "success": false,
      "error": {
        "code": "EMPTY_TEXT_INPUT",
        "message": "The \"text\" field cannot be empty or contain only whitespace."
      }
    }
    ```
  - `429 Too Many Requests`: Exceeded rate limit (120 req/min).
  - `503 Service Unavailable`: Python ML service stopped or model unloaded.
    ```json
    {
      "success": false,
      "error": {
        "code": "ML_SERVICE_UNAVAILABLE",
        "message": "ML inference service is currently unreachable or stopped."
      }
    }
    ```
  - `504 Gateway Timeout`: Python ML service took longer than `ML_SERVICE_TIMEOUT_MS` (5000ms).

---

### 1.3 Batch Text Prediction
- **Endpoint:** `/api/predict/batch`
- **Method:** `POST`
- **Purpose:** Efficiently classifies a batch of texts and persists results to MongoDB.
- **Request Body:**
  ```json
  {
    "texts": [
      "What a wonderful sunny day for a walk in the park.",
      "Brilliant code, it only failed in every single test."
    ]
  }
  ```
- **Validation Rules:**
  - `texts`: Required array of strings, 1–100 items, each 1–5,000 chars.
- **Response `200 OK`:**
  ```json
  {
    "success": true,
    "data": {
      "predictions": [
        {
          "text": "What a wonderful sunny day for a walk in the park.",
          "predicted_class": "Regular",
          "confidence": 0.884,
          "probabilities": { ... },
          "model_name": "SarcasmIronyClassifier-LinearSVM",
          "model_version": "SIC-v1.0"
        },
        {
          "text": "Brilliant code, it only failed in every single test.",
          "predicted_class": "Sarcasm",
          "confidence": 0.791,
          "probabilities": { ... },
          "model_name": "SarcasmIronyClassifier-LinearSVM",
          "model_version": "SIC-v1.0"
        }
      ],
      "total_count": 2,
      "batch_latency_ms": 2.45,
      "model_name": "SarcasmIronyClassifier-LinearSVM",
      "model_version": "SIC-v1.0"
    }
  }
  ```

---

### 1.4 Get Prediction History
- **Endpoint:** `/api/predictions`
- **Method:** `GET`
- **Query Parameters:**
  - `limit`: (optional, default: `20`, max: `100`)
  - `page`: (optional, default: `1`)
  - `predictedClass`: (optional filter: `Regular`, `Sarcasm`, `Irony`, `Figurative`)
- **Response `200 OK`:**
  ```json
  {
    "success": true,
    "data": [
      {
        "id": "6713c230e527dca963df991b",
        "text": "Fantastic, another three-hour meeting.",
        "predicted_class": "Sarcasm",
        "confidence": 0.87,
        "probabilities": { "Regular": 0.03, "Sarcasm": 0.87, "Irony": 0.07, "Figurative": 0.03 },
        "model_name": "SarcasmIronyClassifier-LinearSVM",
        "model_version": "SIC-v1.0",
        "created_at": "2026-10-19T15:30:00.000Z"
      }
    ],
    "pagination": {
      "total": 142,
      "page": 1,
      "limit": 20,
      "pages": 8
    }
  }
  ```

---

### 1.5 Delete Prediction History Item
- **Endpoint:** `/api/predictions/:id`
- **Method:** `DELETE`
- **URL Parameters:**
  - `id`: MongoDB ObjectId hex string
- **Response `200 OK`:**
  ```json
  {
    "success": true,
    "message": "Prediction history record deleted successfully.",
    "data": {
      "id": "6713c230e527dca963df991b"
    }
  }
  ```
- **Error Responses:**
  - `400 Bad Request`: Invalid ObjectId hex string.
  - `404 Not Found`: Record does not exist in the database.

---

## 2. Python FastAPI ML Service Endpoints (`http://localhost:8000`)

### 2.1 Health Check
- **Endpoint:** `/health`
- **Method:** `GET`
- **Response `200 OK`:**
  ```json
  {
    "status": "ok",
    "service": "ml-service",
    "model_loaded": true,
    "model_name": "SarcasmIronyClassifier-LinearSVM",
    "model_version": "SIC-v1.0"
  }
  ```

---

### 2.2 Readiness Probe
- **Endpoint:** `/ready`
- **Method:** `GET`
- **Response `200 OK`:**
  ```json
  {
    "ready": true,
    "model_loaded": true
  }
  ```
- **Response `503 Service Unavailable`:** Returned if model loading failed during startup.

---

### 2.3 Model Information
- **Endpoint:** `/model-info`
- **Method:** `GET`
- **Response `200 OK`:**
  ```json
  {
    "model_name": "SarcasmIronyClassifier-LinearSVM",
    "model_version": "SIC-v1.0",
    "model_type": "LinearSVC (One-vs-Rest) with BoW CountVectorizer",
    "classes": [
      "Regular",
      "Sarcasm",
      "Irony",
      "Figurative"
    ],
    "macro_f1": 0.7719,
    "accuracy": 0.7700
  }
  ```

---

### 2.4 ML Direct Predict
- **Endpoint:** `/predict`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "text": "Great job everyone!"
  }
  ```
- **Response `200 OK`:**
  ```json
  {
    "text": "Great job everyone!",
    "predicted_class": "Regular",
    "confidence": 0.8842,
    "probabilities": {
      "Regular": 0.8842,
      "Sarcasm": 0.0412,
      "Irony": 0.0381,
      "Figurative": 0.0365
    },
    "model_name": "SarcasmIronyClassifier-LinearSVM",
    "model_version": "SIC-v1.0",
    "latency_ms": 0.52
  }
  ```
