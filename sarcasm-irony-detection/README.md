# Sarcasm & Irony Detection System using NLP

**Academic Level:** MCA Final-Year Project  
**Current Phase:** Phase 11 — ML API + Node/Express Backend Integration  
**Selected Final Model:** `SarcasmIronyClassifier-LinearSVM` (`SIC-v1.0`) (Phase 9 & 10)  
**Validation Macro F1:** `0.7719` | **Accuracy:** `77.00%`

---

## 1. Project Overview

This project develops an end-to-end Machine Learning and Natural Language Processing (NLP) system designed to detect and classify nuanced figurative language into four distinct categories:
1. **Regular** (literal, non-sarcastic)
2. **Sarcasm** (irony intended to mock, convey contempt, or express ridicule)
3. **Irony** (situational or incongruous expression without necessary hostility)
4. **Figurative** (metaphorical, hyperbolic, or creative non-literal language)

The project systematically evolved through:
- **Phase 1–3:** Problem formulation, data acquisition, Exploratory Data Analysis, and preprocessing.
- **Phase 4–8:** Benchmarking across Classical ML (BoW, TF-IDF), Deep Learning (GloVe BiLSTM, FastText with Self-Attention), and Transformers (BERT fine-tuning).
- **Phase 9:** Scientific model comparison and selection based on generalization, Macro F1, and computational efficiency. LinearSVC One-vs-Rest on Bag-of-Words was selected as the superior architecture.
- **Phase 10:** Model artifact packaging, checksum validation, serialization, and standalone Python inference engine.
- **Phase 11 (Current):** FastAPI Python ML microservice integration with a Node.js + Express application backend and MongoDB database.

---

## 2. System Architecture

```
                    ┌────────────────────────────┐
                    │    React Frontend (UI)     │
                    │        (Phase 12)          │
                    └─────────────┬──────────────┘
                                  │ HTTP / JSON
                                  ▼
                    ┌────────────────────────────┐
                    │  Node.js + Express Backend │
                    │        (Phase 11)          │
                    │         Port: 5000         │
                    └───────┬────────────┬───────┘
                            │            │
             Inference HTTP │            │ History Storage
             (Internal Only)│            │
                            ▼            ▼
               ┌──────────────────┐ ┌───────────────┐
               │ Python FastAPI   │ │ MongoDB       │
               │ ML Service       │ │ Database      │
               │ Port: 8000       │ │ Port: 27017   │
               └────────┬─────────┘ └───────────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ Final NLP Model  │
               │ (LinearSVC BoW)  │
               └──────────────────┘
```

---

## 3. Technology Stack

| Layer | Technology | Role |
|---|---|---|
| **ML Model** | Scikit-learn + Scipy + Joblib | Final LinearSVC One-vs-Rest model with CountVectorizer & Softmax |
| **ML Service** | Python 3.14 + FastAPI + Uvicorn | Dedicated stateless ML inference microservice (Port 8000) |
| **Backend** | Node.js (v24) + Express | Application logic, API routing, validation, rate limiting (Port 5000) |
| **Database** | MongoDB 8.2 + Mongoose | Prediction history storage with schema validation & indexing |
| **Frontend** | React (Vite) | Interactive user interface (scheduled for Phase 12) |

---

## 4. Repository Structure

```
sarcasm-irony-detection/
├── backend/                         # Node.js + Express Backend Tier
│   ├── src/
│   │   ├── config/                  # DB connection and env variables
│   │   ├── controllers/             # Prediction & health controllers
│   │   ├── middleware/              # Validation, rate limiting, error handlers
│   │   ├── models/                  # Mongoose Prediction schema
│   │   ├── routes/                  # API routes (/api/predict, /api/health)
│   │   ├── services/                # Axios ML service client
│   │   ├── app.js                   # Express application setup
│   │   └── server.js                # Server entrypoint
│   ├── tests/                       # Jest test suites (unit + live E2E)
│   ├── scripts/                     # Performance latency benchmark
│   ├── package.json
│   └── .env.example
│
├── ml-service/                      # Python ML Inference Microservice Tier
│   ├── models/final/                # Deployed model artifacts & metadata
│   ├── src/
│   │   ├── api/                     # FastAPI app, config, and schemas
│   │   └── inference/               # ModelLoader, Preprocessor, Predictor
│   ├── tests/                       # Pytest suites (210 tests)
│   ├── requirements.txt
│   └── .env.example
│
├── docs/                            # Comprehensive Documentation
│   ├── api/
│   │   ├── api_reference.md         # Full REST API specification
│   │   └── postman_collection.json  # Ready-to-import Postman collection
│   ├── backend/
│   │   └── ml_api_integration.md    # Architecture and integration guide
│   └── model_card.md                # NLP Model Card
│
└── README.md
```

---

## 5. Prerequisites & Environment Setup

### Required Software
1. **Python:** Version 3.11–3.14
2. **Node.js:** Version >= 18.0.0
3. **MongoDB:** Community Server >= 6.0 running locally on port 27017

---

## 6. Local Startup Procedure

Follow this exact sequence to start all services locally:

### Step 1: Ensure MongoDB is Running
```powershell
Get-Service -Name MongoDB
```
If not running:
```powershell
Start-Service -Name MongoDB
```

### Step 2: Start the Python ML Service
In Terminal 1:
```powershell
cd ml-service
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
Verify:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

### Step 3: Start the Node/Express Backend
In Terminal 2:
```powershell
cd backend
npm install
npm run dev
```
Verify:
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/api/health"
```

### Step 4: Perform a Live Test Prediction
```powershell
$body = @{ text = "Fantastic, another three-hour meeting on a Friday afternoon." } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/api/predict" -Method Post -Body $body -ContentType "application/json"
```

---

## 7. Testing & Verification

### Running Python ML Service Tests
```bash
cd ml-service
pytest tests/ -v
```
- **Result:** **210 passed** (100% pass rate) covering text preprocessing, artifact validation, predictor logic, and FastAPI endpoints.

### Running Node.js Backend Tests
```bash
cd backend
npm test
```
- **Result:** **34 passed** across 5 test suites (100% pass rate) covering request validation, ML client resilience, routes, and live End-to-End integration.

### Running Latency Benchmark
```bash
cd backend
node scripts/benchmark_latency.js
```
- **Round-trip HTTP Latency:** ~2.27 ms
- **Model Inference Latency:** ~0.50 ms

---

## 8. Next Phase: Phase 12

Phase 12 will implement the React frontend web application consuming `POST /api/predict` and `GET /api/predictions`.
