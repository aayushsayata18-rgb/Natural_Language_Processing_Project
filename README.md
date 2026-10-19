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
- **Phase 11 (Completed):** FastAPI Python ML microservice integration with a Node.js + Express application backend and MongoDB database.

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
d:/Natural_Language_Processing_Project/
├── sarcasm-irony-detection/
│   ├── backend/                         # Node.js + Express Backend Tier
│   │   ├── src/                         # Config, controllers, middleware, models, routes, services
│   │   ├── tests/                       # Jest test suites (unit + live E2E)
│   │   ├── scripts/                     # Performance latency benchmark
│   │   ├── package.json
│   │   └── .env.example
│   │
│   ├── ml-service/                      # Python ML Inference Microservice Tier
│   │   ├── models/final/                # Deployed model artifacts & metadata
│   │   ├── src/api/                     # FastAPI app, config, schemas
│   │   ├── src/inference/               # ModelLoader, Preprocessor, Predictor
│   │   ├── tests/                       # Pytest suites (210 tests)
│   │   ├── requirements.txt
│   │   └── .env.example
│   │
│   ├── docs/                            # Comprehensive Documentation
│   │   ├── api/
│   │   │   ├── api_reference.md         # Full REST API specification
│   │   │   └── postman_collection.json  # Ready-to-import Postman collection
│   │   ├── backend/
│   │   │   └── ml_api_integration.md    # Architecture and integration guide
│   │   └── model_card.md                # NLP Model Card
│   │
│   └── README.md
│
├── Prompts/                             # Phase-by-phase project master prompts
└── README.md
```

---

## 5. Local Startup Procedure

1. **MongoDB**: Start MongoDB server locally (port 27017).
2. **Python ML Service**:
   ```bash
   cd sarcasm-irony-detection/ml-service
   python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   ```
3. **Node/Express Backend**:
   ```bash
   cd sarcasm-irony-detection/backend
   npm install
   npm run dev
   ```
4. **Health Check**:
   ```bash
   curl http://localhost:5000/api/health
   ```
5. **Predict**:
   ```bash
   curl -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" -d "{\"text\": \"Fantastic, another three-hour meeting.\"}"
   ```
