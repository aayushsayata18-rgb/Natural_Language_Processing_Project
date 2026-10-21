# Sarcasm & Irony Detection System using NLP

**Academic Level:** MCA Final-Year Project  
**Current Phase:** Phase 12 — React UI Frontend & Full-Stack Multi-Model Deployment  
**Selected Final Model:** `LinearSVM (BoW)` (`bow_linearsvm` / `SIC-v1.0`) (Phase 9 & 10)  
**Validation Macro F1:** `0.7719` | **Accuracy:** `84.57%` | **Decontaminated Benchmark F1:** `0.7187`

---

## 1. Project Overview

This project develops an end-to-end Machine Learning and Natural Language Processing (NLP) system designed to detect and classify nuanced figurative language into four distinct categories:
1. **Regular** (literal, direct, non-sarcastic)
2. **Sarcasm** (irony intended to mock, convey contempt, or express ridicule)
3. **Irony** (situational or incongruous expression without necessary hostility)
4. **Figurative** (metaphorical, hyperbolic, or creative non-literal language)

The project systematically evolved through:
- **Phase 1–3:** Problem formulation, data acquisition from Kaggle, Exploratory Data Analysis, and preprocessing.
- **Phase 4–8:** Benchmarking across 5 distinct NLP paradigms:
  1. *Phase 4:* Classical Bag of Words (BoW) + LinearSVM
  2. *Phase 5:* Statistical TF-IDF + LinearSVM
  3. *Phase 6:* Pretrained GloVe (100d) + Bidirectional LSTM
  4. *Phase 7:* Subword FastText (300d) + Bi-LSTM with Additive Self-Attention
  5. *Phase 8:* Contextual Transformer (`bert-base-uncased`) fine-tuned end-to-end
- **Phase 9:** Scientific model comparison and selection based on generalization, Macro F1, and computational efficiency. LinearSVC One-vs-Rest on Bag-of-Words was selected as the champion architecture.
- **Phase 10:** Model artifact packaging, checksum validation, serialization, and standalone Python inference engine.
- **Phase 11:** FastAPI Python ML microservice integration with a Node.js + Express application backend and MongoDB database.
- **Phase 12:** React 18 + Vite dark-themed web application with real-time inference, model switcher for all 5 models, probability distributions, and MongoDB history.

---

## 2. Dataset & Corpus Details

- **Dataset Name:** [Tweets with Sarcasm and Irony](https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony)
- **Source Platform:** Kaggle
- **Curated By:** Nikhil John K
- **Benchmark Foundation:** Grounded in the standardized SemEval-2018 Task 3 Subtask B benchmark for Twitter figurative language classification.

### Corpus Summary:
- **Raw Volume:** 89,536 tweets (81,408 train / 8,128 test).
- **Leakage Decontamination:** In Phase 1, an audit detected 2,592 cross-split overlapping records with conflicting labels. The test set was decontaminated down to **5,527 clean benchmark samples**, and the training set was curated to **81,359 clean samples**.
- **Four-Class Test Distribution (5,527 Tweets):**
  - **Regular:** 1,858 samples (33.6%)
  - **Sarcasm:** 1,602 samples (29.0%)
  - **Irony:** 1,301 samples (23.5%)
  - **Figurative:** 766 samples (13.9%)

---

## 3. Cross-Paradigm Benchmark: The 5 Evaluated Models

All 5 models are actively hosted in the ML microservice and can be selected dynamically in the frontend or via API:

| Model ID | Architecture | Phase | Representation | Parameters | Latency (CPU) | Accuracy | Macro F1 | Figurative F1 | Status |
|---|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `bow_linearsvm` | **LinearSVM (BoW)** | Phase 4 | CountVectorizer (n-grams 1-2) | 112,312 | **0.027 ms** | 84.57% | **0.7187** | **0.1378** | ⭐ **Champion Model** |
| `tfidf_linearsvm` | **LinearSVM (TF-IDF)** | Phase 5 | Sublinear TF-IDF | 112,312 | **0.014 ms** | 85.65% | 0.7002 | 0.0421 | Deployed & Active |
| `glove_bilstm` | **GloVe + Bi-LSTM** | Phase 6 | 100d Pretrained GloVe | 3,197,636 | 0.041 ms | **86.12%** | 0.6940 | 0.0026 | Deployed & Active |
| `fasttext_attention` | **FastText + BiLSTM + Attn** | Phase 7 | 300d Subword + Self-Attention | 9,204,828 | 0.098 ms | 86.03% | 0.6928 | 0.0000 | Deployed & Active |
| `bert_transformer` | **BERT Contextual** | Phase 8 | `bert-base-uncased` (12 layers, 768d) | 109,485,316 | 39.33 ms | **86.12%** | 0.6935 | 0.0000 | Deployed & Active |

---

## 4. System Architecture

```
                    ┌────────────────────────────┐
                    │    React Frontend (UI)     │
                    │         Port: 3000         │
                    │  (Dynamic Model Selection) │
                    └─────────────┬──────────────┘
                                  │ HTTP / JSON
                                  ▼
                    ┌────────────────────────────┐
                    │  Node.js + Express Backend │
                    │         Port: 5000         │
                    │ (Rate Limit, History, Log) │
                    └───────┬────────────┬───────┘
                            │            │
             Inference HTTP │            │ History Audit Logs
             (Internal Only)│            │
                            ▼            ▼
                ┌──────────────────┐ ┌───────────────┐
                │  Python FastAPI  │ │    MongoDB    │
                │    ML Service    │ │   Database    │
                │    Port: 8000    │ │  Port: 27017  │
                └────────┬─────────┘ └───────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │   Multi-Model Inference Engine    │
        ├───────────────────────────────────┤
        │ 1. LinearSVM (BoW) [Champion]     │
        │ 2. LinearSVM (TF-IDF)             │
        │ 3. GloVe + Bi-LSTM                │
        │ 4. FastText + Bi-LSTM + Attention │
        │ 5. BERT Contextual Transformer    │
        └───────────────────────────────────┘
```

---

## 5. Technology Stack

| Layer | Technology | Role |
|---|---|---|
| **Frontend** | React 18, Vite, React Router v6, Axios, Vanilla CSS | Modern dark AI UI, model selector, probability chart, history viewer (Port 3000) |
| **Backend** | Node.js (v24), Express, Helmet, Morgan, Mongoose | API gateway, request validation, rate limiting, and database abstraction (Port 5000) |
| **ML Service** | Python 3.14, FastAPI, Uvicorn, Scikit-learn, PyTorch | Dedicated stateless microservice hosting the 5-model inference engine (Port 8000) |
| **Database** | MongoDB Community Server (v8.2) | Stores prediction audit records, latency metrics, and label history |
| **Dataset** | Kaggle ("Tweets with Sarcasm and Irony") | 89,536 tweets curated for 4-class multi-class pragmatic classification |

---

## 6. Repository Structure

```
sarcasm-irony-detection/
├── frontend/                        # React 18 + Vite UI Tier (Phase 12)
│   ├── src/components/              # TextAnalyzer, PredictionResult, ProbabilityChart, etc.
│   ├── src/pages/                   # Detector, History, About, Home
│   ├── src/services/                # Centralized Axios client
│   └── package.json                 # Port 3000
│
├── backend/                         # Node.js + Express Backend Tier (Phase 11)
│   ├── src/config/                  # DB connection and env variables
│   ├── src/controllers/             # Prediction & health controllers
│   ├── src/middleware/              # Validation, rate limiting, error handlers
│   ├── src/models/                  # Mongoose Prediction schema
│   ├── src/routes/                  # API routes (/api/predict, /api/predictions, /api/health)
│   ├── src/services/                # Axios ML service client
│   ├── tests/                       # Jest test suites
│   ├── package.json                 # Port 5000
│   └── .env.example
│
├── ml-service/                      # Python ML Inference Microservice Tier (Phase 10 & 11)
│   ├── models/                      # Serialized model bundles for all 5 paradigms
│   ├── src/api/                     # FastAPI app, config, and schemas
│   ├── src/inference/               # Multi-model inference engine & preprocessor
│   ├── tests/                       # Pytest suites (210 tests)
│   ├── requirements.txt             # Port 8000
│   └── .env.example
│
├── docs/                            # Comprehensive Documentation
│   ├── api/api_reference.md         # Full REST API specification
│   ├── backend/ml_api_integration.md# Multi-tier architectural guide
│   └── model_card.md                # NLP Model Card
│
└── README.md
```

---

## 7. Prerequisites & Environment Setup

### Required Software
1. **Python:** Version >= 3.11 (tested on 3.14)
2. **Node.js:** Version >= 18.0.0 (tested on v24)
3. **MongoDB:** Community Server >= 6.0 running locally on port 27017

---

## 8. Local Startup Procedure

Follow this exact sequence to start all services locally:

### Step 1: Ensure MongoDB is Running
```powershell
Get-Service -Name MongoDB
# If not running: Start-Service -Name MongoDB
```

### Step 2: Start the Python ML Service (Port 8000)
In Terminal 1:
```powershell
cd ml-service
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
Verify:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

### Step 3: Start the Node/Express Backend (Port 5000)
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

### Step 4: Start the React Frontend (Port 3000)
In Terminal 3:
```powershell
cd frontend
npm install
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 9. API Inference Examples

### Run Prediction on Champion Model:
```powershell
$body = @{
    text = "Fantastic, another three-hour meeting on a Friday afternoon."
    model = "bow_linearsvm"
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/api/predict" -Method Post -Body $body -ContentType "application/json"
```

### Run Prediction on BERT Contextual Transformer:
```powershell
$body = @{
    text = "I just love when the internet disconnects during a final exam."
    model = "bert_transformer"
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/api/predict" -Method Post -Body $body -ContentType "application/json"
```

---

## 10. Testing & Verification

### Python ML Service Tests
```bash
cd ml-service
pytest tests/ -v
```
- **Result:** **210 passed** (100% pass rate) covering text preprocessing, artifact validation, predictor logic, and FastAPI endpoints.

### Node.js Backend Tests
```bash
cd backend
npm test
```
- **Result:** **34 passed** across 5 test suites covering validation, ML client resilience, routes, and live End-to-End integration.

### Frontend Component Tests
```bash
cd frontend
npm test
```
- **Result:** **14 passed** across 6 test suites covering UI rendering, form submission, and state hooks.
