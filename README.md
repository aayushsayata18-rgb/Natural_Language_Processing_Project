# Sarcasm & Irony Detection System using NLP

**Academic Level:** MCA Final-Year Project  
**Current Phase:** Phase 12 — React UI Frontend & Full-Stack Multi-Model Deployment  
**Selected Champion Model:** `LinearSVM (BoW)` (`bow_linearsvm` / `SIC-v1.0`)  
**Macro F1 Benchmark:** `0.7187` (Decontaminated Benchmark) / `0.7719` (Stratified Validation) | **Accuracy:** `84.57%`

---

## 1. Project Overview

This project develops an end-to-end Machine Learning and Natural Language Processing (NLP) system designed to detect and classify nuanced figurative language into four distinct pragmatic categories:
1. **Regular** (literal, direct, non-sarcastic)
2. **Sarcasm** (irony intended to mock, convey contempt, or express ridicule)
3. **Irony** (situational or linguistic incongruity between expectations and reality)
4. **Figurative** (metaphorical, hyperbolic, or creative non-literal language)

The project systematically evolved through:
- **Phase 1–3:** Problem formulation, data acquisition from Kaggle, Exploratory Data Analysis, text normalization, and decontamination.
- **Phase 4–8:** Cross-paradigm benchmarking across 5 distinct NLP model families (Classical BoW, Statistical TF-IDF, GloVe Bi-LSTM, FastText with Additive Self-Attention, and fine-tuned BERT Transformer).
- **Phase 9:** Scientific model comparison and selection based on generalization, Macro F1, minority-class collapse prevention, and computational efficiency.
- **Phase 10:** Model artifact packaging, checksum validation, serialization, and standalone Python inference engine.
- **Phase 11:** FastAPI Python ML microservice integration with a Node.js + Express application backend and MongoDB database.
- **Phase 12:** React 18 + Vite dark-themed web application allowing interactive real-time text analysis, dynamic selection across all 5 benchmark models, probability distribution charts, and persistent MongoDB audit logs.

---

## 2. Dataset & Corpus Details

- **Dataset Name:** [Tweets with Sarcasm and Irony](https://www.kaggle.com/datasets/nikhiljohnk/tweets-with-sarcasm-and-irony)
- **Source Platform:** Kaggle
- **Curated By:** Nikhil John K
- **Benchmark Foundation:** Grounded in the standardized SemEval-2018 Task 3 Subtask B benchmark for Twitter figurative language classification.

### Corpus Volume & Quality Decontamination:
- **Original Raw Volume:** 89,536 tweets (81,408 train / 8,128 test).
- **Leakage Prevention & Decontamination:** A strict data audit identified 2,592 cross-split overlapping records with conflicting labels. The test set was strictly decontaminated down to **5,527 untouched, leak-free benchmark samples**, while the training corpus was curated to **81,359 clean samples**.
- **Four-Class Test Distribution (5,527 Tweets):**
  - **Regular:** 1,858 samples (33.6%) — Literal statements without sarcasm or irony.
  - **Sarcasm:** 1,602 samples (29.0%) — Surface praise cloaking sharp ridicule or contempt.
  - **Irony:** 1,301 samples (23.5%) — Incongruity between expectation and reality without malice.
  - **Figurative:** 766 samples (13.9%) — Metaphors, hyperbole, and creative non-literal speech.
- **Preprocessing & Normalization:** Standardized `@user` mentions, stripped non-informative `t.co` URLs, unpacked contractions, and decoded UTF-8 emojis into descriptive text tokens.

---

## 3. The 5 Evaluated & Deployed NLP Models

All 5 architectures were evaluated on the identical decontaminated benchmark of 5,527 test tweets and are deployed live in the ML inference microservice, selectable from the React UI dropdown:

| Model ID | Architecture | Phase | Representation & Technique | Parameters | Latency (CPU) | Accuracy | Macro F1 | Figurative F1 | Selection Status |
|---|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `bow_linearsvm` | **LinearSVM (BoW)** | Phase 4 | CountVectorizer n-grams (1, 2) + LinearSVC (One-vs-Rest) | 112,312 | **0.027 ms** | 84.57% | **0.7187** | **0.1378** | ⭐ **Champion Model** |
| `tfidf_linearsvm` | **LinearSVM (TF-IDF)** | Phase 5 | Sublinear TF-IDF + statistical n-grams | 112,312 | **0.014 ms** | 85.65% | 0.7002 | 0.0421 | Deployed (Active) |
| `glove_bilstm` | **GloVe + Bi-LSTM** | Phase 6 | 100d Pretrained GloVe + 2-layer Bidirectional LSTM | 3,197,636 | 0.041 ms | **86.12%** | 0.6940 | 0.0026 | Deployed (Active) |
| `fasttext_attention` | **FastText + BiLSTM + Attn** | Phase 7 | 300d Subword Embeddings + Additive Self-Attention | 9,204,828 | 0.098 ms | 86.03% | 0.6928 | 0.0000 | Deployed (Active) |
| `bert_transformer` | **BERT Contextual** | Phase 8 | Fine-tuned `bert-base-uncased` (12 layers, 768d, 12 heads) | 109,485,316 | 39.33 ms | **86.12%** | 0.6935 | 0.0000 | Deployed (Active) |

### Key Research Findings:
1. **Why LinearSVM Won on Macro F1:** LinearSVM maximizes geometric margins between hyperplanes. In our imbalanced dataset, standard unweighted cross-entropy in deep neural networks flattened the minority *Figurative* class (13.9% support) into the majority classes, whereas LinearSVM preserved active discrimination (16.97% recall, F1=0.1378).
2. **BERT's Semantic Precision:** BERT achieved **zero errors out of 2,903 tweets** between Sarcasm and Irony, proving contextual transformers excel at phrase-level pragmatic boundary disambiguation.
3. **Inference Efficiency:** LinearSVM executes in **0.027 ms/text** (~1,400× faster than BERT on CPU) with 975× fewer parameters, making it the ideal champion model for real-time web deployment.

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
| **Frontend UI** | React 18, Vite, React Router v6, Axios, Vanilla CSS | Single Page Application with model selector, probability bars, and dark theme (Port 3000) |
| **Backend API** | Node.js (v24), Express, Helmet, Morgan, Mongoose | Application gateway, security headers, rate limiting, and MongoDB integration (Port 5000) |
| **ML Service** | Python 3.14, FastAPI, Uvicorn, Pydantic, Scikit-learn, PyTorch | Dedicated stateless microservice hosting multi-model inference engine (Port 8000) |
| **Database** | MongoDB Community Server (v8.2) | Stores user prediction audit trail, latency metrics, and class distribution |
| **Dataset** | Kaggle ("Tweets with Sarcasm and Irony") | Curated corpus of 89,536 tweets benchmarked across 4 figurative language classes |

---

## 6. Repository Structure

```
d:/Natural_Language_Processing_Project/
├── sarcasm-irony-detection/
│   ├── frontend/                        # React 18 + Vite UI (Phase 12)
│   │   ├── src/components/              # TextAnalyzer, PredictionResult, ProbabilityChart, etc.
│   │   ├── src/pages/                   # Detector, History, About, Home
│   │   ├── src/services/                # Centralized Axios API client
│   │   └── package.json                 # Port 3000
│   │
│   ├── backend/                         # Node.js + Express Backend Tier (Phase 11)
│   │   ├── src/config/                  # Database connection and env variables
│   │   ├── src/controllers/             # Prediction and health controllers
│   │   ├── src/middleware/              # Validation, rate limiting, error handlers
│   │   ├── src/models/                  # Mongoose Prediction schema
│   │   ├── src/routes/                  # /api/predict, /api/predictions, /api/health
│   │   └── package.json                 # Port 5000
│   │
│   ├── ml-service/                      # Python ML Inference Microservice Tier (Phase 10 & 11)
│   │   ├── models/                      # Serialized model artifacts for all 5 paradigms
│   │   ├── src/api/                     # FastAPI app, schemas, lifespan config
│   │   ├── src/inference/               # Multi-model inference engine, preprocessor
│   │   ├── tests/                       # Pytest test suite (210 tests)
│   │   └── requirements.txt             # Port 8000
│   │
│   ├── docs/                            # Comprehensive Project Documentation
│   │   ├── api/api_reference.md         # Full REST API specification
│   │   ├── backend/ml_api_integration.md# Multi-tier architectural guide
│   │   └── model_card.md                # NLP Model Card & benchmark metrics
│   │
│   └── README.md
│
├── Prompts/                             # Phase-by-phase project specifications
└── README.md
```

---

## 7. Local Startup Procedure

Follow this sequence to launch all tiers on your local machine:

### Step 1: Ensure MongoDB is Running
```powershell
Get-Service -Name MongoDB
# If stopped: Start-Service -Name MongoDB
```

### Step 2: Start the Python ML Service (Port 8000)
```powershell
cd sarcasm-irony-detection/ml-service
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
*Health verification:* `curl http://127.0.0.1:8000/health`

### Step 3: Start the Node.js Express Backend (Port 5000)
```powershell
cd sarcasm-irony-detection/backend
npm install
npm run dev
```
*Health verification:* `curl http://localhost:5000/api/health`

### Step 4: Start the React Frontend UI (Port 3000)
```powershell
cd sarcasm-irony-detection/frontend
npm install
npm run dev
```
*Access UI in browser:* **[http://localhost:3000](http://localhost:3000)**

---

## 8. Live API Prediction Examples

### Predict using the Champion Model (`bow_linearsvm`):
```powershell
$body = @{
    text = "Fantastic, another three-hour unscheduled meeting on a Friday afternoon."
    model = "bow_linearsvm"
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/api/predict" -Method Post -Body $body -ContentType "application/json"
```

### Predict using BERT Contextual Transformer (`bert_transformer`):
```powershell
$body = @{
    text = "Oh brilliant, the server went down right before the demonstration!"
    model = "bert_transformer"
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/api/predict" -Method Post -Body $body -ContentType "application/json"
```

### Fetch Available Models & Benchmarks:
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/api/models"
```
