# Machine Learning API & Node/Express Backend Integration Specification

**Document Version:** 1.0.0  
**Phase:** Phase 11 — ML API + Node/Express Backend Integration  
**Project:** MCA Final-Year NLP Project: Sarcasm & Irony Detection System

---

## 1. System Architecture

The project employs a decoupled multi-tier architecture that separates user-facing application management from specialized machine learning inference:

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

## 2. Why Two Backends?

### Why Node.js + Express for Application Logic?
- **Event-Driven I/O & Concurrency:** Ideal for high-throughput HTTP handling, client routing, and concurrent network requests.
- **Application Responsibilities:** Manages database transactions, user session lifecycles, request sanitization, rate limiting, and CORS headers.
- **Frontend Synergy:** Shares JSON-first data structures and JavaScript paradigms with the impending React frontend (Phase 12).

### Why Python + FastAPI for ML Inference?
- **Native ML Ecosystem:** Directly utilizes scientific computing libraries (`scikit-learn`, `scipy`, `numpy`, `joblib`) without unnatural cross-compilation or heavy WASM shims.
- **Model Isolation:** Keeps model memory footprints, matrix vectorizations, and numerical computations separate from the web application loop.
- **Asynchronous & High-Performance:** FastAPI built on Starlette and Uvicorn delivers microsecond-level latency overheads with automatic OpenAPI validation.

---

## 3. End-to-End Request Flow

When an inference request is initiated:

```
User / React UI
   │
   ▼
[POST /api/predict]
   │
Express Middleware (Helmet, CORS, RateLimiter)
   │
Validation Middleware (Empty/Whitespace checks, 5000 character limit)
   │
mlService.js (HTTP Client with timeout & circuit protection)
   │
   ▼
[POST /predict] on Python FastAPI (Port 8000)
   │
SarcasmIronyPredictor (Phase 10 Production Loader)
   │
   ├─► Preprocessing (Text validation & normalization)
   ├─► BoW CountVectorizer (Transform tokenized words)
   ├─► LinearSVC One-vs-Rest (Decision function)
   └─► Softmax Calibration (Generates probabilities across 4 classes)
   │
FastAPI returns JSON with predicted class, probabilities, latency
   │
   ▼
mlService.js validates API contract
   │
predictionController.js persists record to MongoDB (if available)
   │
   ▼
Node/Express returns standardized payload to React Client
```

---

## 4. Resilience & Graceful Degradation

### MongoDB Failure (Section 57 & 59)
- Prediction inference is treated independently from persistence.
- If MongoDB is disconnected or encounters a connection drop, the core ML inference request **still succeeds** and returns predictions immediately with `history_saved: false`.
- Prediction history failures do not crash the user experience.

### Python ML Service Failure (Section 31 & 58)
- If the Python ML service is stopped or unreachable, Node catches the `ECONNREFUSED` error and immediately responds with `HTTP 503 Service Unavailable` and code `ML_SERVICE_UNAVAILABLE`.
- If the model takes longer than `ML_SERVICE_TIMEOUT_MS` (5000ms), Node aborts and responds with `HTTP 504 Gateway Timeout` and code `ML_SERVICE_TIMEOUT`.
- Incomplete, null, or failed inference attempts are **never written** to MongoDB.

---

## 5. Security and Hardening

1. **Helmet:** Express uses `helmet()` to secure HTTP response headers (XSS filter, HSTS, frameguard, remove `X-Powered-By`).
2. **CORS:** Configured to restrict origin requests in production while permitting the local React development dev server (`http://localhost:5173`).
3. **No Secret Leaks:** Error handlers catch all exceptions and format them into safe messages. Python internal tracebacks and filesystem paths (`C:\`, `/Users/`) are strictly masked.
4. **Rate Limiting:** Enforces maximum request limits per window on `/api/predict` to prevent accidental resource exhaustion.
5. **Privacy in Logging:** Morgan request logs only method, endpoint, status code, and latency. The raw text of user sentences is never written into plaintext web server access logs.

---

## 6. Environment Configurations

### Backend (`backend/.env`)
| Variable | Default Value | Description |
|---|---|---|
| `PORT` | `5000` | Port for the Express server |
| `NODE_ENV` | `development` | Environment mode (`development` / `production` / `test`) |
| `MONGO_URI` | `mongodb://localhost:27017/sarcasm_irony_db` | MongoDB connection URI |
| `ML_SERVICE_URL` | `http://localhost:8000` | Base URL of the internal Python FastAPI service |
| `CORS_ORIGIN` | `http://localhost:5173` | Allowed frontend origin for CORS |
| `ML_SERVICE_TIMEOUT_MS` | `5000` | Timeout in ms for Python ML calls |
| `RATE_LIMIT_WINDOW_MS` | `60000` | Window for rate limiting (1 minute) |
| `RATE_LIMIT_MAX` | `120` | Max requests permitted per window |

### Python ML Service (`ml-service/.env`)
| Variable | Default Value | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Host binding for Uvicorn |
| `PORT` | `8000` | Port for the FastAPI server |
| `LOG_LEVEL` | `info` | Python logging severity level |
| `MODEL_PATH` | `models/final` | Relative or absolute path to model artifact bundle |
| `MODEL_VERSION` | `SIC-v1.0` | Expected model version identifier |
| `VERIFY_CHECKSUMS`| `false` | Verify SHA256 checksums on startup |
| `MAX_TEXT_LENGTH` | `5000` | Max text character limit |

---

## 7. Local Startup Order

To run the complete system locally:

### Step 1: Start MongoDB
Ensure MongoDB is running locally:
```bash
# Service status check
Get-Service -Name MongoDB
```

### Step 2: Start Python ML Service (Terminal 1)
```bash
cd ml-service
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
Verify health:
```bash
curl http://localhost:8000/health
```

### Step 3: Start Node/Express Backend (Terminal 2)
```bash
cd backend
npm install
npm run dev
```
Verify backend health:
```bash
curl http://localhost:5000/api/health
```

### Step 4: Verify Prediction End-to-End
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Fantastic, another three-hour meeting."}'
```

---

## 8. Latency Performance Benchmarks

Measured on local hardware across 50 consecutive inference calls:
- **Node -> Python Round-Trip HTTP Latency:**
  - Average: **2.27 ms**
  - Minimum: **1.78 ms**
  - Maximum: **3.27 ms**
  - P95: **3.15 ms**
- **Python Model Inference (Vectorization + Softmax):**
  - Average: **0.50 ms**
  - Minimum: **0.41 ms**
  - Maximum: **0.72 ms**
  - P95: **0.60 ms**

Total round-trip API overhead is **under 3.5 milliseconds**, comfortably supporting real-time interactive user interfaces in Phase 12.

---

## 9. Transition to Phase 12

In Phase 12, the React frontend will be constructed:
- Frontend will call `POST http://localhost:5000/api/predict` via Axios or fetch.
- Frontend will display predicted class badges, probability breakdown bar charts, and recent query history retrieved via `GET http://localhost:5000/api/predictions`.
- The frontend will **never** communicate directly with the internal Python ML service on port 8000.
