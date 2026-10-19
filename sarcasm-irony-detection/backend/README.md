# Node.js + Express Application Backend

**Component:** Application Backend Service  
**Project:** Sarcasm & Irony Detection System using NLP (Phase 11)  
**Port:** `5000` (configurable via `PORT`)

---

## Overview

The Node.js backend handles client application requests, performs input validation, coordinates with the Python ML inference service, and stores prediction history in MongoDB.

### Responsibilities
- Public-facing REST APIs under `/api/*`
- Input text sanitization and size constraint enforcement
- Interfacing with Python ML service (`POST /predict` on port 8000)
- Prediction history persistence in MongoDB
- Health monitoring and degradation reporting
- Rate limiting and HTTP security headers

---

## Directory Structure

```
backend/
├── src/
│   ├── config/
│   │   ├── env.js           # Environment variables & defaults
│   │   └── database.js      # Mongoose connection & lifecycle
│   │
│   ├── controllers/
│   │   ├── predictionController.js  # Prediction & history handlers
│   │   └── healthController.js      # Aggregate health check
│   │
│   ├── routes/
│   │   ├── predictionRoutes.js      # /api/predict & /api/predictions
│   │   └── healthRoutes.js          # /api/health
│   │
│   ├── services/
│   │   └── mlService.js             # HTTP client for Python FastAPI
│   │
│   ├── models/
│   │   └── Prediction.js            # Mongoose schema
│   │
│   ├── middleware/
│   │   ├── validation.js            # Input validation
│   │   ├── errorHandler.js          # Centralized error handler
│   │   └── notFound.js              # 404 handler
│   │
│   ├── app.js                       # Express app configuration
│   └── server.js                    # Server startup script
│
├── tests/
│   ├── validation.test.js
│   ├── mlService.test.js
│   ├── predictionRoutes.test.js
│   ├── healthRoutes.test.js
│   └── integration.test.js          # Live E2E tests (Node -> Python -> MongoDB)
│
├── scripts/
│   └── benchmark_latency.js         # Latency benchmarking
│
├── package.json
├── .env.example
└── README.md
```

---

## Getting Started

### Prerequisites
- Node.js >= 18.0.0 (tested on v24.13.0)
- MongoDB Server running locally on port 27017
- Python ML Service running on port 8000

### Installation
```bash
npm install
```

### Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Available variables:
- `PORT`: Server port (default: `5000`)
- `NODE_ENV`: `development`, `test`, or `production`
- `MONGO_URI`: MongoDB connection string (default: `mongodb://localhost:27017/sarcasm_irony_db`)
- `ML_SERVICE_URL`: URL to Python ML service (default: `http://localhost:8000`)
- `CORS_ORIGIN`: Allowed frontend origin (default: `http://localhost:5173`)
- `ML_SERVICE_TIMEOUT_MS`: Timeout for ML inference calls in ms (default: `5000`)
- `RATE_LIMIT_WINDOW_MS`: Rate limit window in ms (default: `60000`)
- `RATE_LIMIT_MAX`: Max requests per window (default: `120`)

### Running the Backend
Development mode:
```bash
npm run dev
```

Production mode:
```bash
npm start
```

### Running Tests
Execute the full Jest test suite (unit and integration tests):
```bash
npm test
```

### Latency Benchmark
Run the performance latency measurement script:
```bash
node scripts/benchmark_latency.js
```
