# Sarcasm & Irony Detection — React Frontend

**Project:** Sarcasm & Irony Detection System using NLP (Phase 12)  
**Curriculum:** MCA Final-Year NLP Project  
**Tech Stack:** React 18, Vite, React Router v6, Axios, Vanilla CSS (Modern AI Dark Theme), Vitest  

---

## 1. Overview

The React Frontend provides the primary user interface for the Sarcasm & Irony Detection System. It allows users to input text, receive real-time four-class predictions (Regular, Sarcasm, Irony, Figurative) with calibrated confidence and probability distributions, view historical predictions persisted in MongoDB, and review model architecture and performance metrics.

### Architecture Compliance
Following strict decoupled multi-tier design (Section 2 & 94):
- React communicates **ONLY** with the Node.js Express backend (`http://localhost:5000/api/*`).
- React **NEVER** calls the Python ML Service (port 8000) directly.
- React **NEVER** connects directly to MongoDB.
- Prediction results are **NEVER** hard-coded; every analysis passes through the live ML pipeline.

```
USER ──► React Frontend (Port 3000) ──► Node.js Express (Port 5000) ──► Python FastAPI (Port 8000) ──► LinearSVC BoW
                                                │
                                                ▼
                                         MongoDB History
```

---

## 2. Directory Structure

```
frontend/
├── public/
│   └── favicon.svg              # Brain / Sparkle NLP Favicon
├── src/
│   ├── components/
│   │   ├── ErrorBoundary.jsx    # Catches runtime render errors gracefully
│   │   ├── ErrorMessage.jsx     # Actionable error alert with retry
│   │   ├── Footer.jsx           # Academic context & technology metadata
│   │   ├── LoadingState.jsx     # Animated spinner with status text
│   │   ├── Navbar.jsx           # Navigation & live system health indicator
│   │   ├── PredictionHistory.jsx# Table / card view with filters & pagination
│   │   ├── PredictionResult.jsx # Classification badges, confidence & disclaimers
│   │   ├── ProbabilityChart.jsx # Horizontal probability distribution bars
│   │   └── TextAnalyzer.jsx     # Text input, char count, example pills & actions
│   │
│   ├── hooks/
│   │   └── usePrediction.js     # Custom hook for prediction state & race guard
│   │
│   ├── pages/
│   │   ├── About.jsx            # Project methodology, dataset, model & tech stack
│   │   ├── Detector.jsx         # Core prediction interface
│   │   ├── History.jsx          # Audit logs from MongoDB
│   │   └── Home.jsx             # Hero, 4-class taxonomy, 5-step NLP pipeline
│   │
│   ├── services/
│   │   └── api.js               # Centralized Axios client for Node.js API
│   │
│   ├── styles/
│   │   └── global.css           # Modern AI dark design system
│   │
│   ├── test/
│   │   └── setup.js             # Vitest DOM setup
│   │
│   ├── tests/                   # Unit & Integration test suites
│   │   ├── Detector.test.jsx
│   │   ├── formatting.test.js
│   │   ├── PredictionHistory.test.jsx
│   │   ├── PredictionResult.test.jsx
│   │   ├── ProbabilityChart.test.jsx
│   │   └── TextAnalyzer.test.jsx
│   │
│   ├── utils/
│   │   └── formatting.js        # Percentages, dates, text truncation, colors
│   │
│   ├── App.jsx                  # Route definitions
│   └── main.jsx                 # React root mount
│
├── .env.example
├── .env
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

---

## 3. Four-Class Taxonomy

| Class | Linguistic Definition | Accent Color |
|---|---|---|
| **Regular** | Literal, direct communication without sarcasm or irony. | `#10b981` (Emerald) |
| **Sarcasm** | Caustic or mocking expression where surface praise conveys contempt. | `#f59e0b` (Amber) |
| **Irony** | Incongruity between expectation and reality, or words conveying the opposite without malice. | `#a855f7` (Purple) |
| **Figurative** | Metaphorical, hyperbolic, or non-literal expressions that do not fit into direct sarcasm. | `#ec4899` (Rose) |

---

## 4. Setup & Running Locally

### Prerequisites
- Node.js >= 18
- Express Backend running on port 5000 (`cd ../backend && npm start`)
- Python ML Service running on port 8000 (`cd ../ml-service && python -m uvicorn src.api.main:app --port 8000`)

### Installation
```bash
cd frontend
npm install
```

### Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default content:
```env
VITE_API_BASE_URL=http://localhost:5000
```

### Development Server
```bash
npm run dev
```
Runs at `http://localhost:3000`.

### Production Build & Preview
```bash
npm run build
npm run preview
```

### Running Tests
```bash
npm test
```
All 14 unit and integration tests are executed using Vitest and Testing Library.
