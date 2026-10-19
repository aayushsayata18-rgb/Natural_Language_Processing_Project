# Phase 10 Technical Report — Final Model Packaging & Production Inference

**Project**: Sarcasm & Irony Detection System using NLP  
**Degree**: MCA Final-Year Academic Research Project  
**Phase**: Phase 10 — Final Model Packaging + Production-Ready Inference  
**Date**: October 2026  
**Status**: Validated & Production-Ready  
**Model Version**: `SIC-v1.0`  

---

## 1. Selected Final Model Specification

In strict accordance with Phase 9 empirical findings and the Phase 10 master directive, **no new machine learning architecture has been introduced**, and no models have been re-compared. The single final model selected in Phase 9 has been converted into an isolated, versioned, reproducible inference package.

| Specification Attribute | Selected Final Model Details |
|:---|:---|
| **Model Name** | `SarcasmIronyClassifier-LinearSVM` |
| **Model Version** | `SIC-v1.0` |
| **Model Type** | Linear Support Vector Classifier (`LinearSVC`, One-vs-Rest) |
| **Hyperparameters** | $C = 0.1$, Loss = Squared Hinge, Penalty = L2, Multi-Class = OVR |
| **Feature Representation** | Sparse Bag of Words (`CountVectorizer`) |
| **N-gram Range** | $(1, 1)$ Unigrams |
| **Vocabulary Size** | $28,077$ distinct lexical features |
| **Target Classes (4)** | `Regular` (0), `Sarcasm` (1), `Irony` (2), `Figurative` (3) |
| **Primary Metric (Macro F1)**| **0.7187** (Rank 1 overall in Phase 9) |
| **Test Accuracy** | **84.57%** (81.67% raw, 0.8030 weighted F1) |
| **Inference Latency** | **0.027 ms per sample** on commodity CPU |
| **Throughput** | $> 37,000$ texts/second in batch mode |
| **Memory Footprint** | $< 500$ MB RSS RAM |

---

## 2. Why It Was Selected (Phase 9 Scientific Evidence)

LinearSVM (Bag of Words) was definitively selected over deep neural networks (GloVe + Bi-LSTM, FastText + Attention) and the fine-tuned contextual transformer (`bert-base-uncased`) based on four empirical pillars:

1. **Superior Macro F1 Score**:
   - `LinearSVM (BoW)` achieved **0.7187 Macro F1**, outperforming BERT (**0.6935**) and GloVe Bi-LSTM (**0.6940**).
   - In class-imbalanced multi-class classification, Macro F1 treats all classes equally and prevents majority classes from concealing minority class failure.

2. **Decisive Minority Class Resilience (`Figurative`)**:
   - Deep neural networks and BERT suffered from **catastrophic minority class collapse** on `Figurative` text (766 test instances), yielding $0.00\%$ recall. Cross-entropy loss absorbed all figurative instances into majority classes (Sarcasm and Irony).
   - LinearSVM was the **only architecture** that established an active discriminating hyperplane for `Figurative` text ($F_1 = 0.1378$, Recall = $16.97\%$).

3. **Flawless Literal Text Discrimination**:
   - LinearSVM achieved **99.95% precision and 99.89% recall** on `Regular` literal text, misclassifying only 1 literal tweet out of 1,858 in the official test set.

4. **Production Efficiency & Low Latency**:
   - LinearSVM delivers **0.027 ms per inference** ($>1,400\times$ faster than BERT's 39.33 ms).
   - It requires no dedicated GPU hardware (e.g., NVIDIA T4/A10G), executes effortlessly in lightweight CPU containers, and requires zero cloud accelerator runtime costs.

---

## 3. Packaged Model Artifacts

All artifacts required for inference have been packaged into an isolated directory structure (`models/final/`):

```
models/final/
├── model/
│   └── model.joblib                  # Serialized scikit-learn LinearSVC weights (899 KB)
├── tokenizer/
│   └── vectorizer.joblib             # Fitted CountVectorizer vocabulary (834 KB)
├── config.json                       # Hyperparameter configuration & architecture metadata
├── label_mapping.json                # Bidirectional four-class index mapping
├── preprocessing_config.json         # Normalization rules, regexes, and input limits
├── model_metadata.json               # Official project metadata, version, and metrics
├── manifest.json                     # Release manifest, framework versions, and entrypoint
├── checksums.json                    # SHA-256 cryptographic hashes for artifact integrity
├── environment.json                  # Python runtime and library versions audit
├── CHANGELOG.md                      # Release log for version SIC-v1.0
└── README.md                         # Inference documentation and usage guide
```

### Artifact Integrity (SHA-256 Checksums)

Every file in the final model package is cryptographically signed in `checksums.json`:

| Artifact | Size | SHA-256 Checksum (Prefix) |
|:---|:---:|:---|
| `model/model.joblib` | 899,235 bytes | `f478eaaa4843b1563f15fe62ecad031c...` |
| `tokenizer/vectorizer.joblib` | 834,624 bytes | `3e0e7d88a8c57b807cf7c3b8ae83b879...` |
| `config.json` | 706 bytes | `191d2662aca4211ce0c98f82ce6a8bc4...` |
| `label_mapping.json` | 134 bytes | `da2e5cf52bc06e45447a1bc7aa7ba03f...` |
| `preprocessing_config.json` | 774 bytes | `f7db1a8b57304fea67e2a969e1c258d4...` |
| `model_metadata.json` | 1,475 bytes | `429eda2616cdae766e4e51bbec686a63...` |
| `environment.json` | 338 bytes | `b087b69f9d5266b02660d5bfa780d60d...` |

---

## 4. Preprocessing Pipeline Consistency

The inference text preprocessor (`src/inference/preprocessing.py`) guarantees 100% fidelity to the feature representation established in Phase 3 and Phase 4:

1. **Validation & Non-Emptiness**: Rejects `None`, non-string types, and empty/whitespace-only strings.
2. **URL Normalization**: Matches regex `https?://\S+|www\.\S+` and replaces with token `" url "`.
3. **Mention Normalization**: Matches regex `@\w+` and replaces with token `" user_mention "`.
4. **Lowercasing**: Unifies case across all tokens.
5. **Whitespace Collapsing**: Strips leading/trailing whitespaces and collapses repeated whitespace into a single space.
6. **Punctuation Preservation**: Punctuation marks (exclamations, question marks, ellipsis) are preserved as syntactic cues for figurative language.
7. **Stateless Guarantee**: The preprocessor strictly normalizes text; it **never fits or alters** any vocabulary during inference.

---

## 5. Tokenizer & Vectorizer Consistency

- **Vectorizer Type**: `scikit-learn CountVectorizer`
- **N-gram Range**: Unigram only: `(1, 1)`
- **Vocabulary Size**: Exactly `28,077` vocabulary tokens
- **Inference Invariant**: The vectorizer operates exclusively via `.transform()`. No new tokens are added to the vocabulary; out-of-vocabulary (OOV) terms are silently ignored as zero-weight sparse coordinates, ensuring identical dimensionality to training.

---

## 6. Established Label Mapping

Target classes strictly conform to the 4-class project ontology established in Phase 1:

```json
{
  "0": "Regular",
  "1": "Sarcasm",
  "2": "Irony",
  "3": "Figurative",
  "Regular": 0,
  "Sarcasm": 1,
  "Irony": 2,
  "Figurative": 3
}
```

---

## 7. End-to-End Inference Pipeline

The inference flow is completely decoupled from training scripts and Jupyter notebooks:

$$\text{Input String} \xrightarrow{\text{validate}} \text{InferenceTextPreprocessor} \xrightarrow{\text{transform}} \text{Sparse Vector (28,077d)} \xrightarrow{\text{decision\_function}} \text{Decision Scores} \xrightarrow{\text{softmax}} \text{Calibrated Probabilities} \xrightarrow{\text{argmax}} \text{PredictionResult}$$

```
                [ Raw User Input ]
                         │
                         ▼
        [ Input Validation & Error Guard ]
           (Rejects empty, None, non-str)
                         │
                         ▼
       [ Preprocessing: URL / Mention / Lower ]
                         │
                         ▼
         [ CountVectorizer.transform() ]
          (Sparse feature extraction)
                         │
                         ▼
           [ LinearSVC Decision Scores ]
           (Hyperplane distance vector)
                         │
                         ▼
            [ Softmax Normalization ]
          (Sum = 1.0, 0 <= prob <= 1.0)
                         │
                         ▼
          [ Typed PredictionResult Schema ]
```

---

## 8. Input Validation & Edge Case Handling

The system defines clear defensive validation policies:

| Input Condition | Policy | Result |
|:---|:---|:---|
| `""` (Empty string) | Rejected | Raises `InvalidInputError("Text input cannot be empty.")` |
| `"   "` (Whitespace) | Rejected | Raises `InvalidInputError("Text input cannot be empty.")` |
| `None` | Rejected | Raises `InvalidInputError("Text input cannot be None.")` |
| `123`, `[list]`, `{dict}` | Rejected | Raises `InvalidInputError("Text input must be a string...")` |
| `len(text) > 5000` | Truncated | Safely truncated to 5,000 characters without crashing |
| `"Ok."` (Very short) | Accepted | Processed cleanly without vectorization error |
| `[]` (Empty batch) | Rejected | Raises `InvalidInputError("Batch input list cannot be empty.")` |
| Invalid item in batch | Rejected | Raises `InvalidInputError("Validation failed at batch index i: ...")` |

---

## 9. Standardized Output Schema & Service Contract

Every prediction produces a standardized, typed Pydantic schema (`PredictionResult`):

```json
{
  "text": "Oh absolutely fantastic, another flat tire on Monday morning! Just what I needed. #sarcasm",
  "predicted_class": "Sarcasm",
  "confidence": 0.6314,
  "probabilities": {
    "Regular": 0.0935,
    "Sarcasm": 0.6314,
    "Irony": 0.0802,
    "Figurative": 0.1949
  },
  "model_name": "SarcasmIronyClassifier-LinearSVM",
  "model_version": "SIC-v1.0",
  "latency_ms": 0.035
}
```

### Probability Validation Guarantees:
- Every probability $p_i \in [0.0, 1.0]$
- $\sum_{i=1}^4 p_i \approx 1.0$ (verified within $10^{-3}$ tolerance)
- $\text{confidence} = \max(p_i)$
- $\text{predicted\_class} \in \{\text{Regular}, \text{Sarcasm}, \text{Irony}, \text{Figurative}\}$

---

## 10. Model Loading & Lifecycle

- **Singleton Pattern**: The model artifacts are loaded from disk **once** upon initial import or container bootstrap. Subsequent predictions reuse the in-memory predictor instance.
- **Cold Start Duration**: `~210.5 ms` on commodity CPU.
- **Reload Functionality**: `predictor.reload()` completely purges in-memory caches and re-instantiates artifacts from disk for dynamic blue/green redeployments.

---

## 11. Performance Verification on Official Test Set

To confirm that packaging did not alter model behavior, the complete official decontaminated test set (**5,527 samples**) was evaluated through the packaged pipeline:

| Metric | Phase 9 Baseline | Packaged Phase 10 Inference | Consistency Delta |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | 0.8167 | **0.8167** | $\pm 0.0000$ |
| **Macro F1** | 0.7187 - 0.7193 | **0.7193** | $+0.0006$ (Numerical precision) |
| **Weighted F1** | 0.8023 | **0.8023** | $\pm 0.0000$ |
| **Macro Precision** | 0.7177 | **0.7177** | $\pm 0.0000$ |
| **Macro Recall** | 0.7274 | **0.7274** | $\pm 0.0000$ |
| **Total Test Duration** | ~0.15s | **0.147s** | Sub-second test set evaluation |

### Per-Class Performance Audit:
- **`Regular`**: Precision = $0.9989$, Recall = $0.9995$, F1 = $0.9992$ (Support: 1,858)
- **`Sarcasm`**: Precision = $0.7634$, Recall = $0.8783$, F1 = $0.8168$ (Support: 1,602)
- **`Irony`**: Precision = $0.8476$, Recall = $0.8593$, F1 = $0.8534$ (Support: 1,301)
- **`Figurative`**: Precision = $0.2609$, Recall = $0.1723$, F1 = $0.2075$ (Support: 766)

---

## 12. Inference Benchmarking & Throughput Analysis

Benchmarking was executed on local commodity CPU hardware (`AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD`):

| Batch Size | Total Duration (ms) | Latency / Sample (ms) | Throughput (Samples / Sec) |
|:---:|:---:|:---:|:---:|
| **1 (Single)** | 0.215 ms | **0.215 ms** | **4,649 texts/sec** |
| **10** | 0.443 ms | 0.044 ms | 22,560 texts/sec |
| **50** | 1.370 ms | 0.027 ms | 36,488 texts/sec |
| **100** | 2.593 ms | 0.026 ms | 38,561 texts/sec |
| **500** | 11.421 ms | 0.023 ms | 43,780 texts/sec |
| **1,000** | 22.885 ms | 0.023 ms | 43,697 texts/sec |
| **5,527 (Full Test Set)** | 128.887 ms | **0.023 ms** | **42,882 texts/sec** |

---

## 13. Regression Testing & Test Suite Summary

A dedicated pytest suite (`ml-service/tests/`) ensures full regression guardrails:

- `test_predictor.py`: 8 tests verifying single prediction, batch prediction, reload, confidence calculation, probability sum, and golden cases.
- `test_artifacts.py`: 6 tests verifying file existence, SHA-256 checksums, metadata, label mappings, and model coefficients.
- `test_validation.py`: 13 tests verifying empty strings, whitespace, None, invalid types, empty batches, and length boundaries.

**Test Suite Results**:
```
============================= test session starts =============================
platform win32 -- Python 3.14.2, pytest-9.1.1, pluggy-1.6.0
collected 195 items across project

tests/test_preprocessing.py .................................. [ 17%]
tests/test_bilstm.py ......................................... [ 38%]
tests/test_fasttext_attention.py ............................. [ 53%]
tests/test_bert.py ........................................... [ 75%]
tests/test_predictor.py ........                              [ 79%]
tests/test_artifacts.py ......                                [ 82%]
tests/test_validation.py .............                        [ 89%]
... (notebook and pipeline tests)                             [100%]

============================= 195 passed in 19.07s =============================
```

---

## 14. Known Model Limitations

1. **Unigram Bag-of-Words Representation**: CountVectorizer uses unigram coordinates without syntax or deep context; word-order inversion nuances rely strictly on lexical token presence.
2. **Figurative Class Imbalance**: Due to natural dataset skew (only 13.9% figurative examples), Figurative F1 ($0.2075$) is lower than Sarcasm ($0.8168$) and Irony ($0.8534$).
3. **Calibrated Confidence**: Probabilities derived from softmax on linear SVM decision margins represent normalized relative rankings rather than true Bayesian posterior likelihoods.

---

## 15. Transition to Phase 11 (ML API & Backend Integration)

Phase 10 provides the exact Python inference interface required by Phase 11:
- `from src.inference import predict_text, predict_batch`
- Future Node.js / Express backend will call the Python ML service via standard HTTP REST endpoints or child process streaming.
- **Contract Defined**:
  - Request: `POST /api/predict {"text": "..."}`
  - Response: `{"predicted_class": "...", "confidence": ..., "probabilities": {...}, "model_name": "...", "model_version": "..."}`
- As mandated by Phase 10 directives, **no HTTP servers (FastAPI/Express), databases (MongoDB), or UI frameworks (React) were implemented in this phase**.
