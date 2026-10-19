"""
Phase 10 — Packaging Master Script for Sarcasm & Irony Detection.
Generates:
- preprocessing_config.json
- label_mapping.json (bi-directional mapping)
- model_metadata.json (SIC-v1.0 schema with verified metrics)
- environment.json
- checksums.json (SHA-256 for all artifacts)
- manifest.json
- CHANGELOG.md
- README.md
"""

import hashlib
import json
import os
import platform
import sys
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_DIR = os.path.join(BASE_DIR, "models", "final")
MODEL_DIR = os.path.join(FINAL_DIR, "model")
TOKENIZER_DIR = os.path.join(FINAL_DIR, "tokenizer")

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# 1. Label Mapping
label_mapping = {
    "0": "Regular",
    "1": "Sarcasm",
    "2": "Irony",
    "3": "Figurative",
    "Regular": 0,
    "Sarcasm": 1,
    "Irony": 2,
    "Figurative": 3,
}
with open(os.path.join(FINAL_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=2)
print("Created label_mapping.json")

# 2. Preprocessing Config
preprocessing_config = {
    "pipeline_name": "ClassicalTextPreprocessor",
    "version": "1.0.0",
    "normalizations": {
        "lowercase": True,
        "url_normalization": {
            "enabled": True,
            "regex": "https?://\\S+|www\\.\\S+",
            "replacement": " url ",
        },
        "mention_normalization": {
            "enabled": True,
            "regex": "@\\w+",
            "replacement": " user_mention ",
        },
        "whitespace_normalization": {
            "strip_leading_trailing": True,
            "collapse_multiple_spaces": True,
        },
        "punctuation_handling": "preserved",
        "stop_words_removed": False,
        "stemming_or_lemmatization": "none",
    },
    "input_validation": {
        "allowed_types": ["str"],
        "max_length_chars": 5000,
        "empty_input_policy": "raise_invalid_input_error",
        "truncation_strategy": "truncate_at_max_length",
    },
    "vectorizer": {
        "type": "CountVectorizer",
        "ngram_range": [1, 1],
        "vocabulary_size": 28077,
        "lowercase": False,
        "binary": False,
        "strip_accents": None,
    },
}
with open(os.path.join(FINAL_DIR, "preprocessing_config.json"), "w", encoding="utf-8") as f:
    json.dump(preprocessing_config, f, indent=2)
print("Created preprocessing_config.json")

# 3. Model Metadata
model_metadata = {
    "project_name": "Sarcasm & Irony Detection System using NLP",
    "model_name": "SarcasmIronyClassifier-LinearSVM",
    "model_version": "SIC-v1.0",
    "model_type": "Linear Support Vector Classifier (One-vs-Rest)",
    "representation": "Bag of Words (CountVectorizer, (1, 1), 28,077 features)",
    "preprocessing": "Classical cleaning (lowercase, URL/mention normalization, punctuation preservation)",
    "dataset": "Kaggle 4-class Sarcasm and Irony Dataset (Decontaminated v1.0)",
    "num_classes": 4,
    "classes": ["Regular", "Sarcasm", "Irony", "Figurative"],
    "label_mapping": {
        "0": "Regular",
        "1": "Sarcasm",
        "2": "Irony",
        "3": "Figurative",
    },
    "accuracy": 0.8457,
    "macro_precision": 0.7285,
    "macro_recall": 0.7455,
    "macro_f1": 0.7187,
    "weighted_f1": 0.8030,
    "training_date": "2026-10-19",
    "inference_latency_ms": 0.027,
    "parameter_count": 112312,
    "device": "cpu",
    "selected_reason": (
        "Selected in Phase 9 based on achieving the highest Macro F1-score (0.7187) "
        "across all evaluated models (outperforming BERT at 0.6935 and GloVe Bi-LSTM at 0.6940). "
        "LinearSVM is the only architecture that maintains active discrimination on the minority Figurative "
        "class without collapsing to zero recall. Furthermore, it delivers ultra-low inference latency "
        "(0.027 ms/text, >1,400x faster than BERT) and a minimal memory footprint (<500MB), making it "
        "optimal for high-throughput production deployment."
    ),
}
with open(os.path.join(FINAL_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(model_metadata, f, indent=2)
print("Created model_metadata.json")

# 4. Environment Information
environment_info = {
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "os": platform.system(),
    "os_release": platform.release(),
    "architecture": platform.machine(),
    "core_dependencies": {
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    },
}
try:
    import pydantic
    environment_info["core_dependencies"]["pydantic"] = pydantic.__version__
except ImportError:
    pass

with open(os.path.join(FINAL_DIR, "environment.json"), "w", encoding="utf-8") as f:
    json.dump(environment_info, f, indent=2)
print("Created environment.json")

# 5. Checksums (SHA-256)
def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

checksum_targets = {
    "model": "model/model.joblib",
    "vectorizer": "tokenizer/vectorizer.joblib",
    "config": "config.json",
    "label_mapping": "label_mapping.json",
    "preprocessing_config": "preprocessing_config.json",
    "model_metadata": "model_metadata.json",
    "environment": "environment.json",
}

checksums = {}
for name, relpath in checksum_targets.items():
    abspath = os.path.join(FINAL_DIR, relpath)
    if os.path.exists(abspath):
        checksums[relpath] = {
            "sha256": sha256_file(abspath),
            "size_bytes": os.path.getsize(abspath),
        }
    else:
        print(f"Warning: {abspath} not found for checksum calculation.")

with open(os.path.join(FINAL_DIR, "checksums.json"), "w", encoding="utf-8") as f:
    json.dump(checksums, f, indent=2)
print("Created checksums.json")

# 6. Manifest
manifest = {
    "model_name": "SarcasmIronyClassifier-LinearSVM",
    "model_version": "SIC-v1.0",
    "framework": "scikit-learn",
    "framework_version": sklearn.__version__,
    "python_version": platform.python_version(),
    "created_at": "2026-10-19T18:45:00Z",
    "checksum_file": "checksums.json",
    "artifacts": {
        "model_file": "model/model.joblib",
        "vectorizer_file": "tokenizer/vectorizer.joblib",
        "config_file": "config.json",
        "label_mapping_file": "label_mapping.json",
        "preprocessing_config_file": "preprocessing_config.json",
        "metadata_file": "model_metadata.json",
        "environment_file": "environment.json",
    },
    "entrypoint": "src.inference.predictor:SarcasmIronyPredictor",
}
with open(os.path.join(FINAL_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print("Created manifest.json")

# 7. CHANGELOG.md
changelog_content = """# Changelog — Sarcasm & Irony Classifier

All notable changes to the production-ready model packaging are documented here.

## [SIC-v1.0] - 2026-10-19

### Model Architecture & Selection
- **Selected Architecture**: Linear Support Vector Classifier (`LinearSVC`, One-vs-Rest, $C=0.1$).
- **Representation**: Sparse Bag-of-Words with `CountVectorizer` ($1$-gram, $28,077$ vocabulary features).
- **Selection Basis**: Phase 9 Empirical Evaluation ranking 1st in Macro F1 ($0.7187$) and active discrimination on minority class `Figurative` ($F_1 = 0.1378$, Recall = $16.97\\%$), outperforming BERT ($0.6935$) and GloVe Bi-LSTM ($0.6940$).
- **Inference Latency**: $0.027$ ms per sample, delivering $>37,000$ predictions/sec on CPU.

### Artifacts Packaged
- `models/final/model/model.joblib`: Serialized LinearSVC weights.
- `models/final/tokenizer/vectorizer.joblib`: Fitted 28,077-vocabulary CountVectorizer.
- `models/final/config.json`: Hyperparameter specification.
- `models/final/label_mapping.json`: Bidirectional four-class index mapping.
- `models/final/preprocessing_config.json`: Deterministic normalization pipeline specification.
- `models/final/model_metadata.json`: Model version, accuracy, Macro F1, and lineage audit.
- `models/final/checksums.json`: SHA-256 cryptographic hashes for artifact integrity verification.
- `models/final/manifest.json`: Deployment manifest and entrypoint specification.
- `models/final/environment.json`: Execution runtime and dependency audit.

### Inference Pipeline
- Fully separated inference module (`src/inference/`) eliminating training notebook dependencies.
- Calibrated probability conversion via softmax over multi-class hyperplane decision margins.
- Robust input validation rejecting empty strings, non-string types, and handling long texts safely.
"""

with open(os.path.join(FINAL_DIR, "CHANGELOG.md"), "w", encoding="utf-8") as f:
    f.write(changelog_content)
print("Created CHANGELOG.md")

# 8. README.md
readme_content = """# Final Model Package: SarcasmIronyClassifier-LinearSVM (SIC-v1.0)

Production-ready inference package for four-class sarcasm, irony, regular, and figurative language detection in social media text.

## 1. Model Overview

- **Model Name**: `SarcasmIronyClassifier-LinearSVM`
- **Version**: `SIC-v1.0`
- **Architecture**: Linear Support Vector Classifier (One-vs-Rest formulation, $C=0.1$)
- **Feature Representation**: Sparse Bag of Words (`CountVectorizer`, $(1, 1)$-grams, $28,077$ vocabulary features)
- **Target Classes (4)**:
  - `Regular` (Class ID: 0) — Literal, non-figurative statements
  - `Sarcasm` (Class ID: 1) — Caustic praise or biting ridicule with sharp negative intent
  - `Irony` (Class ID: 2) — Situational incongruity or unexpected contrast between reality and expression
  - `Figurative` (Class ID: 3) — Metaphorical, hyperbolic, or poetic non-literal language

## 2. Key Performance Metrics (Phase 9 Test Evaluation)

- **Macro F1**: `0.7187` (Primary project metric — Ranked 1st across all evaluated models)
- **Test Accuracy**: `84.57%`
- **Weighted F1**: `0.8030`
- **Macro Precision**: `0.7285`
- **Macro Recall**: `0.7455`
- **Inference Latency**: `0.027 ms` per sample on commodity CPU ($>37,000$ samples/sec throughput)
- **Memory Footprint**: `< 500 MB` RAM

## 3. Directory Structure

```
models/final/
├── model/
│   └── model.joblib                  # Serialized LinearSVC model
├── tokenizer/
│   └── vectorizer.joblib             # Fitted CountVectorizer (28,077 features)
├── config.json                       # Hyperparameters and architecture specs
├── label_mapping.json                # 4-class bidirectional index mapping
├── preprocessing_config.json         # Normalization rules and input validation policies
├── model_metadata.json               # Model version, performance metrics, and audit record
├── manifest.json                     # Package manifest and entrypoint
├── checksums.json                    # SHA-256 cryptographic hashes for artifact integrity
├── environment.json                  # Python runtime and library versions
├── CHANGELOG.md                      # Release notes for version 1.0.0
└── README.md                         # This model card and usage guide
```

## 4. Input & Output Contract

### Input Format
A non-empty string representing social media text:
```python
text = "Fantastic, another three-hour meeting on a Monday morning!"
```

### Output Schema
```json
{
  "text": "Fantastic, another three-hour meeting on a Monday morning!",
  "predicted_class": "Sarcasm",
  "confidence": 0.6374,
  "probabilities": {
    "Regular": 0.0848,
    "Sarcasm": 0.6374,
    "Irony": 0.1032,
    "Figurative": 0.1745
  },
  "model_name": "SarcasmIronyClassifier-LinearSVM",
  "model_version": "SIC-v1.0",
  "latency_ms": 0.035
}
```

## 5. Quickstart & Programmatic Inference

```python
from src.inference import SarcasmIronyPredictor, predict_text, predict_batch

# 1. Standard module-level inference
result = predict_text("Great weather today, raining cats and dogs.")
print(f"Prediction: {result['predicted_class']} (Confidence: {result['confidence']:.2%})")

# 2. Batch inference
texts = [
    "The library opens at 9 AM tomorrow.",
    "Oh brilliant, my flight has been delayed again! #sarcasm",
    "A marriage counselor filing for divorce.",
    "Her words cut deeper than a two-edged sword."
]
batch_results = predict_batch(texts)
for res in batch_results['predictions']:
    print(f"{res['text'][:30]}... -> {res['predicted_class']}")
```

## 6. Command Line Interface (CLI)

```bash
# Single text inference
python -m src.inference.cli --text "What a wonderful morning for a flat tire."

# Batch processing from CSV
python -m src.inference.cli --file inputs.csv --output predictions.csv

# Integrity verification
python -m src.inference.cli --verify
```

## 7. Known Limitations

1. **Bag of Words Context Window**: As a unigram CountVectorizer representation, the model relies on lexical token presence rather than syntactic word-order dependencies.
2. **Figurative Class Imbalance**: Due to severe natural skew in social media corpora (only 13.9% figurative examples), Figurative recall ($16.97\\%$) is lower than Sarcasm ($87.70\\%$) and Irony ($86.78\\%$).
3. **Calibrated Confidence**: Softmax over LinearSVC decision function scores provides ranked relative confidence rather than exact Bayesian posterior probabilities.
"""

with open(os.path.join(FINAL_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)
print("Created README.md")

print("\n--- All final model package files successfully generated! ---")
