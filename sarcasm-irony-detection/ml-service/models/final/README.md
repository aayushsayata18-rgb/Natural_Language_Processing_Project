# Final Model Package: SarcasmIronyClassifier-LinearSVM (SIC-v1.0)

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
2. **Figurative Class Imbalance**: Due to severe natural skew in social media corpora (only 13.9% figurative examples), Figurative recall ($16.97\%$) is lower than Sarcasm ($87.70\%$) and Irony ($86.78\%$).
3. **Calibrated Confidence**: Softmax over LinearSVC decision function scores provides ranked relative confidence rather than exact Bayesian posterior probabilities.
