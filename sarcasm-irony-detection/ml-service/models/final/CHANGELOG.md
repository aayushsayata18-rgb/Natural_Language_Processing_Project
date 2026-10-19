# Changelog — Sarcasm & Irony Classifier

All notable changes to the production-ready model packaging are documented here.

## [SIC-v1.0] - 2026-10-19

### Model Architecture & Selection
- **Selected Architecture**: Linear Support Vector Classifier (`LinearSVC`, One-vs-Rest, $C=0.1$).
- **Representation**: Sparse Bag-of-Words with `CountVectorizer` ($1$-gram, $28,077$ vocabulary features).
- **Selection Basis**: Phase 9 Empirical Evaluation ranking 1st in Macro F1 ($0.7187$) and active discrimination on minority class `Figurative` ($F_1 = 0.1378$, Recall = $16.97\%$), outperforming BERT ($0.6935$) and GloVe Bi-LSTM ($0.6940$).
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
