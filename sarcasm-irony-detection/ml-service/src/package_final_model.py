"""
Package Final Model Artifacts for Phase 9 and Phase 10 readiness.
"""

import json
import os
import shutil
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "models", "final")
FINAL_SUB_MODEL = os.path.join(FINAL_MODEL_DIR, "model")
FINAL_SUB_TOKENIZER = os.path.join(FINAL_MODEL_DIR, "tokenizer")

os.makedirs(FINAL_SUB_MODEL, exist_ok=True)
os.makedirs(FINAL_SUB_TOKENIZER, exist_ok=True)

# Source artifacts (CAND-01: LinearSVM BoW)
src_model = os.path.join(BASE_DIR, "models", "classical", "bow_linearsvm", "model.joblib")
src_vec = os.path.join(BASE_DIR, "models", "classical", "bow_linearsvm", "vectorizer.joblib")
src_cfg = os.path.join(BASE_DIR, "models", "classical", "bow_linearsvm", "config.json")
src_label_map = os.path.join(BASE_DIR, "data", "processed", "label_mapping.json")

# Copy model and vectorizer
shutil.copy2(src_model, os.path.join(FINAL_SUB_MODEL, "model.joblib"))
shutil.copy2(src_vec, os.path.join(FINAL_SUB_TOKENIZER, "vectorizer.joblib"))

# Write config.json
with open(src_cfg, "r", encoding="utf-8") as f:
    cfg_data = json.load(f)

cfg_data.update({
    "final_model_name": "SarcasmIronyClassifier-LinearSVM-v1",
    "version": "1.0.0",
    "architecture": "Linear Support Vector Classifier (One-vs-Rest, C=0.1)",
    "representation": "Bag of Words (CountVectorizer, ngram_range=(1,1), max_features=28077)",
    "preprocessing": "classical",
    "selected_metric": "Macro F1",
    "selected_score": 0.7187,
})

with open(os.path.join(FINAL_MODEL_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg_data, f, indent=2)

# Write label_mapping.json
with open(src_label_map, "r", encoding="utf-8") as f:
    label_map = json.load(f)

with open(os.path.join(FINAL_MODEL_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(label_map, f, indent=2)

# Write model_metadata.json
model_metadata = {
    "project_name": "Sarcasm & Irony Detection System using NLP",
    "model_name": "SarcasmIronyClassifier-LinearSVM-v1",
    "model_type": "Linear Support Vector Classifier (One-vs-Rest)",
    "representation": "Bag of Words (CountVectorizer, (1, 1), 28,077 features)",
    "preprocessing": "Classical cleaning (lowercase, URL/mention normalization, punctuation preservation)",
    "dataset": "Kaggle 4-class Sarcasm and Irony Dataset (Decontaminated v1.0)",
    "class_mapping": label_map,
    "accuracy": 0.8457,
    "macro_precision": 0.7285,
    "macro_recall": 0.7455,
    "macro_f1": 0.7187,
    "weighted_f1": 0.8030,
    "training_time": 12.4,
    "inference_time": 0.027,
    "parameter_count": 112312,
    "selected_reason": (
        "Selected as the official project model based on achieving the highest Macro F1-score (0.7187) "
        "across all evaluated models (outperforming BERT 0.6935 and GloVe Bi-LSTM 0.6940). "
        "LinearSVM is the only architecture that maintains active discrimination on the minority Figurative "
        "class without collapsing to zero recall. Furthermore, it delivers ultra-low inference latency "
        "(0.027 ms/text, >1,400x faster than BERT) and a minimal memory footprint (<500MB), making it "
        "optimal for high-throughput production deployment."
    ),
    "version": "1.0.0",
}

with open(os.path.join(FINAL_MODEL_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(model_metadata, f, indent=2)

print("Final model packaged successfully into models/final/")
