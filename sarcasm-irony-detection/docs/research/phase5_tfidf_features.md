# Phase 5 — TF-IDF + N-Grams + Feature Engineering

**Project**: Sarcasm & Irony Detection System using NLP  
**Phase**: 5 — TF-IDF + Feature Engineering  
**Dataset**: Phase 1 clean datasets (train: 81,359 / test: 5,527)  
**Classes**: Regular (0), Sarcasm (1), Irony (2), Figurative (3)

---

## 1. Objective

Determine whether TF-IDF, n-gram representations, and linguistic/social-media features improve sarcasm and irony classification compared with Phase 4's Bag-of-Words baselines.

## 2. TF-IDF Theory

TF-IDF (Term Frequency–Inverse Document Frequency) weights terms by:
- **TF**: How often a term appears in a document
- **IDF**: How rare a term is across the corpus
- **sublinear_tf**: Uses `1 + log(tf)` to dampen frequent-term dominance

Unlike raw BoW counts, TF-IDF assigns higher importance to terms that are discriminative within individual documents while penalizing terms that appear everywhere.

## 3. Why TF-IDF Is Appropriate

For sarcasm/irony detection, certain words or phrases may be strong class indicators (e.g., "love" in sarcastic contexts, "totally" in ironic usage). TF-IDF should give these discriminative terms more weight than raw counts.

## 4. N-Gram Methodology

Three configurations tested:

| Configuration | N-gram Range | Vocab Size (classical) |
|---|---|---:|
| Unigram | (1,1) | 23,689 |
| Unigram + Bigram | (1,2) | 50,000 (capped) |
| Unigram + Bigram + Trigram | (1,3) | 50,000 (capped) |

**TfidfVectorizer parameters**: `min_df=2`, `max_df=0.95`, `max_features=50000`, `sublinear_tf=True`, `norm='l2'`

## 5. Feature Engineering

Five controlled feature sets:

| Set | Components | Extra Features |
|---|---|---:|
| A | TF-IDF only | 0 |
| B | TF-IDF + text stats | 3 (word_count, char_count, avg_word_len) |
| C | TF-IDF + social media | 4 (url, mention, hashtag, emoji counts) |
| D | TF-IDF + style | 6 (excl, quest, punct, upper_words, upper_ratio, repeated) |
| E | TF-IDF + all | 13 (B + C + D) |

Features extracted from **raw text** (not preprocessed) to preserve social-media signals.

## 6. Social-Media Features

| Feature | Source |
|---|---|
| url_count | Regex: `https?://\S+` |
| mention_count | Regex: `@\w+` |
| hashtag_count | Regex: `#\w+` |
| emoji_count | `emoji.emoji_count()` |

## 7. Model Configurations

Same three classifiers as Phase 4:
- **Logistic Regression**: C ∈ {0.1, 1.0, 10.0}, solver=lbfgs
- **Multinomial NB**: alpha ∈ {0.1, 0.5, 1.0}
- **Linear SVM**: C ∈ {0.1, 1.0, 10.0}, max_iter=2000

## 8. Experimental Design

Three-stage controlled experiment:

**Stage 1 — Core TF-IDF (8 experiments)**: Classical preprocessing, vary n-gram range and model  
**Stage 2 — Feature sets (5 experiments)**: Best n-gram + model, vary feature set A–E  
**Stage 3 — Preprocessing (4 experiments)**: Best config, vary preprocessing variant

## 9. Leakage Prevention

- TF-IDF vectorizer fitted ONLY on training split
- Feature scaler (MaxAbsScaler) fitted ONLY on training split
- Validation: transform only, no fitting
- Test: evaluated once after all configuration decisions

## 10. Results

### Stage 1 — Core TF-IDF (Validation)

| ID | N-gram | Model | Accuracy | Macro F1 |
|---|---|---|---:|---:|
| TFIDF-01 | (1,1) | LR | 0.7359 | **0.6473** |
| TFIDF-02 | (1,2) | LR | 0.7360 | 0.6468 |
| TFIDF-03 | (1,3) | LR | 0.7359 | 0.6464 |
| TFIDF-04 | (1,1) | NB | 0.5930 | 0.5983 |
| TFIDF-05 | (1,2) | NB | 0.6152 | 0.6020 |
| TFIDF-06 | (1,1) | SVM | 0.7288 | 0.6471 |
| TFIDF-07 | (1,2) | SVM | 0.7289 | 0.6452 |
| TFIDF-08 | (1,3) | SVM | 0.7277 | 0.6447 |

**Finding**: Unigram (1,1) is consistently the best. Higher n-grams do NOT help.

### Stage 2 — Feature Sets (Validation)

| ID | Feature Set | Total Features | Macro F1 |
|---|---|---:|---:|
| FEAT-01 | A (TF-IDF only) | 23,689 | **0.6473** |
| FEAT-02 | B (+ text stats) | 23,692 | 0.6470 |
| FEAT-03 | C (+ social) | 23,693 | 0.6442 |
| FEAT-04 | D (+ style) | 23,695 | 0.6473 |
| FEAT-05 | E (+ all) | 23,702 | 0.6448 |

**Finding**: Feature engineering did NOT improve classification. Feature Set A (text only) tied with D.

### Stage 3 — Preprocessing Comparison (Validation)

| ID | Preprocessing | Vocab Size | Macro F1 |
|---|---|---:|---:|
| PREP-01 | minimal | 27,344 | 0.6469 |
| PREP-02 | lowercase | 27,344 | 0.6469 |
| PREP-03 | lemmatized | 25,020 | 0.6467 |
| PREP-04 | classical | 23,689 | **0.6473** |

**Finding**: Classical preprocessing marginally best, but differences are negligible (<0.001).

## 11. Comparison with Phase 4 BoW

### Test Results

| Phase | Model | Representation | Accuracy | Macro F1 |
|---|---|---|---:|---:|
| Phase 4 | LR | BoW | 0.8542 | 0.7053 |
| Phase 4 | NB | BoW | 0.7849 | 0.6949 |
| **Phase 4** | **SVM** | **BoW** | **0.8457** | **0.7187** |
| Phase 5 | LR | TF-IDF | 0.8583 | 0.6918 |
| Phase 5 | NB | TF-IDF | 0.7574 | 0.6837 |
| Phase 5 | SVM | TF-IDF | 0.8565 | 0.7002 |

**Key finding**: BoW outperformed TF-IDF on test Macro F1 across all three models. The difference for SVM is -0.0185 (BoW is better).

## 12. Feature Analysis

Feature importance extracted from LR and SVM coefficients (top features per class saved to `results/tfidf/important_features/`).

| Metric | Unigram | Uni+Bigram | Uni+Bi+Trigram |
|---|---:|---:|---:|
| Vocab Size | 23,689 | 50,000 | 50,000 |
| Sparsity | 99.95% | 99.98% | 99.98% |
| LR Macro F1 | 0.6473 | 0.6468 | 0.6464 |

## 13. Error Observations

Persistent patterns across both BoW and TF-IDF:
1. **Figurative class failure**: F1 ranges from 0.003 (LR) to 0.23 (NB) — the primary bottleneck
2. **Regular class trivial**: F1 ≈ 0.99+ due to distinctive URL/link patterns
3. **Sarcasm/Irony confusion**: Shared linguistic mechanisms make these hard to separate
4. **NB trades accuracy for Figurative recall**: Only model achieving >0.20 Figurative F1

## 14. Best Traditional NLP Candidates

| Candidate | Phase | Representation | Features | Test Macro F1 |
|---|---|---|---|---:|
| **Best overall** | Phase 4 | BoW | text only | **0.7187** (SVM) |
| Best TF-IDF | Phase 5 | TF-IDF | text only | 0.7002 (SVM) |
| Best Figurative | Phase 5 | TF-IDF | text only | 0.6837 (NB, Fig F1=0.23) |

## 15. Limitations

1. **TF-IDF did not improve over BoW**: L2 normalization + sublinear TF did not provide enough signal lift for this task.
2. **Feature engineering unhelpful**: Social-media and stylistic features (13 total) were overwhelmed by the 23K+ TF-IDF dimensions.
3. **Figurative class remains unsolvable**: All classical approaches fail. Figurative language likely requires semantic understanding beyond bag-of-words/TF-IDF representations.
4. **50K feature cap**: May have constrained bigram/trigram performance.
5. **No cross-validation**: Resource constraints limited to single train/val split.

## 16. Transition to Deep Learning

Phase 5 completes the traditional NLP evaluation. The best classical baseline is:

> **Phase 4 LinearSVM + BoW (unigram, classical preprocessing): Test Macro F1 = 0.7187**

Phase 6 will implement GloVe + Bi-LSTM to test whether:
- Dense word embeddings capture semantic relationships that BoW/TF-IDF cannot
- Sequential models (LSTM) capture word order and context
- The Figurative class benefits from learned representations

---

## File Structure

```
ml-service/
├── src/tfidf/
│   ├── __init__.py
│   ├── vectorizer.py          — TfidfVectorizer wrapper
│   ├── feature_engineering.py — 5 feature sets, sparse hstack
│   ├── feature_analysis.py    — Coefficient-based importance
│   ├── experiments.py         — Experiment definitions + runner
│   └── run_experiments.py     — Main orchestrator
├── models/classical/tfidf/
│   ├── text_only_logisticregression/
│   ├── text_only_multinomialnb/
│   └── text_only_linearsvm/
├── results/tfidf/
│   ├── model_results.csv
│   ├── feature_analysis.csv
│   ├── phase4_vs_phase5.csv
│   ├── phase5_summary.json
│   ├── classification_reports/
│   ├── confusion_matrices/
│   ├── important_features/
│   └── visualizations/
├── notebooks/
│   └── 05_tfidf_feature_engineering.py
└── docs/research/
    └── phase5_tfidf_features.md
```

---

*Phase 5 establishes that TF-IDF and feature engineering do NOT significantly improve over simple BoW for this task. The best traditional NLP model remains Phase 4's LinearSVM + BoW (Macro F1 = 0.7187). Deep learning approaches in Phase 6+ may overcome the Figurative class bottleneck.*
