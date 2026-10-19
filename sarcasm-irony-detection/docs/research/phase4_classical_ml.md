# Phase 4 — Bag of Words + Classical Machine Learning

**Project**: Sarcasm & Irony Detection System using NLP  
**Phase**: 4 — BoW + Classical ML Baselines  
**Dataset**: Phase 1 clean datasets (train: 81,359 / test: 5,527)  
**Classes**: Regular (0), Sarcasm (1), Irony (2), Figurative (3)

---

## 1. Objective

Establish strong, reproducible classical NLP + ML baselines using:
- Bag of Words (CountVectorizer)
- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine

These baselines provide the first real classification results and serve as comparison benchmarks for TF-IDF (Phase 5), Bi-LSTM (Phase 6), and BERT (Phase 8).

## 2. Dataset

| Split | Records | Source |
|---|---:|---|
| Training (80%) | 65,087 | train_clean.csv |
| Validation (20%) | 16,272 | train_clean.csv |
| Test | 5,527 | test_clean.csv |

Stratified split with `random_state=42`. Class distribution is approximately balanced (22.8%–26.1% per class).

## 3. Train / Validation / Test Strategy

- **Training**: Used to fit both the BoW vectorizer and the classifier
- **Validation**: Used for hyperparameter selection and preprocessing comparison (transform only)
- **Test**: Used once for final unbiased evaluation after all decisions are made

**Leakage prevention**: CountVectorizer is fitted ONLY on the training split. Validation and test data are transformed using the fitted vocabulary.

## 4. Bag of Words

Three n-gram configurations were tested:

| Configuration | N-gram Range | Vocabulary Size |
|---|---|---:|
| Unigram | (1,1) | 27,344 |
| Unigram + Bigram | (1,2) | 50,000 (capped) |
| Unigram + Bigram + Trigram | (1,3) | 50,000 (capped) |

**CountVectorizer parameters**: `min_df=2`, `max_df=0.95`, `max_features=50000`

**Sparsity**: 0.9995 (99.95% of the matrix entries are zero)

## 5. Logistic Regression

Multiclass Logistic Regression with L2 regularization.

- **Solver**: lbfgs
- **C values tested**: {0.1, 1.0, 10.0}
- **Best C**: 0.1 (selected on validation Macro F1)
- **max_iter**: 1000 (converged without warnings)

## 6. Multinomial Naive Bayes

- **Alpha values tested**: {0.1, 0.5, 1.0}
- **Best alpha**: 1.0 (selected on validation Macro F1)

## 7. Linear SVM

- **C values tested**: {0.1, 1.0, 10.0}
- **Best C**: 0.1 (selected on validation Macro F1)
- **max_iter**: 2000
- **Note**: LinearSVC does not provide probability outputs. No fake confidence scores are reported.

## 8. Hyperparameters

All hyperparameters were selected by maximizing Macro F1 on the validation set. No class weights, SMOTE, or oversampling were used.

| Model | Hyperparameter | Value |
|---|---|---|
| Logistic Regression | C | 0.1 |
| Multinomial NB | alpha | 1.0 |
| Linear SVM | C | 0.1 |

## 9. Experiments

### Core Experiments (BOW-01 to BOW-07)

| ID | Preprocessing | Representation | Model | Accuracy | Macro F1 |
|---|---|---|---|---:|---:|
| BOW-01 | minimal | Unigram | LR | 0.7127 | 0.6466 |
| BOW-02 | minimal | Uni+Bi | LR | 0.6742 | 0.6305 |
| BOW-03 | minimal | Uni+Bi+Tri | LR | 0.6716 | 0.6288 |
| BOW-04 | minimal | Unigram | NB | 0.6403 | 0.6257 |
| BOW-05 | minimal | Uni+Bi | NB | 0.6215 | 0.6092 |
| BOW-06 | minimal | Unigram | SVM | 0.6742 | 0.6337 |
| BOW-07 | minimal | Uni+Bi | SVM | 0.6063 | 0.5960 |

**Key finding**: Unigram consistently outperforms bigram and trigram. Adding n-grams degrades performance — likely due to the 50K feature cap and sparsity.

### Preprocessing Comparison (BOW-08 to BOW-16)

| ID | Preprocessing | Model | Accuracy | Macro F1 |
|---|---|---|---:|---:|
| BOW-08 | lowercase | LR | 0.7127 | 0.6466 |
| BOW-09 | lowercase | NB | 0.6403 | 0.6257 |
| BOW-10 | lowercase | SVM | 0.6742 | 0.6337 |
| BOW-11 | lemmatized | LR | 0.7154 | 0.6487 |
| BOW-12 | lemmatized | NB | 0.6424 | 0.6260 |
| BOW-13 | lemmatized | SVM | 0.6802 | 0.6354 |
| BOW-14 | classical | LR | 0.7181 | **0.6498** |
| BOW-15 | classical | NB | 0.6517 | 0.6331 |
| BOW-16 | classical | SVM | 0.6827 | 0.6371 |

**Key finding**: Classical preprocessing (lowercase + stopword removal + lemmatization) gives the best validation Macro F1 across all models.

## 10. Validation Results

Best validation result: **BOW-14** (LR, classical preprocessing, unigram)
- Accuracy: 0.7181
- Macro F1: 0.6498

Model ranking by validation Macro F1:
1. Logistic Regression: 0.6498 (BOW-14)
2. Linear SVM: 0.6371 (BOW-16)
3. Multinomial NB: 0.6331 (BOW-15)

## 11. Test Results

Final models retrained on full training data (81,359 samples):

| Model | Preprocessing | Accuracy | Macro F1 | Regular F1 | Sarcasm F1 | Irony F1 | Figurative F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| LR | classical | 0.8542 | 0.7053 | 0.9997 | 0.8514 | 0.9072 | 0.0629 |
| NB | classical | 0.7849 | 0.6949 | 0.9641 | 0.7951 | 0.8295 | 0.1908 |
| SVM | classical | **0.8457** | **0.7187** | 0.9995 | 0.8457 | 0.8917 | 0.1378 |

**Best test Macro F1**: Linear SVM (0.7187)

## 12. Confusion Matrices

### LR Test Confusion Matrix

|  | Regular | Sarcasm | Irony | Figurative |
|---|---:|---:|---:|---:|
| **Regular** | 2042 | 1 | 0 | 0 |
| **Sarcasm** | 2 | 1063 | 21 | 5 |
| **Irony** | 0 | 14 | 962 | 5 |
| **Figurative** | 0 | 328 | 157 | 18 |

### NB Test Confusion Matrix

|  | Regular | Sarcasm | Irony | Figurative |
|---|---:|---:|---:|---:|
| **Regular** | 1905 | 92 | 41 | 5 |
| **Sarcasm** | 5 | 944 | 92 | 50 |
| **Irony** | 0 | 37 | 822 | 122 |
| **Figurative** | 0 | 211 | 199 | 84 |

### SVM Test Confusion Matrix

|  | Regular | Sarcasm | Irony | Figurative |
|---|---:|---:|---:|---:|
| **Regular** | 2042 | 1 | 0 | 0 |
| **Sarcasm** | 2 | 1034 | 32 | 23 |
| **Irony** | 0 | 14 | 928 | 39 |
| **Figurative** | 0 | 305 | 142 | 56 |

## 13. Feature-Space Analysis

| Configuration | Vocab Size | Features | Sparsity |
|---|---:|---:|---:|
| Unigram (minimal) | 27,344 | 27,344 | 99.95% |
| Uni+Bi (minimal) | 50,000 | 50,000 | 99.97% |
| Uni+Bi+Tri (minimal) | 50,000 | 50,000 | 99.98% |
| Unigram (lemmatized) | 25,020 | 25,020 | 99.95% |
| Unigram (classical) | 23,689 | 23,689 | 99.95% |

Lemmatization reduces vocabulary by 8.5%. Classical preprocessing reduces it by 13.4%.

## 14. Error Examples

Key misclassification patterns observed:

1. **Figurative → Sarcasm** (most common): Figurative language without explicit ironic markers is often predicted as Sarcasm
2. **Figurative → Irony** (second most common): Many figurative expressions share linguistic patterns with ironic statements
3. **Sarcasm → Irony**: These classes share underlying mechanisms (saying the opposite of what is meant)
4. **Irony → Figurative**: Non-obvious ironic expressions misclassified as figurative

The Figurative class is the hardest to classify — F1 ranges from 0.06 (LR) to 0.19 (NB). This is the primary bottleneck for overall Macro F1.

## 15. Best Classical Baseline

**Best current classical baseline**: Linear SVM (BOW-16)
- Test Macro F1: 0.7187
- Test Accuracy: 0.8457
- Preprocessing: classical (lowercase + stopword removal + lemmatization)
- Representation: Unigram BoW

**This is NOT the final project model.** It is the best current classical baseline for comparison with TF-IDF, Bi-LSTM, and BERT in subsequent phases.

## 16. Limitations

1. **Figurative class failure**: All models struggle severely with the Figurative class (F1 < 0.20). BoW cannot capture figurative language nuances.
2. **No semantic understanding**: BoW treats words as independent features — it cannot capture word order, context, or implied meaning.
3. **Feature cap**: The 50K feature limit may have constrained bigram/trigram performance.
4. **No word embeddings**: Semantic similarity between words is not captured.
5. **Sparse representations**: 99.95%+ sparsity means most feature dimensions are uninformative for any given document.

## 17. Transition to Phase 5

Phase 5 will investigate:
- **TF-IDF** weighting (compared to raw BoW counts)
- **N-gram** representations with TF-IDF (potentially without the 50K cap)
- **Linguistic features** (hashtag count, emoji count, caps ratio, etc.)
- Feature combination strategies

The Phase 4 BoW baselines stored in `results/classical/model_results.csv` will be directly compared against Phase 5 TF-IDF results.

---

## File Structure

```
ml-service/
├── src/classical/
│   ├── __init__.py
│   ├── bow.py              — CountVectorizer wrapper
│   ├── train_lr.py          — Logistic Regression
│   ├── train_nb.py          — Multinomial Naive Bayes
│   ├── train_svm.py         — Linear SVM
│   ├── evaluate.py          — Metrics, confusion matrix, error analysis
│   ├── experiments.py       — Experiment definitions and runner
│   ├── save_artifacts.py    — Model/vectorizer persistence
│   └── run_experiments.py   — Main orchestrator
├── models/classical/
│   ├── bow_logisticregression/
│   ├── bow_multinomialnb/
│   └── bow_linearsvm/
├── results/classical/
│   ├── model_results.csv
│   ├── phase4_summary.json
│   ├── classification_reports/
│   ├── confusion_matrices/
│   └── visualizations/
├── notebooks/
│   └── 04_bow_classical_ml.py
└── docs/research/
    └── phase4_classical_ml.md
```

---

*Phase 4 establishes classical ML baselines. No TF-IDF, no deep learning, no BERT, no deployment. The best classical model (Linear SVM, Macro F1=0.7187) will be compared against more sophisticated approaches in Phase 5+.*
