# Phase 6 — GloVe + Bi-LSTM Deep Learning

**Project**: Sarcasm & Irony Detection System using NLP  
**Phase**: 6 — GloVe + Bi-LSTM  
**Dataset**: Phase 1 clean datasets (train: 81,359 / test: 5,527)  
**Classes**: Regular (0), Sarcasm (1), Irony (2), Figurative (3)  
**Framework**: PyTorch 2.14.0  
**Device**: CPU (GPU auto-detected if available)

---

## 1. Objective

Phase 6 determines whether a neural sequence model (Bi-LSTM) with dense word embeddings (GloVe) improves upon the traditional NLP approaches developed in:

- **Phase 4**: Bag of Words + Classical ML (best: LinearSVM, Macro F1 = 0.7187)
- **Phase 5**: TF-IDF + N-grams + Feature Engineering (best: LinearSVM, Macro F1 = 0.7002)

The key research question: *"Does sequence modeling with GloVe + Bi-LSTM provide an advantage over traditional sparse text representations?"*

## 2. Why Deep Learning

Traditional TF-IDF/BoW models represent text through statistical word/n-gram features without encoding word order or contextual dependencies. Deep learning models can, in principle, learn richer representations:

- **Dense embeddings** capture semantic relationships (king - man + woman ≈ queen)
- **Recurrent architectures** process sequences and can model context
- **End-to-end learning** may discover patterns that hand-crafted features miss

However, deep learning does NOT automatically "understand" sarcasm. It learns statistical patterns from training data and may or may not outperform simpler approaches.

## 3. Why Word Embeddings

Word embeddings map discrete tokens to continuous vector spaces where semantically similar words are closer:

- **BoW/TF-IDF**: "great" and "wonderful" are orthogonal, unrelated dimensions
- **Embeddings**: "great" ≈ "wonderful" in vector space (close cosine similarity)

This can help classify figurative language where meaning depends on semantic nuance rather than exact word presence.

## 4. Why GloVe

GloVe (Global Vectors for Word Representation) from Stanford NLP was chosen because:

- Pre-trained on large corpora (6B tokens from Wikipedia + Gigaword)
- Captures word co-occurrence statistics at global level
- Available in multiple dimensions (50d, 100d, 200d, 300d)
- Widely validated in NLP research
- Does not require additional training infrastructure

**Source**: Stanford NLP, `glove.6B.100d.txt`  
**Download**: https://nlp.stanford.edu/data/glove.6B.zip

## 5. Why Bi-LSTM

Bidirectional LSTM processes sequences from both directions:

- **Forward LSTM**: past → present (left-to-right context)
- **Backward LSTM**: future → present (right-to-left context)

This captures contextual information from surrounding words in both directions, which is conceptually useful for sarcasm detection where meaning can depend on juxtaposition of positive sentiment words with negative context.

The model does NOT automatically understand sarcasm or irony. It learns statistical patterns from the labeled training data.

## 6. Dataset

| Split | Samples | Source |
|---|---:|---|
| Training (from train_clean.csv) | 65,087 | 80% of train |
| Validation | 16,272 | 20% of train |
| Test | 5,527 | test_clean.csv |

Stratified split with random seed 42. No modifications to original dataset.

## 7. Data Split

- **Train/Validation**: 80/20 stratified split from `train_clean.csv` (random_state=42)
- **Test**: `test_clean.csv` evaluated once after final configuration selection
- **Final model retraining**: Used 90/10 internal split from full training data for early stopping

## 8. Tokenization

Custom word-level tokenizer with minimal neural-friendly preprocessing:

| Parameter | Value |
|---|---|
| Max vocabulary | 30,000 (+ PAD + OOV = 30,002) |
| OOV token | `<UNK>` (index 1) |
| PAD token | `<PAD>` (index 0) |
| Preprocessing | URL → `<URL>`, @mention → `<USER>`, whitespace normalization, lowercase |
| Tokenization | Whitespace splitting after preprocessing |

**Critical**: Tokenizer fitted ONLY on training text (leakage prevention). Validation and test texts are transformed only.

Preprocessing was intentionally minimal compared to Phase 3/4/5 classical preprocessing. Aggressive stemming, stopword removal, and emoji stripping were avoided because the Bi-LSTM benefits from retaining more textual signals.

## 9. Sequence Length Selection

Token-length distribution analysis of training data:

| Metric | Value |
|---|---|
| Mean | 15.2 tokens |
| Median | 15 tokens |
| 95th percentile | ~28 tokens |
| 99th percentile | ~35 tokens |

**Selected MAX_LEN = 30**, covering 99.96% of texts with only 0.04% truncated.

- **Padding**: Pre-padding (zeros prepended)
- **Truncation**: Post-truncation (excess tokens removed from end)

## 10. GloVe Vocabulary Coverage

| Metric | Value |
|---|---|
| Project vocabulary | 30,000 |
| GloVe matched | 14,212 |
| OOV (not in GloVe) | 15,788 |
| **Coverage** | **47.37%** |

The relatively low coverage is attributed to:
- Social media language (hashtags, mentions, abbreviations)
- Internet slang and informal language
- Misspellings and repeated characters
- Username/URL normalization tokens

**OOV Strategy**: Random initialization (uniform ±0.05). OOV vectors are trainable when embedding_trainable=True.

## 11. Architecture

```
Input(batch_size, 30)
    → Embedding(30002, 100, GloVe weights)
    → Bidirectional(LSTM(units))
    → Dropout(0.3)
    → Dense(64, ReLU)
    → Dropout(0.3)
    → Dense(4)  [logits]
    → Softmax  [probabilities]
```

The model uses the final hidden states from both LSTM directions, concatenated to form the sequence representation.

## 12. Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dim | 100 |
| LSTM units | 64 / 128 (tested) |
| Dense units | 64 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Learning rate | 0.001 (with ReduceLROnPlateau) |
| Batch size | 32 |
| Maximum epochs | 30 |
| Random seed | 42 |

## 13. Training Strategy

- **Loss**: CrossEntropyLoss (PyTorch equivalent of SparseCategoricalCrossentropy)
- **Optimizer**: Adam (lr=0.001), only trainable parameters
- **LR schedule**: ReduceLROnPlateau (factor=0.5, patience=2, monitor=val_loss)
- **Early stopping**: patience=5 on val_loss, restoring best weights
- **Checkpointing**: Save best model by val_loss

## 14. Early Stopping

| Experiment | Best Epoch | Total Epochs | Early Stopped |
|---|---:|---:|---|
| BILSTM-01 | 9 | 14 | Yes (patience=5) |
| BILSTM-02 | 1 | 6 | Yes (patience=5) |
| BILSTM-03 | 5 | 10 | Yes (patience=5) |

All experiments triggered early stopping, indicating the models converge quickly on this dataset.

## 15. Evaluation Metrics

- **Primary metric**: Macro F1
- **Also reported**: Accuracy, Macro Precision, Macro Recall
- **Per-class**: Precision, Recall, F1, Support for all 4 classes
- **Confusion matrix**: 4×4 matrix with class order [Regular, Sarcasm, Irony, Figurative]

## 16. Results

### Architecture Comparison (Validation)

| Experiment | Embedding | LSTM Units | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| BILSTM-01 | Frozen GloVe | 64 | 0.7375 | 0.6469 |
| **BILSTM-02** | **Fine-tuned GloVe** | **64** | **0.7382** | **0.6474** |
| BILSTM-03 | Frozen GloVe | 128 | 0.7378 | 0.6471 |

**Finding**: Differences between all three configurations are negligible (<0.001 Macro F1). Fine-tuned GloVe (BILSTM-02) was marginally best.

### Best Model Test Results (BILSTM-02)

| Metric | Value |
|---|---|
| Accuracy | 0.8612 |
| Macro Precision | 0.8990 |
| Macro Recall | 0.7500 |
| **Macro F1** | **0.6940** |

### Per-Class Test Performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Regular | 1.0000 | 1.0000 | 0.9997 | 1,858 |
| Sarcasm | 0.7531 | 0.9994 | 0.8589 | 1,602 |
| Irony | 0.8436 | 0.9992 | 0.9148 | 1,301 |
| **Figurative** | **1.0000** | **0.0013** | **0.0026** | **766** |

## 17. Confusion Matrix

The test confusion matrix reveals:

- **Regular**: Near-perfect classification (trivially easy due to URL/link patterns)
- **Sarcasm**: High recall (99.9%) but lower precision (75.3%) — absorbs Figurative samples
- **Irony**: High recall (99.9%) with decent precision (84.4%) — absorbs Figurative samples
- **Figurative**: Catastrophic failure (1 correct out of 766) — nearly all misclassified as Sarcasm or Irony

The model essentially ignores the Figurative class, assigning almost all samples to Sarcasm or Irony.

## 18. Error Analysis

### Primary Error Patterns

1. **Figurative → Sarcasm** (most common): Tweets with `#sarcasm` hashtags labeled as Figurative are predicted as Sarcasm (confidence ~0.63)
2. **Figurative → Irony** (second most common): Tweets with `#irony`/`#ironic` hashtags labeled as Figurative are predicted as Irony (confidence ~0.68)
3. **Sarcasm → Irony** (minor): Some sarcastic tweets misclassified as ironic

### Error Categories

- **Hashtag-dependent classification**: The model learns hashtag signals strongly, but Figurative texts often contain `#sarcasm` or `#ironic` hashtags
- **Missing context**: Short tweets lack sufficient context for disambiguation
- **Social-media style**: Internet slang reduces GloVe coverage
- **Annotation ambiguity**: Figurative vs. Sarcasm distinction is inherently subjective

## 19. Comparison with Phase 4

| Model | Representation | Test Accuracy | Test Macro F1 |
|---|---|---:|---:|
| **BoW + LinearSVM** | **Bag of Words** | **0.8457** | **0.7187** |
| GloVe + Bi-LSTM | GloVe + Bi-LSTM | 0.8612 | 0.6940 |

Phase 4's BoW + SVM **outperforms** Phase 6 Bi-LSTM by +0.0247 Macro F1. While Bi-LSTM achieves slightly higher accuracy (0.8612 vs 0.8457), the Macro F1 is worse because the Bi-LSTM completely fails on the Figurative class.

## 20. Comparison with Phase 5

| Phase | Model | Representation | Test Accuracy | Test Macro F1 |
|---|---|---|---:|---:|
| Phase 4 | BoW + LinearSVM | Bag of Words | 0.8457 | **0.7187** |
| Phase 5 | TF-IDF + LinearSVM | TF-IDF | 0.8565 | 0.7002 |
| Phase 6 | GloVe + Bi-LSTM (BILSTM-02) | GloVe + Bi-LSTM | 0.8612 | 0.6940 |

**Key findings**:
- **Bi-LSTM < BoW SVM** on Macro F1 (-0.0247)
- **Bi-LSTM < TF-IDF SVM** on Macro F1 (-0.0062)
- **Bi-LSTM > both** on raw Accuracy (+0.0047 to +0.0155)
- The accuracy advantage is misleading — it's driven by strong Regular/Sarcasm/Irony performance while Figurative class collapses

### Research Question Answer

> *"Does sequence modeling with GloVe + Bi-LSTM provide an advantage over traditional sparse text representations?"*

**No, not for this dataset.** The Bi-LSTM achieves marginally higher accuracy but worse Macro F1 than both BoW and TF-IDF approaches. The primary bottleneck remains the Figurative class, which the Bi-LSTM fails to learn entirely (F1 ≈ 0.00 vs BoW's ~0.19).

Possible explanations:
1. **Low GloVe coverage** (47.37%) — social media vocabulary is poorly covered
2. **Class imbalance** — Figurative has the fewest samples (766 test, ~9.5K train)
3. **Shallow architecture** — single-layer Bi-LSTM may not capture the subtle distinctions
4. **No class weighting** was applied in the baseline experiments

## 21. Limitations

1. **GloVe coverage**: Only 47.37% of vocabulary matched — a major limitation for social media text
2. **Figurative class failure**: F1 = 0.0026 on test — the model essentially ignores this class
3. **No class weighting experiment**: Could potentially improve Figurative recall at the cost of other classes
4. **CPU-only training**: Training times (~250-360s per experiment) limited hyperparameter exploration
5. **Single train/val split**: No cross-validation for more robust estimates
6. **Minimal preprocessing**: While appropriate for neural models, the low GloVe coverage suggests some preprocessing (e.g., hashtag splitting) could help
7. **Reproducibility**: Small differences possible across hardware/backend despite fixed seeds

## 22. Transition to Phase 7

Phase 7 will investigate FastText / Attention-based modeling to compare:
1. Alternative word representations (FastText handles OOV via subword information)
2. Attention mechanisms for focusing on sarcasm/irony-indicative words

FastText's subword embeddings could address the low GloVe coverage problem, potentially improving performance on social media text with informal spelling and hashtags.

---

## File Structure

```
ml-service/
├── src/deep_learning/
│   ├── __init__.py
│   └── bilstm/
│       ├── __init__.py
│       ├── tokenizer.py          — Custom word-level tokenizer
│       ├── glove.py              — GloVe loading + embedding matrix
│       ├── model.py              — BiLSTMClassifier (PyTorch)
│       ├── train.py              — Training loop + early stopping
│       ├── evaluate.py           — Metrics, CM, error analysis
│       └── run_experiments.py    — Main orchestrator
├── models/deep_learning/bilstm/
│   ├── best/
│   │   ├── best_bilstm.pt
│   │   ├── tokenizer.json
│   │   ├── config.json
│   │   └── label_mapping.json
│   ├── bilstm-01/
│   ├── bilstm-02/
│   └── bilstm-03/
├── results/bilstm/
│   ├── model_results.csv
│   ├── architecture_comparison.csv
│   ├── phase_comparison.csv
│   ├── glove_coverage.csv
│   ├── oov_analysis.csv
│   ├── classification_report.txt
│   ├── predictions.csv
│   ├── error_analysis.csv
│   ├── phase6_summary.json
│   ├── bilstm-0{1,2,3}_history.csv
│   ├── confusion_matrices/
│   └── visualizations/
├── notebooks/
│   └── 06_glove_bilstm.py
├── tests/
│   └── test_bilstm.py
└── docs/research/
    └── phase6_glove_bilstm.md
```

---

*Phase 6 establishes that GloVe + Bi-LSTM does NOT outperform traditional approaches on Macro F1 for this 4-class sarcasm/irony dataset. The best traditional NLP model remains Phase 4's LinearSVM + BoW (Macro F1 = 0.7187). The Bi-LSTM's primary weakness is catastrophic failure on the Figurative class (F1 ≈ 0.00), likely due to low GloVe coverage (47.37%) and class imbalance. The best current GloVe + Bi-LSTM model is BILSTM-02 (Fine-tuned GloVe, LSTM 64, Test Macro F1 = 0.6940).*
