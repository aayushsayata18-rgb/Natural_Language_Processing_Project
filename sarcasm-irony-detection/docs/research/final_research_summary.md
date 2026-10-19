# Comprehensive Research Summary — NLP Sarcasm & Irony Detection

**Project Title**: Four-Class Sarcasm & Irony Detection System using NLP  
**Degree / Purpose**: MCA Final-Year Academic Research Capstone Project  
**Authors**: MCA NLP Engineering & Research Team  
**Evaluation Scope**: Phases 1 through 9 (Complete Research Lifecycle)  
**Date**: October 2026  

---

## 1. Research Objective & Academic Motivation

Sarcasm and irony are complex forms of non-literal, figurative communication in natural language where speakers deliberately express a meaning that sharply contradicts or inverts the literal semantic content of their words. In automated sentiment analysis, social media monitoring, and customer feedback processing, undetected sarcasm constitutes a primary source of severe polarity classification error (e.g., classifying *"I just love standing in an unmoving queue for two hours!"* as positive customer feedback).

This capstone project investigates the full historical progression of Natural Language Processing (NLP) methodologies applied to the four-class discrimination problem:
1. `Regular` (Class 0): Standard literal communication.
2. `Sarcasm` (Class 1): Sharp, contemptuous mockery conveying negative criticism through surface positive praise.
3. `Irony` (Class 2): Incongruity between expectation and actual state of affairs.
4. `Figurative` (Class 3): Metaphorical, hyperbolic, or poetic non-literal expression.

### Central Research Question:
> *"How do traditional lexical representations, dense static word embeddings, recurrent neural architectures, attention mechanisms, and deep bidirectional contextual transformers compare across four-class classification performance, minority-class sensitivity, computational feasibility, and inference efficiency?"*

---

## 2. Experimental Dataset & Decontamination Protocol

The research utilized the official Kaggle 4-class Sarcasm and Irony Dataset:
- **Raw Volume**: 89,536 total records across `train.csv` (81,408) and `test.csv` (8,128).
- **Phase 1 Validation & Decontamination Audit**:
  - Identified 49 exact duplicate rows in the training split.
  - Uncovered 2,592 cross-split overlapping tweets between train and test with conflicting labels.
  - To prevent test data contamination and invalid evaluation, the test set was decontaminated into a clean, untouched benchmark of **5,527 samples**.
  - The training dataset was cleansed into **81,359 samples**.
- **Official Test Set Class Distribution**:
  - `Regular`: 1,858 (33.6%)
  - `Sarcasm`: 1,602 (29.0%)
  - `Irony`: 1,301 (23.5%)
  - `Figurative`: 766 (13.9%)

---

## 3. Methodological Overview Across Project Phases

The project was executed in a strictly sequential, evidence-based progression:

```
[Phase 1: Dataset Validation & Leakage Decontamination]
                           ↓
[Phase 2: Comprehensive Exploratory Data Analysis (EDA)]
                           ↓
[Phase 3: Standardized Preprocessing Pipeline (Minimal, Classical, Lemmatized, Stemmed)]
                           ↓
[Phase 4: Bag of Words (BoW) + Classical Classifiers (SVM, Naive Bayes, Logistic Regression)]
                           ↓
[Phase 5: TF-IDF + N-Grams + Linguistic & Sentiment Feature Engineering]
                           ↓
[Phase 6: GloVe 100d Dense Word Embeddings + Bidirectional LSTM (Bi-LSTM)]
                           ↓
[Phase 7: FastText 300d Subword Synthesis + Bi-LSTM + Feed-Forward Attention]
                           ↓
[Phase 8: Fine-Tuned BERT Contextual Transformer (bert-base-uncased)]
                           ↓
[Phase 9: Comprehensive Cross-Model Comparison, Error Analysis & Final Model Selection]
```

---

## 4. Summary of Experiments Across Paradigms

### 4.1 Traditional NLP: Bag of Words (Phase 4)
- **Representations**: Sparse word occurrence counts (`CountVectorizer`), unigrams and bigrams, feature dimensions ranging from 27,344 to 50,000.
- **Classifiers**: LinearSVM ($C=0.1$), Multinomial Naive Bayes ($\alpha=1.0$), and Logistic Regression.
- **Key Finding**: `LinearSVM (BOW-16)` with classical preprocessing achieved **84.57% accuracy** and **0.7187 Macro F1**. The linear margin hyperplanes maintained balanced separation across all four classes, yielding a decisive advantage on the minority `Figurative` class ($F_1 = 0.1378$, Recall = $16.97\%$).

### 4.2 Statistical NLP: TF-IDF & Feature Engineering (Phase 5)
- **Representations**: Term Frequency-Inverse Document Frequency (`TfidfVectorizer`, sublinear scaling) combined with 12 engineered lexical/sentiment features (vader sentiment compound, punctuation counts, capitalization ratio, exclamation marks, emoji frequency).
- **Key Finding**: Sublinear TF-IDF boosted overall accuracy to **85.65%** for LinearSVM and **85.83%** for Logistic Regression. However, TF-IDF weighting penalized rare figurative n-grams, reducing Figurative recall to $2.22\%$ ($F_1 = 0.0421$) and lowering Macro F1 to **0.7002**. Handcrafted features contributed marginal gains over raw lexical TF-IDF.

### 4.3 Deep Learning: Dense Static Embeddings & Recurrence (Phase 6)
- **Representations**: Pretrained GloVe 100-dimensional embeddings (`glove.6B.100d`), sequence length 30 tokens.
- **Architecture**: 2-layer Bidirectional LSTM (64 hidden units), dropout (0.3), dense projection (64 units, ReLU), 4-class softmax head (3.2M parameters).
- **Key Finding**: Sequence modeling improved accuracy to **86.12%**. Fine-tuning embedding weights reduced loss rapidly. However, standard unweighted cross-entropy loss caused the network to optimize almost exclusively for the three majority classes, collapsing Figurative recall to $0.13\%$ ($F_1 = 0.0026$) and capping Macro F1 at **0.6940**.

### 4.4 Subwords & Attention: FastText + Bi-LSTM + Attention (Phase 7)
- **Representations**: FastText 300-dimensional embeddings (`wiki-news-300d-1M`) with character n-gram subword synthesis (boosting effective vocabulary coverage from 43.0% to 93.8%).
- **Architectures**: FastText Bi-LSTM (`FT-01`, 9.2M params) and FastText Bi-LSTM with feed-forward additive attention (`FT-02`, 9.2M params).
- **Key Finding**: FastText completely solved out-of-vocabulary misspellings and Twitter hashtags. The attention layer successfully highlighted contrastive tokens (e.g., attending heavily to *"love"* in sarcastic tweets). Both models achieved **86.03% accuracy** and **0.6929 / 0.6928 Macro F1**, but still exhibited the minority-class collapse on Figurative text.

### 4.5 Transformers: Pretrained BERT Contextual Fine-Tuning (Phase 8)
- **Architecture**: `bert-base-uncased` (12 encoder blocks, 768 hidden dimensions, 12 attention heads, 109.5M parameters) fine-tuned end-to-end with AdamW ($lr=2\times 10^{-5}$).
- **Key Finding**: BERT achieved tied top accuracy (**86.12%**) and completely eliminated mutual confusion between Sarcasm and Irony (**0 errors out of 2,903 tweets**). However, cross-entropy fine-tuning pushed all figurative instances into Sarcasm ($n=525$) and Irony ($n=241$), resulting in $0.00\%$ Figurative recall and a Macro F1 of **0.6935**. Training required 1,802 seconds (~30 mins on CPU), and inference was measured at **39.33 ms/text**.

---

## 5. Cross-Paradigm Performance & Complexity Synthesis

| Approach | Architecture | Parameters | Training Duration | Latency / Text | Test Accuracy | Macro F1 (Primary) | Figurative F1 | Sarcasm ↔ Irony Errors |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BoW (Phase 4)** | **LinearSVM** | **112,312** | **12.4s** | **0.027 ms** | 0.8457 | **0.7187** | **0.1378** | **0** |
| TF-IDF (Phase 5) | LinearSVM | 112,312 | 28.6s | 0.014 ms | 0.8565 | 0.7002 | 0.0421 | 0 |
| TF-IDF (Phase 5) | LogisticRegression | 112,312 | 1.55s | 0.015 ms | 0.8583 | 0.6918 | 0.0026 | 0 |
| GloVe (Phase 6) | Bi-LSTM (64u) | 3,197,636 | 184.2s | 0.041 ms | **0.8612** | 0.6940 | 0.0026 | 1 |
| FastText (Phase 7)| Bi-LSTM (64u) | 9,196,508 | 212.4s | 0.085 ms | 0.8603 | 0.6929 | 0.0000 | 1 |
| FastText (Phase 7)| Bi-LSTM + Attn | 9,204,828 | 246.8s | 0.098 ms | 0.8603 | 0.6928 | 0.0000 | 1 |
| **BERT (Phase 8)** | `bert-base-uncased` | 109,485,316 | 1,802.2s | 39.33 ms | **0.8612** | 0.6935 | 0.0000 | **0** |

---

## 6. Error Analysis Insights Across Models

Categorization of 300 representative test errors revealed distinct linguistic failure modes:
1. **Figurative Overlap with Sarcastic Hyperbole (58% of all errors)**:
   - Tweets labeled `Figurative` containing rhetorical exaggerations (e.g., *"I'm drowning in a sea of paperwork"*) share identical lexical markers with sarcastic hyperbole. Without external situational context, models predict `Sarcasm`.
2. **Figurative Overlap with Situational Irony (26% of all errors)**:
   - Figurative descriptions of paradoxical situations (e.g., *"The shoemaker's children go barefoot"*) are predicted as `Irony`.
3. **Ultra-Short Tweets Without Pragmatic Context (10% of all errors)**:
   - Texts under 5 words (e.g., *"Oh great."*, *"Brilliant move."*) lack sufficient context to determine whether the statement is sincere approval or sarcastic mockery.
4. **Punctuation & Ellipsis Dependency (4% of all errors)**:
   - Sarcasm indicated subtly through ellipses (*"Well then..."*) or quotation marks (*"Our 'leader'"*) is occasionally missed when punctuation is stripped or diluted.
5. **False Sarcasm Alarms on Literal Text (<1% error rate)**:
   - Across all models, literal `Regular` tweets achieved over **99.8% precision**, demonstrating that literal factual communication is exceptionally resilient against false sarcasm alarms.

---

## 7. Final Model Selection & Academic Justification

**Selected Model**: `SarcasmIronyClassifier-LinearSVM-v1` (CAND-01: Bag of Words Linear Support Vector Classifier).

### Rationale:
1. **Empirical Supremacy on Primary Metric**: Achieves the highest Macro F1 (**0.7187**) among all seven candidate models.
2. **Superior Class Balance**: It is the only architecture that successfully avoids the majority-class collapse trap on the minority Figurative class ($16.97\%$ recall).
3. **Flawless Regular Text Separation**: 99.95% recall on regular literal text (only 1 false positive out of 1,858 tweets).
4. **Zero Sarcasm ↔ Irony Confusion**: Exactly 0 sarcastic tweets confused as irony, and 0 ironic tweets confused as sarcasm.
5. **Ultra-Fast & Lightweight Inference**:
   - Inference latency: **0.027 ms/text** (>1,400$\times$ faster than BERT).
   - Throughput: **37,000 tweets/sec** on CPU.
   - Memory footprint: <500 MB (runs effortlessly in serverless or edge deployments).

---

## 8. Academic Conclusions: What Did We Learn?

Comparing five distinct NLP eras on a common, decontaminated benchmark produced profound academic conclusions:

1. **On Lexical vs. Contextual Representations**:
   - Contextual transformers (BERT) excel at capturing phrase-level incongruity and eliminating subtle boundaries (flawless Sarcasm-Irony separation).
   - However, classical Bag of Words linear classifiers retain a surprising, powerful advantage in imbalanced multi-class settings: linear margin maximization naturally preserves minority class hyperplanes that gradient descent with unweighted cross-entropy flattens.
2. **On the Accuracy Metric Fallacy**:
   - High overall accuracy can mask severe algorithmic blindness on minority classes. In multi-class pragmatic NLP tasks, Macro F1 must always take precedence over accuracy.
3. **On Static Word Embeddings vs. Recurrence**:
   - Moving from BoW to GloVe (+16$\times$ parameters) and FastText (+82$\times$ parameters) improved test accuracy from 84.57% to 86.12% on majority classes, but failed to resolve figurative ambiguity.
   - FastText's subword character n-grams proved invaluable for social media slang and hashtags, resolving 93.8% of out-of-vocabulary tokens.
4. **On Computational Return on Investment (ROI)**:
   - Fine-tuning BERT required $975\times$ more parameters, $145\times$ more training time, and $1,450\times$ more inference latency on CPU compared to LinearSVM, yet resulted in a lower Macro F1 (0.6935 vs 0.7187). In applied NLP engineering, computational cost and operational complexity must be weighed rigorously against empirical gains.

---

## 9. Limitations & Ethical Considerations

1. **Single-Sentence Scope**: Microblog tweets lack conversational history. Pragmatic intent often requires knowing the preceding conversational turn or the relationship between interlocutors.
2. **Dataset Annotation Subjectivity**: Sarcasm and irony perception varies by culture, age, and dialect. What is perceived as playful irony by one annotator may be classified as literal or offensive by another.
3. **Responsible Deployment**: Automated sarcasm classifiers should assist human moderators and improve aggregate sentiment statistics, but should never be used as punitive or legal determinations of individual malice.

---

## 10. Future Work

1. **Conversational Context Integration**: Integrating thread replies and user metadata into the transformer input sequence.
2. **Class-Weighted & Focal Loss Optimization**: Re-training BERT and Bi-LSTM using Focal Loss ($\gamma=2$) or inverse-frequency class weights to prevent minority class collapse.
3. **Multimodal Analysis**: Incorporating accompanying images, memes, and audio cues to detect visual-textual incongruity.
4. **Model Quantization & Distillation**: Applying 8-bit quantization (INT8) or knowledge distillation (DistilBERT) to evaluate whether compressed transformers can achieve lower inference latency.

---

## 11. Final Roadmap to Phase 10

With the final model selected, validated, packaged in `models/final/`, and accompanied by comprehensive empirical data and visualizations, Phase 9 is concluded. The project is prepared for **Phase 10: Production Model Packaging & Microservice Deployment**.
