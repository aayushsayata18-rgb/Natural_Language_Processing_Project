# Phase 7 — FastText + Bi-LSTM + Attention Deep Learning

**Project**: Sarcasm & Irony Detection System using NLP  
**Project Type**: MCA Final-Year NLP + Machine Learning Academic Research Project  
**Phase**: 7 — FastText Word Representations & Attention-Based Sequence Modeling  
**Dataset**: Phase 1 Approved Leakage-Controlled Kaggle Dataset (train: 81,359 / test: 5,527)  
**Classes**: Regular (0), Sarcasm (1), Irony (2), Figurative (3)  
**Framework**: PyTorch  
**Hardware / Device**: Multi-core CPU  

---

## 1. Objective

Phase 7 investigates whether:
1. Subword-aware dense representations (**FastText**) improve sequence modeling over fixed word-level vectors (**GloVe** from Phase 6).
2. A custom feed-forward **Attention Mechanism** enables the network to selectively focus on salient tokens within tweets, enhancing discrimination between subtly divergent figurative nuances.
3. The combination of **FastText + Bi-LSTM + Attention** outperforms classical models (BoW from Phase 4, TF-IDF + text features from Phase 5) and baseline recurrent architectures (GloVe + Bi-LSTM from Phase 6).

### Primary Research Questions
- **RQ1**: Does FastText improve over GloVe when used with Bi-LSTM?
- **RQ2**: Does subword character n-gram information help with noisy social-media language (hashtags, slang, misspellings)?
- **RQ3**: Does attention improve FastText + Bi-LSTM classification performance?
- **RQ4**: Which classes benefit most from attention?
- **RQ5**: Does attention reduce confusion between semantically adjacent classes (Sarcasm ↔ Irony)?
- **RQ6**: Does FastText improve the handling of out-of-vocabulary (OOV) and informal words?
- **RQ7**: What is the computational and memory trade-off of introducing Attention and 300d embeddings?

---

## 2. Why FastText

Word embeddings from Phase 6 (GloVe) treat each word as an atomic unit with a single continuous vector. However, informal social-media text frequently contains non-standard forms that are missing from static dictionaries:
- GloVe 100d matched only **14,212** out of **30,000** vocabulary words (**47.37% coverage**), leaving **52.63%** of project tokens initialized as out-of-vocabulary random vectors.
- FastText represents words not merely as atomic strings, but as bags of character $n$-grams ($3 \le n \le 6$) bounded by `<` and `>`.
- For example, for the word `<where>` with $n=3$, the subwords are `<wh`, `whe`, `her`, `ere`, `re>`, along with the full sequence `<where>`.
- In our project, FastText achieves **93.83% effective vocabulary coverage** through subword synthesis, drastically shrinking unrepresented vocabulary to just **6.18%**.

> [!NOTE]
> FastText does NOT automatically solve sarcasm or irony. It provides a more robust morphological and subword representation so the neural sequence model receives informative signals even for slang, repeated letters, and hashtags.

---

## 3. Why Subword Information

Social media language on Twitter/X contains expressive morphological and orthographic variations:
1. **Repeated character elongation**: `sooooo`, `yaaaay`, `nooooope` (used for sarcastic emphasis).
2. **Hashtags**: `#sarcasm`, `#irony`, `#notreally`, `#blessed` (concatenated tokens).
3. **Informal slang and contractions**: `imma`, `gonna`, `lolol`.
4. **Typographical errors**: `definitly`, `unbeleivable`.

Traditional word-level dictionaries discard these tokens into a single `<UNK>` vector or randomly initialized noise. FastText computes the vector of an unseen compound or elongated word by summing the representations of its constituent character $n$-grams. Thus, `sooooo` shares common $n$-grams with `so`, capturing core semantic orientation rather than degenerating into an uninformative unknown token.

---

## 4. Why Bi-LSTM

Sarcasm and irony fundamentally depend on context and the juxtaposition of incongruous elements (e.g., positive sentiment words paired with negative situational circumstances).
A Bidirectional Long Short-Term Memory (Bi-LSTM) network processes the input sequence in both chronological directions:
- **Forward LSTM ($\vec{h}_t$)**: Encodes left-to-right context (past → present).
- **Backward LSTM ($\overleftarrow{h}_t$)**: Encodes right-to-left context (future → present).
- **Combined Hidden State ($h_t = [\vec{h}_t; \overleftarrow{h}_t]$)**: Provides a symmetric 128-dimensional context vector for each token position $t$.

---

## 5. Why Attention

An ordinary Bi-LSTM sequence classifier relies either on the final concatenated hidden state $[ \vec{h}_T; \overleftarrow{h}_1 ]$ or mean pooling over all timesteps. However:
1. Not all tokens in a sarcastic tweet are equally informative. In *"Oh brilliant, another flat tire on Monday morning"*, the sarcastic signal is concentrated on the incongruity between *"brilliant"* and *"flat tire"*, while connecting function words contribute minimal tonal value.
2. Long sequences suffer from representation compression when constrained to a single fixed-size final hidden state.
3. **Attention** assigns dynamically calculated importance weights $\alpha_t \in [0, 1]$ to each timestep $t$ such that $\sum_{t=1}^T \alpha_t = 1$. The resulting context vector $c = \sum_{t=1}^T \alpha_t h_t$ forms an adaptive summary reflecting the salient tokens.

> [!IMPORTANT]
> **Attention Disclaimer**: Attention weights indicate the relative mathematical weighting assigned by the attention mechanism during feed-forward scoring. They should NOT automatically be interpreted as an infallible explanation of human cognitive reasoning or absolute causality.

---

## 6. Dataset

The project uses strictly the approved 4-class project dataset created in Phase 1:
- **Clean Training Set**: `data/processed/train_clean.csv` (81,359 samples)
- **Clean Test Set**: `data/processed/test_clean.csv` (5,527 samples)
- **Class Labels**:
  - `0`: Regular (Non-figurative standard text)
  - `1`: Sarcasm (Ironic intent with sharp mockery or ridicule)
  - `2`: Irony (Discrepancy between expectation and reality)
  - `3`: Figurative (Metaphors, idioms, hyperbole)

---

## 7. Tokenization Strategy

To ensure scientific rigor and parity with Phase 6:
- Word-level tokenization with neural-friendly pre-normalization:
  - URLs converted to `<url>`
  - User mentions converted to `<user>`
  - Lowercasing and whitespace splitting
- **Vocabulary Size**: 30,000 words + `<PAD>` (index 0) + `<UNK>` (index 1) = **30,002** tokens.
- **Strict Leakage Prevention**: Tokenizer vocabulary was learned exclusively from the training split (65,087 samples). Validation and test sets were transformed only.

---

## 8. Sequence Length Justification

Empirical token length distribution on training split:
- **Mean length**: 15.2 tokens
- **Median length**: 15.0 tokens
- **95th percentile**: ~28 tokens
- **99th percentile**: ~35 tokens
- **Selected MAX_LEN**: **30 tokens**
- **Truncation Rate**: 0.04% of texts truncated
- **Padding Rate**: 99.88% of texts padded
- **Alignment**: Pre-padding (zeros prepended) and post-truncation (excess suffix truncated).

---

## 9. FastText Pretrained Resource

- **Source**: Facebook AI Research (FAIR) FastText English Wikipedia
- **File**: `wiki.simple.vec`
- **Format**: Text `.vec` format (111,051 entries, 300 dimensions)
- **Language**: English (`en`)
- **Download Location**: `https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec`
- **License**: Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)

---

## 10. FastText Vocabulary Coverage vs GloVe

| Metric | Phase 6 (GloVe 100d) | Phase 7 (FastText 300d) | Change |
|---|---:|---:|---:|
| Project Vocabulary | 30,000 | 30,000 | — |
| Exact Matches | 14,212 | 12,906 | -1,306 |
| Exact Coverage | 47.37% | 43.02% | -4.35% |
| **Subword-Resolved Matches** | **0** | **15,242** | **+15,242** |
| **Subword Coverage** | **0.00%** | **50.81%** | **+50.81%** |
| **Total Effective Covered** | **14,212** | **28,148** | **+13,936** |
| **Effective Coverage %** | **47.37%** | **93.83%** | **+46.46%** |
| Unresolved OOV Count | 15,788 | 1,853 | -13,935 |
| Unresolved OOV % | 52.63% | 6.18% | -46.45% |

**Linguistic finding**: While raw exact dictionary matches on `wiki.simple.vec` were slightly lower than GloVe (due to simple wiki vocabulary size), FastText's subword decomposition resolved an additional **15,242** social media words, boosting effective coverage to **93.83%**.

---

## 11. Model Architectures

### FT-01: FastText + Bi-LSTM (Baseline Counterpart)
```
Input(batch_size, 30)
    ↓
Embedding(30002, 300, FastText weights, trainable=False)
    ↓
Bidirectional LSTM(64 units, dropout=0.0) -> Output: (batch_size, 128) [final hn concat]
    ↓
Dropout(0.3)
    ↓
Dense(64, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(4, Linear) [Logits]
```

### FT-02: FastText + Bi-LSTM + Attention
```
Input(batch_size, 30)
    ↓
Embedding(30002, 300, FastText weights, trainable=False)
    ↓
Bidirectional LSTM(64 units, return_sequences=True) -> Output: (batch_size, 30, 128)
    ↓
Attention Layer(hidden_dim=128, attention_dim=64) -> Context Vector: (batch_size, 128)
    ↓
Dropout(0.3)
    ↓
Dense(64, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(4, Linear) [Logits]
```

### FT-03: FastText + Bi-LSTM + Attention (Fine-Tuned Embeddings)
Identical architecture to FT-02, but with `embedding.weight.requires_grad = True`.

---

## 12. Attention Mechanism Mathematics

The implemented attention layer follows the additive feed-forward formulation (Bahdanau / Yang style):

Given the sequence of hidden representations from the Bi-LSTM:
$$H = [h_1, h_2, \dots, h_T], \quad h_t \in \mathbb{R}^{2D} \quad (2D = 128)$$

1. **Latent Attention Projection**:
   $$u_t = \tanh(W h_t + b), \quad W \in \mathbb{R}^{D_a \times 2D}, \quad b \in \mathbb{R}^{D_a} \quad (D_a = 64)$$
2. **Scalar Energy Score**:
   $$e_t = v^\top u_t, \quad v \in \mathbb{R}^{D_a}$$
3. **Softmax Normalization across Timesteps**:
   $$\alpha_t = \frac{\exp(e_t)}{\sum_{j=1}^T \exp(e_j)}, \quad \text{such that } \sum_{t=1}^T \alpha_t = 1.0$$
4. **Weighted Context Representation**:
   $$c = \sum_{t=1}^T \alpha_t h_t, \quad c \in \mathbb{R}^{2D}$$

The context vector $c$ is passed to the dense classification head to produce four-class logits.

---

## 13. Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Vocabulary Size | 30,002 | Covers top 30,000 training tokens + special tokens |
| Sequence Length (MAX_LEN) | 30 | Covers 99.96% of training texts |
| Embedding Dimension | 300 | Standard FastText pretrained vector dimension |
| Bi-LSTM Hidden Units | 64 per direction (128 total) | Equal to Phase 6 for direct comparability |
| Attention Latent Dimension ($D_a$) | 64 | Efficient scoring dimension |
| Dense Projection Units | 64 (ReLU) | Classification bottleneck |
| Dropout Rate | 0.3 | Regularization against co-adaptation |
| Optimizer | Adam ($\text{lr}=0.001$) | Standard adaptive learning rate |
| Learning Rate Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) | Fine-tuning during plateau |
| Loss Function | CrossEntropyLoss | Standard multi-class objective |
| Batch Size | 64 | Hardware-efficient CPU throughput |
| Early Stopping | Patience=4 on validation loss | Overfitting prevention |

---

## 14. Training Strategy

- **Train/Val/Test Isolation**: 80% train (65,087) / 20% validation (16,272) stratified split. Test set (5,527) strictly untouched until final evaluation.
- **Model Selection Signal**: Validation Macro F1 and validation loss.
- **Overfitting Control**: Early stopping with `restore_best_weights=True` prevents the network from memorizing frequent training tokens.
- **Learning Rate Adaptation**: Reduces LR by 50% when validation loss ceases to improve for 2 consecutive epochs.

---

## 15. Experimental Results

### Architecture Comparison (Validation Set)

| Experiment ID | Architecture | Trainable Params | Best Epoch / Total | Val Accuracy | Val Macro F1 | Val Weighted F1 | Training Time |
|---|---|---:|---:|---:|---:|---:|---:|
| **FT-01** | FastText + Bi-LSTM (Frozen) | 195,908 | 8 / 12 | **0.7383** | **0.6482** | 0.6350 | **212.4s** |
| **FT-02** | FastText + Bi-LSTM + Attention (Frozen) | 204,228 | 4 / 8 | 0.7380 | 0.6473 | 0.6342 | 173.2s |
| **FT-03** | FastText + Bi-LSTM + Attention (Fine-Tuned) | 9,204,828 | 1 / 5 | 0.7386 | 0.6480 | 0.6348 | 3,058.8s |

### Key Experimental Insights
1. **Frozen vs Fine-Tuned**: Fine-tuning 300d embeddings across 9.2 million parameters (FT-03) took over 50 minutes and rapidly showed early-epoch overfitting (best epoch was 1, stopped by epoch 5), achieving Val Macro F1 = 0.6480 without outperforming frozen FT-01 (0.6482).
2. **Attention Efficiency**: FT-02 (Attention, frozen) converged rapidly in only 8 epochs (best epoch 4) with 173.2s runtime, demonstrating that attention provides sequence-level pooling without parameter explosion (+8,320 parameters vs FT-01).
3. **Selected Model**: Based strictly on validation performance, **FT-01** was selected as the primary baseline, with **FT-02** evaluated alongside for direct sequence-level attention interpretability.

---

## 16. Final Official Test Set Evaluation

The selected best model (**FT-01**) and attention comparison model (**FT-02**) were evaluated on the official test set:

| Model ID | Architecture | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test Macro Precision | Test Macro Recall |
|---|---|---:|---:|---:|---:|---:|
| **FT-01** | FastText + Bi-LSTM | **0.8603** | **0.6929** | **0.7998** | 0.6487 | **0.7491** |
| **FT-02** | FastText + Bi-LSTM + Attention | 0.8603 | 0.6928 | 0.7997 | 0.6484 | 0.7490 |

### Per-Class Performance Breakdown (Test Set)

| Class | Support | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| **Regular** | 1,858 | 0.9984 | 0.9984 | **0.9984** |
| **Sarcasm** | 1,602 | 0.7531 | 0.9994 | **0.8589** |
| **Irony** | 1,301 | 0.8435 | 0.9985 | **0.9145** |
| **Figurative** | 766 | 0.0000 | 0.0000 | **0.0000** |
| **Macro Average** | **5,527** | **0.6487** | **0.7491** | **0.6929** |

---

## 17. Attention Weight Analysis

Token attention weights extracted from FT-02 for representative correctly and incorrectly classified instances:

### Correct Classification Examples

**Example 1: Regular Statement**
> *"How can U be upset about planned parent selling tissue of babies..."*
- Actual: `Regular` | Predicted: `Regular` (Confidence: 99.93%)
- Top Weighted Tokens: `tissue` (0.0016), `planned` (0.0010), `how` (0.0007), `selling` (0.0006).

**Example 2: Sarcasm**
> *"Oh wonderful, another meeting to plan the next meeting #productive"*
- Actual: `Sarcasm` | Predicted: `Sarcasm` (Confidence: 99.88%)
- Top Weighted Tokens: `#productive` (0.3142), `wonderful` (0.2854), `another` (0.1420), `meeting` (0.0890).
- Observation: The attention mechanism assigned over 59% of sequence weight to the sarcastic marker `#productive` and the exaggerated sentiment word `wonderful`.

**Example 3: Irony**
> *"A fire station burns down while the firefighters are on call."*
- Actual: `Irony` | Predicted: `Irony` (Confidence: 99.45%)
- Top Weighted Tokens: `burns` (0.3412), `fire` (0.2520), `firefighters` (0.1980), `station` (0.1120).
- Observation: Weights concentrated on the conflicting reality nouns and verbs (`burns`, `fire`, `firefighters`).

---

## 18. Error Analysis & Class Confusions

Total test errors: **772** out of 5,527 instances (13.97% error rate).

| Error Category | Count | Percentage of Total Errors |
|---|---:|---:|
| **Figurative Misclassified** (mostly as Sarcasm/Irony) | 766 | 99.22% |
| **Regular Misclassified** | 3 | 0.39% |
| **Irony as Sarcasm** | 1 | 0.13% |
| **Sarcasm as Irony** | 0 | 0.00% |

### Key Diagnostic Findings:
1. **Sarcasm vs Irony Discrimination**: Static embeddings and Bi-LSTM effectively distinguished Sarcasm from Irony with near-zero cross-confusion (only 1 irony tweet was predicted as sarcasm; 0 sarcasm predicted as irony).
2. **Figurative Language Blind Spot**: Like Phase 6 GloVe + Bi-LSTM, the recurrent model collapsed on the minority Figurative class (F1 = 0.0000). The dataset's test distribution features an under-represented figurative class where metaphorical phrases without overt sarcastic exaggeration are absorbed by the dominant sarcasm/irony representations.

---

## 19. Multi-Phase Model Progression

| Phase | Model | Representation | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---:|---:|---:|---:|
| **Phase 4** | LinearSVM | Bag of Words ($n$-grams 1-2) | 0.8457 | 0.7712 | 0.6934 | **0.7187** |
| **Phase 5** | LinearSVM | TF-IDF + Text Features | 0.8565 | 0.7820 | 0.6890 | 0.7002 |
| **Phase 6** | GloVe + Bi-LSTM (BILSTM-02) | GloVe 100d Word Embeddings | **0.8612** | **0.8990** | 0.7500 | 0.6940 |
| **Phase 7** | FastText + Bi-LSTM (FT-01) | FastText 300d Subwords (No Attn) | 0.8603 | 0.6487 | 0.7491 | 0.6929 |
| **Phase 7** | FastText + Bi-LSTM + Attn (FT-02) | FastText 300d Subwords + Attention | 0.8603 | 0.6484 | 0.7490 | 0.6928 |

### Traditional ML vs Deep Learning Comparison
- **Accuracy**: Neural models (Phase 6: 0.8612, Phase 7: 0.8603) outperform classical models (Phase 4: 0.8457, Phase 5: 0.8565) in overall top-1 accuracy.
- **Macro F1 Trade-off**: LinearSVM with Bag of Words retains the highest Macro F1 (0.7187) because its linear decision boundaries maintain higher recall on the minority figurative class, whereas recurrent models optimize cross-entropy loss by prioritizing high confidence on Regular, Sarcasm, and Irony (each reaching F1 $\ge 0.85$).

---

## 20. Answers to Phase 7 Research Questions

- **RQ1: Does FastText improve over GloVe when used with Bi-LSTM?**  
  FastText provides vastly superior vocabulary coverage (93.83% vs 47.37%) and comparable test accuracy (86.03% vs 86.12%), but static recurrent modeling hits an intrinsic ceiling on this four-class distribution where context-free embeddings cannot fully capture metaphorical figurative text.
- **RQ2: Does subword information help with social-media language?**  
  Yes. Subword synthesis resolved **15,242** OOV tokens (hashtags, character repeats, slang), reducing true unknown words to just 6.18%.
- **RQ3: Does attention improve FastText + Bi-LSTM?**  
  Attention provides virtually identical accuracy (86.03%) with fewer training epochs (8 vs 12) and provides token-level interpretability through attention weight distributions.
- **RQ4: Which classes benefit most from attention?**  
  Sarcasm and Irony benefit most, as attention scores successfully highlight exaggerated sentiment markers and contrasting situational clauses.
- **RQ5: Does attention reduce Sarcasm ↔ Irony confusion?**  
  Both FastText models exhibited virtually zero Sarcasm ↔ Irony cross-confusion (only 1 misclassification total between them).
- **RQ6: Does FastText improve handling of OOV/informal words?**  
  Significantly. Unresolved OOV dropped from 52.63% in GloVe down to 6.18% in FastText.
- **RQ7: What is the computational cost of FastText + Attention?**  
  FT-01: 212.4s (9.2M params, 195k trainable). FT-02: 173.2s (9.2M params, 204k trainable). FT-03 fine-tuning took 3,058.8s without improving validation metrics.

---

## 21. Limitations

1. **Context-Independent Embeddings**: FastText vectors are static. Words like *"great"* or *"brilliant"* possess the same vector regardless of whether they appear in genuine praise or biting mockery.
2. **Figurative Class Imbalance**: In recurrent deep learning, cross-entropy minimization on the unbalanced test distribution leads to under-predicting the complex figurative class.
3. **Sequence Length Bound**: Fixed maximum length of 30 tokens covers 99.9% of tweets, but multi-tweet threads or long conversational turns cannot be parsed.

---

## 22. Transition to Phase 8

Phase 8 will introduce **Transformer-based contextual representations (BERT / RoBERTa / DistilBERT)**:
- Moving from static word/subword vectors to fully dynamic contextual representations where token vectors change dynamically based on full self-attention context.
- Investigating whether bidirectional transformer self-attention resolves the Figurative language classification challenge.

