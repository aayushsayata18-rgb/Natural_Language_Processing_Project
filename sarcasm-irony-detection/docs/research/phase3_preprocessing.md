# Phase 3 — NLP Text Preprocessing

**Project**: Sarcasm & Irony Detection System using NLP  
**Phase**: 3 — NLP Text Preprocessing  
**Dataset**: Phase 1 clean datasets (train: 81,359 / test: 5,527)  
**Classes**: Regular (0), Sarcasm (1), Irony (2), Figurative (3)

---

## 1. Purpose

Phase 3 converts the validated dataset from Phase 1 into reusable, configurable NLP preprocessing pipelines. The goal is NOT to create one blindly cleaned dataset, but to investigate and implement preprocessing strategies that preserve information potentially useful for sarcasm and irony detection.

The output of this phase is used by later phases for:
- Bag of Words (Phase 4)
- TF-IDF & N-grams (Phase 5)
- Bi-LSTM (Phase 6)
- BERT (Phase 8)

## 2. Preprocessing Challenges

Sarcasm and irony detection presents unique preprocessing challenges:

1. **Literal vs. intended meaning**: The literal words may not represent the intended meaning
2. **Social-media noise**: Tweets contain URLs, mentions, hashtags, emojis, and informal language
3. **Signal vs. noise ambiguity**: Elements that appear as "noise" in traditional NLP (e.g., excessive punctuation, capitalization, emojis) may actually carry sarcastic/ironic signals
4. **Short text**: Tweets are limited in length, making every token potentially important
5. **Informal language**: Contractions, slang, repeated characters (e.g., "sooooo")

## 3. Social-Media Characteristics

Based on Phase 2 EDA findings:

| Feature | Observation |
|---|---|
| Hashtags | 99.98% of tweets contain them; `#sarcasm`, `#irony` carry signal |
| Mentions | 33.65% of tweets have @mentions |
| URLs | 37.64% (Regular class has 79.2% — highest) |
| Emojis | 6.72% of tweets contain emojis |
| Capitalization | Uppercase ratio 0.0926; 22.84% have all-caps words |
| Repeated chars | 15.59% of tweets have repeated characters |
| Vocabulary | 71,805 unique words; 68.73% are rare (appear once) |

## 4. Preprocessing Techniques

The following 15 preprocessing stages are implemented as separate, reusable functions:

1. **Text validation** — Safe handling of None, NaN, empty, non-string inputs
2. **Whitespace normalization** — Collapse multiple spaces, tabs, newlines
3. **URL handling** — Replace with `URL` placeholder
4. **Mention handling** — Replace with `USER_MENTION` placeholder
5. **Hashtag handling** — Preserve, strip #, or CamelCase segmentation
6. **Emoji handling** — Preserve, convert to text description, or remove
7. **Repeated-character handling** — Normalize `sooooo` → `soo`
8. **Case normalization** — Optional lowercasing
9. **Punctuation handling** — Preserve, normalize runs, or remove
10. **Negation handling** — Protect negation words during stopword removal
11. **Tokenization** — NLTK TweetTokenizer for social-media text
12. **Stopword handling** — Optional removal with negation protection
13. **Lemmatization** — WordNet lemmatizer with POS tagging
14. **Stemming** — Porter stemmer (experimental)
15. **N-gram preparation** — Token reassembly for vectorization

## 5. Why Aggressive Cleaning Is Avoided

Traditional NLP preprocessing aggressively removes "noise":
- Remove all punctuation
- Remove all emojis
- Lowercase everything
- Remove all stopwords
- Stem everything

For sarcasm/irony detection, this is problematic because:

- **Punctuation** (`!!!`, `???`, `!?`) conveys tone and emphasis
- **Emojis** (😂, 🙃, 😒) signal sarcastic intent
- **Capitalization** (`WHAT A GREAT IDEA`) indicates emphasis/sarcasm
- **Repeated characters** (`soooo good`) indicate exaggeration
- **Negation words** (`not`, `never`) flip meaning — critical for sarcasm
- **Hashtags** (`#sarcasm`, `#irony`) are explicit markers

Instead, we create controlled preprocessing variants and compare them through later model experiments.

## 6. Configuration Variants

Six controlled preprocessing variants are implemented:

### Variant A — Minimal Social Media
- Normalize URLs and mentions
- Preserve everything else (hashtags, emojis, punctuation, case)
- No stopword removal, no lemmatization, no stemming
- **Purpose**: Maximum information preservation

### Variant B — Lowercase
- Same as Variant A + lowercase
- **Purpose**: Test effect of case normalization

### Variant C — Lemmatized
- Same as Variant A + lemmatization (WordNet + POS)
- **Purpose**: Test morphological normalization

### Variant D — Stopword Experiment
- Same as Variant A + stopword removal (with negation protection)
- **Purpose**: Test stopword impact with protected negation

### Variant E — Stemmed
- Same as Variant A + Porter stemming
- **Purpose**: Test aggressive morphological reduction

### Variant F — Classical Clean
- Lowercase, normalize URLs/mentions, strip hashtags, convert emojis to text
- Remove stopwords (protect negation), lemmatize, normalize punctuation
- **Purpose**: Traditional NLP baseline for comparison

## 7. Tokenization

Uses NLTK's `TweetTokenizer`, designed for social-media text:

- Handles hashtags as single tokens (`#sarcasm`)
- Handles mentions as single tokens (`@user`)
- Properly tokenizes contractions (`don't` → `do`, `n't`)
- Separates punctuation as individual tokens
- Preserves emojis as tokens where supported

Example:
```
Input:  "Great!!! Another exam tomorrow 😒"
Tokens: ["Great", "!", "!", "!", "Another", "exam", "tomorrow", "😒"]
```

## 8. Stopword Handling

Two configurations are supported:

- **CONFIG A**: Stopwords retained (default)
- **CONFIG B**: Stopwords removed with negation protection

When stopwords are removed, negation words are explicitly protected:
`not, no, never, nor, neither, nobody, nothing, nowhere, none, cannot, n't, don't, won't, wouldn't, shouldn't, couldn't, doesn't, didn't, isn't, aren't, wasn't, weren't, hasn't, haven't, hadn't`

Example:
```
Input:  "I do not like this at all"
With protection:    "not like"
Without protection: "like"
```

## 9. Lemmatization

Uses NLTK `WordNetLemmatizer` with POS tagging for accurate lemma resolution:

- POS tags are mapped from Penn Treebank to WordNet format
- Verbs, adjectives, and adverbs are lemmatized with correct POS
- Default POS is noun for unknown tags

Example:
```
Input:  ["running", "quickly", "towards", "better", "days"]
Output: ["run", "quickly", "towards", "good", "day"]
```

Lemmatization is optional (`lemmatize=True/False`).

## 10. Stemming

Uses NLTK `PorterStemmer`:

Example:
```
Input:  ["running", "happily", "connections"]
Output: ["run", "happili", "connect"]
```

**Warning**: Stemming can damage informal social-media words. It is an experiment, not a default. Note that `happily` → `happili` is a non-word output — this is expected behavior for stemming.

## 11. Negation Handling

Negation is critical for sarcasm detection because it flips semantic polarity:
- "I love this" → positive
- "I don't love this" → negative/sarcastic

Negation words are:
1. Never removed during preprocessing
2. Protected from stopword removal when `protect_negations=True`
3. Available as features for downstream models

Tested with explicit negation preservation tests.

## 12. URL Handling

URLs are replaced with a consistent `URL` placeholder:
```
"check https://example.com/page" → "check URL"
```

Rationale:
- URL content is not useful for sarcasm classification
- The *presence* of a URL is a structural signal (Regular class has 79.2% URLs)
- `url_count` and `has_url` features are preserved for feature engineering

## 13. Mention Handling

@mentions are replaced with `USER_MENTION`:
```
"@john said hello @jane" → "USER_MENTION said hello USER_MENTION"
```

Rationale:
- Individual usernames do not generalize
- The *presence* of mentions is a social-media pattern feature
- `mention_count` and `has_mention` features are preserved

## 14. Hashtag Handling

Three modes are implemented:

| Mode | Example Input | Output |
|---|---|---|
| preserve | `#sarcasm` | `#sarcasm` |
| strip | `#sarcasm` | `sarcasm` |
| segment | `#WhatAGreatIdea` | `What A Great Idea` |

99.98% of tweets contain hashtags. Top hashtags include `#sarcasm`, `#irony`, `#ironic` — these are explicit class markers and should be investigated, not blindly removed.

## 15. Emoji Handling

Three modes are implemented:

| Mode | Example Input | Output |
|---|---|---|
| preserve | `great 😂` | `great 😂` |
| text | `great 😂` | `great face with tears of joy` |
| remove | `great 😂` | `great` |

6.72% of tweets contain emojis. Sarcasm frequently uses 😂 🙃 😒 😐.

## 16. Punctuation Handling

Three modes are implemented:

| Mode | Example | Output |
|---|---|---|
| preserve | `Great!!!` | `Great!!!` |
| normalize | `Great!!!` | `Great!!` |
| remove | `Great!!!` | `Great` |

Punctuation patterns (`!!!`, `???`, `!?`, `...`) convey tone and are potentially useful for sarcasm detection.

## 17. Model-Specific Preprocessing

Different model families require different preprocessing:

### Classical ML (TF-IDF / BoW)
- URL/mention normalization
- Tokenization
- Optional lowercase, lemmatization
- Controlled punctuation
- N-grams

### Bi-LSTM
- Appropriate normalization
- Tokenizer → integer sequences → padding → embedding
- Do not over-clean

### BERT
- Minimal normalization only
- BERT's own subword tokenizer handles the rest
- Do NOT apply stemming, aggressive stopword removal, or lemmatization before BERT

## 18. Leakage Considerations

**Rule**: Any preprocessing component that LEARNS from text must be fitted only on training data.

| Component | Type | Leakage Risk |
|---|---|---|
| Stopword list | Fixed external resource | None |
| URL/mention regex | Deterministic | None |
| TF-IDF vocabulary | Learned | Must fit on training only |
| Tokenizer vocabulary | Learned | Must fit on training only |
| Embedding vocabulary | Learned | Must fit on training only |

All deterministic preprocessing (regex-based, rule-based) can be safely applied to both train and test. Learned components (Phase 4+) must follow the fit-on-train-only protocol.

## 19. Sample Outputs

Example text: `@john WOW!!! #sarcasm 😂 https://example.com`

| Variant | Output |
|---|---|
| Minimal | `USER_MENTION WOW !!! #sarcasm 😂 URL` |
| Lowercase | `user_mention wow !!! #sarcasm 😂 url` |
| Lemmatized | `USER_MENTION WOW !!! #sarcasm 😂 URL` |
| Stopword | `USER_MENTION WOW !!! #sarcasm 😂 URL` |
| Stemmed | `user_ment WOW !!! #sarcasm 😂 URL` |
| Classical | `user_mention wow !! sarcasm face with tears of joy url` |

## 20. Preprocessing Comparison

The comparison metrics are generated from the full training dataset and saved to `data/reports/preprocessing/preprocessing_comparison.csv`.

Key metrics per variant:
- Records processed
- Empty text percentage
- Average characters
- Average words
- Vocabulary size
- Emoji retention percentage
- Hashtag retention percentage
- Punctuation retention percentage

## 21. Recommendations for Later Experiments

Phase 4/5 should systematically evaluate:

| Experiment | Variant | Hypothesis |
|---|---|---|
| P1 | Minimal | Maximum information preserves sarcasm signals |
| P2 | Lowercase | Case normalization reduces vocabulary without losing too much |
| P3 | Lemmatized | Morphological normalization helps generalization |
| P4 | Stopword | Removing function words focuses on content |
| P5 | Stemmed | Aggressive reduction may harm sarcasm signals |
| P6 | Classical | Traditional NLP baseline |

**Important**: No variant is declared "best" based on preprocessing statistics alone. The optimal strategy must be determined through actual classification performance in Phase 4/5.

---

## File Structure

```
ml-service/
├── src/preprocessing/
│   ├── __init__.py          — Package exports
│   ├── config.py            — PreprocessingConfig + 6 variants
│   ├── text_cleaner.py      — Text validation
│   ├── social_media.py      — URL, mention, hashtag, emoji handling
│   ├── normalization.py     — Whitespace, case, punctuation, negation, stopwords
│   ├── tokenizer.py         — TweetTokenizer wrapper
│   ├── lemmatization.py     — WordNet lemmatizer + POS
│   ├── stemming.py          — Porter stemmer
│   ├── pipeline.py          — Pipeline orchestrator + quality checks
│   └── setup_nltk.py        — NLTK resource downloader
├── tests/
│   └── test_preprocessing.py — 74 unit tests
├── notebooks/
│   └── 03_nlp_preprocessing.py — Interactive demonstration
└── data/
    ├── processed/preprocessed/  — 6 variant output directories
    └── reports/preprocessing/   — Comparison CSV + summary JSON
```

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| Python | 3.14.2 | Runtime |
| Pandas | — | Data manipulation |
| NumPy | — | Numerical operations |
| NLTK | 3.9.4 | Tokenization, lemmatization, stemming, stopwords |
| emoji | 2.15.0 | Emoji detection and conversion |
| re | (stdlib) | Regex patterns for URL/mention/hashtag detection |

---

*Phase 3 prepares text and preprocessing infrastructure. No models are trained. No TF-IDF or BoW is computed. The best preprocessing strategy will be determined through classification experiments in Phase 4/5.*
