"""
frequency_analysis.py — Word frequency, n-gram, stopword, vocabulary, and TF-IDF analysis.

All analysis is EDA-only. The original clean dataset is never modified.

Phase 2: Exploratory Data Analysis
Project: Sarcasm & Irony Detection System
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from typing import Dict, Any, List, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    try:
        _sw = nltk_stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        _sw = nltk_stopwords.words("english")
    STOPWORDS = set(_sw)
    HAS_NLTK = True
except (ImportError, LookupError):
    STOPWORDS = set()
    HAS_NLTK = False


CLASS_ORDER = ["Regular", "Sarcasm", "Irony", "Figurative"]


# ---------------------------------------------------------------------------
# 1. Lightweight EDA Text Normalization
# ---------------------------------------------------------------------------

def eda_normalize(texts: pd.Series) -> pd.Series:
    """
    Create a lightweight normalized copy for frequency analysis ONLY.
    Does NOT modify the original dataset.
    """
    normalized = texts.str.lower()
    # Remove URLs for frequency counting
    normalized = normalized.apply(
        lambda x: re.sub(r'https?://\S+|www\.\S+', '', str(x))
    )
    # Remove @mentions (replace with empty)
    normalized = normalized.apply(
        lambda x: re.sub(r'@\w+', '', str(x))
    )
    # Remove most punctuation but keep # for hashtags
    normalized = normalized.apply(
        lambda x: re.sub(r'[^\w\s#]', ' ', str(x))
    )
    # Collapse whitespace
    normalized = normalized.apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    return normalized


def tokenize_simple(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return str(text).split()


# ---------------------------------------------------------------------------
# 2. Word Frequency
# ---------------------------------------------------------------------------

def word_frequency(
    texts: pd.Series,
    remove_stopwords: bool = False,
    top_n: int = 30,
) -> List[Tuple[str, int]]:
    """Calculate word frequencies from normalized text."""
    counter = Counter()
    for text in texts:
        words = tokenize_simple(text)
        if remove_stopwords and STOPWORDS:
            words = [w for w in words if w not in STOPWORDS]
        # Filter very short tokens
        words = [w for w in words if len(w) > 1]
        counter.update(words)
    return counter.most_common(top_n)


def classwise_word_frequency(
    df: pd.DataFrame,
    text_col: str = "text_normalized",
    remove_stopwords: bool = True,
    top_n: int = 20,
) -> Dict[str, List[Tuple[str, int]]]:
    """Calculate word frequencies per class."""
    result = {}
    for label in CLASS_ORDER:
        subset = df[df["label"] == label]
        result[label] = word_frequency(
            subset[text_col], remove_stopwords=remove_stopwords, top_n=top_n
        )
    return result


# ---------------------------------------------------------------------------
# 3. N-gram Analysis
# ---------------------------------------------------------------------------

def extract_ngrams(
    texts: pd.Series,
    n: int = 2,
    remove_stopwords: bool = True,
    top_n: int = 20,
) -> List[Tuple[str, int]]:
    """Extract top n-grams from texts."""
    counter = Counter()
    for text in texts:
        words = tokenize_simple(text)
        if remove_stopwords and STOPWORDS:
            words = [w for w in words if w not in STOPWORDS]
        words = [w for w in words if len(w) > 1]
        ngrams = list(zip(*[words[i:] for i in range(n)]))
        ngram_strs = [" ".join(ng) for ng in ngrams]
        counter.update(ngram_strs)
    return counter.most_common(top_n)


def classwise_ngrams(
    df: pd.DataFrame,
    n: int = 2,
    text_col: str = "text_normalized",
    remove_stopwords: bool = True,
    top_n: int = 15,
) -> Dict[str, List[Tuple[str, int]]]:
    """Extract top n-grams per class."""
    result = {}
    for label in CLASS_ORDER:
        subset = df[df["label"] == label]
        result[label] = extract_ngrams(
            subset[text_col], n=n,
            remove_stopwords=remove_stopwords, top_n=top_n,
        )
    return result


# ---------------------------------------------------------------------------
# 4. Stopword Analysis
# ---------------------------------------------------------------------------

def stopword_comparison(texts: pd.Series, top_n: int = 20) -> Dict[str, Any]:
    """Compare word frequencies with and without stopword removal."""
    raw_freq = word_frequency(texts, remove_stopwords=False, top_n=top_n)
    clean_freq = word_frequency(texts, remove_stopwords=True, top_n=top_n)

    return {
        "raw_top_words": [{"word": w, "count": c} for w, c in raw_freq],
        "no_stopwords_top_words": [{"word": w, "count": c} for w, c in clean_freq],
        "stopwords_available": bool(STOPWORDS),
        "stopword_count": len(STOPWORDS),
    }


# ---------------------------------------------------------------------------
# 5. Vocabulary Analysis
# ---------------------------------------------------------------------------

def vocabulary_analysis(df: pd.DataFrame, text_col: str = "text_normalized") -> Dict[str, Any]:
    """Analyze vocabulary size, overlap, and class-specific terms."""
    # Overall vocabulary
    all_words = Counter()
    class_vocabs = {}

    for label in CLASS_ORDER:
        subset = df[df["label"] == label]
        class_counter = Counter()
        for text in subset[text_col]:
            words = [w for w in tokenize_simple(text) if len(w) > 1]
            class_counter.update(words)
            all_words.update(words)
        class_vocabs[label] = set(class_counter.keys())

    total_vocab = set(all_words.keys())

    # Rare words (appearing <= 2 times)
    rare_words = {w for w, c in all_words.items() if c <= 2}

    # Common vocabulary (in all classes)
    common_vocab = set.intersection(*class_vocabs.values()) if class_vocabs else set()

    # Class-specific vocabulary
    class_specific = {}
    for label in CLASS_ORDER:
        other_vocabs = set.union(
            *[v for l, v in class_vocabs.items() if l != label]
        ) if len(class_vocabs) > 1 else set()
        class_specific[label] = len(class_vocabs.get(label, set()) - other_vocabs)

    result = {
        "total_vocabulary_size": len(total_vocab),
        "total_word_occurrences": sum(all_words.values()),
        "rare_words_count": len(rare_words),
        "rare_words_percentage": round(len(rare_words) / max(len(total_vocab), 1) * 100, 2),
        "common_vocabulary_size": len(common_vocab),
        "class_vocabulary_sizes": {
            label: len(vocab) for label, vocab in class_vocabs.items()
        },
        "class_specific_words": class_specific,
        "most_frequent_words": [
            {"word": w, "count": c} for w, c in all_words.most_common(20)
        ],
    }
    return result


# ---------------------------------------------------------------------------
# 6. TF-IDF Exploration (EDA Only)
# ---------------------------------------------------------------------------

def tfidf_class_distinctive_terms(
    df: pd.DataFrame,
    text_col: str = "text_normalized",
    top_n: int = 15,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Lightweight TF-IDF analysis to identify class-distinctive terms.
    EDA ONLY — this vectorizer is NOT saved for modeling.
    Fit ONLY on training data.
    """
    if not HAS_SKLEARN:
        return {"error": "scikit-learn not available"}

    result = {}
    try:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            min_df=5,
            max_df=0.95,
        )
        tfidf_matrix = vectorizer.fit_transform(df[text_col])
        feature_names = vectorizer.get_feature_names_out()

        for label in CLASS_ORDER:
            mask = (df["label"] == label).values  # numpy array for sparse indexing
            if mask.sum() == 0:
                continue
            class_mean = tfidf_matrix[mask].mean(axis=0)
            class_tfidf = np.asarray(class_mean).flatten()
            top_indices = class_tfidf.argsort()[-top_n:][::-1]
            result[label] = [
                {"term": feature_names[i], "tfidf_score": round(float(class_tfidf[i]), 4)}
                for i in top_indices
            ]
    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# 7. CSV Export Helpers
# ---------------------------------------------------------------------------

def ngrams_to_dataframe(
    ngrams: List[Tuple[str, int]], label: str = "Overall"
) -> pd.DataFrame:
    """Convert ngram list to a DataFrame."""
    return pd.DataFrame(
        [(label, ng, count) for ng, count in ngrams],
        columns=["Class", "N-gram", "Frequency"],
    )


def classwise_ngrams_to_dataframe(
    cw_ngrams: Dict[str, List[Tuple[str, int]]]
) -> pd.DataFrame:
    """Convert class-wise ngrams to a single DataFrame."""
    frames = []
    for label, ngrams in cw_ngrams.items():
        frames.append(ngrams_to_dataframe(ngrams, label))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
