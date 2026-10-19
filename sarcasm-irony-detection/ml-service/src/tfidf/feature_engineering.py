"""
Phase 5 - Feature Engineering
============================================
Sarcasm & Irony Detection System using NLP

Extracts linguistic and social-media features from raw text
for combining with TF-IDF sparse matrices.

Feature Sets:
    A — TF-IDF only (no additional features)
    B — TF-IDF + basic text features (word_count, char_count, avg_word_len)
    C — TF-IDF + social-media features (url, mention, hashtag, emoji counts)
    D — TF-IDF + style features (punctuation, caps, repeated chars)
    E — TF-IDF + all engineered features (B + C + D)
"""

import logging
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler

logger = logging.getLogger(__name__)

# Reuse Phase 3 patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")

try:
    import emoji
    def _count_emojis(text):
        return emoji.emoji_count(text)
except ImportError:
    def _count_emojis(text):
        return 0


def _safe_str(text):
    """Ensure text is a string."""
    if pd.isna(text):
        return ""
    return str(text)


# ============================================================
# Individual Feature Extractors
# ============================================================

def extract_text_features(texts) -> np.ndarray:
    """
    Extract basic text features:
        - word_count
        - character_count
        - average_word_length
    
    Returns: ndarray of shape (n_samples, 3)
    """
    features = []
    for text in texts:
        t = _safe_str(text)
        words = t.split()
        word_count = len(words)
        char_count = len(t)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        features.append([word_count, char_count, avg_word_len])
    return np.array(features, dtype=np.float64)


def extract_social_features(texts) -> np.ndarray:
    """
    Extract social-media features:
        - url_count
        - mention_count
        - hashtag_count
        - emoji_count
    
    Returns: ndarray of shape (n_samples, 4)
    """
    features = []
    for text in texts:
        t = _safe_str(text)
        url_count = len(URL_PATTERN.findall(t))
        mention_count = len(MENTION_PATTERN.findall(t))
        hashtag_count = len(HASHTAG_PATTERN.findall(t))
        emoji_count = _count_emojis(t)
        features.append([url_count, mention_count, hashtag_count, emoji_count])
    return np.array(features, dtype=np.float64)


def extract_style_features(texts) -> np.ndarray:
    """
    Extract stylistic features:
        - exclamation_count
        - question_count
        - punctuation_count (total)
        - uppercase_count (words that are all caps and len > 1)
        - uppercase_ratio
        - repeated_character_count
    
    Returns: ndarray of shape (n_samples, 6)
    """
    features = []
    for text in texts:
        t = _safe_str(text)
        excl = t.count("!")
        quest = t.count("?")
        punct = sum(1 for c in t if c in "!?.,;:\"'()-")
        words = t.split()
        upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        upper_ratio = sum(1 for c in t if c.isupper()) / max(len(t), 1)
        repeated = len(REPEATED_CHAR_PATTERN.findall(t))
        features.append([excl, quest, punct, upper_words, upper_ratio, repeated])
    return np.array(features, dtype=np.float64)


# ============================================================
# Feature Set Builder
# ============================================================

FEATURE_SET_NAMES = {
    "A": [],                        # TF-IDF only
    "B": ["text"],                  # TF-IDF + basic text
    "C": ["social"],                # TF-IDF + social media
    "D": ["style"],                 # TF-IDF + style
    "E": ["text", "social", "style"],  # TF-IDF + all
}

FEATURE_COLUMNS = {
    "text": ["word_count", "char_count", "avg_word_len"],
    "social": ["url_count", "mention_count", "hashtag_count", "emoji_count"],
    "style": ["excl_count", "quest_count", "punct_count",
              "upper_word_count", "upper_ratio", "repeated_char_count"],
}


def build_feature_set(texts, feature_set_name: str = "A") -> np.ndarray:
    """
    Build numeric feature matrix for a given feature set.

    Args:
        texts: Iterable of raw text strings (NOT preprocessed — use original).
        feature_set_name: One of 'A', 'B', 'C', 'D', 'E'.

    Returns:
        ndarray of shape (n_samples, n_features).
        Returns None for feature set 'A' (TF-IDF only).
    """
    groups = FEATURE_SET_NAMES.get(feature_set_name, [])
    if not groups:
        return None

    parts = []
    for group in groups:
        if group == "text":
            parts.append(extract_text_features(texts))
        elif group == "social":
            parts.append(extract_social_features(texts))
        elif group == "style":
            parts.append(extract_style_features(texts))

    if not parts:
        return None

    features = np.hstack(parts)
    logger.info(f"Feature set '{feature_set_name}': {features.shape[1]} features")
    return features


def get_feature_names(feature_set_name: str) -> list:
    """Get feature column names for a feature set."""
    groups = FEATURE_SET_NAMES.get(feature_set_name, [])
    names = []
    for group in groups:
        names.extend(FEATURE_COLUMNS.get(group, []))
    return names


def combine_tfidf_features(
    tfidf_matrix: sparse.spmatrix,
    numeric_features: np.ndarray,
    fit_scaler: bool = True,
    scaler=None,
):
    """
    Combine TF-IDF sparse matrix with numeric features.

    Scales numeric features with MaxAbsScaler (sparse-compatible)
    then hstacks with TF-IDF.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix.
        numeric_features: Dense numeric feature array.
        fit_scaler: If True, fit a new scaler. If False, use provided scaler.
        scaler: Pre-fitted scaler (used when fit_scaler=False).

    Returns:
        Tuple of (combined_sparse_matrix, fitted_scaler).
    """
    if numeric_features is None:
        return tfidf_matrix, None

    # Scale numeric features
    if fit_scaler:
        scaler = MaxAbsScaler()
        scaled = scaler.fit_transform(numeric_features)
    else:
        scaled = scaler.transform(numeric_features)

    # Convert to sparse
    scaled_sparse = sparse.csr_matrix(scaled)

    # Combine
    combined = sparse.hstack([tfidf_matrix, scaled_sparse], format="csr")

    logger.info(
        f"Combined: TF-IDF({tfidf_matrix.shape[1]}) + "
        f"features({numeric_features.shape[1]}) = {combined.shape[1]} total"
    )
    return combined, scaler
