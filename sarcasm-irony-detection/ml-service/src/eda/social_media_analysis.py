"""
social_media_analysis.py — Social-media pattern analysis for tweets.

Analyzes URLs, @mentions, hashtags, emojis, punctuation, capitalization,
and repeated characters. All analysis is read-only — no text modification.

Phase 2: Exploratory Data Analysis
Project: Sarcasm & Irony Detection System
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List
from collections import Counter

try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False


CLASS_ORDER = ["Regular", "Sarcasm", "Irony", "Figurative"]

# ---------------------------------------------------------------------------
# Pattern regexes
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')
REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{2,}')  # 3+ repeated chars
REPEATED_PUNCT_PATTERN = re.compile(r'([!?.])\1+')
ELLIPSIS_PATTERN = re.compile(r'\.{2,}')
EXCLAM_QUESTION_PATTERN = re.compile(r'[!?]{2,}')


# ---------------------------------------------------------------------------
# 1. Feature Extraction
# ---------------------------------------------------------------------------

def extract_social_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all social-media features from tweet text. Read-only."""
    features = df[["text", "label", "label_id"]].copy()

    # URL features
    features["url_count"] = features["text"].apply(
        lambda x: len(URL_PATTERN.findall(str(x)))
    )
    features["has_url"] = (features["url_count"] > 0).astype(int)

    # Mention features
    features["mention_count"] = features["text"].apply(
        lambda x: len(MENTION_PATTERN.findall(str(x)))
    )
    features["has_mention"] = (features["mention_count"] > 0).astype(int)

    # Hashtag features
    features["hashtag_count"] = features["text"].apply(
        lambda x: len(HASHTAG_PATTERN.findall(str(x)))
    )
    features["has_hashtag"] = (features["hashtag_count"] > 0).astype(int)

    # Emoji features
    if HAS_EMOJI:
        features["emoji_count"] = features["text"].apply(
            lambda x: len([c for c in str(x) if emoji.is_emoji(c)])
        )
    else:
        features["emoji_count"] = 0
    features["has_emoji"] = (features["emoji_count"] > 0).astype(int)

    # Punctuation features
    features["exclamation_count"] = features["text"].str.count("!")
    features["question_count"] = features["text"].str.count(r"\?")
    features["ellipsis_count"] = features["text"].apply(
        lambda x: len(ELLIPSIS_PATTERN.findall(str(x)))
    )
    features["repeated_punct_count"] = features["text"].apply(
        lambda x: len(EXCLAM_QUESTION_PATTERN.findall(str(x)))
    )

    # Capitalization features
    features["uppercase_count"] = features["text"].apply(
        lambda x: sum(1 for c in str(x) if c.isupper())
    )
    features["char_count_alpha"] = features["text"].apply(
        lambda x: sum(1 for c in str(x) if c.isalpha())
    )
    features["uppercase_ratio"] = np.where(
        features["char_count_alpha"] > 0,
        (features["uppercase_count"] / features["char_count_alpha"]).round(4),
        0.0,
    )
    features["has_allcaps_word"] = features["text"].apply(
        lambda x: int(
            any(
                w.isupper() and len(w) > 1
                for w in str(x).split()
            )
        )
    )

    # Repeated character features
    features["repeated_char_count"] = features["text"].apply(
        lambda x: len(REPEATED_CHAR_PATTERN.findall(str(x)))
    )
    features["has_repeated_chars"] = (features["repeated_char_count"] > 0).astype(int)

    return features


# ---------------------------------------------------------------------------
# 2. URL Analysis
# ---------------------------------------------------------------------------

def url_analysis(social_features: pd.DataFrame) -> Dict[str, Any]:
    """Analyze URL presence and distribution by class."""
    total = len(social_features)
    url_tweets = social_features["has_url"].sum()

    result = {
        "total_tweets": total,
        "tweets_with_urls": int(url_tweets),
        "percentage_with_urls": round(url_tweets / total * 100, 2),
        "by_class": _classwise_feature_pct(social_features, "has_url"),
    }
    return result


# ---------------------------------------------------------------------------
# 3. Mention Analysis
# ---------------------------------------------------------------------------

def mention_analysis(social_features: pd.DataFrame) -> Dict[str, Any]:
    """Analyze @mention presence and distribution by class."""
    total = len(social_features)
    mention_tweets = social_features["has_mention"].sum()

    result = {
        "total_tweets": total,
        "tweets_with_mentions": int(mention_tweets),
        "percentage_with_mentions": round(mention_tweets / total * 100, 2),
        "avg_mentions_per_tweet": round(
            social_features["mention_count"].mean(), 2
        ),
        "by_class": _classwise_feature_pct(social_features, "has_mention"),
        "avg_by_class": _classwise_feature_mean(social_features, "mention_count"),
    }
    return result


# ---------------------------------------------------------------------------
# 4. Hashtag Analysis
# ---------------------------------------------------------------------------

def hashtag_analysis(social_features: pd.DataFrame) -> Dict[str, Any]:
    """Analyze hashtag presence, frequency, and distribution by class."""
    total = len(social_features)
    ht_tweets = social_features["has_hashtag"].sum()

    # Extract all hashtags
    all_hashtags = []
    for text in social_features["text"]:
        all_hashtags.extend(HASHTAG_PATTERN.findall(str(text).lower()))

    hashtag_freq = Counter(all_hashtags).most_common(20)

    result = {
        "total_tweets": total,
        "tweets_with_hashtags": int(ht_tweets),
        "percentage_with_hashtags": round(ht_tweets / total * 100, 2),
        "avg_hashtags_per_tweet": round(
            social_features["hashtag_count"].mean(), 2
        ),
        "top_hashtags": [{"hashtag": h, "count": c} for h, c in hashtag_freq],
        "by_class": _classwise_feature_pct(social_features, "has_hashtag"),
        "avg_by_class": _classwise_feature_mean(social_features, "hashtag_count"),
    }
    return result


# ---------------------------------------------------------------------------
# 5. Emoji Analysis
# ---------------------------------------------------------------------------

def emoji_analysis(social_features: pd.DataFrame) -> Dict[str, Any]:
    """Analyze emoji presence and distribution by class."""
    total = len(social_features)
    emoji_tweets = social_features["has_emoji"].sum()

    result = {
        "total_tweets": total,
        "tweets_with_emojis": int(emoji_tweets),
        "percentage_with_emojis": round(emoji_tweets / total * 100, 2),
        "avg_emojis_per_tweet": round(
            social_features["emoji_count"].mean(), 2
        ),
        "by_class": _classwise_feature_pct(social_features, "has_emoji"),
    }

    # Top emojis if emoji library available
    if HAS_EMOJI:
        all_emojis = []
        for text in social_features["text"]:
            all_emojis.extend([c for c in str(text) if emoji.is_emoji(c)])
        emoji_freq = Counter(all_emojis).most_common(15)
        result["top_emojis"] = [{"emoji": e, "count": c} for e, c in emoji_freq]

    return result


# ---------------------------------------------------------------------------
# 6. Punctuation Analysis
# ---------------------------------------------------------------------------

def punctuation_analysis(social_features: pd.DataFrame) -> Dict[str, Any]:
    """Analyze punctuation patterns by class."""
    result = {}
    for feat in ["exclamation_count", "question_count", "ellipsis_count", "repeated_punct_count"]:
        has_feat = (social_features[feat] > 0).astype(int)
        result[feat] = {
            "tweets_with_feature": int(has_feat.sum()),
            "percentage": round(has_feat.sum() / len(social_features) * 100, 2),
            "avg_per_tweet": round(social_features[feat].mean(), 2),
            "by_class": _classwise_feature_pct(
                social_features.assign(**{f"_has_{feat}": has_feat}),
                f"_has_{feat}",
            ),
            "avg_by_class": _classwise_feature_mean(social_features, feat),
        }
    return result


# ---------------------------------------------------------------------------
# 7. Capitalization Analysis
# ---------------------------------------------------------------------------

def capitalization_analysis(social_features: pd.DataFrame) -> Dict[str, Any]:
    """Analyze capitalization patterns by class."""
    result = {
        "avg_uppercase_ratio": round(
            social_features["uppercase_ratio"].mean(), 4
        ),
        "tweets_with_allcaps_words": int(
            social_features["has_allcaps_word"].sum()
        ),
        "pct_with_allcaps_words": round(
            social_features["has_allcaps_word"].sum() / len(social_features) * 100, 2
        ),
        "uppercase_ratio_by_class": _classwise_feature_mean(
            social_features, "uppercase_ratio"
        ),
        "allcaps_by_class": _classwise_feature_pct(
            social_features, "has_allcaps_word"
        ),
    }
    return result


# ---------------------------------------------------------------------------
# 8. Repeated Character Analysis
# ---------------------------------------------------------------------------

def repeated_char_analysis(social_features: pd.DataFrame) -> Dict[str, Any]:
    """Analyze repeated character patterns by class."""
    total = len(social_features)
    rep_tweets = social_features["has_repeated_chars"].sum()

    result = {
        "total_tweets": total,
        "tweets_with_repeated_chars": int(rep_tweets),
        "percentage_with_repeated_chars": round(rep_tweets / total * 100, 2),
        "by_class": _classwise_feature_pct(social_features, "has_repeated_chars"),
    }
    return result


# ---------------------------------------------------------------------------
# 9. Consolidated Class-Wise Comparison
# ---------------------------------------------------------------------------

def classwise_comparison_table(
    text_features: pd.DataFrame,
    social_features: pd.DataFrame,
) -> pd.DataFrame:
    """Create the consolidated class-wise comparison table."""
    rows = []
    for label in CLASS_ORDER:
        tf = text_features[text_features["label"] == label]
        sf = social_features[social_features["label"] == label]
        n = len(sf)
        if n == 0:
            continue
        rows.append({
            "Class": label,
            "Record Count": n,
            "Percentage (%)": round(n / len(social_features) * 100, 2),
            "Mean Words": round(float(tf["word_count"].mean()), 1),
            "Median Words": float(tf["word_count"].median()),
            "Mean Chars": round(float(tf["character_count"].mean()), 1),
            "Median Chars": float(tf["character_count"].median()),
            "URL %": round(sf["has_url"].mean() * 100, 1),
            "Mention %": round(sf["has_mention"].mean() * 100, 1),
            "Hashtag %": round(sf["has_hashtag"].mean() * 100, 1),
            "Emoji %": round(sf["has_emoji"].mean() * 100, 1),
            "Exclamation %": round(
                (sf["exclamation_count"] > 0).mean() * 100, 1
            ),
            "Question %": round(
                (sf["question_count"] > 0).mean() * 100, 1
            ),
            "Uppercase Ratio": round(sf["uppercase_ratio"].mean(), 4),
            "AllCaps Words %": round(sf["has_allcaps_word"].mean() * 100, 1),
            "Repeated Chars %": round(sf["has_repeated_chars"].mean() * 100, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classwise_feature_pct(df: pd.DataFrame, column: str) -> List[Dict]:
    """Calculate percentage of tweets with a binary feature, by class."""
    rows = []
    for label in CLASS_ORDER:
        subset = df[df["label"] == label]
        if len(subset) == 0:
            continue
        rows.append({
            "class": label,
            "count": int(subset[column].sum()),
            "total": len(subset),
            "percentage": round(subset[column].mean() * 100, 2),
        })
    return rows


def _classwise_feature_mean(df: pd.DataFrame, column: str) -> List[Dict]:
    """Calculate mean of a numeric feature, by class."""
    rows = []
    for label in CLASS_ORDER:
        subset = df[df["label"] == label]
        if len(subset) == 0:
            continue
        rows.append({
            "class": label,
            "mean": round(float(subset[column].mean()), 4),
        })
    return rows
