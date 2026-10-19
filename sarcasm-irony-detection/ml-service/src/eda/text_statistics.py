"""
text_statistics.py — Text length features, descriptive statistics, and outlier analysis.

Phase 2: Exploratory Data Analysis
Project: Sarcasm & Irony Detection System
"""

import pandas as pd
import numpy as np
import string
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# 1. Text Length Features
# ---------------------------------------------------------------------------

def compute_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EDA-only text features for each tweet.
    These are analytical features — they do NOT replace the original text.
    """
    features = df[["text", "label", "label_id"]].copy()

    features["character_count"] = features["text"].str.len()
    features["word_count"] = features["text"].str.split().str.len()
    features["punctuation_count"] = features["text"].apply(
        lambda x: sum(1 for c in str(x) if c in string.punctuation)
    )
    features["whitespace_count"] = features["text"].apply(
        lambda x: sum(1 for c in str(x) if c.isspace())
    )

    return features


# ---------------------------------------------------------------------------
# 2. Descriptive Statistics
# ---------------------------------------------------------------------------

def descriptive_stats(features: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Calculate descriptive statistics for a numeric column."""
    series = features[column]
    stats = {
        "min": int(series.min()),
        "max": int(series.max()),
        "mean": round(float(series.mean()), 2),
        "median": float(series.median()),
        "std": round(float(series.std()), 2),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
        "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
    }
    return stats


def classwise_stats(features: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculate descriptive statistics by class."""
    class_order = ["Regular", "Sarcasm", "Irony", "Figurative"]
    rows = []
    for label in class_order:
        subset = features[features["label"] == label][column]
        if len(subset) == 0:
            continue
        rows.append({
            "Class": label,
            "Count": len(subset),
            "Min": int(subset.min()),
            "Max": int(subset.max()),
            "Mean": round(float(subset.mean()), 2),
            "Median": float(subset.median()),
            "Std": round(float(subset.std()), 2),
            "Q1": float(subset.quantile(0.25)),
            "Q3": float(subset.quantile(0.75)),
        })
    return pd.DataFrame(rows)


def overall_and_classwise_stats(
    features: pd.DataFrame,
    columns: List[str] = None,
) -> Dict[str, Any]:
    """Generate overall and class-wise statistics for multiple columns."""
    if columns is None:
        columns = ["character_count", "word_count", "punctuation_count"]

    result = {}
    for col in columns:
        result[col] = {
            "overall": descriptive_stats(features, col),
            "by_class": classwise_stats(features, col).to_dict(orient="records"),
        }
    return result


# ---------------------------------------------------------------------------
# 3. Text Length Statistics Table
# ---------------------------------------------------------------------------

def text_length_statistics_table(features: pd.DataFrame) -> pd.DataFrame:
    """Create a comprehensive text-length statistics table for reporting."""
    class_order = ["Overall", "Regular", "Sarcasm", "Irony", "Figurative"]
    rows = []

    for metric in ["character_count", "word_count"]:
        metric_label = "Characters" if metric == "character_count" else "Words"

        # Overall
        series = features[metric]
        rows.append({
            "Metric": metric_label,
            "Class": "Overall",
            "Count": len(series),
            "Min": int(series.min()),
            "Max": int(series.max()),
            "Mean": round(float(series.mean()), 2),
            "Median": float(series.median()),
            "Std": round(float(series.std()), 2),
        })

        # Per class
        for label in ["Regular", "Sarcasm", "Irony", "Figurative"]:
            subset = features[features["label"] == label][metric]
            if len(subset) == 0:
                continue
            rows.append({
                "Metric": metric_label,
                "Class": label,
                "Count": len(subset),
                "Min": int(subset.min()),
                "Max": int(subset.max()),
                "Mean": round(float(subset.mean()), 2),
                "Median": float(subset.median()),
                "Std": round(float(subset.std()), 2),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Outlier Analysis
# ---------------------------------------------------------------------------

def identify_outliers(
    features: pd.DataFrame,
    column: str,
    multiplier: float = 1.5,
) -> Dict[str, Any]:
    """
    Identify outliers using IQR method.
    Does NOT remove them — only flags and reports.
    """
    q1 = features[column].quantile(0.25)
    q3 = features[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    low_outliers = features[features[column] < lower_bound]
    high_outliers = features[features[column] > upper_bound]

    result = {
        "column": column,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "low_outlier_count": len(low_outliers),
        "high_outlier_count": len(high_outliers),
        "total_outliers": len(low_outliers) + len(high_outliers),
        "outlier_percentage": round(
            (len(low_outliers) + len(high_outliers)) / len(features) * 100, 2
        ),
    }

    # Class distribution of outliers
    all_outliers = pd.concat([low_outliers, high_outliers])
    if len(all_outliers) > 0:
        result["outlier_class_distribution"] = (
            all_outliers["label"].value_counts().to_dict()
        )
    else:
        result["outlier_class_distribution"] = {}

    # Sample extreme values
    if len(high_outliers) > 0:
        top = high_outliers.nlargest(3, column)
        result["extreme_high_samples"] = top[["text", "label", column]].to_dict(
            orient="records"
        )
    if len(low_outliers) > 0:
        bottom = low_outliers.nsmallest(3, column)
        result["extreme_low_samples"] = bottom[["text", "label", column]].to_dict(
            orient="records"
        )

    return result


def outlier_summary(features: pd.DataFrame) -> Dict[str, Any]:
    """Generate outlier analysis for all key text features."""
    columns = ["character_count", "word_count", "punctuation_count"]
    summary = {}
    for col in columns:
        summary[col] = identify_outliers(features, col)
    return summary
