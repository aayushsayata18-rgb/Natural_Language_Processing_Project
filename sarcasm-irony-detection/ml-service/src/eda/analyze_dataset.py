"""
analyze_dataset.py — Dataset overview, quality checks, and class distribution.

Phase 2: Exploratory Data Analysis
Project: Sarcasm & Irony Detection System
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, Tuple


# ---------------------------------------------------------------------------
# 1. Dataset Overview
# ---------------------------------------------------------------------------

def load_datasets(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned train and test CSVs from Phase 1."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def dataset_overview(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Generate a comprehensive overview of a single dataset."""
    overview = {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "unique_labels": int(df["label"].nunique()) if "label" in df.columns else None,
        "unique_texts": int(df["text"].nunique()) if "text" in df.columns else None,
    }
    return overview


def dataset_overview_table(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Create a concise overview comparison table."""
    data = {
        "Dataset": ["Train", "Test", "Total"],
        "Records": [len(train_df), len(test_df), len(train_df) + len(test_df)],
        "Columns": [len(train_df.columns), len(test_df.columns), len(train_df.columns)],
        "Unique Texts": [
            train_df["text"].nunique(),
            test_df["text"].nunique(),
            train_df["text"].nunique() + test_df["text"].nunique(),
        ],
        "Memory (MB)": [
            round(train_df.memory_usage(deep=True).sum() / 1024**2, 2),
            round(test_df.memory_usage(deep=True).sum() / 1024**2, 2),
            round(
                (train_df.memory_usage(deep=True).sum()
                 + test_df.memory_usage(deep=True).sum()) / 1024**2, 2
            ),
        ],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 2. Data Quality Recheck
# ---------------------------------------------------------------------------

def quality_check(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """
    Perform a lightweight EDA-level quality check on the cleaned dataset.
    Does NOT modify the data — only reports issues.
    """
    valid_labels = {"Regular", "Sarcasm", "Irony", "Figurative"}

    checks = {
        "dataset": name,
        "total_rows": len(df),
        # Missing values
        "missing_text": int(df["text"].isna().sum()) if "text" in df.columns else None,
        "missing_label": int(df["label"].isna().sum()) if "label" in df.columns else None,
        "missing_label_id": int(df["label_id"].isna().sum()) if "label_id" in df.columns else None,
        # Empty / whitespace
        "empty_strings": int((df["text"] == "").sum()) if "text" in df.columns else None,
        "whitespace_only": int(df["text"].str.strip().eq("").sum()) if "text" in df.columns else None,
        # Duplicates
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_texts": int(df["text"].duplicated().sum()) if "text" in df.columns else None,
        # Invalid labels
        "invalid_labels": int(
            (~df["label"].isin(valid_labels)).sum()
        ) if "label" in df.columns else None,
        "unique_labels_found": sorted(df["label"].unique().tolist()) if "label" in df.columns else None,
        # Data types
        "text_dtype": str(df["text"].dtype) if "text" in df.columns else None,
        "label_dtype": str(df["label"].dtype) if "label" in df.columns else None,
        "label_id_dtype": str(df["label_id"].dtype) if "label_id" in df.columns else None,
    }

    # Overall assessment
    issues = []
    if checks["missing_text"] and checks["missing_text"] > 0:
        issues.append(f"Found {checks['missing_text']} missing text values")
    if checks["missing_label"] and checks["missing_label"] > 0:
        issues.append(f"Found {checks['missing_label']} missing labels")
    if checks["empty_strings"] and checks["empty_strings"] > 0:
        issues.append(f"Found {checks['empty_strings']} empty strings")
    if checks["whitespace_only"] and checks["whitespace_only"] > 0:
        issues.append(f"Found {checks['whitespace_only']} whitespace-only strings")
    if checks["invalid_labels"] and checks["invalid_labels"] > 0:
        issues.append(f"Found {checks['invalid_labels']} invalid labels")
    if checks["duplicate_rows"] and checks["duplicate_rows"] > 0:
        issues.append(f"Found {checks['duplicate_rows']} duplicate rows")

    checks["issues_found"] = len(issues)
    checks["issues"] = issues
    checks["quality_status"] = "CLEAN" if len(issues) == 0 else "ISSUES_FOUND"

    return checks


# ---------------------------------------------------------------------------
# 3. Class Distribution
# ---------------------------------------------------------------------------

def class_distribution(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Calculate class counts and percentages."""
    counts = df["label"].value_counts()
    pcts = df["label"].value_counts(normalize=True) * 100

    dist = pd.DataFrame({
        "Dataset": name,
        "Class": counts.index,
        "Count": counts.values,
        "Percentage": pcts.values.round(2),
    })

    # Ensure consistent class order
    class_order = ["Regular", "Sarcasm", "Irony", "Figurative"]
    dist["Class"] = pd.Categorical(dist["Class"], categories=class_order, ordered=True)
    dist = dist.sort_values("Class").reset_index(drop=True)

    return dist


def class_balance_analysis(train_dist: pd.DataFrame, test_dist: pd.DataFrame) -> Dict[str, Any]:
    """Analyze class balance and calculate majority/minority ratio."""
    analysis = {}

    for dist, name in [(train_dist, "train"), (test_dist, "test")]:
        counts = dist["Count"].values
        max_count = int(counts.max())
        min_count = int(counts.min())
        ratio = round(max_count / min_count, 2) if min_count > 0 else float("inf")

        # Determine balance category
        if ratio <= 1.1:
            balance = "balanced"
        elif ratio <= 1.3:
            balance = "approximately balanced"
        elif ratio <= 2.0:
            balance = "moderately imbalanced"
        else:
            balance = "heavily imbalanced"

        analysis[name] = {
            "majority_class": dist.loc[dist["Count"].idxmax(), "Class"],
            "minority_class": dist.loc[dist["Count"].idxmin(), "Class"],
            "majority_count": max_count,
            "minority_count": min_count,
            "majority_minority_ratio": ratio,
            "balance_category": balance,
            "implication": _balance_implication(balance),
        }

    return analysis


def _balance_implication(balance: str) -> str:
    """Return evaluation implication based on balance category."""
    if balance in ("balanced", "approximately balanced"):
        return (
            "Accuracy is a useful metric, but Macro F1 should still be reported "
            "to ensure all classes are evaluated fairly."
        )
    elif balance == "moderately imbalanced":
        return (
            "Accuracy may be slightly misleading. Macro F1 and per-class "
            "precision/recall should be primary evaluation metrics. "
            "Consider class weighting during training."
        )
    else:
        return (
            "Accuracy is unreliable. Use Macro F1, per-class metrics, and "
            "confusion matrices. Stratified sampling and class weighting "
            "are strongly recommended."
        )
