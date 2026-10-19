"""
Phase 1 - Dataset Validation & Preparation
============================================
Sarcasm & Irony Detection System using NLP

This script validates the raw dataset and generates comprehensive
quality reports. It does NOT perform any NLP preprocessing.

Usage:
    python src/preprocessing/validate_dataset.py

Outputs:
    data/reports/dataset_summary.json
    data/reports/class_distribution.csv
    data/reports/duplicate_report.csv
    data/reports/overlap_report.csv
    data/reports/missing_values.csv
    data/reports/label_validation.csv
    data/reports/validation_report.md
"""

import os
import sys
import json
import logging
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ml-service/
RAW_DIR = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "data" / "reports"

TRAIN_FILE = RAW_DIR / "train.csv"
TEST_FILE = RAW_DIR / "test.csv"

EXPECTED_TEXT_COL = "tweets"
EXPECTED_LABEL_COL = "class"

STANDARD_LABELS = {"Regular": 0, "Sarcasm": 1, "Irony": 2, "Figurative": 3}

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def verify_files():
    """Step 1: Verify that raw files exist and report sizes."""
    logger.info("=" * 60)
    logger.info("STEP 1: VERIFY FILES")
    logger.info("=" * 60)
    results = {}
    for name, path in [("train.csv", TRAIN_FILE), ("test.csv", TEST_FILE)]:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        logger.info(f"  {name}: exists={exists}, size={size:,} bytes, path={path}")
        results[name] = {"exists": exists, "size_bytes": size, "path": str(path)}
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        logger.error("Required files missing. Cannot continue.")
        sys.exit(1)
    return results


def load_data():
    """Step 2: Load datasets and display basic info."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: LOAD DATA")
    logger.info("=" * 60)
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    for name, df in [("TRAIN", train_df), ("TEST", test_df)]:
        logger.info(f"\n--- {name} ---")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Dtypes:\n{df.dtypes.to_string()}")
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        logger.info(f"  First 3 rows:\n{df.head(3).to_string()}")
        logger.info(f"  Last 3 rows:\n{df.tail(3).to_string()}")
    return train_df, test_df


def validate_columns(train_df, test_df):
    """Step 3: Column validation."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: COLUMN VALIDATION")
    logger.info("=" * 60)
    results = {}
    for name, df in [("train", train_df), ("test", test_df)]:
        cols = list(df.columns)
        has_text = EXPECTED_TEXT_COL in cols
        has_label = EXPECTED_LABEL_COL in cols
        unexpected = [c for c in cols if c not in [EXPECTED_TEXT_COL, EXPECTED_LABEL_COL]]
        whitespace_cols = [c for c in cols if c != c.strip()]
        dup_cols = [c for c in cols if cols.count(c) > 1]
        logger.info(f"\n--- {name} ---")
        logger.info(f"  Columns: {cols}")
        logger.info(f"  Has text column ('{EXPECTED_TEXT_COL}'): {has_text}")
        logger.info(f"  Has label column ('{EXPECTED_LABEL_COL}'): {has_label}")
        logger.info(f"  Unexpected columns: {unexpected if unexpected else 'None'}")
        logger.info(f"  Whitespace in column names: {whitespace_cols if whitespace_cols else 'None'}")
        logger.info(f"  Duplicate column names: {dup_cols if dup_cols else 'None'}")
        results[name] = {
            "columns": cols,
            "has_text_col": has_text,
            "has_label_col": has_label,
            "unexpected_cols": unexpected,
        }
    text_col = EXPECTED_TEXT_COL
    label_col = EXPECTED_LABEL_COL
    if text_col not in train_df.columns:
        logger.error(f"Text column '{text_col}' not found. Actual: {list(train_df.columns)}")
        sys.exit(1)
    if label_col not in train_df.columns:
        logger.error(f"Label column '{label_col}' not found. Actual: {list(train_df.columns)}")
        sys.exit(1)
    return results, text_col, label_col


def validate_text(train_df, test_df, text_col):
    """Step 4: Text validation - analyze the text column."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: TEXT VALIDATION")
    logger.info("=" * 60)
    results = {}
    for name, df in [("train", train_df), ("test", test_df)]:
        text_series = df[text_col]
        null_count = int(text_series.isna().sum())
        non_null = text_series.dropna()
        str_series = non_null.astype(str)
        empty_count = int((str_series == "").sum())
        whitespace_only = int((str_series.str.strip() == "").sum()) - empty_count
        non_string_count = sum(1 for v in df[text_col].dropna() if not isinstance(v, str))

        valid_text = str_series[str_series.str.strip() != ""]
        char_lengths = valid_text.str.len()
        word_counts = valid_text.str.split().str.len()

        stats = {
            "total_rows": len(df),
            "null_values": null_count,
            "empty_strings": empty_count,
            "whitespace_only": whitespace_only,
            "non_string_values": non_string_count,
            "duplicate_texts": int(text_series.dropna().duplicated().sum()),
            "unique_texts": int(text_series.dropna().nunique()),
            "char_length_min": int(char_lengths.min()) if len(char_lengths) > 0 else 0,
            "char_length_max": int(char_lengths.max()) if len(char_lengths) > 0 else 0,
            "char_length_mean": round(float(char_lengths.mean()), 1) if len(char_lengths) > 0 else 0,
            "char_length_median": round(float(char_lengths.median()), 1) if len(char_lengths) > 0 else 0,
            "word_count_min": int(word_counts.min()) if len(word_counts) > 0 else 0,
            "word_count_max": int(word_counts.max()) if len(word_counts) > 0 else 0,
            "word_count_mean": round(float(word_counts.mean()), 1) if len(word_counts) > 0 else 0,
            "word_count_median": round(float(word_counts.median()), 1) if len(word_counts) > 0 else 0,
            "extremely_short_texts": int((char_lengths <= 5).sum()),
            "extremely_long_texts": int((char_lengths > 500).sum()),
        }

        logger.info(f"\n--- {name} ---")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")
        results[name] = stats
    return results


def validate_labels(train_df, test_df, label_col):
    """Step 5: Label validation - inspect all unique values."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: LABEL VALIDATION")
    logger.info("=" * 60)
    label_report_rows = []
    results = {}
    for name, df in [("train", train_df), ("test", test_df)]:
        labels = df[label_col]
        unique_raw = labels.dropna().unique()
        value_counts = labels.value_counts(dropna=False)
        logger.info(f"\n--- {name} ---")
        logger.info(f"  Unique values (incl NaN): {labels.nunique(dropna=False)} ({labels.isna().sum()} NaN)")
        logger.info(f"  Value counts:\n{value_counts.to_string()}")

        for raw_label in unique_raw:
            raw_str = str(raw_label)
            stripped = raw_str.strip()
            matched = None
            for std in STANDARD_LABELS:
                if stripped.lower() == std.lower():
                    matched = std
                    break
            count = int(value_counts.get(raw_label, 0))
            is_valid = matched is not None
            has_whitespace = raw_str != stripped
            label_report_rows.append({
                "dataset": name, "raw_label": raw_str, "stripped": stripped,
                "has_whitespace": has_whitespace, "count": count,
                "valid": is_valid, "standard_label": matched if matched else "UNKNOWN",
            })
            if has_whitespace:
                logger.warning(f"  Label has whitespace: '{raw_str}' -> '{stripped}'")
            if not is_valid:
                logger.warning(f"  INVALID/UNEXPECTED label: '{raw_str}' (count={count})")

        nan_count = int(labels.isna().sum())
        if nan_count > 0:
            label_report_rows.append({
                "dataset": name, "raw_label": "NaN", "stripped": "NaN",
                "has_whitespace": False, "count": nan_count,
                "valid": False, "standard_label": "MISSING",
            })
            logger.warning(f"  Missing labels (NaN): {nan_count}")

        results[name] = {
            "unique_labels": [str(x) for x in unique_raw],
            "nan_count": nan_count,
            "value_counts": {str(k): int(v) for k, v in value_counts.items()},
        }

    label_df = pd.DataFrame(label_report_rows)
    label_path = REPORTS_DIR / "label_validation.csv"
    label_df.to_csv(label_path, index=False)
    logger.info(f"\n  Saved label validation report: {label_path}")
    return results, label_df


def analyze_missing(train_df, test_df, text_col, label_col):
    """Step 6: Missing data analysis."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6: MISSING DATA ANALYSIS")
    logger.info("=" * 60)
    missing_rows = []
    results = {}
    for name, df in [("train", train_df), ("test", test_df)]:
        total = len(df)
        for col in df.columns:
            missing = int(df[col].isna().sum())
            pct = round(missing / total * 100, 2)
            missing_rows.append({
                "dataset": name, "column": col,
                "total_rows": total, "missing_count": missing, "missing_pct": pct,
            })
            logger.info(f"  {name}.{col}: {missing}/{total} missing ({pct}%)")
        results[name] = {col: int(df[col].isna().sum()) for col in df.columns}

    missing_df = pd.DataFrame(missing_rows)
    missing_path = REPORTS_DIR / "missing_values.csv"
    missing_df.to_csv(missing_path, index=False)
    logger.info(f"\n  Saved missing values report: {missing_path}")
    return results, missing_df


def analyze_duplicates(train_df, test_df, text_col, label_col):
    """Step 7: Duplicate analysis (exact rows, text, text+label, conflicting)."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 7: DUPLICATE ANALYSIS")
    logger.info("=" * 60)
    dup_report_rows = []
    results = {}
    for name, df in [("train", train_df), ("test", test_df)]:
        total = len(df)
        exact_dups = int(df.duplicated().sum())
        text_dups = int(df[text_col].dropna().duplicated().sum())
        unique_texts = int(df[text_col].dropna().nunique())
        text_label_dups = int(df[[text_col, label_col]].dropna().duplicated().sum())

        non_null = df.dropna(subset=[text_col, label_col])
        text_groups = non_null.groupby(text_col)[label_col].nunique()
        conflicting_texts = text_groups[text_groups > 1]
        num_conflicting = len(conflicting_texts)

        logger.info(f"\n--- {name} ---")
        logger.info(f"  Total rows: {total}")
        logger.info(f"  A. Exact duplicate rows: {exact_dups}")
        logger.info(f"  B. Duplicate tweet texts: {text_dups} (unique: {unique_texts})")
        logger.info(f"  C. Duplicate (text+label) pairs: {text_label_dups}")
        logger.info(f"  D. Same text with DIFFERENT labels: {num_conflicting}")

        dup_report_rows.append({
            "dataset": name, "total_rows": total,
            "exact_duplicate_rows": exact_dups,
            "duplicate_texts": text_dups, "unique_texts": unique_texts,
            "duplicate_text_label_pairs": text_label_dups,
            "conflicting_label_texts": num_conflicting,
        })
        results[name] = dup_report_rows[-1].copy()

        if num_conflicting > 0:
            logger.info(f"\n  Conflicting label examples (up to 10):")
            for tweet_text in conflicting_texts.index[:10]:
                subset = non_null[non_null[text_col] == tweet_text]
                labels = subset[label_col].value_counts().to_dict()
                logger.info(f"    Text: {str(tweet_text)[:80]}...")
                logger.info(f"    Labels: {labels}")

    dup_df = pd.DataFrame(dup_report_rows)
    dup_path = REPORTS_DIR / "duplicate_report.csv"
    dup_df.to_csv(dup_path, index=False)
    logger.info(f"\n  Saved duplicate report: {dup_path}")
    return results, dup_df


def analyze_overlap(train_df, test_df, text_col, label_col):
    """Step 8/9: Train/test overlap and leakage analysis."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 8-9: TRAIN/TEST OVERLAP and LEAKAGE ANALYSIS")
    logger.info("=" * 60)
    train_texts = set(train_df[text_col].dropna().unique())
    test_texts = set(test_df[text_col].dropna().unique())
    overlap_texts = train_texts & test_texts
    num_overlap = len(overlap_texts)

    logger.info(f"  Unique train texts: {len(train_texts)}")
    logger.info(f"  Unique test texts: {len(test_texts)}")
    logger.info(f"  Overlapping unique texts: {num_overlap}")

    train_affected = train_df[train_df[text_col].isin(overlap_texts)]
    test_affected = test_df[test_df[text_col].isin(overlap_texts)]
    logger.info(f"  Affected training rows: {len(train_affected)}")
    logger.info(f"  Affected test rows: {len(test_affected)}")

    overlap_rows = []
    for text in list(overlap_texts):
        train_labels = train_df[train_df[text_col] == text][label_col].dropna().unique()
        test_labels = test_df[test_df[text_col] == text][label_col].dropna().unique()
        train_label_str = ", ".join(str(x) for x in train_labels)
        test_label_str = ", ".join(str(x) for x in test_labels)
        same_label = set(str(x) for x in train_labels) == set(str(x) for x in test_labels)
        overlap_rows.append({
            "tweet": text, "train_label": train_label_str,
            "test_label": test_label_str, "same_label": same_label,
        })

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_path = REPORTS_DIR / "overlap_report.csv"
    overlap_df.to_csv(overlap_path, index=False)
    logger.info(f"  Saved overlap report: {overlap_path}")

    label_match_count = int(overlap_df["same_label"].sum()) if len(overlap_df) > 0 else 0
    label_mismatch_count = len(overlap_df) - label_match_count
    logger.info(f"\n  LEAKAGE ANALYSIS:")
    logger.info(f"    Overlapping texts with SAME label: {label_match_count}")
    logger.info(f"    Overlapping texts with DIFFERENT label: {label_mismatch_count}")
    logger.info(f"    Leakage risk: {'HIGH' if num_overlap > 0 else 'NONE'}")
    logger.info(f"    Impact: {num_overlap} test texts could inflate performance")

    results = {
        "unique_train_texts": len(train_texts),
        "unique_test_texts": len(test_texts),
        "overlapping_unique_texts": num_overlap,
        "affected_train_rows": len(train_affected),
        "affected_test_rows": len(test_affected),
        "same_label_overlaps": label_match_count,
        "different_label_overlaps": label_mismatch_count,
        "leakage_risk": "HIGH" if num_overlap > 0 else "NONE",
    }
    return results, overlap_df


def analyze_conflicting_labels(train_df, test_df, text_col, label_col):
    """Step 11: Same-text different-label analysis across datasets."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 11: SAME-TEXT DIFFERENT-LABEL ANALYSIS")
    logger.info("=" * 60)
    combined = pd.concat([
        train_df[[text_col, label_col]].assign(source="train"),
        test_df[[text_col, label_col]].assign(source="test"),
    ], ignore_index=True)
    combined_valid = combined.dropna(subset=[text_col, label_col])
    text_labels = combined_valid.groupby(text_col)[label_col].nunique()
    multi_label = text_labels[text_labels > 1]
    logger.info(f"  Total texts with conflicting labels (across train+test): {len(multi_label)}")

    if len(multi_label) > 0:
        pair_counter = Counter()
        for text in multi_label.index:
            labels = sorted(combined_valid[combined_valid[text_col] == text][label_col].unique())
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    pair_counter[(labels[i], labels[j])] += 1
        logger.info(f"\n  Conflicting label pair frequencies:")
        for pair, count in pair_counter.most_common():
            logger.info(f"    {pair[0]} <-> {pair[1]}: {count} texts")

    return {"conflicting_texts_total": len(multi_label)}


def calculate_class_distribution(train_df, test_df, label_col):
    """Step 12/13: Class distribution and balance analysis."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 12-13: CLASS DISTRIBUTION and BALANCE")
    logger.info("=" * 60)
    dist_rows = []
    results = {}
    for name, df in [("train_original", train_df), ("test_original", test_df)]:
        vc = df[label_col].value_counts(dropna=False)
        total = len(df)
        logger.info(f"\n--- {name} ---")
        for label, count in vc.items():
            pct = round(count / total * 100, 2)
            label_str = str(label) if pd.notna(label) else "NaN"
            logger.info(f"  {label_str}: {count} ({pct}%)")
            dist_rows.append({
                "dataset": name, "label": label_str,
                "count": int(count), "percentage": pct,
            })
        valid_counts = vc.dropna()
        if len(valid_counts) > 0:
            ratio = valid_counts.max() / valid_counts.min() if valid_counts.min() > 0 else float("inf")
            if ratio < 1.2:
                balance = "balanced"
            elif ratio < 1.5:
                balance = "approximately balanced"
            elif ratio < 3.0:
                balance = "moderately imbalanced"
            else:
                balance = "heavily imbalanced"
            logger.info(f"  Max/Min class ratio: {ratio:.2f} -> {balance}")
            results[name] = {"ratio": round(ratio, 2), "balance": balance}

    dist_df = pd.DataFrame(dist_rows)
    dist_path = REPORTS_DIR / "class_distribution.csv"
    dist_df.to_csv(dist_path, index=False)
    logger.info(f"\n  Saved class distribution report: {dist_path}")
    return results, dist_df


def generate_summary_json(file_results, text_stats, label_results,
                          missing_results, dup_results, overlap_results,
                          class_results, clean_stats=None):
    """Step 20: Generate structured dataset_summary.json."""
    summary = {
        "dataset_name": "Kaggle 4-class Sarcasm and Irony Dataset",
        "files": file_results,
        "train_records_original": text_stats.get("train", {}).get("total_rows", 0),
        "test_records_original": text_stats.get("test", {}).get("total_rows", 0),
        "columns": {"text_column": EXPECTED_TEXT_COL, "label_column": EXPECTED_LABEL_COL},
        "classes": list(STANDARD_LABELS.keys()),
        "label_mapping": STANDARD_LABELS,
        "text_statistics": text_stats,
        "label_validation": label_results,
        "missing_data": missing_results,
        "duplicates": dup_results,
        "train_test_overlap": overlap_results,
        "class_distribution": class_results,
    }
    if clean_stats:
        summary["train_records_clean"] = clean_stats.get("train_clean", 0)
        summary["test_records_clean"] = clean_stats.get("test_clean", 0)
        summary["cleaning_decisions"] = clean_stats.get("decisions", [])

    summary_path = REPORTS_DIR / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n  Saved dataset summary: {summary_path}")
    return summary


def main():
    """Execute the complete dataset validation pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 1 - DATASET VALIDATION and PREPARATION")
    logger.info("Sarcasm and Irony Detection System using NLP")
    logger.info("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Verify files
    file_results = verify_files()

    # Step 2: Load data
    train_df, test_df = load_data()

    # Step 3: Column validation
    col_results, text_col, label_col = validate_columns(train_df, test_df)

    # Step 4: Text validation
    text_stats = validate_text(train_df, test_df, text_col)

    # Step 5: Label validation
    label_results, label_df = validate_labels(train_df, test_df, label_col)

    # Step 6: Missing data analysis
    missing_results, missing_df = analyze_missing(train_df, test_df, text_col, label_col)

    # Step 7: Duplicate analysis
    dup_results, dup_df = analyze_duplicates(train_df, test_df, text_col, label_col)

    # Step 8-9: Train/test overlap and leakage
    overlap_results, overlap_df = analyze_overlap(train_df, test_df, text_col, label_col)

    # Step 11: Same-text different-label analysis
    conflict_results = analyze_conflicting_labels(train_df, test_df, text_col, label_col)

    # Step 12-13: Class distribution
    class_results, dist_df = calculate_class_distribution(train_df, test_df, label_col)

    # Step 20: Generate summary JSON
    summary = generate_summary_json(
        file_results, text_stats, label_results,
        missing_results, dup_results, overlap_results,
        class_results
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Reports saved to: {REPORTS_DIR}")

    return summary


if __name__ == "__main__":
    main()
