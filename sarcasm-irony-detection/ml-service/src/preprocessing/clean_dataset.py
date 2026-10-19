"""
Phase 1 - Dataset Cleaning
============================================
Sarcasm & Irony Detection System using NLP

Creates clean datasets from validated raw data.
Does NOT perform NLP preprocessing (no stemming, lemmatization, etc.)

Usage:
    python src/preprocessing/clean_dataset.py

Inputs:
    data/raw/train.csv
    data/raw/test.csv

Outputs:
    data/processed/train_clean.csv
    data/processed/test_clean.csv
    data/reports/dataset_summary.json (updated)
    data/reports/validation_report.md
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ml-service/
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
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


def load_raw_data():
    """Load raw datasets."""
    logger.info("Loading raw datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    logger.info(f"  Raw train: {train_df.shape}")
    logger.info(f"  Raw test: {test_df.shape}")
    return train_df, test_df


def standardize_labels(df, label_col, dataset_name):
    """
    Standardize label values: strip whitespace, apply consistent capitalization.
    Does NOT change semantic meaning of labels.
    """
    logger.info(f"\n  Standardizing labels for {dataset_name}...")
    original_labels = df[label_col].dropna().unique()
    logger.info(f"    Original unique labels: {list(original_labels)}")

    # Create mapping from raw to standard
    label_map = {}
    for raw in original_labels:
        raw_str = str(raw).strip()
        for std in STANDARD_LABELS:
            if raw_str.lower() == std.lower():
                label_map[raw] = std
                break

    # Apply mapping
    df[label_col] = df[label_col].map(label_map)

    new_labels = df[label_col].dropna().unique()
    logger.info(f"    Standardized unique labels: {list(new_labels)}")
    logger.info(f"    Label map applied: {label_map}")

    return df, label_map


def remove_missing_text(df, text_col, dataset_name):
    """Remove records with missing or empty text."""
    before = len(df)

    # Remove null text
    null_mask = df[text_col].isna()
    null_count = null_mask.sum()

    # Remove empty / whitespace-only text
    non_null = df[~null_mask]
    empty_mask = non_null[text_col].astype(str).str.strip() == ""
    empty_count = empty_mask.sum()

    # Apply removal
    df = df[~null_mask].copy()
    df = df[df[text_col].astype(str).str.strip() != ""].copy()

    after = len(df)
    removed = before - after

    logger.info(f"  {dataset_name}: Removed {removed} records with missing/empty text "
                f"(null={null_count}, empty/whitespace={empty_count})")

    return df, {"null_text": int(null_count), "empty_text": int(empty_count), "total_removed": int(removed)}


def remove_missing_labels(df, label_col, dataset_name):
    """Remove records with missing labels (cannot be used for supervised learning)."""
    before = len(df)
    missing_mask = df[label_col].isna()
    missing_count = int(missing_mask.sum())

    if missing_count > 0:
        # Log the missing label records before removing
        missing_records = df[missing_mask]
        logger.info(f"  {dataset_name}: Found {missing_count} records with missing labels")
        logger.info(f"    Sample missing-label records:")
        for idx, row in missing_records.head(5).iterrows():
            text_preview = str(row.get(EXPECTED_TEXT_COL, ""))[:80]
            logger.info(f"      Row {idx}: '{text_preview}...'")

        df = df[~missing_mask].copy()

    after = len(df)
    logger.info(f"  {dataset_name}: Removed {missing_count} records with missing labels "
                f"({before} -> {after})")

    return df, {"missing_labels_removed": missing_count}


def remove_invalid_labels(df, label_col, dataset_name):
    """Remove records with labels not in the standard set."""
    valid_labels = set(STANDARD_LABELS.keys())
    invalid_mask = ~df[label_col].isin(valid_labels)
    invalid_count = int(invalid_mask.sum())

    if invalid_count > 0:
        invalid_values = df[invalid_mask][label_col].unique()
        logger.warning(f"  {dataset_name}: Found {invalid_count} records with invalid labels: {list(invalid_values)}")
        df = df[~invalid_mask].copy()
    else:
        logger.info(f"  {dataset_name}: No invalid labels found")

    return df, {"invalid_labels_removed": invalid_count}


def remove_exact_duplicates(df, dataset_name):
    """Remove exact duplicate rows (all columns identical)."""
    before = len(df)
    dup_mask = df.duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    df = df[~dup_mask].copy()
    after = len(df)

    logger.info(f"  {dataset_name}: Removed {dup_count} exact duplicate rows ({before} -> {after})")
    return df, {"exact_duplicates_removed": dup_count}


def handle_train_test_overlap(train_df, test_df, text_col):
    """
    Handle train/test text overlap to prevent data leakage.

    Strategy: Remove overlapping texts from the TEST set (not training).
    This ensures the final evaluation dataset contains only truly unseen text.
    The original test file is preserved in raw/.

    Rationale:
    - We cannot guarantee the model hasn't memorized training examples.
    - Any identical text in both sets creates evaluation leakage.
    - Removing from test (not train) preserves maximum training data.
    """
    logger.info("\n  Handling train/test text overlap (leakage control)...")

    train_texts = set(train_df[text_col].dropna().unique())
    test_texts_before = len(test_df)

    overlap_mask = test_df[text_col].isin(train_texts)
    overlap_count = int(overlap_mask.sum())
    overlap_unique = len(set(test_df[overlap_mask][text_col].dropna().unique()))

    logger.info(f"    Train unique texts: {len(train_texts)}")
    logger.info(f"    Test rows before leakage removal: {test_texts_before}")
    logger.info(f"    Overlapping test rows: {overlap_count}")
    logger.info(f"    Overlapping unique texts: {overlap_unique}")

    # Remove overlapping texts from test
    test_clean = test_df[~overlap_mask].copy()
    test_texts_after = len(test_clean)

    logger.info(f"    Test rows after leakage removal: {test_texts_after}")
    logger.info(f"    Test rows removed for leakage control: {overlap_count}")

    return test_clean, {
        "overlap_unique_texts": overlap_unique,
        "overlap_test_rows_removed": overlap_count,
        "test_before": test_texts_before,
        "test_after": test_texts_after,
    }


def create_clean_format(df, text_col, label_col, dataset_name):
    """
    Create standardized clean dataset format.
    Columns: text, label, label_id
    """
    clean_df = pd.DataFrame({
        "text": df[text_col].values,
        "label": df[label_col].values,
    })
    clean_df["label_id"] = clean_df["label"].map(STANDARD_LABELS)

    logger.info(f"  {dataset_name}: Created clean format with columns: {list(clean_df.columns)}")
    logger.info(f"    Shape: {clean_df.shape}")
    logger.info(f"    Label distribution:\n{clean_df['label'].value_counts().to_string()}")

    return clean_df


def run_integrity_checks(train_clean, test_clean):
    """Step 23: Data integrity checks on clean datasets."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATA INTEGRITY CHECKS")
    logger.info("=" * 60)

    checks_passed = True

    for name, df in [("train_clean", train_clean), ("test_clean", test_clean)]:
        logger.info(f"\n--- {name} ---")

        # Check no missing text
        missing_text = df["text"].isna().sum()
        assert missing_text == 0, f"{name}: Found {missing_text} missing text values"
        logger.info(f"  [PASS] No missing text values")

        # Check no missing labels
        missing_labels = df["label"].isna().sum()
        assert missing_labels == 0, f"{name}: Found {missing_labels} missing labels"
        logger.info(f"  [PASS] No missing labels")

        # Check only four valid labels
        valid_labels = set(STANDARD_LABELS.keys())
        actual_labels = set(df["label"].unique())
        assert actual_labels.issubset(valid_labels), f"{name}: Invalid labels found: {actual_labels - valid_labels}"
        logger.info(f"  [PASS] Only valid labels: {sorted(actual_labels)}")

        # Check label_id mapping is complete
        missing_ids = df["label_id"].isna().sum()
        assert missing_ids == 0, f"{name}: Found {missing_ids} unmapped label IDs"
        logger.info(f"  [PASS] Label mapping complete (no NaN label_ids)")

        # Check expected columns
        expected_cols = {"text", "label", "label_id"}
        assert set(df.columns) == expected_cols, f"{name}: Unexpected columns: {set(df.columns) - expected_cols}"
        logger.info(f"  [PASS] Expected columns present: {sorted(df.columns)}")

        # Check no empty text
        empty_text = (df["text"].astype(str).str.strip() == "").sum()
        assert empty_text == 0, f"{name}: Found {empty_text} empty text values"
        logger.info(f"  [PASS] No empty text values")

    # Check no train/test overlap in clean data
    train_texts = set(train_clean["text"].unique())
    test_texts = set(test_clean["text"].unique())
    overlap = train_texts & test_texts
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping texts in clean data!"
    logger.info(f"\n  [PASS] No train/test text overlap in clean data")

    logger.info(f"\n  ALL INTEGRITY CHECKS PASSED")
    return checks_passed


def save_label_mapping():
    """Save the label mapping configuration."""
    mapping_path = PROCESSED_DIR / "label_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(STANDARD_LABELS, f, indent=2)
    logger.info(f"  Saved label mapping: {mapping_path}")
    return mapping_path


def generate_validation_report(cleaning_log, train_clean, test_clean):
    """Generate the final validation_report.md."""
    report_lines = []
    report_lines.append("# Phase 1 - Dataset Validation Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n## Dataset Source")
    report_lines.append("Kaggle 4-class Sarcasm and Irony Dataset")
    report_lines.append(f"\n## Raw Dataset")
    report_lines.append(f"- Train file: `data/raw/train.csv`")
    report_lines.append(f"- Test file: `data/raw/test.csv`")
    report_lines.append(f"- Original train rows: {cleaning_log['original_train']}")
    report_lines.append(f"- Original test rows: {cleaning_log['original_test']}")
    report_lines.append(f"\n## Cleaning Decisions")
    report_lines.append("")
    for decision in cleaning_log.get("decisions", []):
        report_lines.append(f"### {decision['step']}")
        report_lines.append(f"- **Action**: {decision['action']}")
        report_lines.append(f"- **Justification**: {decision['justification']}")
        report_lines.append(f"- **Impact**: {decision['impact']}")
        report_lines.append("")

    report_lines.append(f"\n## Clean Dataset Summary")
    report_lines.append(f"- Clean train rows: {len(train_clean)}")
    report_lines.append(f"- Clean test rows: {len(test_clean)}")

    report_lines.append(f"\n### Clean Train Class Distribution")
    for label, count in train_clean["label"].value_counts().items():
        pct = round(count / len(train_clean) * 100, 2)
        report_lines.append(f"- {label}: {count} ({pct}%)")

    report_lines.append(f"\n### Clean Test Class Distribution")
    for label, count in test_clean["label"].value_counts().items():
        pct = round(count / len(test_clean) * 100, 2)
        report_lines.append(f"- {label}: {count} ({pct}%)")

    report_lines.append(f"\n## Data Quality Assessment")
    report_lines.append("All integrity checks passed.")
    report_lines.append("No train/test text overlap in final clean data.")

    report_lines.append(f"\n## Important Notes")
    report_lines.append("- Raw files are preserved untouched in `data/raw/`")
    report_lines.append("- No NLP preprocessing was applied in this phase")
    report_lines.append("- Text content (hashtags, emojis, URLs, punctuation) is preserved")
    report_lines.append("- Label mapping: Regular=0, Sarcasm=1, Irony=2, Figurative=3")

    report_text = "\n".join(report_lines)
    report_path = REPORTS_DIR / "validation_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info(f"  Saved validation report: {report_path}")
    return report_path


def update_summary_json(cleaning_log, train_clean, test_clean):
    """Update the dataset_summary.json with clean dataset info."""
    summary_path = REPORTS_DIR / "dataset_summary.json"

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {}

    summary["train_records_clean"] = len(train_clean)
    summary["test_records_clean"] = len(test_clean)
    summary["cleaning_decisions"] = cleaning_log.get("decisions", [])
    summary["clean_train_class_distribution"] = train_clean["label"].value_counts().to_dict()
    summary["clean_test_class_distribution"] = test_clean["label"].value_counts().to_dict()
    summary["label_mapping"] = STANDARD_LABELS
    summary["clean_columns"] = list(train_clean.columns)
    summary["integrity_checks_passed"] = True
    summary["phase1_completed"] = datetime.now().isoformat()

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Updated dataset summary: {summary_path}")


def main():
    """Execute the complete dataset cleaning pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 1 - DATASET CLEANING")
    logger.info("Sarcasm and Irony Detection System using NLP")
    logger.info("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cleaning_log = {"decisions": []}

    # Load raw data
    train_df, test_df = load_raw_data()
    cleaning_log["original_train"] = len(train_df)
    cleaning_log["original_test"] = len(test_df)

    # --- CLEANING STEP 1: Standardize labels ---
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING STEP 1: STANDARDIZE LABELS")
    logger.info("=" * 60)
    train_df, train_label_map = standardize_labels(train_df, EXPECTED_LABEL_COL, "train")
    test_df, test_label_map = standardize_labels(test_df, EXPECTED_LABEL_COL, "test")
    cleaning_log["decisions"].append({
        "step": "Label Standardization",
        "action": "Standardized label capitalization/whitespace to: Regular, Sarcasm, Irony, Figurative",
        "justification": "Ensures consistent label format across pipeline",
        "impact": "No records removed, only label formatting changed",
    })

    # --- CLEANING STEP 2: Remove missing text ---
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING STEP 2: REMOVE MISSING TEXT")
    logger.info("=" * 60)
    train_df, train_text_stats = remove_missing_text(train_df, EXPECTED_TEXT_COL, "train")
    test_df, test_text_stats = remove_missing_text(test_df, EXPECTED_TEXT_COL, "test")
    cleaning_log["decisions"].append({
        "step": "Remove Missing Text",
        "action": f"Removed records with null/empty/whitespace-only text",
        "justification": "Text is required for NLP classification; no text = unusable record",
        "impact": f"Train: -{train_text_stats['total_removed']}, Test: -{test_text_stats['total_removed']}",
    })

    # --- CLEANING STEP 3: Remove missing labels ---
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING STEP 3: REMOVE MISSING LABELS")
    logger.info("=" * 60)
    train_df, train_label_stats = remove_missing_labels(train_df, EXPECTED_LABEL_COL, "train")
    test_df, test_label_stats = remove_missing_labels(test_df, EXPECTED_LABEL_COL, "test")
    cleaning_log["decisions"].append({
        "step": "Remove Missing Labels",
        "action": f"Removed records with missing class labels",
        "justification": "Supervised classification requires labels; cannot fabricate or infer them",
        "impact": f"Train: -{train_label_stats['missing_labels_removed']}, Test: -{test_label_stats['missing_labels_removed']}",
    })

    # --- CLEANING STEP 4: Remove invalid labels ---
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING STEP 4: REMOVE INVALID LABELS")
    logger.info("=" * 60)
    train_df, train_invalid = remove_invalid_labels(train_df, EXPECTED_LABEL_COL, "train")
    test_df, test_invalid = remove_invalid_labels(test_df, EXPECTED_LABEL_COL, "test")
    cleaning_log["decisions"].append({
        "step": "Remove Invalid Labels",
        "action": "Removed records with labels not in {Regular, Sarcasm, Irony, Figurative}",
        "justification": "Only four-class labels are valid for this project",
        "impact": f"Train: -{train_invalid['invalid_labels_removed']}, Test: -{test_invalid['invalid_labels_removed']}",
    })

    # --- CLEANING STEP 5: Remove exact duplicate rows ---
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING STEP 5: REMOVE EXACT DUPLICATE ROWS")
    logger.info("=" * 60)
    train_df, train_dup_stats = remove_exact_duplicates(train_df, "train")
    test_df, test_dup_stats = remove_exact_duplicates(test_df, "test")
    cleaning_log["decisions"].append({
        "step": "Remove Exact Duplicate Rows",
        "action": "Removed rows where all columns are identical (keeping first occurrence)",
        "justification": "Exact duplicates add no information and can bias model training",
        "impact": f"Train: -{train_dup_stats['exact_duplicates_removed']}, Test: -{test_dup_stats['exact_duplicates_removed']}",
    })

    # --- CLEANING STEP 6: Handle train/test overlap (leakage control) ---
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING STEP 6: TRAIN/TEST OVERLAP (LEAKAGE CONTROL)")
    logger.info("=" * 60)
    test_df, overlap_stats = handle_train_test_overlap(train_df, test_df, EXPECTED_TEXT_COL)
    cleaning_log["decisions"].append({
        "step": "Train/Test Overlap Removal (Leakage Control)",
        "action": f"Removed {overlap_stats['overlap_test_rows_removed']} test rows whose text appears in training data",
        "justification": "Identical text in both train and test creates evaluation leakage, inflating metrics. "
                         "Removed from test (not train) to preserve maximum training data.",
        "impact": f"Test: {overlap_stats['test_before']} -> {overlap_stats['test_after']} "
                  f"({overlap_stats['overlap_unique_texts']} unique overlapping texts)",
    })

    # --- CREATE CLEAN FORMAT ---
    logger.info("\n" + "=" * 60)
    logger.info("CREATING CLEAN DATASET FORMAT")
    logger.info("=" * 60)
    train_clean = create_clean_format(train_df, EXPECTED_TEXT_COL, EXPECTED_LABEL_COL, "train_clean")
    test_clean = create_clean_format(test_df, EXPECTED_TEXT_COL, EXPECTED_LABEL_COL, "test_clean")

    # --- INTEGRITY CHECKS ---
    run_integrity_checks(train_clean, test_clean)

    # --- SAVE OUTPUTS ---
    logger.info("\n" + "=" * 60)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 60)

    train_path = PROCESSED_DIR / "train_clean.csv"
    test_path = PROCESSED_DIR / "test_clean.csv"
    train_clean.to_csv(train_path, index=False)
    test_clean.to_csv(test_path, index=False)
    logger.info(f"  Saved clean train: {train_path} ({len(train_clean)} rows)")
    logger.info(f"  Saved clean test: {test_path} ({len(test_clean)} rows)")

    # Save label mapping
    save_label_mapping()

    # Generate validation report
    generate_validation_report(cleaning_log, train_clean, test_clean)

    # Update summary JSON
    update_summary_json(cleaning_log, train_clean, test_clean)

    # --- FINAL SUMMARY ---
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 CLEANING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Original train: {cleaning_log['original_train']} -> Clean train: {len(train_clean)}")
    logger.info(f"  Original test: {cleaning_log['original_test']} -> Clean test: {len(test_clean)}")
    logger.info(f"  Clean train class distribution:")
    for label, count in train_clean["label"].value_counts().items():
        logger.info(f"    {label}: {count} ({round(count/len(train_clean)*100, 1)}%)")
    logger.info(f"  Clean test class distribution:")
    for label, count in test_clean["label"].value_counts().items():
        logger.info(f"    {label}: {count} ({round(count/len(test_clean)*100, 1)}%)")


if __name__ == "__main__":
    main()
