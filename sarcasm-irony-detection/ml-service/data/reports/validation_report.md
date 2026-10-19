# Phase 1 - Dataset Validation Report

Generated: 2026-08-14 14:05:20

## Dataset Source
Kaggle 4-class Sarcasm and Irony Dataset

## Raw Dataset
- Train file: `data/raw/train.csv`
- Test file: `data/raw/test.csv`
- Original train rows: 81408
- Original test rows: 8128

## Cleaning Decisions

### Label Standardization
- **Action**: Standardized label capitalization/whitespace to: Regular, Sarcasm, Irony, Figurative
- **Justification**: Ensures consistent label format across pipeline
- **Impact**: No records removed, only label formatting changed

### Remove Missing Text
- **Action**: Removed records with null/empty/whitespace-only text
- **Justification**: Text is required for NLP classification; no text = unusable record
- **Impact**: Train: -0, Test: -2

### Remove Missing Labels
- **Action**: Removed records with missing class labels
- **Justification**: Supervised classification requires labels; cannot fabricate or infer them
- **Impact**: Train: -0, Test: -7

### Remove Invalid Labels
- **Action**: Removed records with labels not in {Regular, Sarcasm, Irony, Figurative}
- **Justification**: Only four-class labels are valid for this project
- **Impact**: Train: -0, Test: -0

### Remove Exact Duplicate Rows
- **Action**: Removed rows where all columns are identical (keeping first occurrence)
- **Justification**: Exact duplicates add no information and can bias model training
- **Impact**: Train: -49, Test: -0

### Train/Test Overlap Removal (Leakage Control)
- **Action**: Removed 2592 test rows whose text appears in training data
- **Justification**: Identical text in both train and test creates evaluation leakage, inflating metrics. Removed from test (not train) to preserve maximum training data.
- **Impact**: Test: 8119 -> 5527 (2592 unique overlapping texts)


## Clean Dataset Summary
- Clean train rows: 81359
- Clean test rows: 5527

### Clean Train Class Distribution
- Figurative: 21235 (26.1%)
- Irony: 20877 (25.66%)
- Sarcasm: 20678 (25.42%)
- Regular: 18569 (22.82%)

### Clean Test Class Distribution
- Regular: 1858 (33.62%)
- Sarcasm: 1602 (28.98%)
- Irony: 1301 (23.54%)
- Figurative: 766 (13.86%)

## Data Quality Assessment
All integrity checks passed.
No train/test text overlap in final clean data.

## Important Notes
- Raw files are preserved untouched in `data/raw/`
- No NLP preprocessing was applied in this phase
- Text content (hashtags, emojis, URLs, punctuation) is preserved
- Label mapping: Regular=0, Sarcasm=1, Irony=2, Figurative=3