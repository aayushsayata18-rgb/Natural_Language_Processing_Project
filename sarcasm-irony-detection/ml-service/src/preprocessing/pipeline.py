"""
Phase 3 - Preprocessing Pipeline
============================================
Sarcasm & Irony Detection System using NLP

Central pipeline orchestrator that combines all configurable
preprocessing stages into a unified processing flow.

The pipeline applies stages in the correct order:
    1. Text validation
    2. Feature extraction (before normalization)
    3. Social media normalization (URLs, mentions, hashtags, emojis)
    4. Whitespace normalization
    5. Case normalization
    6. Punctuation handling
    7. Repeated character normalization
    8. Tokenization
    9. Stopword removal (optional)
    10. Lemmatization (optional)
    11. Stemming (optional)
    12. Token reassembly

Important:
    - No model is trained here.
    - No TF-IDF or BoW is computed here.
    - This phase prepares text and preprocessing infrastructure.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.preprocessing.config import (
    PreprocessingConfig,
    get_all_variants,
    get_variant,
)
from src.preprocessing.text_cleaner import (
    validate_text,
    get_invalid_input_count,
    reset_invalid_input_count,
)
from src.preprocessing.social_media import (
    normalize_urls,
    normalize_mentions,
    handle_hashtags,
    handle_emojis,
    extract_social_media_features,
)
from src.preprocessing.normalization import (
    normalize_whitespace,
    normalize_case,
    normalize_punctuation,
    normalize_repeated_chars,
    remove_stopwords_from_tokens,
    extract_text_features,
)
from src.preprocessing.tokenizer import tokenize_text, tokens_to_string
from src.preprocessing.lemmatization import lemmatize_tokens
from src.preprocessing.stemming import stem_tokens

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ml-service/
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PREPROCESSED_DIR = PROCESSED_DIR / "preprocessed"
REPORTS_DIR = DATA_DIR / "reports" / "preprocessing"

# --- Logging ---
# Use UTF-8 stream to handle emoji characters on Windows (cp1252)
import io
_utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(_utf8_stdout)]
)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Configurable text preprocessing pipeline.
    
    Applies preprocessing stages in the correct order based on
    a PreprocessingConfig instance.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize pipeline with a configuration.
        
        Args:
            config: PreprocessingConfig controlling which steps are applied.
        """
        config.validate()
        self.config = config
        logger.info(f"Pipeline initialized: {config.name} — {config.description}")
    
    def process_text(self, text: str) -> str:
        """
        Apply the full preprocessing pipeline to a single text.
        
        Pipeline order:
            1. Validate text
            2. Social media normalization
            3. Whitespace normalization
            4. Case normalization
            5. Punctuation handling
            6. Repeated character normalization
            7. Tokenization
            8. Stopword removal (if enabled)
            9. Lemmatization (if enabled)
            10. Stemming (if enabled)
            11. Token reassembly
        
        Args:
            text: Raw text string.
        
        Returns:
            str: Preprocessed text string.
        """
        # Step 1: Validate
        text = validate_text(text)
        if not text:
            return ""
        
        # Step 2: Social media normalization
        if self.config.normalize_urls:
            text = normalize_urls(text, self.config.url_placeholder)
        
        if self.config.normalize_mentions:
            text = normalize_mentions(text, self.config.mention_placeholder)
        
        text = handle_hashtags(text, self.config.hashtag_mode)
        text = handle_emojis(text, self.config.emoji_mode)
        
        # Step 3: Whitespace normalization
        if self.config.normalize_whitespace:
            text = normalize_whitespace(text)
        
        # Step 4: Case normalization
        text = normalize_case(text, self.config.lowercase)
        
        # Step 5: Punctuation handling
        text = normalize_punctuation(text, self.config.punctuation_mode)
        
        # Step 6: Repeated character normalization
        if self.config.normalize_repeated_chars:
            text = normalize_repeated_chars(text, self.config.max_char_repeat)
        
        # Step 7: Tokenization
        tokens = tokenize_text(text)
        
        # Step 8: Stopword removal
        if self.config.remove_stopwords:
            tokens = remove_stopwords_from_tokens(
                tokens,
                protect_negations=self.config.protect_negations
            )
        
        # Step 9: Lemmatization
        if self.config.lemmatize:
            tokens = lemmatize_tokens(tokens)
        
        # Step 10: Stemming
        if self.config.stem:
            tokens = stem_tokens(tokens)
        
        # Step 11: Reassemble
        return tokens_to_string(tokens)
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
    ) -> pd.DataFrame:
        """
        Process an entire DataFrame, adding a 'processed_text' column.
        
        The original text column is preserved.
        Labels are NOT modified.
        
        Args:
            df: DataFrame with text column.
            text_col: Name of the text column.
        
        Returns:
            DataFrame with added 'processed_text' column.
        """
        logger.info(f"Processing {len(df)} records with '{self.config.name}' config...")
        
        reset_invalid_input_count()
        
        result_df = df.copy()
        result_df["processed_text"] = result_df[text_col].apply(self.process_text)
        
        invalid_count = get_invalid_input_count()
        if invalid_count > 0:
            logger.warning(f"Encountered {invalid_count} unexpected invalid inputs")
        
        logger.info(f"Processing complete. {len(result_df)} records processed.")
        return result_df


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_all_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Extract all social-media and text features from a DataFrame.
    
    Must be called on the ORIGINAL text (before normalization)
    to capture true feature values.
    
    Features extracted:
        url_count, has_url, mention_count, has_mention,
        hashtag_count, has_hashtag, emoji_count, has_emoji,
        exclamation_count, question_count, punctuation_count,
        uppercase_count, uppercase_ratio, repeated_character_count
    
    Returns:
        DataFrame with original columns plus all feature columns.
    """
    logger.info(f"Extracting features from {len(df)} records...")
    
    result_df = df.copy()
    
    # Social media features
    sm_features = result_df[text_col].apply(
        lambda t: pd.Series(extract_social_media_features(str(t)))
    )
    
    # Text features
    txt_features = result_df[text_col].apply(
        lambda t: pd.Series(extract_text_features(str(t)))
    )
    
    # Combine
    for col in sm_features.columns:
        result_df[col] = sm_features[col]
    for col in txt_features.columns:
        result_df[col] = txt_features[col]
    
    logger.info(f"Feature extraction complete. {len(result_df.columns)} total columns.")
    return result_df


# ============================================================
# QUALITY CHECKS
# ============================================================

def quality_check(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    variant_name: str,
) -> Dict:
    """
    Verify preprocessing quality.
    
    Checks:
        - No unexpected nulls
        - No accidental empty text
        - Label unchanged
        - Number of records unchanged
        - Class distribution unchanged
    
    Returns:
        dict with quality check results.
    """
    results = {"variant": variant_name, "passed": True, "issues": []}
    
    # Row count check
    if len(original_df) != len(processed_df):
        results["passed"] = False
        results["issues"].append(
            f"Row count mismatch: {len(original_df)} vs {len(processed_df)}"
        )
    
    # Null check in processed_text
    null_count = processed_df["processed_text"].isna().sum()
    if null_count > 0:
        results["passed"] = False
        results["issues"].append(f"Found {null_count} null processed texts")
    
    # Empty text check
    empty_count = (processed_df["processed_text"] == "").sum()
    results["empty_count"] = int(empty_count)
    if empty_count > 0:
        results["issues"].append(
            f"Found {empty_count} empty processed texts ({empty_count/len(processed_df)*100:.2f}%)"
        )
    
    # Label integrity check
    if "label" in original_df.columns and "label" in processed_df.columns:
        if not original_df["label"].equals(processed_df["label"]):
            results["passed"] = False
            results["issues"].append("Labels were modified during preprocessing!")
    
    # Class distribution check
    if "label" in original_df.columns and "label" in processed_df.columns:
        orig_dist = original_df["label"].value_counts().sort_index()
        proc_dist = processed_df["label"].value_counts().sort_index()
        if not orig_dist.equals(proc_dist):
            results["passed"] = False
            results["issues"].append("Class distribution changed during preprocessing!")
    
    # Label ID integrity check
    if "label_id" in original_df.columns and "label_id" in processed_df.columns:
        if not original_df["label_id"].equals(processed_df["label_id"]):
            results["passed"] = False
            results["issues"].append("Label IDs were modified during preprocessing!")
    
    status = "PASSED" if results["passed"] else "FAILED"
    logger.info(f"Quality check [{variant_name}]: {status}")
    for issue in results["issues"]:
        level = logger.warning if "empty" in issue.lower() else logger.error
        level(f"  - {issue}")
    
    return results


# ============================================================
# INFORMATION LOSS ANALYSIS
# ============================================================

def information_loss_metrics(
    df: pd.DataFrame,
    text_col: str = "processed_text",
) -> Dict:
    """
    Calculate information-loss metrics for a preprocessed variant.
    
    Metrics:
        - percentage of empty texts
        - average character count
        - average word count
        - vocabulary size
        - percentage containing emojis
        - percentage containing hashtags
        - percentage containing punctuation
    """
    import string as string_module
    from src.preprocessing.social_media import has_emoji, has_hashtag
    
    texts = df[text_col].fillna("")
    total = len(texts)
    
    empty_pct = (texts == "").sum() / total * 100
    avg_chars = texts.str.len().mean()
    avg_words = texts.str.split().str.len().mean()
    
    # Vocabulary size: unique tokens
    all_tokens = set()
    for text in texts:
        if text:
            all_tokens.update(text.split())
    vocab_size = len(all_tokens)
    
    # Content preservation percentages
    emoji_pct = texts.apply(lambda t: has_emoji(t) if t else False).sum() / total * 100
    hashtag_pct = texts.apply(lambda t: has_hashtag(t) if t else False).sum() / total * 100
    punct_pct = texts.apply(
        lambda t: any(c in string_module.punctuation for c in t) if t else False
    ).sum() / total * 100
    
    return {
        "empty_pct": round(empty_pct, 2),
        "avg_chars": round(avg_chars, 2),
        "avg_words": round(avg_words, 2),
        "vocab_size": vocab_size,
        "emoji_pct": round(emoji_pct, 2),
        "hashtag_pct": round(hashtag_pct, 2),
        "punctuation_pct": round(punct_pct, 2),
    }


# ============================================================
# SAMPLE COMPARISON
# ============================================================

def sample_comparison(
    df: pd.DataFrame,
    text_col: str = "text",
    n_samples: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run all preprocessing variants on a sample and compare outputs.
    
    This is a mandatory sanity check before processing the full dataset.
    Selects samples that include diverse social-media elements.
    """
    logger.info(f"\n{'='*70}")
    logger.info("SAMPLE PREPROCESSING COMPARISON")
    logger.info(f"{'='*70}")
    
    # Try to select diverse samples
    sample_df = _select_diverse_sample(df, text_col, n_samples, seed)
    
    variants = get_all_variants()
    results = []
    
    for idx, row in sample_df.iterrows():
        original = row[text_col]
        result_row = {"original": original}
        
        for var_name, config in variants.items():
            pipeline = PreprocessingPipeline(config)
            result_row[var_name] = pipeline.process_text(original)
        
        results.append(result_row)
    
    comparison_df = pd.DataFrame(results)
    
    # Print comparison
    for i, row in comparison_df.iterrows():
        logger.info(f"\n--- Sample {i+1} ---")
        logger.info(f"  Original:   {row['original'][:120]}")
        for var_name in variants:
            logger.info(f"  {var_name:12s}: {row[var_name][:120]}")
    
    return comparison_df


def _select_diverse_sample(
    df: pd.DataFrame,
    text_col: str,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select a diverse sample that includes various social-media elements.
    
    Tries to include texts with: hashtags, mentions, URLs, emojis,
    punctuation, capitalization, repeated characters, negations.
    """
    from src.preprocessing.social_media import (
        has_url, has_mention, has_emoji,
    )
    from src.preprocessing.normalization import has_repeated_chars
    
    rng = np.random.RandomState(seed)
    
    # Create boolean masks for different features
    texts = df[text_col].fillna("")
    
    masks = {
        "url": texts.apply(has_url),
        "mention": texts.apply(has_mention),
        "emoji": texts.apply(has_emoji),
        "repeated": texts.apply(has_repeated_chars),
        "caps": texts.apply(lambda t: any(w.isupper() and len(w) > 1 for w in t.split())),
        "excl": texts.str.contains("!!", na=False),
        "quest": texts.str.contains(r"\?\?", na=False),
        "negation": texts.str.contains(
            r"\b(?:not|never|no|n\'t|don\'t|can\'t|won\'t)\b",
            case=False, na=False
        ),
    }
    
    # Try to pick one sample per feature, fill rest randomly
    selected_indices = set()
    for feature, mask in masks.items():
        candidates = df[mask].index.tolist()
        if candidates:
            choice = rng.choice(candidates)
            selected_indices.add(choice)
    
    # Fill remaining slots with random samples
    remaining = n_samples - len(selected_indices)
    if remaining > 0:
        available = [i for i in df.index if i not in selected_indices]
        if available:
            extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
            selected_indices.update(extra)
    
    return df.loc[list(selected_indices)[:n_samples]]


# ============================================================
# FULL DATASET PROCESSING
# ============================================================

def process_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: PreprocessingConfig,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Process both train and test datasets with a specific variant.
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        config: Preprocessing configuration.
        output_dir: Directory to save outputs. If None, uses default.
    
    Returns:
        Tuple of (processed_train, processed_test, quality_results).
    """
    pipeline = PreprocessingPipeline(config)
    
    # Process
    train_processed = pipeline.process_dataframe(train_df)
    test_processed = pipeline.process_dataframe(test_df)
    
    # Quality checks
    train_quality = quality_check(train_df, train_processed, f"{config.name}_train")
    test_quality = quality_check(test_df, test_processed, f"{config.name}_test")
    
    # Save if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_processed.to_csv(output_dir / "train.csv", index=False)
        test_processed.to_csv(output_dir / "test.csv", index=False)
        logger.info(f"Saved {config.name} variant to {output_dir}")
    
    return train_processed, test_processed, {
        "train": train_quality,
        "test": test_quality,
    }


def run_all_variants(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict:
    """
    Run all preprocessing variants, save outputs, and generate comparison.
    
    This is the main entry point for full-dataset processing.
    
    Returns:
        dict with comparison metrics and quality results for all variants.
    """
    logger.info(f"\n{'='*70}")
    logger.info("RUNNING ALL PREPROCESSING VARIANTS")
    logger.info(f"{'='*70}")
    
    variants = get_all_variants()
    all_results = {}
    comparison_data = []
    
    for var_name, config in variants.items():
        logger.info(f"\n--- Processing variant: {var_name} ---")
        
        output_dir = PREPROCESSED_DIR / var_name
        
        train_proc, test_proc, quality = process_variant(
            train_df, test_df, config, output_dir
        )
        
        # Information loss metrics (on training set)
        metrics = information_loss_metrics(train_proc)
        metrics["variant"] = var_name
        metrics["records"] = len(train_proc)
        comparison_data.append(metrics)
        
        all_results[var_name] = {
            "quality": quality,
            "metrics": metrics,
            "config": config.to_dict(),
        }
    
    # Save comparison CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df[
        ["variant", "records", "empty_pct", "avg_chars", "avg_words",
         "vocab_size", "emoji_pct", "hashtag_pct", "punctuation_pct"]
    ]
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(REPORTS_DIR / "preprocessing_comparison.csv", index=False)
    logger.info(f"\nPreprocessing comparison saved to {REPORTS_DIR / 'preprocessing_comparison.csv'}")
    
    # Print comparison table
    logger.info(f"\n{'='*70}")
    logger.info("PREPROCESSING COMPARISON")
    logger.info(f"{'='*70}")
    logger.info(comparison_df.to_string(index=False))
    
    return all_results


# ============================================================
# SOCIAL MEDIA FEATURE FILE
# ============================================================

def generate_feature_file(
    train_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate the social-media feature file from training data.
    
    Extracts all social-media and text features and saves to CSV.
    Features are extracted from ORIGINAL text before any normalization.
    """
    logger.info("Generating social-media feature file...")
    
    features_df = extract_all_features(train_df)
    
    if output_path is None:
        output_path = REPORTS_DIR / "social_media_features.csv"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    
    logger.info(f"Feature file saved to {output_path}")
    logger.info(f"  Records: {len(features_df)}")
    logger.info(f"  Columns: {list(features_df.columns)}")
    
    return features_df


# ============================================================
# PREPROCESSING SUMMARY
# ============================================================

def generate_summary(all_results: Dict) -> Dict:
    """
    Generate the preprocessing summary JSON.
    """
    summary = {
        "phase": "Phase 3 - NLP Text Preprocessing",
        "project": "Sarcasm & Irony Detection System",
        "generated": datetime.now().isoformat(),
        "variants": {},
        "quality_summary": {
            "all_passed": True,
            "issues": [],
        },
    }
    
    for var_name, result in all_results.items():
        summary["variants"][var_name] = {
            "config": result["config"],
            "metrics": result["metrics"],
            "quality": {
                "train_passed": result["quality"]["train"]["passed"],
                "test_passed": result["quality"]["test"]["passed"],
            },
        }
        
        if not result["quality"]["train"]["passed"]:
            summary["quality_summary"]["all_passed"] = False
            summary["quality_summary"]["issues"].extend(
                result["quality"]["train"]["issues"]
            )
        if not result["quality"]["test"]["passed"]:
            summary["quality_summary"]["all_passed"] = False
            summary["quality_summary"]["issues"].extend(
                result["quality"]["test"]["issues"]
            )
    
    # Save
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Preprocessing summary saved to {summary_path}")
    return summary


# ============================================================
# NEGATION PRESERVATION TEST
# ============================================================

def test_negation_preservation():
    """
    Explicitly test that negation words are preserved during
    stopword removal.
    """
    logger.info(f"\n{'='*70}")
    logger.info("NEGATION PRESERVATION TEST")
    logger.info(f"{'='*70}")
    
    test_sentences = [
        "I do not like this",
        "She never said that",
        "No one cares about this",
        "I can't believe it",
        "This isn't funny at all",
        "I won't do that",
        "He didn't say anything",
        "We don't need this",
        "Nothing works here",
        "I have not seen it",
    ]
    
    # Test with stopword removal + negation protection
    config = get_variant("stopword")
    pipeline = PreprocessingPipeline(config)
    
    all_passed = True
    negation_keywords = {"not", "never", "no", "n't", "nothing",
                         "don't", "can't", "won't", "didn't", "isn't"}
    
    for sentence in test_sentences:
        processed = pipeline.process_text(sentence)
        tokens = processed.lower().split()
        
        # Check if at least one negation word is present
        has_neg = any(neg in tokens for neg in negation_keywords)
        
        status = "PASS" if has_neg else "FAIL"
        if not has_neg:
            all_passed = False
        
        logger.info(f"  [{status}] \"{sentence}\" -> \"{processed}\"")
    
    logger.info(f"\nNegation preservation: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


# ============================================================
# SOCIAL MEDIA PRESERVATION TEST
# ============================================================

def test_social_media_preservation():
    """
    Test social-media element handling across variants.
    """
    logger.info(f"\n{'='*70}")
    logger.info("SOCIAL MEDIA PRESERVATION TEST")
    logger.info(f"{'='*70}")
    
    test_texts = [
        "@user great job today",
        "#sarcasm is my favorite thing",
        "check this 😂😒🙃",
        "WOW!!! That's AMAZING!!!",
        "Really??? You think so???",
        "https://example.com is great",
        "SO GREAT I could cry",
        "soooo good right now",
        "@john WOW!!! #sarcasm 😂 https://example.com",
    ]
    
    variants = get_all_variants()
    
    for text in test_texts:
        logger.info(f"\n  Original: \"{text}\"")
        for var_name, config in variants.items():
            pipeline = PreprocessingPipeline(config)
            processed = pipeline.process_text(text)
            logger.info(f"    {var_name:12s}: \"{processed}\"")
    
    return True


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Main entry point for Phase 3 preprocessing pipeline.
    
    1. Load Phase 1 clean data
    2. Run sample comparison (sanity check)
    3. Test negation preservation
    4. Test social-media preservation
    5. Extract social-media features
    6. Process all variants on full dataset
    7. Generate comparison and summary reports
    """
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 3 — NLP TEXT PREPROCESSING PIPELINE")
    logger.info(f"{'='*70}")
    
    # Ensure NLTK resources
    from src.preprocessing.setup_nltk import setup_nltk
    setup_nltk()
    
    # Load data
    logger.info("\nLoading Phase 1 clean datasets...")
    train_df = pd.read_csv(PROCESSED_DIR / "train_clean.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_clean.csv")
    logger.info(f"  Train: {train_df.shape}")
    logger.info(f"  Test:  {test_df.shape}")
    
    # Step 1: Sample comparison (mandatory sanity check)
    sample_comp = sample_comparison(train_df)
    
    # Step 2: Negation preservation test
    test_negation_preservation()
    
    # Step 3: Social media preservation test
    test_social_media_preservation()
    
    # Step 4: Extract features from original text
    generate_feature_file(train_df)
    
    # Step 5: Run all variants on full dataset
    all_results = run_all_variants(train_df, test_df)
    
    # Step 6: Generate summary
    summary = generate_summary(all_results)
    
    # Final status
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 3 COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Variants processed: {len(all_results)}")
    logger.info(f"Quality checks: {'ALL PASSED' if summary['quality_summary']['all_passed'] else 'ISSUES FOUND'}")
    logger.info(f"Output directory: {PREPROCESSED_DIR}")
    logger.info(f"Reports directory: {REPORTS_DIR}")
    
    return all_results


if __name__ == "__main__":
    main()
