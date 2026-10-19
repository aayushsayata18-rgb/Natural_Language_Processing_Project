"""
run_eda.py — Master EDA runner script.

Orchestrates all Phase 2 analysis modules, generates all outputs,
reports, and visualizations.

Phase 2: Exploratory Data Analysis
Project: Sarcasm & Irony Detection System

Usage:
    python src/eda/run_eda.py
"""

import sys
import os
import json
import time
import warnings
from datetime import datetime

# Fix Windows console encoding for Unicode output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

from src.eda.analyze_dataset import (
    load_datasets,
    dataset_overview,
    dataset_overview_table,
    quality_check,
    class_distribution,
    class_balance_analysis,
)
from src.eda.text_statistics import (
    compute_text_features,
    overall_and_classwise_stats,
    text_length_statistics_table,
    outlier_summary,
)
from src.eda.social_media_analysis import (
    extract_social_features,
    url_analysis,
    mention_analysis,
    hashtag_analysis,
    emoji_analysis,
    punctuation_analysis,
    capitalization_analysis,
    repeated_char_analysis,
    classwise_comparison_table,
)
from src.eda.frequency_analysis import (
    eda_normalize,
    word_frequency,
    classwise_word_frequency,
    extract_ngrams,
    classwise_ngrams,
    stopword_comparison,
    vocabulary_analysis,
    tfidf_class_distinctive_terms,
    ngrams_to_dataframe,
    classwise_ngrams_to_dataframe,
)
from src.eda import visualization as viz

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train_clean.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test_clean.csv")
REPORT_DIR = os.path.join(PROJECT_ROOT, "data", "reports", "eda")
VIZ_DIR = os.path.join(PROJECT_ROOT, "results", "eda")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ===================================================================
# MAIN PIPELINE
# ===================================================================

def main():
    start_time = time.time()
    ensure_dirs()

    eda_summary = {}  # Accumulates all results for eda_summary.json

    # ==============================================================
    # STEP 1: DATASET OVERVIEW
    # ==============================================================
    section("STEP 1 — Dataset Overview")

    train_df, test_df = load_datasets(TRAIN_PATH, TEST_PATH)
    print(f"  Loaded train: {train_df.shape}")
    print(f"  Loaded test:  {test_df.shape}")

    train_overview = dataset_overview(train_df, "Train")
    test_overview = dataset_overview(test_df, "Test")
    overview_table = dataset_overview_table(train_df, test_df)

    eda_summary["dataset_overview"] = {
        "train": train_overview,
        "test": test_overview,
    }

    print("\n  Dataset Overview Table:")
    print(overview_table.to_string(index=False))
    print(f"\n  Train sample:")
    print(train_df.head(3).to_string())

    # ==============================================================
    # STEP 2: DATA QUALITY RECHECK
    # ==============================================================
    section("STEP 2 — Data Quality Recheck")

    train_quality = quality_check(train_df, "Train")
    test_quality = quality_check(test_df, "Test")

    eda_summary["quality_checks"] = {
        "train": train_quality,
        "test": test_quality,
    }

    for qc in [train_quality, test_quality]:
        status = qc["quality_status"]
        print(f"  {qc['dataset']}: {status}")
        if qc["issues"]:
            for issue in qc["issues"]:
                print(f"    ⚠ {issue}")
        else:
            print("    ✓ All quality checks passed")

    # ==============================================================
    # STEP 3-4: CLASS DISTRIBUTION & BALANCE
    # ==============================================================
    section("STEP 3–4 — Class Distribution & Balance")

    train_dist = class_distribution(train_df, "Train")
    test_dist = class_distribution(test_df, "Test")

    combined_dist = pd.concat([train_dist, test_dist], ignore_index=True)
    combined_dist.to_csv(
        os.path.join(REPORT_DIR, "class_distribution.csv"), index=False
    )
    print("  Saved: class_distribution.csv")

    balance = class_balance_analysis(train_dist, test_dist)
    eda_summary["class_distribution"] = {
        "train": train_dist.to_dict(orient="records"),
        "test": test_dist.to_dict(orient="records"),
    }
    eda_summary["class_balance"] = balance

    print(f"\n  Train balance: {balance['train']['balance_category']}")
    print(f"    Majority: {balance['train']['majority_class']} ({balance['train']['majority_count']:,})")
    print(f"    Minority: {balance['train']['minority_class']} ({balance['train']['minority_count']:,})")
    print(f"    Ratio: {balance['train']['majority_minority_ratio']}")
    print(f"    → {balance['train']['implication']}")

    print(f"\n  Test balance: {balance['test']['balance_category']}")
    print(f"    Majority: {balance['test']['majority_class']} ({balance['test']['majority_count']:,})")
    print(f"    Minority: {balance['test']['minority_class']} ({balance['test']['minority_count']:,})")
    print(f"    Ratio: {balance['test']['majority_minority_ratio']}")

    # Visualize
    viz.plot_class_distribution(
        train_dist,
        "Class Distribution — Training Set",
        os.path.join(VIZ_DIR, "class_distribution.png"),
    )
    viz.plot_class_distribution(
        test_dist,
        "Class Distribution — Test Set",
        os.path.join(VIZ_DIR, "class_distribution_test.png"),
    )

    # ==============================================================
    # STEP 5-7: TEXT LENGTH FEATURES & STATISTICS
    # ==============================================================
    section("STEP 5–7 — Text Length Features & Statistics")

    text_features = compute_text_features(train_df)

    # Save text_length_features.csv
    text_features.to_csv(
        os.path.join(PROCESSED_DIR, "text_length_features.csv"), index=False
    )
    print("  Saved: text_length_features.csv")

    # Statistics
    length_stats = overall_and_classwise_stats(text_features)
    eda_summary["text_length_statistics"] = length_stats

    stats_table = text_length_statistics_table(text_features)
    stats_table.to_csv(
        os.path.join(REPORT_DIR, "text_length_statistics.csv"), index=False
    )
    print("  Saved: text_length_statistics.csv")

    print("\n  Text Length Statistics:")
    print(stats_table.to_string(index=False))

    # Visualizations
    viz.plot_histogram(
        text_features, "character_count",
        "Tweet Character Count Distribution",
        "Character Count",
        os.path.join(VIZ_DIR, "character_count_distribution.png"),
    )
    viz.plot_histogram(
        text_features, "word_count",
        "Tweet Word Count Distribution",
        "Word Count",
        os.path.join(VIZ_DIR, "word_count_distribution.png"),
    )
    viz.plot_boxplot_by_class(
        text_features, "character_count",
        "Character Count by Class",
        "Character Count",
        os.path.join(VIZ_DIR, "character_count_by_class.png"),
    )
    viz.plot_boxplot_by_class(
        text_features, "word_count",
        "Word Count by Class",
        "Word Count",
        os.path.join(VIZ_DIR, "word_count_by_class.png"),
    )

    # ==============================================================
    # STEP 8: OUTLIER ANALYSIS
    # ==============================================================
    section("STEP 8 — Outlier Analysis")

    outliers = outlier_summary(text_features)
    eda_summary["outlier_analysis"] = outliers

    for col, info in outliers.items():
        print(f"\n  {col}:")
        print(f"    IQR bounds: [{info['lower_bound']:.0f}, {info['upper_bound']:.0f}]")
        print(f"    Low outliers: {info['low_outlier_count']}")
        print(f"    High outliers: {info['high_outlier_count']}")
        print(f"    Outlier %: {info['outlier_percentage']}%")
        if info.get("outlier_class_distribution"):
            print(f"    Class distribution: {info['outlier_class_distribution']}")

    # ==============================================================
    # STEP 9-16: SOCIAL MEDIA PATTERN ANALYSIS
    # ==============================================================
    section("STEP 9–16 — Social Media Pattern Analysis")

    social_features = extract_social_features(train_df)

    # URL Analysis
    url_info = url_analysis(social_features)
    print(f"\n  URLs: {url_info['tweets_with_urls']:,} tweets ({url_info['percentage_with_urls']}%)")

    # Mention Analysis
    mention_info = mention_analysis(social_features)
    print(f"  Mentions: {mention_info['tweets_with_mentions']:,} tweets ({mention_info['percentage_with_mentions']}%)")
    print(f"    Avg mentions/tweet: {mention_info['avg_mentions_per_tweet']}")

    # Hashtag Analysis
    hashtag_info = hashtag_analysis(social_features)
    print(f"  Hashtags: {hashtag_info['tweets_with_hashtags']:,} tweets ({hashtag_info['percentage_with_hashtags']}%)")
    print(f"    Avg hashtags/tweet: {hashtag_info['avg_hashtags_per_tweet']}")
    print(f"    Top 5 hashtags: {[h['hashtag'] for h in hashtag_info['top_hashtags'][:5]]}")

    # Emoji Analysis
    emoji_info = emoji_analysis(social_features)
    print(f"  Emojis: {emoji_info['tweets_with_emojis']:,} tweets ({emoji_info['percentage_with_emojis']}%)")

    # Punctuation Analysis
    punct_info = punctuation_analysis(social_features)
    for feat, info in punct_info.items():
        print(f"  {feat}: {info['tweets_with_feature']:,} tweets ({info['percentage']}%)")

    # Capitalization Analysis
    caps_info = capitalization_analysis(social_features)
    print(f"  Uppercase ratio (avg): {caps_info['avg_uppercase_ratio']}")
    print(f"  Tweets w/ all-caps words: {caps_info['tweets_with_allcaps_words']:,} ({caps_info['pct_with_allcaps_words']}%)")

    # Repeated Characters
    repeat_info = repeated_char_analysis(social_features)
    print(f"  Repeated chars: {repeat_info['tweets_with_repeated_chars']:,} ({repeat_info['percentage_with_repeated_chars']}%)")

    # Save social media features
    eda_summary["social_media_analysis"] = {
        "urls": url_info,
        "mentions": mention_info,
        "hashtags": hashtag_info,
        "emojis": emoji_info,
        "punctuation": punct_info,
        "capitalization": caps_info,
        "repeated_characters": repeat_info,
    }

    # Social media features CSV
    social_summary = classwise_comparison_table(text_features, social_features)
    social_summary.to_csv(
        os.path.join(REPORT_DIR, "social_media_features.csv"), index=False
    )
    print("\n  Saved: social_media_features.csv")

    # Visualizations
    viz.plot_feature_by_class(
        url_info["by_class"],
        "URL Presence by Class (%)",
        "Tweets with URLs (%)",
        os.path.join(VIZ_DIR, "urls_by_class.png"),
    )
    viz.plot_feature_by_class(
        mention_info["by_class"],
        "Mention (@) Presence by Class (%)",
        "Tweets with Mentions (%)",
        os.path.join(VIZ_DIR, "mentions_by_class.png"),
    )
    viz.plot_feature_by_class(
        hashtag_info["by_class"],
        "Hashtag (#) Presence by Class (%)",
        "Tweets with Hashtags (%)",
        os.path.join(VIZ_DIR, "hashtags_by_class.png"),
    )
    viz.plot_feature_by_class(
        emoji_info["by_class"],
        "Emoji Presence by Class (%)",
        "Tweets with Emojis (%)",
        os.path.join(VIZ_DIR, "emojis_by_class.png"),
    )
    viz.plot_punctuation_by_class(
        social_features,
        os.path.join(VIZ_DIR, "punctuation_by_class.png"),
    )
    viz.plot_feature_by_class(
        caps_info["allcaps_by_class"],
        "All-Caps Words Presence by Class (%)",
        "Tweets with All-Caps Words (%)",
        os.path.join(VIZ_DIR, "uppercase_by_class.png"),
    )

    # ==============================================================
    # STEP 17-24: WORD FREQUENCY & N-GRAM ANALYSIS
    # ==============================================================
    section("STEP 17–24 — Word Frequency & N-gram Analysis")

    # EDA-only normalization
    analysis_df = train_df.copy()
    analysis_df["text_normalized"] = eda_normalize(analysis_df["text"])

    # Stopword comparison
    print("\n  Stopword Analysis:")
    sw_analysis = stopword_comparison(analysis_df["text_normalized"])
    eda_summary["stopword_analysis"] = sw_analysis
    print(f"    Stopwords available: {sw_analysis['stopwords_available']} ({sw_analysis['stopword_count']})")
    print(f"    Raw top 5:    {[w['word'] for w in sw_analysis['raw_top_words'][:5]]}")
    print(f"    Clean top 5:  {[w['word'] for w in sw_analysis['no_stopwords_top_words'][:5]]}")

    # Top unigrams (overall)
    print("\n  Top Unigrams (after stopword removal):")
    top_unigrams = word_frequency(
        analysis_df["text_normalized"], remove_stopwords=True, top_n=30
    )
    uni_df = ngrams_to_dataframe(top_unigrams, "Overall")
    uni_df.columns = ["Class", "Word", "Frequency"]
    uni_df.to_csv(os.path.join(REPORT_DIR, "top_unigrams.csv"), index=False)
    print("  Saved: top_unigrams.csv")

    # Class-wise unigrams
    cw_uni = classwise_word_frequency(analysis_df, top_n=20)
    cw_uni_rows = []
    for label, freqs in cw_uni.items():
        for word, count in freqs:
            cw_uni_rows.append({"Class": label, "Word": word, "Frequency": count})
    pd.DataFrame(cw_uni_rows).to_csv(
        os.path.join(REPORT_DIR, "classwise_unigrams.csv"), index=False
    )
    print("  Saved: classwise_unigrams.csv")

    # Top bigrams
    print("\n  Top Bigrams:")
    top_bigrams = extract_ngrams(
        analysis_df["text_normalized"], n=2, top_n=30
    )
    bi_df = ngrams_to_dataframe(top_bigrams, "Overall")
    bi_df.to_csv(os.path.join(REPORT_DIR, "top_bigrams.csv"), index=False)
    print("  Saved: top_bigrams.csv")

    cw_bi = classwise_ngrams(analysis_df, n=2, top_n=15)
    classwise_ngrams_to_dataframe(cw_bi).to_csv(
        os.path.join(REPORT_DIR, "classwise_bigrams.csv"), index=False
    )
    print("  Saved: classwise_bigrams.csv")

    # Top trigrams
    print("\n  Top Trigrams:")
    top_trigrams = extract_ngrams(
        analysis_df["text_normalized"], n=3, top_n=30
    )
    tri_df = ngrams_to_dataframe(top_trigrams, "Overall")
    tri_df.to_csv(os.path.join(REPORT_DIR, "top_trigrams.csv"), index=False)
    print("  Saved: top_trigrams.csv")

    cw_tri = classwise_ngrams(analysis_df, n=3, top_n=15)
    classwise_ngrams_to_dataframe(cw_tri).to_csv(
        os.path.join(REPORT_DIR, "classwise_trigrams.csv"), index=False
    )
    print("  Saved: classwise_trigrams.csv")

    # Visualizations
    viz.plot_top_ngrams(
        top_unigrams[:20],
        "Top 20 Unigrams (After Stopword Removal)",
        os.path.join(VIZ_DIR, "top_unigrams.png"),
        color="#4CAF50",
    )
    viz.plot_top_ngrams(
        top_bigrams[:20],
        "Top 20 Bigrams (After Stopword Removal)",
        os.path.join(VIZ_DIR, "top_bigrams.png"),
        color="#2196F3",
    )
    viz.plot_top_ngrams(
        top_trigrams[:20],
        "Top 20 Trigrams (After Stopword Removal)",
        os.path.join(VIZ_DIR, "top_trigrams.png"),
        color="#FF5722",
    )

    # Word clouds
    viz.generate_all_wordclouds(
        analysis_df, "text_normalized", VIZ_DIR
    )

    # ==============================================================
    # STEP 25: CLASS-WISE COMPARISON TABLE
    # ==============================================================
    section("STEP 25 — Class-Wise Comparison")

    print("\n  Consolidated Class-Wise Comparison:")
    print(social_summary.to_string(index=False))

    # ==============================================================
    # STEP 26: FEATURE DISTRIBUTIONS
    # ==============================================================
    section("STEP 26 — Feature Distributions")

    # Merge text and social features for correlation
    merged = text_features.merge(
        social_features.drop(columns=["text", "label", "label_id"]),
        left_index=True, right_index=True,
    )

    viz.plot_feature_distributions(
        merged,
        ["word_count", "character_count", "punctuation_count",
         "hashtag_count", "mention_count", "emoji_count", "uppercase_ratio"],
        os.path.join(VIZ_DIR, "feature_distributions.png"),
    )

    # ==============================================================
    # STEP 27: CORRELATION ANALYSIS
    # ==============================================================
    section("STEP 27 — Correlation Analysis")

    corr_columns = [
        "character_count", "word_count", "punctuation_count",
        "hashtag_count", "mention_count", "emoji_count",
        "uppercase_ratio", "exclamation_count", "question_count",
        "url_count",
    ]

    viz.plot_correlation_heatmap(
        merged,
        corr_columns,
        os.path.join(VIZ_DIR, "feature_correlation_heatmap.png"),
    )
    print("  Note: Correlation is between numeric EDA features only.")
    print("  Class label IDs are NOT included (they are categorical).")

    # ==============================================================
    # STEP 28: VOCABULARY ANALYSIS
    # ==============================================================
    section("STEP 28 — Vocabulary Analysis")

    vocab = vocabulary_analysis(analysis_df, "text_normalized")
    eda_summary["vocabulary_analysis"] = vocab

    print(f"  Total vocabulary: {vocab['total_vocabulary_size']:,} unique words")
    print(f"  Total word occurrences: {vocab['total_word_occurrences']:,}")
    print(f"  Rare words (≤2 occurrences): {vocab['rare_words_count']:,} ({vocab['rare_words_percentage']}%)")
    print(f"  Common vocabulary (in all classes): {vocab['common_vocabulary_size']:,}")
    print(f"  Class vocabulary sizes:")
    for label, size in vocab["class_vocabulary_sizes"].items():
        print(f"    {label}: {size:,}")
    print(f"  Class-specific words (unique to one class):")
    for label, count in vocab["class_specific_words"].items():
        print(f"    {label}: {count:,}")

    # ==============================================================
    # STEP 29: TEXT REPETITION ANALYSIS
    # ==============================================================
    section("STEP 29 — Text Repetition Analysis")

    dup_texts = train_df["text"].duplicated().sum()
    dup_texts_total = train_df["text"].duplicated(keep=False).sum()
    print(f"  Duplicate texts in train: {dup_texts:,}")
    print(f"  Total rows involved in duplicates: {dup_texts_total:,}")
    print("  (Phase 1 removed exact duplicate rows; remaining text duplicates")
    print("   may have different labels — this is a known dataset characteristic)")

    eda_summary["text_repetition"] = {
        "duplicate_texts": int(dup_texts),
        "rows_in_duplicates": int(dup_texts_total),
    }

    # ==============================================================
    # STEP 21: TF-IDF EXPLORATION (EDA ONLY)
    # ==============================================================
    section("STEP 21 — TF-IDF Class-Distinctive Terms (EDA Only)")

    tfidf_terms = tfidf_class_distinctive_terms(analysis_df, "text_normalized")
    eda_summary["tfidf_distinctive_terms"] = tfidf_terms

    if "error" not in tfidf_terms:
        for label, terms in tfidf_terms.items():
            top5 = [t["term"] for t in terms[:5]]
            print(f"  {label}: {top5}")
    else:
        print(f"  Error: {tfidf_terms['error']}")

    print("  Note: This TF-IDF is EDA-only; it is NOT saved for modeling.")

    # ==============================================================
    # STEP 30: SENTIMENT EXPLORATION (OPTIONAL)
    # ==============================================================
    section("STEP 30 — Sentiment Exploration (Optional, Conceptual)")

    print("  Sentiment analysis is NOT a target variable for this project.")
    print("  However, the concept of positive words in negative contexts")
    print("  is relevant to sarcasm detection.")
    print("  Phase 3+ will explore whether lexical sentiment features")
    print("  improve classification, without treating sentiment as a label.")

    eda_summary["sentiment_note"] = (
        "Sentiment analysis is optional and exploratory only. "
        "It is NOT a target variable. Sarcasm often involves positive "
        "wording with negative intent, which simple sentiment tools "
        "misinterpret. This observation informs preprocessing decisions."
    )

    # ==============================================================
    # STEP 31: RESEARCH OBSERVATIONS
    # ==============================================================
    section("STEP 31 — Important Research Observations")

    # Compile key observations from the data
    observations = []

    # Check for positive-looking words in sarcasm
    sarcasm_texts = analysis_df[analysis_df["label"] == "Sarcasm"]["text_normalized"]
    positive_words = ["love", "great", "wonderful", "amazing", "best", "awesome", "fantastic", "brilliant", "perfect", "thanks"]
    sarcasm_positive = {}
    for pw in positive_words:
        count = sarcasm_texts.str.contains(pw, na=False).sum()
        if count > 0:
            sarcasm_positive[pw] = int(count)

    if sarcasm_positive:
        observations.append({
            "pattern": "Positive words in sarcasm",
            "details": f"Sarcasm class contains positive-looking words: {dict(list(sarcasm_positive.items())[:5])}",
            "implication": "Confirms that sarcasm uses verbal irony (positive wording, negative intent)",
        })

    # Exaggeration patterns
    exaggeration_patterns = ["so ", "sooo", "literally", "always", "never", "every", "totally"]
    for label in ["Sarcasm", "Irony"]:
        subset = analysis_df[analysis_df["label"] == label]["text_normalized"]
        for pat in exaggeration_patterns[:3]:
            count = subset.str.contains(pat, na=False).sum()
            if count > 100:
                observations.append({
                    "pattern": f"Exaggeration in {label}",
                    "details": f"'{pat}' appears in {count:,} {label} tweets",
                    "implication": "Exaggeration may serve as a sarcasm/irony signal",
                })
                break

    eda_summary["research_observations"] = observations

    for obs in observations:
        print(f"\n  • {obs['pattern']}")
        print(f"    {obs['details']}")
        print(f"    → {obs['implication']}")

    # ==============================================================
    # SAVE EDA SUMMARY JSON
    # ==============================================================
    section("Saving EDA Summary")

    eda_summary["phase"] = "Phase 2 — Exploratory Data Analysis"
    eda_summary["project"] = "Sarcasm & Irony Detection System"
    eda_summary["timestamp"] = datetime.now().isoformat()
    eda_summary["train_records"] = len(train_df)
    eda_summary["test_records"] = len(test_df)
    eda_summary["total_records"] = len(train_df) + len(test_df)

    # Make JSON serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj

    eda_summary = make_serializable(eda_summary)

    with open(os.path.join(REPORT_DIR, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2, ensure_ascii=False, default=str)
    print("  Saved: eda_summary.json")

    # ==============================================================
    # GENERATE PREPROCESSING RECOMMENDATIONS
    # ==============================================================
    section("Generating Preprocessing Recommendations")

    generate_preprocessing_recommendations(
        eda_summary, hashtag_info, mention_info, url_info,
        emoji_info, caps_info, repeat_info, vocab, balance,
        os.path.join(REPORT_DIR, "preprocessing_recommendations.md"),
    )

    # ==============================================================
    # GENERATE EDA REPORT
    # ==============================================================
    section("Generating EDA Report")

    generate_eda_report(
        eda_summary, train_dist, test_dist, balance,
        social_summary, stats_table, vocab, outliers,
        hashtag_info, tfidf_terms, observations,
        top_unigrams, top_bigrams, top_trigrams,
        os.path.join(REPORT_DIR, "eda_report.md"),
    )

    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================
    elapsed = time.time() - start_time
    section("PHASE 2 COMPLETE")

    print(f"\n  Time elapsed: {elapsed:.1f}s")
    print(f"\n  DATASET: {len(train_df) + len(test_df):,} total records")
    print(f"    Train: {len(train_df):,}")
    print(f"    Test: {len(test_df):,}")
    print(f"\n  CLASS DISTRIBUTION (Train):")
    for _, row in train_dist.iterrows():
        print(f"    {row['Class']}: {row['Count']:,} ({row['Percentage']}%)")
    print(f"\n  AVERAGE TEXT LENGTH:")
    char_mean = text_features['character_count'].mean()
    word_mean = text_features['word_count'].mean()
    print(f"    {char_mean:.1f} characters")
    print(f"    {word_mean:.1f} words")
    print(f"\n  DATA QUALITY: {'Good' if train_quality['quality_status'] == 'CLEAN' and test_quality['quality_status'] == 'CLEAN' else 'Requires attention'}")
    print(f"\n  Files created in:")
    print(f"    Reports: {REPORT_DIR}")
    print(f"    Visualizations: {VIZ_DIR}")

    print("\n  ✓ Phase 2 EDA complete. Do NOT begin Phase 3 without explicit instruction.")


# ---------------------------------------------------------------------------
# Report Generators
# ---------------------------------------------------------------------------

def generate_preprocessing_recommendations(
    summary, hashtag_info, mention_info, url_info,
    emoji_info, caps_info, repeat_info, vocab, balance,
    output_path,
):
    """Generate preprocessing_recommendations.md from EDA findings."""
    lines = [
        "# Preprocessing Recommendations for Phase 3",
        "",
        "Based on Phase 2 EDA findings. These are recommendations to TEST,",
        "not final decisions.",
        "",
        "## 1. Hashtags",
        f"- {hashtag_info['percentage_with_hashtags']}% of tweets contain hashtags",
        f"- Top hashtags: {', '.join([h['hashtag'] for h in hashtag_info['top_hashtags'][:5]])}",
        "- **Recommendation**: PRESERVE hashtags initially. Some hashtags (e.g., #sarcasm, #irony)",
        "  may carry signal. Test with and without hashtag preservation.",
        "",
        "## 2. @Mentions",
        f"- {mention_info['percentage_with_mentions']}% of tweets contain mentions",
        "- **Recommendation**: Replace @mentions with a placeholder token (e.g., @USER)",
        "  to normalize identity references while preserving the mention pattern.",
        "",
        "## 3. URLs",
        f"- {url_info['percentage_with_urls']}% of tweets contain URLs",
        "- **Recommendation**: Replace URLs with a placeholder token (e.g., URL)",
        "  or remove entirely. URL content is not accessible.",
        "",
        "## 4. Emojis",
        f"- {emoji_info['percentage_with_emojis']}% of tweets contain emojis",
        "- **Recommendation**: Test emoji preservation vs. removal. Emojis may",
        "  carry pragmatic meaning relevant to sarcasm/irony.",
        "",
        "## 5. Punctuation",
        "- Exclamation marks and repeated punctuation appear across classes.",
        "- **Recommendation**: Preserve punctuation counts as features. Test whether",
        "  punctuation patterns improve classification.",
        "",
        "## 6. Capitalization",
        f"- Average uppercase ratio: {caps_info['avg_uppercase_ratio']}",
        f"- {caps_info['pct_with_allcaps_words']}% of tweets have all-caps words",
        "- **Recommendation**: Test case preservation vs. lowercasing.",
        "  Extract uppercase ratio as a feature before lowercasing.",
        "",
        "## 7. Repeated Characters",
        f"- {repeat_info['percentage_with_repeated_chars']}% of tweets have repeated chars",
        "- **Recommendation**: Normalize repeated characters (e.g., 'soooo' → 'so')",
        "  but extract a repeated-char count feature first.",
        "",
        "## 8. Stopword Removal",
        "- **Recommendation**: Test with and without stopword removal.",
        "  Stopwords may carry contextual meaning in short tweets.",
        "",
        "## 9. Lemmatization vs. Stemming",
        "- **Recommendation**: Test lemmatization (preserves word meaning)",
        "  vs. stemming (more aggressive) vs. no morphological normalization.",
        "",
        "## 10. N-grams",
        "- Bigrams and trigrams show distinctive patterns per class.",
        "- **Recommendation**: Include bigrams in TF-IDF features.",
        "  Test trigrams if computational budget allows.",
        "",
        "## 11. Vocabulary",
        f"- Total vocabulary: {vocab['total_vocabulary_size']:,} unique words",
        f"- Rare words: {vocab['rare_words_count']:,} ({vocab['rare_words_percentage']}%)",
        "- **Recommendation**: Apply minimum document frequency (min_df) to filter",
        "  rare words. Test min_df values of 2, 5, and 10.",
        "",
        "## 12. Class Balance",
        f"- Train balance: {balance['train']['balance_category']}",
        f"- Test balance: {balance['test']['balance_category']}",
        "- **Recommendation**: Report Macro F1 as primary metric. For the test set,",
        "  consider stratified evaluation. Test class weighting during training.",
        "",
        "---",
        "",
        "*These recommendations are based on EDA observations only.*",
        "*Phase 3 should systematically test each decision.*",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {output_path}")


def generate_eda_report(
    summary, train_dist, test_dist, balance,
    social_summary, stats_table, vocab, outliers,
    hashtag_info, tfidf_terms, observations,
    top_unigrams, top_bigrams, top_trigrams,
    output_path,
):
    """Generate the comprehensive EDA report."""
    lines = [
        "# Phase 2 — EDA Report",
        "",
        f"**Project**: Sarcasm & Irony Detection System",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        f"| Dataset | Records | Columns |",
        f"|---------|--------:|--------:|",
        f"| Train   | {summary['train_records']:,} | 3 |",
        f"| Test    | {summary['test_records']:,} | 3 |",
        f"| **Total** | **{summary['total_records']:,}** | **3** |",
        "",
        "Columns: `text`, `label`, `label_id`",
        "",
        "Label mapping: Regular=0, Sarcasm=1, Irony=2, Figurative=3",
        "",
        "## 2. Data Quality",
        "",
    ]

    for ds in ["train", "test"]:
        qc = summary["quality_checks"][ds]
        lines.append(f"### {qc['dataset']} Set")
        lines.append(f"- Status: **{qc['quality_status']}**")
        lines.append(f"- Missing text: {qc['missing_text']}")
        lines.append(f"- Missing labels: {qc['missing_label']}")
        lines.append(f"- Empty strings: {qc['empty_strings']}")
        lines.append(f"- Duplicate rows: {qc['duplicate_rows']}")
        lines.append(f"- Duplicate texts: {qc['duplicate_texts']}")
        lines.append(f"- Invalid labels: {qc['invalid_labels']}")
        lines.append("")

    # Class Distribution
    lines.extend([
        "## 3. Class Distribution",
        "",
        "### Training Set",
        "",
        "| Class | Count | Percentage |",
        "|-------|------:|-----------:|",
    ])
    for _, row in train_dist.iterrows():
        lines.append(f"| {row['Class']} | {row['Count']:,} | {row['Percentage']:.2f}% |")

    lines.extend(["", "### Test Set", "",
        "| Class | Count | Percentage |",
        "|-------|------:|-----------:|",
    ])
    for _, row in test_dist.iterrows():
        lines.append(f"| {row['Class']} | {row['Count']:,} | {row['Percentage']:.2f}% |")

    lines.extend([
        "",
        "### Class Balance Analysis",
        "",
        f"- **Train**: {balance['train']['balance_category']}",
        f"  - Majority/minority ratio: {balance['train']['majority_minority_ratio']}",
        f"  - {balance['train']['implication']}",
        f"- **Test**: {balance['test']['balance_category']}",
        f"  - Majority/minority ratio: {balance['test']['majority_minority_ratio']}",
        "",
        "![Class Distribution](../../results/eda/class_distribution.png)",
        "",
    ])

    # Text Statistics
    lines.extend([
        "## 4. Text Length Statistics",
        "",
        stats_table.to_markdown(index=False),
        "",
        "![Character Count Distribution](../../results/eda/character_count_distribution.png)",
        "",
        "![Word Count Distribution](../../results/eda/word_count_distribution.png)",
        "",
        "![Character Count by Class](../../results/eda/character_count_by_class.png)",
        "",
        "![Word Count by Class](../../results/eda/word_count_by_class.png)",
        "",
    ])

    # Outliers
    lines.extend(["## 5. Outlier Analysis", ""])
    for col, info in outliers.items():
        lines.append(f"### {col}")
        lines.append(f"- IQR bounds: [{info['lower_bound']:.0f}, {info['upper_bound']:.0f}]")
        lines.append(f"- Low outliers: {info['low_outlier_count']}")
        lines.append(f"- High outliers: {info['high_outlier_count']}")
        lines.append(f"- Outlier percentage: {info['outlier_percentage']}%")
        lines.append(f"- **Note**: Outliers are NOT removed. Sarcasm detection may involve intentionally unusual writing.")
        lines.append("")

    # Social Media
    lines.extend([
        "## 6. Social Media Patterns",
        "",
        "### Class-Wise Comparison",
        "",
        social_summary.to_markdown(index=False),
        "",
        "![Hashtags by Class](../../results/eda/hashtags_by_class.png)",
        "",
        "![Mentions by Class](../../results/eda/mentions_by_class.png)",
        "",
        "![Emojis by Class](../../results/eda/emojis_by_class.png)",
        "",
        "![Punctuation by Class](../../results/eda/punctuation_by_class.png)",
        "",
        "![Uppercase by Class](../../results/eda/uppercase_by_class.png)",
        "",
    ])

    # Top hashtags
    if hashtag_info.get("top_hashtags"):
        lines.extend(["### Top Hashtags", "",
            "| Hashtag | Count |",
            "|---------|------:|",
        ])
        for h in hashtag_info["top_hashtags"][:15]:
            lines.append(f"| {h['hashtag']} | {h['count']:,} |")
        lines.append("")

    # Word Frequency
    lines.extend([
        "## 7. Word Frequency Analysis",
        "",
        "### Top 20 Unigrams (After Stopword Removal)",
        "",
        "| Word | Frequency |",
        "|------|----------:|",
    ])
    for word, count in top_unigrams[:20]:
        lines.append(f"| {word} | {count:,} |")

    lines.extend([
        "",
        "![Top Unigrams](../../results/eda/top_unigrams.png)",
        "",
    ])

    # Bigrams
    lines.extend([
        "### Top 20 Bigrams",
        "",
        "| Bigram | Frequency |",
        "|--------|----------:|",
    ])
    for ng, count in top_bigrams[:20]:
        lines.append(f"| {ng} | {count:,} |")

    lines.extend([
        "",
        "![Top Bigrams](../../results/eda/top_bigrams.png)",
        "",
    ])

    # Trigrams
    lines.extend([
        "### Top 20 Trigrams",
        "",
        "| Trigram | Frequency |",
        "|---------|----------:|",
    ])
    for ng, count in top_trigrams[:20]:
        lines.append(f"| {ng} | {count:,} |")

    lines.extend([
        "",
        "![Top Trigrams](../../results/eda/top_trigrams.png)",
        "",
    ])

    # TF-IDF
    if tfidf_terms and "error" not in tfidf_terms:
        lines.extend(["## 8. TF-IDF Class-Distinctive Terms (EDA Only)", ""])
        for label, terms in tfidf_terms.items():
            lines.append(f"### {label}")
            lines.append(f"| Term | TF-IDF Score |")
            lines.append(f"|------|------------:|")
            for t in terms[:10]:
                lines.append(f"| {t['term']} | {t['tfidf_score']:.4f} |")
            lines.append("")

    # Vocabulary
    lines.extend([
        "## 9. Vocabulary Analysis",
        "",
        f"- Total vocabulary: {vocab['total_vocabulary_size']:,} unique words",
        f"- Total word occurrences: {vocab['total_word_occurrences']:,}",
        f"- Rare words (≤2 occurrences): {vocab['rare_words_count']:,} ({vocab['rare_words_percentage']}%)",
        f"- Common vocabulary (in all classes): {vocab['common_vocabulary_size']:,}",
        "",
        "| Class | Vocabulary Size | Unique Words |",
        "|-------|----------------:|-------------:|",
    ])
    for label, size in vocab["class_vocabulary_sizes"].items():
        unique = vocab["class_specific_words"].get(label, 0)
        lines.append(f"| {label} | {size:,} | {unique:,} |")

    # Word clouds
    lines.extend([
        "",
        "## 10. Word Clouds",
        "",
        "![Overall Word Cloud](../../results/eda/wordcloud_overall.png)",
        "",
        "![Regular Word Cloud](../../results/eda/wordcloud_regular.png)",
        "",
        "![Sarcasm Word Cloud](../../results/eda/wordcloud_sarcasm.png)",
        "",
        "![Irony Word Cloud](../../results/eda/wordcloud_irony.png)",
        "",
        "![Figurative Word Cloud](../../results/eda/wordcloud_figurative.png)",
        "",
    ])

    # Observations
    lines.extend(["## 11. Important Research Observations", ""])
    for obs in observations:
        lines.append(f"- **{obs['pattern']}**: {obs['details']}")
        lines.append(f"  - *Implication*: {obs['implication']}")
        lines.append("")

    # Data limitations
    lines.extend([
        "## 12. Data Limitations",
        "",
        "- Tweet text only — no metadata (timestamp, user info, conversation context)",
        "- Sarcasm/irony labels are subjective and may have inter-annotator disagreement",
        "- Dataset contains duplicate texts with different labels (text ambiguity)",
        "- Test set distribution differs from training due to overlap removal in Phase 1",
        "- Character limit of ~160 chars constrains linguistic analysis",
        "",
    ])

    # Research questions
    lines.extend([
        "## 13. Research Questions for Phase 3+",
        "",
        "1. Does punctuation preservation improve sarcasm classification?",
        "2. Do hashtags (#sarcasm, #irony) provide useful signal or introduce bias?",
        "3. Does emoji preservation help differentiate classes?",
        "4. Does lemmatization improve TF-IDF performance?",
        "5. Are sarcasm and irony text lengths meaningfully different?",
        "6. Does adding bigrams improve classification accuracy?",
        "7. Does social-media feature engineering improve classical ML models?",
        "8. Can uppercase ratio serve as a distinguishing feature?",
        "9. Does class weighting improve Macro F1 on the imbalanced test set?",
        "10. How do repeated character patterns affect model performance?",
        "",
    ])

    # Preprocessing implications
    lines.extend([
        "## 14. Preprocessing Implications",
        "",
        "See `preprocessing_recommendations.md` for detailed recommendations.",
        "",
        "---",
        "",
        "*This report is generated from Phase 2 EDA. No models were trained.*",
        "*All statistics are from the actual cleaned dataset.*",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
