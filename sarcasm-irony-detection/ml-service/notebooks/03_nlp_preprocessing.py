"""
Phase 3 — NLP Text Preprocessing Notebook
============================================
Sarcasm & Irony Detection System using NLP

This script demonstrates the complete preprocessing pipeline
interactively. Run it to see sample outputs and comparisons.

Sections:
    1. Imports & Setup
    2. Load Data
    3. Load Preprocessing Configuration
    4. Sample Inspection
    5. URL Handling
    6. Mention Handling
    7. Hashtag Handling
    8. Emoji Handling
    9. Punctuation Handling
    10. Capitalization
    11. Repeated Characters
    12. Negation Handling
    13. Tokenization
    14. Stopword Experiment
    15. Lemmatization
    16. Stemming
    17. Variant Comparison
    18. Feature Extraction
    19. Quality Checks
    20. Conclusions

Usage:
    python notebooks/03_nlp_preprocessing.py
"""

import sys
import os
from pathlib import Path

# Setup path
NOTEBOOK_DIR = Path(__file__).resolve().parent
ML_SERVICE_DIR = NOTEBOOK_DIR.parent
sys.path.insert(0, str(ML_SERVICE_DIR))

import pandas as pd
import numpy as np

# Import preprocessing modules
from src.preprocessing.setup_nltk import setup_nltk
from src.preprocessing.text_cleaner import validate_text
from src.preprocessing.social_media import (
    normalize_urls, normalize_mentions,
    handle_hashtags, handle_emojis,
    count_urls, count_mentions, count_hashtags, count_emojis,
    extract_social_media_features,
)
from src.preprocessing.normalization import (
    normalize_whitespace, normalize_case,
    normalize_punctuation, normalize_repeated_chars,
    remove_stopwords_from_tokens, extract_text_features,
    NEGATION_WORDS, count_repeated_chars,
)
from src.preprocessing.tokenizer import tokenize_text, tokens_to_string
from src.preprocessing.lemmatization import lemmatize_tokens
from src.preprocessing.stemming import stem_tokens
from src.preprocessing.config import (
    PreprocessingConfig, get_all_variants, get_variant,
)
from src.preprocessing.pipeline import PreprocessingPipeline


def section(title, num):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {num}. {title}")
    print(f"{'='*70}\n")


def main():
    # ============================================================
    # 1. IMPORTS & SETUP
    # ============================================================
    section("IMPORTS & SETUP", 1)
    print("Setting up NLTK resources...")
    setup_nltk()
    print("All imports loaded successfully.\n")
    print("Libraries: pandas, numpy, nltk, emoji, re")
    
    # ============================================================
    # 2. LOAD DATA
    # ============================================================
    section("LOAD DATA", 2)
    
    DATA_DIR = ML_SERVICE_DIR / "data" / "processed"
    train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
    test_df = pd.read_csv(DATA_DIR / "test_clean.csv")
    
    print(f"Training set: {train_df.shape}")
    print(f"Test set:     {test_df.shape}")
    print(f"\nColumns: {train_df.columns.tolist()}")
    print(f"\nLabel distribution (train):")
    print(train_df['label'].value_counts().to_string())
    print(f"\nLabel distribution (test):")
    print(test_df['label'].value_counts().to_string())
    
    # ============================================================
    # 3. LOAD PREPROCESSING CONFIGURATION
    # ============================================================
    section("PREPROCESSING CONFIGURATION", 3)
    
    variants = get_all_variants()
    print(f"Available preprocessing variants: {list(variants.keys())}\n")
    
    for name, config in variants.items():
        print(f"  {name:12s}: {config.description}")
    
    print("\nDefault (minimal) configuration:")
    minimal = get_variant("minimal")
    for key, value in minimal.to_dict().items():
        if key not in ('name', 'description'):
            print(f"  {key}: {value}")
    
    # ============================================================
    # 4. SAMPLE INSPECTION
    # ============================================================
    section("SAMPLE INSPECTION", 4)
    
    # Select diverse samples
    np.random.seed(42)
    sample_indices = np.random.choice(train_df.index, 5, replace=False)
    samples = train_df.loc[sample_indices]
    
    for _, row in samples.iterrows():
        print(f"  Label: {row['label']:12s} | Text: {row['text'][:100]}...")
        print()
    
    # ============================================================
    # 5. URL HANDLING
    # ============================================================
    section("URL HANDLING", 5)
    
    url_examples = [
        "check https://example.com/page for details",
        "link http://t.co/abcXYZ123 here",
        "no url in this text",
        "two urls https://a.com and https://b.com",
    ]
    
    for text in url_examples:
        result = normalize_urls(text)
        n = count_urls(text)
        print(f"  URL count: {n}")
        print(f"  Original:    {text}")
        print(f"  Normalized:  {result}")
        print()
    
    # ============================================================
    # 6. MENTION HANDLING
    # ============================================================
    section("MENTION HANDLING", 6)
    
    mention_examples = [
        "@john said hello @jane",
        "no mentions here",
        "@user1 @user2 @user3 chatting",
    ]
    
    for text in mention_examples:
        result = normalize_mentions(text)
        n = count_mentions(text)
        print(f"  Mention count: {n}")
        print(f"  Original:    {text}")
        print(f"  Normalized:  {result}")
        print()
    
    # ============================================================
    # 7. HASHTAG HANDLING
    # ============================================================
    section("HASHTAG HANDLING", 7)
    
    hashtag_examples = [
        "#sarcasm is great",
        "#WhatAGreatIdea really",
        "#NLP #MachineLearning #AI",
    ]
    
    modes = ["preserve", "strip", "segment"]
    for text in hashtag_examples:
        print(f"  Original: {text}  (hashtag count: {count_hashtags(text)})")
        for mode in modes:
            result = handle_hashtags(text, mode=mode)
            print(f"    {mode:10s}: {result}")
        print()
    
    # ============================================================
    # 8. EMOJI HANDLING
    # ============================================================
    section("EMOJI HANDLING", 8)
    
    emoji_examples = [
        "great job 😂",
        "so tired 😒🙃",
        "love this ❤️✨",
        "no emojis here",
    ]
    
    modes = ["preserve", "text", "remove"]
    for text in emoji_examples:
        print(f"  Original: {text}  (emoji count: {count_emojis(text)})")
        for mode in modes:
            result = handle_emojis(text, mode=mode)
            print(f"    {mode:10s}: {result}")
        print()
    
    # ============================================================
    # 9. PUNCTUATION HANDLING
    # ============================================================
    section("PUNCTUATION HANDLING", 9)
    
    punct_examples = [
        "Great!!!",
        "Really???",
        "Oh..... wonderful",
        "What!? Seriously!?",
        "normal sentence.",
    ]
    
    modes = ["preserve", "normalize", "remove"]
    for text in punct_examples:
        print(f"  Original: {text}")
        for mode in modes:
            result = normalize_punctuation(text, mode=mode)
            print(f"    {mode:10s}: {result}")
        print()
    
    # ============================================================
    # 10. CAPITALIZATION
    # ============================================================
    section("CAPITALIZATION", 10)
    
    cap_examples = [
        "WHAT A GREAT IDEA",
        "Hello World",
        "this is normal",
        "SO AMAZING omg",
    ]
    
    for text in cap_examples:
        preserved = normalize_case(text, lowercase=False)
        lowered = normalize_case(text, lowercase=True)
        print(f"  Original:   {text}")
        print(f"  Preserved:  {preserved}")
        print(f"  Lowercase:  {lowered}")
        print()
    
    # ============================================================
    # 11. REPEATED CHARACTERS
    # ============================================================
    section("REPEATED CHARACTERS", 11)
    
    repeated_examples = [
        "sooooo good",
        "greatttt job",
        "loooool",
        "normal text",
        "goooood moooorning",
    ]
    
    for text in repeated_examples:
        result = normalize_repeated_chars(text, max_repeat=2)
        count = count_repeated_chars(text)
        print(f"  Original:   {text}  (repeated: {count})")
        print(f"  Normalized: {result}")
        print()
    
    # ============================================================
    # 12. NEGATION HANDLING
    # ============================================================
    section("NEGATION HANDLING", 12)
    
    print("Protected negation words:")
    print(f"  {sorted(NEGATION_WORDS)}\n")
    
    negation_examples = [
        "I do not like this",
        "She never said that",
        "No one cares",
        "I can't believe it",
        "This isn't funny at all",
        "He didn't say anything",
    ]
    
    for text in negation_examples:
        tokens = tokenize_text(text)
        with_protection = remove_stopwords_from_tokens(tokens, protect_negations=True)
        without_protection = remove_stopwords_from_tokens(tokens, protect_negations=False)
        
        print(f"  Original:        {text}")
        print(f"  Tokens:          {tokens}")
        print(f"  Stopwords (protected):   {with_protection}")
        print(f"  Stopwords (unprotected): {without_protection}")
        print()
    
    # ============================================================
    # 13. TOKENIZATION
    # ============================================================
    section("TOKENIZATION", 13)
    
    token_examples = [
        "Great!!! Another exam tomorrow 😒",
        "@john WOW!!! #sarcasm 😂 https://example.com",
        "I don't think this is funny",
        "She's running quickly towards the finish line",
        "normal everyday sentence",
    ]
    
    for text in token_examples:
        tokens = tokenize_text(text)
        print(f"  Text:   {text}")
        print(f"  Tokens: {tokens}")
        print(f"  Count:  {len(tokens)}")
        print()
    
    # ============================================================
    # 14. STOPWORD EXPERIMENT
    # ============================================================
    section("STOPWORD EXPERIMENT", 14)
    
    stopword_examples = [
        "I love this but I do not like that",
        "The quick brown fox jumps over the lazy dog",
        "This is not what I expected at all",
    ]
    
    for text in stopword_examples:
        tokens = tokenize_text(text)
        retained = remove_stopwords_from_tokens(tokens, protect_negations=True)
        removed = remove_stopwords_from_tokens(tokens, protect_negations=False)
        
        print(f"  Original:            {text}")
        print(f"  With negation prot:  {tokens_to_string(retained)}")
        print(f"  Without protection:  {tokens_to_string(removed)}")
        print()
    
    # ============================================================
    # 15. LEMMATIZATION
    # ============================================================
    section("LEMMATIZATION", 15)
    
    lemma_examples = [
        ["running", "quickly", "towards", "better", "days"],
        ["cats", "were", "playing", "with", "mice"],
        ["he", "was", "happily", "eating", "apples"],
    ]
    
    for tokens in lemma_examples:
        lemmatized = lemmatize_tokens(tokens)
        print(f"  Tokens:     {tokens}")
        print(f"  Lemmatized: {lemmatized}")
        print()
    
    # ============================================================
    # 16. STEMMING
    # ============================================================
    section("STEMMING", 16)
    
    stem_examples = [
        ["running", "happily", "connections", "beautiful"],
        ["playing", "played", "player", "plays"],
        ["sarcastic", "sarcastically", "sarcasm"],
    ]
    
    for tokens in stem_examples:
        stemmed = stem_tokens(tokens)
        print(f"  Tokens:  {tokens}")
        print(f"  Stemmed: {stemmed}")
        print()
    
    print("Note: Stemming can produce non-word outputs. This is expected.")
    print("Stemming is more aggressive than lemmatization.")
    
    # ============================================================
    # 17. VARIANT COMPARISON
    # ============================================================
    section("VARIANT COMPARISON", 17)
    
    comparison_texts = [
        "@user WOW!!! That's SOOO #sarcastic 😂 https://t.co/abc",
        "I don't think this is funny at all...",
        "GREAT job everyone #NotReally #irony",
        "She's running towards better days 😒😒😒",
    ]
    
    variants = get_all_variants()
    
    for text in comparison_texts:
        print(f"  Original: {text}")
        print()
        for var_name, config in variants.items():
            pipeline = PreprocessingPipeline(config)
            result = pipeline.process_text(text)
            print(f"    {var_name:12s}: {result}")
        print(f"  {'─'*60}")
        print()
    
    # ============================================================
    # 18. FEATURE EXTRACTION
    # ============================================================
    section("FEATURE EXTRACTION", 18)
    
    feature_text = "@john WOW!!! #sarcasm 😂😂 https://example.com sooooo good"
    
    print(f"  Text: {feature_text}\n")
    
    sm_features = extract_social_media_features(feature_text)
    print("  Social Media Features:")
    for k, v in sm_features.items():
        print(f"    {k:20s}: {v}")
    
    txt_features = extract_text_features(feature_text)
    print("\n  Text Features:")
    for k, v in txt_features.items():
        print(f"    {k:25s}: {v}")
    
    # ============================================================
    # 19. QUALITY CHECKS
    # ============================================================
    section("QUALITY CHECKS", 19)
    
    # Quick quality check on a small sample with minimal variant
    sample = train_df.head(100).copy()
    minimal_config = get_variant("minimal")
    pipeline = PreprocessingPipeline(minimal_config)
    processed = pipeline.process_dataframe(sample)
    
    print(f"  Original rows:  {len(sample)}")
    print(f"  Processed rows: {len(processed)}")
    print(f"  Nulls:          {processed['processed_text'].isna().sum()}")
    print(f"  Empty:          {(processed['processed_text'] == '').sum()}")
    print(f"  Labels match:   {sample['label'].equals(processed['label'])}")
    print(f"  Label IDs match: {sample['label_id'].equals(processed['label_id'])}")
    
    # ============================================================
    # 20. CONCLUSIONS
    # ============================================================
    section("CONCLUSIONS", 20)
    
    print("""
  Phase 3 NLP Preprocessing Summary:
  
  1. Six controlled preprocessing variants have been created (A-F)
  2. Social-media signals (hashtags, emojis, punctuation, capitalization)
     are preserved by default in the minimal variant
  3. Negation words are protected during stopword removal
  4. URLs and mentions are normalized to placeholders
  5. Multiple handling modes are available for hashtags and emojis
  6. Lemmatization uses POS-tagged WordNet for accuracy
  7. Stemming (Porter) is available as an experimental option
  8. All preprocessing is configurable and reproducible
  
  IMPORTANT: No preprocessing variant is declared "best" yet.
  The optimal strategy will be determined through classification
  experiments in Phase 4/5.
  
  Recommendations for Phase 4/5 experiments:
  
  Experiment P1: Minimal preprocessing (Variant A)
  Experiment P2: Lowercase (Variant B)
  Experiment P3: Lemmatization (Variant C)
  Experiment P4: Stopword removal with negation protection (Variant D)
  Experiment P5: Stemming (Variant E)
  Experiment P6: Classical cleaning (Variant F)
    """)


if __name__ == "__main__":
    main()
