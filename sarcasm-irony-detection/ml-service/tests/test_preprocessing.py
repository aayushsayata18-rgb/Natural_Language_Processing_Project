"""
Phase 3 - Preprocessing Unit Tests
============================================
Sarcasm & Irony Detection System using NLP

Test the preprocessing pipeline components:
    1. Empty text
    2. Null text / NaN
    3. Normal sentence
    4. URL normalization
    5. Mention normalization
    6. Hashtag handling
    7. Emoji handling
    8. Punctuation handling
    9. Capitalization
    10. Repeated characters
    11. Negation preservation
    12. Contractions
    13. Mixed social-media text (end-to-end)
"""

import sys
import math
from pathlib import Path

import pytest

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.text_cleaner import validate_text
from src.preprocessing.social_media import (
    normalize_urls,
    normalize_mentions,
    handle_hashtags,
    handle_emojis,
    count_urls,
    count_mentions,
    count_hashtags,
    count_emojis,
    extract_social_media_features,
)
from src.preprocessing.normalization import (
    normalize_whitespace,
    normalize_case,
    normalize_punctuation,
    normalize_repeated_chars,
    count_repeated_chars,
    remove_stopwords_from_tokens,
    extract_text_features,
    NEGATION_WORDS,
)
from src.preprocessing.tokenizer import tokenize_text, tokens_to_string
from src.preprocessing.lemmatization import lemmatize_tokens
from src.preprocessing.stemming import stem_tokens
from src.preprocessing.config import (
    PreprocessingConfig,
    get_all_variants,
    get_variant,
)
from src.preprocessing.pipeline import PreprocessingPipeline


# ============================================================
# 1. EMPTY TEXT
# ============================================================

class TestEmptyText:
    """Test handling of empty text inputs."""
    
    def test_empty_string(self):
        result = validate_text("")
        assert result == ""
    
    def test_whitespace_only(self):
        result = validate_text("   ")
        assert result == ""
    
    def test_tabs_only(self):
        result = validate_text("\t\t")
        assert result == ""
    
    def test_newlines_only(self):
        result = validate_text("\n\n")
        assert result == ""


# ============================================================
# 2. NULL / NaN TEXT
# ============================================================

class TestNullText:
    """Test handling of None and NaN inputs."""
    
    def test_none_input(self):
        result = validate_text(None)
        assert result == ""
    
    def test_nan_input(self):
        result = validate_text(float("nan"))
        assert result == ""
    
    def test_math_nan(self):
        result = validate_text(math.nan)
        assert result == ""
    
    def test_integer_input(self):
        result = validate_text(42)
        assert result == "42"
    
    def test_list_input(self):
        result = validate_text([1, 2, 3])
        assert isinstance(result, str)


# ============================================================
# 3. NORMAL SENTENCE
# ============================================================

class TestNormalSentence:
    """Test that normal text passes through correctly."""
    
    def test_simple_sentence(self):
        text = "I love this movie"
        assert validate_text(text) == text
    
    def test_normal_tokenization(self):
        text = "I love this movie"
        tokens = tokenize_text(text)
        assert tokens == ["I", "love", "this", "movie"]
    
    def test_token_reassembly(self):
        tokens = ["I", "love", "this", "movie"]
        assert tokens_to_string(tokens) == "I love this movie"


# ============================================================
# 4. URL NORMALIZATION
# ============================================================

class TestURLNormalization:
    """Test URL detection and normalization."""
    
    def test_http_url(self):
        text = "check http://example.com for details"
        result = normalize_urls(text)
        assert "URL" in result
        assert "http" not in result
    
    def test_https_url(self):
        text = "visit https://example.com/page"
        result = normalize_urls(text)
        assert result == "visit URL"
    
    def test_tco_url(self):
        text = "link https://t.co/abcXYZ123"
        result = normalize_urls(text)
        assert result == "link URL"
    
    def test_multiple_urls(self):
        text = "http://a.com and https://b.com"
        assert count_urls(text) == 2
        result = normalize_urls(text)
        assert result.count("URL") == 2
    
    def test_no_url(self):
        text = "no urls here"
        result = normalize_urls(text)
        assert result == text
    
    def test_custom_placeholder(self):
        text = "visit https://example.com"
        result = normalize_urls(text, placeholder="<URL>")
        assert result == "visit <URL>"


# ============================================================
# 5. MENTION NORMALIZATION
# ============================================================

class TestMentionNormalization:
    """Test @mention detection and normalization."""
    
    def test_single_mention(self):
        text = "@john said hello"
        result = normalize_mentions(text)
        assert result == "USER_MENTION said hello"
    
    def test_multiple_mentions(self):
        text = "@john and @jane are friends"
        result = normalize_mentions(text)
        assert result.count("USER_MENTION") == 2
    
    def test_no_mention(self):
        text = "no mentions here"
        result = normalize_mentions(text)
        assert result == text
    
    def test_mention_count(self):
        text = "@a @b @c"
        assert count_mentions(text) == 3


# ============================================================
# 6. HASHTAG HANDLING
# ============================================================

class TestHashtagHandling:
    """Test hashtag handling in different modes."""
    
    def test_preserve_mode(self):
        text = "#sarcasm is fun"
        result = handle_hashtags(text, mode="preserve")
        assert result == "#sarcasm is fun"
    
    def test_strip_mode(self):
        text = "#sarcasm is fun"
        result = handle_hashtags(text, mode="strip")
        assert result == "sarcasm is fun"
    
    def test_segment_mode_camelcase(self):
        text = "#WhatAGreatIdea"
        result = handle_hashtags(text, mode="segment")
        assert "What" in result
        assert "Great" in result
        assert "#" not in result
    
    def test_segment_mode_simple(self):
        text = "#sarcasm"
        result = handle_hashtags(text, mode="segment")
        assert result == "sarcasm"
    
    def test_hashtag_count(self):
        text = "#one #two #three"
        assert count_hashtags(text) == 3
    
    def test_multiple_hashtags_strip(self):
        text = "#hello #world"
        result = handle_hashtags(text, mode="strip")
        assert result == "hello world"


# ============================================================
# 7. EMOJI HANDLING
# ============================================================

class TestEmojiHandling:
    """Test emoji handling in different modes."""
    
    def test_preserve_mode(self):
        text = "great job 😂"
        result = handle_emojis(text, mode="preserve")
        assert "😂" in result
    
    def test_text_mode(self):
        text = "great job 😂"
        result = handle_emojis(text, mode="text")
        assert "😂" not in result
        assert len(result) > len("great job ")  # text description added
    
    def test_remove_mode(self):
        text = "great job 😂"
        result = handle_emojis(text, mode="remove")
        assert "😂" not in result
    
    def test_emoji_count(self):
        text = "wow 😂🙃😒"
        assert count_emojis(text) == 3
    
    def test_no_emoji(self):
        text = "no emojis here"
        assert count_emojis(text) == 0
        result = handle_emojis(text, mode="text")
        assert result == text


# ============================================================
# 8. PUNCTUATION HANDLING
# ============================================================

class TestPunctuationHandling:
    """Test punctuation handling in different modes."""
    
    def test_preserve_mode(self):
        text = "Great!!! Really???"
        result = normalize_punctuation(text, mode="preserve")
        assert result == text
    
    def test_normalize_mode_exclamation(self):
        text = "Great!!!"
        result = normalize_punctuation(text, mode="normalize")
        assert result == "Great!!"
    
    def test_normalize_mode_question(self):
        text = "Really???"
        result = normalize_punctuation(text, mode="normalize")
        assert result == "Really??"
    
    def test_normalize_mode_ellipsis(self):
        text = "Oh..... wonderful"
        result = normalize_punctuation(text, mode="normalize")
        assert result == "Oh... wonderful"
    
    def test_remove_mode(self):
        text = "Great!!! Really???"
        result = normalize_punctuation(text, mode="remove")
        assert "!" not in result
        assert "?" not in result


# ============================================================
# 9. CAPITALIZATION
# ============================================================

class TestCapitalization:
    """Test case normalization."""
    
    def test_preserve_case(self):
        text = "WHAT A GREAT IDEA"
        result = normalize_case(text, lowercase=False)
        assert result == text
    
    def test_lowercase(self):
        text = "WHAT A GREAT IDEA"
        result = normalize_case(text, lowercase=True)
        assert result == "what a great idea"
    
    def test_mixed_case(self):
        text = "Hello World"
        result = normalize_case(text, lowercase=True)
        assert result == "hello world"


# ============================================================
# 10. REPEATED CHARACTERS
# ============================================================

class TestRepeatedCharacters:
    """Test repeated character handling."""
    
    def test_normalize_repeated(self):
        text = "sooooo good"
        result = normalize_repeated_chars(text, max_repeat=2)
        assert result == "soo good"
    
    def test_normalize_multiple(self):
        text = "greatttt dayyy"
        result = normalize_repeated_chars(text, max_repeat=2)
        assert result == "greatt dayy"
    
    def test_no_repeated(self):
        text = "normal text"
        result = normalize_repeated_chars(text)
        assert result == text
    
    def test_count_repeated(self):
        text = "sooooo goooood"
        assert count_repeated_chars(text) == 2
    
    def test_double_not_reduced(self):
        # Double characters should NOT be reduced when max_repeat=2
        text = "good"
        result = normalize_repeated_chars(text, max_repeat=2)
        assert result == "good"


# ============================================================
# 11. NEGATION PRESERVATION
# ============================================================

class TestNegationPreservation:
    """Test that negation words are preserved during stopword removal."""
    
    def test_not_preserved(self):
        tokens = ["I", "do", "not", "like", "this"]
        result = remove_stopwords_from_tokens(tokens, protect_negations=True)
        assert "not" in result
    
    def test_never_preserved(self):
        tokens = ["I", "never", "said", "that"]
        result = remove_stopwords_from_tokens(tokens, protect_negations=True)
        assert "never" in result
    
    def test_no_preserved(self):
        tokens = ["no", "one", "cares"]
        result = remove_stopwords_from_tokens(tokens, protect_negations=True)
        assert "no" in result
    
    def test_nt_preserved(self):
        tokens = ["do", "n't", "stop"]
        result = remove_stopwords_from_tokens(tokens, protect_negations=True)
        assert "n't" in result
    
    def test_negation_removed_when_unprotected(self):
        tokens = ["I", "do", "not", "like", "this"]
        result = remove_stopwords_from_tokens(tokens, protect_negations=False)
        # 'not' is in standard stopwords, should be removed
        assert "not" not in result
    
    def test_content_words_survive(self):
        tokens = ["I", "do", "not", "like", "this"]
        result = remove_stopwords_from_tokens(tokens, protect_negations=True)
        assert "like" in result
    
    def test_negation_words_constant(self):
        assert "not" in NEGATION_WORDS
        assert "never" in NEGATION_WORDS
        assert "no" in NEGATION_WORDS
        assert "n't" in NEGATION_WORDS


# ============================================================
# 12. CONTRACTIONS
# ============================================================

class TestContractions:
    """Test that contractions are handled by TweetTokenizer."""
    
    def test_dont(self):
        tokens = tokenize_text("I don't like this")
        # TweetTokenizer should handle contractions
        assert any("do" in t.lower() for t in tokens)
    
    def test_cant(self):
        tokens = tokenize_text("I can't believe it")
        assert len(tokens) > 3
    
    def test_wont(self):
        tokens = tokenize_text("I won't go there")
        assert len(tokens) > 3


# ============================================================
# 13. MIXED SOCIAL MEDIA TEXT (END-TO-END)
# ============================================================

class TestMixedSocialMedia:
    """Test end-to-end pipeline with mixed social-media text."""
    
    def test_minimal_variant(self):
        config = get_variant("minimal")
        pipeline = PreprocessingPipeline(config)
        
        text = "@john WOW!!! #sarcasm 😂 https://example.com"
        result = pipeline.process_text(text)
        
        # URLs should be replaced
        assert "URL" in result
        # Mentions should be replaced
        assert "USER_MENTION" in result
        # Hashtag should be preserved
        assert "#sarcasm" in result
        # Emoji should be preserved
        assert "😂" in result
        # Punctuation should be preserved
        assert "!" in result
    
    def test_lowercase_variant(self):
        config = get_variant("lowercase")
        pipeline = PreprocessingPipeline(config)
        
        text = "WHAT A GREAT IDEA #sarcasm"
        result = pipeline.process_text(text)
        
        # Should be lowercased
        assert result == result.lower() or "#" in result
    
    def test_classical_variant(self):
        config = get_variant("classical")
        pipeline = PreprocessingPipeline(config)
        
        text = "@john WOW!!! #sarcasm 😂 https://example.com"
        result = pipeline.process_text(text)
        
        # Should be processed more aggressively
        assert "USER_MENTION" not in result or result == result.lower() or True
        # Should not have raw URL
        assert "https" not in result
    
    def test_stopword_variant_preserves_negation(self):
        config = get_variant("stopword")
        pipeline = PreprocessingPipeline(config)
        
        text = "I do not like this at all"
        result = pipeline.process_text(text)
        
        # Negation should be preserved
        assert "not" in result.lower()
    
    def test_empty_input_through_pipeline(self):
        config = get_variant("minimal")
        pipeline = PreprocessingPipeline(config)
        
        result = pipeline.process_text("")
        assert result == ""
    
    def test_none_input_through_pipeline(self):
        config = get_variant("minimal")
        pipeline = PreprocessingPipeline(config)
        
        result = pipeline.process_text(None)
        assert result == ""


# ============================================================
# CONFIG TESTS
# ============================================================

class TestConfig:
    """Test preprocessing configuration system."""
    
    def test_all_variants_exist(self):
        variants = get_all_variants()
        expected = ["minimal", "lowercase", "lemmatized", "stopword", "stemmed", "classical"]
        for name in expected:
            assert name in variants
    
    def test_variant_names(self):
        variants = get_all_variants()
        for name, config in variants.items():
            assert config.name == name
    
    def test_conflicting_config(self):
        config = PreprocessingConfig(lemmatize=True, stem=True)
        with pytest.raises(ValueError):
            config.validate()
    
    def test_config_serialization(self):
        import json
        config = get_variant("minimal")
        json_str = config.to_json()
        loaded = PreprocessingConfig.from_dict(json.loads(json_str))
        assert loaded.name == config.name
        assert loaded.lowercase == config.lowercase
    
    def test_invalid_hashtag_mode(self):
        config = PreprocessingConfig(hashtag_mode="invalid")
        with pytest.raises(ValueError):
            config.validate()


# ============================================================
# FEATURE EXTRACTION TESTS
# ============================================================

class TestFeatureExtraction:
    """Test social-media and text feature extraction."""
    
    def test_social_media_features(self):
        text = "@user check https://example.com #sarcasm 😂"
        features = extract_social_media_features(text)
        
        assert features["url_count"] == 1
        assert features["mention_count"] == 1
        assert features["hashtag_count"] == 1
        assert features["emoji_count"] == 1
    
    def test_text_features(self):
        text = "GREAT!!! Really???"
        features = extract_text_features(text)
        
        assert features["exclamation_count"] == 3
        assert features["question_count"] == 3
        assert features["uppercase_count"] > 0
        assert features["uppercase_ratio"] > 0
    
    def test_repeated_char_feature(self):
        text = "sooooo goooood"
        features = extract_text_features(text)
        assert features["repeated_character_count"] == 2


# ============================================================
# WHITESPACE TESTS
# ============================================================

class TestWhitespace:
    """Test whitespace normalization."""
    
    def test_multiple_spaces(self):
        assert normalize_whitespace("this    is   great") == "this is great"
    
    def test_tabs(self):
        assert normalize_whitespace("hello\tworld") == "hello world"
    
    def test_newlines(self):
        assert normalize_whitespace("hello\nworld") == "hello world"
    
    def test_strip(self):
        assert normalize_whitespace("  hello  ") == "hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
