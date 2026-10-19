"""
Phase 3 - Social Media Handlers
============================================
Sarcasm & Irony Detection System using NLP

Handles social-media-specific text elements:
    - URL detection and normalization
    - @mention detection and normalization
    - #hashtag handling (preserve / strip / segment)
    - Emoji handling (preserve / text / remove)
    - Social-media feature extraction

Key principle: Do NOT blindly remove social-media signals.
They may carry useful sarcastic/ironic information.
"""

import re
import logging
from typing import Dict

import emoji

logger = logging.getLogger(__name__)

# ============================================================
# REGEX PATTERNS
# ============================================================

# URL pattern: matches http(s), www., and common URL patterns
URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|http\S+",
    re.IGNORECASE
)

# Mention pattern: @username (alphanumeric + underscores)
MENTION_PATTERN = re.compile(r"@\w+")

# Hashtag pattern: #word
HASHTAG_PATTERN = re.compile(r"#(\w+)")

# CamelCase segmentation: split on lowercase-to-uppercase boundaries
CAMEL_CASE_PATTERN = re.compile(r"(?<=[a-z])(?=[A-Z])")


# ============================================================
# URL HANDLING
# ============================================================

def count_urls(text: str) -> int:
    """Count the number of URLs in text."""
    return len(URL_PATTERN.findall(text))


def has_url(text: str) -> bool:
    """Check if text contains any URL."""
    return bool(URL_PATTERN.search(text))


def normalize_urls(text: str, placeholder: str = "URL") -> str:
    """
    Replace all URLs with a consistent placeholder token.
    
    Example:
        "check https://example.com/page" -> "check URL"
    
    The presence of a URL is preserved as a structural signal,
    while the specific URL content (usually not useful for
    sarcasm classification) is normalized.
    """
    return URL_PATTERN.sub(placeholder, text)


# ============================================================
# MENTION HANDLING
# ============================================================

def count_mentions(text: str) -> int:
    """Count the number of @mentions in text."""
    return len(MENTION_PATTERN.findall(text))


def has_mention(text: str) -> bool:
    """Check if text contains any @mention."""
    return bool(MENTION_PATTERN.search(text))


def normalize_mentions(text: str, placeholder: str = "USER_MENTION") -> str:
    """
    Replace all @mentions with a consistent placeholder.
    
    Example:
        "@john said hello @jane" -> "USER_MENTION said hello USER_MENTION"
    
    Individual usernames are not useful for generalization.
    """
    return MENTION_PATTERN.sub(placeholder, text)


# ============================================================
# HASHTAG HANDLING
# ============================================================

def count_hashtags(text: str) -> int:
    """Count the number of hashtags in text."""
    return len(HASHTAG_PATTERN.findall(text))


def has_hashtag(text: str) -> bool:
    """Check if text contains any hashtag."""
    return bool(HASHTAG_PATTERN.search(text))


def _segment_camel_case(word: str) -> str:
    """
    Attempt CamelCase segmentation of a hashtag word.
    
    Example:
        "WhatAGreatIdea" -> "What A Great Idea"
    
    Only applied if the word contains internal uppercase letters,
    suggesting CamelCase. Simple words are returned as-is.
    """
    # Only attempt segmentation if there are internal uppercase letters
    if any(c.isupper() for c in word[1:]):
        segmented = CAMEL_CASE_PATTERN.sub(" ", word)
        return segmented
    return word


def handle_hashtags(text: str, mode: str = "preserve") -> str:
    """
    Handle hashtags according to the specified mode.
    
    Modes:
        "preserve": Keep hashtags as-is (e.g., #sarcasm -> #sarcasm)
        "strip":    Remove # prefix, keep word (e.g., #sarcasm -> sarcasm)
        "segment":  Strip # and attempt CamelCase segmentation
                    (e.g., #WhatAGreatIdea -> What A Great Idea)
    
    Returns:
        str: Text with hashtags handled according to mode.
    """
    if mode == "preserve":
        return text
    
    if mode == "strip":
        return HASHTAG_PATTERN.sub(r"\1", text)
    
    if mode == "segment":
        def _replace_hashtag(match):
            word = match.group(1)
            return _segment_camel_case(word)
        return HASHTAG_PATTERN.sub(_replace_hashtag, text)
    
    raise ValueError(f"Invalid hashtag mode: {mode}")


# ============================================================
# EMOJI HANDLING
# ============================================================

def count_emojis(text: str) -> int:
    """Count the number of emojis in text."""
    return emoji.emoji_count(text)


def has_emoji(text: str) -> bool:
    """Check if text contains any emoji."""
    return emoji.emoji_count(text) > 0


def handle_emojis(text: str, mode: str = "preserve") -> str:
    """
    Handle emojis according to the specified mode.
    
    Modes:
        "preserve": Keep emojis as-is (default for sarcasm detection)
        "text":     Convert emojis to textual descriptions
                    (e.g., 😂 -> :face_with_tears_of_joy:)
        "remove":   Remove all emojis
    
    Sarcasm frequently uses emojis like 😂 🙃 😒 😐.
    Do not remove emojis blindly.
    """
    if mode == "preserve":
        return text
    
    if mode == "text":
        return emoji.demojize(text, delimiters=(" ", " "))
    
    if mode == "remove":
        return emoji.replace_emoji(text, replace="")
    
    raise ValueError(f"Invalid emoji mode: {mode}")


# ============================================================
# SOCIAL MEDIA FEATURE EXTRACTION
# ============================================================

def extract_social_media_features(text: str) -> Dict[str, object]:
    """
    Extract social-media features from text before normalization.
    
    These features will be used in Phase 5 feature engineering.
    Must be called BEFORE text normalization to capture original signals.
    
    Returns:
        dict with keys:
            url_count, has_url,
            mention_count, has_mention,
            hashtag_count, has_hashtag,
            emoji_count, has_emoji
    """
    return {
        "url_count": count_urls(text),
        "has_url": has_url(text),
        "mention_count": count_mentions(text),
        "has_mention": has_mention(text),
        "hashtag_count": count_hashtags(text),
        "has_hashtag": has_hashtag(text),
        "emoji_count": count_emojis(text),
        "has_emoji": has_emoji(text),
    }
