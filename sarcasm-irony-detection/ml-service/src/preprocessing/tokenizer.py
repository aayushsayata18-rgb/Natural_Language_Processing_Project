"""
Phase 3 - Tokenization
============================================
Sarcasm & Irony Detection System using NLP

Tokenization appropriate for social-media text using
NLTK's TweetTokenizer.

TweetTokenizer is designed to handle:
    - words
    - punctuation
    - hashtags
    - @mentions
    - URLs
    - contractions
    - emojis (where supported)

Do NOT use a simplistic split(" ") as the final NLP tokenizer.
"""

import logging
from typing import List

from nltk.tokenize import TweetTokenizer

logger = logging.getLogger(__name__)

# Shared TweetTokenizer instance
# strip_handles=False: we handle mentions separately in social_media.py
# reduce_len=False: we handle repeated chars separately in normalization.py
# preserve_case=True: case handling is done in normalization.py
_tokenizer = TweetTokenizer(
    strip_handles=False,
    reduce_len=False,
    preserve_case=True,
)


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text using NLTK's TweetTokenizer.
    
    Designed for social-media text. Properly handles:
        - Hashtags (#sarcasm -> single token)
        - Mentions (@user -> single token)
        - URLs (if present)
        - Contractions (don't -> "do", "n't")
        - Emojis (as individual tokens where supported)
        - Punctuation (as separate tokens)
    
    Example:
        Input:  "Great!!! Another exam tomorrow 😒"
        Output: ["Great", "!", "!", "!", "Another", "exam", "tomorrow", "😒"]
    
    Returns:
        List[str]: List of tokens.
    """
    if not text or not text.strip():
        return []
    
    try:
        return _tokenizer.tokenize(text)
    except Exception as e:
        logger.warning(f"Tokenization failed for text: {text[:100]}... Error: {e}")
        # Fallback to basic split (should rarely happen)
        return text.split()


def tokens_to_string(tokens: List[str]) -> str:
    """
    Rejoin tokens into a string suitable for downstream vectorization.
    
    Uses space-separated joining for compatibility with TF-IDF,
    BoW, and other text vectorizers.
    
    Args:
        tokens: List of string tokens.
    
    Returns:
        str: Space-joined token string.
    """
    return " ".join(tokens)
