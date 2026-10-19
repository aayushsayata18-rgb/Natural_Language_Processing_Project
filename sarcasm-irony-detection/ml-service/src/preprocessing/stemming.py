"""
Phase 3 - Stemming
============================================
Sarcasm & Irony Detection System using NLP

Optional stemming using NLTK's PorterStemmer.

Stemming reduces words to their root form:
    running -> run
    happily -> happili
    connection -> connect

WARNING: Stemming can damage informal social-media words
and may remove useful semantic information.

Stemming is an experiment, not a default.
Do NOT automatically use stemming in the final pipeline.
"""

import logging
from typing import List

from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

# Shared stemmer instance
_stemmer = PorterStemmer()


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Stem a list of tokens using Porter Stemmer.
    
    Example:
        ["running", "happily", "connections"]
        -> ["run", "happili", "connect"]
    
    Note: Stemming is more aggressive than lemmatization and
    may produce non-word outputs. This is expected behavior.
    
    Args:
        tokens: List of string tokens.
    
    Returns:
        List[str]: Stemmed tokens.
    """
    if not tokens:
        return []
    
    try:
        return [_stemmer.stem(token) for token in tokens]
    except Exception as e:
        logger.warning(f"Stemming failed: {e}. Returning original tokens.")
        return tokens


def stem_text(text: str) -> str:
    """
    Convenience function: stem a text string.
    
    Tokenizes -> stems -> re-joins.
    For use when tokens are not already available.
    """
    from .tokenizer import tokenize_text, tokens_to_string
    tokens = tokenize_text(text)
    stemmed = stem_tokens(tokens)
    return tokens_to_string(stemmed)
