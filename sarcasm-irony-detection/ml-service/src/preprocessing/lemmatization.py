"""
Phase 3 - Lemmatization
============================================
Sarcasm & Irony Detection System using NLP

Optional lemmatization using NLTK's WordNetLemmatizer
with POS tagging for proper lemma resolution.

Lemmatization maps words to their dictionary form:
    running, runs, ran -> run
    better -> good (with POS)

Do NOT make lemmatization mandatory for all model pipelines.
The effect will be evaluated in later phases.
"""

import logging
from typing import List

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

logger = logging.getLogger(__name__)

# Shared lemmatizer instance
_lemmatizer = WordNetLemmatizer()


def _get_wordnet_pos(treebank_tag: str) -> str:
    """
    Map Penn Treebank POS tag to WordNet POS tag for better lemmatization.
    
    The WordNetLemmatizer produces much better results when given
    the correct POS tag (e.g., 'better' as adjective -> 'good').
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        # Default to noun if unknown POS
        return wordnet.NOUN


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lemmatize a list of tokens using WordNet lemmatizer with POS tagging.
    
    Uses NLTK POS tagger to determine the correct part of speech
    before lemmatization, which produces more accurate results.
    
    Example:
        ["running", "quickly", "towards", "better", "days"]
        -> ["run", "quickly", "towards", "good", "day"]
    
    Args:
        tokens: List of string tokens.
    
    Returns:
        List[str]: Lemmatized tokens.
    """
    if not tokens:
        return []
    
    try:
        # POS tag the tokens
        tagged = nltk.pos_tag(tokens)
        
        # Lemmatize with correct POS
        lemmatized = []
        for word, tag in tagged:
            wordnet_pos = _get_wordnet_pos(tag)
            lemma = _lemmatizer.lemmatize(word, pos=wordnet_pos)
            lemmatized.append(lemma)
        
        return lemmatized
    
    except Exception as e:
        logger.warning(f"Lemmatization failed: {e}. Returning original tokens.")
        return tokens


def lemmatize_text(text: str) -> str:
    """
    Convenience function: lemmatize a text string.
    
    Tokenizes -> lemmatizes -> re-joins.
    For use when tokens are not already available.
    """
    from .tokenizer import tokenize_text, tokens_to_string
    tokens = tokenize_text(text)
    lemmatized = lemmatize_tokens(tokens)
    return tokens_to_string(lemmatized)
