"""
Phase 3 - Text Normalization
============================================
Sarcasm & Irony Detection System using NLP

Handles text normalization operations:
    - Whitespace normalization
    - Case normalization
    - Punctuation normalization
    - Repeated character handling
    - Negation handling
    - Stopword removal (with negation protection)
    - Punctuation/capitalization feature extraction

Key principle: Do NOT automatically remove signals that may
carry sarcastic/ironic meaning (punctuation, capitalization,
repeated characters).
"""

import re
import string
import logging
from typing import List, Dict, Set

import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

# Negation words that must be protected during stopword removal
NEGATION_WORDS: Set[str] = {
    "not", "no", "never", "nor", "neither",
    "nobody", "nothing", "nowhere", "none",
    "cannot", "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't",
    "don't",
    "n't",  # Tokenized contraction suffix
}

# Repeated character pattern: 3+ of the same character
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")

# Multiple whitespace pattern
WHITESPACE_PATTERN = re.compile(r"\s+")

# Repeated punctuation patterns for normalization
REPEATED_EXCLAMATION = re.compile(r"!{2,}")
REPEATED_QUESTION = re.compile(r"\?{2,}")
REPEATED_PERIOD = re.compile(r"\.{2,}")


# ============================================================
# WHITESPACE NORMALIZATION
# ============================================================

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace: collapse multiple spaces, tabs,
    unnecessary line breaks into single spaces.
    
    Example:
        "this    is   great" -> "this is great"
    
    Does not remove meaningful punctuation or social-media signals.
    """
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


# ============================================================
# CASE NORMALIZATION
# ============================================================

def normalize_case(text: str, lowercase: bool = False) -> str:
    """
    Optionally lowercase text.
    
    Capitalization can indicate emphasis (e.g., "WHAT A GREAT IDEA").
    Do not blindly lowercase without experimental justification.
    """
    if lowercase:
        return text.lower()
    return text


# ============================================================
# PUNCTUATION HANDLING
# ============================================================

def normalize_punctuation(text: str, mode: str = "preserve") -> str:
    """
    Handle punctuation according to the specified mode.
    
    Modes:
        "preserve":  Keep all punctuation as-is.
                     Important for patterns like !!!, ???, !?, ...
        "normalize": Collapse repeated punctuation to limited form.
                     e.g., "!!!" -> "!!", "???" -> "??"
                     Preserves the signal but reduces noise.
        "remove":    Remove all punctuation characters.
                     WARNING: May destroy sarcasm-relevant signals.
    """
    if mode == "preserve":
        return text
    
    if mode == "normalize":
        # Collapse repeated exclamations to double
        text = REPEATED_EXCLAMATION.sub("!!", text)
        # Collapse repeated questions to double
        text = REPEATED_QUESTION.sub("??", text)
        # Collapse repeated periods to ellipsis
        text = REPEATED_PERIOD.sub("...", text)
        return text
    
    if mode == "remove":
        return text.translate(str.maketrans("", "", string.punctuation))
    
    raise ValueError(f"Invalid punctuation mode: {mode}")


# ============================================================
# REPEATED CHARACTER HANDLING
# ============================================================

def normalize_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """
    Normalize repeated characters in words.
    
    Social-media text often contains elongated words:
        soooo, goooood, loooool, greatttt
    
    This function reduces character repetition to max_repeat.
    
    Example (max_repeat=2):
        "sooooo good" -> "soo good"
        "greatttt" -> "greatt"
    
    The original repeated-character information should be
    captured as features BEFORE this normalization.
    """
    def _replace_repeated(match):
        char = match.group(1)
        return char * max_repeat
    
    return REPEATED_CHAR_PATTERN.sub(_replace_repeated, text)


def count_repeated_chars(text: str) -> int:
    """
    Count instances of repeated characters (3+ of the same char).
    
    Must be called BEFORE repeated-character normalization.
    """
    return len(REPEATED_CHAR_PATTERN.findall(text))


def has_repeated_chars(text: str) -> bool:
    """Check if text contains repeated characters."""
    return bool(REPEATED_CHAR_PATTERN.search(text))


# ============================================================
# NEGATION HANDLING
# ============================================================

def contains_negation(tokens: List[str]) -> bool:
    """
    Check if token list contains any negation word.
    """
    return any(t.lower() in NEGATION_WORDS for t in tokens)


def is_negation_word(token: str) -> bool:
    """Check if a single token is a negation word."""
    return token.lower() in NEGATION_WORDS


# ============================================================
# STOPWORD REMOVAL
# ============================================================

def _get_stopwords(protect_negations: bool = True) -> Set[str]:
    """
    Get the stopword set, optionally with negation words removed.
    
    If protect_negations is True, negation words are excluded
    from the stopword set so they survive stopword removal.
    """
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        logger.warning("NLTK stopwords not found. Downloading...")
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
    
    if protect_negations:
        stop_words -= NEGATION_WORDS
        # Also protect common negation forms
        stop_words -= {"not", "no", "nor", "neither", "never",
                       "don", "don't", "didn", "didn't",
                       "doesn", "doesn't", "hadn", "hadn't",
                       "hasn", "hasn't", "haven", "haven't",
                       "isn", "isn't", "couldn", "couldn't",
                       "shouldn", "shouldn't", "wasn", "wasn't",
                       "weren", "weren't", "won", "won't",
                       "wouldn", "wouldn't", "aren", "aren't",
                       "mustn", "mustn't", "needn", "needn't",
                       "shan", "shan't"}
    
    return stop_words


def remove_stopwords_from_tokens(
    tokens: List[str],
    protect_negations: bool = True
) -> List[str]:
    """
    Remove stopwords from a token list.
    
    IMPORTANT: When protect_negations is True, negation words
    (not, never, no, n't, etc.) are preserved even though they
    appear in standard stopword lists.
    
    Example:
        Input:  ["I", "do", "not", "like", "this"]
        Output (protect_negations=True): ["not", "like"]
        Output (protect_negations=False): ["like"]
    """
    stop_words = _get_stopwords(protect_negations)
    return [t for t in tokens if t.lower() not in stop_words]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_text_features(text: str) -> Dict[str, object]:
    """
    Extract text-level features from text.
    
    These features capture punctuation patterns, capitalization,
    and repeated characters. Must be called BEFORE normalization
    to capture original signals.
    
    Returns:
        dict with keys:
            exclamation_count, question_count, punctuation_count,
            uppercase_count, uppercase_ratio, repeated_character_count
    """
    exclamation_count = text.count("!")
    question_count = text.count("?")
    punctuation_count = sum(1 for c in text if c in string.punctuation)
    
    alpha_chars = [c for c in text if c.isalpha()]
    uppercase_count = sum(1 for c in alpha_chars if c.isupper())
    uppercase_ratio = uppercase_count / len(alpha_chars) if alpha_chars else 0.0
    
    repeated_character_count = count_repeated_chars(text)
    
    return {
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "punctuation_count": punctuation_count,
        "uppercase_count": uppercase_count,
        "uppercase_ratio": round(uppercase_ratio, 4),
        "repeated_character_count": repeated_character_count,
    }
