"""
Phase 7 — FastText Vector Loading, Subword Analysis, and Coverage
=================================================================
Sarcasm & Irony Detection System using NLP

Handles:
- Loading pre-trained FastText vectors from `.vec` format
- FastText subword extraction (character n-grams 3 to 6)
- Subword-based vector synthesis for out-of-vocabulary (OOV) words
- Vocabulary coverage calculation against project vocabulary
- OOV categorization and representative error inspection
"""

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


FASTTEXT_METADATA = {
    "source": "Facebook AI Research (FAIR) FastText English Wikipedia",
    "filename": "wiki.simple.vec",
    "language": "English (en)",
    "dimension": 300,
    "vocabulary_entries": 111051,
    "download_url": "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec",
    "license": "Creative Commons Attribution-Share-Alike License 3.0 (CC BY-SA 3.0)",
}


def load_fasttext_vectors(
    filepath: str,
    dim: int = 300,
    max_vectors: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Loads pre-trained FastText vectors from a .vec file.
    The first line optionally contains '<vocab_size> <dim>'.
    Subsequent lines contain '<word> <v_1> <v_2> ... <v_dim>'.
    """
    vectors: Dict[str, np.ndarray] = {}
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"FastText vector file not found at: {filepath}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip().split()
        # If the first line is header: [vocab_size, dim]
        if len(first_line) == 2 and first_line[0].isdigit() and first_line[1].isdigit():
            expected_dim = int(first_line[1])
            if expected_dim != dim:
                dim = expected_dim
        else:
            # First line is a normal vector line
            if len(first_line) == dim + 1:
                w = first_line[0]
                vec = np.asarray(first_line[1:], dtype=np.float32)
                vectors[w] = vec

        count = len(vectors)
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if len(parts) == dim + 1:
                vec = np.asarray(parts[1:], dtype=np.float32)
                vectors[word] = vec
                count += 1
                if max_vectors and count >= max_vectors:
                    break

    return vectors


def extract_character_ngrams(word: str, min_n: int = 3, max_n: int = 6) -> List[str]:
    """
    Extracts subword character n-grams enclosed by '<' and '>' boundary markers,
    replicating the FastText subword representation strategy (Bojanowski et al., 2017).
    """
    if not word:
        return []
    extended = f"<{word}>"
    length = len(extended)
    ngrams = []
    for n in range(min_n, min(max_n, length) + 1):
        for i in range(length - n + 1):
            ngrams.append(extended[i : i + n])
    return ngrams


def synthesize_subword_vector(
    word: str,
    fasttext_vectors: Dict[str, np.ndarray],
    dim: int = 300,
    min_n: int = 3,
    max_n: int = 6,
) -> Optional[np.ndarray]:
    """
    Synthesizes an embedding vector for an OOV word by finding matching subword
    n-grams in the FastText vocabulary and averaging their vector representations.
    Returns None if no constituent n-grams or morphological roots are found.
    """
    # 1. Check if stripped or root forms match (e.g., removing #, trailing punctuation, repeated letters)
    cleaned = re.sub(r"^[#@]+", "", word.lower())
    if cleaned in fasttext_vectors:
        return fasttext_vectors[cleaned].copy()

    # Reduced repeated characters: e.g. "soooo" -> "so"
    dedup = re.sub(r"(.)\1{2,}", r"\1", cleaned)
    if dedup in fasttext_vectors:
        return fasttext_vectors[dedup].copy()

    # 2. Extract character n-grams and pool matching subword vectors
    ngrams = extract_character_ngrams(cleaned, min_n=min_n, max_n=max_n)
    matched_vecs = [fasttext_vectors[ng] for ng in ngrams if ng in fasttext_vectors]

    if matched_vecs:
        stacked = np.stack(matched_vecs, axis=0)
        mean_vec = np.mean(stacked, axis=0).astype(np.float32)
        # Normalize to unit variance / standard norm
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-7:
            mean_vec = mean_vec / norm
        return mean_vec

    return None


def calculate_fasttext_coverage(
    word2idx: Dict[str, int],
    fasttext_vectors: Dict[str, np.ndarray],
    dim: int = 300,
    use_subwords: bool = True,
) -> Tuple[Dict[str, Union[int, float, str]], List[Dict[str, str]]]:
    """
    Calculates detailed FastText vocabulary coverage, identifying exact matches,
    subword-resolved OOV words, and true unknowns.
    """
    total_vocab = len(word2idx)
    exact_matches = 0
    subword_matches = 0
    true_oov = 0
    oov_details = []

    for word, idx in word2idx.items():
        if idx == 0:  # <PAD>
            continue
        if idx == 1:  # <UNK>
            continue

        if word in fasttext_vectors:
            exact_matches += 1
        elif use_subwords:
            syn_vec = synthesize_subword_vector(word, fasttext_vectors, dim=dim)
            if syn_vec is not None:
                subword_matches += 1
                oov_details.append({"word": word, "status": "subword_resolved", "category": categorize_token(word)})
            else:
                true_oov += 1
                oov_details.append({"word": word, "status": "unresolved_oov", "category": categorize_token(word)})
        else:
            true_oov += 1
            oov_details.append({"word": word, "status": "oov", "category": categorize_token(word)})

    effective_vocab = max(1, total_vocab - 2)  # excluding <PAD> and <UNK>
    exact_coverage_pct = (exact_matches / effective_vocab) * 100.0
    effective_coverage_pct = ((exact_matches + subword_matches) / effective_vocab) * 100.0

    stats = {
        "vocabulary_size": total_vocab,
        "effective_vocab": effective_vocab,
        "exact_matches": exact_matches,
        "exact_coverage_pct": round(exact_coverage_pct, 2),
        "subword_matches": subword_matches,
        "subword_coverage_pct": round((subword_matches / effective_vocab) * 100.0, 2),
        "total_covered": exact_matches + subword_matches,
        "effective_coverage_pct": round(effective_coverage_pct, 2),
        "unresolved_oov": true_oov,
        "oov_pct": round((true_oov / effective_vocab) * 100.0, 2),
        "oov_strategy": "subword_synthesis_with_random_fallback" if use_subwords else "random_fallback",
        "embedding_dim": dim,
        "source": FASTTEXT_METADATA["source"],
    }

    return stats, oov_details


def categorize_token(w: str) -> str:
    """
    Categorizes tokens into linguistic categories for subword analysis.
    """
    if w.startswith("#"):
        return "hashtag"
    if w.startswith("@") or w == "<user>":
        return "mention"
    if w.startswith("http") or w == "<url>":
        return "url"
    if re.search(r"(.)\1{2,}", w):
        return "repeated_characters"
    if any(c.isdigit() for c in w):
        return "numeric_alphanumeric"
    if len(w) <= 2:
        return "short_token"
    if "_" in w or "-" in w:
        return "hyphenated_compound"
    return "slang_misspelling_or_rare"
