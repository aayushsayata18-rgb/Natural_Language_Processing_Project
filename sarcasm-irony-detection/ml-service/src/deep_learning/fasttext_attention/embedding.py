"""
Phase 7 — Embedding Matrix Construction
=======================================
Sarcasm & Irony Detection System using NLP

Constructs the continuous embedding matrix aligned with tokenizer token indices:
- Index 0 (<PAD>) is strictly initialized to all zeros.
- Index 1 (<UNK>) is initialized to uniform random or subword fallback.
- Known words are populated with FastText pre-trained vectors.
- OOV words utilize subword synthesis where morphological/character n-grams match,
  with documented uniform random fallback (±0.05).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.deep_learning.fasttext_attention.fasttext import (
    synthesize_subword_vector,
)


def build_embedding_matrix(
    word2idx: Dict[str, int],
    fasttext_vectors: Dict[str, np.ndarray],
    dim: int = 300,
    use_subwords: bool = True,
    oov_strategy: str = "subword_random",
    random_seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Union[int, float, str]], List[Dict[str, str]]]:
    """
    Constructs embedding matrix for the vocabulary.

    Parameters:
        word2idx: Token to integer mapping.
        fasttext_vectors: Pre-trained FastText dictionary.
        dim: Vector dimension.
        use_subwords: Whether to attempt subword synthesis for OOV words.
        oov_strategy: Strategy name ('subword_random', 'subword_zeros', 'random', 'zeros').
        random_seed: Seed for reproducible random initialization.

    Returns:
        matrix: np.ndarray of shape (vocab_size, dim), dtype float32.
        stats: Dictionary of coverage metrics.
        oov_records: List of OOV token classification records.
    """
    rng = np.random.default_rng(random_seed)
    vocab_size = len(word2idx)
    matrix = np.zeros((vocab_size, dim), dtype=np.float32)

    exact_matches = 0
    subword_matches = 0
    unresolved = 0
    oov_records = []

    for word, idx in word2idx.items():
        if idx == 0:  # <PAD>
            matrix[idx] = np.zeros(dim, dtype=np.float32)
            continue

        if word in fasttext_vectors:
            matrix[idx] = fasttext_vectors[word]
            exact_matches += 1
        elif use_subwords:
            syn_vec = synthesize_subword_vector(word, fasttext_vectors, dim=dim)
            if syn_vec is not None:
                matrix[idx] = syn_vec
                subword_matches += 1
                oov_records.append({"word": word, "resolution": "subword_synthesized"})
            else:
                unresolved += 1
                if "random" in oov_strategy:
                    matrix[idx] = rng.uniform(-0.05, 0.05, size=dim).astype(np.float32)
                else:
                    matrix[idx] = np.zeros(dim, dtype=np.float32)
                oov_records.append({"word": word, "resolution": "random_fallback"})
        else:
            unresolved += 1
            if "random" in oov_strategy:
                matrix[idx] = rng.uniform(-0.05, 0.05, size=dim).astype(np.float32)
            else:
                matrix[idx] = np.zeros(dim, dtype=np.float32)
            oov_records.append({"word": word, "resolution": "random_fallback"})

    effective_vocab = max(1, vocab_size - 2)
    exact_pct = (exact_matches / effective_vocab) * 100.0
    effective_coverage_pct = ((exact_matches + subword_matches) / effective_vocab) * 100.0

    stats = {
        "vocabulary_size": vocab_size,
        "effective_vocab": effective_vocab,
        "exact_matches": exact_matches,
        "exact_coverage_pct": round(exact_pct, 2),
        "subword_matches": subword_matches,
        "subword_coverage_pct": round((subword_matches / effective_vocab) * 100.0, 2),
        "total_covered": exact_matches + subword_matches,
        "effective_coverage_pct": round(effective_coverage_pct, 2),
        "unresolved_oov": unresolved,
        "oov_pct": round((unresolved / effective_vocab) * 100.0, 2),
        "oov_strategy": oov_strategy,
        "embedding_dim": dim,
    }

    return matrix, stats, oov_records
