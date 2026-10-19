"""
GloVe word embedding loading, coverage analysis, and embedding matrix generation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def load_glove_vectors(filepath: str, dim: int = 100) -> Dict[str, np.ndarray]:
    """
    Loads pre-trained GloVe embeddings from text file.
    """
    vectors = {}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = np.asarray(parts[1:], dtype=np.float32)
            if len(vector) == dim:
                vectors[word] = vector
    return vectors


def build_embedding_matrix(
    word2idx: Dict[str, int],
    glove_vectors: Dict[str, np.ndarray],
    dim: int = 100,
    oov_strategy: str = "random",
) -> Tuple[np.ndarray, Dict[str, Union[int, float, str]], List[str]]:
    """
    Constructs the embedding matrix aligned with tokenizer token indices.

    - Index 0 (<PAD>) is strictly initialized to all zeros.
    - Known words are initialized with GloVe vectors.
    - OOV words use the specified strategy ('random' uniform ±0.05 or 'zeros').
    """
    vocab_size = len(word2idx)
    matrix = np.zeros((vocab_size, dim), dtype=np.float32)
    oov_words = []
    matched = 0

    for word, idx in word2idx.items():
        if idx == 0:  # <PAD>
            matrix[idx] = np.zeros(dim, dtype=np.float32)
            continue

        if word in glove_vectors:
            matrix[idx] = glove_vectors[word]
            matched += 1
        else:
            oov_words.append(word)
            if oov_strategy == "random":
                matrix[idx] = np.random.uniform(-0.05, 0.05, size=dim).astype(np.float32)
            elif oov_strategy == "zeros":
                matrix[idx] = np.zeros(dim, dtype=np.float32)
            else:
                matrix[idx] = np.zeros(dim, dtype=np.float32)

    coverage_pct = (matched / max(1, vocab_size)) * 100.0

    stats = {
        "total_vocabulary": vocab_size,
        "glove_matched": matched,
        "oov_count": len(oov_words),
        "coverage_pct": coverage_pct,
        "oov_strategy": oov_strategy,
    }

    return matrix, stats, oov_words


def analyze_oov(oov_words: List[str]) -> pd.DataFrame:
    """
    Categorizes out-of-vocabulary words into descriptive classes.
    """
    categories = []
    for w in oov_words:
        if w.startswith("#"):
            categories.append("hashtag")
        elif w.startswith("@") or w == "<user>":
            categories.append("mention")
        elif w.startswith("http") or w == "<url>":
            categories.append("url")
        elif w.isupper() and len(w) > 1:
            categories.append("uppercase")
        elif any(c.isdigit() for c in w):
            categories.append("contains_numbers")
        elif len(w) <= 2:
            categories.append("short")
        else:
            categories.append("slang_or_misspelling")

    return pd.DataFrame({"word": oov_words, "category": categories})


def save_coverage_report(stats: Dict[str, Union[int, float, str]], filepath: str) -> None:
    """
    Saves coverage statistics to CSV or JSON.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    else:
        df = pd.DataFrame([stats])
        df.to_csv(path, index=False)
