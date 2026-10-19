"""
Custom word-level tokenizer for Bi-LSTM sequence modeling.
"""

import json
import re
from typing import Dict, List, Optional, Union
import numpy as np


class Tokenizer:
    """
    Word-level tokenizer with minimal neural preprocessing, special tokens,
    vocabulary control, sequence encoding, and padding/truncation.
    """

    PAD_TOKEN = "<PAD>"
    OOV_TOKEN = "<UNK>"
    PAD_IDX = 0
    OOV_IDX = 1

    def __init__(
        self,
        max_num_words: int = 30000,
        oov_token: str = "<UNK>",
        pad_token: str = "<PAD>",
    ):
        self.max_num_words = max_num_words
        self.PAD_TOKEN = pad_token
        self.OOV_TOKEN = oov_token
        self.PAD_IDX = 0
        self.OOV_IDX = 1

        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.OOV_TOKEN: self.OOV_IDX,
        }
        self.idx2word: Dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.OOV_IDX: self.OOV_TOKEN,
        }
        self.word_counts: Dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def _tokenize(self, text: Union[str, float]) -> List[str]:
        """
        Tokenizes text with minimal preprocessing:
        - URL normalization to <url>
        - Mention normalization to <user>
        - Lowercasing
        - Whitespace token splitting
        """
        if not isinstance(text, str) or not text.strip():
            return []

        # URL normalization
        t = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
        # Mention normalization
        t = re.sub(r"@\w+", " <user> ", t)
        # Lowercasing and whitespace splitting
        t = t.lower().strip()
        tokens = t.split()
        return tokens

    def fit(self, texts: Union[List[str], np.ndarray]) -> "Tokenizer":
        """
        Builds vocabulary strictly from the provided texts.
        """
        self.word_counts = {}
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1

        # Sort words by frequency (descending)
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)

        # Retain up to max_num_words
        kept_words = sorted_words[: self.max_num_words]

        # Reset mappings
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.OOV_TOKEN: self.OOV_IDX,
        }
        self.idx2word = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.OOV_IDX: self.OOV_TOKEN,
        }

        current_idx = 2
        for word, _ in kept_words:
            if word not in self.word2idx:
                self.word2idx[word] = current_idx
                self.idx2word[current_idx] = word
                current_idx += 1

        return self

    def texts_to_sequences(self, texts: Union[List[str], np.ndarray]) -> List[List[int]]:
        """
        Converts texts into lists of integer token IDs.
        """
        sequences = []
        for text in texts:
            tokens = self._tokenize(text)
            seq = [self.word2idx.get(tok, self.OOV_IDX) for tok in tokens]
            sequences.append(seq)
        return sequences

    def pad_sequences(
        self,
        sequences: List[List[int]],
        max_len: int,
        padding: str = "pre",
        truncating: str = "post",
    ) -> np.ndarray:
        """
        Pads and truncates sequences to a fixed length (max_len).
        """
        padded = np.full((len(sequences), max_len), self.PAD_IDX, dtype=np.int64)

        for i, seq in enumerate(sequences):
            if not seq:
                continue

            # Truncation
            if len(seq) > max_len:
                if truncating == "post":
                    s = seq[:max_len]
                elif truncating == "pre":
                    s = seq[-max_len:]
                else:
                    s = seq[:max_len]
            else:
                s = seq

            # Padding
            if padding == "post":
                padded[i, : len(s)] = s
            elif padding == "pre":
                padded[i, -len(s) :] = s
            else:
                padded[i, -len(s) :] = s

        return padded

    def get_sequence_lengths(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Returns array of token lengths for each text.
        """
        return np.array([len(self._tokenize(t)) for t in texts], dtype=np.int64)

    def save(self, filepath: str) -> None:
        """
        Serializes tokenizer to JSON.
        """
        data = {
            "max_num_words": self.max_num_words,
            "word2idx": self.word2idx,
            "vocab_size": self.vocab_size,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "Tokenizer":
        """
        Loads tokenizer from JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(max_num_words=data.get("max_num_words", 30000))
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(v): k for k, v in tok.word2idx.items()}
        return tok
