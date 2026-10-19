"""
Phase 8 — BERT Tokenizer Manager
================================
Sarcasm & Irony Detection System using NLP

This module encapsulates the Hugging Face BERT WordPiece tokenizer:
- Pretrained vocabulary (bert-base-uncased, vocab_size=30,522)
- Subword tokenization mechanism
- Automatic handling of special tokens ([CLS], [SEP], [PAD], [UNK])
- Sequence length analysis, padding, and truncation
- Input IDs and Attention Mask creation
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BertTokenizerFast

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_MAX_LENGTH = 80


class BertTokenizerManager:
    """
    Manages loading, encoding, sequence-length analysis, and serialization
    of the pretrained BERT tokenizer.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
        )
        logger.info(
            f"Initialized BERT tokenizer from '{model_name_or_path}' "
            f"(vocab_size={self.vocab_size}, default max_length={self.max_length})"
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def cls_token_id(self) -> int:
        return self.tokenizer.cls_token_id

    @property
    def sep_token_id(self) -> int:
        return self.tokenizer.sep_token_id

    @property
    def unk_token_id(self) -> int:
        return self.tokenizer.unk_token_id

    @property
    def special_tokens_map(self) -> Dict[str, str]:
        return self.tokenizer.special_tokens_map

    def tokenize_text(self, text: str) -> List[str]:
        """
        Splits raw text into WordPiece subword tokens without special tokens.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        return self.tokenizer.tokenize(text)

    def inspect_subword_tokens(self, text: str) -> List[Tuple[str, int]]:
        """
        Returns pairs of (token, token_id) including special tokens [CLS] and [SEP].
        Useful for inspecting WordPiece subword decomposition in social media text.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        input_ids = encoding["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        return list(zip(tokens, input_ids))

    def encode_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes a single string into BERT input tensors:
        - input_ids: Tensor of token indices
        - attention_mask: Tensor of 1s (real token) and 0s (padding)
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        length = max_length or self.max_length
        encoded = self.tokenizer(
            text,
            max_length=length,
            padding=padding,
            truncation=truncation,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors=return_tensors,
        )
        return encoded

    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes a list of strings into batched BERT input tensors.
        """
        cleaned_texts = [str(t) if t is not None else "" for t in texts]
        length = max_length or self.max_length
        return self.tokenizer(
            cleaned_texts,
            max_length=length,
            padding=padding,
            truncation=truncation,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors=return_tensors,
        )

    def analyze_sequence_lengths(
        self,
        texts: List[str],
        candidate_lengths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes the tokenized sequence lengths on training texts (including [CLS] and [SEP]).
        Computes mean, median, standard deviation, percentiles (75, 90, 95, 98, 99),
        maximum length, and truncation percentage for candidate sequence lengths.
        """
        if candidate_lengths is None:
            candidate_lengths = [32, 48, 64, 80, 96, 128, 160]

        lengths: List[int] = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            # Count tokens including [CLS] and [SEP]
            ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
            lengths.append(len(ids))

        arr = np.array(lengths)
        total_samples = len(arr)

        stats: Dict[str, Any] = {
            "total_samples": int(total_samples),
            "mean_length": float(np.mean(arr)),
            "median_length": float(np.median(arr)),
            "std_length": float(np.std(arr)),
            "min_length": int(np.min(arr)),
            "max_length": int(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p98": float(np.percentile(arr, 98)),
            "p99": float(np.percentile(arr, 99)),
            "truncation_rates": {},
        }

        for cand in candidate_lengths:
            truncated = int(np.sum(arr > cand))
            rate = float((truncated / total_samples) * 100.0) if total_samples > 0 else 0.0
            stats["truncation_rates"][str(cand)] = {
                "max_len": cand,
                "samples_truncated": truncated,
                "percentage_truncated": round(rate, 3),
                "percentage_preserved": round(100.0 - rate, 3),
            }

        return stats

    def save_tokenizer(self, save_directory: Union[str, Path]) -> str:
        """
        Saves the tokenizer vocabulary, merges, and config to disk for reproducibility.
        """
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(path))
        logger.info(f"Saved BERT tokenizer to {path}")
        return str(path)

    @classmethod
    def load_saved_tokenizer(
        cls,
        load_directory: Union[str, Path],
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> "BertTokenizerManager":
        """
        Loads a saved tokenizer from a directory.
        """
        manager = cls.__new__(cls)
        manager.model_name_or_path = str(load_directory)
        manager.max_length = max_length
        manager.tokenizer = AutoTokenizer.from_pretrained(
            str(load_directory),
            use_fast=True,
        )
        return manager
