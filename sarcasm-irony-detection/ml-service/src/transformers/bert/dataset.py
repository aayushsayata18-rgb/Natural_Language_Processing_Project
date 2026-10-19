"""
Phase 8 — BERT Dataset & DataLoader Module
==========================================
Sarcasm & Irony Detection System using NLP

Handles:
- PyTorch Dataset for BERT inputs (input_ids, attention_mask, token_type_ids, label)
- Stratified 80/20 train/validation split with random_state=42
- Strict data leakage prevention (test set isolated from validation/tuning)
- Dynamic and static padding DataLoader builders
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

from src.transformers.bert.tokenizer import BertTokenizerManager

logger = logging.getLogger(__name__)

DEFAULT_LABEL_MAPPING = {
    "Regular": 0,
    "Sarcasm": 1,
    "Irony": 2,
    "Figurative": 3,
}


def load_label_mapping(mapping_path: Optional[Union[str, Path]] = None) -> Dict[str, int]:
    """
    Loads approved project label mapping from JSON or returns standard default.
    """
    if mapping_path and Path(mapping_path).exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # If keys are label names, return directly; if keys are string ints, invert
            if all(k.isdigit() for k in data.keys()):
                return {v: int(k) for k, v in data.items()}
            return {k: int(v) for k, v in data.items()}
    return DEFAULT_LABEL_MAPPING.copy()


class SarcasmBertDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenized inputs for BERT sequence classification.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer_manager: Optional[BertTokenizerManager] = None,
        max_length: int = 80,
    ) -> None:
        self.texts = [str(t) if t is not None else "" for t in texts]
        self.labels = labels
        self.tokenizer_manager = tokenizer_manager or BertTokenizerManager(max_length=max_length)
        self.max_length = max_length

        if self.labels is not None and len(self.labels) != len(self.texts):
            raise ValueError(
                f"Number of texts ({len(self.texts)}) does not match labels ({len(self.labels)})"
            )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer_manager.encode_text(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if "token_type_ids" in encoding and encoding["token_type_ids"] is not None:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def load_and_split_data(
    train_csv_path: Union[str, Path],
    test_csv_path: Optional[Union[str, Path]] = None,
    val_size: float = 0.2,
    random_state: int = 42,
    label_col: str = "label_id",
    text_col: str = "text",
) -> Dict[str, Any]:
    """
    Loads train_clean.csv, performs stratified train/validation split (80/20),
    and optionally loads untouched test_clean.csv.
    """
    train_path = Path(train_csv_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    df_train_full = pd.read_csv(train_path)
    logger.info(f"Loaded full training dataset from {train_path}: {df_train_full.shape}")

    # Drop any null text rows if present
    df_train_full = df_train_full.dropna(subset=[text_col, label_col]).reset_index(drop=True)

    # Perform stratified split for train and validation
    train_df, val_df = train_test_split(
        df_train_full,
        test_size=val_size,
        stratify=df_train_full[label_col],
        random_state=random_state,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    logger.info(
        f"Stratified Split: Train={len(train_df)} ({100*(1-val_size):.0f}%), "
        f"Validation={len(val_df)} ({100*val_size:.0f}%)"
    )

    test_df = None
    if test_csv_path:
        test_path = Path(test_csv_path)
        if test_path.exists():
            test_df = pd.read_csv(test_path).dropna(subset=[text_col, label_col]).reset_index(drop=True)
            logger.info(f"Loaded test dataset from {test_path}: {test_df.shape} (strictly untouched for training)")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "full_train_df": df_train_full,
    }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer_manager: BertTokenizerManager,
    batch_size: int = 16,
    max_length: int = 80,
    num_workers: int = 0,
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and optional test set.
    """
    train_dataset = SarcasmBertDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label_id"].tolist(),
        tokenizer_manager=tokenizer_manager,
        max_length=max_length,
    )
    val_dataset = SarcasmBertDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label_id"].tolist(),
        tokenizer_manager=tokenizer_manager,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    loaders = {
        "train": train_loader,
        "val": val_loader,
    }

    if test_df is not None:
        test_dataset = SarcasmBertDataset(
            texts=test_df["text"].tolist(),
            labels=test_df["label_id"].tolist(),
            tokenizer_manager=tokenizer_manager,
            max_length=max_length,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        loaders["test"] = test_loader

    return loaders
