"""
Phase 8 — BERT Sequence Classification Model
============================================
Sarcasm & Irony Detection System using NLP

Wraps Hugging Face BertForSequenceClassification with 4 output logits:
0: Regular
1: Sarcasm
2: Irony
3: Figurative

Features:
- Pretrained encoder (bert-base-uncased)
- 4-class classification head (dropout + linear projection to 4 logits)
- CrossEntropyLoss calculation on unnormalized logits
- Softmax probability conversion
- Parameter count inspection (total, trainable, non-trainable)
- Optional attention extraction for academic analysis
- save_pretrained / from_pretrained serialization
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "bert-base-uncased"
NUM_CLASSES = 4


class BertSarcasmClassifier(nn.Module):
    """
    BERT model for four-class sarcasm and irony classification.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        num_labels: int = NUM_CLASSES,
        dropout_rate: Optional[float] = None,
        output_attentions: bool = False,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.output_attentions = output_attentions

        # Configure architecture
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            output_attentions=output_attentions,
        )
        if dropout_rate is not None:
            config.hidden_dropout_prob = dropout_rate
            config.attention_probs_dropout_prob = dropout_rate

        self.config = config
        self.bert: BertForSequenceClassification = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                config=config,
            )
        )
        logger.info(
            f"Initialized BertSarcasmClassifier from '{model_name_or_path}' "
            f"with num_labels={num_labels}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        """
        Forward pass producing raw logits (and loss if labels are provided).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=True,
        )
        return outputs

    def predict_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Converts raw 4-class logits into probability distribution via Softmax.
        """
        return torch.softmax(logits, dim=-1)

    def count_parameters(self) -> Dict[str, int]:
        """
        Calculates total, trainable, and frozen parameters.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "non_trainable_parameters": int(non_trainable),
        }

    def save_pretrained(self, save_directory: Union[str, Path]) -> str:
        """
        Saves the underlying transformer model and config using Hugging Face format.
        """
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.bert.save_pretrained(str(path))
        logger.info(f"Saved BERT sequence classification model to {path}")
        return str(path)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: Union[str, Path],
        num_labels: int = NUM_CLASSES,
        output_attentions: bool = False,
    ) -> "BertSarcasmClassifier":
        """
        Loads a fine-tuned model checkpoint from disk.
        """
        instance = cls.__new__(cls)
        super(BertSarcasmClassifier, instance).__init__()
        instance.model_name_or_path = str(load_directory)
        instance.num_labels = num_labels
        instance.output_attentions = output_attentions

        config = AutoConfig.from_pretrained(
            str(load_directory),
            num_labels=num_labels,
            output_attentions=output_attentions,
        )
        instance.config = config
        instance.bert = AutoModelForSequenceClassification.from_pretrained(
            str(load_directory),
            config=config,
        )
        return instance
