"""
Phase 7 — FastText + Bi-LSTM and FastText + Bi-LSTM + Attention Architectures
============================================================================
Sarcasm & Irony Detection System using NLP

Implements:
1. FastTextBiLSTMClassifier (FT-01):
   Standard sequence model matching Phase 6 architecture for direct representation comparison.
   Input -> FastText Embedding -> Bi-LSTM (final state concat) -> Dropout -> Dense -> Dropout -> Logits (4)

2. FastTextAttentionClassifier (FT-02 / FT-03):
   Attention-augmented sequence model.
   Input -> FastText Embedding -> Bi-LSTM (return_sequences=True) -> Attention Layer -> Context Vector -> Dropout -> Dense -> Dropout -> Logits (4)
   Exposes attention weights for token-level interpretability.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.deep_learning.fasttext_attention.attention import Attention


class FastTextBiLSTMClassifier(nn.Module):
    """
    Standard Bidirectional LSTM text classifier with FastText embeddings (FT-01).
    Direct counterpart of Phase 6 GloVe + Bi-LSTM model for fair scientific comparison.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        lstm_units: int = 64,
        dense_units: int = 64,
        num_classes: int = 4,
        dropout: float = 0.3,
        embedding_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None,
        embedding_trainable: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.embedding_trainable = embedding_trainable
        self.has_attention = False

        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            if isinstance(embedding_matrix, np.ndarray):
                tensor_mat = torch.from_numpy(embedding_matrix).float()
            else:
                tensor_mat = embedding_matrix.float()
            self.embedding.weight.data.copy_(tensor_mat)
        self.embedding.weight.requires_grad = embedding_trainable

        # 2. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Dense classification head
        self.dropout1 = nn.Dropout(dropout)
        self.dense = nn.Linear(lstm_units * 2, dense_units)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(dense_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized logits of shape (batch_size, num_classes).
        """
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, (hn, cn) = self.lstm(emb)

        # hn: (2, batch_size, lstm_units) -> concat forward and backward final hidden states
        feat = torch.cat([hn[0], hn[1]], dim=1)  # (batch_size, lstm_units * 2)

        feat = self.dropout1(feat)
        hidden = self.relu(self.dense(feat))
        hidden = self.dropout2(hidden)
        logits = self.output(hidden)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes softmax prediction probabilities for input sequences.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs

    def count_params(self) -> Tuple[int, int]:
        """
        Returns (total_params, trainable_params).
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_architecture_summary(self) -> str:
        total, trainable = self.count_params()
        return (
            f"FastText + Bi-LSTM Classifier (FT-01):\n"
            f"  - Vocab size: {self.vocab_size}\n"
            f"  - Embedding: {self.embedding_dim}d (trainable={self.embedding_trainable})\n"
            f"  - Bidirectional LSTM: {self.lstm_units} units × 2 = {self.lstm_units * 2}\n"
            f"  - Attention: None (final hidden state concat)\n"
            f"  - Dense layer: {self.dense_units} units (ReLU)\n"
            f"  - Dropout rate: {self.dropout_rate}\n"
            f"  - Output layer: {self.num_classes} classes (logits)\n"
            f"  - Total params: {total:,}\n"
            f"  - Trainable params: {trainable:,}"
        )


class FastTextAttentionClassifier(nn.Module):
    """
    FastText + Bi-LSTM with custom Attention mechanism (FT-02 / FT-03).
    Returns full sequences from Bi-LSTM and computes a weighted context vector via Attention.
    Exposes attention weights for token-level interpretability.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        lstm_units: int = 64,
        attention_dim: int = 64,
        dense_units: int = 64,
        num_classes: int = 4,
        dropout: float = 0.3,
        embedding_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None,
        embedding_trainable: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.attention_dim = attention_dim
        self.dense_units = dense_units
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.embedding_trainable = embedding_trainable
        self.has_attention = True

        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            if isinstance(embedding_matrix, np.ndarray):
                tensor_mat = torch.from_numpy(embedding_matrix).float()
            else:
                tensor_mat = embedding_matrix.float()
            self.embedding.weight.data.copy_(tensor_mat)
        self.embedding.weight.requires_grad = embedding_trainable

        # 2. Bidirectional LSTM returning full sequences (return_sequences=True)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Custom Attention Layer
        self.attention = Attention(
            hidden_dim=lstm_units * 2,
            attention_dim=attention_dim,
        )

        # 4. Dense classification head
        self.dropout1 = nn.Dropout(dropout)
        self.dense = nn.Linear(lstm_units * 2, dense_units)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(dense_units, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        If return_attention is True, returns (logits, attention_weights).
        Otherwise returns logits.
        """
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(emb)  # (batch_size, seq_len, lstm_units * 2)

        # Padding mask: positions where x != 0 (<PAD>)
        mask = x != 0

        # Compute context vector and attention weights
        context, attn_weights = self.attention(
            lstm_out,
            mask=mask,
            return_weights=True,
        )  # context: (batch_size, lstm_units * 2), weights: (batch_size, seq_len)

        feat = self.dropout1(context)
        hidden = self.relu(self.dense(feat))
        hidden = self.dropout2(hidden)
        logits = self.output(hidden)

        if return_attention:
            return logits, attn_weights
        return logits

    def predict_proba(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes softmax prediction probabilities and optionally token attention weights.
        """
        self.eval()
        with torch.no_grad():
            if return_attention:
                logits, attn_weights = self.forward(x, return_attention=True)
                probs = torch.softmax(logits, dim=1)
                return probs, attn_weights
            else:
                logits = self.forward(x, return_attention=False)
                probs = torch.softmax(logits, dim=1)
                return probs

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts normalized attention weights of shape (batch_size, seq_len).
        """
        self.eval()
        with torch.no_grad():
            _, attn_weights = self.forward(x, return_attention=True)
        return attn_weights

    def count_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_architecture_summary(self) -> str:
        total, trainable = self.count_params()
        return (
            f"FastText + Bi-LSTM + Attention Classifier (FT-02/FT-03):\n"
            f"  - Vocab size: {self.vocab_size}\n"
            f"  - Embedding: {self.embedding_dim}d (trainable={self.embedding_trainable})\n"
            f"  - Bidirectional LSTM: {self.lstm_units} units × 2 = {self.lstm_units * 2} (return_sequences=True)\n"
            f"  - Attention: Additive/Bahdanau (latent dim={self.attention_dim})\n"
            f"  - Dense layer: {self.dense_units} units (ReLU)\n"
            f"  - Dropout rate: {self.dropout_rate}\n"
            f"  - Output layer: {self.num_classes} classes (logits)\n"
            f"  - Total params: {total:,}\n"
            f"  - Trainable params: {trainable:,}"
        )
