"""
BiLSTM neural network classifier in PyTorch for 4-class sarcasm and irony classification.
"""

from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM architecture for text sequence classification.

    Architecture:
        Input(batch_size, max_len)
            → Embedding(vocab_size, embedding_dim) [pretrained GloVe]
            → Bidirectional LSTM(lstm_units)
            → Dropout(dropout)
            → Dense(dense_units, ReLU)
            → Dropout(dropout)
            → Dense(num_classes) [logits]
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
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

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            if isinstance(embedding_matrix, np.ndarray):
                tensor_mat = torch.from_numpy(embedding_matrix).float()
            else:
                tensor_mat = embedding_matrix.float()
            self.embedding.weight.data.copy_(tensor_mat)
            self.embedding.weight.requires_grad = embedding_trainable

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
        )

        # Classification head
        self.dropout1 = nn.Dropout(dropout)
        self.dense = nn.Linear(lstm_units * 2, dense_units)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(dense_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized logits.
        """
        # x: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, (hn, cn) = self.lstm(emb)

        # hn: (2, batch_size, lstm_units) -> concat forward and backward final states
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

    def get_architecture_summary(self) -> str:
        """
        Returns a formatted string describing architecture and parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        summary = (
            f"BiLSTM Classifier Architecture:\n"
            f"  - Vocab size: {self.vocab_size}\n"
            f"  - Embedding: {self.embedding_dim}d (trainable={self.embedding_trainable})\n"
            f"  - Bidirectional LSTM: {self.lstm_units} units × 2 = {self.lstm_units * 2}\n"
            f"  - Dense layer: {self.dense_units} units (ReLU)\n"
            f"  - Dropout rate: {self.dropout_rate}\n"
            f"  - Output layer: {self.num_classes} classes (logits)\n"
            f"  - Total params: {total_params:,}\n"
            f"  - Trainable params: {trainable_params:,}"
        )
        return summary
