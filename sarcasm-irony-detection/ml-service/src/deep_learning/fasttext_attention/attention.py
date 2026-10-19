"""
Phase 7 — Attention Mechanism for Sequence Modeling
===================================================
Sarcasm & Irony Detection System using NLP

Custom additive/feed-forward sequence attention mechanism (Bahdanau / Yang style).
Calculates attention energy scores for each Bi-LSTM hidden state, normalizes via softmax,
computes the weighted context vector, and exposes token attention weights for explainability.

Mathematical Formulation:
-------------------------
Given Bi-LSTM hidden sequence H = [h_1, h_2, ..., h_T] where h_t in R^{2D}:
    1. Score projection:  u_t = tanh(W * h_t + b),   W in R^{D_a x 2D}, b in R^{D_a}
    2. Energy scalar:     e_t = v^T * u_t,           v in R^{D_a}
    3. Softmax weights:   alpha_t = exp(e_t) / sum_{j=1}^T exp(e_j)
    4. Context vector:    c = sum_{t=1}^T alpha_t * h_t,   c in R^{2D}

Properties:
- Input shape:  (batch_size, seq_len, 2 * lstm_units)
- Output shape: (batch_size, 2 * lstm_units)
- Weights shape: (batch_size, seq_len)
- Normalization: sum_{t=1}^T alpha_{b, t} = 1.0 (for every sequence b)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Additive / Feed-forward sequence attention layer.

    Parameters:
        hidden_dim: Input feature dimension (typically 2 * lstm_units for Bi-LSTM).
        attention_dim: Internal projection dimension for attention scoring (default: 64).
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        # Projection layer: W * h_t + b
        self.projection = nn.Linear(hidden_dim, attention_dim, bias=True)
        # Context scoring vector: v^T * tanh(...)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Computes the weighted context representation from sequence hidden states.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim).
            mask: Optional boolean or binary mask of shape (batch_size, seq_len).
                  True/1 indicates valid tokens, False/0 indicates padding tokens.
            return_weights: Whether to return the normalized attention weights.

        Returns:
            context: Context vector of shape (batch_size, hidden_dim).
            weights: If return_weights is True, tensor of shape (batch_size, seq_len)
                     satisfying sum(weights, dim=-1) ≈ 1.0; else None.
        """
        # 1. Project hidden states to attention latent space
        # u: (batch_size, seq_len, attention_dim)
        u = torch.tanh(self.projection(hidden_states))

        # 2. Compute scalar score for each token
        # scores: (batch_size, seq_len, 1) -> squeeze to (batch_size, seq_len)
        scores = self.context_vector(u).squeeze(-1)

        # 3. Optional masking for padding tokens
        if mask is not None:
            # Replace masked positions with a very large negative value
            scores = scores.masked_fill(~mask.bool(), -1e9)

        # 4. Softmax normalization across sequence dimension
        # weights: (batch_size, seq_len)
        weights = F.softmax(scores, dim=-1)

        # 5. Weighted sum: sum_t (alpha_t * h_t)
        # weights.unsqueeze(1): (batch_size, 1, seq_len)
        # hidden_states: (batch_size, seq_len, hidden_dim)
        # context: (batch_size, 1, hidden_dim) -> squeeze to (batch_size, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)

        if return_weights:
            return context, weights
        return context, None
