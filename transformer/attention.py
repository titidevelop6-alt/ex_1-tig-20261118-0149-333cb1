"""Scaled dot-product attention and multi-head attention (Vaswani et al., 2017, sec. 3.2)."""

from __future__ import annotations

import math

import torch
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``softmax(Q K^T / sqrt(d_k)) V``.

    Args:
        query: ``(..., query_len, d_k)``
        key: ``(..., key_len, d_k)``
        value: ``(..., key_len, d_v)``
        mask: Optional boolean mask broadcastable to ``(..., query_len, key_len)``;
            ``True`` = allowed, ``False`` = blocked.
        dropout: Optional dropout module applied to the attention weights.

    Returns:
        ``(output, attention_weights)`` with shapes ``(..., query_len, d_v)`` and
        ``(..., query_len, key_len)``.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # Use the dtype's minimum rather than -inf so fully-masked rows
        # produce a uniform distribution instead of NaNs.
        scores = scores.masked_fill(~mask.to(torch.bool), torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, value), weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention: ``Concat(head_1..head_h) W^O``.

    Input and output tensors have shape ``(batch, seq_len, d_model)``.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights: torch.Tensor | None = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, query_len, _ = query.shape
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        out, weights = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
        self.attention_weights = weights.detach()

        out = out.transpose(1, 2).contiguous().view(batch, query_len, self.d_model)
        return self.w_o(out)
