"""Building blocks: feed-forward, positional encoding, residual sublayers, encoder/decoder stacks."""

from __future__ import annotations

import math

import torch
from torch import nn

from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """``FFN(x) = max(0, x W_1 + b_1) W_2 + b_2`` (sec. 3.3)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Return the ``(max_len, d_model)`` sinusoidal table from sec. 3.5.

    ``PE(pos, 2i) = sin(pos / 10000^(2i/d_model))``
    ``PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))``
    """
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(-math.log(10000.0) * two_i / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
    return pe


class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encodings to ``(batch, seq_len, d_model)`` inputs, then dropout."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", sinusoidal_positional_encoding(max_len, d_model), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len {self.pe.size(0)} of the positional encoding"
            )
        return self.dropout(x + self.pe[:seq_len].unsqueeze(0).to(x.dtype))


class ResidualLayerNorm(nn.Module):
    """Post-norm residual connection used in the paper: ``LayerNorm(x + Dropout(Sublayer(x)))``."""

    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer_out))


class EncoderLayer(nn.Module):
    """Self-attention followed by a feed-forward network, each wrapped in residual + LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.residual1 = ResidualLayerNorm(d_model, dropout)
        self.residual2 = ResidualLayerNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.residual1(x, self.self_attn(x, x, x, src_mask))
        return self.residual2(x, self.feed_forward(x))


class DecoderLayer(nn.Module):
    """Masked self-attention, encoder-decoder (cross) attention, then feed-forward."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.residual1 = ResidualLayerNorm(d_model, dropout)
        self.residual2 = ResidualLayerNorm(d_model, dropout)
        self.residual3 = ResidualLayerNorm(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.residual1(x, self.self_attn(x, x, x, tgt_mask))
        x = self.residual2(x, self.cross_attn(x, memory, memory, memory_mask))
        return self.residual3(x, self.feed_forward(x))


class Encoder(nn.Module):
    """Stack of ``num_layers`` identical encoder layers."""

    def __init__(
        self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    """Stack of ``num_layers`` identical decoder layers."""

    def __init__(
        self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x
