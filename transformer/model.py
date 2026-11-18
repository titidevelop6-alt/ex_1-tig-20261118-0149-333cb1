"""Top-level encoder-decoder Transformer (Vaswani et al., 2017)."""

from __future__ import annotations

import math

import torch
from torch import nn

from .layers import Decoder, Encoder, PositionalEncoding
from .masks import make_decoder_self_attention_mask, make_padding_mask


class Transformer(nn.Module):
    """Encoder-decoder Transformer that maps source/target token ids to target-vocabulary logits.

    Defaults follow the paper's base model (d_model=512, 8 heads, 6+6 layers,
    d_ff=2048, dropout=0.1). Vocabulary sizes are required arguments.

    Args:
        src_vocab_size: Size of the source vocabulary.
        tgt_vocab_size: Size of the target vocabulary.
        d_model: Model (embedding) dimension.
        num_heads: Number of attention heads; must divide ``d_model``.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        d_ff: Inner dimension of the position-wise feed-forward network.
        dropout: Dropout probability.
        max_len: Maximum sequence length supported by the positional encoding.
        src_pad_idx: Padding id in source sequences (``None`` disables source padding masks).
        tgt_pad_idx: Padding id in target sequences (``None`` disables target padding masks).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        src_pad_idx: int | None = None,
        tgt_pad_idx: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Keep padding embeddings at zero after re-initialisation.
        with torch.no_grad():
            for emb in (self.src_embedding, self.tgt_embedding):
                if emb.padding_idx is not None:
                    emb.weight[emb.padding_idx].zero_()

    # ------------------------------------------------------------------ masks
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor | None:
        """Padding mask of shape ``(batch, 1, 1, src_len)``, or ``None`` if no pad id is set."""
        if self.src_pad_idx is None:
            return None
        return make_padding_mask(src, self.src_pad_idx)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Causal mask combined with the target padding mask (if any)."""
        return make_decoder_self_attention_mask(tgt, self.tgt_pad_idx)

    # ---------------------------------------------------------------- forward
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode ``(batch, src_len)`` token ids into ``(batch, src_len, d_model)`` memory."""
        x = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode ``(batch, tgt_len)`` token ids into ``(batch, tgt_len, d_model)`` hidden states."""
        x = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        return self.decoder(x, memory, tgt_mask, memory_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits of shape ``(batch, tgt_len, tgt_vocab_size)``.

        If masks are not given, they are built automatically: a padding mask
        for the source (when ``src_pad_idx`` is set) and a causal (+ padding)
        mask for the target.
        """
        if src.dim() != 2 or tgt.dim() != 2:
            raise ValueError("src and tgt must have shape (batch, seq_len)")
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        memory = self.encode(src, src_mask)
        hidden = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.generator(hidden)
