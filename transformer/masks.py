"""Attention mask helpers.

Convention used throughout this package: masks are boolean tensors where
``True`` means "this position may be attended to" and ``False`` means
"blocked". Masks are broadcast against attention scores of shape
``(batch, heads, query_len, key_len)``.
"""

from __future__ import annotations

import torch


def make_padding_mask(tokens: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Mask out padding tokens on the key side.

    Args:
        tokens: Token ids of shape ``(batch, seq_len)``.
        pad_idx: Id of the padding token.

    Returns:
        Boolean tensor of shape ``(batch, 1, 1, seq_len)``; ``False`` where the
        key token is padding.
    """
    if tokens.dim() != 2:
        raise ValueError(f"tokens must have shape (batch, seq_len), got {tuple(tokens.shape)}")
    return (tokens != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """Look-ahead mask: position ``i`` may only attend to positions ``j <= i``.

    Args:
        size: Sequence length.
        device: Optional device for the mask.

    Returns:
        Lower-triangular boolean tensor of shape ``(1, 1, size, size)``.
    """
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def make_decoder_self_attention_mask(tgt: torch.Tensor, pad_idx: int | None) -> torch.Tensor:
    """Combine the causal mask with the target padding mask.

    Returns a boolean tensor broadcastable to ``(batch, 1, tgt_len, tgt_len)``.
    """
    causal = make_causal_mask(tgt.size(1), device=tgt.device)
    if pad_idx is None:
        return causal
    return causal & make_padding_mask(tgt, pad_idx)
