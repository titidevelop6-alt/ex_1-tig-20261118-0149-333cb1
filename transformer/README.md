# transformer

A from-scratch PyTorch implementation of the encoder-decoder Transformer from
*Attention Is All You Need* (Vaswani et al., 2017).

Requirements: Python 3.10+ and PyTorch (`pip install torch`). No other dependencies.

## Modules

| Module | Contents |
| --- | --- |
| `attention.py` | `scaled_dot_product_attention`, `MultiHeadAttention` |
| `layers.py` | `PositionwiseFeedForward`, `sinusoidal_positional_encoding`, `PositionalEncoding`, `ResidualLayerNorm` (post-norm residual), `EncoderLayer`, `DecoderLayer`, `Encoder`, `Decoder` |
| `masks.py` | `make_padding_mask`, `make_causal_mask`, `make_decoder_self_attention_mask` |
| `model.py` | `Transformer` (embeddings × √d_model + positional encoding → encoder → decoder → linear to target vocab) |

Mask convention: boolean tensors where `True` = may attend, `False` = blocked,
broadcast against scores of shape `(batch, heads, query_len, key_len)`.

## Construct the model and run a forward pass

Run from the project root (the directory that contains `transformer/`):

```python
import torch
from transformer import Transformer

model = Transformer(
    src_vocab_size=100,
    tgt_vocab_size=100,
    d_model=32,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=64,
    dropout=0.1,
    src_pad_idx=0,   # optional: enables source padding mask
    tgt_pad_idx=0,   # optional: adds target padding to the causal mask
)

batch, src_len, tgt_len = 2, 7, 5
src = torch.randint(1, 100, (batch, src_len))
tgt = torch.randint(1, 100, (batch, tgt_len))

logits = model(src, tgt)
print(logits.shape)  # torch.Size([2, 5, 100]) == (batch, tgt_len, tgt_vocab_size)
```

`forward` builds masks automatically: a padding mask for the source (when
`src_pad_idx` is set) and a causal look-ahead mask (combined with target
padding when `tgt_pad_idx` is set) for decoder self-attention. You can also pass
`src_mask` / `tgt_mask` explicitly, or call `model.encode(...)` and
`model.decode(...)` separately.

Omitting the hyperparameters gives the paper's base model:
`d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1`.
`d_model` must be divisible by `num_heads`; otherwise a `ValueError` is raised.

## Tests

```bash
python -m unittest discover -s tests -v
```

Training loops, tokenization, datasets, and decoding (greedy/beam search) are
intentionally not included.
