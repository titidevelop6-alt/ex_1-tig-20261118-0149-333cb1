"""Acceptance checks for the transformer package (stdlib unittest; run from the project root).

    python -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from transformer import (  # noqa: E402
    MultiHeadAttention,
    Transformer,
    make_causal_mask,
    make_padding_mask,
    sinusoidal_positional_encoding,
)


def small_model(**kwargs):
    params = dict(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=32,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.1,
    )
    params.update(kwargs)
    return Transformer(**params)


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_forward_shape(self):
        model = small_model(tgt_vocab_size=120)
        src = torch.randint(0, 100, (2, 7))
        tgt = torch.randint(0, 120, (2, 5))
        logits = model(src, tgt)
        self.assertEqual(tuple(logits.shape), (2, 5, 120))
        self.assertTrue(torch.isfinite(logits).all())

    def test_backward(self):
        model = small_model(src_pad_idx=0, tgt_pad_idx=0)
        src = torch.randint(1, 100, (2, 6))
        src[0, -2:] = 0
        tgt = torch.randint(1, 100, (2, 4))
        tgt[1, -1] = 0
        logits = model(src, tgt)
        logits.sum().backward()
        self.assertIsNotNone(model.encoder.layers[0].self_attn.w_q.weight.grad)

    def test_multi_head_attention_shape(self):
        mha = MultiHeadAttention(d_model=32, num_heads=4)
        x = torch.randn(3, 9, 32)
        self.assertEqual(mha(x, x, x).shape, x.shape)

    def test_multi_head_attention_invalid_heads(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            MultiHeadAttention(d_model=30, num_heads=4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            small_model(d_model=30)

    def test_causal_mask_lower_triangular(self):
        mask = make_causal_mask(5)[0, 0]
        for i in range(5):
            for j in range(5):
                self.assertEqual(bool(mask[i, j]), j <= i)

    def test_future_tokens_do_not_affect_past_logits(self):
        model = small_model()
        model.eval()
        src = torch.randint(0, 100, (1, 6))
        tgt = torch.randint(0, 100, (1, 5))
        tgt_changed = tgt.clone()
        tgt_changed[0, 3] = (tgt[0, 3] + 1) % 100
        with torch.no_grad():
            a = model(src, tgt)
            b = model(src, tgt_changed)
        self.assertTrue(torch.allclose(a[:, :3], b[:, :3], atol=1e-6))
        self.assertFalse(torch.allclose(a[:, 3:], b[:, 3:]))

    def test_padding_mask(self):
        tokens = torch.tensor([[5, 6, 0, 0], [1, 2, 3, 0]])
        mask = make_padding_mask(tokens, pad_idx=0)
        self.assertEqual(tuple(mask.shape), (2, 1, 1, 4))
        self.assertEqual(mask[0, 0, 0].tolist(), [True, True, False, False])
        self.assertEqual(mask[1, 0, 0].tolist(), [True, True, True, False])

    def test_source_padding_is_ignored(self):
        model = small_model(src_pad_idx=0)
        model.eval()
        src = torch.tensor([[4, 5, 6, 0, 0]])
        src_other_pad_len = torch.tensor([[4, 5, 6, 0, 0, 0, 0]])
        tgt = torch.randint(1, 100, (1, 3))
        with torch.no_grad():
            a = model(src, tgt)
            b = model(src_other_pad_len, tgt)
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_positional_encoding_formula(self):
        for d_model in (6, 7):
            pe = sinusoidal_positional_encoding(max_len=10, d_model=d_model)
            for pos in range(10):
                for dim in range(d_model):
                    i2 = dim - (dim % 2)
                    angle = pos / (10000 ** (i2 / d_model))
                    expected = math.sin(angle) if dim % 2 == 0 else math.cos(angle)
                    self.assertAlmostEqual(pe[pos, dim].item(), expected, places=5)


if __name__ == "__main__":
    unittest.main()
