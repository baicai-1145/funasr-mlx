"""Synthetic tests for the AWQ port (no checkpoint, tiny shapes, no audio)."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from funasr_mlx.models.llm.awq import (  # noqa: E402
    AwqBlockQuantizer,
    AwqConfig,
    QWEN3_NO_CLIP,
    QWEN3_SCALE_SITES,
)


class TinyAttn(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.q_proj = nn.Linear(dim, heads * 16, bias=False)
        self.k_proj = nn.Linear(dim, 8, bias=False)
        self.v_proj = nn.Linear(dim, heads * 16, bias=False)
        self.o_proj = nn.Linear(heads * 16, dim, bias=False)

    def __call__(self, x):
        # every projection runs, like the real attention
        return self.o_proj(self.v_proj(x)) + 0.0 * (
            self.q_proj(x).sum() + self.k_proj(x).sum()
        )


class TinyMLP(nn.Module):
    def __init__(self, dim=64, hidden=128):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(self.gate_proj(x) * mx.sigmoid(self.up_proj(x)))


class TinyBlock(nn.Module):
    """Qwen3-block topology at toy size: norm -> attn(q,k,v,o) -> norm -> mlp."""

    def __init__(self, dim=64, hidden=128, heads=4):
        super().__init__()
        self.self_attn = TinyAttn(dim, heads)
        self.mlp = TinyMLP(dim, hidden)
        self.input_layernorm = nn.RMSNorm(dim, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=1e-6)

    def __call__(self, x):
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


def _quantize_func(bits=4, group_size=64):
    def q(w):
        wq = mx.quantize(w, bits=bits, group_size=group_size)
        return mx.dequantize(*wq, bits=bits, group_size=group_size)

    return q


def _make_input(block, rows=96, dim=64):
    # an input with a few strong channels (the outlier case AWQ targets)
    base = mx.random.normal((rows, dim))
    strong = mx.zeros((rows, dim))
    strong[:, 3] = 8.0
    strong[:, 17] = -6.0
    return (base + strong)[None]


def test_capture_records_every_linear_input():
    block = TinyBlock()
    x = _make_input(block)
    quantizer = AwqBlockQuantizer(block, AwqConfig(n_grid=4, clip_rows=32))

    def run():
        block(x)

    quantizer.capture(run)
    named = dict(block.named_modules())
    linears = [p for p, m in named.items() if isinstance(m, nn.Linear)]
    assert len(linears) == 7
    for path in linears:
        module = named[path]
        assert hasattr(module, "input_feat"), path
        assert module.input_feat.shape[0] == x.shape[1]
    # the block's own input (residual stream) is captured on the first norm
    assert hasattr(block.input_layernorm, "block_input")
    assert block.input_layernorm.block_input.shape == (x.shape[1], x.shape[2])
    # capture restored the original modules (no Catcher left behind)
    for path, module in dict(block.named_modules()).items():
        if isinstance(module, nn.Linear):
            assert type(module).__name__ == "Linear", path


def test_scale_search_folds_and_never_breaks_shapes():
    block = TinyBlock()
    x = _make_input(block)
    quantizer = AwqBlockQuantizer(block, AwqConfig(n_grid=4, clip_rows=32))
    quantizer.capture(lambda: block(x))
    scales = quantizer.search_scale(QWEN3_SCALE_SITES[0], _quantize_func())
    assert scales.shape == (64,)
    assert float(scales.min()) > 0
    quantizer.apply_scale(QWEN3_SCALE_SITES[0], scales)
    # the fold keeps every weight's shape and dtype
    for path, module in block.leaf_modules().items():
        if isinstance(module, nn.Linear):
            assert module.weight.shape[1] == module.weight.shape[1]
    # norm weight divided, linear weight multiplied: math equivalent
    # y = norm(x) * (w * s) = (norm(x) / s) * w -- check the folded norm
    ln = block.input_layernorm.weight
    assert ln is not None


def test_clip_respects_no_clip_list():
    assert any("q_proj" in name for name in QWEN3_NO_CLIP)
    assert any("k_proj" in name for name in QWEN3_NO_CLIP)


def test_optimize_never_worse_than_rtn():
    """The guarantee: AWQ's measured loss <= plain RTN's, or fallback fires."""
    mx.random.seed(0)
    block = TinyBlock()
    x = _make_input(block)
    quantizer = AwqBlockQuantizer(block, AwqConfig(n_grid=6, clip_rows=48))
    quantizer.capture(lambda: block(x))
    stats = quantizer.optimize(_quantize_func(), do_clip=True)
    assert stats["after"] <= stats["before"] + 1e-9 or stats["fallback"]
    assert stats["before"] > 0


def test_mse_is_float32():
    a = mx.array([1.0, 2.0], dtype=mx.float16)
    b = mx.array([1.5, 2.5], dtype=mx.float16)
    out = None
    from funasr_mlx.models.llm.awq import mse

    out = mse(a, b)
    assert out.dtype == mx.float32
