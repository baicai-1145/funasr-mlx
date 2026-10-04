"""Tiny, offline, network-free tests for the GLM-ASR audio tower's dtype contract.

Deliberately tiny (4 mel bins, 8 channels, 2 blocks) so the file stays far below
the project memory ceiling.  Structural / algebraic checks only -- not
real-checkpoint acceptance.

The dtype test is the regression guard for a real bug: the tower's partial-RoPE
tables are built in float32, and ``x * cos`` promoted the whole encoder to
float32 activations against bf16 weights.  Casting only the mel (which an
earlier change did, measured 1.00x, and skipped the rest) does NOT fix it --
both must follow the weight dtype, together worth 1.44x on the tower.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core", reason="mlx is required for the model ports")

from mlx.utils import tree_map  # noqa: E402

from funasr_mlx.models.glm_asr import (  # noqa: E402
    GlmAsrAudioConfig,
    GlmAsrAudioEncoder,
)

N_MELS = 4
HIDDEN = 8
BLOCKS = 2


def _config() -> GlmAsrAudioConfig:
    return GlmAsrAudioConfig(
        num_mel_bins=N_MELS,
        hidden_size=HIDDEN,
        num_hidden_layers=BLOCKS,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=HIDDEN * 4,
        max_position_embeddings=64,
        partial_rotary_factor=0.5,
        rope_theta=10000.0,
        projector_dim=16,
        output_size=8,
        projector_groups=4,
    )


def _encoder(dtype=mx.bfloat16) -> GlmAsrAudioEncoder:
    enc = GlmAsrAudioEncoder(_config())
    # Stand in for a bf16 checkpoint.
    enc.update(tree_map(lambda a: a.astype(dtype), enc.parameters()))
    return enc.eval()


def test_rope_tables_stay_float32_for_accuracy():
    """The table is built in float32; the CAST happens at use time."""
    enc = _encoder()
    attn = next(m for m in enc.modules() if type(m).__name__ == "_Attn")
    assert attn._cos.dtype == mx.float32
    assert attn._sin.dtype == mx.float32


def test_rope_output_follows_the_activation_dtype():
    """Regression guard: a float32 table must not promote a bf16 stream."""
    enc = _encoder(mx.bfloat16)
    attn = next(m for m in enc.modules() if type(m).__name__ == "_Attn")
    x = mx.ones((1, attn._h, 5, attn._dh), dtype=mx.bfloat16)
    out = attn._rope(x)
    mx.eval(out)
    assert out.dtype == mx.bfloat16, (
        f"partial RoPE promoted {x.dtype} to {out.dtype}; the float32 cos/sin "
        "tables must be cast to the activation dtype"
    )


def test_rope_output_follows_a_float32_activation():
    """A float32 activation keeps float32 (the cast must not downcast)."""
    enc = _encoder(mx.float32)
    attn = next(m for m in enc.modules() if type(m).__name__ == "_Attn")
    x = mx.ones((1, attn._h, 5, attn._dh), dtype=mx.float32)
    out = attn._rope(x)
    mx.eval(out)
    assert out.dtype == mx.float32


def test_tower_output_follows_the_weight_dtype_for_a_float32_mel():
    """The encoder casts the mel, so the whole tower stays in the weight dtype."""
    enc = _encoder(mx.bfloat16)
    mel = mx.zeros((N_MELS, 32), dtype=mx.float32)
    out = enc(mel.astype(enc.conv1.weight.dtype))
    mx.eval(out)
    assert out.dtype == mx.bfloat16


def test_rope_is_bit_identical_to_an_explicit_float32_table():
    """Casting the table must not change the math for a float32 activation."""
    enc = _encoder(mx.float32)
    attn = next(m for m in enc.modules() if type(m).__name__ == "_Attn")
    x = mx.random.uniform(shape=(1, attn._h, 5, attn._dh)).astype(mx.float32)
    got = attn._rope(x)
    # reference: the original float32 formula
    r = attn._rotary
    x1, x2 = x[..., : r // 2], x[..., r // 2 : r]
    cos = mx.concatenate([attn._cos, attn._cos], axis=-1)[None, None, : x.shape[2]]
    sin = mx.concatenate([attn._sin, attn._sin], axis=-1)[None, None, : x.shape[2]]
    rot = mx.concatenate([-x2, x1], axis=-1)
    ref = mx.concatenate([x[..., :r] * cos + rot * sin, x[..., r:]], axis=-1)
    mx.eval(got, ref)
    assert mx.array_equal(got, ref)
