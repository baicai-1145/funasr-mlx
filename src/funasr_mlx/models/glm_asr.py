"""GLM-ASR-Nano audio encoder + projector, ported from the checkpoint's code.

Structure verified from ``model.safetensors`` (zai-org/GLM-ASR-Nano-2512,
MIT): ``conv1`` (1280, 128, 3) stride 1 + ``conv2`` (1280, 1280, 3) stride 2
=> a single 2x downsample; 32 pre-norm blocks with Llama-style names
(``input_layernorm``/``post_attention_layernorm``), attention q/k/v/o where
q, v, o carry biases and k does not, ``partial_rotary_factor`` 0.5; final
``norm``; projector ``linear_1`` 1280->4096 + gelu + ``linear_2`` 4096->2048.

Text side is a Llama-shaped decoder (see :mod:`.llm.llama`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as mxnn

__all__ = ["GlmAsrAudioConfig", "GlmAsrAudioEncoder", "glmasr_audio_config_from_dict"]


@dataclass
class GlmAsrAudioConfig:
    num_mel_bins: int = 128
    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    head_dim: int = 64
    intermediate_size: int = 5120
    max_position_embeddings: int = 1500
    partial_rotary_factor: float = 0.5
    rope_theta: float = 10000.0
    projector_dim: int = 4096
    output_size: int = 2048  # GLM text hidden
    # intermediate_size (5120) == hidden (1280) * groups (4): the projector
    # merges 4 encoder frames into one row
    projector_groups: int = 4


def glmasr_audio_config_from_dict(data: dict[str, Any]) -> GlmAsrAudioConfig:
    known = {f.name for f in __import__("dataclasses").fields(GlmAsrAudioConfig)}
    return GlmAsrAudioConfig(**{k: v for k, v in data.items() if k in known})


def _downsample_length(n: int) -> int:
    """One stride-2 conv (kernel 3, padding 1): floor((L + 1) / 2)."""
    return (n + 1) // 2


class _Block(mxnn.Module):
    """Pre-norm block, Llama naming, partial RoPE on q/k."""

    def __init__(self, cfg: GlmAsrAudioConfig) -> None:
        super().__init__()
        d = cfg.hidden_size
        self.input_layernorm = mxnn.LayerNorm(d)
        self.self_attn = _Attn(cfg)
        self.post_attention_layernorm = mxnn.LayerNorm(d)
        self.mlp = _Mlp(cfg)

    def __call__(self, x: mx.array) -> mx.array:
        r = x
        x = self.self_attn(self.input_layernorm(x))
        x = r + x
        r = x
        x = self.mlp(self.post_attention_layernorm(x))
        return r + x


class _Attn(mxnn.Module):
    """q/v/o with bias, k without; partial RoPE (first half of head dim)."""

    def __init__(self, cfg: GlmAsrAudioConfig) -> None:
        super().__init__()
        d, h, kv = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
        self.q_proj = mxnn.Linear(d, h * cfg.head_dim, bias=True)
        self.k_proj = mxnn.Linear(d, kv * cfg.head_dim, bias=False)
        self.v_proj = mxnn.Linear(d, kv * cfg.head_dim, bias=True)
        self.o_proj = mxnn.Linear(h * cfg.head_dim, d, bias=True)
        self._h, self._kv, self._dh = h, kv, cfg.head_dim
        r = int(cfg.head_dim * cfg.partial_rotary_factor)
        self._rotary = r
        half = r // 2
        pos = mx.arange(cfg.max_position_embeddings, dtype=mx.float32)
        inv = cfg.rope_theta ** (-mx.arange(half, dtype=mx.float32) / half)
        angles = pos[:, None] * inv[None, :]
        self._cos = mx.cos(angles)
        self._sin = mx.sin(angles)

    def _rope(self, x: mx.array) -> mx.array:
        r = self._rotary
        x1, x2 = x[..., : r // 2], x[..., r // 2 : r]
        cos = mx.concatenate([self._cos, self._cos], axis=-1)[None, None, : x.shape[2]]
        sin = mx.concatenate([self._sin, self._sin], axis=-1)[None, None, : x.shape[2]]
        rot = mx.concatenate([-x2, x1], axis=-1)
        return mx.concatenate([x[..., :r] * cos + rot * sin, x[..., r:]], axis=-1)

    def __call__(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape

        def split(v, heads):
            return mx.transpose(mx.reshape(v, (b, t, heads, self._dh)), (0, 2, 1, 3))

        q = self._rope(split(self.q_proj(x), self._h))
        k = self._rope(split(self.k_proj(x), self._kv))
        v = split(self.v_proj(x), self._kv)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self._dh**-0.5)
        o = mx.reshape(mx.transpose(o, (0, 2, 1, 3)), (b, t, -1))
        return self.o_proj(o)


class _Mlp(mxnn.Module):
    def __init__(self, cfg: GlmAsrAudioConfig) -> None:
        super().__init__()
        self.fc1 = mxnn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = mxnn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(mxnn.GELU()(self.fc1(x)))


class GlmAsrAudioEncoder(mxnn.Module):
    """``glmasr_encoder`` (inference): mel (F, T) -> (1, T', output_size)."""

    def __init__(self, cfg: GlmAsrAudioConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv1 = mxnn.Conv1d(cfg.num_mel_bins, cfg.hidden_size, kernel_size=3, padding=1)
        self.conv2 = mxnn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=3, stride=2, padding=1)
        self.layers = [_Block(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = mxnn.LayerNorm(cfg.hidden_size)
        self.linear_1 = mxnn.Linear(cfg.intermediate_size, cfg.projector_dim)
        self.linear_2 = mxnn.Linear(cfg.projector_dim, cfg.output_size)

    def __call__(self, mel: mx.array) -> mx.array:
        """mel: ``(F, T)``; returns ``(1, T'', output_size)`` embeddings.

        The official projector consumes the encoder output RE-GROUPED into
        ``intermediate_size``-wide rows: every 4 frames of 1280 become one
        5120-wide row (a 4-frame merge), then linear_1 5120->4096 + gelu +
        linear_2 4096->2048.
        """
        x = mxnn.GELU()(
            self.conv1(mx.transpose(mel[None], (0, 2, 1)))
        )  # (1, T, d) stride 1
        x = mxnn.GELU()(self.conv2(x))  # (1, T', d) stride 2
        # positions are handled by the per-block partial RoPE ONLY; the
        # official GlmAsrEncoder has no additive sinusoid table
        for blk in self.layers:
            x = blk(x)
        x = self.norm(x)
        b, t, d = x.shape
        g = self.cfg.projector_groups  # frames merged per projector row
        # the reference pads the mel to 3000 frames so L % 4 == 0; on real
        # length pad the conv output to a multiple of g, reshape, then keep
        # the first (t - g) // g + 1 rows (the padding rows are dropped)
        if t % g:
            x = mx.pad(x, [(0, 0), (0, g - t % g), (0, 0)])
        n = (t - g) // g + 1
        x = mx.reshape(x, (b, -1, d * g))[:, :n]
        x = mxnn.GELU()(self.linear_1(x))
        return self.linear_2(x)
