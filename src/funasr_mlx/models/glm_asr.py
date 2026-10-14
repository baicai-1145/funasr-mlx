"""Port notes: docs/port-notes/glm_asr.py.md"""

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
    projector_groups: int = 4


def glmasr_audio_config_from_dict(data: dict[str, Any]) -> GlmAsrAudioConfig:
    known = {f.name for f in __import__("dataclasses").fields(GlmAsrAudioConfig)}
    return GlmAsrAudioConfig(**{k: v for k, v in data.items() if k in known})


def _downsample_length(n: int) -> int:
    return (n + 1) // 2


class _Block(mxnn.Module):

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
        # see docs/port-notes/comments.md :: glm_asr.py:72
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
        # see docs/port-notes/comments.md :: glm_asr.py:84
        cos = cos.astype(x.dtype)
        sin = sin.astype(x.dtype)
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
    # see docs/port-notes/functions.md :: GlmAsrAudioEncoder

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
        # see docs/port-notes/functions.md :: __call__
        x = mxnn.GELU()(
            self.conv1(mx.transpose(mel[None], (0, 2, 1)))
        )  # (1, T, d) stride 1
        x = mxnn.GELU()(self.conv2(x))  # (1, T', d) stride 2
        for blk in self.layers:
            x = blk(x)
        x = self.norm(x)
        b, t, d = x.shape
        g = self.cfg.projector_groups  # frames merged per projector row
        if t % g:
            x = mx.pad(x, [(0, 0), (0, g - t % g), (0, 0)])
        n = (t - g) // g + 1
        x = mx.reshape(x, (b, -1, d * g))[:, :n]
        x = mxnn.GELU()(self.linear_1(x))
        return self.linear_2(x)
