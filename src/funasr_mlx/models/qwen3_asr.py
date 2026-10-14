"""Port notes: docs/port-notes/qwen3_asr.py.md"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as mxnn

__all__ = [
    "Qwen3ASRAudioConfig",
    "Qwen3ASRAudioEncoder",
    "qwen3asr_audio_config_from_dict",
]


@dataclass
class Qwen3ASRAudioConfig:
    d_model: int = 896
    encoder_layers: int = 18
    encoder_attention_heads: int = 14
    encoder_ffn_dim: int = 4864
    downsample_hidden_size: int = 480
    output_dim: int = 1024
    n_mels: int = 128
    n_window: int = 50
    n_window_infer: int = 800
    conv_chunksize: int = 500
    activation_function: str = "gelu"
    max_source_positions: int = 4096


def qwen3asr_audio_config_from_dict(data: dict[str, Any]) -> Qwen3ASRAudioConfig:
    data = dict(data)
    if "num_mel_bins" in data and "n_mels" not in data:
        data["n_mels"] = data.pop("num_mel_bins")
    known = {f.name for f in __import__("dataclasses").fields(Qwen3ASRAudioConfig)}
    return Qwen3ASRAudioConfig(**{k: v for k, v in data.items() if k in known})


def _get_feat_extract_output_lengths(n: int) -> int:
    out = n
    for _ in range(3):
        out = (out + 1) // 2
    return out


class _Conv2dStride2(mxnn.Module):
    # see docs/port-notes/functions.md :: _Conv2dStride2

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = mxnn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x and output are NHWC (B, F, T, C) -- MLX Conv2d's native layout
        return self.conv(x)


class _AudioAttention(mxnn.Module):

    def __init__(self, cfg: Qwen3ASRAudioConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self._h = cfg.encoder_attention_heads
        self._dh = d // self._h
        self.q_proj = mxnn.Linear(d, d, bias=True)
        self.k_proj = mxnn.Linear(d, d, bias=True)
        self.v_proj = mxnn.Linear(d, d, bias=True)
        self.out_proj = mxnn.Linear(d, d, bias=True)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        b, t, d = x.shape

        def split(v):
            return mx.transpose(mx.reshape(v, (b, t, self._h, self._dh)), (0, 2, 1, 3))

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self._dh**-0.5, mask=mask)
        o = mx.reshape(mx.transpose(o, (0, 2, 1, 3)), (b, t, d))
        return self.out_proj(o)


class _AudioEncoderLayer(mxnn.Module):

    def __init__(self, cfg: Qwen3ASRAudioConfig) -> None:
        super().__init__()
        self.self_attn = _AudioAttention(cfg)
        self.self_attn_layer_norm = mxnn.LayerNorm(cfg.d_model)
        self.fc1 = mxnn.Linear(cfg.d_model, cfg.encoder_ffn_dim)
        self.fc2 = mxnn.Linear(cfg.encoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = mxnn.LayerNorm(cfg.d_model)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        r = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, mask)
        x = r + x
        r = x
        x = self.final_layer_norm(x)
        x = self.fc2(mxnn.GELU()(self.fc1(x)))
        return r + x


class _Sinusoids(mxnn.Module):

    def __init__(self, length: int, channels: int) -> None:
        super().__init__()
        # reference: inv_timescales[i] = exp(-log(1e4)/(C/2-1) * i) -- the
        # exponent multiplies arange BEFORE exp; my old form computed
        # exp(-log(1e4)*i)/(C/2-1) which is a different frequency ladder.
        position = mx.arange(length, dtype=mx.float32)[:, None]
        half = channels // 2
        exponent = -math.log(10000.0) / (half - 1) * mx.arange(half, dtype=mx.float32)
        inv_timescales = mx.exp(exponent)
        args = position * inv_timescales[None, :]
        self.table = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)

    def __call__(self, n: int) -> mx.array:
        return self.table[:n]


class Qwen3ASRAudioEncoder(mxnn.Module):
    # see docs/port-notes/functions.md :: Qwen3ASRAudioEncoder

    def __init__(self, cfg: Qwen3ASRAudioConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.downsample_hidden_size
        self.conv2d1 = _Conv2dStride2(1, h)
        self.conv2d2 = _Conv2dStride2(h, h)
        self.conv2d3 = _Conv2dStride2(h, h)
        # conv_out: Linear over (h * F_after_cnn)
        f_after = _get_feat_extract_output_lengths(cfg.n_mels)  # 80 -> 10
        self.conv_out = mxnn.Linear(h * f_after, cfg.d_model, bias=False)  # checkpoint ships no bias
        self.positional_embedding = _Sinusoids(cfg.max_source_positions, cfg.d_model)
        self.layers = [_AudioEncoderLayer(cfg) for _ in range(cfg.encoder_layers)]
        self.ln_post = mxnn.LayerNorm(cfg.d_model)
        self.proj1 = mxnn.Linear(cfg.d_model, cfg.d_model)
        self.proj2 = mxnn.Linear(cfg.d_model, cfg.output_dim)

    def __call__(self, mel: mx.array) -> mx.array:
        cfg = self.cfg
        t = mel.shape[1]

        # The reference time-splits into 2*n_window-mel chunks, pads every
        # chunk to the full window (pad_sequence), convolves per chunk, adds
        # positional embeddings computed for the PADDED length, then keeps
        # only each chunk's real frames (padded_mask_after_cnn). Verified
        # per-block against the reference hooks: padding the tail chunk
        # before the conv matters (4.2 abs otherwise), and the pos table is
        # applied on the padded 13 frames before slicing.
        win = cfg.n_window * 2
        n_chunks = max(1, -(-t // win))
        pe_len = _get_feat_extract_output_lengths(win)
        pe = self.positional_embedding(pe_len)
        outs = []
        for i in range(n_chunks):
            piece = mel[:, i * win : (i + 1) * win]
            real = _get_feat_extract_output_lengths(piece.shape[1])
            if piece.shape[1] < win:
                piece = mx.pad(piece, [(0, 0), (0, win - piece.shape[1])])
            x = mx.transpose(piece[None, None], (0, 2, 3, 1))  # (1, F, win, 1)
            e = mxnn.GELU()(self.conv2d1(x))
            e = mxnn.GELU()(self.conv2d2(e))
            e = mxnn.GELU()(self.conv2d3(e))  # (1, F', L', H)
            e = mx.transpose(e, (0, 2, 3, 1))  # (1, L', H, F')
            e = mx.reshape(e, (e.shape[0], e.shape[1], -1))
            e = self.conv_out(e) + pe[None, :, :]
            outs.append(e[:, :real, :])
        e = mx.concatenate(outs, axis=1)  # (1, T', d_model)

        t_out = e.shape[1]

        # Reference windowing: after concatenating the chunk frames, the
        # layers attend within windows of window_aftercnn = padded_chunk_frames
        # * (n_window_infer / (n_window*2)) AFTER-CNN frames (modeling file's
        # cu_chunk_lens over aftercnn_lens). For the default n_window_infer=
        # 800 and n_window=50 that is 104 frames; utterances shorter than one
        # window attend globally (mask None).
        win_mel = cfg.n_window * 2
        padded_chunk_frames = _get_feat_extract_output_lengths(win_mel)
        window_aftercnn = int(padded_chunk_frames * (cfg.n_window_infer // win_mel))
        aftercnn_total = _get_feat_extract_output_lengths(t)
        mask = None
        if aftercnn_total > window_aftercnn:
            block = mx.zeros((t_out, t_out), dtype=mx.float32)
            pos = 0
            while pos < t_out:
                end = min(pos + window_aftercnn, t_out)
                block[pos:end, pos:end] = 1.0
                pos = end
            mask = mx.broadcast_to(block[None, None, :, :], (1, 1, t_out, t_out))
            mask = mx.where(mask > 0, 0.0, -1e9)

        for layer in self.layers:
            e = layer(e, mask)
        e = self.ln_post(e)
        e = mxnn.GELU()(self.proj1(e))
        return self.proj2(e)
