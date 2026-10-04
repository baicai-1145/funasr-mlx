"""MOSS-Transcribe-Diarize: Whisper-Medium encoder + VQAdaptor + Qwen3-0.6B.

Reference: OpenMOSS-Team/MOSS-Transcribe-Diarize ``modeling_moss_transcribe_diarize.py``
(ships with the checkpoint; MIT). The audio tower is the **HF transformers**
``WhisperEncoder`` layout (NOT openai/whisper): conv1 stride 2 + conv2
stride 1 (one 2x downsample total), a LEARNED ``embed_positions`` table
(1500, 1024), 24 pre-norm blocks with ``self_attn_layer_norm`` /
``final_layer_norm`` (LayerNorm with bias), MHA with q/k/v/out biases,
fc1/fc2 GELU, no top-level norm. Text side reuses :mod:`.llm.qwen3`
(q_norm/k_norm, tie_word_embeddings).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as mxnn

from .llm.kv_cache import KVCache
from .llm.qwen3 import Qwen3ForCausalLM


class _HfWhisperBlock(mxnn.Module):
    """HF WhisperEncoderLayer: pre-norm MHA + GELU MLP, LN with bias."""

    def __init__(self, d: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.self_attn_layer_norm = mxnn.LayerNorm(d)
        self.self_attn = _HfWhisperAttn(d, heads)
        self.final_layer_norm = mxnn.LayerNorm(d)
        self.fc1 = mxnn.Linear(d, ffn)
        self.fc2 = mxnn.Linear(ffn, d)

    def __call__(self, x: mx.array) -> mx.array:
        r = x
        x = self.self_attn(self.self_attn_layer_norm(x))
        x = r + x
        r = x
        x = self.fc2(mxnn.GELU()(self.fc1(self.final_layer_norm(x))))
        return r + x


class _HfWhisperAttn(mxnn.Module):
    """HF ``WhisperAttention``: q/v/out carry biases, **k does not**.

    ``k_proj`` is bias-free in the reference (``nn.Linear(embed_dim,
    embed_dim, bias=False)`` in transformers' ``modeling_whisper.py``), and the
    checkpoint agrees: ``...self_attn.k_proj.bias`` is absent from all 24
    layers.  Building it with ``bias=True`` therefore left a random,
    never-trained key bias in the audio tower -- measured on the previous
    release: exists=True, absmax 0.0312, absmean 0.0158.  The old permissive
    loaders never noticed, which is why MOSS tower parity sat at 1.3e-3.
    """

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.q_proj = mxnn.Linear(d, d, bias=True)
        self.k_proj = mxnn.Linear(d, d, bias=False)
        self.v_proj = mxnn.Linear(d, d, bias=True)
        self.out_proj = mxnn.Linear(d, d, bias=True)
        self._h = heads
        self._dh = d // heads

    def __call__(self, x: mx.array) -> mx.array:
        b, t, d = x.shape

        def split(v: mx.array) -> mx.array:
            return mx.transpose(mx.reshape(v, (b, t, self._h, self._dh)), (0, 2, 1, 3))

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self._dh**-0.5)
        o = mx.reshape(mx.transpose(o, (0, 2, 1, 3)), (b, t, d))
        return self.out_proj(o)


class HfWhisperEncoder(mxnn.Module):
    """HF ``WhisperEncoder`` (inference): mel (F, T) -> (1, T/2, d)."""

    def __init__(self, n_mels: int, d: int, layers: int, heads: int, ffn: int,
                 max_source_positions: int) -> None:
        super().__init__()
        self.conv1 = mxnn.Conv1d(n_mels, d, kernel_size=3, padding=1)  # stride 1
        self.conv2 = mxnn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)  # stride 2
        self.embed_positions = mxnn.Embedding(max_source_positions, d)
        self.layers = [_HfWhisperBlock(d, heads, ffn) for _ in range(layers)]
        self.layer_norm = mxnn.LayerNorm(d)  # top-level (HF 5.x)

    def __call__(self, mel: mx.array) -> mx.array:
        """mel: ``(F, T)``; returns ``(1, T/2, d)``."""
        x = mxnn.GELU()(self.conv1(mx.transpose(mel[None], (0, 2, 1))))
        x = mxnn.GELU()(self.conv2(x))  # (1, T/2, d) channels-last
        x = x + self.embed_positions.weight[: x.shape[1]]
        for blk in self.layers:
            x = blk(x)
        return self.layer_norm(x)


class VQAdaptor(mxnn.Module):
    """Linear(input_dim->hidden) + SiLU + Linear(hidden->hidden) + LN(1e-6)."""

    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.layers = [
            mxnn.Linear(input_dim, hidden, bias=True),
            mxnn.SiLU(),
            mxnn.Linear(hidden, hidden, bias=True),
            mxnn.LayerNorm(hidden, eps=1e-6),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x) if callable(layer) else x
        return x


class MossTranscribeDiarize(mxnn.Module):
    """whisper_encoder + vq_adaptor + qwen3 (audio-token splice)."""

    def __init__(
        self,
        whisper_cfg: dict[str, Any],
        text_config: dict[str, Any],
        audio_merge_size: int = 4,
        audio_token_id: int = 151671,
    ) -> None:
        super().__init__()
        self.whisper_encoder = HfWhisperEncoder(
            n_mels=whisper_cfg.get("num_mel_bins", 80),
            d=whisper_cfg.get("d_model", 1024),
            layers=whisper_cfg.get("encoder_layers", 24),
            heads=whisper_cfg.get("encoder_attention_heads", 16),
            ffn=whisper_cfg.get("encoder_ffn_dim", 4096),
            max_source_positions=whisper_cfg.get("max_source_positions", 1500),
        )
        hidden = text_config.hidden_size
        self.vq_adaptor = VQAdaptor(hidden * audio_merge_size, hidden)
        self.language_model = Qwen3ForCausalLM(text_config)
        self.audio_merge_size = audio_merge_size
        self.audio_token_id = audio_token_id

    def encode_audio(self, mel: mx.array) -> mx.array:
        """mel (F, T) -> whisper (1, T/2, 1024) -> adaptor (1, T/8, hidden)."""
        whisper_hidden = self.whisper_encoder(mel)
        return self.encode_audio_from_hidden(whisper_hidden)

    def encode_audio_from_hidden(self, whisper_hidden: mx.array) -> mx.array:
        """Merged/chunked whisper output (1, T, 1024) -> adaptor (1, T/4, hidden).

        Trims to a multiple of the merge size first (official time_merge).
        """
        m = self.audio_merge_size
        b, t, d = whisper_hidden.shape
        trim = (t // m) * m
        x = whisper_hidden[:, :trim, :]
        x = mx.reshape(x, (b, trim // m, d * m))
        return self.vq_adaptor(x)

    def forward_with_audio(
        self,
        token_ids: mx.array,
        audio_embeds: mx.array,
        pad_span: int,
        caches: list[KVCache] | None = None,
    ) -> mx.array:
        """Replace the audio-token run with adaptor output; one decoder pass.

        Prefill needs the causal mask even with pre-created KVCache objects
        (same pattern as the Qwen3-ASR / GLM thinkers).
        """
        match = token_ids == self.audio_token_id
        first = int(mx.argmax(match).item())
        h = self.language_model.model.embed_tokens(token_ids)
        h = mx.concatenate(
            [h[:, :first, :], audio_embeds, h[:, first + pad_span :, :]], axis=1
        )
        prefill = caches is None or all(
            c is None or c.offset == 0 for c in caches
        )
        mask = "causal" if prefill and h.shape[1] > 1 else None
        out = self.language_model.model(input_embeddings=h, cache=caches, mask=mask)
        if self.language_model.args.tie_word_embeddings:
            return self.language_model.model.embed_tokens.as_linear(out)
        return self.language_model.lm_head(out)
