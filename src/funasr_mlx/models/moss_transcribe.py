"""Port notes: docs/port-notes/moss_transcribe.py.md"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as mxnn

from .llm.kv_cache import KVCache
from .llm.splice import splice_and_forward
from .llm.qwen3 import Qwen3ForCausalLM


class _HfWhisperBlock(mxnn.Module):

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
    # see docs/port-notes/functions.md :: _HfWhisperAttn

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
    # see docs/port-notes/functions.md :: HfWhisperEncoder

    def __init__(self, n_mels: int, d: int, layers: int, heads: int, ffn: int,
                 max_source_positions: int) -> None:
        super().__init__()
        self.conv1 = mxnn.Conv1d(n_mels, d, kernel_size=3, padding=1)  # stride 1
        self.conv2 = mxnn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)  # stride 2
        self.embed_positions = mxnn.Embedding(max_source_positions, d)
        self.layers = [_HfWhisperBlock(d, heads, ffn) for _ in range(layers)]
        self.layer_norm = mxnn.LayerNorm(d)  # top-level (HF 5.x)

    def __call__(self, mel: mx.array) -> mx.array:
        x = mxnn.GELU()(self.conv1(mx.transpose(mel[None], (0, 2, 1))))
        x = mxnn.GELU()(self.conv2(x))  # (1, T/2, d) channels-last
        x = x + self.embed_positions.weight[: x.shape[1]]
        for blk in self.layers:
            x = blk(x)
        return self.layer_norm(x)


class VQAdaptor(mxnn.Module):
    # see docs/port-notes/functions.md :: VQAdaptor

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
    # see docs/port-notes/functions.md :: MossTranscribeDiarize

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
        # see docs/port-notes/functions.md :: encode_audio
        whisper_hidden = self.whisper_encoder(mel)
        return self.encode_audio_from_hidden(whisper_hidden)

    def encode_audio_from_hidden(self, whisper_hidden: mx.array) -> mx.array:
        # see docs/port-notes/functions.md :: encode_audio_from_hidden
        m = self.audio_merge_size
        b, t, d = whisper_hidden.shape
        trim = (t // m) * m
        x = whisper_hidden[:, :trim, :]
        x = mx.reshape(x, (b, trim // m, d * m))
        return self.vq_adaptor(x)

    def _decoder_dtype(self) -> mx.Dtype:
        # see docs/port-notes/functions.md :: _decoder_dtype
        from .llm.quant import activation_dtype

        return activation_dtype(self.language_model.model.embed_tokens)

    def forward_with_audio(
        self,
        token_ids: mx.array,
        audio_embeds: mx.array,
        pad_span: int,
        caches: list[KVCache] | None = None,
    ) -> mx.array:
        # see docs/port-notes/functions.md :: forward_with_audio
        lm = self.language_model
        return splice_and_forward(
            token_ids, audio_embeds, pad_span, caches,
            audio_token_id=self.audio_token_id,
            embed_tokens=lm.model.embed_tokens,
            decoder=lm.model,
            embeds_as_linear=lm.model.embed_tokens.as_linear,
            lm_head=getattr(lm, "lm_head", None),
            tied=lm.args.tie_word_embeddings,
            decoder_dtype=self._decoder_dtype(),
        )
