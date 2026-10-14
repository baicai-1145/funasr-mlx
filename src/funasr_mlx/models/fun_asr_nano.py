"""FunASRNano: audio encoder + adaptor + Qwen3 LLM, MLX port.

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/fun_asr_nano.py.md
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as mxnn

from .sensevoice import SenseVoiceEncoderSmall, SenseVoiceEncoderSmallConfig
from .llm import KVCache, Qwen3ForCausalLM, Qwen3ModelArgs

__all__ = ["AdaptorConfig", "FunASRNanoConfig", "FunASRNano", "nano_config_from_dict"]


@dataclass
class AdaptorConfig:

    downsample_rate: int = 1
    encoder_dim: int = 512
    llm_dim: int = 1024
    ffn_dim: int = 2048
    n_layer: int = 2
    attention_heads: int = 8


class TransformerAdaptor(mxnn.Module):
    # see docs/port-notes/functions.md :: TransformerAdaptor

    def __init__(self, cfg: AdaptorConfig) -> None:
        super().__init__()
        self.k = cfg.downsample_rate
        self.linear1 = mxnn.Linear(cfg.encoder_dim * self.k, cfg.ffn_dim)
        self.linear2 = mxnn.Linear(cfg.ffn_dim, cfg.llm_dim)
        self._n_layer = cfg.n_layer
        self._heads = cfg.attention_heads
        self._dim = cfg.llm_dim
        self.blocks = self._build_blocks()

    def _build_blocks(self) -> list[mxnn.Module]:
        blocks = []
        for _ in range(self._n_layer):
            attn = StandardMHA(self._heads, self._dim)
            ffn = PositionwiseFeedForward(self._dim, self._dim // 4, 0.0)
            blocks.append(_EncoderLayer(self._dim, attn, ffn))
        return blocks

    def __call__(self, x: mx.array) -> mx.array:
        b, t, d = x.shape
        chunk_num = (t - 1) // self.k + 1
        pad_num = chunk_num * self.k - t
        if pad_num:
            x = mx.pad(x, [(0, 0), (0, pad_num), (0, 0)])
        x = mx.reshape(x, (b, chunk_num, d * self.k))
        x = mx.maximum(self.linear1(x), 0.0)
        x = self.linear2(x)
        for block in self.blocks:
            x = block(x)
        return x


class StandardMHA(mxnn.Module):
    # see docs/port-notes/functions.md :: StandardMHA

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self._h = n_head
        self._d_k = n_feat // n_head
        self.linear_q = mxnn.Linear(n_feat, n_feat)
        self.linear_k = mxnn.Linear(n_feat, n_feat)
        self.linear_v = mxnn.Linear(n_feat, n_feat)
        self.linear_out = mxnn.Linear(n_feat, n_feat)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        b, t, _ = x.shape

        def split(v):
            return mx.transpose(mx.reshape(v, (b, t, self._h, self._d_k)), (0, 2, 1, 3))

        q, k, v = split(self.linear_q(x)), split(self.linear_k(x)), split(self.linear_v(x))
        scores = mx.matmul(q * (self._d_k**-0.5), mx.swapaxes(k, -1, -2))
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (b, t, self._h * self._d_k))
        return self.linear_out(out)


from .sanm import PositionwiseFeedForward  # noqa: E402


class _EncoderLayer(mxnn.Module):

    def __init__(self, size: int, self_attn: mxnn.Module, feed_forward: mxnn.Module) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = mxnn.LayerNorm(size)
        self.norm2 = mxnn.LayerNorm(size)
        self._size = size

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        attn = self.self_attn(self.norm1(x), mask)
        x = x + attn
        return x + self.feed_forward(self.norm2(x))


@dataclass
class FunASRNanoConfig:
    audio_encoder_conf: SenseVoiceEncoderSmallConfig = field(
        default_factory=SenseVoiceEncoderSmallConfig
    )
    adaptor_conf: AdaptorConfig = field(default_factory=AdaptorConfig)
    llm_conf: dict[str, Any] = field(default_factory=dict)
    input_size: int = 560

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FunASRNanoConfig":
        cfg = dict(data)
        enc = SenseVoiceEncoderSmallConfig.from_dict(cfg.get("audio_encoder_conf", {}))
        ad = AdaptorConfig(
            **{
                k: v
                for k, v in cfg.get("audio_adaptor_conf", {}).items()
                if k in AdaptorConfig.__dataclass_fields__
            }
        )
        return cls(
            audio_encoder_conf=enc,
            adaptor_conf=ad,
            llm_conf=dict(cfg.get("llm_conf", {})),
            input_size=int(cfg.get("input_size", 560)),
        )


class FunASRNano(mxnn.Module):
    # see docs/port-notes/functions.md :: FunASRNano

    def __init__(self, config: FunASRNanoConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        cfg = config or nano_config_from_dict(None, **kwargs)
        self.config = cfg
        self.audio_encoder = SenseVoiceEncoderSmall(cfg.input_size, cfg.audio_encoder_conf)
        self.audio_adaptor = TransformerAdaptor(cfg.adaptor_conf)
        llm_args = Qwen3ModelArgs.from_dict(cfg.llm_conf.get("args", {}))
        self.llm = Qwen3ForCausalLM(llm_args)

    def encode_audio(self, speech: mx.array, speech_lengths: mx.array) -> mx.array:
        # see docs/port-notes/functions.md :: encode_audio
        enc, _ = self.audio_encoder(speech, speech_lengths)
        return self.audio_adaptor(enc)

    def embed_with_speech(
        self,
        token_ids: mx.array,
        speech_embeds: mx.array,
        speech_beg: int,
    ) -> mx.array:
        # see docs/port-notes/functions.md :: embed_with_speech
        h = self.llm.model.embed_tokens(token_ids)
        n = speech_embeds.shape[1]
        h = mx.concatenate([h[:, :speech_beg, :], speech_embeds, h[:, speech_beg + n :, :]], axis=1)
        return h

    def forward(
        self,
        token_ids: mx.array,
        speech_embeds: mx.array,
        speech_beg: int,
        caches: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_with_speech(token_ids, speech_embeds, speech_beg)
        return self.llm.model(input_embeddings=h, cache=caches)


def nano_config_from_dict(data: Mapping[str, Any] | None = None, **kwargs: Any) -> FunASRNanoConfig:
    data = dict(data or {})
    data.update(kwargs)
    return FunASRNanoConfig.from_dict(data)


def nano_config_from_nano_yaml(path: str) -> FunASRNanoConfig:
    # see docs/port-notes/functions.md :: nano_config_from_nano_yaml
    import yaml

    raw = yaml.safe_load(open(path).read())
    return nano_config_from_dict(
        {
            "audio_encoder_conf": raw.get("audio_encoder_conf") or {},
            "audio_adaptor_conf": raw.get("audio_adaptor_conf") or {},
            "input_size": 560,
            "llm_conf": {
                "args": {
                    "hidden_size": 1024,
                    "num_hidden_layers": 28,
                    "intermediate_size": 3072,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "vocab_size": 151936,
                    "tie_word_embeddings": True,
                    "max_position_embeddings": 40960,
                    "rope_theta": 1000000.0,
                }
            },
        }
    )
