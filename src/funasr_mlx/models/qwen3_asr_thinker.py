"""Port notes: docs/port-notes/qwen3_asr_thinker.py.md"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as mxnn

from .llm import KVCache, Qwen3ForCausalLM, Qwen3ModelArgs
from .llm.splice import splice_and_forward
from .qwen3_asr import (
    Qwen3ASRAudioConfig,
    Qwen3ASRAudioEncoder,
    qwen3asr_audio_config_from_dict,
)

__all__ = ["Qwen3ASRConfig", "Qwen3ASR", "qwen3asr_config_from_dict"]


@dataclass
class Qwen3ASRConfig:
    audio: Qwen3ASRAudioConfig
    text: Qwen3ModelArgs
    audio_start_token_id: int = 151669
    audio_pad_token_id: int = 151676
    audio_end_token_id: int = 151670


def qwen3asr_config_from_dict(data: dict[str, Any]) -> Qwen3ASRConfig:
    tk = dict(data.get("thinker_config", data))
    ac = qwen3asr_audio_config_from_dict(tk.get("audio_config", {}))
    tc = dict(tk.get("text_config", {}))
    text = Qwen3ModelArgs.from_dict(tc)
    return Qwen3ASRConfig(
        audio=ac,
        text=text,
        audio_start_token_id=int(tk.get("audio_start_token_id", 151669)),
        audio_pad_token_id=int(tk.get("audio_token_id", 151676)),
        audio_end_token_id=int(tk.get("audio_end_token_id", 151670)),
    )


class Qwen3ASR(mxnn.Module):

    def __init__(self, config: Qwen3ASRConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_tower = Qwen3ASRAudioEncoder(config.audio)
        self.thinker = Qwen3ForCausalLM(config.text)

    def _decoder_dtype(self) -> mx.Dtype:
        # see docs/port-notes/functions.md :: _decoder_dtype
        from .llm.quant import activation_dtype

        return activation_dtype(self.thinker.model.embed_tokens)

    def _tower_dtype(self) -> mx.Dtype:
        # see docs/port-notes/functions.md :: _tower_dtype
        return self.audio_tower.conv2d1.conv.weight.dtype

    def encode_audio(self, mel: mx.array) -> mx.array:
        # see docs/port-notes/functions.md :: encode_audio
        return self.audio_tower(mel.astype(self._tower_dtype()))

    def forward_with_audio(
        self,
        token_ids: mx.array,
        audio_embeds: mx.array,
        pad_span: int,
        caches: list[KVCache] | None = None,
    ) -> mx.array:
        th = self.thinker
        cfg = self.config
        return splice_and_forward(
            token_ids, audio_embeds, pad_span, caches,
            audio_token_id=cfg.audio_pad_token_id,
            embed_tokens=th.model.embed_tokens,
            decoder=th.model,
            embeds_as_linear=th.model.embed_tokens.as_linear,
            lm_head=getattr(th, "lm_head", None),
            tied=cfg.text.tie_word_embeddings,
            decoder_dtype=self._decoder_dtype(),
        )
