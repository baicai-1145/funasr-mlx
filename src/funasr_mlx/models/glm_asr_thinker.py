"""Port notes: docs/port-notes/glm_asr_thinker.py.md"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as mxnn

from .glm_asr import GlmAsrAudioEncoder, GlmAsrAudioConfig, glmasr_audio_config_from_dict
from .llm.splice import splice_and_forward
from .llm.kv_cache import KVCache
from .llm.llama import LlamaForCausalLM


class GlmAsrThinkerConfig:
    def __init__(
        self,
        audio: GlmAsrAudioConfig,
        text: dict[str, Any],
        audio_token_id: int = 59260,
        begin_audio_token_id: int = 59261,
        end_audio_token_id: int = 59262,
    ) -> None:
        self.audio = audio
        self.text = text
        self.audio_token_id = audio_token_id
        self.begin_audio_token_id = begin_audio_token_id
        self.end_audio_token_id = end_audio_token_id


def glmasr_thinker_config_from_dict(data: dict[str, Any]) -> GlmAsrThinkerConfig:
    return GlmAsrThinkerConfig(
        audio=glmasr_audio_config_from_dict(data.get("audio_config", {})),
        text=data.get("text_config", {}),
        audio_token_id=data.get("audio_token_id", 59260),
    )


class GlmAsrThinker(mxnn.Module):
    # see docs/port-notes/functions.md :: GlmAsrThinker

    def __init__(self, cfg: GlmAsrThinkerConfig) -> None:
        super().__init__()
        self.config = cfg
        self.audio_tower = GlmAsrAudioEncoder(cfg.audio)
        self.language_model = LlamaForCausalLM(cfg.text)

    def _decoder_dtype(self) -> mx.Dtype:
        # see docs/port-notes/functions.md :: _decoder_dtype
        from .llm.quant import activation_dtype

        return activation_dtype(self.language_model.model.embed_tokens)

    def _tower_dtype(self) -> mx.Dtype:
        # see docs/port-notes/functions.md :: _tower_dtype
        return self.audio_tower.conv1.weight.dtype

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
        lm = self.language_model
        return splice_and_forward(
            token_ids, audio_embeds, pad_span, caches,
            audio_token_id=self.config.audio_token_id,
            embed_tokens=lm.model.embed_tokens,
            decoder=lm.model,
            embeds_as_linear=lm.model.embed_tokens.as_linear,
            lm_head=getattr(lm, "lm_head", None),
            tied=False if hasattr(lm, "lm_head") else True,
            decoder_dtype=self._decoder_dtype(),
        )
