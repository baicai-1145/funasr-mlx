"""GLM-ASR-Nano thinker: audio tower + projector + Llama decoder assembly.

Prompt (from the checkpoint's chat_template.jinja, verified against the
processor): ``<|user|>\\n<|begin_of_audio|><|pad|>*N<|end_of_audio|>\\n<|assistant|>\\n``
where N = number of projector output frames. The ``<|pad|>`` run's
embeddings are replaced with the tower+projector output.

N for raw audio of T mel frames (no 30s padding -- the reference pads the
mel to 3000 but the encoder is position-local so running real length only
is equivalent):
  conv1 s1: T -> T; conv2 s2: T -> (T+1)//2; projector 4-merge: -> (L-4)//4+1
where L = (T+1)//2.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as mxnn

from .glm_asr import GlmAsrAudioEncoder, GlmAsrAudioConfig, glmasr_audio_config_from_dict
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
    """audio_tower (with projector folded in) + Llama causal LM."""

    def __init__(self, cfg: GlmAsrThinkerConfig) -> None:
        super().__init__()
        self.config = cfg
        self.audio_tower = GlmAsrAudioEncoder(cfg.audio)
        self.language_model = LlamaForCausalLM(cfg.text)

    def encode_audio(self, mel: mx.array) -> mx.array:
        """mel (F, T) -> (1, N, text_hidden) embeddings for the pad run."""
        return self.audio_tower(mel)

    def forward_with_audio(
        self,
        token_ids: mx.array,
        audio_embeds: mx.array,
        pad_span: int,
        caches: list[KVCache] | None = None,
    ) -> mx.array:
        """Replace the pad run with tower output; one decoder pass.

        The prefill pass needs the causal mask even when KVCache objects
        already exist (see Qwen3-ASR thinker: the generic model call only
        builds a mask when cache is None as a whole).
        """
        cfg = self.config
        match = token_ids == cfg.audio_token_id
        first = int(mx.argmax(match).item())
        h = self.language_model.model.embed_tokens(token_ids)
        h = mx.concatenate(
            [h[:, :first, :], audio_embeds, h[:, first + pad_span :, :]], axis=1
        )
        prefill = caches is None or all(
            c is None or c.offset == 0 for c in caches
        )
        mask = "causal" if prefill and h.shape[1] > 1 else None
        out = self.language_model.model(
            input_embeddings=h, cache=caches, mask=mask
        )
        if hasattr(self.language_model, "lm_head"):
            return self.language_model.lm_head(out)
        return self.language_model.model.embed_tokens.as_linear(out)
