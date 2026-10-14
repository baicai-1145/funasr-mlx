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
    """audio_tower (with projector folded in) + Llama causal LM."""

    def __init__(self, cfg: GlmAsrThinkerConfig) -> None:
        super().__init__()
        self.config = cfg
        self.audio_tower = GlmAsrAudioEncoder(cfg.audio)
        self.language_model = LlamaForCausalLM(cfg.text)

    def _decoder_dtype(self) -> mx.Dtype:
        """Dtype the decoder activations must follow.

        Read through :func:`activation_dtype`, not ``weight.dtype``: a quantized
        embedding stores packed ``uint32`` codes in ``weight``, and casting the
        spliced audio embeddings to ``uint32`` destroys them.
        """
        from .llm.quant import activation_dtype

        return activation_dtype(self.language_model.model.embed_tokens)

    def _tower_dtype(self) -> mx.Dtype:
        """Dtype of the audio tower's checkpoint weights (first conv)."""
        return self.audio_tower.conv1.weight.dtype

    def encode_audio(self, mel: mx.array) -> mx.array:
        """mel (F, T) -> (1, N, text_hidden) embeddings for the pad run.

        The mel is cast to the tower's weight dtype, as whisper's AudioEncoder
        and qwen3-asr's tower do.  This cast ALONE measures 1.00x here (see
        .tmp/task4/tower-tight-*.json and the re-measurement in
        .tmp/t7/glm-isolate.json), which is why it was originally skipped -- but
        that A/B was misleading: the tower's partial-RoPE tables were float32,
        so attention re-promoted the stream to float32 regardless.  With the
        tables also following the activation dtype (see ``_Attn._rope``) the
        two casts together give 1.44x on the tower and 28.7% on the pipeline.
        """
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
