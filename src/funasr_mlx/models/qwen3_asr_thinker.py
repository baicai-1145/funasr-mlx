"""Qwen3-ASR thinker: audio tower + Qwen3 text decoder assembly (inference).

Model structure (``thinker.*`` of the official checkpoint):
``thinker.audio_tower`` (the conv encoder in :mod:`funasr_mlx.models.qwen3_asr`)
+ ``thinker.model`` / ``thinker.lm_head`` -- a Qwen3-0.6B-shaped decoder that
REUSES :class:`funasr_mlx.models.llm.Qwen3ForCausalLM` (the thinker's
text_config matches Qwen3-0.6B; rope/mask/qk-norm semantics are identical;
ASR never needs the MRoPE grid variant because all three position streams
carry the same ids for pure-audio turns).

Prompt (``get_generation_prompt`` of the qwen-asr package):
``<|im_start|>system\\n{context}<|im_end|>\\n<|im_start|>user\\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\\n<|im_start|>assistant\\n``
where N = audio-tower frames; the ``<|audio_pad|>`` span's embeddings are
replaced by the tower output. With ``language`` forced:
``...assistant\\nlanguage {lang}<asr_text>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as mxnn

from .llm import KVCache, Qwen3ForCausalLM, Qwen3ModelArgs
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
    """Audio tower + Qwen3 decoder with the audio span spliced in."""

    def __init__(self, config: Qwen3ASRConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_tower = Qwen3ASRAudioEncoder(config.audio)
        self.thinker = Qwen3ForCausalLM(config.text)

    def encode_audio(self, mel: mx.array) -> mx.array:
        """``(F, T)`` mel -> ``(1, T', text.hidden_size)`` tower embeddings."""
        return self.audio_tower(mel)

    def forward_with_audio(
        self,
        token_ids: mx.array,
        audio_embeds: mx.array,
        pad_span: int,
        caches: list[KVCache] | None = None,
    ) -> mx.array:
        """Replace the ``pad_span``-wide ``<|audio_pad|>*`` run with the tower
        output and run one decoder pass.

        The prefill pass (fresh caches, L > 1) needs the causal mask -- the
        generic model call only builds one when ``cache is None`` as a whole,
        so pass an explicit mask here for the prefill step.
        """
        cfg = self.config
        match = token_ids == cfg.audio_pad_token_id
        first = int(mx.argmax(match).item())
        h = self.thinker.model.embed_tokens(token_ids)
        h = mx.concatenate(
            [h[:, :first, :], audio_embeds, h[:, first + pad_span :, :]], axis=1
        )
        prefill = caches is None or all(
            c is None or c.offset == 0 for c in caches
        )
        mask = "causal" if prefill and h.shape[1] > 1 else None
        out = self.thinker.model(input_embeddings=h, cache=caches, mask=mask)
        if cfg.text.tie_word_embeddings:
            return self.thinker.model.embed_tokens.as_linear(out)
        return self.thinker.lm_head(out)
