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
    """Audio tower + Qwen3 decoder with the audio span spliced in."""

    def __init__(self, config: Qwen3ASRConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_tower = Qwen3ASRAudioEncoder(config.audio)
        self.thinker = Qwen3ForCausalLM(config.text)

    def _decoder_dtype(self) -> mx.Dtype:
        """Dtype the decoder activations must follow.

        The token embedding produces the activations the decoder consumes, so
        its OUTPUT dtype is the one to cast spliced audio to.  Read through
        :func:`activation_dtype`, not ``weight.dtype``: a quantized embedding
        stores packed ``uint32`` codes in ``weight``, and casting the audio
        embeddings to ``uint32`` destroys them (see the helper's docstring).
        """
        from .llm.quant import activation_dtype

        return activation_dtype(self.thinker.model.embed_tokens)

    def _tower_dtype(self) -> mx.Dtype:
        """Dtype of the audio tower's checkpoint weights (first conv)."""
        return self.audio_tower.conv2d1.conv.weight.dtype

    def encode_audio(self, mel: mx.array) -> mx.array:
        """``(F, T)`` mel -> ``(1, T', text.hidden_size)`` tower embeddings.

        The front-end always produces float32 mel, so the mel is cast to the
        tower's **weight** dtype first (same rule as whisper's
        ``AudioEncoder.__call__``); without it an fp16 checkpoint runs the
        conv stack and every tower Linear in mixed precision.  Interleaved
        A/B on M4 (25 rounds, .tmp/task4/tower-tight-*.json): 55.7 -> 44.0 ms
        at the real 420-frame shape and 273.9 -> 214.7 ms at 30 s, i.e. 1.27x
        at both dtypes.  For a float32 checkpoint the cast is a no-op, so
        those numerics stay bit-identical.
        """
        return self.audio_tower(mel.astype(self._tower_dtype()))

    def forward_with_audio(
        self,
        token_ids: mx.array,
        audio_embeds: mx.array,
        pad_span: int,
        caches: list[KVCache] | None = None,
    ) -> mx.array:
        """Replace the ``pad_span`` audio run with tower output (shared splice)."""
        th = self.thinker
        cfg = self.config
        return splice_and_forward(
            token_ids, audio_embeds, pad_span, caches,
            audio_token_id=cfg.audio_token_id,
            embed_tokens=th.model.embed_tokens,
            decoder=th.model,
            embeds_as_linear=th.model.embed_tokens.as_linear,
            lm_head=th.lm_head,
            tied=cfg.text.tie_word_embeddings,
            decoder_dtype=self._decoder_dtype(),
        )
