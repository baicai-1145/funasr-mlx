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

    def _decoder_dtype(self) -> mx.Dtype:
        """Dtype of the decoder's checkpoint weights (``embed_tokens``).

        This is the dtype the decoder activations must follow: the token
        embedding produces it, so splicing anything else into the stream
        promotes every Linear in the stack (see :meth:`forward_with_audio`).
        """
        return self.thinker.model.embed_tokens.weight.dtype

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
        # Follow the decoder's own weight dtype, exactly as whisper's
        # TextDecoder does.  The tower emits float32 embeds (float32 mel in),
        # so splicing them unchanged locked all 28 decoder layers into fp32
        # activations against fp16 weights: a mixed-dtype promotion at every
        # Linear, every step (measured on M4: 109.6 vs 20.7 ms/step).  A
        # float32 checkpoint is unaffected -- the cast is a no-op there.
        audio_embeds = audio_embeds.astype(self._decoder_dtype())
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
