"""Shared audio-splice prefill for the LLM-ASR thinkers (cluster D).

The glm/qwen3/moss thinkers each reimplemented the same forward:
find the audio-token run, splice tower embeds over it (cast to the
decoder's dtype first -- see the dtype note in each family), run one
decoder pass with an explicit causal mask for prefill, and pick the
output head by tie_word_embeddings. The only real differences are where
each family keeps its attributes:

* embedder/decoder call: ``self.thinker(...)`` vs ``self.language_model(...)``
* audio token id: ``cfg.audio_token_id`` vs ``self.audio_token_id``
* head selection: tie flag location differs; glm probes lm_head presence.

``splice_and_forward`` takes those as explicit parameters; each thinker
keeps its public ``forward_with_audio`` as a one-line adapter.
"""
from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx

from .kv_cache import KVCache


def splice_and_forward(
    token_ids: mx.array,
    audio_embeds: mx.array,
    pad_span: int,
    caches: list[KVCache] | None,
    *,
    audio_token_id: int,
    embed_tokens: Callable[[mx.array], mx.array],
    decoder: Callable[..., mx.array],
    embeds_as_linear: Callable[[mx.array], mx.array],
    lm_head: Callable[[mx.array], mx.array] | None,
    tied: bool | None,
    decoder_dtype: mx.Dtype,
) -> mx.array:
    """Replace the ``pad_span``-wide audio-token run with tower output; decode.

    Prefill needs the causal mask even when KVCache objects already exist:
    the generic model call only builds one when ``cache is None`` as a whole.
    ``decoder_dtype``: the tower emits float32 embeds (float32 mel in); the
    cast prevents a mixed-dtype promotion at every Linear of the stack (a
    no-op for float32 checkpoints).
    """
    match = token_ids == audio_token_id
    first = int(mx.argmax(match).item())
    h = embed_tokens(token_ids)
    h = mx.concatenate(
        [h[:, :first, :], audio_embeds.astype(decoder_dtype), h[:, first + pad_span :, :]],
        axis=1,
    )
    prefill = caches is None or all(c is None or c.offset == 0 for c in caches)
    mask = "causal" if prefill and h.shape[1] > 1 else None
    out = decoder(input_embeddings=h, cache=caches, mask=mask)
    if lm_head is not None and not tied:
        return lm_head(out)
    return embeds_as_linear(out)
