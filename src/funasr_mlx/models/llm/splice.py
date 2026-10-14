"""Port notes: docs/port-notes/splice.py.md"""
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
    # see docs/port-notes/functions.md :: splice_and_forward
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
