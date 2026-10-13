"""Minimal KV cache for LLM-ASR decoders, ported from mlx-lm.

Source: mlx-lm 0.32.0 ``mlx_lm/models/cache.py`` (Apache 2.0,
Copyright © 2023-2024 Apple Inc.) -- only the basic ``KVCache`` used by
the Qwen3-family models; the quantized/rotating/paged variants of
upstream are not ported.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["KVCache"]


class KVCache:
    """mlx-lm ``KVCache``: append-only keys/values with an offset cursor.

    ``reserve`` pre-allocates the buffer once (call it after prefill, when the
    final length is known: audio positions + prompt + max decode steps). Without
    it, the cache grows in ``step``-sized chunks, and every growth allocates a
    fresh buffer and copies the whole history -- measured ~3 ms per event on the
    0.6B, a 2-3x tax on a 7 ms decode step, hitting exactly when a long
    utterance crosses a 256-position boundary mid-decode.
    """

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self.step = 256

    def reserve(self, positions: int, like: "KVCache | None" = None) -> None:
        """Grow the backing store to ``positions`` capacity immediately.

        ``like`` may carry the (B, H, D) shape when this cache is still empty.
        """
        if self.keys is not None and self.keys.shape[2] >= positions:
            return
        if self.keys is not None:
            B, H, _, D = self.keys.shape
            new_k = mx.zeros((B, H, positions, D), self.keys.dtype)
            new_v = mx.zeros((B, H, positions, D), self.values.dtype)
            new_k[:, :, : self.keys.shape[2]] = self.keys
            new_v[:, :, : self.values.shape[2]] = self.values
            self.keys, self.values = new_k, new_v
            mx.eval(self.keys, self.values)
        elif like is not None and like.keys is not None:
            B, H, _, D = like.keys.shape
            self.keys = mx.zeros((B, H, positions, D), like.keys.dtype)
            self.values = mx.zeros((B, H, positions, D), like.values.dtype)

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        offset = self.offset
        if self.keys is None or (offset + keys.shape[2]) > self.keys.shape[2]:
            B, H, _, D = keys.shape
            n = max(self.step, keys.shape[2])
            if self.keys is not None:
                n = max(n, self.step)
            new_k = mx.zeros((B, H, offset + n, D), keys.dtype)
            new_v = mx.zeros((B, H, offset + n, D), values.dtype)
            if self.keys is not None:
                # The backing store may be LARGER than offset (capacity is
                # step-quantized); copy the whole store, not just [:offset] --
                # a prefill of 253 into a 256 buffer followed by a 5-token
                # speculative block would otherwise try to squeeze 256 columns
                # into a 253-column slice.
                new_k[:, :, : self.keys.shape[2]] = self.keys
                new_v[:, :, : self.values.shape[2]] = self.values
            self.keys, self.values = new_k, new_v
        self.offset += keys.shape[2]
        self.keys[:, :, offset : self.offset] = keys
        self.values[:, :, offset : self.offset] = values
        return self.keys[:, :, : self.offset], self.values[:, :, : self.offset]

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        return (self.keys, self.values) if self.keys is not None else (None, None)
