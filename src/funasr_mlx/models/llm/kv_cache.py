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
    """mlx-lm ``KVCache``: append-only keys/values with an offset cursor."""

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self.step = 256

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
                new_k[:, :, :offset] = self.keys
                new_v[:, :, :offset] = self.values
            self.keys, self.values = new_k, new_v
        self.offset += keys.shape[2]
        self.keys[:, :, offset : self.offset] = keys
        self.values[:, :, offset : self.offset] = values
        return self.keys[:, :, : self.offset], self.values[:, :, : self.offset]

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        return (self.keys, self.values) if self.keys is not None else (None, None)
