"""Port notes: docs/port-notes/kv_cache.py.md"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["KVCache"]


class KVCache:
    # see docs/port-notes/functions.md :: KVCache

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self.step = 256

    def reserve(self, positions: int, like: "KVCache | None" = None) -> None:
        # see docs/port-notes/functions.md :: reserve
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
