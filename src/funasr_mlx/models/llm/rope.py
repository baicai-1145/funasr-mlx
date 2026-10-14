"""Port notes: docs/port-notes/rope.py.md"""

from __future__ import annotations

from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn

__all__ = ["swiglu", "initialize_rope"]


def swiglu(gate: mx.array, x: mx.array) -> mx.array:
    return nn.silu(gate) * x


def initialize_rope(
    dims: int,
    base: float = 10000.0,
    traditional: bool = False,
    scaling_config: Optional[dict[str, Union[float, str]]] = None,
    max_position_embeddings: Optional[int] = None,  # noqa: ARG001 (upstream signature)
) -> nn.RoPE:
    # see docs/port-notes/functions.md :: initialize_rope
    if scaling_config is None:
        return nn.RoPE(dims, traditional=traditional, base=base, scale=1.0)
    rope_type = scaling_config.get("type") or scaling_config.get("rope_type", "default")
    if rope_type == "default":
        return nn.RoPE(dims, traditional=traditional, base=base, scale=1.0)
    if rope_type == "linear":
        scale = 1.0 / float(scaling_config["factor"])
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)
    raise NotImplementedError(
        f"RoPE scaling {rope_type!r} is not ported (supported: default, linear)"
    )
