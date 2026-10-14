"""LLM decoder components for LLM-ASR models, ported from mlx-lm.

Sources (mlx-lm 0.32.0, Apache 2.0, Copyright © 2023-2024 Apple Inc.):
* ``mlx_lm/models/activations.py`` -> :func:`swiglu`
* ``mlx_lm/models/rope_utils.py`` -> :func:`initialize_rope` (inference subset)

Qwen3 checkpoints carry no ``rope_scaling``, so the default path uses
MLX's built-in :class:`mlx.nn.RoPE` exactly like upstream; only the
``linear`` scaling variant is additionally ported. The yarn/llama3
variants and the distributed ``shard`` machinery of upstream are not
ported on purpose (not used by any FunASR LLM-ASR checkpoint).
"""

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
    """mlx-lm ``models/rope_utils.py::initialize_rope`` (inference subset).

    ``default`` (and absent config) and ``linear`` are supported; anything
    else raises rather than silently degrading position handling.
    """
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
