# rope.py — port notes

## module

```
LLM decoder components for LLM-ASR models, ported from mlx-lm.

Sources (mlx-lm 0.32.0, Apache 2.0, Copyright © 2023-2024 Apple Inc.):
* ``mlx_lm/models/activations.py`` -> :func:`swiglu`
* ``mlx_lm/models/rope_utils.py`` -> :func:`initialize_rope` (inference subset)

Qwen3 checkpoints carry no ``rope_scaling``, so the default path uses
MLX's built-in :class:`mlx.nn.RoPE` exactly like upstream; only the
``linear`` scaling variant is additionally ported. The yarn/llama3
variants and the distributed ``shard`` machinery of upstream are not
ported on purpose (not used by any FunASR LLM-ASR checkpoint).

```

## module

```
LLM decoder components for LLM-ASR models, ported from mlx-lm.

Sources (mlx-lm 0.32.0, Apache 2.0, Copyright © 2023-2024 Apple Inc.):
* ``mlx_lm/models/activations.py`` -> :func:`swiglu`
* ``mlx_lm/models/rope_utils.py`` -> :func:`initialize_rope` (inference subset)

Qwen3 checkpoints carry no ``rope_scaling``, so the default path uses
MLX's built-in :class:`mlx.nn.RoPE` exactly like upstream; only the
``linear`` scaling variant is additionally ported. The yarn/llama3
variants and the distributed ``shard`` machinery of upstream are not
ported on purpose (not used by any FunASR LLM-ASR checkpoint).

```

## module

```
LLM decoder components for LLM-ASR models, ported from mlx-lm.

Sources (mlx-lm 0.32.0, Apache 2.0, Copyright © 2023-2024 Apple Inc.):
* ``mlx_lm/models/activations.py`` -> :func:`swiglu`
* ``mlx_lm/models/rope_utils.py`` -> :func:`initialize_rope` (inference subset)

Qwen3 checkpoints carry no ``rope_scaling``, so the default path uses
MLX's built-in :class:`mlx.nn.RoPE` exactly like upstream; only the
``linear`` scaling variant is additionally ported. The yarn/llama3
variants and the distributed ``shard`` machinery of upstream are not
ported on purpose (not used by any FunASR LLM-ASR checkpoint).

```
