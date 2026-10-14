# qwen3.py — port notes

## module

```
Qwen3 decoder, ported from mlx-lm for LLM-ASR.

Source: mlx-lm 0.32.0 ``mlx_lm/models/qwen3.py`` (Apache 2.0,
Copyright © 2023-2024 Apple Inc.). Parameter names match the HF
``Qwen3ForCausalLM`` state dict exactly, so Fun-ASR-Nano checkpoints
(whose ``llm.*`` subtree is that state dict) load unchanged.

Dropped from upstream: ``shard`` (multi-GPU), the distributed
``shard_linear`` import, and ``BaseModelArgs`` plumbing (a plain
dataclass with ``from_dict`` serves the same role without the mixin).
The ``input_embeddings`` argument of the model call is the hook the
audio adaptor uses to splice speech embeddings into the token stream --
exactly how FunASR's ``FunASRNano`` writes ``inputs_embeds``.

```

## module

```
Qwen3 decoder, ported from mlx-lm for LLM-ASR.

Source: mlx-lm 0.32.0 ``mlx_lm/models/qwen3.py`` (Apache 2.0,
Copyright © 2023-2024 Apple Inc.). Parameter names match the HF
``Qwen3ForCausalLM`` state dict exactly, so Fun-ASR-Nano checkpoints
(whose ``llm.*`` subtree is that state dict) load unchanged.

Dropped from upstream: ``shard`` (multi-GPU), the distributed
``shard_linear`` import, and ``BaseModelArgs`` plumbing (a plain
dataclass with ``from_dict`` serves the same role without the mixin).
The ``input_embeddings`` argument of the model call is the hook the
audio adaptor uses to splice speech embeddings into the token stream --
exactly how FunASR's ``FunASRNano`` writes ``inputs_embeds``.

```

## module

```
Qwen3 decoder, ported from mlx-lm for LLM-ASR.

Source: mlx-lm 0.32.0 ``mlx_lm/models/qwen3.py`` (Apache 2.0,
Copyright © 2023-2024 Apple Inc.). Parameter names match the HF
``Qwen3ForCausalLM`` state dict exactly, so Fun-ASR-Nano checkpoints
(whose ``llm.*`` subtree is that state dict) load unchanged.

Dropped from upstream: ``shard`` (multi-GPU), the distributed
``shard_linear`` import, and ``BaseModelArgs`` plumbing (a plain
dataclass with ``from_dict`` serves the same role without the mixin).
The ``input_embeddings`` argument of the model call is the hook the
audio adaptor uses to splice speech embeddings into the token stream --
exactly how FunASR's ``FunASRNano`` writes ``inputs_embeds``.

```
