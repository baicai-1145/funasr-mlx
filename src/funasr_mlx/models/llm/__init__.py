"""LLM decoder components (Qwen3 first), ported from mlx-lm 0.32.0.

No runtime dependency on mlx-lm: the model definitions here are trimmed
ports (Apache 2.0, Apple Inc.), see each module's docstring for the
upstream file it mirrors.
"""

from .kv_cache import KVCache
from .qwen3 import Qwen3ForCausalLM, Qwen3Model, Qwen3ModelArgs
from .rope import initialize_rope, swiglu

__all__ = [
    "KVCache",
    "Qwen3ForCausalLM",
    "Qwen3Model",
    "Qwen3ModelArgs",
    "initialize_rope",
    "swiglu",
]
