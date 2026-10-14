"""Port notes: docs/port-notes/qwen3.py.md"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .kv_cache import KVCache
from .rope import initialize_rope, swiglu

__all__ = ["Qwen3ModelArgs", "Qwen3Model"]


@dataclass
class Qwen3ModelArgs:

    model_type: str = "qwen3"
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rope_theta: float = 1000000.0
    head_dim: int = 64
    tie_word_embeddings: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3ModelArgs":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class Qwen3Attention(nn.Module):
    def __init__(self, args: Qwen3ModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Qwen3MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen3TransformerBlock(nn.Module):
    def __init__(self, args: Qwen3ModelArgs) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(args)
        self.mlp = Qwen3MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


class Qwen3Model(nn.Module):

    def __init__(self, args: Qwen3ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Qwen3TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array | None = None,
        cache: list[KVCache | None] | None = None,
        input_embeddings: mx.array | None = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
            if mask is None and h.shape[1] > 1:
                mask = "causal"
        for layer, c in zip(self.layers, cache):
            x = layer.input_layernorm(h)
            r = layer.self_attn(x, mask, c)
            h = h + r
            x = layer.post_attention_layernorm(h)
            h = h + layer.mlp(x)
        return self.norm(h)


class Qwen3ForCausalLM(nn.Module):
    # see docs/port-notes/functions.md :: Qwen3ForCausalLM

    def __init__(self, args: Qwen3ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model = Qwen3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array | None = None,
        cache: list[KVCache | None] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights
