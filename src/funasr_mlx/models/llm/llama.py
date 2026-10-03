"""Llama-style decoder for GLM-ASR (text_config), ported from mlx-lm llama.

Vendored and trimmed from mlx-lm (Apache 2.0, Copyright © 2023-2024 Apple
Inc.) -- no runtime dependency: layout mirrors
mlx_lm.models.llama so weight binding is key-prefix only.

The GLM-ASR text side is Llama-shaped: 2048d, 28 layers, 16 q heads / 4 kv
heads (GQA), head_dim 128, gate/up/down MLP, RMSNorm, tied or untied head
(the checkpoint ships a separate lm_head).
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as mxnn

from .kv_cache import KVCache
from .rope import initialize_rope


class _LlamaAttention(mxnn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        d = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg.get("num_key_value_heads", self.n_heads)
        self.head_dim = cfg.get("head_dim", d // self.n_heads)
        self.scale = self.head_dim**-0.5
        self.q_proj = mxnn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = mxnn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.v_proj = mxnn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.o_proj = mxnn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.rope = initialize_rope(
            self.head_dim,
            base=cfg.get("rope_theta", 10000.0),
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[str | mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        b, t, _ = x.shape

        def split(v: mx.array, heads: int) -> mx.array:
            return mx.transpose(mx.reshape(v, (b, t, heads, self.head_dim)), (0, 2, 1, 3))

        offset = cache.offset if cache is not None else 0
        q = self.rope(split(self.q_proj(x), self.n_heads), offset=offset)
        k = self.rope(split(self.k_proj(x), self.n_kv), offset=offset)
        v = split(self.v_proj(x), self.n_kv)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = mx.reshape(mx.transpose(o, (0, 2, 1, 3)), (b, t, -1))
        return self.o_proj(o)


class _LlamaMLP(mxnn.Module):
    def __init__(self, cfg: dict[str, Any], quant: bool = False) -> None:
        super().__init__()
        d = cfg["hidden_size"]
        hidden = cfg["intermediate_size"]
        self.gate_proj = mxnn.Linear(d, hidden, bias=False)
        self.down_proj = mxnn.Linear(hidden, d, bias=False)
        self.up_proj = mxnn.Linear(d, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(mxnn.silu(self.gate_proj(x)) * self.up_proj(x))


class _LlamaDecoderLayer(mxnn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        d = cfg["hidden_size"]
        self.self_attn = _LlamaAttention(cfg)
        self.mlp = _LlamaMLP(cfg)
        self.input_layernorm = mxnn.RMSNorm(d, eps=cfg.get("rms_norm_eps", 1e-5))
        self.post_attention_layernorm = mxnn.RMSNorm(d, eps=cfg.get("rms_norm_eps", 1e-5))

    def __call__(
        self,
        h: mx.array,
        mask: Optional[str | mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        r = h
        h = self.self_attn(self.input_layernorm(h), mask, cache)
        h = r + h
        r = h
        h = self.mlp(self.post_attention_layernorm(h))
        return r + h


class LlamaModel(mxnn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.embed_tokens = mxnn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = [
            _LlamaDecoderLayer(cfg) for _ in range(cfg["num_hidden_layers"])
        ]
        self.norm = mxnn.RMSNorm(cfg["hidden_size"], eps=cfg.get("rms_norm_eps", 1e-5))

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: list[KVCache | None] | None = None,
        input_embeddings: Optional[mx.array] = None,
        mask: Optional[str | mx.array] = None,
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
            h = layer(h, mask, c)
        return self.norm(h)


class LlamaForCausalLM(mxnn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model = LlamaModel(cfg)
        if not cfg.get("tie_word_embeddings", False):
            self.lm_head = mxnn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: list[KVCache | None] | None = None,
        input_embeddings: Optional[mx.array] = None,
        mask: Optional[str | mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings, mask)
        if hasattr(self, "lm_head"):
            return self.lm_head(out)
        return self.model.embed_tokens.as_linear(out)
