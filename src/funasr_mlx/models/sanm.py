"""Shared SANM (self-attention with memory) blocks for the MLX ASR ports.

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/sanm.py.md
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Iterator, Mapping, Sequence

import mlx.core as mx
from mlx import nn as mxnn

from ._layout_state import convert_tensor as _convert_tensor
from ._layout_state import layout_rule as _layout_rule


__all__ = [
    "CONV1D_LAYOUT",
    "CONV2D_LAYOUT",
    "IDENTITY_LAYOUT",
    "LAYOUT_RULES",
    "attention_blocked",
    "CheckpointKeyError",
    "Conv2dSubsampling",
    "Conv2dSubsampling2",
    "DecoderLayerSANM",
    "EncoderLayerSANM",
    "Float32LayerNorm",
    "LayerNorm",
    "ModuleStack",
    "MultiHeadedAttentionCrossAtt",
    "MultiHeadedAttentionSANM",
    "MultiHeadedAttentionSANMDecoder",
    "PositionalEncoding",
    "PositionwiseFeedForward",
    "PositionwiseFeedForwardDecoderSANM",
    "SinusoidalPositionEncoder",
    "TooShortUttError",
    "checkpoint_state_dict",
    "check_short_utt",
    "constant_pad_1d",
    "convert_checkpoint_tensor",
    "decimate_mask",
    "flatten_state_dict",
    "flatten_state_dict_with_rules",
    "load_strict",
    "make_pad_mask",
    "repeat",
    "sequence_mask",
]


def sequence_mask(lengths: Any, maxlen: int | None = None, dtype: Any = mx.float32) -> mx.array:
    # see docs/port-notes/functions.md :: sequence_mask
    lens = mx.asarray(lengths)
    if lens.ndim > 1:
        lens = lens.reshape(-1)
    if maxlen is None:
        maxlen = int(mx.max(lens).item())
    maxlen = int(maxlen)
    positions = mx.arange(0, maxlen)
    return (positions[None, :] < lens[:, None]).astype(dtype)


def make_pad_mask(lengths: Any, maxlen: int | None = None) -> mx.array:
    # see docs/port-notes/functions.md :: make_pad_mask
    return ~sequence_mask(lengths, maxlen, dtype=mx.bool_)


def constant_pad_1d(
    x: mx.array, left: int, right: int, *, axis: int = -1, value: float = 0.0
) -> mx.array:
    # see docs/port-notes/functions.md :: constant_pad_1d
    if left == 0 and right == 0:
        return x
    axis = axis % x.ndim
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (int(left), int(right))
    return mx.pad(x, pad_width, constant_values=value)


def decimate_mask(x_mask: mx.array, stride: int = 2, end_offset: int = 2) -> mx.array:
    # see docs/port-notes/functions.md :: decimate_mask
    length = x_mask.shape[-1]
    indices = [i for i in range(0, length - end_offset, stride)]
    if not indices:
        indices = [0]
    return mx.take(x_mask, mx.array(indices, dtype=mx.int32), axis=-1)


IDENTITY_LAYOUT = "identity"
CONV1D_LAYOUT = "conv1d_torch_out_ink"
CONV2D_LAYOUT = "conv2d_torch_out_ink"

_RULE_NAMES = {"conv1d": CONV1D_LAYOUT, "conv2d": CONV2D_LAYOUT, "identity": IDENTITY_LAYOUT}
_IDENTITY_RULES = (IDENTITY_LAYOUT, None)

_LAYOUT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(^|\.)fsmn_block\.weight$"), CONV1D_LAYOUT),
    (re.compile(r"(^|\.)cif_conv1d\.weight$"), CONV1D_LAYOUT),
    (re.compile(r"(^|\.)conv\.\d+\.weight$"), CONV2D_LAYOUT),
)

LAYOUT_RULES: Mapping[str, str] = {
    "fsmn_block.weight": CONV1D_LAYOUT,
    "cif_conv1d.weight": CONV1D_LAYOUT,
    "conv.<n>.weight": CONV2D_LAYOUT,
}


class CheckpointKeyError(KeyError):
    """Raised by :func:`load_strict` when a checkpoint does not match a module.

    The message always lists the offending keys so a failed load is never
    silently downgraded to a partial load.
    """


def _module_layout_rule(module: mxnn.Module) -> str:
    return _layout_rule(module, _RULE_NAMES)


def convert_checkpoint_tensor(name: str, array: mx.array, rule: str | None = None) -> mx.array:
    # see docs/port-notes/functions.md :: convert_checkpoint_tensor
    if rule is None:
        rule = _rule_for_name(name)
    out = _convert_tensor(name, array, rule, (CONV1D_LAYOUT, CONV2D_LAYOUT))
    if out is array and rule not in _IDENTITY_RULES:
        raise ValueError(f"unknown layout rule {rule!r} for parameter {name!r}")
    return out


def _rule_for_name(name: str) -> str:
    for pattern, rule in _LAYOUT_RULES:
        if pattern.search(name):
            return rule
    return IDENTITY_LAYOUT


def flatten_state_dict(module: mxnn.Module, prefix: str = "") -> dict[str, mx.array]:
    # see docs/port-notes/functions.md :: flatten_state_dict
    return _flatten(module, prefix)[0]


def flatten_state_dict_with_rules(
    module: mxnn.Module, prefix: str = ""
) -> tuple[dict[str, mx.array], dict[str, str]]:
    # see docs/port-notes/functions.md :: flatten_state_dict_with_rules
    return _flatten(module, prefix)


def _flatten(module: mxnn.Module, prefix: str = "") -> tuple[dict[str, mx.array], dict[str, str]]:
    tensors: dict[str, mx.array] = {}
    rules: dict[str, str] = {}
    _flatten_into(module, prefix, tensors, rules)
    return tensors, rules


def attention_blocked(mask: mx.array) -> mx.array:
    """Boolean ``True`` where attention is blocked, for any mask dtype.

    Full notes: docs/port-notes/functions.md :: attention_blocked
    """
    if mask.dtype == mx.bool_:
        blocked = mx.logical_not(mask)
    else:
        blocked = mask == 0
    return mx.expand_dims(blocked, 1)


def _flatten_into(module: Any, prefix: str, tensors: dict, rules: dict) -> None:
    """Collect tensors from a module tree.

    Full notes: docs/port-notes/functions.md :: _flatten_into
    """
    for key, value in dict.items(module):
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, mxnn.Module):
            _flatten_into(value, name, tensors, rules)
        elif isinstance(value, mx.array):
            tensors[name] = value
            rules[name] = _module_layout_rule(module) if key == "weight" else IDENTITY_LAYOUT
        elif isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, mx.array):
                    tensors[f"{name}.{sub_key}"] = sub_value
                    rules[f"{name}.{sub_key}"] = _module_layout_rule(module)
                elif isinstance(sub_value, mxnn.Module):
                    _flatten_into(sub_value, f"{name}.{sub_key}", tensors, rules)


def checkpoint_state_dict(module: mxnn.Module) -> dict[str, mx.array]:
    # see docs/port-notes/functions.md :: checkpoint_state_dict
    return flatten_state_dict(module)


def _resolve_child(module: mxnn.Module, key: str) -> Any:
    if key in dict.keys(module):
        return dict.__getitem__(module, key)
    return getattr(module, key)


def _assign(module: mxnn.Module, key_parts: Sequence[str], array: mx.array) -> None:
    node: Any = module
    for part in key_parts[:-1]:
        node = _resolve_child(node, part)
    setattr(node, key_parts[-1], array)


def load_strict(
    module: mxnn.Module,
    tensors: Mapping[str, Any],
    *,
    prefix: str = "",
    convert: bool = False,
    dtype: Any | None = None,
) -> list[str]:
    """Load ``tensors`` into ``module``, refusing any key mismatch.

    Full notes: docs/port-notes/functions.md :: load_strict
    """
    expected, rules = flatten_state_dict_with_rules(module)
    expected_names = {f"{prefix}{k}" for k in expected}
    incoming = dict(tensors)

    missing = sorted(expected_names - set(incoming))
    unexpected = sorted(set(incoming) - expected_names)
    if missing or unexpected:
        raise CheckpointKeyError(
            "checkpoint/module key mismatch: "
            f"{len(missing)} missing, {len(unexpected)} unexpected. "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    loaded: list[str] = []
    for name in sorted(incoming):
        short = name[len(prefix) :]
        array = mx.asarray(incoming[name])
        if convert:
            array = convert_checkpoint_tensor(short, array, rules.get(short))
        if dtype is not None:
            array = array.astype(dtype)
        target = expected[short]
        if tuple(array.shape) != tuple(target.shape):
            raise ValueError(
                f"shape mismatch for {name!r}: checkpoint {tuple(array.shape)} "
                f"vs model {tuple(target.shape)}"
            )
        _assign(module, short.split("."), array)
        loaded.append(name)
    return loaded


class ModuleStack(mxnn.Module):
    """A stack of sibling modules with PyTorch-compatible parameter names.

    Full notes: docs/port-notes/functions.md :: ModuleStack
    """

    def __init__(self, blocks: Sequence[mxnn.Module]) -> None:
        super().__init__()
        for index, block in enumerate(blocks):
            setattr(self, str(index), block)
        self._length = len(blocks)

    def __len__(self) -> int:
        return dict.__len__(self)

    def __iter__(self) -> Iterator[mxnn.Module]:
        for index in range(dict.__len__(self)):
            yield dict.__getitem__(self, str(index))

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            if key < 0:
                key += dict.__len__(self)
            key = str(key)
        return dict.__getitem__(self, key)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # see docs/port-notes/functions.md :: __call__
        if not args:
            raise TypeError(f"{type(self).__name__}.__call__() requires at least one argument")
        head, *rest = args
        for block in self:
            head = block(head, *rest, **kwargs)
        return head


def repeat(num_blocks: int, factory: Callable[[int], mxnn.Module]) -> ModuleStack:
    # see docs/port-notes/functions.md :: repeat
    if num_blocks < 0:
        raise ValueError(f"num_blocks must be >= 0, got {num_blocks}")
    return ModuleStack([factory(index) for index in range(num_blocks)])


class LayerNorm(mxnn.LayerNorm):
    # see docs/port-notes/functions.md :: LayerNorm

    def __init__(self, num_features: int, dim: int = -1, eps: float = 1e-12) -> None:
        super().__init__(num_features, eps=eps)
        self._dim = dim


class Float32LayerNorm(mxnn.LayerNorm):
    # see docs/port-notes/functions.md :: Float32LayerNorm

    def __init__(self, num_features: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__(num_features, eps=eps)
        self._dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        out_dtype = x.dtype
        return super().__call__(x.astype(mx.float32)).astype(out_dtype)


class PositionalEncoding(mxnn.Module):
    """``funasr.models.transformer.embedding.PositionalEncoding``.

    Full notes: docs/port-notes/functions.md :: PositionalEncoding
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float = 0.1,
        max_len: int = 5000,
        reverse: bool = False,
    ) -> None:
        super().__init__()
        self._d_model = int(d_model)
        self._max_len = int(max_len)
        self._reverse = bool(reverse)
        self._xscale = math.sqrt(float(d_model))

    def _table(self, length: int, dtype: Any) -> mx.array:
        if self._reverse:
            positions = mx.arange(length - 1, -1, -1).astype(mx.float32)[:, None]
        else:
            positions = mx.arange(0, length).astype(mx.float32)[:, None]
        div_term = mx.exp(
            mx.arange(0, self._d_model, 2).astype(mx.float32) * -(math.log(10000.0) / self._d_model)
        )
        phase = positions * div_term  # (length, d_model // 2)
        return (
            mx.stack([mx.sin(phase), mx.cos(phase)], axis=-1)
            .reshape(length, self._d_model)
            .astype(dtype)
        )

    def __call__(self, x: mx.array) -> mx.array:
        return x * self._xscale + self._table(x.shape[1], x.dtype)[None, : x.shape[1]]


class SinusoidalPositionEncoder(mxnn.Module):
    # see docs/port-notes/functions.md :: SinusoidalPositionEncoder

    def __init__(self, d_model: int = 80, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self._d_model = int(d_model)

    def encode(self, positions: mx.array, depth: int, dtype: Any = mx.float32) -> mx.array:
        log_timescale_increment = math.log(10000.0) / (depth / 2 - 1)
        inv_timescales = mx.exp(
            mx.arange(0, math.ceil(depth / 2)).astype(mx.float32) * (-log_timescale_increment)
        )
        scaled_time = positions.astype(mx.float32).reshape(1, -1, 1) * inv_timescales.reshape(
            1, 1, -1
        )
        encoding = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=2)
        return encoding.astype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        timesteps = x.shape[1]
        positions = mx.arange(1, timesteps + 1)[None, :]
        position_encoding = self.encode(positions, x.shape[2], x.dtype)
        return x + position_encoding


class PositionwiseFeedForward(mxnn.Module):
    # see docs/port-notes/functions.md :: PositionwiseFeedForward

    def __init__(self, idim: int, hidden_units: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.w_1 = mxnn.Linear(idim, hidden_units)
        self.w_2 = mxnn.Linear(hidden_units, idim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_2(mx.maximum(self.w_1(x), 0.0))


class PositionwiseFeedForwardDecoderSANM(mxnn.Module):
    # see docs/port-notes/functions.md :: PositionwiseFeedForwardDecoderSANM

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float = 0.1,
        adim: int | None = None,
    ) -> None:
        super().__init__()
        self.w_1 = mxnn.Linear(idim, hidden_units)
        self.w_2 = mxnn.Linear(hidden_units, idim if adim is None else adim, bias=False)
        self.norm = LayerNorm(hidden_units)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_2(self.norm(mx.maximum(self.w_1(x), 0.0)))


class MultiHeadedAttentionSANM(mxnn.Module):
    # see docs/port-notes/functions.md :: MultiHeadedAttentionSANM

    def __init__(
        self,
        n_head: int,
        in_feat: int,
        n_feat: int,
        dropout_rate: float = 0.0,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
    ) -> None:
        super().__init__()
        if n_feat % n_head != 0:
            raise ValueError(f"n_feat ({n_feat}) must be divisible by n_head ({n_head})")
        self._h = n_head
        self._d_k = n_feat // n_head
        self._n_feat = n_feat
        self.linear_q_k_v = mxnn.Linear(in_feat, n_feat * 3)
        self.linear_out = mxnn.Linear(n_feat, n_feat)
        self.fsmn_block = mxnn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        self._left_padding = int(left_padding)
        self._right_padding = int(kernel_size - 1 - left_padding)
        self._kernel_size = int(kernel_size)

    def _split_heads(self, x: mx.array) -> mx.array:
        # see docs/port-notes/functions.md :: _split_heads
        b, t, _ = x.shape
        return mx.transpose(mx.reshape(x, (b, t, self._h, self._d_k)), (0, 2, 1, 3))

    def forward_qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        q_k_v = self.linear_q_k_v(x)
        q, k, v = mx.split(q_k_v, 3, axis=-1)
        return self._split_heads(q), self._split_heads(k), self._split_heads(v), v

    def forward_fsmn(self, inputs: mx.array, mask: mx.array | None) -> mx.array:
        b = inputs.shape[0]
        if mask is not None:
            mask = mx.reshape(mask, (b, -1, 1))
            inputs = inputs * mask
        x = constant_pad_1d(inputs, self._left_padding, self._right_padding, axis=1)
        x = self.fsmn_block(x)
        x = x + inputs
        if mask is not None:
            x = x * mask
        return x

    def forward_attention(self, value: mx.array, scores: mx.array, mask: mx.array | None):
        if mask is not None:
            blocked = attention_blocked(mask)
            neg_inf = mx.array(-float("inf"), scores.dtype)
            scores = mx.where(blocked, neg_inf, scores)
            attn = mx.softmax(scores, axis=-1)
            attn = mx.where(blocked, mx.array(0.0, attn.dtype), attn)
        else:
            attn = mx.softmax(scores, axis=-1)
        x = mx.matmul(attn, value)
        b, _, t, _ = x.shape
        x = mx.reshape(mx.transpose(x, (0, 2, 1, 3)), (b, t, self._h * self._d_k))
        return self.linear_out(x)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        q_k_v = self.linear_q_k_v(x)
        q, k, v = mx.split(q_k_v, 3, axis=-1)
        q_h, k_h, v_h, v = self._split_heads(q), self._split_heads(k), self._split_heads(v), v
        fsmn_memory = self.forward_fsmn(v, mask)
        # see docs/port-notes/comments.md :: sanm.py:484
        if mask is None:
            attn = mx.fast.scaled_dot_product_attention(q_h, k_h, v_h, scale=self._d_k**-0.5)
            b, _, t, _ = attn.shape
            x = mx.reshape(mx.transpose(attn, (0, 2, 1, 3)), (b, t, self._h * self._d_k))
            return self.linear_out(x) + fsmn_memory
        q_h = q_h * (self._d_k**-0.5)
        scores = mx.matmul(q_h, mx.swapaxes(k_h, -1, -2))
        return self.forward_attention(v_h, scores, mask) + fsmn_memory


class MultiHeadedAttentionSANMDecoder(mxnn.Module):
    # see docs/port-notes/functions.md :: MultiHeadedAttentionSANMDecoder

    def __init__(
        self, n_feat: int, dropout_rate: float = 0.0, kernel_size: int = 11, sanm_shfit: int = 0
    ) -> None:
        super().__init__()
        self.fsmn_block = mxnn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        self._left_padding = int(left_padding)
        self._right_padding = int(kernel_size - 1 - left_padding)
        self._kernel_size = int(kernel_size)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        # see docs/port-notes/functions.md :: __call__
        b = inputs.shape[0]
        t = inputs.shape[1]
        if mask is not None:
            mask = mx.reshape(mask, (b, -1, 1))
            inputs = inputs * mask

        if cache is None:
            x = constant_pad_1d(inputs, self._left_padding, self._right_padding, axis=1)
        else:
            x = mx.concatenate([cache[:, 1:, :], inputs], axis=1)
            x = x[:, -(self._kernel_size + t - 1) :, :]
        new_cache = x
        x = self.fsmn_block(x)
        x = x + inputs
        if mask is not None:
            x = x * mask
        return x, new_cache


class MultiHeadedAttentionCrossAtt(mxnn.Module):
    # see docs/port-notes/functions.md :: MultiHeadedAttentionCrossAtt

    def __init__(
        self,
        n_head: int,
        n_feat: int,
        dropout_rate: float = 0.0,
        encoder_output_size: int | None = None,
        *,
        lora_list: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if lora_list is not None:
            raise NotImplementedError(
                "LoRA variants of the SANM blocks are not ported; drop lora_list "
                "and load the dense checkpoint instead"
            )
        if n_feat % n_head != 0:
            raise ValueError(f"n_feat ({n_feat}) must be divisible by n_head ({n_head})")
        self._h = n_head
        self._d_k = n_feat // n_head
        kv_in = n_feat if encoder_output_size is None else encoder_output_size
        self.linear_q = mxnn.Linear(n_feat, n_feat)
        self.linear_k_v = mxnn.Linear(kv_in, n_feat * 2)
        self.linear_out = mxnn.Linear(n_feat, n_feat)

    def _split_heads(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape
        return mx.transpose(mx.reshape(x, (b, t, self._h, self._d_k)), (0, 2, 1, 3))

    def forward_qkv(self, x: mx.array, memory: mx.array):
        q_h = self._split_heads(self.linear_q(x))
        k_v = self.linear_k_v(memory)
        k, v = mx.split(k_v, 2, axis=-1)
        return q_h, self._split_heads(k), self._split_heads(v)

    def forward_attention(
        self,
        value: mx.array,
        scores: mx.array,
        mask: mx.array | None,
        ret_attn: bool = False,
    ):
        if mask is not None:
            blocked = attention_blocked(mask)
            neg_inf = mx.array(-float("inf"), scores.dtype)
            scores = mx.where(blocked, neg_inf, scores)
            attn = mx.softmax(scores, axis=-1)
            attn = mx.where(blocked, mx.array(0.0, attn.dtype), attn)
        else:
            attn = mx.softmax(scores, axis=-1)
        x = mx.matmul(attn, value)
        b, _, t, _ = x.shape
        x = mx.reshape(mx.transpose(x, (0, 2, 1, 3)), (b, t, self._h * self._d_k))
        out = self.linear_out(x)
        return (out, attn) if ret_attn else out

    def __call__(
        self,
        x: mx.array,
        memory: mx.array,
        memory_mask: mx.array | None = None,
        ret_attn: bool = False,
    ):
        q_h, k_h, v_h = self.forward_qkv(x, memory)
        q_h = q_h * (self._d_k**-0.5)
        scores = mx.matmul(q_h, mx.swapaxes(k_h, -1, -2))
        return self.forward_attention(v_h, scores, memory_mask, ret_attn)


class EncoderLayerSANM(mxnn.Module):
    # see docs/port-notes/functions.md :: EncoderLayerSANM

    def __init__(
        self,
        in_size: int,
        size: int,
        self_attn: mxnn.Module,
        feed_forward: mxnn.Module,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self._in_size = in_size
        self._size = size
        self._normalize_before = normalize_before
        self._concat_after = concat_after
        if concat_after:
            self.concat_linear = mxnn.Linear(size + size, size)

    def __call__(self, x: mx.array, mask: mx.array | None):
        residual = x
        if self._normalize_before:
            x = self.norm1(x)

        if self._concat_after:
            x_concat = mx.concatenate([x, self.self_attn(x, mask)], axis=-1)
            if self._in_size == self._size:
                x = residual + self.concat_linear(x_concat)
            else:
                x = self.concat_linear(x_concat)
        else:
            attn_out = self.self_attn(x, mask)
            x = residual + attn_out if self._in_size == self._size else attn_out
        if not self._normalize_before:
            x = self.norm1(x)

        residual = x
        if self._normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self._normalize_before:
            x = self.norm2(x)
        return x, mask


class DecoderLayerSANM(mxnn.Module):
    # see docs/port-notes/functions.md :: DecoderLayerSANM

    def __init__(
        self,
        size: int,
        self_attn: mxnn.Module | None,
        src_attn: mxnn.Module | None,
        feed_forward: mxnn.Module,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
    ) -> None:
        super().__init__()
        self._size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self._normalize_before = normalize_before
        self._concat_after = concat_after
        if self_attn is not None:
            self.norm2 = LayerNorm(size)
        if src_attn is not None:
            self.norm3 = LayerNorm(size)

    def __call__(self, tgt, tgt_mask, memory, memory_mask=None):
        residual = tgt
        if self._normalize_before:
            tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn is not None:
            if self._normalize_before:
                tgt = self.norm2(tgt)
            attn_out, _ = self.self_attn(tgt, tgt_mask)
            x = residual + attn_out

        if self.src_attn is not None:
            residual = x
            if self._normalize_before:
                x = self.norm3(x)
            x_src_attn = self.src_attn(x, memory, memory_mask, ret_attn=False)
            x = residual + x_src_attn
        return x, tgt_mask, memory, memory_mask


class TooShortUttError(Exception):
    # see docs/port-notes/functions.md :: TooShortUttError

    def __init__(self, message: str, actual_size: int, limit: int) -> None:
        super().__init__(message)
        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(embed: mxnn.Module, size: int) -> tuple[bool, int]:
    # see docs/port-notes/functions.md :: check_short_utt
    if isinstance(embed, Conv2dSubsampling2) and size < 3:
        return True, 3
    if isinstance(embed, Conv2dSubsampling) and size < 7:
        return True, 7
    return False, -1


class Conv2dSubsampling(mxnn.Module):
    # see docs/port-notes/functions.md :: Conv2dSubsampling

    def __init__(self, idim: int, odim: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.conv = ModuleStack(
            [
                mxnn.Conv2d(1, odim, 3, stride=2),
                mxnn.ReLU(),
                mxnn.Conv2d(odim, odim, 3, stride=2),
                mxnn.ReLU(),
            ]
        )
        self.out = ModuleStack(
            [
                mxnn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
                PositionalEncoding(odim, dropout_rate),
            ]
        )
        self._idim = idim
        self._odim = odim

    def __call__(self, x: mx.array, x_mask: mx.array | None):
        x = mx.expand_dims(x, -1)  # (B, T, F, C=1), MLX NHWC
        x = self.conv(x)
        b, t, f, c = x.shape
        x = mx.reshape(mx.transpose(x, (0, 1, 3, 2)), (b, t, c * f))
        x = self.out(x)
        if x_mask is None:
            return x, None
        return x, decimate_mask(decimate_mask(x_mask, 2, 2), 2, 2)


class Conv2dSubsampling2(mxnn.Module):
    # see docs/port-notes/functions.md :: Conv2dSubsampling2

    def __init__(self, idim: int, odim: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.conv = ModuleStack(
            [
                mxnn.Conv2d(1, odim, 3, stride=2),
                mxnn.ReLU(),
                mxnn.Conv2d(odim, odim, 3, stride=1),
                mxnn.ReLU(),
            ]
        )
        self.out = ModuleStack(
            [
                mxnn.Linear(odim * ((idim - 1) // 2 - 2), odim),
                PositionalEncoding(odim, dropout_rate),
            ]
        )
        self._idim = idim
        self._odim = odim

    def __call__(self, x: mx.array, x_mask: mx.array | None):
        x = mx.expand_dims(x, -1)
        x = self.conv(x)
        b, t, f, c = x.shape
        x = mx.reshape(mx.transpose(x, (0, 1, 3, 2)), (b, t, c * f))
        x = self.out(x)
        if x_mask is None:
            return x, None
        return x, decimate_mask(decimate_mask(x_mask, 2, 2), 1, 2)
