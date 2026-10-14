"""Native MLX inference port of ``emotion2vec_plus_large``.

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/emotion2vec.py.md
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn

from ._strict_load import _require
from mlx.utils import tree_flatten, tree_unflatten

__all__ = [
    "INFERENCE_ONLY_PREFIXES",
    "AltAttention",
    "AltBlock",
    "AudioEncoder",
    "BlockEncoder",
    "Emotion2VecConfig",
    "Emotion2VecModel",
    "EmotionResult",
    "Fp32LayerNorm",
    "ModuleListStack",
    "OrderedModules",
    "PositionalEncoder",
    "SequentialStack",
    "TransposeLast",
    "CheckpointKeyError",
    "conv_feature_output_length",
    "convert_checkpoint_tensors",
    "is_conv_weight",
    "convert_checkpoint_tensors",
    "flat_parameter_names",
    "get_alibi",
    "get_slopes",
    "load_state_dict_strict",
    "parse_feature_encoder_spec",
    "preprocess_waveform",
]

#: Prefixes that exist in the published checkpoint but carry no inference
#: parameters.  They are dropped by :func:`convert_checkpoint_tensors` only when
#: this exact prefix list matches, and the dropped names are returned so the
#: caller can log them.  Any *other* unexpected key is an error.
INFERENCE_ONLY_PREFIXES: tuple[str, ...] = ("modality_encoders.AUDIO.decoder.",)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _as_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float, got {type(value).__name__}")
    return float(value)


def get_slopes(n: int) -> list[float]:

    def power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if math.log2(n).is_integer():
        return power_of_2(n)
    closest = 2 ** math.floor(math.log2(n))
    return power_of_2(closest) + get_slopes(2 * closest)[0::2][: n - closest]


def get_alibi(max_positions: int, attention_heads: int) -> mx.array:
    """``get_alibi(max_positions, attention_heads, dims=1)``.

    1-D manhattan ALiBi: ``slopes[h] * -|i - j|`` with shape
    ``(heads, max_positions, max_positions)``.
    """
    slopes = mx.array(get_slopes(attention_heads), dtype=mx.float32)
    pos = mx.arange(max_positions, dtype=mx.float32)
    pos_bias = -mx.abs(pos[:, None] - pos[None, :])  # (T, T)
    return slopes.reshape(-1, 1, 1) * pos_bias[None, :, :]


# --------------------------------------------------------------------------- #
# containers that preserve PyTorch Sequential / ModuleList key indices
# --------------------------------------------------------------------------- #
class OrderedModules(nn.Module):
    """An ``nn.Sequential`` whose children keep their integer index in the key.

    ``nn.Module`` is a dict subclass and ``tree_flatten`` walks ``.items()``, so
    registering a child under the *string* ``"2"`` reproduces PyTorch's
    ``...2.1.weight`` naming exactly.  A plain list would also be captured by
    ``Module.__setattr__`` and would pollute the key space, hence the tuple.
    """

    def __init__(self, modules: Mapping[str, nn.Module]) -> None:
        super().__init__()
        self._order = tuple(str(name) for name in modules)
        for name, module in modules.items():
            setattr(self, str(name), module)

    def __call__(self, x: mx.array) -> mx.array:
        for name in self._order:
            x = getattr(self, name)(x)
        return x

    def __len__(self) -> int:
        return len(self._order)

    def __getitem__(self, index: int | str) -> nn.Module:
        """Fetch by Sequential position (``self[0]``) **or** by raw key (``self["0"]``).

        Both spellings are used: ``__call__`` walks ``_order`` positionally while
        :class:`PositionalBlock` needs named access so it can reorder the
        transposes around the channel-last conv.
        """
        if isinstance(index, str):
            return getattr(self, index)
        return getattr(self, self._order[index])


class SequentialStack(OrderedModules):
    """``nn.Sequential`` with automatic ``0..n-1`` indices."""


class ModuleListStack(nn.Module):
    """``nn.ModuleList``; children registered under their decimal index."""

    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__()
        self._length = len(modules)
        for index, module in enumerate(modules):
            setattr(self, str(index), module)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for index in range(self._length):
            yield getattr(self, str(index))

    def __getitem__(self, index: int | str) -> nn.Module:
        """Fetch by position (``self[0]``) or by the mapping key (``self["0"]``).

        ``nn.Module.update`` / ``tree_unflatten`` hand back the *string* keys that
        were registered here, so a positional-only ``index < 0`` comparison raises
        ``TypeError: '<' not supported between instances of 'str' and 'int'``.
        """
        if isinstance(index, str):
            if not index.lstrip("-").isdigit():
                raise KeyError(index)
            position = int(index)
        else:
            position = index
        if position < 0:
            position += self._length
        if not 0 <= position < self._length:
            raise IndexError(index)
        return getattr(self, str(position))


class TransposeLast(nn.Module):
    """``fairseq.modules.TransposeLast``: swap the last two axes."""

    def __call__(self, x: mx.array) -> mx.array:
        return mx.swapaxes(x, -2, -1)


class Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


# --------------------------------------------------------------------------- #
# norms and activations
# --------------------------------------------------------------------------- #
class Fp32LayerNorm(nn.LayerNorm):
    """``fairseq.layers.fp32_group_norm.Fp32LayerNorm``.

    A ``nn.LayerNorm`` that upcasts to float32, normalises and casts back, so
    the reduction never happens in half precision.
    """

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        return super().__call__(x.astype(mx.float32)).astype(dtype)


class Gelu(nn.Module):
    """``nn.GELU()`` with the default (exact, erf-based) approximation."""

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(x)


class Mlp(nn.Module):
    """``fairseq`` ``Mlp``: ``fc1 -> act -> fc2`` with dropout around."""

    def __init__(self, in_features: int, hidden_features: int, act_layer: Callable) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


# --------------------------------------------------------------------------- #
# ConvFeatureExtractionModel (fairseq v0.12.2 wav2vec2.py:819-895)
# --------------------------------------------------------------------------- #
def parse_feature_encoder_spec(spec: str) -> list[tuple[int, int, int]]:
    """Parse ``feature_encoder_spec`` with ``ast`` -- ``eval`` is never used.

    Accepts only literal arithmetic over tuples of ints, i.e. ``+``, ``*`` and
    ``[a, b, c]``.  Anything else (names, calls, attributes, comprehensions)
    raises.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("feature_encoder_spec must be a non-empty string")
    try:
        tree = ast.parse(spec.strip(), mode="eval")
    except SyntaxError as exc:  # pragma: no cover - message only
        raise ValueError(f"feature_encoder_spec is not parseable: {spec!r}") from exc

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise ValueError(f"only int literals are allowed, got {node.value!r}")
            return node.value
        if isinstance(node, ast.Tuple | ast.List):
            return tuple(evaluate(item) for item in node.elts)
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Mult):
                return left * right
            raise ValueError(
                f"unsupported operator in feature_encoder_spec: {type(node.op).__name__}"
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        raise ValueError(f"unsupported expression in feature_encoder_spec: {type(node).__name__}")

    layers = evaluate(tree)
    if not isinstance(layers, tuple) or not layers:
        raise ValueError("feature_encoder_spec must evaluate to a non-empty sequence")
    resolved: list[tuple[int, int, int]] = []
    for entry in layers:
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise ValueError(f"each layer must be a (dim, kernel, stride) triple, got {entry!r}")
        dim, kernel, stride = entry
        for name, value in (("dim", dim), ("kernel", kernel), ("stride", stride)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an int literal, got {value!r}")
        if dim < 1 or kernel < 1 or stride < 1:
            raise ValueError(f"dim/kernel/stride must be >= 1, got {entry!r}")
        resolved.append((int(dim), int(kernel), int(stride)))
    return resolved


def conv_feature_output_length(samples: int, layers: Sequence[tuple[int, int, int]]) -> int:
    length = int(samples)
    for _, kernel, stride in layers:
        length = math.floor((length - kernel) / stride + 1)
        if length < 1:
            raise ValueError(f"input of {samples} samples is too short for this feature encoder")
    return length


class ConvFeatureBlock(OrderedModules):
    """``wav2vec2.py:839-860`` ``block(...)`` for ``is_layer_norm=True``.

    ``Conv1d -> Dropout -> Sequential(TransposeLast, Fp32LayerNorm(affine=True),
    TransposeLast) -> GELU``.  The double transpose turns the ``NCT`` conv
    output into ``NTC`` so the LayerNorm normalises the **512 channels**, and the
    norm is **affine**, i.e. its ``weight``/``bias`` are real checkpoint tensors.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        eps: float,
        conv_bias: bool,
    ) -> None:
        super().__init__(
            {
                "0": nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=conv_bias,
                ),
                "1": Identity(),
                "2": OrderedModules(
                    {
                        "0": TransposeLast(),
                        "1": Fp32LayerNorm(out_channels, eps=eps, affine=True),
                        "2": TransposeLast(),
                    }
                ),
                "3": Gelu(),
            }
        )
        self.out_channels = int(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """``x`` is ``NCT`` (PyTorch layout); returns ``NCT``.

        Full notes: docs/port-notes/functions.md :: __call__
        """
        ntc = self["0"](mx.swapaxes(x, 1, 2))  # NTC conv boundary -> NTC out
        nct = mx.swapaxes(ntc, 1, 2)  # back to NCT for the rest of the chain
        nct = self["1"](nct)  # Identity (Dropout at inference)
        # self["2"] is Sequential(TransposeLast, Fp32LayerNorm, TransposeLast):
        # NCT -> NTC puts the 512 channels on the norm's last axis -> NCT.
        nct = self["2"](nct)
        return self["3"](nct)  # GELU is elementwise; returns NCT


class ConvFeatureExtraction(nn.Module):
    """``ConvFeatureExtractionModel`` with ``mode="layer_norm"``."""

    def __init__(
        self, layers: Sequence[tuple[int, int, int]], *, eps: float, conv_bias: bool
    ) -> None:
        super().__init__()
        blocks = []
        in_d = 1
        for dim, kernel, stride in layers:
            blocks.append(ConvFeatureBlock(in_d, dim, kernel, stride, eps=eps, conv_bias=conv_bias))
            in_d = dim
        self.conv_layers = ModuleListStack(blocks)
        self.out_channels = int(layers[-1][0])
        self.spec = tuple(tuple(entry) for entry in layers)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 2:
            raise ValueError(f"expected (B, samples), got shape {x.shape}")
        y = x[:, None, :]  # BxT -> Bx1xT
        for block in self.conv_layers:
            y = block(y)
        return y


# --------------------------------------------------------------------------- #
# relative positional encoder (audio.py:83-106)
# --------------------------------------------------------------------------- #
class SamePad(OrderedModules):
    """``fairseq.modules.SamePad`` (non-causal), ``same_pad.py:18-22``.

    Full notes: docs/port-notes/functions.md :: SamePad
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__({})
        self.kernel_size = int(kernel_size)
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def __call__(self, x: mx.array) -> mx.array:
        if self.remove > 0:
            # reference: x[:, :, : -remove] -- the final slot on the last axis
            x = x[..., : -self.remove]
        return x


class PositionalBlock(OrderedModules):

    def __init__(self, dim: int, kernel_size: int, groups: int, eps: float) -> None:
        super().__init__(
            {
                "0": nn.Conv1d(
                    dim,
                    dim,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    dilation=1,
                    groups=groups,
                    bias=True,
                ),
                "1": SamePad(kernel_size),
                "2": TransposeLast(),
                "3": Fp32LayerNorm(dim, eps=eps, affine=False),
                "4": TransposeLast(),
                "5": Gelu(),
            }
        )

    def __call__(self, x: mx.array) -> mx.array:
        """``x`` is ``NTC`` (channels-last); returns ``NTC``.

        Full notes: docs/port-notes/functions.md :: __call__
        """
        ntc = self["0"](mx.swapaxes(x, 1, 2))  # NTC conv boundary -> NTC out
        nct = mx.swapaxes(ntc, 1, 2)  # NCT: SamePad's last axis is time
        nct = self["1"](nct)  # SamePad crops the last axis == T
        nct = self["2"](nct)  # TransposeLast: -> NTC
        nct = self["3"](nct)  # LayerNorm over C
        nct = self["4"](nct)  # TransposeLast: -> NCT
        return self["5"](nct)  # GELU is elementwise; returns NCT


class PositionalEncoder(OrderedModules):
    """``Sequential(TransposeLast, 5 x PositionalBlock, TransposeLast)``.

    With ``conv_pos_pre_ln=false`` (the large config) the conv's
    ``padding = k // 2 = 9`` preserves ``T`` and ``SamePad(19)`` is the identity
    on top of it, so the output length is always exactly ``T``.

    Time axis contract: the encoder takes ``NTC`` and returns ``NTC``; each
    :class:`PositionalBlock` takes ``NTC`` and returns ``NCT``, so the encoder
    restores ``NTC`` for the final output.  Each :class:`PositionalBlock` is
    ``NCT`` in and ``NCT`` out, so nothing is transposed *between* blocks.
    """

    def __init__(self, dim: int, depth: int, kernel_size: int, groups: int, eps: float) -> None:
        modules: dict[str, nn.Module] = {"0": TransposeLast()}
        for index in range(depth):
            modules[str(index + 1)] = PositionalBlock(dim, kernel_size, groups, eps)
        modules[str(depth + 1)] = TransposeLast()
        super().__init__(modules)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
        self.groups = int(groups)

    def __call__(self, x: mx.array) -> mx.array:
        """``NTC`` in, ``NTC`` out -- the untouched source chain.

        The leading ``TransposeLast`` converts ``NTC -> NCT`` for the blocks (each
        ``NCT -> NCT``) and the trailing one restores ``NTC``.
        """
        return super().__call__(x)


# --------------------------------------------------------------------------- #
# AltBlock / AltAttention (modules.py)
# --------------------------------------------------------------------------- #
class AltAttention(nn.Module):
    """``modules.py:255-300`` ``AltAttention`` (cosine attention is not used)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, alibi_bias: mx.array | None) -> mx.array:
        b, n, c = x.shape
        qkv = self.qkv(x)  # (B, N, 3C)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim)
        q = mx.swapaxes(qkv[:, :, 0], 1, 2)  # (B, H, N, d)
        k = mx.swapaxes(qkv[:, :, 1], 1, 2)
        v = mx.swapaxes(qkv[:, :, 2], 1, 2)
        # The entire score pipeline (scale + QK^T + bias + softmax + @V) is one
        # fused SDPA kernel; the ALiBi bias goes in as the additive ``mask``
        # argument and SDPA's internal softmax runs in float32. Equivalent to
        # the manual path (covered by the alignment suite) minus four kernel
        # launches per layer.
        bias = None
        if alibi_bias is not None:
            _require(
                alibi_bias.shape == (b, self.num_heads, n, n),
                f"alibi_bias must be (B, {self.num_heads}, N, N), got {alibi_bias.shape}",
            )
            bias = alibi_bias.astype(q.dtype) if alibi_bias.dtype != q.dtype else alibi_bias
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=bias)
        out = mx.swapaxes(out, 1, 2).reshape(b, n, self.num_heads * self.head_dim)
        return self.proj(out)


class AltBlock(nn.Module):
    """``modules.py:180-245`` ``AltBlock`` with ``layer_norm_first=false``.

    Post-norm ordering::

        x = x + attn(norm1(x))
        x = x + mlp(norm2(x))

    and the second return value is the post-norm ``x`` when ``ffn_targets`` else
    the pre-norm mlp output.  That value is training-only (it supervises the
    teacher), so the port returns it for API parity but never consumes it.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        eps: float,
        affine: bool,
        layer_norm_first: bool,
        ffn_targets: bool,
    ) -> None:
        super().__init__()
        self.layer_norm_first = bool(layer_norm_first)
        self.ffn_targets = bool(ffn_targets)
        self.norm1 = Fp32LayerNorm(dim, eps=eps, affine=affine)
        self.attn = AltAttention(dim, num_heads)
        hidden = int(dim * mlp_ratio)
        self.norm2 = Fp32LayerNorm(dim, eps=eps, affine=affine)
        self.mlp = Mlp(dim, hidden, Gelu)

    def __call__(self, x: mx.array, alibi_bias: mx.array | None) -> tuple[mx.array, mx.array]:
        """``modules.py:227-244`` with every dropout/drop-path removed.

        Note the post-norm branch feeds the **raw** ``x`` to the attention and
        only then applies ``norm1``; ``t`` is the pre-``norm2`` mlp output when
        ``ffn_targets`` and the final ``x`` otherwise.  ``t`` is training-only
        (it supervises the teacher) and is returned for API parity only.
        """
        if self.layer_norm_first:
            x = x + self.attn(self.norm1(x), alibi_bias)
            r = x = self.mlp(self.norm2(x))
            t = x
            x = r + x
            if not self.ffn_targets:
                t = x
        else:
            x = x + self.attn(x, alibi_bias)
            r = x = self.norm1(x)
            x = self.mlp(x)
            t = x
            x = self.norm2(r + x)
            if not self.ffn_targets:
                t = x
        return x, t


class BlockEncoder(nn.Module):

    def __init__(
        self, blocks: Sequence[nn.Module], norm: nn.Module | None, layer_norm_first: bool
    ) -> None:
        super().__init__()
        self.blocks = ModuleListStack(blocks)
        self.norm = norm
        self.layer_norm_first = bool(layer_norm_first)

    def __call__(
        self, x: mx.array, alibi_bias: mx.array | None, alibi_scale: mx.array | None = None
    ) -> mx.array:
        if self.norm is not None and not self.layer_norm_first:
            x = self.norm(x)
        for index, block in enumerate(self.blocks):
            bias = alibi_bias
            if bias is not None and alibi_scale is not None:
                scale = alibi_scale[index] if alibi_scale.shape[0] > 1 else alibi_scale.squeeze(0)
                bias = bias * scale.astype(bias.dtype)
            x, _ = block(x, bias)
        return x


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Emotion2VecConfig:
    """Validated mirror of the audio modality config plus the model fields."""

    # model
    embed_dim: int = 1024
    depth: int = 8
    num_heads: int = 16
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-5
    norm_affine: bool = True
    layer_norm_first: bool = False
    end_of_block_targets: bool = False

    # feature encoder
    feature_encoder_spec: str = "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)]"
    extractor_mode: str = "layer_norm"
    conv_bias: bool = False
    normalize: bool = True

    # relative positional encoder
    conv_pos_depth: int = 5
    conv_pos_width: int = 95
    conv_pos_groups: int = 16
    conv_pos_pre_ln: bool = False

    # context encoder
    prenet_depth: int = 4

    # extra tokens / alibi
    num_extra_tokens: int = 10
    init_extra_token_zero: bool = True
    use_alibi_encoder: bool = True
    num_alibi_heads: int = 16
    alibi_scale_init: float = 1.0
    alibi_max_pos: int | None = None
    learned_alibi: bool = False
    learned_alibi_scale: bool = True
    learned_alibi_scale_per_head: bool = True
    learned_alibi_scale_per_layer: bool = False

    # classifier
    vocab_size: int = -1
    labels: tuple[str, ...] = ()

    # memory bounds (attention is O(T^2) in 16 * heads * frames)
    max_samples: int = 160_000
    max_frames: int = 640

    conv_layers: tuple[tuple[int, int, int], ...] = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "embed_dim",
            "depth",
            "num_heads",
            "conv_pos_depth",
            "conv_pos_width",
            "conv_pos_groups",
            "prenet_depth",
            "num_alibi_heads",
            "max_samples",
            "max_frames",
            "vocab_size",
        ):
            object.__setattr__(self, name, _as_int(getattr(self, name), name))
        for name in ("mlp_ratio", "norm_eps", "alibi_scale_init"):
            object.__setattr__(self, name, _as_float(getattr(self, name), name))
        object.__setattr__(
            self,
            "num_extra_tokens",
            _as_int(self.num_extra_tokens, "num_extra_tokens", minimum=0),
        )
        for name in (
            "embed_dim",
            "num_heads",
            "conv_pos_depth",
            "conv_pos_groups",
            "prenet_depth",
            "num_alibi_heads",
            "max_samples",
            "max_frames",
        ):
            _require(getattr(self, name) >= 1, f"{name} must be >= 1")
        _require(
            self.embed_dim % self.num_heads == 0,
            f"embed_dim {self.embed_dim} must be divisible by num_heads {self.num_heads}",
        )
        _require(
            self.extractor_mode == "layer_norm",
            f"only extractor_mode='layer_norm' is ported, got {self.extractor_mode!r}",
        )
        _require(
            self.num_alibi_heads == self.num_heads,
            f"num_alibi_heads {self.num_alibi_heads} must equal num_heads {self.num_heads}",
        )
        _require(
            self.embed_dim % self.conv_pos_groups == 0,
            f"embed_dim {self.embed_dim} must be divisible by "
            f"conv_pos_groups {self.conv_pos_groups}",
        )
        _require(self.depth >= 1, "depth must be >= 1")
        _require(
            self.vocab_size == -1 or self.vocab_size >= 0, "vocab_size must be -1 (no head) or >= 0"
        )
        object.__setattr__(self, "labels", tuple(self.labels))
        if self.vocab_size > 0 and self.labels:
            _require(
                len(self.labels) == self.vocab_size,
                f"labels ({len(self.labels)}) must match vocab_size ({self.vocab_size})",
            )
        if self.learned_alibi:
            _require(self.alibi_max_pos is not None, "learned_alibi requires alibi_max_pos")
        resolved = parse_feature_encoder_spec(self.feature_encoder_spec)
        object.__setattr__(self, "conv_layers", tuple(resolved))

    @property
    def feature_embed_dim(self) -> int:
        return self.conv_layers[-1][0]

    @property
    def pos_kernel_size(self) -> int:
        """``k = max(3, conv_pos_width // conv_pos_depth)`` (``audio.py:87``)."""
        return max(3, self.conv_pos_width // self.conv_pos_depth)

    @property
    def ffn_targets(self) -> bool:
        """``ffn_targets = not end_of_block_targets`` (``emotion2vec.py``)."""
        return not self.end_of_block_targets

    def frames_for(self, samples: int) -> int:
        _require(samples >= 1, "samples must be >= 1")
        _require(
            samples <= self.max_samples, f"{samples} samples exceeds max_samples={self.max_samples}"
        )
        frames = conv_feature_output_length(samples, self.conv_layers)
        total = frames + self.num_extra_tokens
        _require(
            total <= self.max_frames,
            f"{total} frames (with {self.num_extra_tokens} extra tokens) exceeds "
            f"max_frames={self.max_frames}",
        )
        return frames


# --------------------------------------------------------------------------- #
# modality encoder
# --------------------------------------------------------------------------- #
class AudioEncoder(nn.Module):

    def __init__(self, config: Emotion2VecConfig) -> None:
        super().__init__()
        self.config = config
        self.local_encoder = ConvFeatureExtraction(
            config.conv_layers, eps=config.norm_eps, conv_bias=config.conv_bias
        )
        self.project_features = SequentialStack(
            {
                "0": TransposeLast(),
                "1": nn.LayerNorm(config.feature_embed_dim, eps=config.norm_eps),
                "2": nn.Linear(config.feature_embed_dim, config.embed_dim),
            }
        )
        self.relative_positional_encoder = PositionalEncoder(
            config.embed_dim,
            config.conv_pos_depth,
            config.pos_kernel_size,
            config.conv_pos_groups,
            config.norm_eps,
        )
        prenet_blocks = [
            AltBlock(
                config.embed_dim,
                config.num_heads,
                config.mlp_ratio,
                eps=config.norm_eps,
                affine=config.norm_affine,
                layer_norm_first=config.layer_norm_first,
                ffn_targets=config.ffn_targets,
            )
            for _ in range(config.prenet_depth)
        ]
        prenet_norm = (
            None
            if config.layer_norm_first
            else Fp32LayerNorm(config.embed_dim, eps=config.norm_eps, affine=config.norm_affine)
        )
        self.context_encoder = BlockEncoder(prenet_blocks, prenet_norm, config.layer_norm_first)
        self.extra_tokens = mx.zeros((1, config.num_extra_tokens, config.embed_dim))
        if config.learned_alibi_scale and config.use_alibi_encoder:
            rows = config.prenet_depth + config.depth if config.learned_alibi_scale_per_layer else 1
            self.alibi_scale = mx.full(
                (
                    rows,
                    1,
                    config.num_alibi_heads if config.learned_alibi_scale_per_head else 1,
                    1,
                    1,
                ),
                float(config.alibi_scale_init),
                dtype=mx.float32,
            )
        else:
            self.alibi_scale = None

    # -- pieces, kept separate so the tests can drive them individually ----- #
    def _weight_dtype(self) -> mx.Dtype:
        """Dtype of this encoder's checkpoint weights (first conv always exists)."""
        return self.local_encoder.conv_layers["0"]["0"].weight.dtype

    def local_features(self, samples: mx.array) -> mx.array:
        """Waveform ``(B, samples)`` -> projected features ``(B, T, embed_dim)``.

        The conv stack returns ``NCT`` and ``project_features`` is the source's
        ``Sequential(TransposeLast, LayerNorm(512), Linear)``.  Its leading
        ``TransposeLast`` is a **real** transpose, so the ``NCT`` tensor is fed
        straight in and the norm correctly reduces the **512 channels**.
        """
        x = self.local_encoder(samples)  # (B, C, frames) == NCT
        x = self.project_features(x)  # -> NTC -> LayerNorm(512) -> NTC -> Linear
        return x

    def alibi_bias(self, batch_size: int, time_steps: int) -> mx.array:
        """``get_alibi_bias`` -> ``(B, heads, T, T)``."""
        base = get_alibi(time_steps, self.config.num_alibi_heads)  # (H, T, T)
        return mx.broadcast_to(base[None], (batch_size, *base.shape))

    def contextualized_features(
        self, samples: mx.array
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        """``contextualized_features(..., mask=False, remove_masked=False)``.

        Returns ``(x, alibi_bias, alibi_scale_for_model_blocks)``.  When the
        learned scale has a single row it is folded into ``alibi_bias`` here and
        the third element is ``None``; with one row **per layer** the rows stay
        live and the tail (``alibi_scale[prenet_depth:]``) is handed back so the
        shared blocks pick their own row, exactly like the reference.
        """
        if samples.ndim != 2:
            raise ValueError(f"expected (B, samples), got shape {samples.shape}")
        # Bound the O(T^2) attention here too, not only via Emotion2VecModel.
        self.config.frames_for(int(samples.shape[-1]))

        # Follow the module's own weight dtype, exactly as whisper's
        # ``AudioEncoder.__call__`` does.  The waveform is normalised in float32
        # (``preprocess_waveform``), so without this cast an fp16 checkpoint
        # runs the conv stack and all 12 blocks with float32 activations against
        # fp16 weights and pays a mixed-dtype promotion at every op (measured on
        # M4, 3.3 s window: 51.7 ms vs 37.2 ms, 1.39x).
        weight_dtype = self._weight_dtype()
        samples = samples.astype(weight_dtype)

        x = self.local_features(samples)
        x = x + self.relative_positional_encoder(x)

        alibi_bias = self.alibi_bias(x.shape[0], x.shape[1])
        alibi_scale = self.alibi_scale
        remaining: mx.array | None = None
        if alibi_scale is not None:
            alibi_scale = mx.maximum(alibi_scale, 0)
            if alibi_scale.shape[0] == 1:
                alibi_bias = alibi_bias * alibi_scale.squeeze(0).astype(alibi_bias.dtype)
                alibi_scale = None
            else:
                remaining = alibi_scale[self.config.prenet_depth :]

        num = self.extra_tokens.shape[1]
        if num:
            x = mx.concatenate(
                [mx.broadcast_to(self.extra_tokens, (x.shape[0], num, x.shape[2])), x], axis=1
            )
            if alibi_bias is not None:
                alibi_bias = mx.pad(alibi_bias, [(0, 0), (0, 0), (num, 0), (num, 0)])

        x = self.context_encoder(x, alibi_bias, alibi_scale)
        return x, alibi_bias, remaining

    def __call__(self, samples: mx.array) -> tuple[mx.array, mx.array, mx.array | None]:
        return self.contextualized_features(samples)


# --------------------------------------------------------------------------- #
# top level model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmotionResult:
    """Mirrors the keys FunASR's ``Emotion2vec.inference`` returns."""

    key: str
    labels: tuple[str, ...]
    scores: tuple[float, ...]
    feats: mx.array


def preprocess_waveform(waveform: mx.array, normalize: bool = True, eps: float = 1e-5) -> mx.array:
    """``source = F.layer_norm(source, source.shape)`` then ``view(1, -1)``.

    The reference normalises over the *entire* 1-D waveform (one global
    normalisation), not per frame.
    """
    x = waveform
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"expected (samples,) or (B, samples), got shape {x.shape}")
    if normalize:
        x32 = x.astype(mx.float32)
        mean = mx.mean(x32, axis=-1, keepdims=True)
        centred = x32 - mean
        var = mx.mean(centred * centred, axis=-1, keepdims=True)
        x = centred / mx.sqrt(var + eps)
    return x


class Emotion2VecModel(nn.Module):
    """``Emotion2vec`` (FunASR wrapper) as native MLX modules.

    The checkpoint is a plain ``state_dict`` of the ``Emotion2vec`` module, so the
    parameter prefixes are ``modality_encoders.AUDIO.*``, ``blocks.N.*`` and
    ``proj.*``.  ``self.modality_encoders`` is a plain dict so the keys come out
    as ``modality_encoders.AUDIO.<...>`` exactly like the reference
    ``nn.ModuleDict``.
    """

    def __init__(self, config: Emotion2VecConfig) -> None:
        super().__init__()
        self.config = config
        audio = AudioEncoder(config)
        self.modality_encoders = {"AUDIO": audio}
        self.blocks = ModuleListStack(
            [
                AltBlock(
                    config.embed_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    eps=config.norm_eps,
                    affine=config.norm_affine,
                    layer_norm_first=config.layer_norm_first,
                    ffn_targets=config.ffn_targets,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = (
            Fp32LayerNorm(config.embed_dim, eps=config.norm_eps, affine=config.norm_affine)
            if config.layer_norm_first
            else None
        )
        self.proj = (
            nn.Linear(config.embed_dim, config.vocab_size) if config.vocab_size > 0 else None
        )

    # -- forward paths ------------------------------------------------------ #
    def extract_features(self, samples: mx.array, *, remove_extra_tokens: bool = True) -> mx.array:
        """Waveform ``(B, samples)`` -> frame features ``(B, T', embed_dim)``.

        ``remove_extra_tokens=True`` strips the ``num_extra_tokens`` cls tokens,
        which is what FunASR does before the utterance-level pooling.  The encoder
        is evaluated exactly **once**.
        """
        if samples.ndim != 2:
            raise ValueError(f"expected (B, samples), got shape {samples.shape}")
        # Attention is O(T^2) * num_heads, so the bound is enforced here rather
        # than only by callers who pre-flight with config.frames_for.
        self.config.frames_for(int(samples.shape[-1]))
        encoder = self.modality_encoders["AUDIO"]
        x, alibi_bias, alibi_scale = encoder.contextualized_features(samples)
        for index, block in enumerate(self.blocks):
            bias = alibi_bias
            if bias is not None and alibi_scale is not None:
                scale = alibi_scale[index] if alibi_scale.shape[0] > 1 else alibi_scale.squeeze(0)
                bias = bias * scale.astype(bias.dtype)
            x, _ = block(x, bias)
        if self.norm is not None:
            x = self.norm(x)
        if remove_extra_tokens:
            num = encoder.config.num_extra_tokens
            if num:
                x = x[:, num:, :]
        return x

    def classify(self, samples: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Return ``(features (B, embed_dim), logits (B, vocab_size), frames)``.

        The head is ``nn.Linear(embed_dim, vocab_size)`` applied to the mean over
        time **after** the extra tokens have been removed
        (``Emotion2vec.inference``: ``x = x.mean(dim=1); x = self.proj(x)``).  The
        frame features are returned so a caller that also wants them does not have
        to run the whole encoder a second time.
        """
        if self.proj is None:
            raise RuntimeError(
                "no classifier head: construct with vocab_size > 0. The reference "
                "only builds self.proj when vocab_size > 0."
            )
        frames = self.extract_features(samples, remove_extra_tokens=True)
        pooled = mx.mean(frames, axis=1)  # (B, D)
        return pooled, self.proj(pooled), frames

    def emotion_scores(self, samples: mx.array) -> EmotionResult:
        """Full FunASR-equivalent result: labels + softmax scores + features.

        Runs the encoder exactly once; ``feats`` is the utterance-level pooled
        embedding the classifier consumed, which is the same
        ``x.mean(dim=1)`` the reference uses, so no second pass is needed.
        """
        if self.proj is None:
            raise RuntimeError("emotion_scores requires a classifier head (vocab_size > 0)")
        if samples.ndim != 2:
            raise ValueError(f"expected (B, samples), got shape {samples.shape}")
        if samples.shape[0] != 1:
            raise ValueError(
                f"emotion_scores returns a single EmotionResult, got batch "
                f"{samples.shape[0]}; use classify() for batched output"
            )
        pooled, logits, _frames = self.classify(samples)
        labels = self.config.labels
        for index, label in enumerate(labels):
            if label.startswith("unuse"):
                logits = mx.where(
                    mx.arange(logits.shape[-1], dtype=logits.dtype) == index,
                    mx.array(float("-inf"), logits.dtype),
                    logits,
                )
        probs = mx.softmax(logits, axis=-1)
        row = probs[0].tolist()
        selected = [(lab, row[i]) for i, lab in enumerate(labels) if not lab.startswith("unuse")]
        return EmotionResult(
            key="",
            labels=tuple(lab for lab, _ in selected),
            scores=tuple(float(score) for _, score in selected),
            feats=pooled,
        )


# --------------------------------------------------------------------------- #
# checkpoint loading
# --------------------------------------------------------------------------- #
def flat_parameter_names(module: nn.Module) -> dict[str, mx.array]:
    """``{dotted.name: array}`` for every parameter leaf.

    Full notes: docs/port-notes/functions.md :: flat_parameter_names
    """
    flat = tree_flatten(module, destination={})
    return {key: value for key, value in flat.items() if isinstance(value, mx.array)}


#: Exact checkpoint keys whose layout differs between PyTorch and MLX.  These
#: are written out in full rather than derived from a positional guess, because
#: the sequence index sits at ``parts[-3]`` and a ``parts[-3] == "conv_layers"``
#: check silently stops matching.
#:
#: * ``conv_layers.<i>.0.weight`` -- ``nn.Conv1d`` weight ``(out, in//g, K)``
#: * ``relative_positional_encoder.<i>.0.weight`` -- same ``nn.Conv1d``
CONV_WEIGHT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^modality_encoders\.AUDIO\.local_encoder\.conv_layers\.\d+\.0\.weight$"),
    re.compile(r"^modality_encoders\.AUDIO\.relative_positional_encoder\.\d+\.0\.weight$"),
)


def is_conv_weight(name: str) -> bool:
    """True for the depthwise ``Conv1d`` weights listed in :data:`CONV_WEIGHT_PATTERNS`."""
    return any(pattern.match(name) for pattern in CONV_WEIGHT_PATTERNS)


def convert_checkpoint_tensors(
    state: Mapping[str, mx.array | Any],
) -> tuple[dict[str, mx.array], tuple[str, ...]]:
    """Convert a published ``state_dict`` and report what was discarded.

    Only ``Conv1d`` weights change layout: ``(out, in//g, K) -> (out, K, in//g)``.
    Linears, LayerNorms, ``extra_tokens`` and ``alibi_scale`` already match MLX.
    """
    converted: dict[str, mx.array] = {}
    discarded: list[str] = []
    for name, value in state.items():
        if any(name.startswith(prefix) for prefix in INFERENCE_ONLY_PREFIXES):
            discarded.append(name)
            continue
        if not isinstance(value, mx.array):
            value = mx.array(value)
        if is_conv_weight(name):
            if value.ndim != 3:
                raise ValueError(
                    f"{name}: expected a 3-D Conv1d weight (out, in//g, K), got shape {value.shape}"
                )
            value = mx.transpose(value, (0, 2, 1))
        converted[name] = value
    return converted, tuple(sorted(discarded))


class CheckpointKeyError(ValueError):
    """Raised when a checkpoint does not match the model exactly."""


def _assign(root: dict[str, Any], dotted: str, value: mx.array) -> None:
    parts = dotted.split(".")
    node = root
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def load_state_dict_strict(
    module: nn.Module,
    state: Mapping[str, mx.array | Any],
    *,
    allow_training_only: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load ``state`` into ``module``; return ``(discarded, unexpected)``.

    Never silently skips a key: anything outside
    :data:`INFERENCE_ONLY_PREFIXES` must match a parameter by name *and* shape.
    """
    converted, discarded = convert_checkpoint_tensors(state)
    if not allow_training_only and discarded:
        raise CheckpointKeyError(f"training-only keys present but not allowed: {list(discarded)}")
    target = flat_parameter_names(module)
    missing = sorted(set(target) - set(converted))
    unexpected = sorted(set(converted) - set(target))
    mismatched = sorted(
        name
        for name in set(target) & set(converted)
        if tuple(target[name].shape) != tuple(converted[name].shape)
    )
    if missing or unexpected or mismatched:
        raise CheckpointKeyError(
            "checkpoint does not match the model: "
            f"missing={missing} unexpected={unexpected} mismatched={mismatched}"
        )
    nested: dict[str, Any] = {}
    for name, array in converted.items():
        _assign(nested, name, array)
    module.update(tree_unflatten(nested))
    return discarded, ()
