"""Native MLX inference port of ``emotion2vec_plus_large``.

Source of truth (read-only, pinned):

* ``github.com/ddlBoJack/emotion2vec`` @ ``main``, ``upstream/models/``
  ``audio.py`` (blob ``329222b2``), ``base.py`` (``c63de01c``),
  ``emotion2vec.py`` (``fe71a344``), ``modules.py`` (``dae553af``),
  ``config.py`` (``29eae511``).
* ``huggingface/emotion2vec/emotion2vec_plus_large`` @ ``6c303ba9``,
  ``config.yaml``.
* ``ConvFeatureExtractionModel`` from ``fairseq`` v0.12.2
  ``fairseq/models/wav2vec/wav2vec2.py``, lines 819-895.  Note this file is
  *not* ``conv_feature.py`` (that path 404s and is absent from fairseq's
  history); ``wav2vec2.py`` is where the class lives.
* Classifier head from the FunASR wrapper
  ``modelscope/FunASR`` @ ``main``, ``funasr/models/emotion2vec/model.py``
  (``Emotion2vec.inference``).  The upstream ``emotion2vec.py`` contains only
  ``Data2VecMultiModel`` and has no head, and the Hub repository ships no
  ``auto_map``/modeling file, so that wrapper is the only source for it.

Architecture for ``emotion2vec_plus_large`` (from the pinned ``config.yaml``):
``embed_dim=1024``, ``depth=8``, ``num_heads=16``, ``mlp_ratio=4.0``,
``norm_eps=1e-5``, ``norm_affine=true``, ``layer_norm_first=false`` (post-norm),
``feature_encoder_spec='[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] +
[(512, 2, 2)]'``, ``extractor_mode=layer_norm``, ``conv_bias=false``,
``normalize=true``, ``conv_pos_depth=5`` / ``conv_pos_width=95`` /
``conv_pos_groups=16`` / ``conv_pos_pre_ln=false`` (so ``k = max(3, 95 // 5) =
19`` and ``SamePad(19)`` runs *in addition to* the conv's ``padding = 9``),
``prenet_depth=4``, ``num_extra_tokens=10``, ``learned_alibi=false``,
``learned_alibi_scale=true`` with ``per_head=true`` / ``per_layer=false``,
``use_alibi_encoder=true``, ``num_alibi_heads=16``, ``alibi_scale=1.0``.

Deliberately excluded (training-only, see :data:`INFERENCE_ONLY_PREFIXES`):
``modality_encoders.AUDIO.decoder.*`` (:class:`Decoder1d`), the EMA, mask
construction (``compute_mask``/``apply_mask``/``MaskInfo``), layerdrop,
drop-path, and every dropout.  At inference they are identities.

Layout rules (checkpoint ``state_dict`` -> MLX)

=========================  =========================  =====================
key suffix                 PyTorch shape             MLX shape
=========================  =========================  =====================
``*.conv_layers.N.0.weight``  ``(out, in//g, K)``     ``(out, K, in//g)``
``*.project_features.2.weight`` ``(out, in)``       ``(out, in)``
``*.relative_positional_encoder.N.0.weight`` ``(C, C//g, K)`` ``(C, K, C//g)``
``*.norm1.weight`` / ``.bias``  ``(D,)``           ``(D,)``
``*.extra_tokens``         ``(1, N, D)``            ``(1, N, D)``
``*.alibi_scale``          ``(1, 1, H, 1, 1)``       ``(1, 1, H, 1, 1)``
=========================  =========================  =====================

Traps this file deliberately avoids: MLX ``nn.Module`` stores children in its
*dict* part (so the loader uses ``module.parameters()``, never ``vars(module)``);
``mx.transpose`` takes a single axes sequence; every ``mx.nn.Conv*`` is
channels-last so each call site transposes ``NCT -> NTC`` and back; ``mx.take``
is always given an explicit ``axis``.

No numerical test was executed for this port; see the task report for what is
and is not verified.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn

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
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
    """``get_alibi.get_slopes`` (verbatim, including the power-of-2 branch)."""

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

    def __getitem__(self, index: int) -> nn.Module:
        return getattr(self, self._order[index])


class SequentialStack(OrderedModules):
    """``nn.Sequential`` with automatic ``0..n-1`` indices."""


class ModuleListStack(nn.Module):
    """``nn.ModuleList``; children are registered under their decimal index."""

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

    def __getitem__(self, index: int) -> nn.Module:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        return getattr(self, str(index))


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
        raise ValueError(
            f"unsupported expression in feature_encoder_spec: {type(node).__name__}"
        )

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
    """``floor((L - kernel) / stride + 1)`` folded over the stack."""
    length = int(samples)
    for _, kernel, stride in layers:
        length = math.floor((length - kernel) / stride + 1)
        if length < 1:
            raise ValueError(
                f"input of {samples} samples is too short for this feature encoder"
            )
    return length


class ConvFeatureBlock(nn.Module):
    """``wav2vec2.py:839-860`` ``block(...)`` for ``is_layer_norm=True``.

    ``Conv1d -> Dropout -> Sequential(TransposeLast, Fp32LayerNorm(affine=True),
    TransposeLast) -> GELU``.  The double transpose turns the ``NCT`` conv
    output into ``NTC`` so the LayerNorm normalises the **512 channels**, and the
    norm is **affine**, i.e. its ``weight``/``bias`` are real checkpoint tensors.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 eps: float, conv_bias: bool) -> None:
        super().__init__()
        # The children are registered *only* through the OrderedModules so that the
        # keys are ``conv_layers.N.0.weight`` / ``conv_layers.N.2.1.weight``; keeping
        # an extra alias attribute would duplicate every leaf in the key space.
        self.block = OrderedModules(
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
        """``x`` is ``NCT`` (PyTorch layout); returns ``NCT``."""
        y = mx.swapaxes(x, 1, 2)  # NCT -> NTC
        y = self.block(y)  # NTC -> NTC
        return mx.swapaxes(y, 1, 2)  # NTC -> NCT


class ConvFeatureExtraction(nn.Module):
    """``ConvFeatureExtractionModel`` with ``mode="layer_norm"``."""

    def __init__(self, layers: Sequence[tuple[int, int, int]], *, eps: float,
                 conv_bias: bool) -> None:
        super().__init__()
        blocks = []
        in_d = 1
        for dim, kernel, stride in layers:
            blocks.append(
                ConvFeatureBlock(in_d, dim, kernel, stride, eps=eps, conv_bias=conv_bias)
            )
            in_d = dim
        self.conv_layers = ModuleListStack(blocks)
        self.out_channels = int(layers[-1][0])
        self.spec = tuple(tuple(entry) for entry in layers)

    def __call__(self, x: mx.array) -> mx.array:
        """``x`` is ``(B, samples)``; returns ``(B, C, frames)``."""
        if x.ndim != 2:
            raise ValueError(f"expected (B, samples), got shape {x.shape}")
        y = x[:, None, :]  # BxT -> Bx1xT
        for block in self.conv_layers:
            y = block(y)
        return y


# --------------------------------------------------------------------------- #
# relative positional encoder (audio.py:83-106)
# --------------------------------------------------------------------------- #
class SamePad(nn.Module):
    """``fairseq.modules.SamePad`` (non-causal).

    ``remove = 1 if k is even else 0``; left-pad by ``remove`` then right-pad by
    one when the length became even.  For the odd ``k = 19`` used by
    ``emotion2vec_plus_large`` the only effect is a **+1 frame right pad whenever
    the incoming length is odd**, which the port reproduces exactly.
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def __call__(self, x: mx.array) -> mx.array:
        if self.remove > 0:
            x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(self.remove, 0)])
        if x.shape[-1] % 2 == 0:
            x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, 1)])
        return x


class PositionalBlock(nn.Module):
    """One of the ``conv_pos_depth`` blocks (``audio.py:88-100``)."""

    def __init__(self, dim: int, kernel_size: int, groups: int, eps: float) -> None:
        super().__init__()
        # Registered only through ``block`` to keep the key ``....N.0.weight``.
        self.block = OrderedModules(
            {
                "0": nn.Conv1d(
                    dim, dim, kernel_size, stride=1, padding=kernel_size // 2,
                    dilation=1, groups=groups, bias=True,
                ),
                "1": SamePad(kernel_size),
                "2": TransposeLast(),
                "3": Fp32LayerNorm(dim, eps=eps, affine=False),
                "4": TransposeLast(),
                "5": Gelu(),
            }
        )

    def __call__(self, x: mx.array) -> mx.array:
        """``x`` is ``NTC`` (channels-last); returns ``NTC``."""
        y = mx.swapaxes(x, 1, 2)  # NTC -> NCT
        y = self.block(y)  # keeps NCT inside the Sequential
        return mx.swapaxes(y, 1, 2)  # NCT -> NTC


class PositionalEncoder(nn.Module):
    """``Sequential(TransposeLast, 5 x PositionalBlock, TransposeLast)``.

    With ``conv_pos_pre_ln=false`` (the large config) the conv's
    ``padding = k // 2 = 9`` and ``SamePad(19)`` are *both* applied, so the
    output length is ``L`` for even ``L`` and ``L + 1`` for odd ``L``.
    """

    def __init__(self, dim: int, depth: int, kernel_size: int, groups: int,
                 eps: float) -> None:
        super().__init__()
        modules: dict[str, nn.Module] = {"0": TransposeLast()}
        for index in range(depth):
            modules[str(index + 1)] = PositionalBlock(dim, kernel_size, groups, eps)
        modules[str(depth + 1)] = TransposeLast()
        self.encoder = SequentialStack(modules)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
        self.groups = int(groups)

    def __call__(self, x: mx.array) -> mx.array:
        return self.encoder(x)


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
        scores = q @ mx.swapaxes(k, -1, -2)
        scores = scores * (self.head_dim**-0.5)
        if alibi_bias is not None:
            _require(
                alibi_bias.shape == (b, self.num_heads, n, n),
                f"alibi_bias must be (B, {self.num_heads}, N, N), got {alibi_bias.shape}",
            )
            scores = scores + alibi_bias
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v  # (B, H, N, d)
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

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, eps: float,
                 affine: bool, layer_norm_first: bool, ffn_targets: bool) -> None:
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
    """``modules.py:63-95`` ``BlockEncoder`` (the prenet / context encoder)."""

    def __init__(self, blocks: Sequence[nn.Module], norm: nn.Module | None,
                 layer_norm_first: bool) -> None:
        super().__init__()
        self.blocks = ModuleListStack(blocks)
        self.norm = norm
        self.layer_norm_first = bool(layer_norm_first)

    def __call__(self, x: mx.array, alibi_bias: mx.array | None) -> mx.array:
        if self.norm is not None and not self.layer_norm_first:
            x = self.norm(x)
        for block in self.blocks:
            x, _ = block(x, alibi_bias)
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
            "embed_dim", "depth", "num_heads", "conv_pos_depth", "conv_pos_width",
            "conv_pos_groups", "prenet_depth", "num_alibi_heads", "max_samples",
            "max_frames", "vocab_size",
        ):
            object.__setattr__(self, name, _as_int(getattr(self, name), name))
        for name in ("mlp_ratio", "norm_eps", "alibi_scale_init"):
            object.__setattr__(self, name, _as_float(getattr(self, name), name))
        object.__setattr__(
            self, "num_extra_tokens",
            _as_int(self.num_extra_tokens, "num_extra_tokens", minimum=0),
        )
        for name in ("embed_dim", "num_heads", "conv_pos_depth", "conv_pos_groups",
                     "prenet_depth", "num_alibi_heads", "max_samples", "max_frames"):
            _require(getattr(self, name) >= 1, f"{name} must be >= 1")
        _require(self.embed_dim % self.num_heads == 0,
                 f"embed_dim {self.embed_dim} must be divisible by num_heads {self.num_heads}")
        _require(self.extractor_mode == "layer_norm",
                 f"only extractor_mode='layer_norm' is ported, got {self.extractor_mode!r}")
        _require(self.num_alibi_heads == self.num_heads,
                 f"num_alibi_heads {self.num_alibi_heads} must equal num_heads {self.num_heads}")
        _require(
            self.embed_dim % self.conv_pos_groups == 0,
            f"embed_dim {self.embed_dim} must be divisible by "
            f"conv_pos_groups {self.conv_pos_groups}",
        )
        _require(self.depth >= 1, "depth must be >= 1")
        _require(self.vocab_size == -1 or self.vocab_size >= 0,
                 "vocab_size must be -1 (no head) or >= 0")
        object.__setattr__(self, "labels", tuple(self.labels))
        if self.vocab_size > 0 and self.labels:
            _require(len(self.labels) == self.vocab_size,
                     f"labels ({len(self.labels)}) must match vocab_size ({self.vocab_size})")
        if self.learned_alibi:
            _require(self.alibi_max_pos is not None,
                     "learned_alibi requires alibi_max_pos")
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
        _require(samples <= self.max_samples,
                 f"{samples} samples exceeds max_samples={self.max_samples}")
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
    """``ModalitySpecificEncoder`` specialised to ``AUDIO`` (``audio.py``)."""

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
        self.context_encoder = BlockEncoder(
            prenet_blocks, prenet_norm, config.layer_norm_first
        )
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
    def local_features(self, samples: mx.array) -> mx.array:
        """Waveform ``(B, samples)`` -> projected features ``(B, T, embed_dim)``."""
        x = self.local_encoder(samples)  # (B, C, frames)
        x = self.project_features(x)  # TransposeLast + LayerNorm + Linear
        return x

    def alibi_bias(self, batch_size: int, time_steps: int) -> mx.array:
        """``get_alibi_bias`` -> ``(B, heads, T, T)``."""
        base = get_alibi(time_steps, self.config.num_alibi_heads)  # (H, T, T)
        return mx.broadcast_to(base[None], (batch_size, *base.shape))

    def contextualized_features(self, samples: mx.array) -> tuple[mx.array, mx.array]:
        """``contextualized_features(..., mask=False, remove_masked=False)``.

        Returns ``(x, alibi_bias_after_prenet_scaling)``.
        """
        x = self.local_features(samples)
        x = x + self.relative_positional_encoder(x)

        alibi_bias = self.alibi_bias(x.shape[0], x.shape[1])
        alibi_scale = self.alibi_scale
        if alibi_scale is not None:
            alibi_scale = mx.maximum(alibi_scale, 0)
            if alibi_scale.shape[0] == 1:
                alibi_bias = alibi_bias * alibi_scale.squeeze(0).astype(alibi_bias.dtype)
                alibi_scale = None

        num = self.extra_tokens.shape[1]
        if num:
            x = mx.concatenate(
                [mx.broadcast_to(self.extra_tokens, (x.shape[0], num, x.shape[2])), x], axis=1
            )
            if alibi_bias is not None:
                alibi_bias = mx.pad(alibi_bias, [(0, 0), (0, 0), (num, 0), (num, 0)])

        x = self.context_encoder(x, alibi_bias)
        return x, alibi_bias

    def __call__(self, samples: mx.array) -> tuple[mx.array, mx.array]:
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


def preprocess_waveform(waveform: mx.array, normalize: bool = True,
                        eps: float = 1e-5) -> mx.array:
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
            nn.Linear(config.embed_dim, config.vocab_size)
            if config.vocab_size > 0
            else None
        )

    # -- forward paths ------------------------------------------------------ #
    def extract_features(self, samples: mx.array, *, remove_extra_tokens: bool = True) -> mx.array:
        """Waveform ``(B, samples)`` -> frame features ``(B, T', embed_dim)``.

        ``remove_extra_tokens=True`` strips the ``num_extra_tokens`` cls tokens,
        which is what FunASR does before the utterance-level pooling.
        """
        encoder = self.modality_encoders["AUDIO"]
        # Attention is O(T^2) * num_heads, so the bound is enforced on the forward
        # path, not only by callers that pre-flight with config.frames_for.
        self.config.frames_for(int(samples.shape[-1]))
        x, alibi_bias = encoder.contextualized_features(samples)
        for block in self.blocks:
            x, _ = block(x, alibi_bias)
        if self.norm is not None:
            x = self.norm(x)
        if remove_extra_tokens:
            num = encoder.config.num_extra_tokens
            if num:
                x = x[:, num:, :]
        return x

    def classify(self, samples: mx.array) -> tuple[mx.array, mx.array]:
        """Return ``(features (B, embed_dim), logits (B, vocab_size))``.

        The head is ``nn.Linear(embed_dim, vocab_size)`` applied to the mean over
        time **after** the extra tokens have been removed
        (``Emotion2vec.inference``: ``x = x.mean(dim=1); x = self.proj(x)``).
        """
        if self.proj is None:
            raise RuntimeError(
                "no classifier head: construct with vocab_size > 0. The reference "
                "only builds self.proj when vocab_size > 0."
            )
        frames = self.extract_features(samples, remove_extra_tokens=True)
        pooled = mx.mean(frames, axis=1)  # (B, D)
        return pooled, self.proj(pooled)

    def emotion_scores(self, samples: mx.array) -> EmotionResult:
        """Full FunASR-equivalent result: labels + softmax scores + features."""
        if self.proj is None:
            raise RuntimeError("emotion_scores requires a classifier head (vocab_size > 0)")
        _features, logits = self.classify(samples)
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
            feats=mx.mean(self.extract_features(samples, remove_extra_tokens=True), axis=1),
        )


# --------------------------------------------------------------------------- #
# checkpoint loading
# --------------------------------------------------------------------------- #
def flat_parameter_names(module: nn.Module) -> dict[str, mx.array]:
    """``{dotted.name: array}`` for every parameter leaf.

    Uses ``mlx.utils.tree_flatten`` on the module itself -- **not**
    ``vars(module)`` and **not** ``module.parameters()``:

    * ``nn.Module`` is a ``dict`` subclass and ``__setattr__`` stores arrays and
      submodules in the dict part, so ``vars()`` reports zero tensors;
    * ``parameters()`` applies ``valid_parameter_filter``, which drops biases,
      LayerNorm weights, ``extra_tokens`` and ``alibi_scale`` -- all of which
      are real tensors in this checkpoint.

    Non-array leaves (ints, strings, tuples) are filtered out.
    """
    flat = mx.utils.tree_flatten(module)
    return {
        key: value for key, value in flat.items() if isinstance(value, mx.array)
    }


def _is_conv_weight(name: str) -> bool:
    """True for ``*.conv_layers.<i>.0.weight`` and ``*.relative_positional_encoder.<j>.0.weight``."""  # noqa: E501
    if not name.endswith(".weight"):
        return False
    parts = name.split(".")
    if parts[-2] != "0":
        return False
    return parts[-3] in ("conv_layers",) or "relative_positional_encoder" in parts


def convert_checkpoint_tensors(
    state: Mapping[str, mx.array | Any],
) -> tuple[dict[str, mx.array], tuple[str, ...]]:
    """Convert a published ``state_dict`` and report what was discarded.

    Only ``Conv1d``/``Conv2d`` weights change layout:
    ``(out, in//g, K) -> (out, K, in//g)``.  Linears, LayerNorms, ``extra_tokens``
    and ``alibi_scale`` already match MLX.
    """
    converted: dict[str, mx.array] = {}
    discarded: list[str] = []
    for name, value in state.items():
        if any(name.startswith(prefix) for prefix in INFERENCE_ONLY_PREFIXES):
            discarded.append(name)
            continue
        if not isinstance(value, mx.array):
            value = mx.array(value)
        if _is_conv_weight(name):
            if value.ndim == 3:  # Conv1d: (out, in//g, K)
                value = mx.transpose(value, (0, 2, 1))
            elif value.ndim == 4:  # Conv2d: (out, in//g, KH, KW)
                value = mx.transpose(value, (0, 2, 3, 1))
            else:
                raise ValueError(f"{name}: expected a 3-D or 4-D conv weight, got {value.shape}")
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
        raise CheckpointKeyError(
            f"training-only keys present but not allowed: {list(discarded)}"
        )
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
    module.update(mx.utils.tree_unflatten(nested))
    return discarded, ()