"""Native MLX inference port of FunASR's CAM++ speaker-verification model.

Reference (read-only, ``/Users/baicai1145/repos/funasr``):

* ``funasr/models/campplus/components.py``
* ``funasr/models/campplus/model.py``
* ``funasr/models/campplus/template.yaml`` (published defaults)

This module is deliberately **self-contained** (it duplicates the small
container / checkpoint helpers instead of importing ``funasr_mlx.models.sanm``)
so that it stays correct while the ASR modules are under review, and so that it
adds no dependency between model files.

Layout rules that matter
------------------------
1. **Channels-last convolutions.**  MLX's ``Conv1d`` consumes ``NLC`` and
   ``Conv2d`` consumes ``NHWC``; PyTorch uses ``NCL``/``NCHW``.  This port keeps
   the network's *activation* layout channels-last and therefore transposes only
   the reference's two explicit layout changes (FCM's final reshape and
   ``output_level == "frame"``).  See :meth:`FCM.__call__`.
2. **BatchNorm buffers.**  ``nn.BatchNorm{1,2}d`` carries ``weight``/``bias``
   (only when ``affine=True``) plus the persistent buffers ``running_mean`` and
   ``running_var``, and a torch-only scalar ``num_batches_tracked`` that MLX has
   no equivalent for.  :func:`load_strict` reports the dropped torch-only buffer
   explicitly instead of skipping it silently.
3. **Statistics pooling.**  The reference reduces the *time* axis of an NCL
   tensor.  This port reduces axis 1 of an NLC tensor, which is exactly the same
   quantity; see :class:`StatsPool`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Iterator, Mapping, Sequence

import mlx.core as mx
from mlx import nn as mxnn

__all__ = [
    "BasicResBlock",
    "CAMPlusConfig",
    "CAMDenseTDNNBlock",
    "CAMDenseTDNNLayer",
    "CAMLayer",
    "CAMPPlus",
    "BatchNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "CheckpointKeyError",
    "DenseLayer",
    "FCM",
    "LayoutReport",
    "LoadReport",
    "ModuleStack",
    "NamedLayerList",
    "NamedModuleDict",
    "StatsPool",
    "TDNNLayer",
    "TransitLayer",
    "TORCH_ONLY_BUFFERS",
    "campplus_config_from_dict",
    "convert_checkpoint_tensor",
    "flatten_state_dict",
    "get_nonlinear",
    "kaiming_normal_",
    "load_strict",
    "statistics_pooling",
]

#: Buffer names that exist in a PyTorch BatchNorm but have no MLX counterpart.
TORCH_ONLY_BUFFERS = ("num_batches_tracked",)


# --------------------------------------------------------------------------- #
# containers that reproduce PyTorch parameter names
# --------------------------------------------------------------------------- #
class ModuleStack(mxnn.Module):
    """Ordered children registered under their numeric index.

    ``torch.nn.Sequential`` yields ``<parent>.<i>.<leaf>`` parameter names; MLX's
    ``nn.Sequential`` would insert an extra ``layers`` level, so children are
    registered directly under ``"0"``, ``"1"``, ...
    """

    def __init__(self, blocks: Sequence[mxnn.Module]) -> None:
        super().__init__()
        for index, block in enumerate(blocks):
            setattr(self, str(index), block)
        self._length = len(blocks)

    def __len__(self) -> int:
        # Keep dict semantics: MLX may call len() on this object.
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

    def names(self) -> list[str]:
        return [str(index) for index in range(dict.__len__(self))]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Chain blocks with canonical ``nn.Sequential`` semantics.

        The leading argument is threaded through the chain while the remaining
        positional arguments and keywords stay fixed.  Feeding a block's return
        value back as ``*args`` would unpack an ``mx.array`` along axis 0 and hand
        the next block ``B`` separate arguments, which only works at batch size 1.
        """
        if not args:
            raise TypeError(f"{type(self).__name__}.__call__() requires at least one argument")
        head, *rest = args
        for block in self:
            head = block(head, *rest, **kwargs)
        return head


class NamedModuleDict(mxnn.Module):
    """Named children applied in order; keeps the ``<parent>.<name>.<leaf>`` keys.

    Used for ``CAMPPlus.xvector``, which the reference builds as an
    ``nn.Sequential`` of named sub-modules.
    """

    def __init__(self) -> None:
        super().__init__()
        self._names: list[str] = []

    def add(self, name: str, module: mxnn.Module) -> mxnn.Module:
        setattr(self, name, module)
        self._names.append(name)
        return module

    def __len__(self) -> int:
        return len(self._names)

    def names(self) -> list[str]:
        """Registered names in insertion order."""
        return list(self._names)

    def __iter__(self) -> Iterator[mxnn.Module]:
        for name in self._names:
            yield getattr(self, name)

    def __getitem__(self, key: Any) -> mxnn.Module:
        name = key if isinstance(key, str) else self._names[key]
        return getattr(self, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not args:
            raise TypeError(f"{type(self).__name__}.__call__() requires at least one argument")
        head, *rest = args
        for module in self:
            head = module(head, *rest, **kwargs)
        return head


class NamedLayerList(mxnn.Module):
    """Named children iterated in insertion order.

    Mirrors ``torch.nn.ModuleList`` with explicit names (``CAMDenseTDNNBlock``
    uses ``tdnnd1``, ``tdnnd2``, ...).
    """

    def __init__(self) -> None:
        super().__init__()
        self._names: list[str] = []

    def add(self, name: str, module: mxnn.Module) -> mxnn.Module:
        setattr(self, name, module)
        self._names.append(name)
        return module

    def __len__(self) -> int:
        return len(self._names)

    def names(self) -> list[str]:
        """Registered names in insertion order (``tdnnd1``, ``tdnnd2``, ...)."""
        return list(self._names)

    def __iter__(self) -> Iterator[mxnn.Module]:
        for name in self._names:
            yield getattr(self, name)

    def __getitem__(self, index: int) -> mxnn.Module:
        return getattr(self, self._names[index])

    def __call__(self, x: mx.array) -> mx.array:
        for module in self:
            x = mx.concatenate([x, module(x)], axis=-1)
        return x


# --------------------------------------------------------------------------- #
# checkpoint plumbing
# --------------------------------------------------------------------------- #
#: Layout rule names produced by :func:`flatten_state_dict`.
IDENTITY_LAYOUT = "identity"
CONV1D_LAYOUT = "conv1d_torch_out_ink_to_mlx_out_ki"
CONV2D_LAYOUT = "conv2d_torch_out_inkij_to_mlx_out_kij"


def convert_checkpoint_tensor(name: str, array: mx.array, rule: str = IDENTITY_LAYOUT):
    """Convert one PyTorch checkpoint tensor into the MLX layout.

    ``rule`` is the module-derived rule from :func:`flatten_state_dict`; the name
    is kept for diagnostics and error messages only.
    """
    if rule == CONV1D_LAYOUT:
        # torch (out, in/groups, k)  ->  MLX (out, k, in/groups)
        return mx.transpose(array, (0, 2, 1))
    if rule == CONV2D_LAYOUT:
        # torch (out, in, kh, kw)  ->  MLX (out, kh, kw, in)
        return mx.transpose(array, (0, 2, 3, 1))
    if rule != IDENTITY_LAYOUT:
        raise ValueError(f"unknown layout rule {rule!r} for parameter {name!r}")
    return array


def _conv_rule(module: mxnn.Module) -> str:
    if isinstance(module, mxnn.Conv1d):
        return CONV1D_LAYOUT
    if isinstance(module, mxnn.Conv2d):
        return CONV2D_LAYOUT
    return IDENTITY_LAYOUT


def _flatten_into(module: Any, prefix: str, tensors: dict, rules: dict) -> None:
    """Collect tensors from a module tree.

    ``mlx.nn.Module`` **subclasses ``dict``** (``class Module(dict)`` in
    ``mlx/nn/layers/base.py``).  Its ``__setattr__`` routes arrays, dicts, lists,
    tuples and *sub-modules* (a ``Module`` is itself a ``dict``, so
    ``isinstance(v, dict)`` holds) into the **mapping** storage via
    ``self[key] = val``; everything else -- ints, strings, dataclasses and the
    ``_``-prefixed configuration attached here -- goes to ``__dict__``.

    Consequently ``vars(module)`` contains **no tensors at all** (only
    ``_no_grad``/``_training``), and flattening through it silently yields an
    empty dict.  The mapping storage is walked instead.
    """
    for key, value in dict.items(module):
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, mxnn.Module):
            _flatten_into(value, name, tensors, rules)
        elif isinstance(value, mx.array):
            tensors[name] = value
            rules[name] = _conv_rule(module)


@dataclass
class LayoutReport:
    """Flat parameter tensors plus the layout rule that applies to each key."""

    tensors: dict[str, mx.array]
    rules: dict[str, str]

    def __contains__(self, name: str) -> bool:
        return name in self.tensors


def flatten_state_dict(module: mxnn.Module, prefix: str = "") -> LayoutReport:
    """Flatten a module tree into FunASR/PyTorch-compatible parameter names.

    Unlike a name-pattern heuristic, the layout rule is derived from the *owning
    module type*, so it is correct for any naming the architecture happens to
    use.  Attributes starting with ``_`` are configuration, never weights.
    """
    tensors: dict[str, mx.array] = {}
    rules: dict[str, str] = {}
    _flatten_into(module, prefix, tensors, rules)
    return LayoutReport(tensors=tensors, rules=rules)


class CheckpointKeyError(KeyError):
    """Raised by :func:`load_strict` when a checkpoint does not match a module."""


@dataclass
class LoadReport:
    """Outcome of a :func:`load_strict` call.

    ``dropped`` lists the torch-only buffers that were intentionally ignored.
    It is never empty silently: callers can assert on it.
    """

    loaded: list[str]
    dropped: list[str]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LoadReport(loaded={len(self.loaded)}, dropped={self.dropped})"


def load_strict(
    module: mxnn.Module,
    tensors: Mapping[str, Any],
    *,
    prefix: str = "",
    convert: bool = False,
    dtype: Any | None = None,
    allow_torch_only_buffers: bool = True,
) -> LoadReport:
    """Load ``tensors`` into ``module``, refusing any key mismatch.

    Parameters
    ----------
    module:
        Destination module.
    tensors:
        ``{name: array}`` using FunASR/PyTorch names, e.g.
        ``xvector.block1.tdnnd1.cam_layer.linear1.weight``.  Arrays must
        already be in MLX layout unless ``convert=True``.
    prefix:
        Optional shared key prefix (e.g. ``"model."``).
    convert:
        ``True`` when ``tensors`` come straight from a PyTorch ``state_dict``.
    dtype:
        Optional cast applied after the layout conversion.
    allow_torch_only_buffers:
        Accept (and report) torch-only BatchNorm scalars such as
        ``num_batches_tracked``, which MLX does not model.  Set ``False`` to make
        their presence an error.

    Raises
    ------
    CheckpointKeyError
        On missing or unexpected keys (other than the allowed buffers).
    ValueError
        On any shape mismatch.
    """
    report = flatten_state_dict(module)
    expected = {f"{prefix}{k}": v for k, v in report.tensors.items()}
    incoming = dict(tensors)

    dropped: list[str] = []
    if allow_torch_only_buffers:
        for key in list(incoming):
            short = key.rsplit(".", 1)[-1]
            if short in TORCH_ONLY_BUFFERS and key not in expected:
                dropped.append(key)
                del incoming[key]

    missing = sorted(set(expected) - set(incoming))
    unexpected = sorted(set(incoming) - set(expected))
    if missing or unexpected:
        raise CheckpointKeyError(
            "checkpoint/module key mismatch: "
            f"{len(missing)} missing, {len(unexpected)} unexpected. "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    loaded: list[str] = []
    for name in sorted(incoming):
        array = mx.asarray(incoming[name])
        if convert:
            short = name[len(prefix) :]
            array = convert_checkpoint_tensor(
                short, array, report.rules.get(short, IDENTITY_LAYOUT)
            )
        if dtype is not None:
            array = array.astype(dtype)
        target = expected[name]
        if tuple(array.shape) != tuple(target.shape):
            raise ValueError(
                f"shape mismatch for {name!r}: checkpoint {tuple(array.shape)} "
                f"vs model {tuple(target.shape)}"
            )
        _assign(module, name[len(prefix) :].split("."), array)
        loaded.append(name)
    return LoadReport(loaded=loaded, dropped=dropped)


def _resolve_child(module: mxnn.Module, key: str) -> Any:
    """Fetch a child by mapping key, falling back to a plain attribute."""
    if key in dict.keys(module):
        return dict.__getitem__(module, key)
    return getattr(module, key)


def _assign(module: mxnn.Module, parts: Sequence[str], array: mx.array) -> None:
    node: Any = module
    for part in parts[:-1]:
        node = _resolve_child(node, part)
    setattr(node, parts[-1], array)


# --------------------------------------------------------------------------- #
# initialisation (matches the reference's post-build kaiming pass)
# --------------------------------------------------------------------------- #
def kaiming_normal_(weight: mx.array) -> mx.array:
    """In-place-style ``torch.nn.init.kaiming_normal_`` for MLX conv/linear weights.

    The reference re-initialises every ``Conv1d``/``Linear`` weight with
    ``kaiming_normal_`` and zeroes the biases *after* building ``CAMPPlus``.
    ``fan_in`` is the number of input elements per output element, which is the
    product of every axis except the output axis in both layouts.
    """
    fan_in = 1
    for dim in weight.shape[1:]:
        fan_in *= dim
    return mx.random.normal(weight.shape, dtype=weight.dtype) * math.sqrt(2.0 / max(fan_in, 1))


def _reset_conv_linear(module: mxnn.Module) -> None:
    if isinstance(module, (mxnn.Conv1d, mxnn.Conv2d, mxnn.Linear)):
        module.weight = kaiming_normal_(module.weight)
        bias = vars(module).get("bias")
        if isinstance(bias, mx.array):
            module.bias = mx.zeros_like(bias)
    for value in vars(module).values():
        if isinstance(value, mxnn.Module):
            _reset_conv_linear(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, mxnn.Module):
                    _reset_conv_linear(item)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
class BatchNorm(mxnn.Module):
    """``torch.nn.BatchNorm{1,2}d`` for channels-last MLX activations.

    The input's **last** axis must be the channel axis, so statistics are
    reduced over every leading axis, exactly like the reference reduces over
    ``(N, L)`` / ``(N, H, W)``.

    Eval semantics (the only mode used for inference) are
    ``(x - running_mean) / sqrt(running_var + eps) * weight + bias`` with
    ``running_var`` holding the *unbiased* sample variance, as in PyTorch.

    ``torch.nn.BatchNorm*`` additionally persists a scalar
    ``num_batches_tracked`` that has no computational role and no MLX analogue;
    :func:`load_strict` reports it in ``LoadReport.dropped``.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self._num_features = int(num_features)
        self._eps = float(eps)
        self._momentum = float(momentum)
        if affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))
        else:
            self.weight = None
            self.bias = None
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

    def __call__(self, x: mx.array) -> mx.array:
        if x.shape[-1] != self._num_features:
            raise ValueError(
                f"BatchNorm expects {self._num_features} channels on the last axis, "
                f"got {x.shape[-1]} for input shape {tuple(x.shape)}"
            )
        if self.training:
            axes = tuple(range(x.ndim - 1))
            batch_mean = mx.mean(x, axis=axes, keepdims=True)
            batch_var = mx.var(x, axis=axes, ddof=0, keepdims=True)
            unbiased = mx.var(x, axis=axes, ddof=1, keepdims=True)
            self.running_mean = (
                1.0 - self._momentum
            ) * self.running_mean + self._momentum * batch_mean
            self.running_var = (1.0 - self._momentum) * self.running_var + self._momentum * unbiased
            mean, var = batch_mean, batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        y = (x - mean) * mx.rsqrt(var + self._eps)
        if self.weight is not None:
            y = y * self.weight + self.bias
        return y


class BatchNorm1d(BatchNorm):
    """``torch.nn.BatchNorm1d`` (channels-last input ``(B, L, C)``)."""


class BatchNorm2d(BatchNorm):
    """``torch.nn.BatchNorm2d`` (channels-last input ``(B, H, W, C)``)."""


def get_nonlinear(config_str: str, channels: int) -> ModuleStack:
    """Port of ``components.get_nonlinear``.

    ``"batchnorm-relu"`` -> ``[BatchNorm1d(C), ReLU]``; ``"batchnorm_"`` uses
    ``affine=False`` and therefore contributes no parameters; ``"prelu"`` maps to
    ``mxnn.PReLU`` (a single learnable slope, matching ``nn.PReLU``'s default
    ``num_parameters=1``).
    """
    blocks: list[mxnn.Module] = []
    for name in config_str.split("-"):
        if name == "relu":
            blocks.append(mxnn.ReLU())
        elif name == "prelu":
            blocks.append(mxnn.PReLU())
        elif name == "batchnorm":
            blocks.append(BatchNorm1d(channels))
        elif name == "batchnorm_":
            blocks.append(BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module ({name}).")
    return ModuleStack(blocks)


# --------------------------------------------------------------------------- #
# statistics pooling
# --------------------------------------------------------------------------- #
def statistics_pooling(
    x: mx.array,
    *,
    unbiased: bool = True,
    keepdim: bool = True,
    eps: float = 1e-2,
) -> mx.array:
    """Time-axis mean/std pooling of an ``(B, L, C)`` tensor -> ``(B, 1, 2C)``.

    The reference's ``statistics_pooling(x, dim=-1)`` reduces the *time* axis of
    an NCL tensor and concatenates along the channel axis, giving ``(B, 2C)``.
    Reducing axis 1 of the channels-last tensor and concatenating on the last
    axis gives ``(B, 1, 2C)`` with the same numbers in the same order, which is
    what the following ``DenseLayer`` consumes.  ``eps`` is accepted for
    signature parity; the reference never uses it.
    """
    if x.ndim != 3:
        raise ValueError(
            "statistics_pooling expects channels-last (B, L, C); got "
            f"{tuple(x.shape)}. NCL input is not accepted because the reduction "
            "axis would be ambiguous."
        )
    ddof = 1 if unbiased else 0
    mean = mx.mean(x, axis=1, keepdims=keepdim)
    std = mx.std(x, axis=1, ddof=ddof, keepdims=keepdim)
    return mx.concatenate([mean, std], axis=-1)


class StatsPool(mxnn.Module):
    """``components.StatsPool`` (channels-last; see :func:`statistics_pooling`)."""

    def __call__(self, x: mx.array) -> mx.array:
        return statistics_pooling(x)


# --------------------------------------------------------------------------- #
# FCM front-end
# --------------------------------------------------------------------------- #
class BasicResBlock(mxnn.Module):
    """``components.BasicResBlock`` (channels-last NHWC).

    The torch shortcut is an ``nn.Sequential`` (empty => identity), which puts
    its parameters at ``shortcut.0.weight`` / ``shortcut.1.weight``; a
    :class:`ModuleStack` (empty in the identity case) reproduces those names
    exactly.
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = mxnn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = mxnn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ModuleStack(
                [
                    mxnn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=(stride, 1),
                        bias=False,
                    ),
                    BatchNorm2d(self.expansion * planes),
                ]
            )
        else:
            self.shortcut = ModuleStack([])  # identity, contributes no parameters
        self._identity_shortcut = len(self.shortcut) == 0

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.maximum(self.bn1(self.conv1(x)), 0.0)
        out = self.bn2(self.conv2(out))
        out = out + (x if self._identity_shortcut else self.shortcut(x))
        return mx.maximum(out, 0.0)


class FCM(mxnn.Module):
    """``components.FCM`` -- the mel-axis feature-map module.

    The reference treats the mel axis as the spatial *height* (it halves it
    eight times in total via two stride-2 residual stages and a stride-2 conv)
    and the time axis as the *width*.  Its final
    ``out.reshape(B, C * H, W)`` therefore folds channels into the mel axis,
    producing ``(B, m_channels * (feat_dim // 8), T)``.

    Here the tensor is ``(B, H, W, C)`` throughout, so the equivalent result is
    ``(B, W, C * H)`` obtained by transposing to ``(B, W, C, H)`` before the
    reshape -- which keeps PyTorch's ``(C, H)`` row-major ordering.
    """

    def __init__(
        self,
        block: type = BasicResBlock,
        num_blocks: Sequence[int] = (2, 2),
        m_channels: int = 32,
        feat_dim: int = 80,
    ) -> None:
        super().__init__()
        if feat_dim % 8 != 0:
            raise ValueError(f"feat_dim must be divisible by 8 for FCM, got {feat_dim}")
        in_planes = m_channels
        self.conv1 = mxnn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(m_channels)

        def _make_layer(planes: int, num: int, stride: int) -> ModuleStack:
            nonlocal in_planes
            strides = [stride] + [1] * (num - 1)
            layers: list[mxnn.Module] = []
            for current in strides:
                layers.append(block(in_planes, planes, current))
                in_planes = planes * block.expansion
            return ModuleStack(layers)

        # Reference quirk preserved: components.py:72-73 passes ``num_blocks[0]``
        # to BOTH ``_make_layer`` calls, so ``num_blocks[1]`` is never consulted.
        # Irrelevant for the default (2, 2), but kept faithful rather than
        # "corrected", so a checkpoint built with the reference cannot diverge.
        self.layer1 = _make_layer(m_channels, num_blocks[0], 2)
        self.layer2 = _make_layer(m_channels, num_blocks[0], 2)

        self.conv2 = mxnn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)
        self._feat_dim = feat_dim

    def __call__(self, x: mx.array) -> mx.array:
        """``x`` is ``(B, F, T)`` (mel-major, as in the reference) -> ``(B, T, C*F/8)``."""
        x = mx.expand_dims(x, -1)  # (B, F, T, 1)
        out = mx.maximum(self.bn1(self.conv1(x)), 0.0)
        out = self.layer1(out)
        out = self.layer2(out)
        out = mx.maximum(self.bn2(self.conv2(out)), 0.0)
        b, h, w, c = out.shape
        # torch: (B, C, H, W) -> (B, C*H, W);  here: (B, H, W, C) -> (B, W, C*H)
        out = mx.reshape(mx.transpose(out, (0, 2, 3, 1)), (b, w, c * h))
        if c * h != self.out_channels:
            raise RuntimeError(
                f"FCM produced {c * h} channels but out_channels={self.out_channels}; "
                f"check feat_dim={self._feat_dim} / m_channels"
            )
        return out


# --------------------------------------------------------------------------- #
# TDNN building blocks (all channels-last NLC, matching MLX Conv1d)
# --------------------------------------------------------------------------- #
class TDNNLayer(mxnn.Module):
    """``components.TDNNLayer``.

    ``padding < 0`` selects the reference's same-padding rule
    ``padding = (kernel_size - 1) // 2 * dilation`` and requires an odd kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        if padding < 0:
            if kernel_size % 2 == 0:
                raise AssertionError(
                    f"Expect equal paddings, but got even kernel size ({kernel_size})"
                )
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = mxnn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.nonlinear(self.linear(x))


class CAMLayer(mxnn.Module):
    """``components.CAMLayer`` (channels attention over a dilated TDNN).

    ``context`` mixes the *global* time mean with a non-overlapping segment mean
    (``seg_len=100``, average pooling with ``ceil_mode=True``), squeezes it
    through ``linear1``/``relu``/``linear2``/``sigmoid``, and rescales the local
    branch.  Because the channel axis is last, the time axis is axis 1.
    """

    def __init__(
        self,
        bn_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool,
        reduction: int = 2,
    ) -> None:
        super().__init__()
        self.linear_local = mxnn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = mxnn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = mxnn.ReLU()
        self.linear2 = mxnn.Conv1d(bn_channels // reduction, out_channels, 1)

    def seg_pooling(self, x: mx.array, seg_len: int = 100, stype: str = "avg") -> mx.array:
        """Non-overlapping segment mean (or max) broadcast back over time."""
        if stype not in ("avg", "max"):
            raise ValueError("Wrong segment pooling type.")
        length = x.shape[1]
        n_seg = -(-length // seg_len)  # ceil, matching ceil_mode=True
        total = n_seg * seg_len
        pad_width = [(0, 0)] * x.ndim
        pad_width[1] = (0, total - length)
        padded = mx.pad(x, pad_width)
        valid = mx.pad(
            mx.ones((1, length, 1), mx.float32),
            [(0, 0), (0, total - length), (0, 0)],
        )
        seg = mx.reshape(padded, (x.shape[0], n_seg, seg_len, x.shape[2]))
        seg_valid = mx.reshape(valid, (1, n_seg, seg_len, 1))
        if stype == "avg":
            # torch's avg_pool with ceil_mode=True excludes padded positions from
            # the mean (it pads with -inf internally).
            pooled = mx.sum(seg, axis=2) / mx.sum(seg_valid, axis=2)
        else:
            pooled = mx.max(seg + (seg_valid - 1.0) * 1e9, axis=2)
        pooled = mx.repeat(pooled, seg_len, axis=1)
        return pooled[:, :length, :]

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear_local(x)
        context = mx.mean(x, axis=1, keepdims=True) + self.seg_pooling(x)
        context = self.linear1(context)
        context = mx.maximum(context, 0.0)
        m = mx.sigmoid(self.linear2(context))
        return y * m


class CAMDenseTDNNLayer(mxnn.Module):
    """``components.CAMDenseTDNNLayer``.

    ``memory_efficient`` only gates ``torch.utils.checkpoint`` during *training*
    and is therefore inert here (documented, not silently dropped).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise AssertionError(f"Expect equal paddings, but got even kernel size ({kernel_size})")
        padding = (kernel_size - 1) // 2 * dilation
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = mxnn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self._memory_efficient = bool(memory_efficient)

    def __call__(self, x: mx.array) -> mx.array:
        return self.cam_layer(self.nonlinear2(self.linear1(self.nonlinear1(x))))


class CAMDenseTDNNBlock(NamedLayerList):
    """``components.CAMDenseTDNNBlock`` -- dense concatenation along channels."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add(
                f"tdnnd{i + 1}",
                CAMDenseTDNNLayer(
                    in_channels=in_channels + i * out_channels,
                    out_channels=out_channels,
                    bn_channels=bn_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    config_str=config_str,
                    memory_efficient=memory_efficient,
                ),
            )


class TransitLayer(mxnn.Module):
    """``components.TransitLayer`` (nonlinearity **before** the linear map)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        config_str: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = mxnn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(self.nonlinear(x))


class DenseLayer(mxnn.Module):
    """``components.DenseLayer``.

    The reference special-cases 2-D input by squeezing through a length-1 axis so
    that a 1x1 ``Conv1d`` can be used as a plain ``Linear``; that branch is
    reproduced (channels-last, so the squeezed axis is the time axis).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        self.linear = mxnn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 2:
            x = self.linear(mx.expand_dims(x, -1))
            x = mx.squeeze(x, -2)
        else:
            x = self.linear(x)
        return self.nonlinear(x)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def _filter(cls: type, data: Mapping[str, Any] | None) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in (data or {}).items() if k in known}


@dataclass
class CAMPlusConfig:
    """Mirror of FunASR's ``model_conf`` for ``CAMPPlus``.

    Defaults are the published ``speech_campplus_sv_zh-cn_16k-common`` values
    taken from ``funasr/models/campplus/template.yaml``.
    """

    feat_dim: int = 80
    embedding_size: int = 192
    growth_rate: int = 32
    bn_size: int = 4
    init_channels: int = 128
    config_str: str = "batchnorm-relu"
    memory_efficient: bool = True
    output_level: str = "segment"
    m_channels: int = 32
    num_blocks: tuple[int, int] = (2, 2)
    block_spec: tuple[tuple[int, int, int], ...] = ((12, 3, 1), (24, 3, 2), (16, 3, 2))
    tdnn_kernel_size: int = 5
    tdnn_stride: int = 2
    tdnn_dilation: int = 1
    tdnn_padding: int = -1

    def __post_init__(self) -> None:
        self.num_blocks = tuple(self.num_blocks)
        self.block_spec = tuple(tuple(item) for item in self.block_spec)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None, **kwargs: Any):
        merged = dict(_filter(cls, data))
        merged.update(_filter(cls, kwargs))
        return cls(**merged)


def campplus_config_from_dict(
    data: Mapping[str, Any] | None = None, **kwargs: Any
) -> CAMPlusConfig:
    """Build a :class:`CAMPlusConfig` from a FunASR ``model.yaml`` mapping."""
    data = dict(data or {})
    if "model_conf" in data and isinstance(data["model_conf"], Mapping):
        merged = dict(data["model_conf"])
        for key in ("encoder_conf", "frontend_conf"):
            if key in data and key not in merged:
                merged[key] = data[key]
        data = merged
    return CAMPlusConfig.from_dict(data)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class CAMPPlus(mxnn.Module):
    """``funasr.models.campplus.model.CAMPPlus`` (inference only).

    Input is 80-dim Kaldi fbank **without** the LFR stack and **without** CMVN;
    ``funasr/models/campplus/utils.py::extract_feature`` mean-normalises each
    utterance across all frames and zero-pads the batch.  Both steps stay
    outside this module (frontend scope).

    ``output_level="segment"`` returns ``(B, embedding_size)``;
    ``output_level="frame"`` returns ``(B, T, channels)``.
    """

    def __init__(self, config: CAMPlusConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        cfg = config if config is not None else campplus_config_from_dict(None, **kwargs)
        if cfg.output_level not in ("segment", "frame"):
            raise AssertionError(
                f"`output_level` should be set to 'segment' or 'frame', got {cfg.output_level!r}"
            )
        self.config = cfg
        self.output_level = cfg.output_level

        self.head = FCM(
            BasicResBlock,
            num_blocks=cfg.num_blocks,
            m_channels=cfg.m_channels,
            feat_dim=cfg.feat_dim,
        )
        channels = self.head.out_channels

        # The reference keeps everything after ``head`` inside ``self.xvector``,
        # an ``nn.Sequential`` built by ``add_module``; keeping that wrapper
        # preserves the ``xvector.`` prefix in the checkpoint keys.
        self.xvector = NamedModuleDict()
        self.xvector.add(
            "tdnn",
            TDNNLayer(
                channels,
                cfg.init_channels,
                cfg.tdnn_kernel_size,
                stride=cfg.tdnn_stride,
                dilation=cfg.tdnn_dilation,
                padding=cfg.tdnn_padding,
                config_str=cfg.config_str,
            ),
        )
        channels = cfg.init_channels

        self._block_names: list[str] = []
        self._transit_names: list[str] = []
        for i, (num_layers, kernel_size, dilation) in enumerate(cfg.block_spec):
            block_name = f"block{i + 1}"
            self.xvector.add(
                block_name,
                CAMDenseTDNNBlock(
                    num_layers=num_layers,
                    in_channels=channels,
                    out_channels=cfg.growth_rate,
                    bn_channels=cfg.bn_size * cfg.growth_rate,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    config_str=cfg.config_str,
                    memory_efficient=cfg.memory_efficient,
                ),
            )
            channels = channels + num_layers * cfg.growth_rate
            transit_name = f"transit{i + 1}"
            self.xvector.add(
                transit_name,
                TransitLayer(channels, channels // 2, bias=False, config_str=cfg.config_str),
            )
            channels //= 2
        # Introspection only.  These two lists must NEVER be used for execution
        # order: the reference interleaves block_i / transit_i, so
        # ``xvector.names()`` is the single source of truth (see
        # ``CAMPPlus.frame_features``).
        self._block_names = [n for n in self.xvector.names() if n.startswith("block")]
        self._transit_names = [n for n in self.xvector.names() if n.startswith("transit")]

        self.xvector.add("out_nonlinear", get_nonlinear(cfg.config_str, channels))
        self._frame_channels = channels

        if self.output_level == "segment":
            self.xvector.add("stats", StatsPool())
            self.xvector.add(
                "dense",
                DenseLayer(channels * 2, cfg.embedding_size, config_str="batchnorm_"),
            )
            self.embedding_size = cfg.embedding_size
        else:
            self.embedding_size = channels

        _reset_conv_linear(self)

    # -- checkpoint helpers ------------------------------------------------- #
    def state_dict(self) -> LayoutReport:
        """Flat ``{pytorch_name: mlx_array}`` plus per-key layout rules."""
        return flatten_state_dict(self)

    def load_checkpoint(self, tensors: Mapping[str, Any], **kwargs: Any) -> LoadReport:
        """Strict load; see :func:`load_strict`."""
        return load_strict(self, tensors, **kwargs)

    # -- forward ------------------------------------------------------------ #
    def frame_features(self, x: mx.array) -> mx.array:
        """Run everything up to (but excluding) pooling; returns ``(B, T, C)``.

        The stage order is taken from the ``xvector`` container's own insertion
        order, which mirrors the reference's ``nn.Sequential``: ``tdnn``, then
        ``block1, transit1, block2, transit2, ...``, then ``out_nonlinear``.
        Iterating separate block/transit name lists instead would apply **all**
        blocks before **any** transit, which both misorders the network and
        desynchronises the channel arithmetic (each transit halves the channel
        count feeding the next block).
        """
        x = mx.transpose(x, (0, 2, 1))  # (B, T, F) -> (B, F, T)
        x = self.head(x)  # (B, T', C*F/8)
        for name in self.xvector.names():
            if name == "out_nonlinear":
                break
            x = self.xvector[name](x)
        return self.xvector["out_nonlinear"](x)

    def forward(self, x: mx.array) -> mx.array:
        """``x`` is fbank ``(B, T, feat_dim)``.

        Literal port of the reference ``forward``: the batch is assumed to be
        already padded (or to contain a single utterance), because the reference
        pools without any padding mask.  See :meth:`embed` for the length-aware
        variant.
        """
        features = self.frame_features(x)
        if self.output_level == "frame":
            return features  # already (B, T, C); torch returns x.transpose(1, 2)
        pooled = self.xvector["stats"](features)  # (B, 1, 2C)
        return mx.squeeze(self.xvector["dense"](pooled), 1)  # (B, embedding_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)

    def embed(self, speech: mx.array, speech_lengths: mx.array | None = None) -> mx.array:
        """Length-aware embedding extraction for a padded batch.

        The reference zero-pads a batch and pools over **all** frames, so padded
        frames contaminate the segment statistics.  This method restricts the
        mean/std pooling to each utterance's true length, which is identical to
        :meth:`forward` for a single unpadded utterance and is the behaviour you
        want for a real batch.
        """
        if self.output_level == "frame":
            return self.frame_features(speech)
        if speech_lengths is None:
            return self.forward(speech)

        features = self.frame_features(speech)
        lengths = mx.asarray(speech_lengths).reshape(-1)
        batch, frames, _ = features.shape
        if lengths.shape[0] != batch:
            raise ValueError(
                f"speech_lengths has {lengths.shape[0]} entries for a batch of {batch}"
            )
        keep = (mx.arange(frames)[None, :] < lengths[:, None]).astype(features.dtype)
        mask = mx.expand_dims(keep, -1)  # (B, T, 1)
        count = mx.expand_dims(mx.sum(mask, axis=1), -1)  # (B, 1, 1)
        safe_count = mx.maximum(count, 1.0)
        mean = mx.sum(features * mask, axis=1, keepdims=True) / safe_count
        centred = (features - mean) * mask
        denom = mx.maximum(count - 1.0, 1.0)
        std = mx.sqrt(mx.sum(centred * centred, axis=1, keepdims=True) / denom)
        pooled = mx.concatenate([mean, std], axis=-1)  # (B, 1, 2C)
        return mx.squeeze(self.xvector["dense"](pooled), 1)

    def speaker_embedding(
        self, speech: mx.array, speech_lengths: mx.array | None = None
    ) -> mx.array:
        """Alias of :meth:`embed` for readability at call sites."""
        return self.embed(speech, speech_lengths)
