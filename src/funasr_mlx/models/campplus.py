"""Native MLX inference port of FunASR's CAM++ speaker-verification model.

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/campplus.py.md
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Iterator, Sequence

import mlx.core as mx
from mlx import nn as mxnn

from ._layout_state import layout_rule as _layout_rule
from ._strict_load import _filter

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
    "flatten_state_dict_with_rules",
    "get_nonlinear",
    "kaiming_normal_",
    "load_strict",
    "statistics_pooling",
]

#: Buffer names that exist in a PyTorch BatchNorm but have no MLX counterpart.
TORCH_ONLY_BUFFERS = ("num_batches_tracked",)
CAMPPLUS_FRONTEND_DEFAULTS = {"window": "povey", "upsacle_samples": False}


# containers that reproduce PyTorch parameter names
class ModuleStack(mxnn.Module):
    # see docs/port-notes/functions.md :: ModuleStack

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
        if not args:
            raise TypeError(f"{type(self).__name__}.__call__() requires at least one argument")
        head, *rest = args
        for block in self:
            head = block(head, *rest, **kwargs)
        return head


class NamedModuleDict(mxnn.Module):
    # see docs/port-notes/functions.md :: NamedModuleDict

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
    # see docs/port-notes/functions.md :: NamedLayerList

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
        return list(self._names)

    def __iter__(self) -> Iterator[mxnn.Module]:
        for name in self._names:
            yield getattr(self, name)

    def __getitem__(self, index: int | str) -> mxnn.Module:
        if isinstance(index, str):
            return dict.__getitem__(self, index)
        return getattr(self, self._names[index])

    def __call__(self, x: mx.array) -> mx.array:
        for module in self:
            x = mx.concatenate([x, module(x)], axis=-1)
        return x


# checkpoint plumbing
#: Layout rule names produced by :func:`flatten_state_dict`.
IDENTITY_LAYOUT = "identity"
CONV1D_LAYOUT = "conv1d_torch_out_ink_to_mlx_out_ki"
CONV2D_LAYOUT = "conv2d_torch_out_inkij_to_mlx_out_kij"


def _conv_rule(module: mxnn.Module) -> str:
    names = {"conv1d": CONV1D_LAYOUT, "conv2d": CONV2D_LAYOUT, "identity": IDENTITY_LAYOUT}
    return _layout_rule(module, names)


def convert_checkpoint_tensor(name: str, array: mx.array, rule: str = IDENTITY_LAYOUT):
    """Convert one PyTorch checkpoint tensor into the MLX layout.

    Full notes: docs/port-notes/functions.md :: convert_checkpoint_tensor
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
            rules[name] = _conv_rule(module) if key == "weight" else IDENTITY_LAYOUT


@dataclass
class LayoutReport(Mapping):
    # see docs/port-notes/functions.md :: LayoutReport

    tensors: dict[str, mx.array]
    rules: dict[str, str]

    def __getitem__(self, name: str) -> mx.array:
        return self.tensors[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.tensors)

    def __len__(self) -> int:
        return len(self.tensors)

    def __contains__(self, name: object) -> bool:
        return name in self.tensors

    def keys(self):
        return self.tensors.keys()

    def values(self):
        return self.tensors.values()

    def items(self):
        return self.tensors.items()

    def get(self, name: str, default: Any = None) -> Any:
        return self.tensors.get(name, default)


def flatten_state_dict(module: mxnn.Module, prefix: str = "") -> LayoutReport:
    # see docs/port-notes/functions.md :: flatten_state_dict
    tensors: dict[str, mx.array] = {}
    rules: dict[str, str] = {}
    _flatten_into(module, prefix, tensors, rules)
    return LayoutReport(tensors=tensors, rules=rules)


def flatten_state_dict_with_rules(
    module: mxnn.Module, prefix: str = ""
) -> tuple[dict[str, mx.array], dict[str, str]]:
    # see docs/port-notes/functions.md :: flatten_state_dict_with_rules
    report = flatten_state_dict(module, prefix)
    return report.tensors, report.rules


class CheckpointKeyError(KeyError):
    """Raised by :func:`load_strict` when a checkpoint does not match a module."""


@dataclass
class LoadReport:
    # see docs/port-notes/functions.md :: LoadReport

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

    Full notes: docs/port-notes/functions.md :: load_strict
    """
    report = flatten_state_dict(module)
    expected = {f"{prefix}{k}": v for k, v in report.tensors.items()}
    incoming = dict(tensors)

    dropped: list[str] = []
    if allow_torch_only_buffers:
        allowed = {
            f"{prefix}{path}.{buffer}" if path else f"{prefix}{buffer}"
            for path, child in module.named_modules()
            if isinstance(child, BatchNorm)
            for buffer in TORCH_ONLY_BUFFERS
        }
        for key in sorted(allowed.intersection(incoming)):
            counter = mx.asarray(incoming[key])
            if counter.shape != () or counter.dtype != mx.int64:
                raise ValueError(f"Invalid BatchNorm training counter: {key}")
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
    if key in dict.keys(module):
        return dict.__getitem__(module, key)
    return getattr(module, key)


def _assign(module: mxnn.Module, parts: Sequence[str], array: mx.array) -> None:
    node: Any = module
    for part in parts[:-1]:
        node = _resolve_child(node, part)
    setattr(node, parts[-1], array)


# initialisation (matches the reference's post-build kaiming pass)
def kaiming_normal_(weight: mx.array) -> mx.array:
    # see docs/port-notes/functions.md :: kaiming_normal_
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


# normalisation
class BatchNorm(mxnn.Module):
    """``torch.nn.BatchNorm{1,2}d`` for channels-last MLX activations.

    Full notes: docs/port-notes/functions.md :: BatchNorm
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


def get_nonlinear(config_str: str, channels: int) -> NamedModuleDict:
    # see docs/port-notes/functions.md :: get_nonlinear
    blocks = NamedModuleDict()
    for name in config_str.split("-"):
        if name == "relu":
            child = mxnn.ReLU()
        elif name == "prelu":
            child = mxnn.PReLU(channels)
        elif name == "batchnorm":
            child = BatchNorm1d(channels)
        elif name == "batchnorm_":
            child = BatchNorm1d(channels, affine=False)
        else:
            raise ValueError(f"Unexpected module ({name}).")
        key = "batchnorm" if name == "batchnorm_" else name
        if key in blocks.names():
            setattr(blocks, key, child)
        else:
            blocks.add(key, child)
    return blocks


# statistics pooling
def statistics_pooling(
    x: mx.array,
    *,
    unbiased: bool = True,
    keepdim: bool = True,
    eps: float = 1e-2,
) -> mx.array:
    # see docs/port-notes/functions.md :: statistics_pooling
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

    def __call__(self, x: mx.array) -> mx.array:
        return statistics_pooling(x)


# FCM front-end
class BasicResBlock(mxnn.Module):
    # see docs/port-notes/functions.md :: BasicResBlock

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

    Full notes: docs/port-notes/functions.md :: FCM
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
        # see docs/port-notes/functions.md :: __call__
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


# TDNN building blocks (all channels-last NLC, matching MLX Conv1d)
class TDNNLayer(mxnn.Module):
    # see docs/port-notes/functions.md :: TDNNLayer

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
    # see docs/port-notes/functions.md :: CAMLayer

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
    # see docs/port-notes/functions.md :: CAMDenseTDNNLayer

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
    # see docs/port-notes/functions.md :: TransitLayer

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
    # see docs/port-notes/functions.md :: DenseLayer

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


# configuration

@dataclass
class CAMPlusConfig:
    # see docs/port-notes/functions.md :: CAMPlusConfig

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
    # see docs/port-notes/functions.md :: campplus_config_from_dict
    data = dict(data or {})
    if "model_conf" in data and isinstance(data["model_conf"], Mapping):
        merged = dict(data["model_conf"])
        for key in ("encoder_conf", "frontend_conf"):
            if key in data and key not in merged:
                merged[key] = data[key]
        data = merged
    return CAMPlusConfig.from_dict(data)


# model
class CAMPPlus(mxnn.Module):
    # see docs/port-notes/functions.md :: CAMPPlus

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

    def state_dict(self) -> LayoutReport:
        # see docs/port-notes/functions.md :: state_dict
        return flatten_state_dict(self)

    def load_checkpoint(self, tensors: Mapping[str, Any], **kwargs: Any) -> LoadReport:
        # see docs/port-notes/functions.md :: load_checkpoint
        return load_strict(self, tensors, **kwargs)

    def frame_features(self, x: mx.array) -> mx.array:
        # see docs/port-notes/functions.md :: frame_features
        x = mx.transpose(x, (0, 2, 1))  # (B, T, F) -> (B, F, T)
        x = self.head(x)  # (B, T', C*F/8)
        for name in self.xvector.names():
            if name == "out_nonlinear":
                break
            x = self.xvector[name](x)
        return self.xvector["out_nonlinear"](x)

    def forward(self, x: mx.array) -> mx.array:
        # see docs/port-notes/functions.md :: forward
        features = self.frame_features(x)
        if self.output_level == "frame":
            return features  # already (B, T, C); torch returns x.transpose(1, 2)
        pooled = self.xvector["stats"](features)  # (B, 1, 2C)
        return mx.squeeze(self.xvector["dense"](pooled), 1)  # (B, embedding_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)

    def embed(self, speech: mx.array, speech_lengths: mx.array | None = None) -> mx.array:
        """Length-aware embedding extraction for a padded batch.

        Full notes: docs/port-notes/functions.md :: embed
        """
        if speech_lengths is None or self.output_level == "frame":
            return self.forward(speech)

        lengths = mx.asarray(speech_lengths).reshape(-1)
        batch, frames, _ = speech.shape
        if lengths.shape[0] != batch:
            raise ValueError(
                f"speech_lengths has {lengths.shape[0]} entries for a batch of {batch}"
            )
        values = [int(lengths[b].item()) for b in range(batch)]
        if not 0 < min(values) <= frames:
            raise ValueError(f"speech_lengths {values} out of range for {frames} input frames")
        rows = [self.forward(speech[b : b + 1, : values[b], :]) for b in range(batch)]
        return mx.concatenate(rows, axis=0)

    def speaker_embedding(
        self, speech: mx.array, speech_lengths: mx.array | None = None
    ) -> mx.array:
        return self.embed(speech, speech_lengths)
