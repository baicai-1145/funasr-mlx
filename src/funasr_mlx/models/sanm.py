"""Shared SANM (self-attention with memory) blocks for the MLX ASR ports.

This module is the single source of truth for every block that FunASR's
``paraformer`` and ``sense_voice`` models share.  It is a direct port of the
following reference files (read-only, ``/Users/baicai1145/repos/funasr``):

* ``funasr/models/sanm/attention.py``
  (``MultiHeadedAttentionSANM``, ``MultiHeadedAttentionSANMDecoder``,
  ``MultiHeadedAttentionCrossAtt``)
* ``funasr/models/sanm/encoder.py`` (``EncoderLayerSANM``)
* ``funasr/models/sanm/positionwise_feed_forward.py``
  (``PositionwiseFeedForwardDecoderSANM``)
* ``funasr/models/transformer/positionwise_feed_forward.py``
  (``PositionwiseFeedForward``)
* ``funasr/models/transformer/layer_norm.py`` (``LayerNorm``, ``eps=1e-12``)
* ``funasr/models/transformer/embedding.py``
  (``PositionalEncoding``, ``SinusoidalPositionEncoder``)
* ``funasr/models/transformer/utils/subsampling.py``
  (``Conv2dSubsampling``, ``Conv2dSubsampling2``)
* ``funasr/models/transformer/utils/nets_utils.py`` (``make_pad_mask``)
* ``funasr/models/scama/utils.py`` (``sequence_mask``)
* ``funasr/models/transformer/utils/repeat.py`` (``repeat``/``MultiSequential``)

Design rules enforced here
--------------------------
1. **Checkpoint-compatible parameter names.**  FunASR keeps its repeated
   encoder/decoder layers in ``nn.Sequential`` (via ``repeat``) or in
   ``nn.ModuleList``.  Both produce ``"<parent>.<index>.<leaf>"`` keys in a
   PyTorch ``state_dict``.  MLX's ``nn.Sequential``/``nn.ModuleList`` would add
   an extra ``layers`` level, so :class:`ModuleStack` registers children under
   their bare numeric index instead.
2. **Explicit tensor-layout conversion rules.**  PyTorch stores conv weights as
   ``(out, in/groups, k)`` / ``(out, in, kh, kw)``; MLX uses
   ``(out, k, in/groups)`` / ``(out, kh, kw, in)``.  See
   :data:`LAYOUT_RULES` and :func:`convert_checkpoint_tensor`.
   MLX convolutions are also **channels-last** (``mx.conv1d`` consumes ``NLC``,
   ``mx.conv2d`` consumes ``NHWC``) whereas PyTorch uses ``NCL``/``NCHW``; every
   conv call site here therefore feeds a channels-last tensor and pads the time
   axis explicitly with ``axis=1``.
3. **Strict loading.**  :func:`load_strict` refuses missing/unexpected keys and
   never silently skips a mismatch.
4. **Inference only.**  Every dropout in the reference is an identity in eval
   mode and is therefore elided; stochastic depth (train-time only) is dropped.
   No training losses are ported.

Known deviations and deliberately not-ported surface
-----------------------------------------------------
These are intentional and reviewed; none of them is a silent partial
implementation.

* **Streaming / chunked paths are not ported.**  ``forward_chunk``,
  ``mask_shfit_chunk``, ``mask_chunk_predictor``, ``forward_one_step``,
  ``forward_asf2``, ``SANMVadEncoder`` / ``SANMEncoder.forward(vad_indexes=...)``,
  ``mask_att_chunk_encoder``, and the ``fa-zh`` timestamp prediction are
  training/streaming-only in the reference and are absent here.
* **``MultiHeadedAttentionSANMDecoder`` cache layout deviates (NLC, not NCL).**
  The reference keeps the streaming cache in ``(b, d, L)`` (``cache = x`` taken
  before the transpose); this port keeps ``(b, L, d)``.  Internally consistent
  and behaviourally identical for offline inference, but it is a *different
  runtime contract* from the reference's export/streaming path, so a cache
  written by one cannot be read by the other.  Two further reference branches
  are streaming-only and therefore absent: the
  ``if x.size(1) != inputs.size(1): inputs = inputs[:, -1, :]`` step-position
  guard, and populating the cache on the first (``cache is None``) call.
* **``decimate_mask`` emulates ``x_mask[:, :, :-2:2][:, :, :-2:2]``** with an
  explicit index list plus ``mx.take(..., axis=-1)``.  Behaviourally identical to
  the slice, just not as cheap.
* **CIF accumulation is float32.**  The reference uses float64 in
  ``cif_v1``/``cif_wo_hidden_v1``; the peak-index thresholding in ``cif_v1`` is
  the sensitive part and can differ by a hair near an exact tie.
* **``cif_v1`` clamps ``max_label_len = min(max_label_len, t)``**; the reference
  has no clamp (``cif_predictor.py:880``).  It is unreachable in practice since
  ``alphas <= 1`` per frame plus at most one 0.45 tail gives ``floor(sum) <= t``.
* **``PositionalEncoding(reverse=True)``** flips the position index where the
  reference reverses an already-built table of length ``max_len``.  Identical for
  ``reverse=False``, the only configured path.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Iterator, Mapping, Sequence

import mlx.core as mx
from mlx import nn as mxnn

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


# --------------------------------------------------------------------------- #
# masking / padding helpers
# --------------------------------------------------------------------------- #
def sequence_mask(lengths: Any, maxlen: int | None = None, dtype: Any = mx.float32) -> mx.array:
    """Port of ``funasr.models.scama.utils.sequence_mask``.

    Returns a ``(batch, maxlen)`` array holding ``1.0`` on valid positions and
    ``0.0`` on padded ones.  ``lengths`` may be ``(B,)`` or ``(B, 1)``.
    """
    lens = mx.asarray(lengths)
    if lens.ndim > 1:
        lens = lens.reshape(-1)
    if maxlen is None:
        maxlen = int(mx.max(lens).item())
    maxlen = int(maxlen)
    positions = mx.arange(0, maxlen)
    return (positions[None, :] < lens[:, None]).astype(dtype)


def make_pad_mask(lengths: Any, maxlen: int | None = None) -> mx.array:
    """Port of ``funasr.models.transformer.utils.nets_utils.make_pad_mask``.

    Returns a boolean ``(batch, maxlen)`` array that is ``True`` on padded
    positions (i.e. the logical inverse of :func:`sequence_mask`).
    """
    return ~sequence_mask(lengths, maxlen, dtype=mx.bool_)


def constant_pad_1d(
    x: mx.array, left: int, right: int, *, axis: int = -1, value: float = 0.0
) -> mx.array:
    """``torch.nn.ConstantPad1d`` port.

    MLX convolutions are **channels-last**: ``mx.conv1d`` consumes ``NLC`` and
    ``mx.conv2d`` consumes ``NHWC``.  The reference SANM modules therefore pad
    the *time* axis of an ``NLC`` tensor, which is ``axis=1`` -- not the last
    axis as it is for PyTorch's ``NCL`` tensors.  ``axis`` is explicit so the
    intent is visible at every call site.
    """
    if left == 0 and right == 0:
        return x
    axis = axis % x.ndim
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (int(left), int(right))
    return mx.pad(x, pad_width, constant_values=value)


def decimate_mask(x_mask: mx.array, stride: int = 2, end_offset: int = 2) -> mx.array:
    """Port of the ``x_mask[:, :, :-2:2][:, :, :-2:2]`` subsampling masks.

    ``torch`` slices with an explicit stop index (``[:-2]`` stops at
    ``length - 2``); MLX negative-stop semantics are easy to get subtly wrong,
    so the equivalent index vector is built explicitly and applied with
    :func:`mx.take`.  ``stride=2`` models ``[:-2:2]`` and ``stride=1`` models
    ``[:-2:1]``; both stop at ``length - end_offset``.
    """
    length = x_mask.shape[-1]
    indices = [i for i in range(0, length - end_offset, stride)]
    if not indices:
        indices = [0]
    return mx.take(x_mask, mx.array(indices, dtype=mx.int32), axis=-1)


# --------------------------------------------------------------------------- #
# checkpoint plumbing
# --------------------------------------------------------------------------- #
IDENTITY_LAYOUT = "identity"
CONV1D_LAYOUT = "conv1d_torch_out_ink"
CONV2D_LAYOUT = "conv2d_torch_out_ink"

_LAYOUT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(^|\.)fsmn_block\.weight$"), CONV1D_LAYOUT),
    (re.compile(r"(^|\.)cif_conv1d\.weight$"), CONV1D_LAYOUT),
    (re.compile(r"(^|\.)conv\.\d+\.weight$"), CONV2D_LAYOUT),
)

#: Public mapping ``"<key suffix regex>" -> layout rule name`` for consumers that
#: need to build a conversion table themselves.
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
    """Layout rule derived from the *owning module type*, not from its name."""
    if isinstance(module, mxnn.Conv1d):
        return CONV1D_LAYOUT
    if isinstance(module, mxnn.Conv2d):
        return CONV2D_LAYOUT
    return IDENTITY_LAYOUT


def convert_checkpoint_tensor(name: str, array: mx.array, rule: str | None = None) -> mx.array:
    """Convert one PyTorch checkpoint tensor into the MLX layout.

    ``Linear``/``LayerNorm``/``Embedding`` weights already match.  Only
    ``Conv1d``/``Conv2d`` kernels need a transpose.

    Parameters
    ----------
    name:
        Parameter name, used only when ``rule`` is not supplied.
    array:
        The tensor to convert.
    rule:
        Explicit rule name (from :func:`flatten_state_dict_with_rules`).  Preferred
        over name matching because it cannot mis-convert a renamed submodule.
    """
    if rule is None:
        rule = _rule_for_name(name)
    if rule == CONV1D_LAYOUT:
        # torch: (out, in/groups, k) -> mlx: (out, k, in/groups)
        return mx.transpose(array, (0, 2, 1))
    if rule == CONV2D_LAYOUT:
        # torch: (out, in, kh, kw) -> mlx: (out, kh, kw, in)
        return mx.transpose(array, (0, 2, 3, 1))
    if rule not in (IDENTITY_LAYOUT, None):
        raise ValueError(f"unknown layout rule {rule!r} for parameter {name!r}")
    return array


def _rule_for_name(name: str) -> str:
    """Name-based fallback for :func:`convert_checkpoint_tensor`."""
    for pattern, rule in _LAYOUT_RULES:
        if pattern.search(name):
            return rule
    return IDENTITY_LAYOUT


def flatten_state_dict(module: mxnn.Module, prefix: str = "") -> dict[str, mx.array]:
    """Flatten a module tree into FunASR/PyTorch-compatible parameter names.

    Names are built from the Module's **mapping storage** (see
    :func:`_flatten_into`), which is the only place MLX can hold a tensor.  The
    result matches what ``torch.nn.Module.state_dict()`` produces for the
    corresponding PyTorch modules.
    """
    return _flatten(module, prefix)[0]


def flatten_state_dict_with_rules(
    module: mxnn.Module, prefix: str = ""
) -> tuple[dict[str, mx.array], dict[str, str]]:
    """Like :func:`flatten_state_dict`, but also returns the per-key layout rule."""
    return _flatten(module, prefix)


def _flatten(module: mxnn.Module, prefix: str = "") -> tuple[dict[str, mx.array], dict[str, str]]:
    tensors: dict[str, mx.array] = {}
    rules: dict[str, str] = {}
    _flatten_into(module, prefix, tensors, rules)
    return tensors, rules


def attention_blocked(mask: mx.array) -> mx.array:
    """Boolean ``True`` where attention is blocked, for any mask dtype.

    The reference does ``mask.unsqueeze(1).eq(0)`` on the **float** 1.0/0.0 mask
    that FunASR's ``sequence_mask`` produces (its ``dtype`` defaults to
    ``torch.float32``), so ``.eq(0)`` marks the padded positions.

    Rather than re-implementing ``== 0`` on a dtype-agnostic array -- whose
    answer depends on MLX's comparison type promotion between ``bool`` and
    ``int``, i.e. whether ``False == 0`` is honoured -- this branches on the
    mask's own dtype explicitly:

    * ``bool`` masks mean "True = valid", so blocked is the logical negation.
      ``logical_not`` is unambiguous and needs no promotion.
    * numeric masks reproduce the reference's ``.eq(0)`` exactly, comparing
      within a single dtype.

    Both conversions are exact for the float masks this port actually produces,
    so behaviour is unchanged while a boolean mask stops being silently ignored.
    """
    if mask.dtype == mx.bool_:
        blocked = mx.logical_not(mask)
    else:
        blocked = mask == 0
    return mx.expand_dims(blocked, 1)


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
            # Only the kernel needs a transpose.  The rule follows the OWNING
            # module, so a conv's 1-D ``bias`` would inherit it and then be
            # transposed with three axes; the bias is IDENTITY in both layouts.
            rules[name] = _module_layout_rule(module) if key == "weight" else IDENTITY_LAYOUT
        elif isinstance(value, Mapping):
            # A plain dict of tensors stored directly on a Module.
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, mx.array):
                    tensors[f"{name}.{sub_key}"] = sub_value
                    rules[f"{name}.{sub_key}"] = _module_layout_rule(module)
                elif isinstance(sub_value, mxnn.Module):
                    _flatten_into(sub_value, f"{name}.{sub_key}", tensors, rules)


def checkpoint_state_dict(module: mxnn.Module) -> dict[str, mx.array]:
    """Return the module's parameters as ``{pytorch_name: mlx_array}``.

    The names are byte-compatible with the FunASR ``.pt`` checkpoints; the
    arrays are already in MLX layout.
    """
    return flatten_state_dict(module)


def _resolve_child(module: mxnn.Module, key: str) -> Any:
    """Fetch a child by mapping key, falling back to a plain attribute.

    Numeric keys (``"0"``, ``"1"``, ...) live in the mapping storage and are
    reachable through ``Module.__getattr__``, but going through the mapping
    explicitly keeps this independent of MLX's attribute fallbacks.
    """
    if key in dict.keys(module):
        return dict.__getitem__(module, key)
    return getattr(module, key)


def _assign(module: mxnn.Module, key_parts: Sequence[str], array: mx.array) -> None:
    node: Any = module
    for part in key_parts[:-1]:
        node = _resolve_child(node, part)
    # ``mx.array`` values go into the mapping storage via MLX's ``__setattr__``.
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

    Parameters
    ----------
    module:
        Destination module.
    tensors:
        ``{name: array}`` using FunASR/PyTorch names, e.g.
        ``encoder.encoders.0.self_attn.fsmn_block.weight``.  The *arrays* must
        already be in MLX layout unless ``convert=True``.
    prefix:
        Optional key prefix shared by ``tensors`` (e.g. ``"model."``).
    convert:
        ``True`` when ``tensors`` come straight from a PyTorch ``state_dict``;
        applies :func:`convert_checkpoint_tensor` (conv weight transposes).
    dtype:
        Optional cast applied *after* the layout conversion.

    Returns
    -------
    list[str]
        The names that were loaded, sorted.

    Raises
    ------
    CheckpointKeyError
        If any expected parameter is missing or any checkpoint tensor is
        unexpected.  Shape mismatches raise ``ValueError``.
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
            # The rule comes from the owning module type, so a renamed
            # parameter still gets the right layout conversion.
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


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #
class ModuleStack(mxnn.Module):
    """A stack of sibling modules with PyTorch-compatible parameter names.

    FunASR's ``repeat()`` builds an ``nn.Sequential`` (``encoders.0.<leaf>``) and
    the SenseVoice encoder uses ``nn.ModuleList`` (``encoders.0.<leaf>``).
    MLX's own containers would nest everything under ``layers``, so children
    are registered directly under their numeric index.

    ``mlx.nn.Module`` subclasses ``dict``, so those children live in the mapping
    storage under the keys ``"0"``, ``"1"``, ... -- which is exactly the PyTorch
    ``state_dict`` spelling.  The dunders below therefore forward to ``dict`` and
    must not shadow it: MLX internally calls ``dst[key]``, ``key in dst`` and
    ``dst.items()``, so ``__getitem__`` accepts string keys unchanged.
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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Chain blocks with canonical ``nn.Sequential`` semantics.

        The leading argument is threaded through the chain while the remaining
        positional arguments and the keywords stay fixed.  Feeding a block's
        *return value* back as ``*args`` instead would unpack an ``mx.array``
        along axis 0 and hand the next block ``B`` separate arguments, which
        only works at batch size 1 or when the arities happen to line up.
        """
        if not args:
            raise TypeError(f"{type(self).__name__}.__call__() requires at least one argument")
        head, *rest = args
        for block in self:
            head = block(head, *rest, **kwargs)
        return head


def repeat(num_blocks: int, factory: Callable[[int], mxnn.Module]) -> ModuleStack:
    """Port of ``funasr.models.transformer.utils.repeat.repeat``."""
    if num_blocks < 0:
        raise ValueError(f"num_blocks must be >= 0, got {num_blocks}")
    return ModuleStack([factory(index) for index in range(num_blocks)])


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
class LayerNorm(mxnn.LayerNorm):
    """``funasr.models.transformer.layer_norm.LayerNorm`` (eps ``1e-12``).

    Parameter names are ``weight``/``bias`` and the normalisation axis is the
    last one, exactly like the PyTorch original.  MLX performs the reduction in
    float32 internally, which matches ``torch.nn.LayerNorm`` for float32 inputs.
    """

    def __init__(self, num_features: int, dim: int = -1, eps: float = 1e-12) -> None:
        super().__init__(num_features, eps=eps)
        self._dim = dim


class Float32LayerNorm(mxnn.LayerNorm):
    """``funasr.models.sense_voice.model.LayerNorm`` (eps ``1e-5``, fp32 math).

    SenseVoice keeps its own ``LayerNorm`` that inherits the ``torch`` default
    ``eps=1e-5`` and forces the reduction to float32 before casting the result
    back to the input dtype.  Reproducing the explicit cast keeps the fp16 path
    numerically identical to the reference.
    """

    def __init__(self, num_features: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__(num_features, eps=eps)
        self._dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        out_dtype = x.dtype
        return super().__call__(x.astype(mx.float32)).astype(out_dtype)


# --------------------------------------------------------------------------- #
# positional encodings
# --------------------------------------------------------------------------- #
class PositionalEncoding(mxnn.Module):
    """``funasr.models.transformer.embedding.PositionalEncoding``.

    Fixed sin/cos table scaled by ``sqrt(d_model)``.  The table is not a
    checkpoint parameter in FunASR (it is a plain attribute there), so it is
    regenerated on demand here instead of being stored as a buffer.

    **Interleaved, not concatenated.**  The reference writes
    ``pe[:, 0::2] = sin(...)`` / ``pe[:, 1::2] = cos(...)``, i.e. even columns
    hold sin and odd columns hold cos.  Do not "simplify" this into a
    ``concatenate([sin, cos])``: that is a different table and every column from
    ``d_model // 2`` onward would be wrong.  Note that
    :class:`SinusoidalPositionEncoder` genuinely *does* concatenate (see
    ``embedding.py``), so the two must not be made to match.
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
        # ``reverse`` flips the position index here; the reference instead
        # reverses an already-built table of length ``max_len``.  Identical for
        # ``reverse=False``, which is the only configured path.
        if self._reverse:
            positions = mx.arange(length - 1, -1, -1).astype(mx.float32)[:, None]
        else:
            positions = mx.arange(0, length).astype(mx.float32)[:, None]
        # reference: torch.arange(0, d_model, 2) * -(log(10000) / d_model)
        div_term = mx.exp(
            mx.arange(0, self._d_model, 2).astype(mx.float32) * -(math.log(10000.0) / self._d_model)
        )
        phase = positions * div_term  # (length, d_model // 2)
        # stack(...).reshape(...) yields [sin0, cos0, sin1, cos1, ...], which is
        # exactly pe[:, 0::2] = sin / pe[:, 1::2] = cos.
        return (
            mx.stack([mx.sin(phase), mx.cos(phase)], axis=-1)
            .reshape(length, self._d_model)
            .astype(dtype)
        )

    def __call__(self, x: mx.array) -> mx.array:
        return x * self._xscale + self._table(x.shape[1], x.dtype)[None, : x.shape[1]]


class SinusoidalPositionEncoder(mxnn.Module):
    """``funasr.models.transformer.embedding.SinusoidalPositionEncoder``.

    Note the reference divides ``log(10000)`` by ``depth / 2 - 1`` (not by
    ``depth / 2``), i.e. positions start at ``1`` and the wavelength exponent
    denominator is ``depth / 2 - 1``.  Both quirks are preserved.
    """

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


# --------------------------------------------------------------------------- #
# feed forward
# --------------------------------------------------------------------------- #
class PositionwiseFeedForward(mxnn.Module):
    """``funasr.models.transformer.positionwise_feed_forward.PositionwiseFeedForward``."""

    def __init__(self, idim: int, hidden_units: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.w_1 = mxnn.Linear(idim, hidden_units)
        self.w_2 = mxnn.Linear(hidden_units, idim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_2(mx.maximum(self.w_1(x), 0.0))


class PositionwiseFeedForwardDecoderSANM(mxnn.Module):
    """``funasr.models.sanm.positionwise_feed_forward.PositionwiseFeedForwardDecoderSANM``.

    Two details matter for checkpoint compatibility: ``w_2`` has **no bias** and
    the ``norm`` sits *between* the two linears and uses ``eps=1e-12``.
    """

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


# --------------------------------------------------------------------------- #
# attention
# --------------------------------------------------------------------------- #
class MultiHeadedAttentionSANM(mxnn.Module):
    """``funasr.models.sanm.attention.MultiHeadedAttentionSANM``.

    Fused QKV projection (``linear_q_k_v``) plus a depth-wise FSMN memory block
    whose output is added to the attention output.  The reference applies
    ``d_k ** -0.5`` to the queries *after* splitting heads (equivalent to
    scaling the scores by ``1/sqrt(d_k)``).
    """

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
        """``(B, T, n_feat)`` -> ``(B, head, T, d_k)``."""
        b, t, _ = x.shape
        return mx.transpose(mx.reshape(x, (b, t, self._h, self._d_k)), (0, 2, 1, 3))

    def forward_qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        q_k_v = self.linear_q_k_v(x)
        q, k, v = mx.split(q_k_v, 3, axis=-1)
        return self._split_heads(q), self._split_heads(k), self._split_heads(v), v

    def forward_fsmn(self, inputs: mx.array, mask: mx.array | None) -> mx.array:
        """Depth-wise FSMN memory branch; ``inputs`` is ``(B, T, D)`` (MLX NLC)."""
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
            # (B, 1, T) -> (B, 1, 1, T) boolean "is padding" mask.
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
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask)
        q_h = q_h * (self._d_k**-0.5)
        scores = mx.matmul(q_h, mx.swapaxes(k_h, -1, -2))
        return self.forward_attention(v_h, scores, mask) + fsmn_memory


class MultiHeadedAttentionSANMDecoder(mxnn.Module):
    """``funasr.models.sanm.attention.MultiHeadedAttentionSANMDecoder``.

    Pure FSMN memory block over the decoder input.  ``cache`` mirrors the
    reference streaming state; offline inference passes ``None``.

    Deviation: the cache is kept channels-last ``(b, L, d)`` whereas the
    reference keeps ``(b, d, L)``.  See the module docstring's "Known
    deviations" section -- this matters only for the streaming/export contract,
    not for offline inference.
    """

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
        """``inputs`` is ``(B, T, D)`` (MLX NLC); ``cache`` mirrors the reference."""
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
    """``funasr.models.sanm.attention.MultiHeadedAttentionCrossAtt``."""

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


# --------------------------------------------------------------------------- #
# encoder / decoder layers
# --------------------------------------------------------------------------- #
class EncoderLayerSANM(mxnn.Module):
    """``funasr.models.sanm.encoder.EncoderLayerSANM`` (offline inference path).

    The first layer (``encoders0``) has ``in_size != size``; in that case the
    self-attention residual is dropped exactly like the reference does.
    """

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
    """``funasr.models.paraformer.decoder.DecoderLayerSANM``.

    Note the ordering: the feed-forward runs **first** and the self-attention
    residual is taken against the block *input* (``residual``), not against the
    feed-forward output.  ``src_attn`` may be ``None`` (the ``decoders2`` stack).
    """

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


# --------------------------------------------------------------------------- #
# subsampling front-ends
# --------------------------------------------------------------------------- #
class TooShortUttError(Exception):
    """Port of ``funasr.models.transformer.utils.subsampling.TooShortUttError``."""

    def __init__(self, message: str, actual_size: int, limit: int) -> None:
        super().__init__(message)
        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(embed: mxnn.Module, size: int) -> tuple[bool, int]:
    """Port of ``funasr.models.transformer.utils.subsampling.check_short_utt``."""
    if isinstance(embed, Conv2dSubsampling2) and size < 3:
        return True, 3
    if isinstance(embed, Conv2dSubsampling) and size < 7:
        return True, 7
    return False, -1


class Conv2dSubsampling(mxnn.Module):
    """``funasr.models.transformer.utils.subsampling.Conv2dSubsampling`` (1/4).

    Two strided 3x3 convolutions over the ``(B, T, F, 1)`` view followed by a
    linear projection.  Parameter keys stay ``conv.<i>.weight`` and
    ``out.0.weight`` thanks to :class:`ModuleStack`.

    MLX convolutions are channels-last, so the channel axis comes **last**
    (``(B, T, F, C)``).  The flatten before ``out.0`` is transposed back to
    PyTorch's ``(C, F)`` row-major order, which is what ``out.0`` was trained on.
    """

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
        # (B, T, F, C) -> (B, T, C, F) -> flatten in PyTorch's (C, F) order.
        x = mx.reshape(mx.transpose(x, (0, 1, 3, 2)), (b, t, c * f))
        x = self.out(x)
        if x_mask is None:
            return x, None
        return x, decimate_mask(decimate_mask(x_mask, 2, 2), 2, 2)


class Conv2dSubsampling2(mxnn.Module):
    """``funasr.models.transformer.utils.subsampling.Conv2dSubsampling2`` (1/2)."""

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
