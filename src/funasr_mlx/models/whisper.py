"""Native MLX inference port of **canonical OpenAI Whisper** as used by FunASR.

Authoritative source
--------------------
``funasr/models/whisper/model.py`` (FunASR wrapper) does ``import whisper``
(line 11) and builds ``whisper.model.Whisper`` (line 60), i.e. the **external
``openai-whisper`` package**, not the vendored SenseVoice copy.  The encoder,
decoder, attention, residual block and cache semantics here are therefore ported
from the canonical ``whisper/model.py`` (OpenAI release v20250625, pinned at
``/Users/baicai1145/repos/funasr-mlx/.tmp/openai-whisper-model.py``).

The vendored ``funasr/models/sense_voice/whisper_lib/`` tree is a **different
encoder** and must not be used as the reference for the Whisper path.  The two
differ in ways that silently corrupt a canonical checkpoint:

* ``whisper_lib``'s ``conv1`` has ``stride=2``; canonical's has **no stride**.
  Canonical halves the time axis exactly once, so the input contract is
  ``2 * n_audio_ctx`` mel frames.
* ``whisper_lib``'s ``ResidualAttentionBlock`` scales the branches by
  ``attn_pct`` / ``1 - attn_pct`` with ``attn_pct = 0.0`` on cross-attention
  blocks, discarding self-attention in the decoder.  Canonical has no gates and
  orders the branches ``attn -> cross_attn -> mlp``.

``funasr/models/sense_voice/whisper_lib/audio.py`` remains the reference for the
log-mel front-end arithmetic only, since it is numerically the same STFT/Slaney
computation the canonical package performs.

There is **no** dependency on ``mlx_whisper`` / ``openai-whisper`` / any torch
package, and nothing is ever downloaded: the model is a pure
``mlx.core``/``mlx.nn`` module tree loaded from a converted safetensors file.

Scope and honesty boundary
--------------------------
The tokenizer, feature file loading, beam search, temperature fallback,
timestamp rules and the ``ct-punc`` post-processor are **not** implemented here.
What this module does expose is the raw material the lead needs to integrate
them:

* :func:`log_mel_spectrogram` -- waveform ``(B, T)`` -> log-mel ``(B, T', 80)``
* :func:`pad_or_trim` -- waveform -> exactly ``n_samples``
* :func:`greedy_decode` -- **token ids only**, never text
* :func:`detect_language` -- raw per-language-token log-probabilities, given the
  language token ids supplied by the caller

Layout conventions (documented, not accidental)
----------------------------------------------
* Mel features are **channels-last**: ``(B, T', n_mels)``.  The reference uses
  ``(B, n_mels, T')`` and a ``permute(0, 2, 1)`` after ``conv2``; MLX's Conv1d
  is already channels-last, so that permute disappears.
* MLX convolutions are channels-last, so ``encoder.conv{1,2}.weight`` is stored
  as ``(n_state, kernel, in_channels)`` and converted from torch's
  ``(n_state, in_channels, kernel)`` by :func:`convert_checkpoint_tensor`.
* The causal mask is registered with ``persistent=False`` in the reference, so it
  is regenerated here and is **not** part of the expected parameter set.
* ``encoder.positional_embedding`` *is* a persistent buffer in the reference, so
  it is materialised as a real array and round-trips through strict loading
  (its regenerated value is identical).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Iterator, Sequence

import mlx.core as mx
from mlx import nn as mxnn
from mlx.nn import log_softmax

__all__ = [
    "AudioEncoder",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_N_MELS",
    "DEFAULT_N_FFT",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_N_SAMPLES",
    "GreedyDecodeConfig",
    "GreedyDecodeResult",
    "IDENTITY_LAYOUT",
    "CONV1D_LAYOUT",
    "LayoutReport",
    "Linear",
    "LoadReport",
    "ModelDimensions",
    "NON_PERSISTENT_BUFFERS",
    "ModuleStack",
    "MultiHeadAttention",
    "ResidualAttentionBlock",
    "TextDecoder",
    "Whisper",
    "WhisperConfig",
    "CheckpointKeyError",
    "convert_checkpoint_tensor",
    "detect_language",
    "flatten_state_dict",
    "flatten_state_dict_with_rules",
    "greedy_decode",
    "load_strict",
    "log_mel_spectrogram",
    "mel_filterbank",
    "pad_or_trim",
    "reflect_pad_1d",
    "sinusoids",
    "whisper_config_from_dict",
]

#: OpenAI Whisper defaults (see ``whisper/__init__.py`` of the reference).
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_MELS = 80
DEFAULT_N_FFT = 400
DEFAULT_HOP_LENGTH = 160
DEFAULT_N_SAMPLES = 480000  # 30 s
CHUNK_LENGTH = 30

IDENTITY_LAYOUT = "identity"
CONV1D_LAYOUT = "conv1d_torch_out_ink_to_mlx_out_ki"


# --------------------------------------------------------------------------- #
# containers / checkpoint plumbing (self-contained by design)
# --------------------------------------------------------------------------- #
class ModuleStack(mxnn.Module):
    """Ordered children registered under their numeric index.

    ``torch.nn.Sequential`` yields ``<parent>.<i>.<leaf>`` parameter names; MLX's
    ``nn.Sequential`` would insert a ``layers`` level, so children are registered
    directly under ``"0"``, ``"1"``, ...
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


def convert_checkpoint_tensor(name: str, array: mx.array, rule: str = IDENTITY_LAYOUT):
    """Convert one PyTorch checkpoint tensor into the MLX layout.

    Parameters
    ----------
    name:
        Parameter name, used only for diagnostics and error messages.
    array:
        The tensor to convert.
    rule:
        **Explicit** rule name, normally taken from
        :func:`flatten_state_dict_with_rules`, which derives it from the owning
        module type.  There is deliberately **no name-based inference**: here the
        same attribute name (``conv1``/``conv2``/``linear``) can belong to an
        ``nn.Linear`` or to a ``nn.Conv1d``, so guessing from the name could
        silently apply the wrong transpose.  The default is identity, and a wrong
        choice surfaces as a clear shape mismatch at load time rather than as a
        silently mis-converted weight.
    """
    if rule == CONV1D_LAYOUT:
        # torch (out, in, k) -> MLX (out, k, in)
        return mx.transpose(array, (0, 2, 1))
    if rule != IDENTITY_LAYOUT:
        raise ValueError(f"unknown layout rule {rule!r} for parameter {name!r}")
    return array


def _rule_for(module: mxnn.Module) -> str:
    """Conv1d is the only module whose MLX layout differs from PyTorch's."""
    return CONV1D_LAYOUT if isinstance(module, mxnn.Conv1d) else IDENTITY_LAYOUT


#: Buffers canonical registers with ``persistent=False``, which are therefore
#: **absent from every published checkpoint**: its derived causal ``mask``
#: (OpenAI release v20250625 ``model.py:225``) and ``alignment_heads``
#: (``:276``, ``:285``).
#:
#: The actual table is :data:`NON_PERSISTENT_BUFFERS`, defined **after** the
#: class definitions it names and reached through
#: :func:`_is_non_persistent_buffer`, which resolves the global at call time.
#: It pairs each buffer with the module type that owns it, rather than matching
#: a leaf name, so an unrelated parameter called ``_mask`` on some other module
#: is still collected.
#:
#: Either way this is an explicit list of exactly those buffers.  A blanket
#: "skip every private array" rule would swallow real parameters -- both
#: ``positional_embedding`` buffers are *persistent* (``:181``) and must stay.


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
            # Owner + native key, so the full model, a standalone submodule and
            # any caller prefix are all covered by one entry.
            if _is_non_persistent_buffer(module, key):
                continue
            tensors[name] = value
            rules[name] = _rule_for(module) if key == "weight" else IDENTITY_LAYOUT


@dataclass
class LayoutReport(Mapping):
    """Flat parameter tensors plus the layout rule that applies to each key.

    Behaves like a read-only ``Mapping`` of ``{parameter_name: mlx_array}``, so it
    can be iterated, indexed, ``len()``-ed and fed to ``dict(...)`` exactly like a
    plain state dict, while also carrying the per-key rule used for the
    torch -> MLX layout conversion.
    """

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
    """Flatten a module tree into FunASR/PyTorch-compatible parameter names.

    The layout rule is derived from the owning module type rather than from a
    name pattern, so it cannot silently mis-convert a renamed submodule.
    Attributes starting with ``_`` are configuration, never weights.
    """
    tensors: dict[str, mx.array] = {}
    rules: dict[str, str] = {}
    _flatten_into(module, prefix, tensors, rules)
    return LayoutReport(tensors=tensors, rules=rules)


def flatten_state_dict_with_rules(
    module: mxnn.Module, prefix: str = ""
) -> tuple[dict[str, mx.array], dict[str, str]]:
    """Like :func:`flatten_state_dict`, but also returns the per-key layout rule.

    Returns ``(tensors, rules)`` with the same two dictionaries that the
    :class:`LayoutReport` carries -- nothing is dropped and the rule metadata is
    not recomputed by a second walk.  This is the contract
    :func:`funasr_mlx.models.sanm.flatten_state_dict_with_rules` and
    :func:`funasr_mlx.models.campplus.flatten_state_dict_with_rules` expose, so
    the three model modules agree on one name for the same operation; callers
    that want the richer object should use :func:`flatten_state_dict` instead.
    """
    report = flatten_state_dict(module, prefix)
    return report.tensors, report.rules


class CheckpointKeyError(KeyError):
    """Raised by :func:`load_strict` when a checkpoint does not match a module."""


@dataclass
class LoadReport:
    """Outcome of a :func:`load_strict` call."""

    loaded: list[str]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LoadReport(loaded={len(self.loaded)})"


def load_strict(
    module: mxnn.Module,
    tensors: Mapping[str, Any],
    *,
    prefix: str = "",
    convert: bool = False,
    dtype: Any | None = None,
) -> LoadReport:
    """Load ``tensors`` into ``module``, refusing any key mismatch.

    Arrays must already be in MLX layout unless ``convert=True`` (raw PyTorch
    ``state_dict``).  Raises :class:`CheckpointKeyError` listing the offending
    keys, and ``ValueError`` on any shape mismatch -- never a partial load.
    """
    report = flatten_state_dict(module)
    expected = {f"{prefix}{k}": v for k, v in report.tensors.items()}
    incoming = dict(tensors)

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
        if tuple(array.shape) != tuple(expected[name].shape):
            raise ValueError(
                f"shape mismatch for {name!r}: checkpoint {tuple(array.shape)} "
                f"vs model {tuple(expected[name].shape)}"
            )
        _assign(module, name[len(prefix) :].split("."), array)
        loaded.append(name)
    return LoadReport(loaded=loaded)


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
# building blocks
# --------------------------------------------------------------------------- #
class Linear(mxnn.Linear):
    """``torch.nn.Linear`` alias.

    MLX already uses the ``weight (out, in)`` / ``bias (out,)`` layout and the
    same parameter names, so this is purely a readability marker and the anchor
    for the conv-layout rule detection above.
    """


class LayerNorm(mxnn.LayerNorm):
    """``whisper.model.LayerNorm`` -- fp32 reduction, torch's default ``eps=1e-5``.

    The reference casts the input to ``float32`` before normalising and casts
    the result back to the input dtype; MLX's ``LayerNorm`` already accumulates
    in float32, so the explicit round-trip is kept for clarity and for the fp16
    path.
    """

    def __init__(self, n_state: int, eps: float = 1e-5) -> None:
        super().__init__(n_state, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        out_dtype = x.dtype
        return super().__call__(x.astype(mx.float32)).astype(out_dtype)


def sinusoids(length: int, channels: int, max_timescale: int = 10000) -> mx.array:
    """Port of ``whisper.model.sinusoids``.

    Note this is **0-based** and uses ``-log_increment * arange(channels // 2)``
    -- different from FunASR's ``SinusoidalPositionEncoder`` (1-based, positive
    increments), which the SANM blocks use.  Both quirks matter and both are
    preserved exactly.
    """
    if channels % 2 != 0:
        raise ValueError(f"sinusoids requires an even channel count, got {channels}")
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(
        -log_timescale_increment * mx.arange(0, channels // 2).astype(mx.float32)
    )
    scaled_time = mx.arange(0, length).astype(mx.float32)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


#: ``torch.nn.GELU()``'s default is the *exact* erf formulation, and MLX's
#: ``nn.GELU`` defaults to ``approx="none"`` (exact), so the two agree.  If MLX's
#: default ever changed, this constant would silently switch to the tanh
#: approximation -- hence the explicit note rather than an inline call.
_GELU = mxnn.GELU()


class MultiHeadAttention(mxnn.Module):
    """``whisper.model.MultiHeadAttention`` with an explicit MLX KV cache.

    Cache semantics are a direct port of the reference's forward hooks
    (``install_kv_cache_hooks``) plus the ``self.key not in kv_cache`` branch in
    ``forward``:

    * **Self-attention** (``xa is None``) always recomputes ``k``/``v`` and
      *appends* them to the cache, so ``k`` covers ``[0, offset + t)``.
    * **Cross-attention** (``xa is not None``) computes once; later steps read the
      cached tensors unchanged.

    Cache keys are strings of the form ``"blocks.<i>.attn.key"`` /
    ``"blocks.<i>.cross_attn.value"`` so the cache is inspectable and
    deterministic, instead of the reference's module-object keys.
    """

    def __init__(self, n_state: int, n_head: int, prefix: str = "") -> None:
        super().__init__()
        self._prefix = prefix
        self._n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self._n_state = n_state
        self._cache_limit: int | None = None

    @property
    def cache_prefix(self) -> str:
        """Cache key prefix for this attention (e.g. ``"blocks.0.attn"``)."""
        return self._prefix

    def _split(self, x: mx.array) -> mx.array:
        """``(B, T, C)`` -> ``(B, H, T, d)``."""
        b, t, _ = x.shape
        return mx.transpose(mx.reshape(x, (b, t, self._n_head, -1)), (0, 2, 1, 3))

    def qkv_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Port of ``MultiHeadAttention.qkv_attention`` (the non-SDPA path).

        Layouts follow the reference exactly: ``q`` is ``(B, H, T, d)`` and
        ``k`` is **transposed** to ``(B, H, d, T)``.  That ``.permute(0, 2, 3, 1)``
        in the reference is what makes ``q @ k`` contract the head dimension and
        produce the ``(B, H, T, T)`` score matrix; leaving ``k`` as
        ``(B, H, T, d)`` would contract ``T`` with ``T`` instead.

        Whisper applies ``(n_state // n_head) ** -0.25`` to *both* ``q`` and
        ``k``, which is exactly a ``d_k ** -0.5`` factor on the scores.
        """
        n_ctx = q.shape[2]
        scale = (self._n_state // self._n_head) ** -0.25
        q = q * scale
        k = k * scale
        qk = mx.matmul(q, k)  # (B, H, T, T)
        if mask is not None:
            key_len = k.shape[-1]
            # The reference slices the same ``n_ctx`` from both axes; that only
            # broadcasts for a full-length pass (n_ctx == key_len) or for the
            # single-token incremental steps this decoder performs.
            if n_ctx not in (key_len, 1):
                raise ValueError(
                    f"causal mask slice [:{n_ctx}, :{n_ctx}] does not broadcast "
                    f"against key length {key_len}; the reference only supports "
                    "full-length or one-token-at-a-time decoding"
                )
            qk = qk + mask[:n_ctx, :n_ctx]
        # The reference computes the scores in fp32, takes the softmax there and
        # casts the weights back to the query dtype.
        qk = qk.astype(mx.float32)
        attn = mx.softmax(qk, axis=-1).astype(q.dtype)
        out = mx.matmul(attn, v)
        b, _, t, _ = out.shape
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (b, t, self._n_state))
        return self.out(out), qk

    def __call__(
        self,
        x: mx.array,
        xa: mx.array | None = None,
        mask: mx.array | None = None,
        kv_cache: dict[str, mx.array] | None = None,
    ) -> mx.array:
        q = self.query(x)
        src = x if xa is None else xa
        use_cache = kv_cache is not None
        key_name = f"{self._prefix}.key"
        value_name = f"{self._prefix}.value"

        # ``xa is None`` forces the recompute branch for self-attention, which is
        # what makes the cache append and cover the current step.
        if not use_cache or xa is None or key_name not in kv_cache:
            k = self.key(src)
            v = self.value(src)
        else:
            k = kv_cache[key_name]
            v = kv_cache[value_name]

        if use_cache:
            for name, out in ((key_name, k), (value_name, v)):
                if name not in kv_cache or (
                    self._cache_limit is not None and out.shape[1] > self._cache_limit
                ):
                    kv_cache[name] = out
                else:
                    kv_cache[name] = mx.concatenate([kv_cache[name], out], axis=1)
            k = kv_cache[key_name]
            v = kv_cache[value_name]

        # Fused path: with no mask (cross-attention or any unmasked pass) the
        # whole scale+QK^T+softmax+@V chain is one SDPA kernel -- at T=1500 the
        # manual path materializes a (B,H,T,T) score matrix whose traffic
        # dominates the encoder. Whisper's dual ``d^-0.25`` scaling is exactly
        # SDPA's ``scale=d_k**-0.5``, and SDPA's internal softmax runs in
        # float32 like the reference. The causal-masked self-attention path
        # keeps the manual implementation (mask semantics + qk return value
        # are covered by tests against the reference).
        if mask is None:
            attn = mx.fast.scaled_dot_product_attention(
                self._split(q),
                self._split(k),
                self._split(v),
                scale=(self._n_state // self._n_head) ** -0.5,
            )
            b, _, t, _ = attn.shape
            attn = mx.reshape(mx.transpose(attn, (0, 2, 1, 3)), (b, t, self._n_state))
            return self.out(attn)

        attn_out, _ = self.qkv_attention(
            self._split(q),
            mx.swapaxes(self._split(k), -1, -2),
            self._split(v),
            mask,
        )
        return attn_out


class ResidualAttentionBlock(mxnn.Module):
    """``whisper.model.ResidualAttentionBlock`` (canonical OpenAI, v20250625).

    Plain pre-norm residuals with **no** weighting: ``attn``, then
    ``cross_attn``, then ``mlp``, each added at full strength.

    The SenseVoice vendored fork (``funasr/models/sense_voice/whisper_lib``)
    scales these branches by ``attn_pct`` / ``1 - attn_pct``, where
    ``attn_pct`` is ``0.0`` for cross-attention blocks -- i.e. it *discards
    self-attention entirely* in the decoder.  That fork is **not** what
    ``funasr/models/whisper/model.py`` runs: that module does ``import whisper``
    (line 11) and builds ``whisper.model.Whisper`` (line 60).  The gates are
    therefore removed, and the cross-attention branch is applied **before** the
    MLP as in the canonical source.
    """

    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
        n_text_ctx: int = 448,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head, prefix=f"{prefix}.attn")
        self.attn_ln = LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp = ModuleStack([Linear(n_state, n_mlp), mxnn.GELU(), Linear(n_mlp, n_state)])
        self.mlp_ln = LayerNorm(n_state)
        self.cross_attn = (
            MultiHeadAttention(n_state, n_head, prefix=f"{prefix}.cross_attn")
            if cross_attention
            else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        self._n_ctx = n_text_ctx

    def __call__(
        self,
        x: mx.array,
        xa: mx.array | None = None,
        mask: mx.array | None = None,
        kv_cache: dict[str, mx.array] | None = None,
    ) -> mx.array:
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa=xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


# --------------------------------------------------------------------------- #
# dimensions / configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModelDimensions:
    """Mirror of ``whisper.model.ModelDimensions``."""

    n_mels: int = 80
    n_audio_ctx: int = 1500
    n_audio_state: int = 384
    n_audio_head: int = 6
    n_audio_layer: int = 4
    n_vocab: int = 51865
    n_text_ctx: int = 448
    n_text_state: int = 384
    n_text_head: int = 6
    n_text_layer: int = 4

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None, **kwargs: Any):
        data = dict(data or {})
        data.update(kwargs)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def is_multilingual(self) -> bool:
        return self.n_vocab >= 51865

    @property
    def num_languages(self) -> int:
        return self.n_vocab - 51765 - int(self.is_multilingual)


@dataclass
class WhisperConfig:
    """Mirror of ``funasr/models/whisper/model.py::Whisper``'s inference config."""

    dims: ModelDimensions = None  # type: ignore[assignment]
    device: str = "cpu"
    fp16: bool = False
    temperature: float = 0.0
    best_of: int = 5
    beam_size: int = 5
    sample_len: int | None = None
    suppress_blank: bool = True
    suppress_tokens: tuple[int, ...] = ()
    without_timestamps: bool = True
    max_initial_timestamp: float = 1.0
    language: str | None = "en"
    task: str = "transcribe"
    model_name: str = "large-v3"

    def __post_init__(self) -> None:
        if self.dims is None:
            self.dims = ModelDimensions()
        elif isinstance(self.dims, Mapping):
            self.dims = ModelDimensions.from_dict(self.dims)
        self.suppress_tokens = tuple(int(t) for t in self.suppress_tokens)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None, **kwargs: Any):
        data = dict(data or {})
        data.update(kwargs)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def whisper_config_from_dict(data: Mapping[str, Any] | None = None, **kwargs: Any) -> WhisperConfig:
    """Build a :class:`WhisperConfig` from a FunASR ``model.yaml``-like mapping.

    ``WhisperParameters`` is also accepted as an alias for the ``parameters``
    section of FunASR's document.
    """
    data = dict(data or {})
    for section in ("model_conf", "parameters", "WhisperParameters", "parameters_inferred"):
        if section in data and isinstance(data[section], Mapping):
            data = dict(data[section])
            break
    dims = data.pop("dims", None)
    if dims is None:
        dims = {
            key: data.pop(key)
            for key in list(data)
            if key in {f.name for f in fields(ModelDimensions)}
        }
    known = {f.name for f in fields(WhisperConfig)}
    kwargs_ = {k: v for k, v in data.items() if k in known}
    return WhisperConfig(dims=ModelDimensions.from_dict(dims), **kwargs_)


# --------------------------------------------------------------------------- #
# encoder / decoder / model
# --------------------------------------------------------------------------- #
class AudioEncoder(mxnn.Module):
    """``whisper.model.AudioEncoder``.

    ``mel`` is channels-last ``(B, T', n_mels)``; the reference's
    ``permute(0, 2, 1)`` after ``conv2`` is unnecessary under MLX's channels-last
    Conv1d.  ``positional_embedding`` is materialised (the reference registers it
    as a persistent buffer) so it round-trips through strict loading; the
    regenerated value is identical to :func:`sinusoids`.
    """

    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int) -> None:
        super().__init__()
        self.conv1 = mxnn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = mxnn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = sinusoids(n_ctx, n_state)
        self.blocks = ModuleStack(
            [
                ResidualAttentionBlock(n_state, n_head, prefix=f"blocks.{i}", n_text_ctx=n_ctx)
                for i in range(n_layer)
            ]
        )
        self.ln_post = LayerNorm(n_state)
        self._n_state = n_state
        self._n_ctx = n_ctx

    def __call__(self, x: mx.array) -> mx.array:
        """``x`` is ``(B, T', n_mels)`` -> ``(B, T'/2, n_state)``.

        ``conv1`` has **no** stride and only ``conv2`` halves the time axis, so
        the contract is exactly ``2 * n_ctx`` mel frames in and ``n_ctx`` frames
        out.
        """
        x = _GELU(self.conv1(x))
        x = _GELU(self.conv2(x))

        # The canonical encoder *asserts* an exact length rather than adapting to
        # whatever it produced: ``positional_embedding`` has ``n_ctx`` rows.  The
        # previous version sliced the table down to whatever length it got, which
        # silently returned a short encoder output and is precisely what hid the
        # ``conv1`` stride bug from the test suite.
        if x.shape[1] != self.positional_embedding.shape[0]:
            raise ValueError(
                "incorrect audio shape: the encoder produced "
                f"{x.shape[1]} frames but positional_embedding expects "
                f"{self.positional_embedding.shape[0]} "
                "(pass exactly 2 * n_audio_ctx mel frames)"
            )
        x = (x + self.positional_embedding).astype(x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.ln_post(x)


class TextDecoder(mxnn.Module):
    """``whisper.model.TextDecoder`` with an explicit MLX KV cache.

    The causal mask is **not** a parameter: the reference registers it with
    ``persistent=False``, so it never appears in a ``state_dict`` and is
    regenerated on demand here.
    """

    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int) -> None:
        super().__init__()
        self.token_embedding = mxnn.Embedding(n_vocab, n_state)
        self.positional_embedding = mx.zeros((n_ctx, n_state))
        self.blocks = ModuleStack(
            [
                ResidualAttentionBlock(
                    n_state,
                    n_head,
                    cross_attention=True,
                    n_text_ctx=n_ctx,
                    prefix=f"blocks.{i}",
                )
                for i in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)
        mask = mx.arange(0, n_ctx)
        # torch: empty(n_ctx, n_ctx).fill_(-inf).triu_(1)
        self._mask = mx.where(mask[None, :] > mask[:, None], -float("inf"), mx.array(0.0))
        for block in self.blocks:
            block.attn._cache_limit = n_ctx
            if block.cross_attn is not None:
                block.cross_attn._cache_limit = n_ctx
        self._n_ctx = n_ctx
        self._offset_key = "blocks.0.attn.key" if n_layer > 0 else None

    @property
    def mask(self) -> mx.array:
        """The ``(n_ctx, n_ctx)`` lower-triangular ``-inf`` mask."""
        return self._mask

    def offset(self, kv_cache: Mapping[str, mx.array] | None) -> int:
        """Absolute position of the next token, from the self-attention cache.

        The reference reads ``next(iter(kv_cache.values())).shape[1]``, which
        relies on ``dict`` insertion order.  Here it is read deterministically
        from the first block's self-attention key.
        """
        if not kv_cache or self._offset_key is None:
            return 0
        entry = kv_cache.get(self._offset_key)
        return 0 if entry is None else int(entry.shape[1])

    def __call__(
        self,
        x: mx.array,
        xa: mx.array,
        mask: mx.array | None = None,
        kv_cache: dict[str, mx.array] | None = None,
    ) -> mx.array:
        offset = self.offset(kv_cache)
        if mask is None:
            mask = self.mask
        if x.ndim == 1:
            x = mx.expand_dims(x, 0)
        emb = self.token_embedding(x)
        pos = self.positional_embedding[offset : offset + x.shape[1]]
        emb = emb + pos.reshape(1, pos.shape[0], pos.shape[1])
        emb = emb.astype(xa.dtype)
        for block in self.blocks:
            emb = block(emb, xa=xa, mask=mask, kv_cache=kv_cache)
        emb = self.ln(emb)
        # Tied projection (reference: x @ token_embedding.weight.T), with the weight
        # cast to the activation dtype first.  The logits are always float32.
        weight = mx.swapaxes(self.token_embedding.weight.astype(emb.dtype), -1, -2)
        return mx.matmul(emb, weight).astype(mx.float32)


class Whisper(mxnn.Module):
    """``whisper.model.Whisper``.

    Only inference helpers are provided; :func:`greedy_decode` and
    :func:`detect_language` in this module consume this class.
    """

    def __init__(self, dims: ModelDimensions | None = None) -> None:
        super().__init__()
        self.dims = dims or ModelDimensions()
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        self.alignment_heads = None

    @property
    def is_multilingual(self) -> bool:
        return self.dims.is_multilingual

    @property
    def num_languages(self) -> int:
        return self.dims.num_languages

    @property
    def device(self) -> str:
        return "mlx"

    @property
    def is_loaded(self) -> bool:
        return True

    def embed_audio(self, mel: mx.array) -> mx.array:
        return self.encoder(mel)

    def logits(
        self,
        tokens: mx.array,
        audio_features: mx.array,
        kv_cache: dict[str, mx.array] | None = None,
    ) -> mx.array:
        return self.decoder(tokens, audio_features, mask=self.decoder.mask, kv_cache=kv_cache)

    def forward(self, mel: mx.array, tokens: mx.array) -> mx.array:
        return self.logits(tokens, self.embed_audio(mel))

    def __call__(self, mel: mx.array, tokens: mx.array) -> mx.array:
        return self.forward(mel, tokens)

    # -- checkpoint helpers ------------------------------------------------- #
    def state_dict(self) -> LayoutReport:
        return flatten_state_dict(self)

    def load_checkpoint(self, tensors: Mapping[str, Any], **kwargs: Any) -> LoadReport:
        return load_strict(self, tensors, **kwargs)


# --------------------------------------------------------------------------- #
# front-end helpers (raw mel only -- no tokenizer, no downloads)
# --------------------------------------------------------------------------- #
def reflect_pad_1d(x: mx.array, pad: int) -> mx.array:
    """``torch.nn.functional.pad(..., mode="reflect")`` along the last axis.

    The edge value is **not** repeated.  With ``x = [1, 2, 3, 4]`` and ``pad=2``
    on both sides the result is ``[3, 2, 1, 2, 3, 4, 3, 2]``: the left filler
    reads left-to-right as ``x[pad] ... x[1]`` and the right filler as
    ``x[n-2] ... x[n-1-pad]`` -- both *descending* index runs.
    """
    if pad <= 0:
        return x
    length = x.shape[-1]
    if length < pad + 1:
        raise ValueError(f"reflect padding of {pad} needs an axis longer than {pad}, got {length}")
    left = mx.take(x, mx.arange(pad, 0, -1, dtype=mx.int32), axis=-1)
    right = mx.take(x, mx.arange(length - 2, length - pad - 2, -1, dtype=mx.int32), axis=-1)
    return mx.concatenate([left, x, right], axis=-1)


def pad_or_trim(x: mx.array, length: int = DEFAULT_N_SAMPLES, axis: int = -1) -> mx.array:
    """Port of ``whisper.audio.pad_or_trim`` (zero-pad or truncate)."""
    if x.shape[axis] > length:
        index = [slice(None)] * x.ndim
        index[axis] = slice(0, length)
        return x[tuple(index)]
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, length - x.shape[axis])
    return mx.pad(x, pad_width)


def _hz_to_mel(frequencies: mx.array) -> mx.array:
    """Slaney mel scale (librosa default, ``htk=False``)."""
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = mx.log(mx.array(6.4)) / 27.0
    mels = frequencies / f_sp
    log_region = min_log_mel + mx.log(mx.maximum(frequencies, 1e-10) / min_log_hz) / logstep
    return mx.where(frequencies >= min_log_hz, log_region, mels)


def _mel_to_hz(mels: mx.array) -> mx.array:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = mx.log(mx.array(6.4)) / 27.0
    freqs = f_sp * mels
    log_region = min_log_hz * mx.exp(logstep * (mels - min_log_mel))
    return mx.where(mels >= min_log_mel, log_region, freqs)


NON_PERSISTENT_BUFFERS: frozenset[tuple[type, str]] = frozenset(
    {
        (TextDecoder, "_mask"),
        (Whisper, "alignment_heads"),
    }
)
"""``(owning module type, native parameter name)`` pairs canonical registers with
``persistent=False``, so they appear in no published ``state_dict``.

Defined here rather than beside :func:`_flatten_into` because it names the two
classes, which do not exist yet at that point in the module -- referencing them
in a literal there would be a ``NameError`` at import time.  :func:`_flatten_into`
looks the global up when it runs, so the ordering costs nothing.

Pairing the buffer with its owner matters: matching the leaf name alone would
silently drop an unrelated trainable parameter called ``_mask`` on any other
module, which is the blanket-private-drop failure this exists to avoid.
"""


def _is_non_persistent_buffer(owner: Any, key: str) -> bool:
    """Whether ``owner``'s parameter ``key`` is a derived, non-persistent buffer.

    ``isinstance`` rather than an exact type match, so a subclass still matches.
    The result is independent of any flatten prefix or of whether the owning
    module is reached standalone or as part of a full ``Whisper``.
    """
    return any(
        isinstance(owner, owner_type) and key == name for owner_type, name in NON_PERSISTENT_BUFFERS
    )


def mel_filterbank(
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> mx.array:
    """Slaney-normalised mel filterbank, shape ``(n_mels, n_fft // 2 + 1)``.

    Re-implementation of ``librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)``,
    which is how the reference's ``mel_filters.npz`` asset was produced.  **Not
    numerically verified against the official asset** -- if the official array is
    available, load it and pass it to :func:`log_mel_spectrogram` directly instead
    of relying on this function.
    """
    if n_fft % 2 != 0:
        raise ValueError(f"this filterbank builder assumes an even n_fft, got {n_fft}")
    if fmax is None:
        fmax = sample_rate / 2.0
    fftfreqs = mx.arange(0, n_fft // 2 + 1).astype(mx.float32) * (sample_rate / n_fft)
    mel_points = mx.linspace(
        float(_hz_to_mel(mx.array([fmin], mx.float32))[0].item()),
        float(_hz_to_mel(mx.array([fmax], mx.float32))[0].item()),
        n_mels + 2,
    )
    mel_f = _mel_to_hz(mel_points.astype(mx.float32))
    fdiff = mel_f[1:] - mel_f[:-1]
    ramps = mel_f[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = mx.maximum(mx.array(0.0), mx.minimum(lower, upper))
    # Slaney normalisation
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    return weights * enorm[:, None]


def log_mel_spectrogram(
    audio: mx.array,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    *,
    center: bool = True,
    filters: mx.array | None = None,
    max_frames: int | None = None,
) -> mx.array:
    """Port of ``whisper.audio.log_mel_spectrogram`` (channel axis last).

    Parameters
    ----------
    audio:
        Waveform ``(B, T)`` (float).  Use :func:`pad_or_trim` first.
    center:
        ``True`` reproduces ``torch.stft``'s reflect padding by ``n_fft // 2``.
    filters:
        Precomputed ``(n_mels, n_fft // 2 + 1)`` mel filterbank.  Defaults to
        :func:`mel_filterbank`.
    max_frames:
        Optional hard cap on the number of output frames, as a guard against
        unbounded allocation.

    Returns
    -------
    ``(B, T' - 1, n_mels)`` where ``T'`` is the number of STFT frames -- i.e.
    ``n_samples // hop_length`` frames, matching the reference's ``[..., :-1]``
    drop of the final frame.
    """
    if audio.ndim != 2:
        raise ValueError(f"audio must be (B, T), got {tuple(audio.shape)}")
    window = 0.5 - 0.5 * mx.cos(2.0 * math.pi * mx.arange(0, n_fft).astype(mx.float32) / n_fft)
    if center:
        audio = reflect_pad_1d(audio, n_fft // 2)
    length = audio.shape[-1]
    if length < n_fft:
        raise ValueError(f"audio needs at least {n_fft} samples after padding, got {length}")
    n_frames = 1 + (length - n_fft) // hop_length
    if max_frames is not None and n_frames > max_frames:
        raise ValueError(
            f"input yields {n_frames} mel frames, above the max_frames={max_frames} guard"
        )
    indices = mx.arange(n_frames)[:, None] * hop_length + mx.arange(n_fft)[None, :]
    flat = mx.reshape(indices, (-1,))
    frames = mx.take(mx.reshape(audio, (-1,)), flat)
    frames = mx.reshape(frames, (audio.shape[0], n_frames, n_fft)) * window
    spectrum = mx.fft.rfft(frames, axis=-1)
    # torch slices ``[..., :-1]`` along the *frame* axis.
    spectrum = spectrum[:, : n_frames - 1, :]
    magnitudes = mx.abs(spectrum) ** 2
    if filters is None:
        filters = mel_filterbank(n_mels=n_mels, n_fft=n_fft)
    # MLX's matmul does not accept a 2-D right-hand side against a 3-D left-hand
    # side (it treats the pair as batched and then requires the feature axis to
    # match), so the transposed filterbank gets an explicit leading batch axis.
    filters_t = mx.expand_dims(mx.swapaxes(filters.astype(magnitudes.dtype), -1, -2), 0)
    # (B, frames-1, freq) @ (1, freq, n_mels) -> (B, frames-1, n_mels)
    log_spec = mx.matmul(magnitudes, filters_t)
    log_spec = mx.log10(mx.maximum(log_spec, 1e-10))
    log_spec = mx.maximum(log_spec, mx.max(log_spec) - 8.0)
    return (log_spec + 4.0) / 4.0


# --------------------------------------------------------------------------- #
# raw decoding helpers (token ids only -- no text, no tokenizer)
# --------------------------------------------------------------------------- #
@dataclass
class GreedyDecodeConfig:
    """Bounds and logit filters for :func:`greedy_decode`.

    ``eot_token_id`` and ``space_token_id`` **must** be supplied by the caller
    (they depend on the BPE vocabulary of the particular checkpoint) or left as
    ``None`` to disable the corresponding filter.  No default is guessed.
    """

    max_tokens: int = 224
    eot_token_id: int | None = None
    space_token_id: int | None = None
    suppress_blank_at_first_step: bool = True
    suppress_tokens: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.suppress_tokens = tuple(int(t) for t in self.suppress_tokens)


@dataclass
class GreedyDecodeResult:
    """Token-level greedy output (ids only; decoding to text is not provided)."""

    tokens: list[list[int]]
    tokens_including_prompt: list[list[int]]
    sum_logprobs: list[float]
    avg_logprobs: list[float]


def _suppress(logits: mx.array, ids: Sequence[int]) -> mx.array:
    """Set the given vocabulary columns to ``-inf`` (a port of the logit filters).

    ``vocab`` (V,) is broadcast **against** the banned ids (N,) rather than the
    other way round, and the membership test is reduced over the N axis.  Building
    a ``(1, V)`` and a ``(1, N)`` array and comparing them would only broadcast
    when ``N == 1`` or ``N == V``; with two or more suppressed tokens and a
    vocabulary of a different size it raises instead of masking.
    """
    banned = [int(t) for t in ids]
    if not banned:
        return logits
    vocab_size = logits.shape[-1]
    vocab = mx.arange(vocab_size, dtype=mx.int32)[:, None]  # (V, 1)
    banned_arr = mx.array(banned, dtype=mx.int32)[None, :]  # (1, N)
    blocked = mx.any(vocab == banned_arr, axis=-1)  # (V,)
    return mx.where(mx.expand_dims(blocked, 0), mx.array(-float("inf"), logits.dtype), logits)


def greedy_decode(
    model: Whisper,
    audio_features: mx.array,
    prompt_tokens: mx.array,
    config: GreedyDecodeConfig | None = None,
) -> GreedyDecodeResult:
    """Autoregressive greedy decoding with a bounded KV cache.

    Semantics ported from ``whisper.decoding.DecodingTask`` with ``beam_size``
    and ``best_of`` both 1:

    * the full prompt is processed first (building the cache),
    * afterwards exactly **one** token per step is fed,
    * ``SuppressBlank`` forbids ``[space, eot]`` at the first generated position,
    * ``SuppressTokens`` forbids the caller-supplied ids at every step,
    * decoding stops at ``eot_token_id`` or after ``max_tokens``.

    Returns token ids only.  There is no beam search, no temperature fallback
    and no text conversion.
    """
    cfg = config or GreedyDecodeConfig()
    tokens = mx.asarray(prompt_tokens)
    if tokens.ndim == 1:
        tokens = mx.expand_dims(tokens, 0)
    prompt_length = int(tokens.shape[-1])
    if prompt_length == 0:
        raise ValueError("prompt_tokens must contain at least one token")
    if cfg.max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {cfg.max_tokens}")

    batch = tokens.shape[0]
    eot = cfg.eot_token_id
    kv_cache: dict[str, mx.array] = {}

    generated: list[list[int]] = [[] for _ in range(batch)]
    full: list[list[int]] = [[int(t) for t in row] for row in tokens.tolist()]
    sum_logprobs = [0.0] * batch
    finished = [False] * batch

    for step in range(cfg.max_tokens):
        logits = model.logits(tokens, audio_features, kv_cache)
        last = logits[:, -1, :]
        if cfg.suppress_blank_at_first_step and step == 0:
            last = _suppress(last, [t for t in (cfg.space_token_id, eot) if t is not None])
        last = _suppress(last, cfg.suppress_tokens)
        logprobs = log_softmax(last.astype(mx.float32), axis=-1)
        next_tokens = mx.argmax(logprobs, axis=-1)
        rows = next_tokens.tolist()
        lps = logprobs.tolist()

        still_running = False
        for b in range(batch):
            if finished[b]:
                continue
            token = int(rows[b])
            if eot is not None and token == eot:
                finished[b] = True
                continue
            generated[b].append(token)
            full[b].append(token)
            sum_logprobs[b] += float(lps[b][token])
            still_running = True
        if not still_running:
            break
        tokens = mx.expand_dims(next_tokens, -1)

    # Only the SELF-attention cache grows, one step at a time.  Cross-attention is
    # keyed and stored once at ``n_audio_ctx`` and never appended, so including
    # it would compare an audio length against a text bound -- and since
    # ``n_audio_ctx`` is larger than ``n_text_ctx`` for every published Whisper
    # (1500 vs 448), that trips the check on the very first decode.
    self_cache_len = max(
        (v.shape[1] for k, v in kv_cache.items() if "cross_attn" not in k),
        default=0,
    )
    bound = model.dims.n_text_ctx + cfg.max_tokens
    if self_cache_len > bound:
        raise RuntimeError(
            f"KV cache grew to {self_cache_len} steps, above the {bound} bound implied "
            f"by n_text_ctx={model.dims.n_text_ctx} and max_tokens={cfg.max_tokens}"
        )

    avg = [(sum_logprobs[b] / len(generated[b])) if generated[b] else 0.0 for b in range(batch)]
    return GreedyDecodeResult(
        tokens=generated,
        tokens_including_prompt=full,
        sum_logprobs=sum_logprobs,
        avg_logprobs=avg,
    )


def detect_language(
    model: Whisper,
    audio_features: mx.array,
    language_token_ids: Sequence[int],
    prompt_tokens: Sequence[int],
) -> list[tuple[int, float]]:
    """Raw per-language log-probabilities for language identification.

    Tokenizer-free port of ``whisper.decoding.detect_language``.  The caller
    supplies

    * ``prompt_tokens`` -- the SOT sequence (for the multilingual models this is
      ``tokenizer.sot_sequence_including_notimestamps``), and
    * ``language_token_ids`` -- the ``<|lang|>`` token ids,

    and receives ``(token_id, logprob)`` pairs sorted by descending
    log-probability.  Only the language-token columns are sampled, as in the
    reference.  Mapping ids back to language names is the tokenizer owner's job.
    """
    ids = [int(t) for t in language_token_ids]
    if not ids:
        raise ValueError("language_token_ids must be non-empty")
    prompt = list(int(t) for t in prompt_tokens)
    if not prompt:
        raise ValueError("prompt_tokens must be non-empty")
    tokens = mx.array([prompt], dtype=mx.int32)
    logits = model.logits(tokens, audio_features)
    logprobs = log_softmax(logits[:, -1, :].astype(mx.float32), axis=-1)
    values = logprobs[0].tolist()
    scored = [(ids[i], float(values[tok])) for i, tok in enumerate(ids)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
