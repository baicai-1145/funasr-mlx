"""Native MLX inference port of OpenAI Whisper as vendored by FunASR.

Reference (read-only, ``/Users/baicai1145/repos/funasr``):

* ``funasr/models/sense_voice/whisper_lib/model.py``  (encoder/decoder core)
* ``funasr/models/sense_voice/whisper_lib/audio.py``   (log-mel front-end)
* ``funasr/models/sense_voice/whisper_lib/decoding.py`` (greedy decoding filters)
* ``funasr/models/whisper/model.py``                   (FunASR wrapper)

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
from dataclasses import dataclass, fields
from typing import Any, Iterator, Mapping, Sequence

import mlx.core as mx
from mlx import nn as mxnn

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
        return self._length

    def __iter__(self) -> Iterator[mxnn.Module]:
        for index in range(self._length):
            yield getattr(self, str(index))

    def __getitem__(self, index: int) -> mxnn.Module:
        if index < 0:
            index += self._length
        return getattr(self, str(index))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result: Any = args
        for block in self:
            result = block(*result, **kwargs)
        return result


def convert_checkpoint_tensor(name: str, array: mx.array, rule: str = IDENTITY_LAYOUT):
    """Convert one PyTorch checkpoint tensor into the MLX layout."""
    if rule == CONV1D_LAYOUT:
        # torch (out, in, k) -> MLX (out, k, in)
        return mx.transpose(array, (0, 2, 1))
    if rule != IDENTITY_LAYOUT:
        raise ValueError(f"unknown layout rule {rule!r} for parameter {name!r}")
    return array


def _rule_for(module: mxnn.Module) -> str:
    """Conv1d is the only module whose MLX layout differs from PyTorch's."""
    return CONV1D_LAYOUT if isinstance(module, mxnn.Conv1d) else IDENTITY_LAYOUT


def _flatten_into(module: Any, prefix: str, tensors: dict, rules: dict) -> None:
    for key, value in vars(module).items():
        if key.startswith("_"):
            continue
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, mxnn.Module):
            _flatten_into(value, name, tensors, rules)
        elif isinstance(value, mx.array):
            tensors[name] = value
            rules[name] = _rule_for(module)


@dataclass
class LayoutReport:
    """Flat parameter tensors plus the layout rule that applies to each key."""

    tensors: dict[str, mx.array]
    rules: dict[str, str]

    def __contains__(self, name: str) -> bool:
        return name in self.tensors


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


def _assign(module: mxnn.Module, parts: Sequence[str], array: mx.array) -> None:
    node: Any = module
    for part in parts[:-1]:
        node = getattr(node, part)
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

        Whisper applies ``(n_state // n_head) ** -0.25`` to *both* ``q`` and
        ``k``, which is exactly a ``d_k ** -0.5`` factor on the scores.
        """
        n_ctx = q.shape[2]
        scale = (self._n_state // self._n_head) ** -0.25
        q = q * scale
        k = k * scale
        qk = mx.matmul(q, k)
        if mask is not None:
            key_len = k.shape[3]
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
        # float32 softmax, then back to the query dtype (reference behaviour).
        attn = mx.softmax(qk.astype(mx.float32), axis=-1).astype(q.dtype)
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

        attn_out, _ = self.qkv_attention(q, self._split(k), self._split(v), mask)
        return attn_out


class ResidualAttentionBlock(mxnn.Module):
    """``whisper.model.ResidualAttentionBlock``.

    Both attention branches read ``self.attn_pct``/``self.resid_pct`` computed
    in ``__init__``; the exact float arithmetic (``1 - pct``) is preserved.
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

        attn_pct = 0.0 if cross_attention else 0.5
        self._attn_pct = attn_pct
        self._resid_pct = 1.0 - attn_pct
        self._n_ctx = n_text_ctx

    def __call__(
        self,
        x: mx.array,
        xa: mx.array | None = None,
        mask: mx.array | None = None,
        kv_cache: dict[str, mx.array] | None = None,
    ) -> mx.array:
        x = x + self._attn_pct * self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        x = x + self._resid_pct * self.mlp(self.mlp_ln(x))
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), kv_cache=kv_cache, xa=xa)
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
        self.conv1 = mxnn.Conv1d(n_mels, n_state, kernel_size=3, stride=2, padding=1)
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
        """``x`` is ``(B, T', n_mels)`` -> ``(B, T''/2, n_state)``."""
        x = _GELU(self.conv1(x))
        x = _GELU(self.conv2(x))
        x = x + self.positional_embedding[None, : x.shape[1], :]
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
        if x.ndim == 1:
            x = mx.expand_dims(x, 0)
        emb = self.token_embedding(x)
        pos = self.positional_embedding[offset : offset + x.shape[1]]
        emb = emb + pos.reshape(1, pos.shape[0], pos.shape[1])
        emb = emb.astype(xa.dtype)
        for block in self.blocks:
            emb = block(emb, xa=xa, mask=mask, kv_cache=kv_cache)
        emb = self.ln(emb)
        weight = mx.swapaxes(self.token_embedding.weight, -1, -2)
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

    Reflection excludes the edge element, i.e. ``x[..., -1]`` is not repeated.
    """
    if pad <= 0:
        return x
    length = x.shape[-1]
    if length < pad + 1:
        raise ValueError(f"reflect padding of {pad} needs an axis longer than {pad}, got {length}")
    left = mx.take(x, mx.arange(1, pad + 1, dtype=mx.int32), axis=-1)
    right = mx.take(x, mx.arange(length - pad - 1, length - 1, dtype=mx.int32), axis=-1)
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
    # (B, frames-1, freq) @ (freq, n_mels) -> (B, frames-1, n_mels)
    log_spec = mx.matmul(magnitudes, filters.astype(magnitudes.dtype))
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
    """Set the given vocabulary columns to ``-inf`` (a port of the logit filters)."""
    banned = [int(t) for t in ids]
    if not banned:
        return logits
    vocab = mx.arange(logits.shape[-1], dtype=mx.int32)[None, :]
    mask = vocab == mx.array(banned, dtype=mx.int32)[None, :]
    return mx.where(mask, mx.array(-float("inf"), logits.dtype), logits)


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
        logprobs = mx.logsoftmax(last.astype(mx.float32), axis=-1)
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

    cache_len = max((v.shape[1] for v in kv_cache.values()), default=0)
    bound = model.dims.n_text_ctx + cfg.max_tokens
    if cache_len > bound:
        raise RuntimeError(
            f"KV cache grew to {cache_len} steps, above the {bound} bound implied "
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
    logprobs = mx.logsoftmax(logits[:, -1, :].astype(mx.float32), axis=-1)
    values = logprobs[0].tolist()
    scored = [(ids[i], float(values[tok])) for i, tok in enumerate(ids)]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
