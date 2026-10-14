"""Native MLX inference port of **canonical OpenAI Whisper** as used by FunASR.

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/whisper.py.md
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Iterator, Sequence

import mlx.core as mx
from mlx import nn as mxnn

from ._layout_state import layout_rule as _layout_rule
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


# containers / checkpoint plumbing (self-contained by design)
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


def convert_checkpoint_tensor(name: str, array: mx.array, rule: str = IDENTITY_LAYOUT):
    """Convert one PyTorch checkpoint tensor into the MLX layout.

    Full notes: docs/port-notes/functions.md :: convert_checkpoint_tensor
    """
    if rule == CONV1D_LAYOUT:
        # torch (out, in, k) -> MLX (out, k, in)
        return mx.transpose(array, (0, 2, 1))
    if rule != IDENTITY_LAYOUT:
        raise ValueError(f"unknown layout rule {rule!r} for parameter {name!r}")
    return array


def _rule_for(module: mxnn.Module) -> str:
    # see docs/port-notes/functions.md :: _rule_for
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

    Full notes: docs/port-notes/functions.md :: _flatten_into
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
    # see docs/port-notes/functions.md :: load_strict
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
    if key in dict.keys(module):
        return dict.__getitem__(module, key)
    return getattr(module, key)


def _assign(module: mxnn.Module, parts: Sequence[str], array: mx.array) -> None:
    node: Any = module
    for part in parts[:-1]:
        node = _resolve_child(node, part)
    setattr(node, parts[-1], array)


# building blocks
class Linear(mxnn.Linear):
    """``torch.nn.Linear`` alias.

    MLX already uses the ``weight (out, in)`` / ``bias (out,)`` layout and the
    same parameter names, so this is purely a readability marker and the anchor
    for the conv-layout rule detection above.
    """


class LayerNorm(mxnn.LayerNorm):
    # see docs/port-notes/functions.md :: LayerNorm

    def __init__(self, n_state: int, eps: float = 1e-5) -> None:
        super().__init__(n_state, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        out_dtype = x.dtype
        return super().__call__(x.astype(mx.float32)).astype(out_dtype)


def sinusoids(length: int, channels: int, max_timescale: int = 10000) -> mx.array:
    # see docs/port-notes/functions.md :: sinusoids
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

    Full notes: docs/port-notes/functions.md :: MultiHeadAttention
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
        # see docs/port-notes/functions.md :: cache_prefix
        return self._prefix

    def _split(self, x: mx.array) -> mx.array:
        # see docs/port-notes/functions.md :: _split
        b, t, _ = x.shape
        return mx.transpose(mx.reshape(x, (b, t, self._n_head, -1)), (0, 2, 1, 3))

    def qkv_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        # see docs/port-notes/functions.md :: qkv_attention
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

    Full notes: docs/port-notes/functions.md :: ResidualAttentionBlock
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


# dimensions / configuration
@dataclass
class ModelDimensions:
    # see docs/port-notes/functions.md :: ModelDimensions

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
    # see docs/port-notes/functions.md :: WhisperConfig

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
    # see docs/port-notes/functions.md :: whisper_config_from_dict
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


# encoder / decoder / model
class AudioEncoder(mxnn.Module):
    # see docs/port-notes/functions.md :: AudioEncoder

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

    def _weight_dtype(self) -> mx.Dtype:
        # see docs/port-notes/functions.md :: _weight_dtype
        return self.conv1.weight.dtype

    def __call__(self, x: mx.array) -> mx.array:
        """``x`` is ``(B, T', n_mels)`` -> ``(B, T'/2, n_state)``.

        Full notes: docs/port-notes/functions.md :: __call__
        """
        x = x.astype(self._weight_dtype())
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
        # Evaluate block by block instead of letting the whole 32-block graph
        # accumulate.  MLX frees a block's intermediates only once its output is
        # materialised; left lazy, every block's activation stays live until the
        # single `mx.eval` at the call site.  On the 30 s window (1500 frames)
        # that transient is the largest single allocation in the encoder path.
        # Measured on M4 (whisper-large-v3, 30 s, peak `Physical footprint` from
        # vmmap, 6 alternating rounds x 3 calls): 3600 MB -> 3000 MB, i.e.
        # -600 MB (-16.7%), reproducible to +-100 MB.  Speed is unchanged:
        # paired in-process A/B over 30 pairs gives -0.04%, CI95 [-0.36, +0.14]%
        # (median 973.6 vs 976.6 ms).  The arithmetic is untouched -- the same
        # `block(x)` calls in the same order, only the evaluation boundary moves
        # -- so results are bit-identical.
        for block in self.blocks:
            x = block(x)
            mx.eval(x)
        return self.ln_post(x)


class TextDecoder(mxnn.Module):
    # see docs/port-notes/functions.md :: TextDecoder

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
        return self._mask

    def _weight_dtype(self) -> mx.Dtype:
        # see docs/port-notes/functions.md :: _weight_dtype
        return self.ln.weight.dtype

    def offset(self, kv_cache: Mapping[str, mx.array] | None) -> int:
        # see docs/port-notes/functions.md :: offset
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
        # Follow the **decoder's own weight dtype**, not ``xa``/the mel dtype.
        # The front-end mel is float32, so casting to ``xa.dtype`` used to lock
        # the entire decoder into float32 activations against fp16 weights: a
        # mixed-dtype promotion at every Linear, every step (measured on M4:
        # 9.12 ms/step vs 2.02 ms/step for a 24-step decode).  A float32
        # checkpoint is unaffected -- the cast is a no-op there.
        emb = emb.astype(self._weight_dtype())
        for block in self.blocks:
            emb = block(emb, xa=xa, mask=mask, kv_cache=kv_cache)
        emb = self.ln(emb)
        # Tied projection (reference: x @ token_embedding.weight.T), with the weight
        # cast to the activation dtype first.  The logits are always float32.
        weight = mx.swapaxes(self.token_embedding.weight.astype(emb.dtype), -1, -2)
        return mx.matmul(emb, weight).astype(mx.float32)


class Whisper(mxnn.Module):

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
        # see docs/port-notes/functions.md :: embed_audio
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

    def state_dict(self) -> LayoutReport:
        return flatten_state_dict(self)

    def load_checkpoint(self, tensors: Mapping[str, Any], **kwargs: Any) -> LoadReport:
        return load_strict(self, tensors, **kwargs)


# front-end helpers (raw mel only -- no tokenizer, no downloads)
def reflect_pad_1d(x: mx.array, pad: int) -> mx.array:
    # see docs/port-notes/functions.md :: reflect_pad_1d
    if pad <= 0:
        return x
    length = x.shape[-1]
    if length < pad + 1:
        raise ValueError(f"reflect padding of {pad} needs an axis longer than {pad}, got {length}")
    left = mx.take(x, mx.arange(pad, 0, -1, dtype=mx.int32), axis=-1)
    right = mx.take(x, mx.arange(length - 2, length - pad - 2, -1, dtype=mx.int32), axis=-1)
    return mx.concatenate([left, x, right], axis=-1)


def pad_or_trim(x: mx.array, length: int = DEFAULT_N_SAMPLES, axis: int = -1) -> mx.array:
    # see docs/port-notes/functions.md :: pad_or_trim
    if x.shape[axis] > length:
        index = [slice(None)] * x.ndim
        index[axis] = slice(0, length)
        return x[tuple(index)]
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, length - x.shape[axis])
    return mx.pad(x, pad_width)


def _hz_to_mel(frequencies: mx.array) -> mx.array:
    # see docs/port-notes/functions.md :: _hz_to_mel
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
    # see docs/port-notes/functions.md :: _is_non_persistent_buffer
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
    # see docs/port-notes/functions.md :: mel_filterbank
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

    Full notes: docs/port-notes/functions.md :: log_mel_spectrogram
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


# raw decoding helpers (token ids only -- no text, no tokenizer)
@dataclass
class GreedyDecodeConfig:
    # see docs/port-notes/functions.md :: GreedyDecodeConfig

    max_tokens: int = 224
    eot_token_id: int | None = None
    space_token_id: int | None = None
    suppress_blank_at_first_step: bool = True
    suppress_tokens: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.suppress_tokens = tuple(int(t) for t in self.suppress_tokens)


@dataclass
class GreedyDecodeResult:
    # see docs/port-notes/functions.md :: GreedyDecodeResult

    tokens: list[list[int]]
    tokens_including_prompt: list[list[int]]
    sum_logprobs: list[float]
    avg_logprobs: list[float]


def _suppress(logits: mx.array, ids: Sequence[int]) -> mx.array:
    # see docs/port-notes/functions.md :: _suppress
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

    Full notes: docs/port-notes/functions.md :: greedy_decode
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

    Full notes: docs/port-notes/functions.md :: detect_language
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
