"""MLX inference port of FunASR's ``Paraformer``.

Reference (read-only, ``/Users/baicai1145/repos/funasr``):

* ``funasr/models/sanm/encoder.py`` -> :class:`SANMEncoder`
* ``funasr/models/paraformer/cif_predictor.py`` -> :class:`CifPredictorV2`,
  :func:`cif_v1`, :func:`cif_wo_hidden_v1`
* ``funasr/models/paraformer/decoder.py`` -> :class:`ParaformerSANMDecoder`
* ``funasr/models/paraformer/model.py`` -> :class:`Paraformer`
* ``funasr/models/ctc/ctc.py`` -> :class:`CTC`
* ``funasr/utils/timestamp_tools.py`` ->
  :func:`ts_prediction_lfr6_standard` (timestamp projection)

What is ported
--------------
``encoder -> CIF predictor -> SANM decoder -> log_softmax``, i.e. the complete
non-autoregressive inference path, plus greedy token decoding and the
CIF-peak-based character timestamps.

What is *not* ported (deliberately)
----------------------------------
Label-smoothing / CTC / MAE losses, the training-time ``sampler``
(``sampling_ratio``), beam search with CTC or n-gram scorers, SpecAug and
feature extraction.  LoRA variants of the SANM blocks are rejected explicitly
instead of being silently ignored.

Parity notes
------------
* The decoder consumes the **acoustic embeddings produced by the CIF predictor**
  as ``ys_in_pad``; ``self.embed`` is *not* applied during inference (it is only
  used by the training sampler), which is why the checkpoint still carries
  ``decoder.embed.0.weight``.
* ``CifPredictorV2.tail_process_fn`` reproduces the reference's shift-based mask
  trick exactly, including the behaviour that a full-length (unpadded) utterance
  gets a zero tail.
* ``decoder.decoders2.w_2`` has **no bias** (``PositionwiseFeedForwardDecoderSANM``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import mlx.core as mx
from mlx import nn as mxnn

from .sanm import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    DecoderLayerSANM,
    EncoderLayerSANM,
    LayerNorm,
    ModuleStack,
    MultiHeadedAttentionCrossAtt,
    MultiHeadedAttentionSANM,
    MultiHeadedAttentionSANMDecoder,
    PositionalEncoding,
    PositionwiseFeedForward,
    PositionwiseFeedForwardDecoderSANM,
    SinusoidalPositionEncoder,
    checkpoint_state_dict,
    check_short_utt,
    constant_pad_1d,
    load_strict,
    make_pad_mask,
    repeat,
    sequence_mask,
)

__all__ = [
    "CTC",
    "CIF_DEVICE",
    "CifPredictorConfig",
    "CifPredictorV2",
    "Paraformer",
    "ParaformerConfig",
    "ParaformerDecoderConfig",
    "ParaformerSANMDecoder",
    "SANMEncoder",
    "SANMEncoderConfig",
    "cif_v1",
    "cif_wo_hidden_v1",
    "paraformer_config_from_dict",
    "ts_prediction_lfr6_standard",
]


def _filter(cls: type, data: Mapping[str, Any] | None) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in (data or {}).items() if k in known}


_POS_ENC_CLASSES = {
    "SinusoidalPositionEncoder": SinusoidalPositionEncoder,
    "PositionalEncoding": PositionalEncoding,
}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class SANMEncoderConfig:
    """Mirror of FunASR's ``encoder_conf`` for ``SANMEncoder`` (Paraformer side)."""

    output_size: int = 512
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 50
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    input_layer: str = "pe"
    pos_enc_class: str = "SinusoidalPositionEncoder"
    normalize_before: bool = True
    concat_after: bool = False
    positionwise_layer_type: str = "linear"
    positionwise_conv_kernel_size: int = 1
    padding_idx: int = -1
    kernel_size: int = 11
    sanm_shfit: int = 0
    selfattention_layer_type: str = "sanm"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None, **kwargs: Any):
        merged = dict(_filter(cls, data))
        merged.update(_filter(cls, kwargs))
        return cls(**merged)


@dataclass
class ParaformerDecoderConfig:
    """Mirror of FunASR's ``decoder_conf`` for ``ParaformerSANMDecoder``."""

    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 16
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    self_attention_dropout_rate: float = 0.1
    src_attention_dropout_rate: float = 0.1
    input_layer: str = "embed"
    use_output_layer: bool = True
    wo_input_layer: bool = False
    normalize_before: bool = True
    concat_after: bool = False
    att_layer_num: int = 16
    kernel_size: int = 11
    sanm_shfit: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None, **kwargs: Any):
        merged = dict(_filter(cls, data))
        merged.update(_filter(cls, kwargs))
        return cls(**merged)


@dataclass
class CifPredictorConfig:
    """Mirror of FunASR's ``predictor_conf`` for ``CifPredictorV2``."""

    idim: int = 512
    threshold: float = 1.0
    l_order: int = 1
    r_order: int = 1
    dropout: float = 0.1
    smooth_factor: float = 1.0
    noise_threshold: float = 0.0
    tail_threshold: float = 0.45
    tail_mask: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None, **kwargs: Any):
        merged = dict(_filter(cls, data))
        merged.update(_filter(cls, kwargs))
        return cls(**merged)


@dataclass
class CTCConfig:
    """Mirror of FunASR's ``ctc_conf`` (inference-relevant fields only)."""

    dropout_rate: float = 0.0
    ctc_type: str = "builtin"
    reduce: bool = True
    ignore_nan_grad: bool = True
    extra_linear: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None, **kwargs: Any):
        merged = dict(_filter(cls, data))
        merged.update(_filter(cls, kwargs))
        return cls(**merged)


@dataclass
class ParaformerConfig:
    """Mirror of FunASR's ``model_conf`` for ``Paraformer``.

    Defaults follow the published ``paraformer-zh`` configuration
    (``input_size=560`` LFR fbank, 50 SANM blocks of width 512, 16 decoder
    blocks, CIF ``tail_threshold=0.45``).
    """

    input_size: int = 560
    vocab_size: int = 8408
    ignore_id: int = -1
    blank_id: int = 0
    sos: int = 1
    eos: int = 2
    ctc_weight: float = 0.0
    lsm_weight: float = 0.1
    length_normalized_loss: bool = True
    predictor_weight: float = 1.0
    predictor_bias: int = 1
    sampling_ratio: float = 0.75
    encoder: str = "SANMEncoder"
    decoder: str = "ParaformerSANMDecoder"
    predictor: str = "CifPredictorV2"
    encoder_conf: SANMEncoderConfig = None  # type: ignore[assignment]
    decoder_conf: ParaformerDecoderConfig = None  # type: ignore[assignment]
    predictor_conf: CifPredictorConfig = None  # type: ignore[assignment]
    ctc_conf: CTCConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if isinstance(self.encoder_conf, Mapping):
            self.encoder_conf = SANMEncoderConfig.from_dict(self.encoder_conf)
        if isinstance(self.decoder_conf, Mapping):
            self.decoder_conf = ParaformerDecoderConfig.from_dict(self.decoder_conf)
        if isinstance(self.predictor_conf, Mapping):
            self.predictor_conf = CifPredictorConfig.from_dict(self.predictor_conf)
        if isinstance(self.ctc_conf, Mapping):
            self.ctc_conf = CTCConfig.from_dict(self.ctc_conf)
        if self.encoder_conf is None:
            self.encoder_conf = SANMEncoderConfig()
        if self.decoder_conf is None:
            self.decoder_conf = ParaformerDecoderConfig()
        if self.predictor_conf is None:
            # ``idim`` must follow the encoder width unless given explicitly.
            self.predictor_conf = CifPredictorConfig(idim=self.encoder_conf.output_size)
        if self.ctc_conf is None:
            self.ctc_conf = CTCConfig()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None, **kwargs: Any):
        data = dict(data or {})
        data.update(kwargs)
        encoder_conf = data.pop("encoder_conf", None)
        decoder_conf = data.pop("decoder_conf", None)
        predictor_conf = data.pop("predictor_conf", None)
        ctc_conf = data.pop("ctc_conf", None)
        merged = dict(_filter(cls, data))
        sections: dict[str, Any] = {
            "encoder_conf": SANMEncoderConfig.from_dict(encoder_conf),
            "decoder_conf": ParaformerDecoderConfig.from_dict(decoder_conf),
            "ctc_conf": CTCConfig.from_dict(ctc_conf),
        }
        if predictor_conf is not None:
            sections["predictor_conf"] = CifPredictorConfig.from_dict(predictor_conf)
        return cls(**sections, **merged)


def paraformer_config_from_dict(data: Mapping[str, Any] | None = None, **kwargs: Any):
    """Build a :class:`ParaformerConfig` from a FunASR ``model.yaml`` mapping."""
    data = dict(data or {})
    if "model_conf" in data and isinstance(data["model_conf"], Mapping):
        merged = dict(data["model_conf"])
        for key in ("encoder_conf", "decoder_conf", "predictor_conf", "ctc_conf"):
            if key in data and key not in merged:
                merged[key] = data[key]
        data = merged
    return ParaformerConfig.from_dict(data)


# --------------------------------------------------------------------------- #
# CTC head (shared shape with the SenseVoice port, kept local for independence)
# --------------------------------------------------------------------------- #
class CTC(mxnn.Module):
    """``funasr.models.ctc.ctc.CTC`` inference path (``ctc_lo`` projection)."""

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: bool = True,
        extra_linear: bool = True,
    ) -> None:
        super().__init__()
        self.ctc_lo = mxnn.Linear(encoder_output_size, odim) if extra_linear else None

    def __call__(self, hs_pad: mx.array) -> mx.array:
        return hs_pad if self.ctc_lo is None else self.ctc_lo(hs_pad)

    def log_softmax(self, hs_pad: mx.array) -> mx.array:
        return mxnn.log_softmax(self(hs_pad), axis=-1)

    def argmax(self, hs_pad: mx.array) -> mx.array:
        return mx.argmax(self(hs_pad), axis=-1)


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #
class SANMEncoder(mxnn.Module):
    """``funasr.models.sanm.encoder.SANMEncoder`` (offline inference path).

    Differences from the reference that do not affect checkpoint parity:
    intermediate CTC / ``interctc_layer_idx`` outputs and streaming
    ``forward_chunk`` are not ported (both are training / streaming features).
    """

    def __init__(
        self,
        input_size: int,
        config: SANMEncoderConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        cfg = config or SANMEncoderConfig.from_dict(kwargs)
        if cfg.selfattention_layer_type != "sanm":
            raise NotImplementedError(
                f"only selfattention_layer_type='sanm' is supported, got "
                f"{cfg.selfattention_layer_type!r}"
            )
        if cfg.positionwise_layer_type != "linear":
            raise NotImplementedError(
                f"only positionwise_layer_type='linear' is supported, got "
                f"{cfg.positionwise_layer_type!r}"
            )
        pos_enc_name = cfg.pos_enc_class
        if pos_enc_name not in _POS_ENC_CLASSES:
            raise NotImplementedError(f"unsupported pos_enc_class: {pos_enc_name!r}")
        pos_enc_cls = _POS_ENC_CLASSES[pos_enc_name]

        self._output_size = cfg.output_size
        self._normalize_before = cfg.normalize_before
        self.embed = self._build_embed(cfg, input_size, pos_enc_cls)

        def _encoder_layer(in_size: int) -> EncoderLayerSANM:
            return EncoderLayerSANM(
                in_size,
                cfg.output_size,
                MultiHeadedAttentionSANM(
                    cfg.attention_heads,
                    in_size,
                    cfg.output_size,
                    cfg.attention_dropout_rate,
                    cfg.kernel_size,
                    cfg.sanm_shfit,
                ),
                PositionwiseFeedForward(cfg.output_size, cfg.linear_units, cfg.dropout_rate),
                cfg.dropout_rate,
                cfg.normalize_before,
                cfg.concat_after,
            )

        self.encoders0 = repeat(1, lambda _i: _encoder_layer(input_size))
        self.encoders = repeat(cfg.num_blocks - 1, lambda i: _encoder_layer(cfg.output_size))
        self.after_norm = LayerNorm(cfg.output_size) if cfg.normalize_before else None

    @staticmethod
    def _build_embed(cfg: SANMEncoderConfig, input_size: int, pos_enc_cls):
        layer = cfg.input_layer
        if layer == "linear":
            return ModuleStack(
                [
                    mxnn.Linear(input_size, cfg.output_size),
                    LayerNorm(cfg.output_size),
                    mxnn.ReLU(),
                    pos_enc_cls(cfg.output_size, cfg.positional_dropout_rate),
                ]
            )
        if layer == "conv2d":
            return Conv2dSubsampling(input_size, cfg.output_size, cfg.dropout_rate)
        if layer == "conv2d2":
            return Conv2dSubsampling2(input_size, cfg.output_size, cfg.dropout_rate)
        if layer == "embed":
            # FunASR's ``nn.Embedding(..., padding_idx=...)`` only affects weight
            # *initialisation*; MLX's ``Embedding`` has no such argument, and the
            # forward pass is identical, so it is intentionally dropped here.
            return ModuleStack(
                [
                    mxnn.Embedding(input_size, cfg.output_size),
                    SinusoidalPositionEncoder(),
                ]
            )
        if layer is None:
            if input_size == cfg.output_size:
                return None
            return mxnn.Linear(input_size, cfg.output_size)
        if layer == "pe":
            return SinusoidalPositionEncoder()
        raise NotImplementedError(
            f"unsupported input_layer {layer!r}; ported: 'linear', 'conv2d', 'conv2d2', "
            "'embed', 'pe', None"
        )

    def output_size(self) -> int:
        return self._output_size

    def __call__(self, xs_pad: mx.array, ilens: mx.array) -> tuple[mx.array, mx.array]:
        masks = mx.expand_dims(~make_pad_mask(ilens, xs_pad.shape[1]), 1)  # (B, 1, T)
        xs_pad = xs_pad * (self.output_size() ** 0.5)

        embed = self.embed
        if embed is None:
            pass
        elif isinstance(embed, (Conv2dSubsampling, Conv2dSubsampling2)):
            short_status, limit_size = check_short_utt(embed, xs_pad.shape[1])
            if short_status:
                raise ValueError(
                    f"has {xs_pad.shape[1]} frames and is too short for subsampling "
                    f"(it needs more than {limit_size} frames), return empty results"
                )
            xs_pad, masks = embed(xs_pad, masks)
        else:
            xs_pad = embed(xs_pad)

        for layer in self.encoders0:
            xs_pad, masks = layer(xs_pad, masks)
        for layer in self.encoders:
            xs_pad, masks = layer(xs_pad, masks)

        if self._normalize_before:
            xs_pad = self.after_norm(xs_pad)
        olens = mx.sum(mx.squeeze(masks, 1), axis=-1).astype(mx.int32)
        return xs_pad, olens


# --------------------------------------------------------------------------- #
# CIF predictor
# --------------------------------------------------------------------------- #
CIF_DEVICE = mx.cpu
"""Device the CIF predictor evaluates on.

Deterministic **CPU** by default.  On Metal the index arrays this code depends
on (``time_of_rank``, and therefore ``cur_p`` / ``cur_h`` / ``cur_f``) have been
observed to come back holding *float* bit patterns instead of indices once the
whole graph is evaluated, so results vary between calls and between runs.  The
failure is upstream of any algebraic detail: barriers and ``cache_limit=0`` do
not suppress it, and the same input arrives unchanged.  Evaluation is
deterministic on CPU, so the predictor is pinned there.

This is not a warmup and not a retry: a warmup would still execute on the
corrupting device, and there is no warmup that makes the graph itself correct.
It is also **not** "all CPU" -- the caller passes arrays on whatever stream it
is already using, only this subtree evaluates on CPU, and the returned arrays
are ordinary MLX arrays that move back freely.  The surrounding encoder, decoder
and the rest of the Paraformer are untouched and stay on the GPU.

Passing ``device=mx.gpu`` explicitly re-enables the GPU path.  It is retained
for future verification only and must not be treated as correct until the
upstream corruption is resolved.
"""


def _cif_stream(device: mx.array | None):
    """Resolve the evaluation device; ``None`` means the deterministic default."""
    return CIF_DEVICE if device is None else device


def _cif_wo_hidden_v1_impl(alphas: mx.array) -> tuple[mx.array, mx.array]:
    """The math of ``cif_wo_hidden_v1`` with no device context of its own.

    Private so the public wrapper can hold one CPU stream around the *whole*
    computation; a nested or partially applied context would let an
    intermediate land on the default device.
    """
    # The reference accumulates in float64 and only then narrows:
    #   torch.cumsum(prefix_sum, dim=1, dtype=torch.float64).to(torch.float32)
    # ``mx.cumsum(..., dtype=...)`` casts the input to that type before summing,
    # so this is the same operation, not an approximation of it.  The narrowing
    # cast matters: a running sum that lands just below an integer in float64
    # can round up to exactly that integer in float32 and fire a token the
    # reference would not.
    prefix_sum = mx.cumsum(alphas, axis=1, dtype=mx.float64).astype(mx.float32)
    prefix_sum_floor = mx.floor(prefix_sum)
    dislocation_prefix_sum = mx.roll(prefix_sum, shift=1, axis=1)
    dislocation_prefix_sum_floor = mx.floor(dislocation_prefix_sum)
    b, t = dislocation_prefix_sum_floor.shape
    dislocation_prefix_sum_floor = mx.concatenate(
        [mx.zeros((b, 1), dislocation_prefix_sum_floor.dtype), dislocation_prefix_sum_floor[:, 1:]],
        axis=1,
    )
    dislocation_diff = prefix_sum_floor - dislocation_prefix_sum_floor
    fire_idxs = dislocation_diff > 0
    one = mx.array(1.0, alphas.dtype)
    zero = mx.array(0.0, alphas.dtype)
    fires = mx.where(fire_idxs, one, zero) + prefix_sum - prefix_sum_floor
    return fires, fire_idxs


def cif_wo_hidden_v1(
    alphas: mx.array,
    threshold: float = 1.0,
    return_fire_idxs: bool = False,
    *,
    device: mx.array | None = None,
):
    """Port of ``funasr.models.paraformer.cif_predictor.cif_wo_hidden_v1``.

    ``threshold`` is unused in the reference implementation (it only gates the
    downstream comparison) but is kept for signature parity.

    Evaluated on :data:`CIF_DEVICE` unless ``device`` overrides it; the whole
    subtree is forced inside that stream so no intermediate escapes.

    The prefix sum accumulates in ``float64`` and is narrowed to ``float32``
    afterwards, exactly like the reference's
    ``torch.cumsum(..., dtype=torch.float64).to(torch.float32)``.
    """
    with mx.stream(_cif_stream(device)):
        fires, fire_idxs = _cif_wo_hidden_v1_impl(alphas)
        mx.eval(fires, fire_idxs)
    if return_fire_idxs:
        return fires, fire_idxs
    return fires


def _cif_v1_impl(
    hidden: mx.array, alphas: mx.array, threshold: float = 1.0
) -> tuple[mx.array, mx.array]:
    """The math of :func:`cif_v1` with no device context of its own."""
    fires, fire_idxs = _cif_wo_hidden_v1_impl(alphas)
    b, t, d = hidden.shape

    total_fires = int(mx.sum(mx.sum(fire_idxs.astype(mx.int32), axis=-1)).item())
    if total_fires == 0:
        # Very short / silent segment: the reference returns zero frames here
        # (the upstream fix that avoids an IndexError in the export variant).
        max_label_len = int(mx.max(mx.round(mx.sum(alphas, axis=-1))).item())
        return mx.zeros((b, max_label_len, d), hidden.dtype), fires

    batch_len = mx.sum(fire_idxs.astype(mx.int32), axis=-1)  # (B,)
    max_label_len = int(mx.max(mx.round(mx.sum(alphas, axis=-1))).item())
    max_label_len = max(max_label_len, 1)
    max_label_len = min(max_label_len, t)

    # ``time_of_rank[b, j]`` = time of the j-th fire of utterance b.
    # The sort key is **tie-free**: fires carry 0..batch_len-1 and non-fires carry
    # t..2t-1, so the resulting order never depends on sort stability.  Keying all
    # non-fires with a single sentinel value left the tail of the permutation
    # backend-dependent, which is a needless nondeterminism source.
    rank = mx.cumsum(fire_idxs.astype(mx.int32), axis=1) - 1
    filler = t + mx.arange(t, dtype=mx.int32)[None, :]
    time_of_rank = mx.argsort(mx.where(fire_idxs, rank, filler), axis=1)  # (B, T)

    # Gather along the **time** axis natively rather than flattening the batch.
    # The reference's ``prefix_sum_hidden[fire_idxs]`` over a (B, T, D) tensor is a
    # batched gather on axis 1; the flattened form instead reshapes (B, T, D) to
    # (B*T, D), adds a per-row ``b * T`` offset and takes on axis 0, then reshapes
    # back.  That round-trip through a flat view is the part worth falsifying on
    # GPU, and it is also why a missing row offset was silent at B=1.
    def _gather(array: mx.array) -> mx.array:
        """Gather time steps of ``array`` in fire order, independently per row.

        ``mx.take_along_axis`` indexes the time axis directly, so no flatten,
        offset arithmetic or output reshape is involved.  Per the installed stub
        (``mlx/core/__init__.pyi:1901-1909``) ``indices`` must be "broadcastable
        with the input array excluding the ``axis`` dimension", so a trailing
        singleton axis broadcasts one time index across every feature lane and
        the result keeps ``(B, T, D)``.
        """
        if array.ndim == 2:
            # The fires signal carries no feature axis, so index it directly.
            return mx.take_along_axis(array, time_of_rank, axis=1)
        indices = mx.expand_dims(time_of_rank, -1)  # (B, T, 1) broadcasts over D
        return mx.take_along_axis(array, indices, axis=1)

    prefix_sum_hidden = mx.cumsum(mx.expand_dims(alphas, -1) * hidden, axis=1)

    # One boundary per fire, then subtract the previous one.
    #
    #   boundary[j] = P[t_j] - rem[t_j] * h[t_j]
    #   frames[j]   = boundary[j] - boundary[j-1]          (boundary[-1] = 0)
    #
    # Expanding the second line gives exactly the reference expression
    #   P[t_j] - P[t_{j-1}] + rem[t_{j-1}] * h[t_{j-1}] - rem[t_j] * h[t_j]
    # with ``rem[t] = fires[t] - floor(fires[t])`` and every previous term zero at
    # j = 0.  That is the same algebra the reference reaches by rolling the
    # gathered frames by one and zeroing the first entry of each batch
    # (``cif_predictor.py:881-895``); this form needs a single gather per operand
    # instead of two, and builds no separate "previous" tensors and no
    # first-position ``where`` masks.
    cur_p = _gather(prefix_sum_hidden)[:, :max_label_len]
    cur_h = _gather(hidden)[:, :max_label_len]
    cur_f = _gather(fires)[:, :max_label_len]
    boundary = cur_p - mx.expand_dims(cur_f - mx.floor(cur_f), -1) * cur_h

    previous = mx.concatenate([mx.zeros((b, 1, d), boundary.dtype), boundary[:, :-1]], axis=1)
    frames = boundary - previous

    valid = mx.arange(max_label_len, dtype=mx.int32)[None, :] < batch_len[:, None]
    frames = mx.where(mx.expand_dims(valid, -1), frames, mx.array(0.0, frames.dtype))
    return frames, fires


def cif_v1(
    hidden: mx.array,
    alphas: mx.array,
    threshold: float = 1.0,
    *,
    device: mx.array | None = None,
):
    """Port of ``funasr.models.paraformer.cif_predictor.cif_v1``.

    Returns ``(frame_fires, fires)`` where ``frame_fires`` has shape
    ``(B, max_label_len, D)``.  ``fires`` is the ``cif_peak`` signal used for
    timestamps (values ``>= 1.0`` mark an integration boundary).

    The fire ordering, the boundary algebra and the lengths mask are the
    reference's; only the *device* differs, and it is pinned to
    :data:`CIF_DEVICE` because the Metal backend has been observed to corrupt
    the index arrays this depends on.  See :data:`CIF_DEVICE` for the evidence
    and for why this is not a warmup.  ``device=mx.gpu`` opts back into the GPU
    path for future verification; it is not known to be correct.

    ``mx.eval`` returns ``None`` here (unlike ``torch``), so the values are
    read back from the already-materialised arrays rather than its return value.
    """
    with mx.stream(_cif_stream(device)):
        frames, fires = _cif_v1_impl(hidden, alphas, threshold)
        mx.eval(frames, fires)
    return frames, fires


class CifPredictorV2(mxnn.Module):
    """``funasr.models.paraformer.cif_predictor.CifPredictorV2`` (inference path).

    Parameters: ``cif_conv1d`` (a **dense**, i.e. ``groups=1``, ``Conv1d`` in V2)
    and ``cif_output``.  ``__call__`` returns the 4-tuple
    ``(acoustic_embeds, token_num, alphas, cif_peak)`` of the reference.
    """

    def __init__(
        self,
        idim: int,
        l_order: int = 1,
        r_order: int = 1,
        threshold: float = 1.0,
        dropout: float = 0.1,
        smooth_factor: float = 1.0,
        noise_threshold: float = 0.0,
        tail_threshold: float = 0.0,
        tail_mask: bool = True,
    ) -> None:
        super().__init__()
        self._l_order = l_order
        self._r_order = r_order
        self._threshold = threshold
        self._smooth_factor = smooth_factor
        self._noise_threshold = noise_threshold
        self._tail_threshold = tail_threshold
        self._tail_mask = tail_mask
        self.cif_conv1d = mxnn.Conv1d(idim, idim, l_order + r_order + 1)
        self.cif_output = mxnn.Linear(idim, 1)

    def tail_process_fn(self, hidden: mx.array, alphas: mx.array, mask: mx.array | None):
        """Port of ``CifPredictorV2.tail_process_fn`` with ``tail_mask=True``.

        The reference builds ``tail_threshold * (cat([ones, mask]) - cat([mask,
        zeros]))`` and adds it to a zero-padded ``alphas``.  Working through that
        shift-and-subtract, the coefficient is ``1.0`` at index ``i`` when
        ``mask[i-1] != mask[i]``, and at the **appended** slot when
        ``mask[T-1] == 1``.  So:

        * a **full** mask (the utterance fills the padded window) puts the
          threshold on the **appended** column -- it is *not* zeroed;
        * a **padded** mask puts it on the **first padded frame**, and the
          appended column stays zero.

        Both branches are reproduced verbatim; see
        ``tests/test_asr.py::test_cif_tail_lands_on_the_first_padded_frame_or_the_appended_slot``.

        ``token_num`` is floored here (reference ``cif_predictor.py:461``), which
        is the only place a floor belongs.
        """
        b, _, d = hidden.shape
        if mask is not None and self._tail_mask:
            zeros_t = mx.zeros((b, 1), mx.float32)
            ones_t = mx.ones((b, 1), mx.float32)
            mask_1 = mx.concatenate([mask, zeros_t], axis=1)
            mask_2 = mx.concatenate([ones_t, mask], axis=1)
            tail_threshold = (mask_2 - mask_1) * self._tail_threshold
            alphas = mx.concatenate([alphas, zeros_t], axis=1)
            alphas = alphas + tail_threshold
        else:
            tail = mx.array([[self._tail_threshold]], alphas.dtype)
            tail = mx.broadcast_to(tail, (b, 1)) if b > 1 else tail
            alphas = mx.concatenate([alphas, tail], axis=1)
        hidden = mx.concatenate([hidden, mx.zeros((b, 1, d), hidden.dtype)], axis=1)
        token_num = mx.sum(alphas, axis=-1)
        return hidden, alphas, mx.floor(token_num)

    def _alphas(self, hidden: mx.array, mask: mx.array | None) -> tuple[mx.array, mx.array | None]:
        # MLX conv1d is channels-last (NLC), so ``hidden`` (B, T, D) feeds the
        # Conv1d directly and both reference transposes disappear.  Padding is
        # applied on the time axis, i.e. axis=1.
        queries = constant_pad_1d(hidden, self._l_order, self._r_order, axis=1)
        output = mx.maximum(self.cif_conv1d(queries), 0.0)  # (B, T, D)
        output = self.cif_output(output)  # (B, T, 1)
        alphas = mx.sigmoid(output)
        alphas = mx.maximum(alphas * self._smooth_factor - self._noise_threshold, 0.0)
        mask_2d = None
        if mask is not None:
            mask_2d = mx.transpose(mask, (0, 2, 1)).astype(mx.float32).squeeze(-1)
            alphas = alphas * mx.expand_dims(mask_2d, -1)
        return alphas.squeeze(-1), mask_2d

    def __call__(self, hidden: mx.array, mask: mx.array | None = None):
        """Return ``(acoustic_embeds, token_num, alphas, cif_peak)``."""
        alphas, mask_2d = self._alphas(hidden, mask)

        if self._tail_threshold > 0.0:
            hidden, alphas, token_num = self.tail_process_fn(hidden, alphas, mask_2d)
        else:
            # Reference ``cif_predictor.py:293`` is a plain ``alphas.sum(-1)`` with
            # **no floor**; ``hidden`` is likewise left unextended because the
            # reference has no ``else`` branch at all when tail_threshold <= 0.
            # Flooring here would change ``round(token_num)`` downstream, which
            # the reference's ``Paraformer.forward`` applies to the raw sum.
            token_num = mx.sum(alphas, axis=-1)

        acoustic_embeds, cif_peak = cif_v1(hidden, alphas, self._threshold)
        if self._tail_threshold > 0.0:
            token_num_int = int(mx.max(token_num).item())
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]
        return acoustic_embeds, token_num, alphas, cif_peak


# --------------------------------------------------------------------------- #
# decoder
# --------------------------------------------------------------------------- #
class ParaformerSANMDecoder(mxnn.Module):
    """``funasr.models.paraformer.decoder.ParaformerSANMDecoder`` (offline path).

    Three stacks: ``decoders`` (``att_layer_num`` layers with SANM self-attention
    *and* cross-attention), optional ``decoders2`` (self-attention only) and a
    single ``decoders3`` feed-forward-only layer.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        config: ParaformerDecoderConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        cfg = config or ParaformerDecoderConfig.from_dict(kwargs)
        attention_dim = encoder_output_size
        self._normalize_before = cfg.normalize_before

        if cfg.wo_input_layer:
            self.embed = None
        elif cfg.input_layer == "embed":
            # The reference intentionally drops the positional encoder here.
            self.embed = ModuleStack([mxnn.Embedding(vocab_size, attention_dim)])
        elif cfg.input_layer == "linear":
            self.embed = ModuleStack(
                [
                    mxnn.Linear(vocab_size, attention_dim),
                    LayerNorm(attention_dim),
                    mxnn.ReLU(),
                    PositionalEncoding(attention_dim, cfg.positional_dropout_rate),
                ]
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {cfg.input_layer}")

        self.after_norm = LayerNorm(attention_dim) if cfg.normalize_before else None
        self.output_layer = mxnn.Linear(attention_dim, vocab_size) if cfg.use_output_layer else None

        sanm_shfit = (cfg.kernel_size - 1) // 2 if cfg.sanm_shfit is None else cfg.sanm_shfit

        def _cross_attn() -> MultiHeadedAttentionCrossAtt:
            return MultiHeadedAttentionCrossAtt(
                cfg.attention_heads, attention_dim, cfg.src_attention_dropout_rate
            )

        def _self_attn(shift: int) -> MultiHeadedAttentionSANMDecoder:
            return MultiHeadedAttentionSANMDecoder(
                attention_dim, cfg.self_attention_dropout_rate, cfg.kernel_size, shift
            )

        def _ff() -> PositionwiseFeedForwardDecoderSANM:
            return PositionwiseFeedForwardDecoderSANM(
                attention_dim, cfg.linear_units, cfg.dropout_rate
            )

        self.decoders = repeat(
            cfg.att_layer_num,
            lambda i: DecoderLayerSANM(
                attention_dim,
                _self_attn(sanm_shfit),
                _cross_attn(),
                _ff(),
                cfg.dropout_rate,
                cfg.normalize_before,
                cfg.concat_after,
            ),
        )
        remaining = cfg.num_blocks - cfg.att_layer_num
        self.decoders2 = (
            repeat(
                remaining,
                lambda i: DecoderLayerSANM(
                    attention_dim,
                    _self_attn(0),
                    None,
                    _ff(),
                    cfg.dropout_rate,
                    cfg.normalize_before,
                    cfg.concat_after,
                ),
            )
            if remaining > 0
            else None
        )
        self.decoders3 = repeat(
            1,
            lambda i: DecoderLayerSANM(
                attention_dim,
                None,
                None,
                _ff(),
                cfg.dropout_rate,
                cfg.normalize_before,
                cfg.concat_after,
            ),
        )

    def __call__(
        self,
        hs_pad: mx.array,
        hlens: mx.array,
        ys_in_pad: mx.array,
        ys_in_lens: mx.array,
        return_hidden: bool = False,
    ):
        """``hs_pad`` is the encoder/acoustic memory ``(B, T1, D)``.

        Following the reference (``funasr/models/paraformer/decoder.py:397-450``)
        the offline path does **not** run ``self.embed``: ``x = tgt`` with
        ``tgt = ys_in_pad``.  Only ``forward_one_step`` (``:604``) embeds.  In
        offline Paraformer the decoder input is therefore the CIF acoustic
        embedding, already ``(B, T2, D)`` -- not token ids.
        """
        tgt = ys_in_pad
        tgt_mask = mx.expand_dims(sequence_mask(ys_in_lens, tgt.shape[1]), 2)  # (B, T2, 1)
        memory = hs_pad
        memory_mask = mx.expand_dims(sequence_mask(hlens, memory.shape[1]), 1)  # (B, 1, T1)

        x = tgt
        for stack in (self.decoders, self.decoders2, self.decoders3):
            if stack is None:
                continue
            for layer in stack:
                x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory, memory_mask)

        if self._normalize_before:
            hidden = self.after_norm(x)
        else:
            hidden = x

        olens = mx.sum(mx.squeeze(tgt_mask, -1), axis=-1).astype(mx.int32)
        if self.output_layer is not None and not return_hidden:
            return self.output_layer(hidden), olens
        return hidden, olens


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@dataclass
class ParaformerOutputs:
    """Bundle returned by :meth:`Paraformer.forward`."""

    encoder_out: mx.array
    encoder_out_lens: mx.array
    acoustic_embeds: mx.array
    token_length: mx.array
    alphas: mx.array
    cif_peak: mx.array
    decoder_out: mx.array | None
    decoder_out_lens: mx.array | None


class Paraformer(mxnn.Module):
    """``funasr.models.paraformer.model.Paraformer`` (inference only)."""

    def __init__(self, config: ParaformerConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        cfg = config if config is not None else paraformer_config_from_dict(None, **kwargs)
        self.config = cfg
        self.encoder = SANMEncoder(input_size=cfg.input_size, config=cfg.encoder_conf)
        encoder_output_size = self.encoder.output_size()

        self.blank_id = cfg.blank_id
        self.sos = cfg.sos if cfg.sos is not None else cfg.vocab_size - 1
        self.eos = cfg.eos if cfg.eos is not None else cfg.vocab_size - 1
        self.vocab_size = cfg.vocab_size
        self.ignore_id = cfg.ignore_id
        self.encoder_output_size = encoder_output_size

        if cfg.ctc_weight == 1.0:
            self.decoder = None
        else:
            self.decoder = ParaformerSANMDecoder(
                vocab_size=cfg.vocab_size,
                encoder_output_size=encoder_output_size,
                config=cfg.decoder_conf,
            )
        self.ctc = (
            None
            if cfg.ctc_weight == 0.0
            else CTC(
                odim=cfg.vocab_size, encoder_output_size=encoder_output_size, **vars(cfg.ctc_conf)
            )
        )
        self.predictor = CifPredictorV2(**vars(cfg.predictor_conf))
        self.predictor_bias = cfg.predictor_bias

    # -- checkpoint helpers ------------------------------------------------- #
    def state_dict(self) -> dict[str, mx.array]:
        """``{pytorch_name: mlx_array}`` in MLX layout."""
        return checkpoint_state_dict(self)

    def load_checkpoint(self, tensors: Mapping[str, Any], **kwargs: Any) -> list[str]:
        """Strict load; see :func:`funasr_mlx.models.sanm.load_strict`."""
        return load_strict(self, tensors, **kwargs)

    # -- forward ------------------------------------------------------------ #
    def encode(self, speech: mx.array, speech_lengths: mx.array) -> tuple[mx.array, mx.array]:
        if speech.ndim == 2:
            speech = mx.expand_dims(speech, 0)
        return self.encoder(speech, speech_lengths)

    def calc_predictor(self, encoder_out: mx.array, encoder_out_lens: mx.array):
        encoder_out_mask = mx.expand_dims(~make_pad_mask(encoder_out_lens, encoder_out.shape[1]), 1)
        return self.predictor(encoder_out, encoder_out_mask)

    def forward(self, speech: mx.array, speech_lengths: mx.array) -> ParaformerOutputs:
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        acoustic_embeds, token_length, alphas, cif_peak = self.calc_predictor(
            encoder_out, encoder_out_lens
        )
        token_length = mx.round(token_length).astype(mx.int32)

        decoder_out = None
        decoder_out_lens = None
        if self.decoder is not None and int(mx.max(token_length).item()) >= 1:
            logits, decoder_out_lens = self.decoder(
                encoder_out, encoder_out_lens, acoustic_embeds, token_length
            )
            decoder_out = mxnn.log_softmax(logits, axis=-1)
        return ParaformerOutputs(
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            acoustic_embeds=acoustic_embeds,
            token_length=token_length,
            alphas=alphas,
            cif_peak=cif_peak,
            decoder_out=decoder_out,
            decoder_out_lens=decoder_out_lens,
        )

    def __call__(self, speech: mx.array, speech_lengths: mx.array) -> ParaformerOutputs:
        return self.forward(speech, speech_lengths)

    def _greedy_from_outputs(self, outs: ParaformerOutputs) -> list[list[int]]:
        if outs.decoder_out is None:
            return [[] for _ in range(outs.token_length.shape[0])]
        ids = mx.argmax(outs.decoder_out, axis=-1)  # (B, T2, V) -> (B, T2)
        results: list[list[int]] = []
        for b in range(ids.shape[0]):
            length = int(outs.token_length[b].item())
            length = max(0, min(length, ids.shape[1]))
            tokens = [int(t) for t in ids[b, :length].tolist()]
            results.append([t for t in tokens if t not in (self.eos, self.sos, self.blank_id)])
        return results

    def greedy_decode(self, speech: mx.array, speech_lengths: mx.array) -> list[list[int]]:
        """Greedy token ids per utterance.

        Mirrors ``Paraformer.inference`` without beam search: ``argmax`` over the
        decoder posteriors followed by dropping ``sos``/``eos``/``blank_id``
        (the reference wraps the sequence in sos/eos and then strips it).
        """
        return self._greedy_from_outputs(self.forward(speech, speech_lengths))

    def timestamps(
        self,
        speech: mx.array,
        speech_lengths: mx.array,
        num_tokens: Sequence[int] | None = None,
        upsample_rate: int = 3,
    ):
        """CIF-peak based character timestamps in milliseconds.

        Returns ``(tokens, timestamps)`` where ``timestamps[i]`` is a list of
        ``[start_ms, end_ms]`` pairs aligned with ``tokens[i]``.
        """
        outs = self.forward(speech, speech_lengths)
        tokens = self._greedy_from_outputs(outs)
        stamps: list[list[list[int]]] = []
        width = int(outs.alphas.shape[1])
        for b, token_ids in enumerate(tokens):
            count = num_tokens[b] if num_tokens is not None else len(token_ids)
            stamps.append(
                ts_prediction_lfr6_standard(
                    outs.alphas[b, :width], outs.cif_peak[b, :width], count, upsample_rate
                )
            )
        return tokens, stamps


def ts_prediction_lfr6_standard(
    alphas: mx.array, peaks: mx.array, num_chars: int, upsample_rate: int = 3
) -> list[list[int]]:
    """Port of ``funasr.utils.timestamp_tools.ts_prediction_lfr6_standard``.

    Only the millisecond ``res`` output of the reference is produced here (the
    text dump and the ``<sil>`` bookkeeping are dropped).  ``upsample_rate`` must
    be ``1`` to match Paraformer's inference call site.
    """
    start_end_threshold = 5
    max_token_duration = 12
    time_rate = 10.0 * 6 / 1000 / upsample_rate

    alphas_list = [float(v) for v in alphas.tolist()]
    peaks_list = [float(v) for v in peaks.tolist()]
    num_frames = len(peaks_list)
    if num_frames == 0 or num_chars <= 0:
        return []

    fire_place = [i for i, p in enumerate(peaks_list) if p >= 1.0 - 1e-4]
    if len(fire_place) != num_chars + 1:
        total = sum(alphas_list) or 1.0
        alphas_list = [a / total * (num_chars + 1) for a in alphas_list]
        fire_place = _cif_fire_positions(alphas_list, threshold=1.0 - 1e-4)
    if not fire_place:
        # Degenerate case (alphas all zero): no integration boundary at all.
        return [[0, int(num_frames * time_rate * 1000)] for _ in range(num_chars)]

    timestamps: list[list[float]] = []
    if fire_place[0] > start_end_threshold:
        timestamps.append([0.0, fire_place[0] * time_rate])
    for i in range(len(fire_place) - 1):
        if max_token_duration < 0 or fire_place[i + 1] - fire_place[i] <= max_token_duration:
            timestamps.append([fire_place[i] * time_rate, fire_place[i + 1] * time_rate])
        else:
            split = fire_place[i] + max_token_duration
            timestamps.append([fire_place[i] * time_rate, split * time_rate])
            timestamps.append([split * time_rate, fire_place[i + 1] * time_rate])
    if num_frames - fire_place[-1] > start_end_threshold:
        end = (num_frames + fire_place[-1]) * 0.5
        timestamps[-1][1] = end * time_rate
        timestamps.append([end * time_rate, num_frames * time_rate])
    elif timestamps:
        timestamps[-1][1] = num_frames * time_rate

    return [[int(start * 1000), int(end * 1000)] for start, end in timestamps]


def _cif_fire_positions(alphas: Sequence[float], threshold: float) -> list[int]:
    """Fallback used when the CIF peaks do not line up with the token count.

    Mirrors ``funasr.models.paraformer.cif_predictor.cif_wo_hidden``: a frame
    "fires" when the running integral crosses an integer boundary.
    """
    integrate = 0.0
    fires: list[int] = []
    for t, alpha in enumerate(alphas):
        integrate += alpha
        if integrate >= threshold:
            fires.append(t)
            integrate -= 1.0
    return fires
