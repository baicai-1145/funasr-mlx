"""MLX inference port of FunASR's ``SenseVoiceSmall``.

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/sensevoice.py.md
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import mlx.core as mx
from mlx import nn as mxnn

from ._strict_load import _filter
from .sanm import (
    EncoderLayerSANM,
    Float32LayerNorm,
    MultiHeadedAttentionSANM,
    PositionwiseFeedForward,
    SinusoidalPositionEncoder,
    checkpoint_state_dict,
    load_strict,
    repeat,
    sequence_mask,
)

__all__ = [
    "CTC",
    "EMO_DICT",
    "EVENT_EMO_IDS",
    "LID_DICT",
    "LID_INT_DICT",
    "NUM_METADATA_EMBEDDINGS",
    "NUM_PREFIX_FRAMES",
    "SenseVoiceEncoderSmall",
    "SenseVoiceEncoderSmallConfig",
    "SenseVoiceSmall",
    "SenseVoiceSmallConfig",
    "TEXTNORM_DICT",
    "TEXTNORM_INT_DICT",
    "ctc_greedy_decode",
    "sensevoice_config_from_dict",
]


# --------------------------------------------------------------------------- #
# metadata dictionaries (verbatim from sense_voice/model.py)
# --------------------------------------------------------------------------- #
LID_DICT: dict[str, int] = {
    "auto": 0,
    "zh": 3,
    "en": 4,
    "yue": 7,
    "ja": 11,
    "ko": 12,
    "nospeech": 13,
}
LID_INT_DICT: dict[int, int] = {
    24884: 3,
    24885: 4,
    24888: 7,
    24892: 11,
    24896: 12,
    24992: 13,
}
TEXTNORM_DICT: dict[str, int] = {"withitn": 14, "woitn": 15}
TEXTNORM_INT_DICT: dict[int, int] = {25016: 14, 25017: 15}
EMO_DICT: dict[str, int] = {
    "unk": 25009,
    "happy": 25001,
    "sad": 25002,
    "angry": 25003,
    "neutral": 25004,
}

#: ``self.embed = nn.Embedding(7 + len(lid_dict) + len(textnorm_dict), input_size)``
#: i.e. rows ``0..6`` are reserved for the rich-transcription/event slots.
NUM_METADATA_EMBEDDINGS = 7 + len(LID_DICT) + len(TEXTNORM_DICT)

#: Index pairs used verbatim by ``SenseVoiceSmall.encode`` / ``.inference``.
LANGUAGE_SLOT = 0
EVENT_ID, EMO_ID = 1, 2
TEXTNORM_SLOT_RANGE = (LID_DICT["nospeech"], TEXTNORM_DICT["woitn"] + 1)
EVENT_EMO_IDS = (EVENT_ID, EMO_ID)

#: language + event + emotion + textnorm
NUM_PREFIX_FRAMES = 4



# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class SenseVoiceEncoderSmallConfig:
    """Mirror of FunASR's ``encoder_conf`` for ``SenseVoiceEncoderSmall``."""

    output_size: int = 512
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 50
    tp_blocks: int = 20
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    input_layer: str = "pe"
    normalize_before: bool = True
    concat_after: bool = False
    positionwise_layer_type: str = "linear"
    positionwise_conv_kernel_size: int = 1
    padding_idx: int = -1
    kernel_size: int = 11
    sanm_shfit: int = 0
    selfattention_layer_type: str = "sanm"
    use_checkpoint: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None, **kwargs: Any):
        merged = dict(_filter(cls, data))
        merged.update(_filter(cls, kwargs))
        return cls(**merged)


@dataclass
class CTCConfig:
    """Mirror of FunASR's ``ctc_conf`` (only the inference-relevant fields)."""

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
class SenseVoiceSmallConfig:
    """Mirror of FunASR's ``model_conf`` for ``SenseVoiceSmall``.

    The defaults are the published ``SenseVoiceSmall`` values
    (``input_size=560``, 50 SANM blocks + 20 tail blocks, ``vocab_size=55465``).
    """

    input_size: int = 560
    vocab_size: int = 55465
    ignore_id: int = -1
    blank_id: int = 0
    sos: int = 1
    eos: int = 2
    length_normalized_loss: bool = True
    lsm_weight: float = 0.0
    encoder: str = "SenseVoiceEncoderSmall"
    encoder_conf: SenseVoiceEncoderSmallConfig = None  # type: ignore[assignment]
    ctc_conf: CTCConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if isinstance(self.encoder_conf, Mapping):
            self.encoder_conf = SenseVoiceEncoderSmallConfig.from_dict(self.encoder_conf)
        if isinstance(self.ctc_conf, Mapping):
            self.ctc_conf = CTCConfig.from_dict(self.ctc_conf)
        if self.encoder_conf is None:
            self.encoder_conf = SenseVoiceEncoderSmallConfig()
        if self.ctc_conf is None:
            self.ctc_conf = CTCConfig()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None, **kwargs: Any):
        data = dict(data or {})
        data.update(kwargs)
        encoder_conf = data.pop("encoder_conf", None)
        ctc_conf = data.pop("ctc_conf", None)
        merged = dict(_filter(cls, data))
        return cls(
            encoder_conf=SenseVoiceEncoderSmallConfig.from_dict(encoder_conf),
            ctc_conf=CTCConfig.from_dict(ctc_conf),
            **merged,
        )


def sensevoice_config_from_dict(data: Mapping[str, Any] | None = None, **kwargs: Any):
    """Build a :class:`SenseVoiceSmallConfig` from a FunASR ``model.yaml`` mapping.

    Accepts either the flat ``model_conf`` block or the whole document with
    ``model_conf``/``encoder_conf``/``ctc_conf`` sections.
    """
    data = dict(data or {})
    if "model_conf" in data and isinstance(data["model_conf"], Mapping):
        merged = dict(data["model_conf"])
        for key in ("encoder_conf", "ctc_conf"):
            if key in data and key not in merged:
                merged[key] = data[key]
        data = merged
    return SenseVoiceSmallConfig.from_dict(data)


# --------------------------------------------------------------------------- #
# modules
# --------------------------------------------------------------------------- #
class CTC(mxnn.Module):
    """``funasr.models.ctc.ctc.CTC`` restricted to its inference path.

    The only parameter is ``ctc_lo`` (``Linear(encoder_output_size, odim)``);
    the loss/argmax helpers of the reference are re-implemented on top of it.
    """

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
        if self.ctc_lo is None:
            return hs_pad
        return self.ctc_lo(hs_pad)

    def log_softmax(self, hs_pad: mx.array) -> mx.array:
        # ``mx.logsoftmax`` does not exist in MLX; the max-shifted functional
        # ``mlx.nn.log_softmax`` is the API.  Qualified as ``mxnn.`` because this
        # class also defines a *method* named ``log_softmax``.
        return mxnn.log_softmax(self(hs_pad), axis=-1)

    def argmax(self, hs_pad: mx.array) -> mx.array:
        return mx.argmax(self(hs_pad), axis=-1)


class SenseVoiceEncoderSmall(mxnn.Module):
    """``funasr.models.sense_voice.model.SenseVoiceEncoderSmall``.

    Full notes: docs/port-notes/functions.md :: SenseVoiceEncoderSmall
    """

    def __init__(
        self,
        input_size: int,
        config: SenseVoiceEncoderSmallConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        cfg = config or SenseVoiceEncoderSmallConfig.from_dict(kwargs)
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

        self._output_size = cfg.output_size
        # The reference hard-codes SinusoidalPositionEncoder regardless of
        # ``input_layer`` (SenseVoice has no Conv2dSubsampling front-end).
        self.embed = SinusoidalPositionEncoder()
        self._normalize_before = cfg.normalize_before

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
        self.after_norm = Float32LayerNorm(cfg.output_size)
        self.tp_encoders = repeat(cfg.tp_blocks, lambda i: _encoder_layer(cfg.output_size))
        self.tp_norm = Float32LayerNorm(cfg.output_size)

    def _weight_dtype(self) -> mx.Dtype:
        """Dtype of this encoder's checkpoint weights (``encoders0`` always exists).

        Read through :func:`~funasr_mlx.models.llm.quant.activation_dtype` rather
        than ``weight.dtype`` directly: the SANM stack is a documented
        quantization target (``audio_encoder.`` is in fun-asr-nano's scope), and
        a quantized Linear's ``weight`` holds packed ``uint32`` codes.  Casting
        the fbank input to ``uint32`` would destroy it exactly as the decoder
        embedding cast did.  The helper returns the side tensor's dtype for a
        quantized module, so this stays correct either way.
        """
        from .llm.quant import activation_dtype

        return activation_dtype(self.encoders0[0].self_attn.linear_q_k_v)

    def output_size(self) -> int:
        return self._output_size

    def __call__(self, xs_pad: mx.array, ilens: mx.array) -> tuple[mx.array, mx.array]:
        # The front-end always produces float32 fbank.  Without this cast an
        # fp16 checkpoint runs the whole 70-layer SANM stack with float32
        # activations against fp16 weights and pays a mixed-dtype promotion at
        # every Linear (measured on M4, 3.275 s window: 37.5 ms vs 19.2 ms,
        # 1.95x).  For a float32 checkpoint the cast is a no-op, so those
        # numerics stay bit-identical.
        xs_pad = xs_pad.astype(self._weight_dtype())
        maxlen = xs_pad.shape[1]
        masks = mx.expand_dims(sequence_mask(ilens, maxlen=maxlen), 1)  # (B, 1, T)
        # When every sample fills the whole window the mask is uniformly 1.0:
        # every ``where(blocked, ...)`` is a no-op and the mask products in the
        # FSMN branch are identity. Fold it to None so each layer can take the
        # fused SDPA path (numerically identical; covered by the alignment
        # suite). ``olens`` is then just ``ilens``.
        uniform = bool(mx.all(masks == 1.0).item())
        layer_mask = None if uniform else masks

        xs_pad = xs_pad * (self.output_size() ** 0.5)
        xs_pad = self.embed(xs_pad)

        for layer in self.encoders0:
            xs_pad, _ = layer(xs_pad, layer_mask)
        for layer in self.encoders:
            xs_pad, _ = layer(xs_pad, layer_mask)

        xs_pad = self.after_norm(xs_pad)
        # ``olens`` is derived here, before the tail blocks (faithful to the
        # reference; the mask is unchanged by the tail blocks anyway).
        olens = (
            mx.sum(mx.squeeze(masks, 1), axis=-1).astype(mx.int32)
            if not uniform
            else mx.asarray(ilens).reshape(-1).astype(mx.int32)
        )

        for layer in self.tp_encoders:
            xs_pad, _ = layer(xs_pad, layer_mask)
        xs_pad = self.tp_norm(xs_pad)
        return xs_pad, olens


class SenseVoiceSmall(mxnn.Module):
    """``funasr.models.sense_voice.model.SenseVoiceSmall`` (inference only)."""

    def __init__(
        self,
        config: SenseVoiceSmallConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        cfg = config if config is not None else sensevoice_config_from_dict(None, **kwargs)
        self.config = cfg
        self.encoder = SenseVoiceEncoderSmall(input_size=cfg.input_size, config=cfg.encoder_conf)
        self.ctc = CTC(
            odim=cfg.vocab_size,
            encoder_output_size=self.encoder.output_size(),
            **(vars(cfg.ctc_conf) if cfg.ctc_conf else {}),
        )
        self.embed = mxnn.Embedding(NUM_METADATA_EMBEDDINGS, cfg.input_size)

        self.blank_id = cfg.blank_id
        self.sos = cfg.sos if cfg.sos is not None else cfg.vocab_size - 1
        self.eos = cfg.eos if cfg.eos is not None else cfg.vocab_size - 1
        self.vocab_size = cfg.vocab_size
        self.ignore_id = cfg.ignore_id
        self.encoder_output_size = self.encoder.output_size()

    # -- checkpoint helpers ------------------------------------------------- #
    def state_dict(self) -> dict[str, mx.array]:
        """``{pytorch_name: mlx_array}`` in MLX layout (no key rewriting)."""
        return checkpoint_state_dict(self)

    def load_checkpoint(self, tensors: Mapping[str, Any], **kwargs: Any) -> list[str]:
        """Strict load; see :func:`funasr_mlx.models.sanm.load_strict`."""
        return load_strict(self, tensors, **kwargs)

    # -- forward ------------------------------------------------------------ #
    def _metadata_prefix(
        self,
        batch_size: int,
        language: str | int,
        use_itn: bool,
        text_norm: str | None,
    ) -> mx.array:
        if isinstance(language, str):
            lid = LID_DICT.get(language, 0)
        else:
            lid = int(language)
        if text_norm is None:
            text_norm = "withitn" if use_itn else "woitn"
        tn = TEXTNORM_DICT.get(text_norm, TEXTNORM_DICT["woitn"])

        ids = mx.array([[lid, EVENT_ID, EMO_ID, tn]], dtype=mx.int32)
        # One row of ``NUM_PREFIX_FRAMES`` token ids, tiled across the batch.
        # Note the broadcast target is ``(B, NUM_PREFIX_FRAMES)``: ``ids`` is
        # token *indices*, so its width is the number of prefix frames and has
        # nothing to do with ``input_size``.
        return self.embed(mx.broadcast_to(ids, (batch_size, NUM_PREFIX_FRAMES)))

    def encode(
        self,
        speech: mx.array,
        speech_lengths: mx.array,
        language: str | int = "auto",
        use_itn: bool = False,
        text_norm: str | None = None,
    ) -> tuple[mx.array, mx.array]:
        if speech.ndim == 2:
            speech = mx.expand_dims(speech, 0)
        batch_size = speech.shape[0]
        prefix = self._metadata_prefix(batch_size, language, use_itn, text_norm)
        speech = mx.concatenate([prefix, speech], axis=1)
        speech_lengths = speech_lengths + NUM_PREFIX_FRAMES
        return self.encoder(speech, speech_lengths)

    def forward_logits(
        self,
        speech: mx.array,
        speech_lengths: mx.array,
        language: str | int = "auto",
        use_itn: bool = False,
        text_norm: str | None = None,
        ban_emo_unk: bool = False,
    ) -> mx.array:
        encoder_out, _ = self.encode(speech, speech_lengths, language, use_itn, text_norm)
        logits = self.ctc.log_softmax(encoder_out)
        if ban_emo_unk:
            unk = EMO_DICT["unk"]
            if unk >= logits.shape[-1]:
                raise ValueError(f"emo_dict['unk']={unk} is outside vocab_size={logits.shape[-1]}")
            vocab_ids = mx.arange(logits.shape[-1], dtype=mx.int32)[None, None, :]
            logits = mx.where(
                vocab_ids == unk,
                mx.array(-float("inf"), logits.dtype),
                logits,
            )
        return logits

    def __call__(
        self,
        speech: mx.array,
        speech_lengths: mx.array,
        language: str | int = "auto",
        use_itn: bool = False,
        text_norm: str | None = None,
        ban_emo_unk: bool = False,
    ) -> mx.array:
        return self.forward_logits(
            speech, speech_lengths, language, use_itn, text_norm, ban_emo_unk
        )

    def greedy_decode(
        self,
        speech: mx.array,
        speech_lengths: mx.array,
        language: str | int = "auto",
        use_itn: bool = False,
        text_norm: str | None = None,
        ban_emo_unk: bool = False,
    ) -> list[list[int]]:
        """Greedy CTC token ids per utterance (mirrors ``SenseVoiceSmall.inference``)."""
        logits = self.forward_logits(
            speech, speech_lengths, language, use_itn, text_norm, ban_emo_unk
        )
        encoder_out_lens = speech_lengths + NUM_PREFIX_FRAMES
        if encoder_out_lens.ndim > 1:
            encoder_out_lens = encoder_out_lens.reshape(-1)
        return ctc_greedy_decode(logits, encoder_out_lens, self.blank_id)


def ctc_greedy_decode(
    log_probs: mx.array,
    lengths: mx.array | Sequence[int] | None = None,
    blank_id: int = 0,
) -> list[list[int]]:
    """``argmax`` -> ``unique_consecutive`` -> drop ``blank_id``.

    Faithful to ``SenseVoiceSmall.inference``: consecutive duplicates collapse
    *before* blanks are removed, so ``[a, blank, a]`` decodes to ``[a, a]``.
    """
    ids = mx.argmax(log_probs, axis=-1)
    if lengths is None:
        lengths_arr = mx.full((ids.shape[0],), ids.shape[1], dtype=mx.int32)
    else:
        lengths_arr = mx.asarray(lengths)
        if lengths_arr.ndim > 1:
            lengths_arr = lengths_arr.reshape(-1)
    results: list[list[int]] = []
    for b in range(ids.shape[0]):
        length = int(lengths_arr[b].item())
        length = max(0, min(length, ids.shape[1]))
        row = [int(v) for v in ids[b, :length].tolist()]
        collapsed: list[int] = []
        prev: int | None = None
        for token in row:
            if token != prev:
                collapsed.append(token)
            prev = token
        results.append([t for t in collapsed if t != blank_id])
    return results
