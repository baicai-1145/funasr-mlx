"""Native MLX inference for FunASR's FSMN voice activity detector (``fsmn-vad``).

FunASR reference sources (read-only, ported verbatim):

* ``funasr/models/fsmn_vad_streaming/model.py`` -- ``FsmnVADStreaming``,
  ``VADXOptions``, ``Stats``, ``WindowDetector``, ``E2EVadSpeechBufWithDoa``,
  ``E2EVadFrameProb``, the frame-state machine (``GetFrameState``,
  ``DetectOneFrame``, ``DetectCommonFrames``, ``DetectLastFrames``) and the
  endpoint bookkeeping (``PopDataBufTillFrame``, ``PopDataToOutputBuf``,
  ``DropCachedFrames``, ``ResetDetection``, ``init_cache``).
* ``funasr/models/fsmn_vad_streaming/encoder.py`` -- ``FSMN``, ``BasicBlock``,
  ``FSMNBlock``, ``LinearTransform``, ``AffineTransform``, ``RectifiedLinear``,
  ``FsmnStack``.
* ``funasr/models/fsmn_vad_streaming/template.yaml`` -- the reference config.

The module is self-contained (no dependency on the other ``funasr_mlx`` model
modules) and imports ``mlx`` at module scope only.

Architecture notes / exact semantics reproduced:

* ``FSMNBlock`` uses a PyTorch ``Conv2d(D, D, (lorder, 1), dilation=(lstride, 1),
  groups=D)`` applied to a ``(B, D, T, 1)`` view.  MLX has no Conv2d, so it is
  expressed as ``mx.conv1d`` over the time axis with layout ``(B, T, D)`` and a
  weight of shape ``(D, K, 1)``.  The cache is the trailing
  ``(lorder - 1) * lstride`` frames of the left-padded tensor, stored as
  ``(B, D, keep)`` (the reference's trailing singleton axis is a Conv2d artifact
  and carries no information).
* ``cache is not None`` selects streaming mode.  FunASR's ``init_cache`` always
  passes a (possibly empty) dict, so VAD inference always runs in streaming
  mode; the cache is created full-length on first use.
* Lookahead / lookback semantics live in the decision layer and are preserved:
  start point is delayed by ``win_size_frame + lookback_time_start_point/frame_in_ms``
  frames, the end point is cut back by
  ``max_end_sil_frame_cnt_thresh/frame_in_ms - lookahead_time_end_point/frame_in_ms - 1``
  frames (``-1`` only when ``do_extend``).
* ``GetFrameState`` computes ``noise_prob = log(sum(silence probs)) *
  speech_2_noise_ratio`` and ``speech_prob = log(1 - sum(...))``, then compares
  ``exp(speech_prob) >= exp(noise_prob) + speech_noise_thres``.  The literal
  ``log``/``exp`` round trip is kept (not simplified to ``1 - p >= p + thres``).
* Decibel is ``10 * log10(sum(frame ** 2) + 1e-6)`` over 25 ms frames shifted by
  10 ms, computed in float32.

Deliberate, behaviour-preserving deviations:

* ``stats.data_buf`` / ``stats.data_buf_all`` are tracked as *sample counts*
  instead of tensors.  FunASR only ever reads ``len(...)`` and slices to advance
  the cursor; the samples themselves are never consumed, so keeping the waveform
  resident would waste O(duration) memory for nothing.
* ``PopDataBufTillFrame`` gets a bounded-loop guard.  FunASR's ``while`` loop can
  spin forever when the buffer runs dry; here it breaks with a warning instead.
* Dropout (identity at inference) and the dynamic-silence schedule that lives in
  ``FsmnVADStreaming.inference``'s Python-level chunking loop are out of scope for
  the tensor model; see the module docstring of the reference.

Weight layout conversion (PyTorch -> MLX):

============================  =========================  ==========================
FunASR / PyTorch key          PyTorch shape              MLX shape
============================  =========================  ==========================
``encoder.in_linear1.linear.weight``   ``(A, D)``      identical
``encoder.fsmn.N.linear.weight``      ``(P, L)``      identical
``encoder.fsmn.N.fsmn_block.conv_left.weight``  ``(D, 1, K, 1)``  ``(D, K, 1)``
``encoder.fsmn.N.affine.linear.weight``  ``(L, P)``    identical
============================  =========================  ==========================

Loading is strict; any missing/unexpected/mismatched key is an error.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "AudioChangeState",
    "BasicBlock",
    "FSMN",
    "FSMNBlock",
    "FsmnEncoderConfig",
    "FsmnVadModel",
    "FrameProb",
    "FrameState",
    "ModuleLayerList",
    "SpeechSegment",
    "VadDetectMode",
    "VadOptions",
    "VadStateMachine",
    "VadStats",
    "WindowDetector",
    "convert_checkpoint_tensors",
    "flat_parameter_names",
    "load_state_dict_strict",
]

logger = logging.getLogger(__name__)

# Depthwise FSMN convolutions: ``(D, 1, K, 1)`` -> ``(D, K, 1)``.
CONV2D_WEIGHT_SUFFIXES = (".fsmn_block.conv_left.weight", ".fsmn_block.conv_right.weight")

# FunASR's ``ComputeDecibel`` adds this before the log.
DECIBEL_EPSILON = 0.000001


# --------------------------------------------------------------------------- #
# enums (values mirror funasr/models/fsmn_vad_streaming/model.py)
# --------------------------------------------------------------------------- #
class FrameState(enum.Enum):
    INVALID = -1
    SIL = 0
    SPEECH = 1


class AudioChangeState(enum.Enum):
    SPEECH_TO_SPEECH = 0
    SPEECH_TO_SIL = 1
    SIL_TO_SIL = 2
    SIL_TO_SPEECH = 3
    NO_BEGIN = 4
    INVALID = 5


class VadStateMachine(enum.Enum):
    START_POINT_NOT_DETECTED = 1
    IN_SPEECH_SEGMENT = 2
    END_POINT_DETECTED = 3


class VadDetectMode(enum.Enum):
    SINGLE_UTTERANCE = 0
    MULTIPLE_UTTERANCE = 1


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    return float(value)


def _known_fields(cls: type, mapping: Mapping[str, Any], context: str) -> dict[str, Any]:
    allowed = {f.name for f in dataclass_fields(cls)}
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {context} keys: {unknown}; allowed: {sorted(allowed)}")
    return {k: v for k, v in mapping.items() if k in allowed}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FsmnEncoderConfig:
    """Validated mirror of ``encoder_conf`` for FunASR's ``FSMN`` network."""

    input_dim: int = 400
    input_affine_dim: int = 140
    fsmn_layers: int = 4
    linear_dim: int = 250
    proj_dim: int = 128
    lorder: int = 20
    rorder: int = 0
    lstride: int = 1
    rstride: int = 0
    output_affine_dim: int = 140
    output_dim: int = 248
    use_softmax: bool = True

    def __post_init__(self) -> None:
        for name in ("input_dim", "input_affine_dim", "fsmn_layers", "linear_dim",
                     "proj_dim", "lorder", "output_affine_dim", "output_dim"):
            minimum = 2 if name == "fsmn_layers" else 1
            object.__setattr__(self, name, _as_int(getattr(self, name), name, minimum=minimum))
        object.__setattr__(self, "rorder", _as_int(self.rorder, "rorder", minimum=0))
        object.__setattr__(self, "lstride", _as_int(self.lstride, "lstride", minimum=1))
        object.__setattr__(self, "rstride", _as_int(self.rstride, "rstride", minimum=0))
        object.__setattr__(
            self, "use_softmax", bool(self.use_softmax)
        )
        _require(
            self.lorder >= 1,
            "lorder must be >= 1 so the left context cache is well defined",
        )
        if self.rorder > 0:
            _require(
                self.rstride >= 1,
                "rorder > 0 requires rstride >= 1 (0 would be an invalid dilation)",
            )

    @property
    def cache_frames(self) -> int:
        """Left-context cache length ``(lorder - 1) * lstride``."""
        return (self.lorder - 1) * self.lstride

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "FsmnEncoderConfig":
        if not isinstance(config, Mapping):
            raise TypeError(f"encoder_conf must be a mapping, got {type(config).__name__}")
        return cls(**_known_fields(cls, config, "encoder_conf"))


@dataclass
class VadOptions:
    """Mutable mirror of FunASR's ``VADXOptions``.

    Mutable because ``ComputeScores`` overwrites ``nn_eval_block_size`` with the
    number of frames in the current block, exactly like the reference.
    """

    sample_rate: int = 16000
    detect_mode: int = VadDetectMode.MULTIPLE_UTTERANCE.value
    snr_mode: int = 0
    max_end_silence_time: int = 800
    max_start_silence_time: int = 3000
    do_start_point_detection: bool = True
    do_end_point_detection: bool = True
    window_size_ms: int = 200
    sil_to_speech_time_thres: int = 150
    speech_to_sil_time_thres: int = 150
    speech_2_noise_ratio: float = 1.0
    do_extend: int = 1
    lookback_time_start_point: int = 200
    lookahead_time_end_point: int = 100
    max_single_segment_time: int = 60000
    nn_eval_block_size: int = 8
    dcd_block_size: int = 4
    snr_thres: float = -100.0
    noise_frame_num_used_for_snr: int = 100
    decibel_thres: float = -100.0
    speech_noise_thres: float = 0.6
    fe_prior_thres: float = 1e-4
    silence_pdf_num: int = 1
    sil_pdf_ids: tuple[int, ...] = (0,)
    speech_noise_thresh_low: float = -0.1
    speech_noise_thresh_high: float = 0.3
    output_frame_probs: bool = False
    frame_in_ms: int = 10
    frame_length_ms: int = 25

    def __post_init__(self) -> None:
        for name in ("sample_rate", "max_end_silence_time", "max_start_silence_time",
                     "window_size_ms", "sil_to_speech_time_thres",
                     "speech_to_sil_time_thres", "lookback_time_start_point",
                     "lookahead_time_end_point", "max_single_segment_time",
                     "nn_eval_block_size", "dcd_block_size", "noise_frame_num_used_for_snr",
                     "silence_pdf_num", "frame_in_ms", "frame_length_ms"):
            object.__setattr__(self, name, _as_int(getattr(self, name), name, minimum=0))
        object.__setattr__(self, "detect_mode", _as_int(self.detect_mode, "detect_mode", minimum=0))
        object.__setattr__(
            self, "do_extend", _as_int(self.do_extend, "do_extend", minimum=0)
        )
        object.__setattr__(self, "sil_pdf_ids", tuple(int(i) for i in self.sil_pdf_ids))
        for name in ("speech_2_noise_ratio", "snr_thres", "decibel_thres",
                     "speech_noise_thres", "fe_prior_thres", "speech_noise_thresh_low",
                     "speech_noise_thresh_high"):
            object.__setattr__(self, name, _as_float(getattr(self, name), name))

        _require(self.sample_rate > 0, "sample_rate must be positive")
        _require(self.frame_in_ms > 0, "frame_in_ms must be positive")
        _require(
            self.frame_length_ms >= self.frame_in_ms,
            "frame_length_ms must be >= frame_in_ms "
            f"({self.frame_length_ms} < {self.frame_in_ms})",
        )
        _require(
            self.detect_mode in tuple(mode.value for mode in VadDetectMode),
            f"detect_mode must be 0 or 1, got {self.detect_mode}",
        )
        _require(self.window_size_ms > 0, "window_size_ms must be positive")
        _require(
            len(self.sil_pdf_ids) == self.silence_pdf_num,
            "sil_pdf_ids must have exactly silence_pdf_num entries "
            f"({len(self.sil_pdf_ids)} != {self.silence_pdf_num})",
        )
        _require(len(self.sil_pdf_ids) > 0, "sil_pdf_ids must not be empty")
        _require(
            self.max_end_silence_time >= self.speech_to_sil_time_thres,
            "max_end_silence_time must be >= speech_to_sil_time_thres "
            f"({self.max_end_silence_time} < {self.speech_to_sil_time_thres})",
        )
        _require(
            self.noise_frame_num_used_for_snr > 0,
            "noise_frame_num_used_for_snr must be positive",
        )

    # -- derived quantities -------------------------------------------------
    @property
    def frame_sample_length(self) -> int:
        return int(self.frame_length_ms * self.sample_rate / 1000)

    @property
    def frame_shift_length(self) -> int:
        return int(self.frame_in_ms * self.sample_rate / 1000)

    @property
    def max_end_sil_frame_cnt_thresh(self) -> int:
        """``Stats.max_end_sil_frame_cnt_thresh`` at cache-initialisation time."""
        return self.max_end_silence_time - self.speech_to_sil_time_thres

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "VadOptions":
        if not isinstance(config, Mapping):
            raise TypeError(f"model_conf must be a mapping, got {type(config).__name__}")
        return cls(**_known_fields(cls, config, "model_conf"))


# --------------------------------------------------------------------------- #
# modules
# --------------------------------------------------------------------------- #
class ModuleLayerList(nn.Module):
    """``nn.ModuleList`` keyed by *string* index (see ``ct_punc.py`` for why).

    MLX's own ``ModuleList`` nests children under ``layers`` and would shift every
    checkpoint key; FunASR's ``FsmnStack`` (an ``nn.Sequential``) indexes blocks
    as ``fsmn.0.…``, ``fsmn.1.…``, so children must sit directly under ``"0"``,
    ``"1"``, …  ``nn.Module.update()`` can only overwrite existing keys, so
    children are inserted as dict items.
    """

    def __init__(self, modules: Sequence[nn.Module]) -> None:
        super().__init__()
        for index, module in enumerate(modules):
            self[str(index)] = module

    def __len__(self) -> int:
        return dict.__len__(self)

    def __iter__(self):
        for index in range(dict.__len__(self)):
            yield self[str(index)]

    def __getitem__(self, index: int | str) -> nn.Module:
        return self[str(index)] if isinstance(index, int) else dict.__getitem__(self, index)


class LinearTransform(nn.Module):
    """``LinearTransform``: bias-free linear (checkpoint key ``…linear.weight``)."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class AffineTransform(nn.Module):
    """``AffineTransform``: linear with bias."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class FSMNBlock(nn.Module):
    """``FSMNBlock``: left/right dilated depthwise memory block.

    The reference applies ``nn.Conv2d`` to a ``(B, D, T, 1)`` view; here the time
    axis is the sequence axis of ``mx.conv1d`` and the trailing singleton axis is
    dropped (it carries no data).

    Cache shapes: ``cache`` is ``(B, D, (lorder - 1) * lstride)`` or ``None``.
    """

    def __init__(
        self, dim: int, lorder: int, rorder: int, lstride: int, rstride: int
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.lorder = int(lorder)
        self.rorder = int(rorder)
        self.lstride = int(lstride)
        self.rstride = int(rstride)
        self.keep = (self.lorder - 1) * self.lstride

        self.conv_left = nn.Conv1d(
            self.dim, self.dim, self.lorder, stride=1, padding=0,
            dilation=self.lstride, groups=self.dim, bias=False,
        )
        self.conv_right = (
            nn.Conv1d(
                self.dim, self.dim, self.rorder, stride=1, padding=0,
                dilation=self.rstride, groups=self.dim, bias=False,
            )
            if self.rorder > 0
            else None
        )

    def __call__(self, x: mx.array, cache: mx.array | None = None):
        """Return ``(output, new_cache)`` with ``output`` shaped ``(B, T, D)``."""
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, D) input, got shape {x.shape}")
        x_per = mx.transpose(x, 0, 2, 1)  # (B, D, T)
        if cache is not None:
            if cache.ndim != 3 or cache.shape[1] != self.dim:
                raise ValueError(
                    f"cache must be (B, {self.dim}, keep); got {cache.shape}"
                )
            y_left = mx.concatenate([cache, x_per], axis=2)
            start = max(0, y_left.shape[2] - self.keep)
            new_cache = y_left[:, :, start:]
        else:
            y_left = mx.pad(x_per, [(0, 0), (0, 0), (self.keep, 0)])
            new_cache = None

        out = x_per + mx.transpose(self.conv_left(mx.transpose(y_left, 0, 2, 1)), 0, 2, 1)

        if self.conv_right is not None:
            # FunASR right-pads by rorder*rstride then drops rstride frames, which
            # is exactly "slice the first rstride frames, then right-pad by the
            # same amount".
            if x_per.shape[2] >= self.rstride:
                future = x_per[:, :, self.rstride :]
                future = mx.pad(
                    future, [(0, 0), (0, 0), (0, self.rorder * self.rstride)]
                )
                out = out + mx.transpose(
                    self.conv_right(mx.transpose(future, 0, 2, 1)), 0, 2, 1
                )
            else:
                logger.warning(
                    "rorder=%d needs at least rstride=%d frames; right context skipped",
                    self.rorder, self.rstride,
                )
        return mx.transpose(out, 0, 2, 1), new_cache


class BasicBlock(nn.Module):
    """``BasicBlock``: linear -> FSMN -> affine -> relu, with a per-layer cache."""

    def __init__(
        self, linear_dim: int, proj_dim: int, lorder: int, rorder: int,
        lstride: int, rstride: int, stack_layer: int,
    ) -> None:
        super().__init__()
        self.stack_layer = int(stack_layer)
        self.lorder = int(lorder)
        self.lstride = int(lstride)
        self.cache_key = f"cache_layer_{self.stack_layer}"
        self.linear = LinearTransform(linear_dim, proj_dim)
        self.fsmn_block = FSMNBlock(proj_dim, lorder, rorder, lstride, rstride)
        self.affine = AffineTransform(proj_dim, linear_dim)

    def __call__(self, x: mx.array, cache: dict[str, mx.array] | None = None):
        x1 = self.linear(x)  # (B, T, P)
        if cache is None:
            x2, _ = self.fsmn_block(x1, None)
        else:
            key = self.cache_key
            if key not in cache:
                cache[key] = mx.zeros(
                    (x1.shape[0], x1.shape[-1], self.fsmn_block.keep), dtype=x1.dtype
                )
            x2, cache[key] = self.fsmn_block(x1, cache[key])
        return nn.relu(self.affine(x2))


class FSMN(nn.Module):
    """``FSMN`` encoder (``funasr/models/fsmn_vad_streaming/encoder.py``).

    ``cache=None`` runs the non-streaming path (zero left padding); passing a dict
    runs the streaming path and updates ``cache["cache_layer_<i>"]`` in place,
    exactly like the reference.
    """

    def __init__(self, config: FsmnEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.use_softmax = bool(config.use_softmax)
        self.in_linear1 = AffineTransform(config.input_dim, config.input_affine_dim)
        self.in_linear2 = AffineTransform(config.input_affine_dim, config.linear_dim)
        self.fsmn = ModuleLayerList(
            [
                BasicBlock(
                    config.linear_dim, config.proj_dim, config.lorder, config.rorder,
                    config.lstride, config.rstride, index,
                )
                for index in range(config.fsmn_layers)
            ]
        )
        self.out_linear1 = AffineTransform(config.linear_dim, config.output_affine_dim)
        self.out_linear2 = AffineTransform(config.output_affine_dim, config.output_dim)

    def output_size(self) -> int:
        return self.config.output_dim

    def __call__(self, x: mx.array, cache: dict[str, mx.array] | None = None) -> mx.array:
        h = nn.relu(self.in_linear2(self.in_linear1(x)))
        for layer in self.fsmn:
            h = layer(h, cache)
        h = self.out_linear2(self.out_linear1(h))
        return nn.softmax(h, axis=-1) if self.use_softmax else h

    def forward_with_caches(self, x: mx.array, cache: dict[str, mx.array]):
        """ONNX-export style: return ``(scores, [cache_layer_0, …])``."""
        scores = self.__call__(x, cache)
        caches = [
            cache[layer.cache_key] for layer in self.fsmn if layer.cache_key in cache
        ]
        return scores, caches


# --------------------------------------------------------------------------- #
# decision logic (pure Python; mirrors funasr/models/fsmn_vad_streaming/model.py)
# --------------------------------------------------------------------------- #
class WindowDetector:
    """``WindowDetector``: sliding window frame-state machine."""

    def __init__(
        self, window_size_ms: int, sil_to_speech_time: int, speech_to_sil_time: int,
        frame_size_ms: int,
    ) -> None:
        self.window_size_ms = int(window_size_ms)
        self.sil_to_speech_time = int(sil_to_speech_time)
        self.speech_to_sil_time = int(speech_to_sil_time)
        self.frame_size_ms = int(frame_size_ms)

        self.win_size_frame = int(self.window_size_ms / self.frame_size_ms)
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame
        self.cur_win_pos = 0
        self.pre_frame_state = FrameState.SIL
        self.cur_frame_state = FrameState.SIL
        self.sil_to_speech_frmcnt_thres = int(self.sil_to_speech_time / self.frame_size_ms)
        self.speech_to_sil_frmcnt_thres = int(self.speech_to_sil_time / self.frame_size_ms)
        self.voice_last_frame_count = 0
        self.noise_last_frame_count = 0
        self.hydre_frame_count = 0

    def reset(self) -> None:
        self.cur_win_pos = 0
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame
        self.pre_frame_state = FrameState.SIL
        self.cur_frame_state = FrameState.SIL
        self.voice_last_frame_count = 0
        self.noise_last_frame_count = 0
        self.hydre_frame_count = 0

    def get_window_size(self) -> int:
        return int(self.win_size_frame)

    def frame_size_ms(self) -> int:
        return int(self.frame_size_ms)

    def detect_one_frame(
        self, frame_state: FrameState, frame_count: int = 0, cache: dict | None = None
    ) -> AudioChangeState:
        """Feed one frame; returns the resulting transition."""
        del frame_count, cache  # unused, kept for signature parity
        if frame_state == FrameState.SPEECH:
            current = 1
        elif frame_state == FrameState.SIL:
            current = 0
        else:
            return AudioChangeState.INVALID

        self.win_sum -= self.win_state[self.cur_win_pos]
        self.win_sum += current
        self.win_state[self.cur_win_pos] = current
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame

        if (
            self.pre_frame_state == FrameState.SIL
            and self.win_sum >= self.sil_to_speech_frmcnt_thres
        ):
            self.pre_frame_state = FrameState.SPEECH
            return AudioChangeState.SIL_TO_SPEECH
        if (
            self.pre_frame_state == FrameState.SPEECH
            and self.win_sum <= self.speech_to_sil_frmcnt_thres
        ):
            self.pre_frame_state = FrameState.SIL
            return AudioChangeState.SPEECH_TO_SIL
        if self.pre_frame_state == FrameState.SIL:
            return AudioChangeState.SIL_TO_SIL
        return AudioChangeState.SPEECH_TO_SPEECH


@dataclass
class SpeechSegment:
    """``E2EVadSpeechBufWithDoa`` (DOA is always 0 for fsmn-vad)."""

    start_ms: int = 0
    end_ms: int = 0
    contain_seg_start_point: bool = False
    contain_seg_end_point: bool = False
    doa: int = 0

    def reset(self) -> None:
        self.start_ms = 0
        self.end_ms = 0
        self.contain_seg_start_point = False
        self.contain_seg_end_point = False
        self.doa = 0

    def as_pair(self) -> list[int]:
        return [int(self.start_ms), int(self.end_ms)]


@dataclass
class FrameProb:
    """``E2EVadFrameProb`` (populated only when ``output_frame_probs`` is set)."""

    noise_prob: float = 0.0
    speech_prob: float = 0.0
    score: float = 0.0
    frame_id: int = 0
    frm_state: int = 0


@dataclass
class VadStats:
    """``Stats`` with the waveform buffers reduced to sample counts."""

    sil_pdf_ids: tuple[int, ...]
    max_end_sil_frame_cnt_thresh: int
    speech_noise_thres: float

    data_buf_start_frame: int = 0
    frm_cnt: int = 0
    latest_confirmed_speech_frame: int = 0
    lastest_confirmed_silence_frame: int = -1
    continous_silence_frame_count: int = 0
    vad_state_machine: VadStateMachine = VadStateMachine.START_POINT_NOT_DETECTED
    confirmed_start_frame: int = -1
    confirmed_end_frame: int = -1
    number_end_time_detected: int = 0
    sil_frame: int = 0
    noise_average_decibel: float = -100.0
    pre_end_silence_detected: bool = False
    next_seg: bool = True
    output_data_buf: list[SpeechSegment] = field(default_factory=list)
    output_data_buf_offset: int = 0
    frame_probs: list[FrameProb] = field(default_factory=list)
    scores: mx.array | None = None
    max_time_out: bool = False
    decibel: list[float] = field(default_factory=list)
    #: ``sum(scores[0][t][sil_pdf_id] for sil_pdf_id in sil_pdf_ids)`` per frame,
    #: kept host-side so the decision loop stays pure Python.  Trimmed in lockstep
    #: with ``decibel`` by :meth:`FsmnVadModel.drop_cached_frames`.
    noise_scores: list[float] = field(default_factory=list)
    waveform: mx.array | None = None
    last_drop_frames: int = 0
    #: Sample counts replacing FunASR's ``data_buf`` / ``data_buf_all`` tensors.
    data_buf_len: int = 0
    data_buf_all_len: int = 0
    #: ``True`` once ``data_buf_all_len`` holds real samples (FunASR tests
    #: ``data_buf_all is None``; a 0-length buffer is not the same as "unset").
    data_buf_all_init: bool = False


# --------------------------------------------------------------------------- #
# model + driver
# --------------------------------------------------------------------------- #
def _require_cache(cache: dict | None) -> dict:
    if not isinstance(cache, dict) or "stats" not in cache:
        raise ValueError("cache must be initialised with FsmnVadModel.init_cache()")
    return cache


class FsmnVadModel(nn.Module):
    """FSMN-VAD: streaming/offline speech endpointing.

    ``__call__(feats, waveform, cache, is_final=…, is_streaming_input=…)`` mirrors
    ``FsmnVADStreaming.forward`` and returns ``[[[start_ms, end_ms], …], …]``
    (batch x segment; batch size 1 is the only supported case, as in FunASR).
    """

    def __init__(
        self,
        config: FsmnEncoderConfig,
        vad_options: VadOptions | None = None,
        *,
        max_chunk_frames: int = 6000,
    ) -> None:
        super().__init__()
        if max_chunk_frames <= 0:
            raise ValueError(f"max_chunk_frames must be positive, got {max_chunk_frames}")
        self.config = config
        self.vad_opts = vad_options if vad_options is not None else VadOptions()
        self.max_chunk_frames = int(max_chunk_frames)
        self.encoder = FSMN(config)

    # -- factory -------------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        config: Mapping[str, Any],
        *,
        max_chunk_frames: int = 6000,
    ) -> "FsmnVadModel":
        """Build from a FunASR ``config.yaml`` mapping."""
        if not isinstance(config, Mapping):
            raise TypeError(f"config must be a mapping, got {type(config).__name__}")
        model_conf = dict(config.get("model_conf") or {})
        encoder_conf = dict(config.get("encoder_conf") or {})
        if not model_conf and not encoder_conf:
            raise ValueError("config must contain 'model_conf' and/or 'encoder_conf'")
        options = VadOptions.from_dict(model_conf) if model_conf else VadOptions()
        return cls(
            FsmnEncoderConfig.from_dict(encoder_conf),
            options,
            max_chunk_frames=max_chunk_frames,
        )

    # -- cache lifecycle -----------------------------------------------------
    def init_cache(
        self,
        cache: dict | None = None,
        *,
        max_end_silence_time: int | None = None,
        speech_noise_thres: float | None = None,
    ) -> dict:
        """``init_cache``: fresh encoder caches, window detector and stats."""
        cache = {} if cache is None else cache
        if max_end_silence_time is not None:
            self.vad_opts.max_end_silence_time = _as_int(
                max_end_silence_time, "max_end_silence_time"
            )
        threshold = self.vad_opts.max_end_silence_time - self.vad_opts.speech_to_sil_time_thres

        cache["prev_samples"] = None
        cache["encoder"] = {}
        cache["windows_detector"] = WindowDetector(
            self.vad_opts.window_size_ms,
            self.vad_opts.sil_to_speech_time_thres,
            self.vad_opts.speech_to_sil_time_thres,
            self.vad_opts.frame_in_ms,
        )
        cache["windows_detector"].reset()
        cache["stats"] = VadStats(
            sil_pdf_ids=self.vad_opts.sil_pdf_ids,
            max_end_sil_frame_cnt_thresh=threshold,
            speech_noise_thres=(
                self.vad_opts.speech_noise_thres
                if speech_noise_thres is None
                else _as_float(speech_noise_thres, "speech_noise_thres")
            ),
        )
        return cache

    def reset(self, cache: dict | None = None, **kwargs: Any) -> dict:
        """Full reset between utterances (``init_cache``)."""
        return self.init_cache(cache, **kwargs)

    def drop_cached_frames(self, drop_frames: int, cache: dict) -> None:
        """``DropCachedFrames``: release decibel/score history before a frame."""
        stats = cache["stats"]
        if drop_frames > stats.frm_cnt:
            raise RuntimeError(
                f"cannot drop through frame {drop_frames}; only {stats.frm_cnt} frames exist"
            )
        real_drop_frames = drop_frames - stats.last_drop_frames
        if real_drop_frames <= 0:
            return

        frame_shift_length = self.vad_opts.frame_shift_length
        stats.data_buf_all_len = max(
            0, stats.data_buf_all_len - real_drop_frames * frame_shift_length
        )
        stats.decibel = stats.decibel[real_drop_frames:]
        stats.noise_scores = stats.noise_scores[real_drop_frames:]
        if stats.scores is not None:
            stats.scores = stats.scores[:, real_drop_frames:, :]
        stats.last_drop_frames = drop_frames
        offset = max(stats.data_buf_start_frame - stats.last_drop_frames, 0) * frame_shift_length
        stats.data_buf_len = max(0, stats.data_buf_all_len - offset)

    def reset_detection(self, cache: dict) -> None:
        """``ResetDetection``: soft reset after a detected endpoint."""
        stats = cache["stats"]
        stats.continous_silence_frame_count = 0
        stats.latest_confirmed_speech_frame = 0
        stats.lastest_confirmed_silence_frame = -1
        stats.confirmed_start_frame = -1
        stats.confirmed_end_frame = -1
        stats.vad_state_machine = VadStateMachine.START_POINT_NOT_DETECTED
        cache["windows_detector"].reset()
        stats.sil_frame = 0
        stats.frame_probs = []
        if stats.output_data_buf:
            if not stats.output_data_buf[-1].contain_seg_end_point:
                raise RuntimeError("reset_detection requires the last segment to be closed")
            drop_frames = int(stats.output_data_buf[-1].end_ms / self.vad_opts.frame_in_ms)
            self.drop_cached_frames(drop_frames, cache=cache)

    # -- feature / score stages ---------------------------------------------
    def compute_decibel(self, waveform: mx.array, cache: dict, frame_count: int) -> None:
        """``ComputeDecibel``: 10*log10(frame energy + 1e-6) per 10 ms frame."""
        if frame_count <= 0:
            return
        stats = cache["stats"]
        frame_sample_length = self.vad_opts.frame_sample_length
        frame_shift_length = self.vad_opts.frame_shift_length

        wave = waveform.reshape(-1)
        aligned_sample_count = (frame_count - 1) * frame_shift_length + frame_sample_length
        if wave.shape[0] != aligned_sample_count:
            raise RuntimeError(
                "VAD score frames and waveform samples are not aligned: expected "
                f"{aligned_sample_count}, got {wave.shape[0]}"
            )

        new_sample_count = frame_count * frame_shift_length
        if not stats.data_buf_all_init:
            stats.data_buf_all_init = True
            stats.data_buf_all_len = int(wave.shape[0])
        else:
            stats.data_buf_all_len += int(min(new_sample_count, wave.shape[0]))
        offset = (
            max(stats.data_buf_start_frame - stats.last_drop_frames, 0) * frame_shift_length
        )
        stats.data_buf_len = max(0, stats.data_buf_all_len - offset)

        total = int(wave.shape[0])
        available = max((total - frame_sample_length) // frame_shift_length + 1, 0)
        n_frames = min(available, frame_count)
        if n_frames <= 0:
            return
        starts = mx.arange(n_frames, dtype=mx.int32) * frame_shift_length
        offsets = starts[:, None] + mx.arange(frame_sample_length, dtype=mx.int32)[None, :]
        frames = wave[offsets]
        energy = mx.sum(mx.square(frames.astype(mx.float32)), axis=1)
        decibel = 10.0 * mx.log10(energy + DECIBEL_EPSILON)
        stats.decibel.extend(float(v) for v in decibel.tolist())

    def compute_scores(self, feats: mx.array, cache: dict) -> None:
        """``ComputeScores``: run the FSMN encoder and append the probabilities."""
        stats = cache["stats"]
        scores = self.encoder(feats, cache=cache["encoder"])
        if scores.shape[1] != feats.shape[1]:
            raise RuntimeError(
                "the shape between feats and scores does not match "
                f"({scores.shape[1]} != {feats.shape[1]})"
            )
        self.vad_opts.nn_eval_block_size = scores.shape[1]
        stats.frm_cnt += scores.shape[1]
        if stats.scores is None:
            stats.scores = scores
        else:
            stats.scores = mx.concatenate([stats.scores, scores], axis=1)

        ids = [int(i) for i in self.vad_opts.sil_pdf_ids]
        noise = mx.sum(scores[0][:, ids], axis=-1)
        stats.noise_scores.extend(float(v) for v in noise.tolist())

    # -- endpoint bookkeeping ------------------------------------------------
    def pop_data_buf_till_frame(self, frame_idx: int, cache: dict) -> None:
        """``PopDataBufTillFrame``: advance the sample cursor to ``frame_idx``."""
        stats = cache["stats"]
        frame_shift_length = self.vad_opts.frame_shift_length
        stalled = 0
        while stats.data_buf_start_frame < frame_idx:
            if stats.data_buf_len >= frame_shift_length:
                stats.data_buf_start_frame += 1
                offset = (
                    stats.data_buf_start_frame - stats.last_drop_frames
                ) * frame_shift_length
                stats.data_buf_len = max(0, stats.data_buf_all_len - offset)
                stalled = 0
            else:
                # FunASR's loop here spins forever; break out instead of hanging.
                stalled += 1
                if stalled > 2:
                    logger.warning(
                        "waveform buffer exhausted at frame %d (wanted %d); stopping",
                        stats.data_buf_start_frame, frame_idx,
                    )
                    break

    def pop_data_to_output_buf(
        self,
        start_frm: int,
        frm_cnt: int,
        first_frm_is_start_point: bool,
        last_frm_is_end_point: bool,
        end_point_is_sent_end: bool,
        cache: dict,
    ) -> None:
        """``PopDataToOutputBuf``: open/extend/close an output segment."""
        stats = cache["stats"]
        self.pop_data_buf_till_frame(start_frm, cache=cache)

        expected_sample_number = int(
            frm_cnt * self.vad_opts.sample_rate * self.vad_opts.frame_in_ms / 1000
        )
        if last_frm_is_end_point:
            expected_sample_number += int(
                max(
                    0,
                    self.vad_opts.frame_sample_length
                    - self.vad_opts.sample_rate * self.vad_opts.frame_in_ms / 1000,
                )
            )
        if end_point_is_sent_end:
            expected_sample_number = max(expected_sample_number, stats.data_buf_len)
        if stats.data_buf_len < expected_sample_number:
            logger.warning("error in calling pop_data_buf: buffer too short")

        if not stats.output_data_buf or first_frm_is_start_point:
            segment = SpeechSegment()
            segment.start_ms = start_frm * self.vad_opts.frame_in_ms
            segment.end_ms = segment.start_ms
            stats.output_data_buf.append(segment)
        current = stats.output_data_buf[-1]

        if current.end_ms != start_frm * self.vad_opts.frame_in_ms:
            logger.warning("segment cursor mismatch in pop_data_to_output_buf")

        data_to_pop = (
            expected_sample_number
            if end_point_is_sent_end
            else int(frm_cnt * self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000)
        )
        if data_to_pop > stats.data_buf_len:
            logger.warning("VAD data_to_pop is bigger than the waveform buffer")
            data_to_pop = stats.data_buf_len

        stats.data_buf_start_frame += frm_cnt
        current.end_ms = (start_frm + frm_cnt) * self.vad_opts.frame_in_ms
        if first_frm_is_start_point:
            current.contain_seg_start_point = True
        if last_frm_is_end_point:
            current.contain_seg_end_point = True

    def on_silence_detected(self, valid_frame: int, cache: dict) -> None:
        stats = cache["stats"]
        stats.lastest_confirmed_silence_frame = valid_frame
        if stats.vad_state_machine == VadStateMachine.START_POINT_NOT_DETECTED:
            self.pop_data_buf_till_frame(valid_frame, cache=cache)

    def on_voice_detected(self, valid_frame: int, cache: dict) -> None:
        stats = cache["stats"]
        stats.latest_confirmed_speech_frame = valid_frame
        self.pop_data_to_output_buf(valid_frame, 1, False, False, False, cache=cache)

    def on_voice_start(
        self, start_frame: int, fake_result: bool = False, cache: dict | None = None
    ) -> None:
        cache = _require_cache(cache)
        stats = cache["stats"]
        if stats.confirmed_start_frame != -1:
            # FunASR only prints here and keeps the stale value; mirror that.
            logger.warning(
                "VAD not reset properly: confirmed_start_frame=%d is already set",
                stats.confirmed_start_frame,
            )
        else:
            stats.confirmed_start_frame = start_frame
        if (
            not fake_result
            and stats.vad_state_machine == VadStateMachine.START_POINT_NOT_DETECTED
        ):
            self.pop_data_to_output_buf(
                stats.confirmed_start_frame, 1, True, False, False, cache=cache
            )

    def on_voice_end(
        self, end_frame: int, fake_result: bool, is_last_frame: bool, cache: dict
    ) -> None:
        stats = cache["stats"]
        for frame in range(stats.latest_confirmed_speech_frame + 1, end_frame):
            self.on_voice_detected(frame, cache=cache)
        if stats.confirmed_end_frame != -1:
            # FunASR only prints here and keeps the stale value; mirror that.
            logger.warning(
                "VAD not reset properly: confirmed_end_frame=%d is already set",
                stats.confirmed_end_frame,
            )
        else:
            stats.confirmed_end_frame = end_frame
        if not fake_result:
            stats.sil_frame = 0
            self.pop_data_to_output_buf(
                stats.confirmed_end_frame, 1, False, True, is_last_frame, cache=cache
            )
        stats.number_end_time_detected += 1

    def maybe_on_voice_end_if_last_frame(
        self, is_final_frame: bool, cur_frm_idx: int, cache: dict
    ) -> None:
        if is_final_frame:
            self.on_voice_end(cur_frm_idx, False, True, cache=cache)
            cache["stats"].vad_state_machine = VadStateMachine.END_POINT_DETECTED

    def get_latency(self, cache: dict) -> int:
        return int(self.latency_frame_count(cache=cache) * self.vad_opts.frame_in_ms)

    def latency_frame_count(self, cache: dict) -> int:
        latency = cache["windows_detector"].get_window_size()
        if self.vad_opts.do_extend:
            latency += int(self.vad_opts.lookback_time_start_point / self.vad_opts.frame_in_ms)
        return latency

    # -- per-frame decision --------------------------------------------------
    def get_frame_state(self, t: int, cache: dict) -> FrameState:
        """``GetFrameState``: decibel gate + FSMN log-odds comparison."""
        stats = cache["stats"]
        if t >= len(stats.decibel):
            return FrameState.SIL
        cur_decibel = stats.decibel[t]
        cur_snr = cur_decibel - stats.noise_average_decibel
        if cur_decibel < self.vad_opts.decibel_thres:
            return FrameState.SIL
        if t >= len(stats.noise_scores):
            return FrameState.SIL

        sum_score = stats.noise_scores[t]
        noise_prob = math.log(sum_score) * self.vad_opts.speech_2_noise_ratio
        sum_score = 1.0 - sum_score
        speech_prob = math.log(sum_score)

        if self.vad_opts.output_frame_probs:
            prob = FrameProb(
                noise_prob=noise_prob,
                speech_prob=speech_prob,
                score=sum_score,
                frame_id=t,
            )
            stats.frame_probs.append(prob)

        if math.exp(speech_prob) >= math.exp(noise_prob) + stats.speech_noise_thres:
            frame_state = (
                FrameState.SPEECH
                if (cur_snr >= self.vad_opts.snr_thres
                    and cur_decibel >= self.vad_opts.decibel_thres)
                else FrameState.SIL
            )
        else:
            frame_state = FrameState.SIL
            if stats.noise_average_decibel < -99.9:
                stats.noise_average_decibel = cur_decibel
            else:
                stats.noise_average_decibel = (
                    cur_decibel
                    + stats.noise_average_decibel
                    * (self.vad_opts.noise_frame_num_used_for_snr - 1)
                ) / self.vad_opts.noise_frame_num_used_for_snr
        return frame_state

    def detect_one_frame(
        self, cur_frm_state: FrameState, cur_frm_idx: int, is_final_frame: bool, cache: dict
    ) -> None:
        """``DetectOneFrame``: the full state machine step for a single frame."""
        stats = cache["stats"]
        if cur_frm_state == FrameState.SPEECH:
            if abs(1.0) > self.vad_opts.fe_prior_thres:
                tmp_state = FrameState.SPEECH
            else:
                tmp_state = FrameState.SIL
        elif cur_frm_state == FrameState.SIL:
            tmp_state = FrameState.SIL
        else:
            tmp_state = FrameState.INVALID

        state_change = cache["windows_detector"].detect_one_frame(tmp_state, cur_frm_idx)
        frm_shift_in_ms = self.vad_opts.frame_in_ms

        if state_change == AudioChangeState.SIL_TO_SPEECH:
            stats.continous_silence_frame_count = 0
            stats.pre_end_silence_detected = False
            if stats.vad_state_machine == VadStateMachine.START_POINT_NOT_DETECTED:
                start_frame = max(
                    stats.data_buf_start_frame,
                    cur_frm_idx - self.latency_frame_count(cache=cache),
                )
                self.on_voice_start(start_frame, cache=cache)
                stats.vad_state_machine = VadStateMachine.IN_SPEECH_SEGMENT
                for frame in range(start_frame + 1, cur_frm_idx + 1):
                    self.on_voice_detected(frame, cache=cache)
            elif stats.vad_state_machine == VadStateMachine.IN_SPEECH_SEGMENT:
                for frame in range(stats.latest_confirmed_speech_frame + 1, cur_frm_idx):
                    self.on_voice_detected(frame, cache=cache)
                if cur_frm_idx - stats.confirmed_start_frame + 1 > (
                    self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.on_voice_end(cur_frm_idx, False, False, cache=cache)
                    stats.vad_state_machine = VadStateMachine.END_POINT_DETECTED
                elif not is_final_frame:
                    self.on_voice_detected(cur_frm_idx, cache=cache)
                else:
                    self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx, cache=cache)

        elif state_change == AudioChangeState.SPEECH_TO_SIL:
            stats.continous_silence_frame_count = 0
            if stats.vad_state_machine == VadStateMachine.IN_SPEECH_SEGMENT:
                if cur_frm_idx - stats.confirmed_start_frame + 1 > (
                    self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.on_voice_end(cur_frm_idx, False, False, cache=cache)
                    stats.vad_state_machine = VadStateMachine.END_POINT_DETECTED
                elif not is_final_frame:
                    self.on_voice_detected(cur_frm_idx, cache=cache)
                else:
                    self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx, cache=cache)

        elif state_change == AudioChangeState.SPEECH_TO_SPEECH:
            stats.continous_silence_frame_count = 0
            if stats.vad_state_machine == VadStateMachine.IN_SPEECH_SEGMENT:
                if cur_frm_idx - stats.confirmed_start_frame + 1 > (
                    self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    stats.max_time_out = True
                    self.on_voice_end(cur_frm_idx, False, False, cache=cache)
                    stats.vad_state_machine = VadStateMachine.END_POINT_DETECTED
                elif not is_final_frame:
                    self.on_voice_detected(cur_frm_idx, cache=cache)
                else:
                    self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx, cache=cache)

        elif state_change == AudioChangeState.SIL_TO_SIL:
            stats.continous_silence_frame_count += 1
            if stats.vad_state_machine == VadStateMachine.START_POINT_NOT_DETECTED:
                timed_out = (
                    self.vad_opts.detect_mode == VadDetectMode.SINGLE_UTTERANCE.value
                    and stats.continous_silence_frame_count * frm_shift_in_ms
                    > self.vad_opts.max_start_silence_time
                )
                if timed_out or (is_final_frame and stats.number_end_time_detected == 0):
                    for frame in range(stats.lastest_confirmed_silence_frame + 1, cur_frm_idx):
                        self.on_silence_detected(frame, cache=cache)
                    self.on_voice_start(0, True, cache=cache)
                    self.on_voice_end(0, True, False, cache=cache)
                    stats.vad_state_machine = VadStateMachine.END_POINT_DETECTED
                elif cur_frm_idx >= self.latency_frame_count(cache=cache):
                    self.on_silence_detected(
                        cur_frm_idx - self.latency_frame_count(cache=cache), cache=cache
                    )
            elif stats.vad_state_machine == VadStateMachine.IN_SPEECH_SEGMENT:
                if stats.continous_silence_frame_count * frm_shift_in_ms >= (
                    stats.max_end_sil_frame_cnt_thresh
                ):
                    lookback_frame = int(stats.max_end_sil_frame_cnt_thresh / frm_shift_in_ms)
                    if self.vad_opts.do_extend:
                        lookback_frame -= int(
                            self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                        )
                        lookback_frame -= 1
                        lookback_frame = max(0, lookback_frame)
                    self.on_voice_end(cur_frm_idx - lookback_frame, False, False, cache=cache)
                    stats.vad_state_machine = VadStateMachine.END_POINT_DETECTED
                elif cur_frm_idx - stats.confirmed_start_frame + 1 > (
                    self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.on_voice_end(cur_frm_idx, False, False, cache=cache)
                    stats.vad_state_machine = VadStateMachine.END_POINT_DETECTED
                elif self.vad_opts.do_extend and not is_final_frame:
                    if stats.continous_silence_frame_count <= int(
                        self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                    ):
                        self.on_voice_detected(cur_frm_idx, cache=cache)
                else:
                    self.maybe_on_voice_end_if_last_frame(is_final_frame, cur_frm_idx, cache=cache)

        if (
            stats.vad_state_machine == VadStateMachine.END_POINT_DETECTED
            and self.vad_opts.detect_mode == VadDetectMode.MULTIPLE_UTTERANCE.value
        ):
            self.reset_detection(cache=cache)

    def detect_common_frames(self, cache: dict) -> int:
        """``DetectCommonFrames``: replay the block backwards (streaming)."""
        stats = cache["stats"]
        if stats.vad_state_machine == VadStateMachine.END_POINT_DETECTED:
            return 0
        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = self.get_frame_state(
                stats.frm_cnt - 1 - i - stats.last_drop_frames, cache=cache
            )
            self.detect_one_frame(frame_state, stats.frm_cnt - 1 - i, False, cache=cache)
        return 0

    def detect_last_frames(self, cache: dict) -> int:
        """``DetectLastFrames``: replay the block backwards, final frame flagged."""
        stats = cache["stats"]
        if stats.vad_state_machine == VadStateMachine.END_POINT_DETECTED:
            return 0
        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = self.get_frame_state(
                stats.frm_cnt - 1 - i - stats.last_drop_frames, cache=cache
            )
            if i != 0:
                self.detect_one_frame(frame_state, stats.frm_cnt - 1 - i, False, cache=cache)
            else:
                self.detect_one_frame(frame_state, stats.frm_cnt - 1, True, cache=cache)
        return 0

    # -- public entry points -------------------------------------------------
    def __call__(
        self,
        feats: mx.array,
        waveform: mx.array,
        cache: dict,
        *,
        is_final: bool = False,
        is_streaming_input: bool = True,
    ) -> list[list[list[int]]]:
        """``FsmnVADStreaming.forward``.

        ``feats`` is ``(1, T, input_dim)`` fbank (the frontend is owned by the
        shared runtime); ``waveform`` is the exact ``(1, N)`` (or ``(N,)``)
        sample span those ``T`` frames cover.
        """
        if not isinstance(cache, dict) or "stats" not in cache:
            raise ValueError("cache must be initialised with FsmnVadModel.init_cache()")
        if feats.size == 0:
            return []
        if feats.ndim != 3:
            raise ValueError(f"expected (B, T, D) feats, got shape {feats.shape}")
        if feats.shape[0] != 1:
            raise ValueError(f"fsmn-vad only supports batch_size=1, got {feats.shape[0]}")
        if feats.shape[1] > self.max_chunk_frames:
            raise ValueError(
                f"chunk of {feats.shape[1]} frames exceeds max_chunk_frames="
                f"{self.max_chunk_frames}"
            )

        stats = cache["stats"]
        stats.waveform = waveform
        self.compute_decibel(waveform, cache=cache, frame_count=feats.shape[1])
        self.compute_scores(feats, cache=cache)
        if not is_final:
            self.detect_common_frames(cache=cache)
        else:
            self.detect_last_frames(cache=cache)
        self.drop_cached_frames(stats.data_buf_start_frame, cache=cache)

        segments: list[list[list[int]]] = []
        for _ in range(feats.shape[0]):
            batch: list[list[int]] = []
            if stats.output_data_buf:
                for i in range(stats.output_data_buf_offset, len(stats.output_data_buf)):
                    segment = stats.output_data_buf[i]
                    if is_streaming_input:
                        if not segment.contain_seg_start_point:
                            continue
                        if not stats.next_seg and not segment.contain_seg_end_point:
                            continue
                        start_ms = segment.start_ms if stats.next_seg else -1
                        if segment.contain_seg_end_point:
                            end_ms = segment.end_ms
                            stats.next_seg = True
                            stats.output_data_buf_offset += 1
                        else:
                            end_ms = -1
                            stats.next_seg = False
                        batch.append([int(start_ms), int(end_ms)])
                    else:
                        if not is_final and (
                            not segment.contain_seg_start_point
                            or not segment.contain_seg_end_point
                        ):
                            continue
                        batch.append(segment.as_pair())
                        stats.output_data_buf_offset += 1
            if batch:
                segments.append(batch)
        return segments

    def detect(
        self,
        feats: mx.array,
        waveform: mx.array,
        *,
        cache: dict | None = None,
        is_final: bool = True,
        is_streaming_input: bool = False,
        **init_kwargs: Any,
    ) -> list[list[int]]:
        """Convenience wrapper for the offline case (auto-initialised cache)."""
        if cache is None:
            cache = self.init_cache(**init_kwargs)
        batches = self.__call__(
            feats, waveform, cache,
            is_final=is_final, is_streaming_input=is_streaming_input,
        )
        if not batches:
            return []
        return batches[0]


# --------------------------------------------------------------------------- #
# strict checkpoint loading
# --------------------------------------------------------------------------- #
def flat_parameter_names(module: nn.Module) -> dict[str, mx.array]:
    """Flatten ``module.parameters()`` into FunASR-style dotted names."""
    flat: dict[str, mx.array] = {}

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, mx.array):
            flat[prefix] = obj
        elif isinstance(obj, dict):
            for key, value in obj.items():
                walk(f"{prefix}.{key}" if prefix else str(key), value)
        elif isinstance(obj, (list, tuple)):
            for index, value in enumerate(obj):
                walk(f"{prefix}.{index}" if prefix else str(index), value)
        elif obj is not None:
            raise TypeError(f"unexpected parameter container {type(obj).__name__}")

    walk("", module.parameters())
    return flat


def convert_checkpoint_tensors(state: Mapping[str, mx.array | Any]) -> dict[str, mx.array]:
    """Convert a FunASR/PyTorch ``state_dict`` into MLX layout.

    Only the depthwise ``Conv2d`` memory kernels change:
    ``(D, 1, K, 1) -> (D, K, 1)``.  Every ``nn.Linear`` weight already uses
    MLX's ``(out, in)`` layout.
    """
    converted: dict[str, mx.array] = {}
    for name, value in state.items():
        if not isinstance(value, mx.array):
            value = mx.array(value)
        if name.endswith(CONV2D_WEIGHT_SUFFIXES):
            if value.ndim != 4:
                raise ValueError(
                    f"{name}: expected a 4-D depthwise conv weight (D, 1, K, 1), "
                    f"got shape {value.shape}"
                )
            d, _, k, w = value.shape
            if w != 1:
                raise ValueError(f"{name}: only a trailing width of 1 is supported, got {w}")
            value = mx.transpose(value, 0, 2, 1, 3).reshape(d, k, 1)
        converted[name] = value
    return converted


def _nest(flat: Mapping[str, mx.array]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for name, array in flat.items():
        parts = name.split(".")
        node = root
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = array
    return root


def load_state_dict_strict(
    model: nn.Module,
    state: Mapping[str, mx.array | Any],
    *,
    ignore_prefixes: Sequence[str] = (),
) -> dict[str, mx.array]:
    """Strictly load ``state`` into ``model``; raise on any mismatch."""
    expected = flat_parameter_names(model)
    provided = convert_checkpoint_tensors(state)

    missing = sorted(set(expected) - set(provided))
    if ignore_prefixes:
        missing = [
            name
            for name in missing
            if not any(name.startswith(prefix) for prefix in ignore_prefixes)
        ]
    unexpected = sorted(set(provided) - set(expected))
    if missing or unexpected:
        raise KeyError(
            "checkpoint does not match the model exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )

    mismatched = [
        (name, tuple(expected[name].shape), tuple(provided[name].shape))
        for name in sorted(expected)
        if tuple(expected[name].shape) != tuple(provided[name].shape)
    ]
    if mismatched:
        details = ", ".join(f"{n}: expected {e}, got {g}" for n, e, g in mismatched[:8])
        raise ValueError(
            f"checkpoint tensor shapes do not match ({len(mismatched)} tensor(s)): {details}"
        )

    applied = {name: provided[name] for name in expected}
    model.update(_nest(applied))
    return applied
