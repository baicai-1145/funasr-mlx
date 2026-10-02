"""Bounded native MLX audio features; no PyTorch inference dependency."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Mapping

import numpy as np


def read_audio(path: str | Path, *, sample_rate: int = 16000, max_seconds: float = 30.0):
    import soundfile as sf

    info = sf.info(path)
    if info.samplerate != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz, received {info.samplerate} Hz")
    if not 1 <= info.channels <= 8:
        raise ValueError("Audio must have between 1 and 8 channels")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if info.frames > int(sample_rate * max_seconds):
        raise ValueError(f"Audio exceeds the {max_seconds:g}-second input bound")
    samples, _ = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains non-finite samples")
    return samples.mean(axis=1, dtype=np.float32)


def load_cmvn(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.stat().st_size > 1_000_000:
        raise ValueError("CMVN file exceeds the metadata bound")
    values = {}
    current = None
    for line in path.read_text().splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] in {"<AddShift>", "<Rescale>"}:
            current = tokens[0]
        elif current and "[" in tokens and "]" in tokens:
            left, right = tokens.index("["), tokens.index("]")
            values[current] = [float(value) for value in tokens[left + 1 : right]]
            current = None
    shift, scale = values.get("<AddShift>"), values.get("<Rescale>")
    if not shift or not scale or len(shift) != len(scale):
        raise ValueError("CMVN needs equally sized AddShift and Rescale vectors")
    result = np.asarray([shift, scale], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("CMVN must contain finite values")
    return result


def apply_lfr(features, m: int = 7, n: int = 6):
    import mlx.core as mx

    if m <= 0 or n <= 0 or features.ndim != 2:
        raise ValueError("LFR needs rank-2 features and positive m/n")
    count, dimension = features.shape
    if count == 0:
        return mx.zeros((0, dimension * m), dtype=features.dtype)
    indices = mx.arange((count + n - 1) // n)[:, None] * n + mx.arange(m)[None, :]
    indices = mx.clip(indices - (m - 1) // 2, 0, count - 1)
    return features[indices].reshape(-1, dimension * m)


@dataclass(frozen=True)
class FbankConfig:
    fs: int = 16000
    n_mels: int = 80
    frame_length: float = 25.0
    frame_shift: float = 10.0
    window: str = "hamming"
    dither: float = 0.0
    snip_edges: bool = True
    lfr_m: int = 1
    lfr_n: int = 1
    upsacle_samples: bool = True
    low_freq: float = 20.0
    high_freq: float = 0.0
    preemphasis: float = 0.97
    max_seconds: float = 30.0
    cmvn_file: str | None = None

    def __post_init__(self):
        if self.fs not in {8000, 16000, 22050, 24000, 44100, 48000}:
            raise ValueError("Unsupported sample rate")
        if not 4 <= self.n_mels <= 256 or not 1 <= self.lfr_m <= 15 or self.lfr_n <= 0:
            raise ValueError("Invalid Mel or LFR dimensions")
        if self.window not in {"hamming", "hanning", "povey", "rectangular"}:
            raise ValueError("Unsupported Kaldi window")
        if not 1 <= self.frame_length <= 100 or not 1 <= self.frame_shift <= 100:
            raise ValueError("Frame sizes must be between 1 and 100 ms")
        if not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 60:
            raise ValueError("max_seconds must be in (0, 60]")
        if not math.isfinite(self.dither) or self.dither < 0 or not 0 <= self.preemphasis <= 1:
            raise ValueError("Invalid dither/preemphasis")
        high = self.fs / 2 + self.high_freq if self.high_freq <= 0 else self.high_freq
        if not 0 <= self.low_freq < high <= self.fs / 2:
            raise ValueError("Invalid Mel frequency bounds")

    @classmethod
    def from_dict(cls, mapping: Mapping):
        allowed = {field.name for field in fields(cls)}
        unknown = set(mapping) - allowed - {"filter_length_min", "filter_length_max"}
        if unknown:
            raise ValueError(f"Unsupported frontend options: {sorted(unknown)}")
        return cls(**{key: value for key, value in mapping.items() if key in allowed})


def _kaldi_filters(config: FbankConfig, n_fft: int):
    import mlx.core as mx

    high = config.fs / 2 + config.high_freq if config.high_freq <= 0 else config.high_freq
    low_mel = 1127 * math.log1p(config.low_freq / 700)
    high_mel = 1127 * math.log1p(high / 700)
    delta = (high_mel - low_mel) / (config.n_mels + 1)
    left = low_mel + mx.arange(config.n_mels)[:, None] * delta
    center, right = left + delta, left + 2 * delta
    frequency = config.fs / n_fft * mx.arange(n_fft // 2)
    mel = 1127 * mx.log1p(frequency / 700)[None, :]
    filters = mx.maximum(
        0, mx.minimum((mel - left) / (center - left), (right - mel) / (right - center))
    )
    return mx.pad(filters, [(0, 0), (0, 1)])


_CONFIG_CMVN = object()


class WavFrontend:
    def __init__(self, config: FbankConfig | None = None, *, cmvn=_CONFIG_CMVN):
        self.config = config or FbankConfig()
        cmvn = self.config.cmvn_file if cmvn is _CONFIG_CMVN else cmvn
        self.cmvn = load_cmvn(cmvn) if isinstance(cmvn, (str, Path)) else cmvn
        if self.cmvn is not None:
            expected = self.config.n_mels * self.config.lfr_m
            if np.shape(self.cmvn) != (2, expected):
                raise ValueError(f"Expected CMVN shape (2, {expected})")

    def fbank(self, samples):
        import mlx.core as mx

        cfg = self.config
        if len(samples) > int(cfg.fs * cfg.max_seconds):
            raise ValueError("Audio exceeds frontend max_seconds")
        waveform = mx.array(samples, dtype=mx.float32)
        if waveform.ndim != 1:
            raise ValueError("Frontend expects mono rank-1 waveform")
        size = waveform.shape[0]
        if size == 0:
            return mx.zeros((0, cfg.n_mels))
        if size < 2:
            raise ValueError("Audio needs at least two samples")
        if not bool(mx.all(mx.isfinite(waveform)).item()):
            raise ValueError("Audio samples must be finite")
        if cfg.upsacle_samples:
            waveform = waveform * 32768.0
        width = min(int(cfg.fs * cfg.frame_length / 1000), size)
        shift = int(cfg.fs * cfg.frame_shift / 1000)
        if cfg.snip_edges:
            count = 1 + (size - width) // shift
            indices = mx.arange(count)[:, None] * shift + mx.arange(width)[None, :]
        else:
            count = (size + shift // 2) // shift
            if count == 0:
                return mx.zeros((0, cfg.n_mels))
            indices = (
                mx.arange(count)[:, None] * shift
                + mx.arange(width)[None, :]
                - (width // 2 - shift // 2)
            )
            indices = indices % (2 * size)
            indices = mx.where(indices < size, indices, 2 * size - 1 - indices)
        frames = waveform[indices]
        if cfg.dither:
            frames = frames + cfg.dither * mx.random.normal(frames.shape)
        frames = frames - mx.mean(frames, axis=1, keepdims=True)
        previous = mx.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
        frames = frames - cfg.preemphasis * previous
        phase = 2 * math.pi * mx.arange(width) / (width - 1)
        if cfg.window == "hamming":
            window = 0.54 - 0.46 * mx.cos(phase)
        elif cfg.window in {"hanning", "povey"}:
            window = 0.5 - 0.5 * mx.cos(phase)
            if cfg.window == "povey":
                window = window**0.85
        else:
            window = mx.ones((width,))
        n_fft = 1 << (width - 1).bit_length()
        spectrum = mx.abs(mx.fft.rfft(frames * window, n=n_fft, axis=-1)) ** 2
        power = spectrum @ _kaldi_filters(cfg, n_fft).T
        return mx.log(mx.maximum(power, np.finfo(np.float32).eps))

    def __call__(self, samples):
        import mlx.core as mx

        features = apply_lfr(self.fbank(samples), self.config.lfr_m, self.config.lfr_n)
        if self.cmvn is not None:
            features = (features + mx.array(self.cmvn[0])) * mx.array(self.cmvn[1])
        return features


def _slaney_filters(n_mels: int, n_fft: int = 400, sample_rate: int = 16000):
    def to_mel(frequency):
        frequency = np.asarray(frequency)
        linear = frequency / (200 / 3)
        return np.where(
            frequency >= 1000,
            15 + np.log(np.maximum(frequency, 1) / 1000) / (math.log(6.4) / 27),
            linear,
        )

    mel = np.linspace(0, to_mel(sample_rate / 2), n_mels + 2)
    frequencies = np.where(
        mel >= 15, 1000 * np.exp((mel - 15) * (math.log(6.4) / 27)), mel * (200 / 3)
    )
    fft = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    lower = (fft[None, :] - frequencies[:-2, None]) / np.diff(frequencies)[:-1, None]
    upper = (frequencies[2:, None] - fft[None, :]) / np.diff(frequencies)[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2 / (frequencies[2:] - frequencies[:-2]))[:, None]
    return weights.astype(np.float32)


def whisper_log_mel(samples, *, n_mels: int = 80, pad_to_samples: int = 480000):
    import mlx.core as mx

    if n_mels not in {80, 128} or not 400 <= pad_to_samples <= 480000:
        raise ValueError("Whisper needs 80/128 Mel bins and at most 30 seconds")
    if len(samples) > pad_to_samples:
        raise ValueError("Audio exceeds the Whisper segment bound")
    waveform = mx.array(samples, dtype=mx.float32)
    if waveform.ndim != 1 or not bool(mx.all(mx.isfinite(waveform)).item()):
        raise ValueError("Whisper expects a finite mono waveform")
    waveform = mx.pad(waveform, [(0, pad_to_samples - waveform.shape[0])])
    indices = mx.arange(pad_to_samples // 160)[:, None] * 160 + mx.arange(400)[None, :] - 200
    indices = mx.where(indices < 0, -indices, indices)
    indices = mx.where(indices >= pad_to_samples, 2 * pad_to_samples - 2 - indices, indices)
    window = 0.5 - 0.5 * mx.cos(2 * math.pi * mx.arange(400) / 400)
    power = mx.abs(mx.fft.rfft(waveform[indices] * window, n=400)) ** 2
    log_spec = mx.log10(mx.maximum(power @ mx.array(_slaney_filters(n_mels)).T, 1e-10))
    log_spec = mx.maximum(log_spec, mx.max(log_spec) - 8.0)
    return ((log_spec + 4.0) / 4.0).T
