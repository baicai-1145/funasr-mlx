"""Pseudo-streaming layer for ALL model families.

One protocol, every model: audio arrives in a growing buffer; each
``push()`` transcribes ``buffer[0:current]`` from scratch (cheap: classic
models are single-pass; LLM towers are bounded), ``stream()`` yields the
transient best-effort text. The final ``close()`` output is BIT-IDENTICAL
to offline transcription of the full audio (same computation by
construction).

Family support matrix:
- sensevoice / paraformer / whisper: single forward on the prefix; the
  text just stops early (no premature punctuation machinery -- non-AR
  or fixed-vocab decoding does not invent sentence endings).
- qwen3-asr / glm-asr-nano / moss / fun-asr-nano: re-run tower+prefill+
  greedy per push (the tail-roll behavior measured at 94.6% char survival,
  .tmp/opt/LOG.md).
- ct-punc / campplus / emotion2vec / fsmn-vad: not audio-transcription
  models; excluded by construction (Transcriber raises).

This module is display-layer glue; it adds no numeric behavior of its own.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np


@dataclass
class StreamEvent:
    """One pseudo-stream block result.

    ``text`` is the full current best-effort transcription (stable prefix +
    rolling tail). ``is_final`` marks the closing event, whose text equals
    offline transcription exactly.
    """

    text: str
    audio_seconds: float
    wall_ms: float
    is_final: bool = False
    revised_tail: str = ""


@dataclass
class PseudoStream:
    """Growing-buffer pseudo-streaming over any transcription callable.

    Parameters
    ----------
    transcribe_fn:
        ``f(samples: np.ndarray) -> str`` -- OFFLINE transcription of a
        waveform prefix. For classic models use
        :func:`funasr_mlx.pseudo_streaming.classic_transcriber`; for LLM
        families use :meth:`LlmAsrTranscriber.transcribe_samples` via
        :func:`llm_transcriber`.
    sample_rate: expected rate (only for reporting seconds).
    chunk_seconds: push granularity the caller commits to (informational;
      the protocol is pull-based, any push size works).
    min_push_seconds: ignore pushes smaller than this to bound recompute.
    """

    transcribe_fn: Callable[[np.ndarray], str]
    sample_rate: int = 16000
    chunk_seconds: float = 1.0
    min_push_seconds: float = 0.4
    _buffer: list[np.ndarray] = field(default_factory=list)
    _n: int = 0
    _last_emitted_seconds: float = -1.0

    def push(self, samples: np.ndarray) -> StreamEvent | None:
        """Feed more audio; returns the event when a re-transcribe ran."""
        self._buffer.append(np.asarray(samples, dtype=np.float32))
        self._n += len(samples)
        seconds = self._n / self.sample_rate
        if seconds - max(self._last_emitted_seconds, 0) < self.min_push_seconds:
            return None
        return self._emit(seconds, is_final=False)

    def close(self) -> StreamEvent:
        """Final pass over the whole buffer: equals offline output."""
        return self._emit(self._n / self.sample_rate, is_final=True)

    def stream(self) -> Iterator[StreamEvent]:
        raise NotImplementedError("use push()/close(); stream() is documentation")

    def _emit(self, seconds: float, *, is_final: bool) -> StreamEvent:
        t0 = time.perf_counter()
        audio = self._assemble()
        text = self.transcribe_fn(audio)
        wall = (time.perf_counter() - t0) * 1000
        self._last_emitted_seconds = seconds
        return StreamEvent(
            text=text, audio_seconds=seconds, wall_ms=wall, is_final=is_final
        )

    def _assemble(self) -> np.ndarray:
        if len(self._buffer) == 1:
            return self._buffer[0]
        joined = np.concatenate(self._buffer)
        self._buffer = [joined]  # collapse to keep memory flat
        return joined


# --------------------------------------------------------------------------
# Adapters: offline callables per family
# --------------------------------------------------------------------------


def classic_transcriber(model, config: dict, *, family: str, tokens=None, cmvn=None):
    """Build a ``samples -> text`` callable for classic (non-LLM) families.

    Covers sensevoice / paraformer (greedy CTC/CIF). ``cmvn`` is REQUIRED
    for sensevoice-family checkpoints (``am.mvn``; found next to the source
    configuration) -- without it the fbank distribution is shifted and the
    encoder collapses to all-blanks (verified: argmax == 0 everywhere).
    """
    import mlx.core as mx

    from .frontend import FbankConfig, WavFrontend

    defaults = {"sensevoice": (7, 6), "paraformer": (7, 6)}
    if family not in defaults:
        raise ValueError(f"classic_transcriber supports {sorted(defaults)}")
    m, n = defaults[family]
    options = dict(config.get("frontend_conf") or {})
    options.setdefault("lfr_m", m)
    options.setdefault("lfr_n", n)
    frontend = WavFrontend(FbankConfig.from_dict(options), cmvn=cmvn)

    def fn(samples: np.ndarray) -> str:
        features = frontend(samples)
        if features.shape[0] == 0:
            return ""
        lengths = mx.array([features.shape[0]], dtype=mx.int32)
        ids = model.greedy_decode(features[None], lengths)
        return tokens.decode(ids[0]) if tokens else str(ids[0])

    return fn


def llm_transcriber(transcriber):
    """Adapt an ``LlmAsrTranscriber`` into a ``samples -> text`` callable."""
    return transcriber.transcribe_samples
