"""Pseudo-streaming layer for ALL model families.

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/pseudo_streaming.py.md
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np


@dataclass
class StreamEvent:
    # see docs/port-notes/functions.md :: StreamEvent

    text: str
    audio_seconds: float
    wall_ms: float
    is_final: bool = False
    revised_tail: str = ""


@dataclass
class PseudoStream:
    """Growing-buffer pseudo-streaming over any transcription callable.

    Full notes: docs/port-notes/functions.md :: PseudoStream
    """

    transcribe_fn: Callable[[np.ndarray], str]
    sample_rate: int = 16000
    chunk_seconds: float = 1.0
    min_push_seconds: float = 0.4
    _buffer: list[np.ndarray] = field(default_factory=list)
    _n: int = 0
    _last_emitted_seconds: float = -1.0

    def push(self, samples: np.ndarray) -> StreamEvent | None:
        self._buffer.append(np.asarray(samples, dtype=np.float32))
        self._n += len(samples)
        seconds = self._n / self.sample_rate
        if seconds - max(self._last_emitted_seconds, 0) < self.min_push_seconds:
            return None
        return self._emit(seconds, is_final=False)

    def close(self) -> StreamEvent:
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
    # see docs/port-notes/functions.md :: classic_transcriber
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
    # see docs/port-notes/functions.md :: llm_transcriber
    return transcriber.transcribe_samples
