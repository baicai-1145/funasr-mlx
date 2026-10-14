"""Port notes: docs/port-notes/api.py.md"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def download_model(repo, *, variant=None, dest=".", owner=None):
    # see docs/port-notes/functions.md :: download_model
    from .download import download

    kwargs = {"variant": variant, "dest": dest}
    if owner is not None:
        kwargs["owner"] = owner
    return download(repo, **kwargs)


def pseudo_stream(model, *, family: str | None = None, chunk_seconds: float = 1.0):
    # see docs/port-notes/functions.md :: pseudo_stream
    from .pseudo_streaming import PseudoStream

    transcriber = model if hasattr(model, "transcribe_samples") else TranscribeWorker(
        model, family=family
    )._impl
    return PseudoStream(
        transcribe_fn=transcriber.transcribe_samples,
        chunk_seconds=chunk_seconds,
    )


__all__ = ["transcribe", "TranscribeWorker", "download_model", "pseudo_stream"]


class TranscribeWorker:

    def __init__(self, checkpoint: str | Path, *, family: str | None = None, max_tokens: int = 96):
        from .llm_asr import LlmAsrTranscriber, resolve_family

        checkpoint = Path(checkpoint)
        self._family = resolve_family(checkpoint, family)
        self._impl = LlmAsrTranscriber(checkpoint, family=self._family, max_tokens=max_tokens)

    @property
    def family(self) -> str:
        return self._family

    def transcribe(self, audio: str | Path) -> dict[str, Any]:
        return self._impl.transcribe_file(audio).to_dict()

    def pseudo_stream(self, chunk_seconds: float = 1.0):
        from .pseudo_streaming import PseudoStream

        return PseudoStream(
            transcribe_fn=self._impl.transcribe_samples,
            chunk_seconds=chunk_seconds,
        )


def transcribe(
    audio: str | Path,
    *,
    model: str | Path,
    family: str | None = None,
    max_tokens: int = 96,
) -> dict[str, Any]:
    try:
        worker = TranscribeWorker(model, family=family, max_tokens=max_tokens)
        return worker.transcribe(audio)
    except MemoryBudgetExceeded:
        pass  # fall through to the guarded CLI subprocess

    cmd = [sys.executable, "-m", "funasr_mlx.cli", "transcribe", str(model),
           "--audio", str(audio), "--max-tokens", str(max_tokens)]
    if family:
        cmd += ["--family", family]
    env = dict(os.environ)
    env.pop("FUNASR_MLX_MEMORY_POLICY", None)  # let the CLI parent set its own
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    out, err = proc.communicate(timeout=900)
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{") and '"text"' in line:
            return json.loads(line)
    raise RuntimeError(f"guarded transcribe failed (rc={proc.returncode}): {err[-400:]}")
