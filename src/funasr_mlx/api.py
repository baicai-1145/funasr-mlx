"""First-class Python API for LLM-ASR transcription.

``transcribe()`` routes through the guarded CLI subprocess (same memory
policy as ``funasr-mlx transcribe``) and parses its JSON output.
``TranscribeWorker`` gives callers a long-lived in-process handle when they
already run inside their own guard (resident model, KV reset-reuse).

Example:
    from funasr_mlx.api import transcribe
    result = transcribe("audio.wav",
                        model="/path/to/qwen3-asr-0.6b-w4a16-awq-w8tower-g32")
    print(result["text"])
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .memory import GB, MB, MemoryPolicy, guarded_child_active

def download_model(repo, *, variant=None, dest=".", owner=None):
    """Download a published checkpoint (see funasr_mlx.download.download)."""
    from .download import download

    kwargs = {"variant": variant, "dest": dest}
    if owner is not None:
        kwargs["owner"] = owner
    return download(repo, **kwargs)


__all__ = ["transcribe", "TranscribeWorker", "download_model"]


class TranscribeWorker:
    """Long-lived transcription worker (use inside an existing guard)."""

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


def transcribe(
    audio: str | Path,
    *,
    model: str | Path,
    family: str | None = None,
    max_tokens: int = 96,
) -> dict[str, Any]:
    """One-shot transcription via the guarded CLI subprocess."""
    if guarded_child_active():
        worker = TranscribeWorker(model, family=family, max_tokens=max_tokens)
        return worker.transcribe(audio)

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
