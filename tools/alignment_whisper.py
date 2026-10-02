"""Separate-backend FP32 Whisper tiny diagnosis on complete official weights.

The saved decoder logits are conditioned on a shared official tokenizer prompt.
This is full-model forward parity, not transcription or corpus accuracy acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_inputs(manifests):
    samples = []
    seen = set()
    for manifest in manifests:
        path = Path(manifest)
        if path.stat().st_size > 1_000_000:
            raise ValueError("Diagnostic manifest exceeds metadata bound")
        document = json.loads(path.read_text())
        entries = document.get("samples") if isinstance(document, dict) else document
        if not isinstance(entries, list) or not entries:
            raise ValueError("Expected a nonempty samples list")
        for entry in entries:
            key = entry["key"]
            if key in seen:
                raise ValueError(f"Duplicate input key: {key}")
            if not 0 < entry["duration_seconds"] <= 30:
                raise ValueError("Input exceeds the full-model audio bound")
            if not isinstance(entry.get("language"), str) or not entry["language"]:
                raise ValueError("Prepared inputs must declare their language")
            if not entry.get("sha256") or not entry.get("corpus"):
                raise ValueError("Prepared inputs need audio hashes and corpus provenance")
            seen.add(key)
            samples.append(entry)
            if len(samples) > 128:
                raise ValueError("Diagnostic input count exceeds the 128-sample bound")
    return samples


def read_waveform(entry):
    import numpy as np
    import soundfile as sf

    path = Path(entry["audio"])
    if path.stat().st_size > 32_000_000:
        raise ValueError("Input file exceeds byte bound")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Prepared audio hash mismatch: {path}")
    samples, rate = sf.read(path, dtype="float32")
    if rate != 16000 or samples.ndim != 1 or not 0 < samples.size <= 480000:
        raise ValueError("Whisper input must be mono 16k and at most 30 seconds")
    if not np.isfinite(samples).all():
        raise ValueError("Non-finite audio input")
    if abs(samples.size / rate - entry["duration_seconds"]) > 1 / rate:
        raise ValueError("Prepared duration disagrees with waveform")
    return samples


def save_case(destination, entry, mode, arrays, metadata):
    import numpy as np

    for name, value in arrays.items():
        if value.size == 0 or value.size > 4_000_000 or not np.isfinite(value).all():
            raise ValueError(f"Unbounded, empty or non-finite Whisper stage: {name}")
    key = "whisper_" + entry["key"]
    temporary = destination / f"{key}.{mode}.npz.partial"
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(destination / f"{key}.{mode}.npz")
    (destination / f"{key}.{mode}.json").write_text(
        json.dumps(
            {
                "key": key,
                "audio_sha256": entry["sha256"],
                "corpus": entry["corpus"],
                "duration_seconds": entry["duration_seconds"],
                "dtype": "float32",
                "scope": "full encoder and prompt-conditioned decoder; not transcription acceptance",
                **metadata,
            },
            indent=2,
        )
    )


def run_reference(root, entries, destination):
    from funasr_mlx.memory import guarded_child_active

    if not guarded_child_active():
        raise RuntimeError("Whisper reference execution requires the guarded launcher")
    import numpy as np
    import torch
    import whisper
    from whisper.tokenizer import get_tokenizer

    package_version = whisper.__version__
    if package_version != "20250625":
        raise ValueError(f"Whisper reference version drift: {package_version}")
    whisper.model.MultiHeadAttention.use_sdpa = False
    checkpoint = root / "models/whisper/tiny.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model = whisper.model.Whisper(whisper.model.ModelDimensions(**payload["dims"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.float().eval()
    del payload
    for entry in entries:
        samples = read_waveform(entry)
        language = entry["language"]
        tokenizer = get_tokenizer(model.is_multilingual, language=language, task="transcribe")
        prompt = list(tokenizer.sot_sequence_including_notimestamps)
        tokens = torch.tensor([prompt], dtype=torch.int64)
        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(
                whisper.pad_or_trim(torch.from_numpy(samples)), n_mels=model.dims.n_mels
            )
            encoded = model.encoder(mel[None])
            logits = model.decoder(tokens, encoded)
        save_case(
            destination,
            entry,
            "reference",
            {
                "frontend": mel.detach().numpy().astype(np.float32),
                "encoder": encoded.detach().numpy().astype(np.float32),
                "decoder_logits": logits.detach().numpy().astype(np.float32),
                "prompt_tokens": np.asarray(prompt, dtype=np.int32),
                "next_token": np.asarray(logits[0, -1].argmax().item(), dtype=np.int32),
            },
            {
                "backend": "torch-cpu",
                "package": f"openai-whisper=={package_version}",
                "torch_version": torch.__version__,
                "sdpa": False,
                "language": language,
                "prompt_tokens": prompt,
                "checkpoint": str(checkpoint),
                "dimensions": vars(model.dims),
            },
        )
        print(
            json.dumps({"mode": "reference", "key": entry["key"], "language": language}), flush=True
        )


def run_native(root, entries, destination):
    import sys
    import mlx.core as mx
    import numpy as np
    from funasr_mlx.frontend import whisper_log_mel
    from funasr_mlx.pipeline import load_model

    model, _ = load_model(root / "native/whisper", family="whisper")
    for entry in entries:
        reference = json.loads((destination / f"whisper_{entry['key']}.reference.json").read_text())
        if reference["audio_sha256"] != entry["sha256"]:
            raise ValueError("Reference used different prepared audio")
        prompt = reference["prompt_tokens"]
        samples = read_waveform(entry)
        frames = whisper_log_mel(
            samples, n_mels=model.dims.n_mels, pad_to_samples=2 * model.dims.n_audio_ctx * 160
        )
        if frames.shape != (model.dims.n_mels, 2 * model.dims.n_audio_ctx):
            raise ValueError(f"Unexpected Whisper mel dimensions: {frames.shape}")
        encoded = model.embed_audio(mx.swapaxes(frames, 0, 1)[None])
        tokens = mx.array([prompt], dtype=mx.int32)
        logits = model.logits(tokens, encoded)
        mx.eval(encoded, logits)
        save_case(
            destination,
            entry,
            "native",
            {
                "frontend": np.asarray(frames, dtype=np.float32),
                "encoder": np.asarray(encoded, dtype=np.float32),
                "decoder_logits": np.asarray(logits, dtype=np.float32),
                "prompt_tokens": np.asarray(prompt, dtype=np.int32),
                "next_token": np.asarray(mx.argmax(logits[0, -1]).item(), dtype=np.int32),
            },
            {
                "backend": "mlx-metal",
                "device": str(mx.default_device()),
                "prompt_tokens": prompt,
                "checkpoint": str(root / "native/whisper"),
                "language": reference["language"],
            },
        )
        mx.clear_cache()
        print(json.dumps({"mode": "native", "key": entry["key"]}), flush=True)
    if "torch" in sys.modules:
        raise RuntimeError("Native Whisper diagnosis must not import PyTorch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=("reference", "native"), required=True)
    arguments = parser.parse_args()
    entries = load_inputs(arguments.manifest)
    destination = arguments.root / "results/whisper"
    destination.mkdir(parents=True, exist_ok=True)
    (run_reference if arguments.mode == "reference" else run_native)(
        arguments.root, entries, destination
    )


if __name__ == "__main__":
    main()
