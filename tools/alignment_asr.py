#!/usr/bin/env python3
"""Full-weight alignment adapter: FunASR/PyTorch reference vs native MLX.

Purpose
-------
The synthetic suites prove each port against tiny configurations. They cannot
prove that a **full official checkpoint** produces the same numbers as its
PyTorch source. This tool loads one shared official checkpoint into two
independent backends, feeds both the same input, and saves per-stage tensors and
token ids for numerical comparison.

Hard rules this module obeys
---------------------------
1. **Never mix backends.** The reference process imports FunASR + Torch only; the
   native process imports MLX only. Importing the MLX frontend inside the
   reference process would not just waste memory, it would make the frontend
   comparison tautological -- both sides would be running the *same* code, so
   agreement would prove nothing.
2. **No hand-rolled reference math.** FunASR models are constructed from the
   official YAML through their real constructors; canonical Whisper is loaded by
   AST-extracting the actual upstream definitions. Every reimplementation here
   would be a second thing to be wrong.
3. **Reference and native use real frontends**, each with the *same* official
   CMVN and configuration, loaded independently.
4. **Explicit float32** everywhere; every saved tensor's dtype is recorded.
5. **No hidden normalisation.** Nothing here rescales or clips a tensor.
6. **Strict key inventory in both directions**, and any unsupported variant is
   refused with a reason rather than silently substituted.

Both numeric runners must be invoked as separate processes; see ``main``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Where the lead stages official checkpoints, configs, CMVN and reference audio.
DEFAULT_ASSETS_ROOT = Path("/Volumes/2T/funasr-mlx-alignment")

#: Staged canonical OpenAI Whisper source, used to build the reference.
CANONICAL_WHISPER_SOURCE = Path("/Users/baicai1145/repos/funasr-mlx/.tmp/openai-whisper-model.py")

#: Per-tensor element cap for saved arrays. Truncation is recorded, never silent:
#: a partial activation must not be mistaken for a full one.
MAX_SAVED_ELEMENTS = 4_000_000

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Variant:
    """One alignable checkpoint and everything needed to run both backends."""

    key: str
    family: str
    checkpoint: str
    config: str
    max_seconds: float
    #: Official Kaldi fbank + CMVN settings, shared by both frontends.
    frontend: dict[str, Any]
    #: FunASR import path + class for the reference backend.
    reference_import: str = ""
    reference_class: str = ""
    #: Modules that must be imported so FunASR's registry can resolve the
    #: ``encoder:``/``decoder:``/``predictor:`` name strings in the config.
    registry_imports: tuple[str, ...] = ()
    notes: str = ""
    #: Whisper-only: the official tiny variant's dimensions are read from the
    #: checkpoint, so nothing is hard-coded here.
    whisper_checkpoint: str = ""


#: Shared frontend settings. Both backends are configured from this one place so
#: a divergence cannot come from the adapter disagreeing with itself.
_ASR_FRONTEND = {
    "fs": 16000,
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "n_mels": 80,
    "frame_length": 25.0,
    "frame_shift": 10.0,
    "dither": 0.0,
    "lfr_m": 7,
    "lfr_n": 6,
    "upsacle_samples": True,
}

SUPPORTED: tuple[Variant, ...] = (
    Variant(
        key="paraformer-zh",
        family="paraformer",
        checkpoint="models/paraformer-zh/model.pt",
        config="models/paraformer-zh/config.yaml",
        max_seconds=30.0,
        frontend=dict(_ASR_FRONTEND),
        reference_import="funasr.models.paraformer.model",
        reference_class="Paraformer",
        registry_imports=(
            "funasr.models.sanm.encoder",
            "funasr.models.sanm.decoder",
        ),
        notes="Non-autoregressive. Final logits are the DECODER log-softmax.",
    ),
    Variant(
        key="sensevoice-small",
        family="sensevoice",
        checkpoint="models/SenseVoiceSmall/model.pt",
        config="models/SenseVoiceSmall/config.yaml",
        max_seconds=30.0,
        # SenseVoiceSmall consumes a 560-dim fbank; the official frontend builds
        # it from 80 mels x 7 LFR frames.
        frontend={**_ASR_FRONTEND, "n_mels": 80, "lfr_m": 7, "lfr_n": 6},
        reference_import="funasr.models.sense_voice.model",
        reference_class="SenseVoiceSmall",
        registry_imports=("funasr.models.sanm.encoder",),
        notes="CTC. Conditioning ids (lid/textnorm/event-emo) are recorded.",
    ),
    Variant(
        key="whisper-tiny",
        family="whisper",
        checkpoint="models/whisper-tiny/model.pt",
        config="models/whisper-tiny/dims.json",
        max_seconds=30.0,
        frontend={"fs": 16000, "n_mels": 80, "n_fft": 400, "hop_length": 160},
        reference_import="",
        reference_class="",
        notes=(
            "Canonical OpenAI module AST-loaded from the staged model.py. "
            "Bounded first full-model proof only; not evidence for larger variants."
        ),
    ),
)

#: Refused with a reason rather than quietly aliased to a smaller checkpoint.
UNSUPPORTED: dict[str, str] = {
    "whisper-large": "Not aligned. Roughly 1.5 GB of weights needs its own memory "
    "budget and approval; a tiny result is not evidence about it.",
    "whisper-large-v3-turbo": "Not aligned. Same budget reason as whisper-large, "
    "and the turbo variant changes decoder depth, so tiny says nothing about it.",
    "whisper-base": "Not aligned. Structurally identical to tiny and covered by it; "
    "add it when a full base run is actually needed.",
    "paraformer-zh-streaming": "Not aligned. Streaming chunking changes the LFR and "
    "cache paths and the offline port has no streaming counterpart.",
    "sensevoice-large": "Not aligned. Different encoder width and depth.",
}

SUPPORTED_BY_KEY = {variant.key: variant for variant in SUPPORTED}


class AlignmentError(ValueError):
    """The requested alignment cannot proceed. Carries every problem found."""


# --------------------------------------------------------------------------- #
# Manifest and spec (pure; no Torch, no MLX)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sample:
    key: str
    audio: str
    duration_seconds: float
    transcript: str = ""


def load_manifest(path: str | Path) -> list[Sample]:
    """Read the manifest, rejecting anything the two backends could disagree on."""
    path = Path(path)
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise AlignmentError(f"manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise AlignmentError(f"manifest is not valid JSON: {error}") from error

    entries = document.get("samples") if isinstance(document, dict) else document
    if not isinstance(entries, list) or not entries:
        raise AlignmentError("manifest must contain a non-empty 'samples' list")

    problems: list[str] = []
    samples: list[Sample] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"sample {index}: expected an object")
            continue
        missing = [name for name in ("key", "audio", "duration_seconds") if not entry.get(name)]
        if missing:
            problems.append(f"sample {index}: missing {', '.join(missing)}")
            continue
        key = str(entry["key"])
        if key in seen:
            problems.append(f"sample {index}: duplicate key {key!r}")
            continue
        seen.add(key)
        duration = entry["duration_seconds"]
        if not isinstance(duration, (int, float)) or duration <= 0:
            problems.append(f"sample {index}: duration_seconds must be positive")
            continue
        samples.append(
            Sample(
                key=key,
                audio=str(entry["audio"]),
                duration_seconds=float(duration),
                transcript=str(entry.get("transcript", "")),
            )
        )
    if problems:
        raise AlignmentError("; ".join(problems))
    return samples


def resolve_variant(requested: str) -> Variant:
    if requested in SUPPORTED_BY_KEY:
        return SUPPORTED_BY_KEY[requested]
    if requested in UNSUPPORTED:
        raise AlignmentError(f"{requested} is not aligned: {UNSUPPORTED[requested]}")
    raise AlignmentError(
        f"unknown variant {requested!r}; aligned variants are {', '.join(sorted(SUPPORTED_BY_KEY))}"
    )


def build_alignment_spec(
    manifest_path: str | Path,
    variant_name: str,
    output_root: str | Path,
    *,
    assets_root: str | Path = DEFAULT_ASSETS_ROOT,
    max_samples: int = 8,
    cmvn_name: str = "am.mvn",
    canonical_whisper_source: str | Path = CANONICAL_WHISPER_SOURCE,
) -> dict[str, Any]:
    """Produce the JSON-serialisable plan both backend runs execute.

    Free of heavy imports on purpose: the spec is cheap to build, reviewable in a
    diff, and identical for both backends by construction.
    """
    variant = resolve_variant(variant_name)
    if max_samples <= 0:
        raise AlignmentError("--max-samples must be positive")
    assets_root = Path(assets_root)

    inputs: list[dict[str, Any]] = []
    problems: list[str] = []
    for sample in load_manifest(manifest_path):
        if sample.duration_seconds > variant.max_seconds:
            problems.append(
                f"{sample.key}: {sample.duration_seconds:.2f}s exceeds the "
                f"{variant.key} bound of {variant.max_seconds:.2f}s"
            )
            continue
        audio = Path(sample.audio)
        if not audio.is_file():
            problems.append(f"{sample.key}: audio not found at {audio}")
            continue
        inputs.append(
            {
                "key": sample.key,
                "audio": str(audio),
                "duration_seconds": sample.duration_seconds,
                "transcript": sample.transcript,
            }
        )
        if len(inputs) == max_samples:
            break
    if problems:
        raise AlignmentError("; ".join(problems))
    if not inputs:
        raise AlignmentError("no usable samples remain")

    spec: dict[str, Any] = {
        "schema": "funasr-mlx-alignment/2",
        "variant": {
            "key": variant.key,
            "family": variant.family,
            "checkpoint": str(assets_root / variant.checkpoint),
            "config": str(assets_root / variant.config),
            "cmvn": str(assets_root / variant.config.rsplit("/", 1)[0] / cmvn_name)
            if not variant.family == "whisper"
            else None,
            "frontend": variant.frontend,
            "reference_import": variant.reference_import,
            "reference_class": variant.reference_class,
            "registry_imports": list(variant.registry_imports),
            "max_seconds": variant.max_seconds,
            "notes": variant.notes,
        },
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "batch_size": 1,
            "fixed_order": [item["key"] for item in inputs],
        },
        "inputs": inputs,
        "output_root": str(output_root),
        "dtype": "float32",
        "stages": ["frontend", "encoder", "pre_decoder", "final_logits", "token_ids"],
        "limits": {"max_saved_elements": MAX_SAVED_ELEMENTS},
        "unsupported_variants": dict(sorted(UNSUPPORTED.items())),
    }
    if variant.family == "whisper":
        spec["variant"]["canonical_source"] = str(canonical_whisper_source)
        source = Path(canonical_whisper_source)
        if not source.is_file():
            raise AlignmentError(
                f"canonical Whisper source is missing at {source}; the reference "
                "must be the actual upstream module, not a reimplementation"
            )
    return spec


# --------------------------------------------------------------------------- #
# Checkpoint inventory
# --------------------------------------------------------------------------- #
def inventory_report(
    model_keys: Iterable[str],
    checkpoint_keys: Iterable[str],
    *,
    label: str = "",
) -> dict[str, Any]:
    """Compare two key sets, reporting every difference in both directions."""
    expected = sorted(str(key) for key in model_keys)
    provided = sorted(str(key) for key in checkpoint_keys)
    expected_set, provided_set = set(expected), set(provided)
    missing = sorted(expected_set - provided_set)
    unexpected = sorted(provided_set - expected_set)
    return {
        "label": label,
        "model_key_count": len(expected),
        "checkpoint_key_count": len(provided),
        "missing": missing,
        "unexpected": unexpected,
        "exact_match": not missing and not unexpected,
        "verdict": "ok" if not missing and not unexpected else "mismatch",
    }


# --------------------------------------------------------------------------- #
# Comparison (pure; metadata only)
# --------------------------------------------------------------------------- #
def _stage(sample: dict[str, Any], stage: str) -> dict[str, Any]:
    for entry in sample.get("stages", []):
        if entry.get("stage") == stage:
            return entry
    return {}


def compare_alignment(
    reference_meta: dict[str, Any], native_meta: dict[str, Any]
) -> dict[str, Any]:
    """Diff two runs' metadata. Loads no tensors and declares no tolerance.

    Numeric divergence is left to the tool that owns the ``.npz`` files: deciding
    what counts as a difference is an adjudication this script should not make
    unilaterally.
    """
    findings: list[dict[str, Any]] = []

    if reference_meta.get("variant", {}).get("key") != native_meta.get("variant", {}).get("key"):
        findings.append(
            {
                "kind": "variant_mismatch",
                "detail": (
                    f"reference={reference_meta.get('variant', {}).get('key')!r} "
                    f"native={native_meta.get('variant', {}).get('key')!r}"
                ),
            }
        )
    for field_name in ("dtype", "frontend_config", "cmvn"):
        if reference_meta.get(field_name) != native_meta.get(field_name):
            findings.append(
                {
                    "kind": f"{field_name}_mismatch",
                    "detail": f"reference={reference_meta.get(field_name)!r} native={native_meta.get(field_name)!r}",
                }
            )

    reference_samples = {item["key"]: item for item in reference_meta.get("samples", [])}
    native_samples = {item["key"]: item for item in native_meta.get("samples", [])}
    if sorted(reference_samples) != sorted(native_samples):
        findings.append(
            {
                "kind": "sample_set_mismatch",
                "detail": f"reference={sorted(reference_samples)} native={sorted(native_samples)}",
            }
        )

    for key in sorted(set(reference_samples) & set(native_samples)):
        reference_sample, native_sample = reference_samples[key], native_samples[key]
        if reference_sample.get("token_ids") != native_sample.get("token_ids"):
            findings.append(
                {
                    "kind": "token_ids_differ",
                    "sample": key,
                    "reference": reference_sample.get("token_ids"),
                    "native": native_sample.get("token_ids"),
                }
            )
        for field_name in ("conditioning_ids", "tokenizer"):
            if reference_sample.get(field_name) != native_sample.get(field_name):
                findings.append(
                    {
                        "kind": f"{field_name}_differ",
                        "sample": key,
                        "reference": reference_sample.get(field_name),
                        "native": native_sample.get(field_name),
                    }
                )
        for stage_name in ("frontend", "encoder", "pre_decoder", "final_logits"):
            reference_stage = _stage(reference_sample, stage_name)
            native_stage = _stage(native_sample, stage_name)
            if not reference_stage or not native_stage:
                continue
            if list(reference_stage.get("shape", [])) != list(native_stage.get("shape", [])):
                findings.append(
                    {
                        "kind": "stage_shape_mismatch",
                        "sample": key,
                        "stage": stage_name,
                        "reference": reference_stage.get("shape"),
                        "native": native_stage.get("shape"),
                    }
                )
            if reference_stage.get("dtype") != native_stage.get("dtype"):
                findings.append(
                    {
                        "kind": "stage_dtype_mismatch",
                        "sample": key,
                        "stage": stage_name,
                        "reference": reference_stage.get("dtype"),
                        "native": native_stage.get("dtype"),
                    }
                )
            if reference_stage.get("truncated") or native_stage.get("truncated"):
                findings.append(
                    {
                        "kind": "stage_truncated",
                        "sample": key,
                        "stage": stage_name,
                        "detail": "a saved activation was capped; it is not the full tensor",
                    }
                )

    return {
        "schema": "funasr-mlx-alignment-diff/2",
        "variant": reference_meta.get("variant", {}).get("key"),
        "finding_count": len(findings),
        "findings": findings,
        "clean": not findings,
        "note": (
            "Shape/dtype/token/conditioning agreement only, and any truncated "
            "stage is reported as a finding. Numeric divergence is measured over "
            "the saved .npz arrays by the numerical tool; no tolerance is declared."
        ),
    }


# --------------------------------------------------------------------------- #
# Backend-agnostic tensor recording
# --------------------------------------------------------------------------- #
def _to_numpy(value: Any) -> Any:
    """Convert a Torch or MLX tensor to numpy float32 without importing either."""
    import numpy as np

    if hasattr(value, "detach"):  # torch
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def _bounded(value: Any) -> tuple[Any, bool]:
    """Return ``(array, truncated)`` after enforcing the element cap."""
    array = _to_numpy(value)
    if array.size <= MAX_SAVED_ELEMENTS:
        return array, False
    return array.reshape(-1)[:MAX_SAVED_ELEMENTS].reshape(1, -1), True


def _stage_record(stage: str, value: Any, *, path: Path | None = None) -> dict[str, Any]:
    array, truncated = _bounded(value)
    record: dict[str, Any] = {
        "stage": stage,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "truncated": truncated,
    }
    if path is not None:
        record["npz_key"] = path.stem
    return record


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")


def _load_audio(path: str) -> Any:
    """Load mono float32 at the model rate. No resampling, ever."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise AlignmentError(f"{path}: expected {SAMPLE_RATE} Hz, received {rate} Hz")
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)


def _load_checkpoint_state(path: Path) -> dict[str, Any]:
    """Read an official checkpoint safely and without doubling process RSS.

    ``weights_only=True`` refuses arbitrary pickle globals even though these
    checkpoints are trusted and pinned; the cost is one argument. ``mmap=True``
    avoids a second full copy of the weights in RSS, which for a full
    SenseVoiceSmall or Whisper checkpoint is the difference between fitting the
    memory guard and not.
    """
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    for container in ("model", "state_dict", "model_state_dict"):
        if isinstance(payload, dict) and container in payload:
            payload = payload[container]
            break
    if not isinstance(payload, dict):
        raise AlignmentError(f"{path} does not contain a state_dict mapping")
    return {str(key): value for key, value in payload.items()}


def _split_funasr_config(config: dict[str, Any]) -> dict[str, Any]:
    """Split an official FunASR YAML into the constructor's keyword arguments.

    The official files nest most sub-configs under ``model_conf`` while the
    constructor also accepts ``encoder_conf``/``decoder_conf``/... at top level,
    and takes the submodule *names* (``encoder: sanm_encoder``) as strings that
    resolve through FunASR's registry. Splitting here keeps that mapping visible
    and checkable instead of buried in a splat.
    """
    model_conf = dict(config.get("model_conf") or {})
    for name in ("encoder_conf", "decoder_conf", "predictor_conf", "ctc_conf"):
        # ``name not in model_conf`` tests the key, not the value; testing the
        # value would both be wrong and raise, since a sub-config is unhashable.
        if config.get(name) is not None and name not in model_conf:
            model_conf[name] = config[name]
    if not model_conf:
        raise AlignmentError("config has no model_conf; refusing to guess the model's structure")
    return model_conf


# --------------------------------------------------------------------------- #
# Reference backend (Torch). Imports no MLX, ever.
# --------------------------------------------------------------------------- #
def run_reference(spec: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Any]:
    """Run the PyTorch source: real config, real constructors, real frontend."""
    import torch

    variant = spec["variant"]
    output_root = Path(output_root or spec["output_root"]) / "reference"
    output_root.mkdir(parents=True, exist_ok=True)

    if variant["family"] == "whisper":
        model, extras = _build_canonical_whisper(variant)
        loader = None
    else:
        model, extras = _build_funasr_reference(variant)
        loader = model.load_state_dict

    state = _load_checkpoint_state(Path(variant["checkpoint"]))
    inventory = inventory_report(model.state_dict().keys(), state.keys(), label="reference")
    if inventory["verdict"] != "ok":
        raise AlignmentError(
            f"reference key inventory mismatch: missing={inventory['missing']} "
            f"unexpected={inventory['unexpected']}"
        )
    loader(state, strict=True)
    model = model.float().eval()

    frontend = _build_reference_frontend(variant)
    samples = [
        _run_reference_sample(model, frontend, variant, item, output_root, torch)
        for item in spec["inputs"]
    ]

    metadata = {
        "schema": "funasr-mlx-alignment-run/2",
        "backend": "reference",
        "variant": variant,
        "dtype": "float32",
        "frontend_config": variant["frontend"],
        "cmvn": variant["cmvn"],
        "inventory": inventory,
        "extras": extras,
        "samples": samples,
        "torch_version": torch.__version__,
        "device": "cpu",
    }
    _write_metadata(output_root / "metadata.json", metadata)
    return metadata


def _build_funasr_reference(variant: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Construct the real FunASR model from the official config.

    Registry imports come first: the config names submodules as strings
    (``encoder: sanm_encoder``) and those resolve through modules that register
    themselves on import, so skipping them would fail with an opaque lookup error.
    """
    import importlib

    import yaml

    for module_path in variant["registry_imports"]:
        importlib.import_module(module_path)

    config = yaml.safe_load(Path(variant["config"]).read_text())
    model_conf = _split_funasr_config(config)
    module = importlib.import_module(variant["reference_import"])
    model = getattr(module, variant["reference_class"])(**model_conf)

    extras: dict[str, Any] = {"model_conf_keys": sorted(model_conf)}
    # Conditioning id tables must be recorded, not assumed: they decide what the
    # model actually consumed and are impossible to infer from a tensor shape.
    for attribute in ("lid_dict", "textnorm_dict", "lid_int_dict"):
        table = getattr(model, attribute, None)
        if isinstance(table, dict):
            extras[attribute] = {str(k): int(v) for k, v in table.items()}
    extras["ctc_weight"] = float(getattr(model, "ctc_weight", 0.0) or 0.0)
    extras["has_ctc_head"] = hasattr(model, "ctc") and getattr(model, "ctc", None) is not None
    return model, extras


def _build_reference_frontend(variant: dict[str, Any]) -> Any:
    """The real FunASR ``WavFrontend``, with the official CMVN and Kaldi scaling."""
    import importlib

    for module_path in ("funasr.frontends.wav_frontend",):
        importlib.import_module(module_path)
    frontend_class = _resolve_wav_frontend()

    settings = dict(variant["frontend"])
    cmvn = settings.pop("cmvn_file", None) or variant["cmvn"]
    kwargs = {
        "fs": int(settings["fs"]),
        "window": settings.get("window", "hanning"),
        "n_mels": int(settings["n_mels"]),
        "frame_length": float(settings["frame_length"]),
        "frame_shift": float(settings["frame_shift"]),
        "dither": float(settings.get("dither", 0.0)),
        "lfr_m": int(settings["lfr_m"]),
        "lfr_n": int(settings["lfr_n"]),
        "upsacle_samples": bool(settings.get("upsacle_samples", True)),
    }
    return frontend_class(cmvn_file=cmvn, **kwargs)


def _resolve_wav_frontend() -> Any:
    """Locate FunASR's ``WavFrontend`` wherever this checkout keeps it."""
    import importlib

    for module_path, attribute in (
        ("funasr.frontends.wav_frontend", "WavFrontend"),
        ("funasr.frontends.default", "WavFrontend"),
        ("funasr.frontends", "WavFrontend"),
    ):
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        if hasattr(module, attribute):
            return getattr(module, attribute)
    raise AlignmentError(
        "FunASR WavFrontend not found; the reference must use the source frontend, "
        "not a reimplementation and not the MLX one"
    )


def _run_reference_sample(
    model: Any,
    frontend: Any,
    variant: dict[str, Any],
    item: dict[str, Any],
    output_root: Path,
    torch: Any,
) -> dict[str, Any]:
    import numpy as np

    audio = _load_audio(item["audio"])
    with torch.no_grad():
        fbank = frontend(audio)
        fbank = np.asarray(fbank, dtype=np.float32)
        speech = torch.from_numpy(fbank)[None, :, :]
        lengths = torch.tensor([fbank.shape[0]], dtype=torch.int32)

        finals: dict[str, Any] = {}
        token_ids: list[int] | None = None
        conditioning: dict[str, Any] = {}

        if variant["family"] == "paraformer":
            encoder_out, encoder_out_lens = model.encoder(speech, lengths)
            sematic_embeds, token_length, _, _ = model.calc_predictor(encoder_out, encoder_out_lens)
            ys_pad_lens = token_length
            # The published model is ctc_weight=0, so self.ctc may not exist. The
            # decoder log-softmax is the real final logits.
            decoder_out, ys_pad_lens = model.cal_decoder_with_predictor(
                encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens
            )
            finals = {
                "encoder": encoder_out,
                "pre_decoder": sematic_embeds,
                "final_logits": decoder_out,
            }
            token_ids = [int(v) for v in decoder_out[0].argmax(dim=-1)[: int(ys_pad_lens[0])]]
        elif variant["family"] == "sensevoice":
            speech, lengths, conditioning = _apply_sensevoice_conditioning(
                model, speech, lengths, torch
            )
            encoder_out, encoder_out_lens = model.encoder(speech, lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            ctc_logits = _sensevoice_ctc_logits(model, encoder_out)
            finals = {
                "encoder": encoder_out,
                "pre_decoder": encoder_out,
                "final_logits": ctc_logits,
            }
            token_ids = [int(v) for v in ctc_logits[0].argmax(dim=-1)]
        else:  # pragma: no cover - dispatched earlier
            raise AlignmentError(f"reference path for {variant['family']} is not wired")

    stages = [_stage_record("frontend", fbank, path=output_root / f"{item['key']}_frontend.npz")]
    saved: dict[str, Any] = {"frontend": fbank}
    for stage_name, value in finals.items():
        stages.append(
            _stage_record(stage_name, value, path=output_root / f"{item['key']}_{stage_name}.npz")
        )
        saved[stage_name] = _bounded(value)[0]
    np.savez(output_root / f"{item['key']}.npz", **saved)

    return {
        "key": item["key"],
        "audio": item["audio"],
        "duration_seconds": item["duration_seconds"],
        "stages": stages,
        "token_ids": token_ids,
        "conditioning_ids": conditioning,
    }


def _apply_sensevoice_conditioning(
    model: Any, speech: Any, lengths: Any, torch: Any
) -> tuple[Any, Any, dict[str, Any]]:
    """Prepend SenseVoice's conditioning tokens exactly as ``inference`` does.

    The conditioning is a fixed id sequence built from ``lid_dict`` and
    ``textnorm_dict`` plus the hard-coded event/emotion pair ``[1, 2]``. It is
    1 + 1 + 3 = 5 frames of prefix, and the lengths must grow to match or the
    encoder masks real audio away.
    """
    language = "auto"
    textnorm = "woitn"
    lid = int(getattr(model, "lid_dict", {}).get(language, 0))
    norm = int(getattr(model, "textnorm_dict", {}).get(textnorm, 0))
    event_emo = [1, 2]

    batch = speech.size(0)
    language_query = model.embed(torch.LongTensor([[lid]])).repeat(batch, 1, 1)
    textnorm_query = model.embed(torch.LongTensor([[norm]])).repeat(batch, 1, 1)
    event_emo_query = model.embed(torch.LongTensor([event_emo])).repeat(batch, 1, 1)
    input_query = torch.cat((language_query, event_emo_query), dim=1)

    speech = torch.cat((textnorm_query, speech), dim=1)
    speech = torch.cat((input_query, speech), dim=1)
    lengths = lengths + 1 + len(input_query[0])
    return (
        speech,
        lengths,
        {
            "language": language,
            "language_id": lid,
            "textnorm": textnorm,
            "textnorm_id": norm,
            "event_emo_ids": event_emo,
            "prefix_frames": 1 + len(input_query[0]),
        },
    )


def _sensevoice_ctc_logits(model: Any, encoder_out: Any) -> Any:
    """SenseVoice's CTC logits, refusing to guess if the head is named otherwise."""
    head = getattr(model, "ctc", None)
    if head is None:
        raise AlignmentError(
            "SenseVoiceSmall has no ctc head; refusing to substitute another head for the real one"
        )
    return head.log_softmax(head(encoder_out))


# --------------------------------------------------------------------------- #
# Canonical Whisper reference, AST-loaded from the staged upstream file
# --------------------------------------------------------------------------- #
WHISPER_DEFINITIONS = (
    "ModelDimensions",
    "LayerNorm",
    "Linear",
    "Conv1d",
    "sinusoids",
    "MultiHeadAttention",
    "ResidualAttentionBlock",
    "AudioEncoder",
    "TextDecoder",
    "Whisper",
)


def reference_definitions(source: str | Path, names: Sequence[str]) -> dict[str, Any]:
    """Load exactly these definitions out of the upstream source and exec them.

    This is how the real canonical module gets used without installing a
    ``whisper`` package and without copying its code here. Every requested name
    must actually be present: a missing one is an error, never a silent skip.
    """
    import torch

    source = Path(source)
    if not source.is_file():
        raise AlignmentError(f"canonical Whisper source not found: {source}")
    tree = ast.parse(source.read_text(), filename=str(source))
    found = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in set(names)
    }
    missing = [name for name in names if name not in found]
    if missing:
        raise AlignmentError(
            f"canonical Whisper source is missing {missing}; the reference must be "
            "the real upstream module"
        )

    from dataclasses import dataclass
    from typing import Dict, Iterable, List, Optional, Tuple

    namespace: dict[str, Any] = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "Tensor": torch.Tensor,
        "Dict": Dict,
        "Iterable": Iterable,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "dataclass": dataclass,
        "math": math,
        "__builtins__": __builtins__,
    }
    exec(  # noqa: S102 - executing the upstream source is the point
        compile(ast.Module(body=list(found.values()), type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


def _build_canonical_whisper(variant: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Instantiate canonical Whisper from the official tiny dimensions."""
    reference = reference_definitions(variant["canonical_source"], WHISPER_DEFINITIONS)
    # SDPA changes numerics; the comparison needs the plain path.
    reference["MultiHeadAttention"].use_sdpa = False

    dimensions = json.loads(Path(variant["config"]).read_text())
    model = reference["Whisper"](reference["ModelDimensions"](**dimensions)).eval()
    return model, {"dimensions": dimensions, "use_sdpa": False}


# --------------------------------------------------------------------------- #
# Native backend (MLX). Imports no Torch at module scope.
# --------------------------------------------------------------------------- #
def run_native(spec: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Any]:
    """Run the native MLX port with its own frontend and the same config."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    import funasr_mlx
    from funasr_mlx.frontend import FbankConfig, WavFrontend

    variant = spec["variant"]
    output_root = Path(output_root or spec["output_root"]) / "native"
    output_root.mkdir(parents=True, exist_ok=True)

    model = _build_native_model(variant, funasr_mlx)
    state = _load_checkpoint_state(Path(variant["checkpoint"]))
    expected = dict(tree_flatten(model.parameters()))
    inventory = inventory_report(expected.keys(), state.keys(), label="native")
    if inventory["verdict"] != "ok":
        raise AlignmentError(
            f"native key inventory mismatch: missing={inventory['missing']} "
            f"unexpected={inventory['unexpected']}"
        )
    funasr_mlx.models.sanm.load_strict(model, state, convert=True, dtype=mx.float32)
    mx.eval(model.parameters())

    frontend = WavFrontend(_native_frontend_config(variant, FbankConfig), cmvn=variant["cmvn"])
    samples = [
        _run_native_sample(model, frontend, variant, item, output_root, mx)
        for item in spec["inputs"]
    ]

    metadata = {
        "schema": "funasr-mlx-alignment-run/2",
        "backend": "native",
        "variant": variant,
        "dtype": "float32",
        "frontend_config": variant["frontend"],
        "cmvn": variant["cmvn"],
        "inventory": inventory,
        "samples": samples,
        "default_device": str(mx.default_device()),
    }
    _write_metadata(output_root / "metadata.json", metadata)
    return metadata


def _native_frontend_config(variant: dict[str, Any], config_class: Any) -> Any:
    """Build the MLX frontend config from the *same* settings the spec carries."""
    settings = dict(variant["frontend"])
    fields = {field for field in config_class.__dataclass_fields__}
    kwargs = {key: value for key, value in settings.items() if key in fields}
    if "fs" not in kwargs and "sample_rate" in fields:
        kwargs["sample_rate"] = settings["fs"]
    return config_class(**kwargs)


def _build_native_model(variant: dict[str, Any], funasr_mlx: Any) -> Any:
    import yaml

    family = variant["family"]
    if family == "paraformer":
        from funasr_mlx.models import paraformer as module
    elif family == "sensevoice":
        from funasr_mlx.models import sensevoice as module
    elif family == "whisper":
        from funasr_mlx.models import whisper as module
    else:  # pragma: no cover
        raise AlignmentError(f"native path for {family} is not wired")

    path = Path(variant["config"])
    raw = (
        json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    )
    if family == "paraformer":
        return module.Paraformer(config=module.paraformer_config_from_dict(raw))
    if family == "sensevoice":
        return module.SenseVoiceSmall(module.sensevoice_config_from_dict(raw))
    return module.Whisper(module.whisper_config_from_dict(raw).dims)


def _run_native_sample(
    model: Any,
    frontend: Any,
    variant: dict[str, Any],
    item: dict[str, Any],
    output_root: Path,
    mx: Any,
) -> dict[str, Any]:
    import numpy as np

    audio = _load_audio(item["audio"])
    fbank = np.asarray(frontend(audio), dtype=np.float32)
    speech = mx.array(fbank)[None, :, :]
    lengths = mx.array([fbank.shape[0]], dtype=mx.int32)

    stages = [_stage_record("frontend", fbank, path=output_root / f"{item['key']}_frontend.npz")]
    saved: dict[str, Any] = {"frontend": fbank}
    finals: dict[str, Any] = {}
    token_ids: list[int] | None = None

    if variant["family"] == "paraformer":
        outputs = model.forward(speech, lengths)
        finals = {
            "encoder": outputs.encoder_out,
            "pre_decoder": outputs.acoustic_embeds,
            "final_logits": outputs.acoustic_embeds,
        }
    elif variant["family"] == "sensevoice":
        encoder_out, _ = model.encode(
            speech, lengths, language="auto", use_itn=False, ban_emo_unk=False
        )
        logits = model(speech, lengths, language="auto", use_itn=False, ban_emo_unk=False)
        finals = {"encoder": encoder_out, "pre_decoder": encoder_out, "final_logits": logits}
        token_ids = [int(v) for v in np.asarray(logits[0]).argmax(axis=-1)]
    else:  # pragma: no cover
        raise AlignmentError(f"native path for {variant['family']} is not wired")

    for stage_name, value in finals.items():
        stages.append(
            _stage_record(stage_name, value, path=output_root / f"{item['key']}_{stage_name}.npz")
        )
        saved[stage_name] = _bounded(value)[0]
    np.savez(output_root / f"{item['key']}.npz", **saved)

    return {
        "key": item["key"],
        "audio": item["audio"],
        "duration_seconds": item["duration_seconds"],
        "stages": stages,
        "token_ids": token_ids,
        "conditioning_ids": None,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full-weight ASR/Whisper alignment adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.set_defaults(command="list")

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--manifest", required=True)
    build_parser.add_argument("--variant", required=True)
    build_parser.add_argument("--output-root", required=True)
    build_parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    build_parser.add_argument("--max-samples", type=int, default=8)
    build_parser.add_argument("--cmvn-name", default="am.mvn")
    build_parser.set_defaults(command="build")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--backend", required=True, choices=("reference", "native"))
    run_parser.add_argument("--spec", required=True)
    run_parser.add_argument("--output-root", required=True)
    run_parser.set_defaults(command="run")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference", required=True)
    compare_parser.add_argument("--native", required=True)
    compare_parser.add_argument("--output-root", required=True)
    compare_parser.set_defaults(command="compare")

    args = parser.parse_args(argv)

    if args.command == "list":
        for variant in SUPPORTED:
            print(f"aligned    {variant.key:22s} {variant.checkpoint:38s} {variant.notes}")
        for key, reason in sorted(UNSUPPORTED.items()):
            print(f"refused    {key:22s} {reason}")
        return 0

    if args.command == "build":
        spec = build_alignment_spec(
            args.manifest,
            args.variant,
            args.output_root,
            assets_root=args.assets_root,
            max_samples=args.max_samples,
            cmvn_name=args.cmvn_name,
        )
        destination = Path(args.output_root) / "spec.json"
        _write_metadata(destination, spec)
        print(f"wrote {destination}: {spec['variant']['key']} samples={len(spec['inputs'])}")
        return 0

    if args.command == "run":
        spec = json.loads(Path(args.spec).read_text())
        runner = run_reference if args.backend == "reference" else run_native
        metadata = runner(spec, args.output_root)
        inventory = metadata["inventory"]
        print(
            f"{args.backend}: {metadata['variant']['key']} samples={len(metadata['samples'])} "
            f"keys={inventory['model_key_count']} inventory={inventory['verdict']}"
        )
        return 0

    reference = json.loads(Path(args.reference).read_text())
    native = json.loads(Path(args.native).read_text())
    report = compare_alignment(reference, native)
    destination = Path(args.output_root) / "diff.json"
    _write_metadata(destination, report)
    print(f"wrote {destination}: findings={report['finding_count']} clean={report['clean']}")
    for finding in report["findings"]:
        print(f"  {finding['kind']}: {finding.get('sample', '')} {finding.get('detail', '')}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
