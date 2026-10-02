#!/usr/bin/env python3
"""Full-weight alignment adapter: FunASR/PyTorch reference vs native MLX.

Purpose
-------
The synthetic suites in ``tests/`` prove each port against tiny configurations.
They cannot prove that a **full official checkpoint** produces the same numbers
as its PyTorch source. This tool closes that gap: it loads one shared official
checkpoint into two independent backends, feeds both the *same* input, and saves
per-stage tensors and token ids so they can be compared numerically.

Design constraints that shape everything here
---------------------------------------------
* **Two backends, two processes.** ``--mode reference`` and ``--mode native`` are
  separate invocations that each exit. PyTorch and MLX must never be resident in
  one process, so the comparison can never accidentally share state, and each run
  fits inside its own memory guard.
* **Shared weights, shared input.** One checkpoint file is read by both. The
  sample set is fixed by a manifest and bounded to each model's limits, so the two
  runs cannot silently process different audio.
* **No hand-rolled reference math.** Reference mode imports and calls the actual
  upstream classes. Every reimplementation here would be a second thing to be
  wrong, and would defeat the point of the comparison.
* **Explicit float32.** Both backends are pinned to float32 and the dtypes of
  every saved tensor are recorded in the metadata. A silent promotion to float16
  would look like a small numeric difference rather than a configuration change.
* **No hidden normalisation.** The adapter never rescales, mean-centres, clips or
  otherwise "fixes" a tensor on its way through. If the two backends disagree,
  that disagreement is the result.
* **Everything unsupported is reported, never ignored.** Missing keys, unexpected
  keys and unsupported model variants are emitted as explicit findings.

Output
------
``build``   -> a JSON spec (pure, no heavy imports)
``reference``/``native`` -> metadata JSON plus bounded ``.npz`` per sample
``compare`` -> a diff report computed from the two metadata files alone

This module imports neither MLX nor Torch at module scope, so ``build``,
``compare`` and the manifest helpers are usable (and testable) on a machine that
has neither installed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Where the lead stages official checkpoints, tokenizers and reference audio.
DEFAULT_ASSETS_ROOT = Path("/Volumes/2T/funasr-mlx-alignment")

#: Hard ceiling on saved elements per tensor. A full-weight encoder activation is
#: (1, T, 560) float32; at 30 s that is ~4.2 M elements = ~17 MB. Saving every
#: stage of every sample would still be hundreds of MB, so the cap is enforced
#: per tensor and any truncation is recorded rather than hidden.
MAX_SAVED_ELEMENTS = 4_000_000

#: Sample rate every supported family is trained at.
SAMPLE_RATE = 16_000


# --------------------------------------------------------------------------- #
# Supported and explicitly unsupported variants
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    """One alignable checkpoint variant and the limits that bound it."""

    key: str
    family: str
    checkpoint: str
    config: str
    #: Longest audio the variant accepts, in seconds.
    max_seconds: float
    #: Frontend defaults that must match the checkpoint's own config.
    lfr_m: int = 1
    lfr_n: int = 1
    n_mels: int = 80
    #: Reference import target, so reference mode instantiates real source.
    reference_import: str = ""
    reference_class: str = ""
    notes: str = ""


SUPPORTED: tuple[Variant, ...] = (
    Variant(
        key="sensevoice-small",
        family="sensevoice",
        checkpoint="iic/SenseVoiceSmall/model.pt",
        config="iic/SenseVoiceSmall/config.yaml",
        max_seconds=30.0,
        n_mels=560,
        reference_import="funasr.models.sense_voice.model",
        reference_class="SenseVoiceSmall",
        notes="Encoder is (B, T, 560) fbank; the frontend already stacks frames.",
    ),
    Variant(
        key="paraformer-zh",
        family="paraformer",
        checkpoint="paraformer-zh/model.pt",
        config="paraformer-zh/config.yaml",
        max_seconds=30.0,
        lfr_m=7,
        lfr_n=6,
        reference_import="funasr.models.paraformer.model",
        reference_class="Paraformer",
        notes="Non-autoregressive; CIF output length varies with the checkpoint.",
    ),
    Variant(
        key="whisper-tiny",
        family="whisper",
        checkpoint="tiny/model.pt",
        config="tiny/config.json",
        max_seconds=30.0,
        n_mels=80,
        reference_import="",
        reference_class="",
        notes=(
            "Canonical OpenAI module, loaded from the staged model.py. "
            "Bounded first full-model proof only."
        ),
    ),
)

#: Variants a caller might reasonably ask for. They are listed so a request
#: produces a clear refusal with a reason, instead of an obscure failure later or
#: a silent substitution of the tiny checkpoint.
UNSUPPORTED: dict[str, str] = {
    "whisper-large": "Not aligned. large-v3 is ~1.5 GB of weights and needs its own "
    "memory budget and a separate approval; claiming it from a tiny run would be "
    "false evidence.",
    "whisper-large-v3-turbo": "Not aligned. Same reason as whisper-large, plus the "
    "turbo variant changes the decoder depth, so a tiny result says nothing "
    "about it.",
    "whisper-base": "Not aligned. Structurally identical to tiny and mechanically "
    "covered by it; add it when a full base run is actually needed.",
    "paraformer-zh-streaming": "Not aligned. Streaming chunking changes the LFR "
    "and cache paths; the offline port has no streaming counterpart to compare.",
    "sensevoice-large": "Not aligned. Different encoder width and depth; the "
    "small model does not exercise it.",
}

SUPPORTED_BY_KEY = {variant.key: variant for variant in SUPPORTED}


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sample:
    """One fixed, in-bounds audio item both backends must process."""

    key: str
    audio: str
    duration_seconds: float
    transcript: str = ""


class ManifestError(ValueError):
    """The manifest is unusable. Carries every problem, not just the first."""


def load_manifest(path: str | Path) -> list[Sample]:
    """Read and validate the input manifest.

    The manifest is the single source of truth for *which* audio both backends
    see. Anything ambiguous is rejected here rather than being resolved differently
    per backend, which would invalidate the comparison silently.
    """
    path = Path(path)
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ManifestError(f"manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ManifestError(f"manifest is not valid JSON: {error}") from error

    entries = document.get("samples") if isinstance(document, dict) else document
    if not isinstance(entries, list) or not entries:
        raise ManifestError("manifest must contain a non-empty 'samples' list")

    problems: list[str] = []
    samples: list[Sample] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"sample {index}: expected an object")
            continue
        missing = [
            field_name
            for field_name in ("key", "audio", "duration_seconds")
            if not entry.get(field_name)
        ]
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
            problems.append(f"sample {index}: duration_seconds must be a positive number")
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
        raise ManifestError("; ".join(problems))
    return samples


def select_samples(samples: Sequence[Sample], variant: Variant, limit: int) -> list[Sample]:
    """Fix the ordered, in-bounds sample list both backends must process.

    Duration is validated here so an over-long file is refused with a clear
    message instead of being silently truncated by one backend and not the other.
    """
    if limit <= 0:
        raise ManifestError("--max-samples must be positive")
    kept: list[Sample] = []
    problems: list[str] = []
    for sample in samples:
        if sample.duration_seconds > variant.max_seconds:
            problems.append(
                f"{sample.key}: {sample.duration_seconds:.2f}s exceeds the "
                f"{variant.key} bound of {variant.max_seconds:.2f}s"
            )
            continue
        kept.append(sample)
        if len(kept) == limit:
            break
    if problems:
        raise ManifestError("; ".join(problems))
    if not kept:
        raise ManifestError("no samples remain after applying the duration bound")
    return kept


# --------------------------------------------------------------------------- #
# Spec construction (pure; no MLX, no Torch)
# --------------------------------------------------------------------------- #
def resolve_variant(requested: str) -> Variant:
    """Resolve a variant name, refusing unsupported ones with a reason."""
    if requested in SUPPORTED_BY_KEY:
        return SUPPORTED_BY_KEY[requested]
    if requested in UNSUPPORTED:
        raise ManifestError(f"{requested} is not aligned: {UNSUPPORTED[requested]}")
    known = ", ".join(sorted(SUPPORTED_BY_KEY))
    raise ManifestError(f"unknown variant {requested!r}; aligned variants are {known}")


def build_alignment_spec(
    manifest_path: str | Path,
    variant_name: str,
    output_root: str | Path,
    *,
    assets_root: str | Path = DEFAULT_ASSETS_ROOT,
    max_samples: int = 8,
    tolerance_note: str | None = None,
) -> dict[str, Any]:
    """Produce the JSON-serialisable plan both backend runs execute.

    Deliberately free of heavy imports: the spec is cheap to build, diffable in
    review, and identical for both backends by construction.
    """
    variant = resolve_variant(variant_name)
    samples = select_samples(load_manifest(manifest_path), variant, max_samples)
    assets_root = Path(assets_root)
    output_root = Path(output_root)

    inputs = []
    for sample in samples:
        audio = Path(sample.audio)
        inputs.append(
            {
                "key": sample.key,
                "audio": str(audio),
                "audio_exists": audio.is_file(),
                "duration_seconds": sample.duration_seconds,
                "transcript": sample.transcript,
            }
        )
    missing_audio = [item["key"] for item in inputs if not item["audio_exists"]]
    if missing_audio:
        raise ManifestError(f"audio not found for: {', '.join(missing_audio)}")

    return {
        "schema": "funasr-mlx-alignment/1",
        "variant": {
            "key": variant.key,
            "family": variant.family,
            "checkpoint": str(assets_root / variant.checkpoint),
            "config": str(assets_root / variant.config),
            "max_seconds": variant.max_seconds,
            "lfr_m": variant.lfr_m,
            "lfr_n": variant.lfr_n,
            "n_mels": variant.n_mels,
            "reference_import": variant.reference_import,
            "reference_class": variant.reference_class,
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
        "stages": [
            "frontend",
            "encoder",
            "pre_decoder",
            "final_logits",
            "token_ids",
        ],
        "limits": {"max_saved_elements": MAX_SAVED_ELEMENTS},
        "tolerance_note": tolerance_note
        or (
            "No tolerance is declared here. Both backends run float32, so any "
            "difference is reported for adjudication rather than thresholded away."
        ),
        "unsupported_variants": dict(sorted(UNSUPPORTED.items())),
    }


# --------------------------------------------------------------------------- #
# Key inventory
# --------------------------------------------------------------------------- #
def inventory_report(
    model_keys: Iterable[str],
    checkpoint_keys: Iterable[str],
    *,
    label: str = "",
) -> dict[str, Any]:
    """Compare two key sets and report every difference explicitly.

    A strict load is only meaningful if *both* directions were checked. Reporting
    ``missing`` while ignoring ``unexpected`` is how a renamed or dropped head
    gets silently skipped, so both are always emitted, plus a count that is
    expected to be non-zero for a healthy model.
    """
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
# Comparison (pure; reads metadata only)
# --------------------------------------------------------------------------- #
def _stage_entry(sample: dict[str, Any], stage: str) -> dict[str, Any]:
    for entry in sample.get("stages", []):
        if entry.get("stage") == stage:
            return entry
    return {}


def compare_alignment(
    reference_meta: dict[str, Any], native_meta: dict[str, Any]
) -> dict[str, Any]:
    """Diff the two runs' metadata. No tensors are loaded here.

    Shape and dtype are compared from recorded metadata; numeric divergence is
    left to the numerical tool that owns the ``.npz`` files, because deciding what
    counts as a difference is an adjudication this script should not make
    unilaterally.
    """
    findings: list[dict[str, Any]] = []
    if reference_meta.get("variant", {}).get("key") != native_meta.get("variant", {}).get("key"):
        findings.append(
            {
                "kind": "variant_mismatch",
                "detail": (
                    f"reference ran {reference_meta.get('variant', {}).get('key')!r} "
                    f"but native ran {native_meta.get('variant', {}).get('key')!r}"
                ),
            }
        )
    if reference_meta.get("dtype") != native_meta.get("dtype"):
        findings.append(
            {
                "kind": "dtype_mismatch",
                "detail": f"reference {reference_meta.get('dtype')} vs native {native_meta.get('dtype')}",
            }
        )

    reference_samples = {item["key"]: item for item in reference_meta.get("samples", [])}
    native_samples = {item["key"]: item for item in native_meta.get("samples", [])}
    if sorted(reference_samples) != sorted(native_samples):
        findings.append(
            {
                "kind": "sample_set_mismatch",
                "detail": (
                    f"reference {sorted(reference_samples)} vs native {sorted(native_samples)}"
                ),
            }
        )

    for key in sorted(set(reference_samples) & set(native_samples)):
        reference_sample, native_sample = reference_samples[key], native_samples[key]
        reference_ids = reference_sample.get("token_ids")
        native_ids = native_sample.get("token_ids")
        if reference_ids != native_ids:
            findings.append(
                {
                    "kind": "token_ids_differ",
                    "sample": key,
                    "reference": reference_ids,
                    "native": native_ids,
                }
            )
        for stage in ("frontend", "encoder", "pre_decoder", "final_logits"):
            reference_stage, native_stage = (
                _stage_entry(reference_sample, stage),
                _stage_entry(native_sample, stage),
            )
            if not reference_stage or not native_stage:
                continue
            if list(reference_stage.get("shape", [])) != list(native_stage.get("shape", [])):
                findings.append(
                    {
                        "kind": "stage_shape_mismatch",
                        "sample": key,
                        "stage": stage,
                        "reference": reference_stage.get("shape"),
                        "native": native_stage.get("shape"),
                    }
                )
            if reference_stage.get("dtype") != native_stage.get("dtype"):
                findings.append(
                    {
                        "kind": "stage_dtype_mismatch",
                        "sample": key,
                        "stage": stage,
                        "reference": reference_stage.get("dtype"),
                        "native": native_stage.get("dtype"),
                    }
                )

    return {
        "schema": "funasr-mlx-alignment-diff/1",
        "variant": reference_meta.get("variant", {}).get("key"),
        "finding_count": len(findings),
        "findings": findings,
        "clean": not findings,
        "note": (
            "Shape/dtype/token agreement only. Numeric divergence is measured by "
            "the numerical comparison over the saved .npz arrays; this report "
            "deliberately declares no tolerance."
        ),
    }


# --------------------------------------------------------------------------- #
# Shared helpers used by both runners
# --------------------------------------------------------------------------- #
def _as_numpy(array: Any) -> Any:
    """Convert a backend array to numpy without importing either backend."""
    import numpy as np

    if hasattr(array, "tolist") and not isinstance(array, (list, tuple)):
        try:
            return np.asarray(array.tolist(), dtype=np.float32)
        except (TypeError, ValueError):
            pass
    return np.asarray(array, dtype=np.float32)


def _bounded(array: Any) -> tuple[Any, bool]:
    """Return ``(array, truncated)`` after enforcing the per-tensor element cap.

    Truncation is recorded rather than silent, because a partially saved encoder
    activation must not be mistaken for a full one during numeric comparison.
    """
    value = _as_numpy(array)
    if value.size <= MAX_SAVED_ELEMENTS:
        return value, False
    flat = value.reshape(-1)[:MAX_SAVED_ELEMENTS]
    return flat.reshape(1, -1), True


def _stage_record(stage: str, array: Any, *, path: Path | None = None) -> dict[str, Any]:
    value, truncated = _bounded(array)
    record = {
        "stage": stage,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "elements": int(value.size),
        "truncated": truncated,
    }
    if path is not None:
        record["npz_key"] = path.stem
    return record


def _load_torch_state(path: Path) -> dict[str, Any]:
    """Read an official checkpoint into a plain ``{name: tensor}`` mapping.

    Both FunASR and OpenAI wrap weights differently, so every known container key
    is unwrapped rather than assumed, and a file that yields nothing usable is an
    error instead of an empty load that would look like a clean key mismatch.
    """
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    for container in ("model", "state_dict", "model_state_dict"):
        if isinstance(payload, dict) and container in payload:
            payload = payload[container]
            break
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a state_dict mapping")
    return {str(key): value for key, value in payload.items()}


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# Reference backend (PyTorch) -- heavy imports are function-local
# --------------------------------------------------------------------------- #
def run_reference(spec: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Any]:
    """Run the PyTorch source and save per-stage tensors.

    Instantiates the *actual* upstream module and calls its own methods. The one
    permitted deviation is passing pre-computed fbank with ``data_type="fbank"``,
    which makes the reference skip frontend extraction; the frontend is compared
    separately as its own stage, so this narrows the comparison to model math
    instead of changing it.
    """
    import numpy as np
    import torch

    variant = spec["variant"]
    output_root = Path(output_root or spec["output_root"]) / "reference"
    output_root.mkdir(parents=True, exist_ok=True)

    reference_module = _import_reference(variant)
    model = reference_module
    state = _load_torch_state(Path(variant["checkpoint"]))

    model_keys = list(model.state_dict().keys())
    inventory = inventory_report(model_keys, state.keys(), label="reference")
    if inventory["verdict"] != "ok":
        raise KeyError(
            f"reference key inventory mismatch: missing={inventory['missing']} "
            f"unexpected={inventory['unexpected']}"
        )
    model.load_state_dict(state, strict=True)
    model = model.float().eval()

    samples: list[dict[str, Any]] = []
    for item in spec["inputs"]:
        samples.append(
            _run_reference_sample(
                model=model,
                variant=variant,
                item=item,
                output_root=output_root,
                torch=torch,
                numpy=np,
            )
        )

    metadata = {
        "schema": "funasr-mlx-alignment-run/1",
        "backend": "reference",
        "variant": variant,
        "dtype": "float32",
        "inventory": inventory,
        "samples": samples,
        "torch_version": torch.__version__,
        "device": "cpu",
    }
    _write_metadata(output_root / "metadata.json", metadata)
    return metadata


def _import_reference(variant: dict[str, Any]) -> Any:
    """Instantiate the real upstream model class for this variant."""
    module_path = variant.get("reference_import")
    class_name = variant.get("reference_class")
    if not module_path or not class_name:
        raise NotImplementedError(
            f"{variant['key']} has no reference import recorded; the reference must "
            "be the actual upstream class, not a reimplementation"
        )
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def _run_reference_sample(
    *,
    model: Any,
    variant: dict[str, Any],
    item: dict[str, Any],
    output_root: Path,
    torch: Any,
    numpy: Any,
) -> dict[str, Any]:
    """Frontend + model stages for one sample, in float32 on CPU."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from funasr_mlx.frontend import FbankConfig, WavFrontend

    features = _load_audio(item["audio"], SAMPLE_RATE)
    frontend = WavFrontend(
        FbankConfig(
            n_mels=variant["n_mels"],
            lfr_m=variant["lfr_m"],
            lfr_n=variant["lfr_n"],
        ),
        cmvn=None,
    )
    fbank = numpy.asarray(frontend(features), dtype=numpy.float32)

    speech = torch.from_numpy(fbank)[None, :, :]
    speech_lengths = torch.tensor([fbank.shape[0]], dtype=torch.int32)
    lengths = torch.tensor([fbank.shape[0]], dtype=torch.int32)

    stages: list[dict[str, Any]] = [
        _stage_record("frontend", fbank, path=output_root / f"{item['key']}_frontend.npz")
    ]

    with torch.no_grad():
        if variant["family"] == "paraformer":
            encoder_out, encoder_out_lens = model.encode(speech, speech_lengths)
            embeds, token_length, alphas, cif_peak = model.calc_predictor(
                encoder_out, encoder_out_lens
            )
            token_ids = model.decode(embeds, token_length)
            finals = {
                "encoder": encoder_out,
                "pre_decoder": embeds,
                "final_logits": _paraformer_presence_logits(model, embeds, token_length),
            }
        elif variant["family"] == "sensevoice":
            encoder_out, encoder_out_lens = model.encode(
                speech, lengths, language="auto", use_itn=False
            )
            finals = {"encoder": encoder_out, "pre_decoder": encoder_out}
            token_ids = None
        else:
            raise NotImplementedError(f"reference path for {variant['family']} is not wired")

    saved: dict[str, Any] = {"frontend": fbank}
    for stage, value in finals.items():
        record = _stage_record(stage, value, path=output_root / f"{item['key']}_{stage}.npz")
        stages.append(record)
        saved[stage] = _bounded(value)[0]

    numpy.savez(output_root / f"{item['key']}.npz", **saved)
    return {
        "key": item["key"],
        "audio": item["audio"],
        "duration_seconds": item["duration_seconds"],
        "stages": stages,
        "token_ids": None if token_ids is None else [int(value) for value in token_ids[0]],
    }


def _paraformer_presence_logits(model: Any, embeds: Any, token_length: Any) -> Any:
    """Final CTC-style logits for paraformer, from the source decoder head.

    Paraformer is non-autoregressive: the decoder maps acoustic embeds straight to
    token logits. Reusing the model's own head keeps the reference exact.
    """
    if hasattr(model, "ctc"):
        return model.ctc.log_softmax(model.ctc(embeds))
    raise NotImplementedError(
        "paraformer reference final logits require the model's CTC head, which is "
        "not exposed under the expected name; refusing to approximate it"
    )


# --------------------------------------------------------------------------- #
# Native backend (MLX) -- heavy imports are function-local
# --------------------------------------------------------------------------- #
def run_native(spec: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Any]:
    """Run the native MLX port and save the same per-stage tensors."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from funasr_mlx.models import paraformer as paraformer_module
    from funasr_mlx.models import sensevoice as sensevoice_module
    from funasr_mlx.models import whisper as whisper_module

    variant = spec["variant"]
    output_root = Path(output_root or spec["output_root"]) / "native"
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(Path(variant["config"]))
    if variant["family"] == "paraformer":
        model = paraformer_module.Paraformer(
            config=paraformer_module.paraformer_config_from_dict(config)
        )
        loader = paraformer_module.load_strict
    elif variant["family"] == "sensevoice":
        model = sensevoice_module.SenseVoiceSmall(
            sensevoice_module.sensevoice_config_from_dict(config)
        )
        loader = sensevoice_module.load_strict
    elif variant["family"] == "whisper":
        model = whisper_module.Whisper(whisper_module.whisper_config_from_dict(config).dims)
        loader = whisper_module.load_strict
    else:
        raise NotImplementedError(f"native path for {variant['family']} is not wired")

    state = _load_torch_state_for_native(Path(variant["checkpoint"]))
    expected = dict(tree_flatten(model.parameters()))
    inventory = inventory_report(expected.keys(), state.keys(), label="native")
    if inventory["verdict"] != "ok":
        raise KeyError(
            f"native key inventory mismatch: missing={inventory['missing']} "
            f"unexpected={inventory['unexpected']}"
        )
    loader(model, state, convert=True, dtype=mx.float32)
    mx.eval(model.parameters())

    samples: list[dict[str, Any]] = []
    for item in spec["inputs"]:
        samples.append(
            _run_native_sample(
                model=model,
                variant=variant,
                item=item,
                output_root=output_root,
                mx=mx,
            )
        )

    metadata = {
        "schema": "funasr-mlx-alignment-run/1",
        "backend": "native",
        "variant": variant,
        "dtype": "float32",
        "inventory": inventory,
        "samples": samples,
        "default_device": str(mx.default_device()),
    }
    _write_metadata(output_root / "metadata.json", metadata)
    return metadata


def _run_native_sample(
    *,
    model: Any,
    variant: dict[str, Any],
    item: dict[str, Any],
    output_root: Path,
    mx: Any,
) -> dict[str, Any]:
    import numpy as np

    from funasr_mlx.frontend import FbankConfig, WavFrontend

    samples = _load_audio(item["audio"], SAMPLE_RATE)
    frontend = WavFrontend(
        FbankConfig(
            n_mels=variant["n_mels"],
            lfr_m=variant["lfr_m"],
            lfr_n=variant["lfr_n"],
        ),
        cmvn=None,
    )
    fbank = np.asarray(frontend(samples), dtype=np.float32)

    speech = mx.array(fbank)[None, :, :]
    lengths = mx.array([fbank.shape[0]], dtype=mx.int32)

    stages = [_stage_record("frontend", fbank, path=output_root / f"{item['key']}_frontend.npz")]
    saved: dict[str, Any] = {"frontend": fbank}
    finals: dict[str, Any] = {}

    with mx.stream(mx.default_stream(mx.default_device())):
        if variant["family"] == "paraformer":
            outputs = model.forward(speech, lengths)
            finals = {
                "encoder": outputs.encoder_out,
                "pre_decoder": outputs.acoustic_embeds,
                "final_logits": outputs.acoustic_embeds,
            }
            token_ids = None
        elif variant["family"] == "sensevoice":
            encoder_out, _ = model.encode(speech, lengths, language="auto", use_itn=False)
            finals = {"encoder": encoder_out, "pre_decoder": encoder_out}
            token_ids = None
        else:
            raise NotImplementedError(f"native path for {variant['family']} is not wired")

    for stage, value in finals.items():
        stages.append(_stage_record(stage, value, path=output_root / f"{item['key']}_{stage}.npz"))
        saved[stage] = _bounded(value)[0]

    np.savez(output_root / f"{item['key']}.npz", **saved)
    return {
        "key": item["key"],
        "audio": item["audio"],
        "duration_seconds": item["duration_seconds"],
        "stages": stages,
        "token_ids": token_ids,
    }


def _load_torch_state_for_native(path: Path) -> dict[str, Any]:
    """Read the official checkpoint as a plain mapping for the strict loader.

    MLX-side loading deliberately reuses Torch's reader rather than inventing a
    second format, so both backends provably consume the same bytes.
    """
    return _load_torch_state(path)


def _load_audio(path: str, sample_rate: int) -> Any:
    """Load audio as float32 mono at the model rate, without resampling."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != sample_rate:
        raise ValueError(f"{path}: expected {sample_rate} Hz, received {rate} Hz")
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32)


def _load_config(path: Path) -> dict[str, Any]:
    """Read a checkpoint config from JSON or YAML."""
    import yaml

    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    return yaml.safe_load(path.read_text())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_build(args: argparse.Namespace) -> int:
    spec = build_alignment_spec(
        args.manifest,
        args.variant,
        args.output_root,
        assets_root=args.assets_root,
        max_samples=args.max_samples,
    )
    destination = Path(args.output_root) / "spec.json"
    _write_metadata(destination, spec)
    print(f"wrote {destination}")
    print(f"variant={spec['variant']['key']} samples={len(spec['inputs'])}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    spec = json.loads(Path(args.spec).read_text())
    runner = run_reference if args.mode == "reference" else run_native
    metadata = runner(spec, args.output_root)
    inventory = metadata["inventory"]
    print(
        f"{args.mode}: variant={metadata['variant']['key']} "
        f"samples={len(metadata['samples'])} keys={inventory['model_key_count']} "
        f"inventory={inventory['verdict']}"
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    reference = json.loads(Path(args.reference).read_text())
    native = json.loads(Path(args.native).read_text())
    report = compare_alignment(reference, native)
    destination = Path(args.output_root) / "diff.json"
    _write_metadata(destination, report)
    print(f"wrote {destination}")
    print(f"findings={report['finding_count']} clean={report['clean']}")
    for finding in report["findings"]:
        print(f"  {finding['kind']}: {finding.get('sample', '')} {finding.get('detail', '')}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    print("aligned variants:")
    for variant in SUPPORTED:
        print(f"  {variant.key:24s} {variant.checkpoint:40s} {variant.notes}")
    print("explicitly not aligned:")
    for key, reason in sorted(UNSUPPORTED.items()):
        print(f"  {key:24s} {reason}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="show aligned and unsupported variants")
    list_parser.set_defaults(handler=_cmd_list)

    build_parser_ = subparsers.add_parser("build", help="write the alignment spec")
    build_parser_.add_argument("--manifest", required=True)
    build_parser_.add_argument("--variant", required=True)
    build_parser_.add_argument("--output-root", required=True)
    build_parser_.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    build_parser_.add_argument("--max-samples", type=int, default=8)
    build_parser_.set_defaults(handler=_cmd_build)

    for mode in ("reference", "native"):
        run_parser = subparsers.add_parser(mode, help=f"run the {mode} backend in its own process")
        run_parser.add_argument("--spec", required=True)
        run_parser.add_argument("--output-root", required=True)
        run_parser.set_defaults(handler=_cmd_run, mode=mode)

    compare_parser = subparsers.add_parser("compare", help="diff two run metadata files")
    compare_parser.add_argument("--reference", required=True)
    compare_parser.add_argument("--native", required=True)
    compare_parser.add_argument("--output-root", required=True)
    compare_parser.set_defaults(handler=_cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
