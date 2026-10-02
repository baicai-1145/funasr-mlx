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
6. **Strict key inventory in both directions**, plus an explicit **shape**
   check, and any unsupported variant is refused with a reason rather than
   silently substituted.
7. **The reference allocates no weights.** The FunASR model is constructed on
   Torch's ``meta`` device and the official checkpoint is *bound* into it with
   ``load_state_dict(..., assign=True)``, so peak RSS is the checkpoint plus
   activations rather than the checkpoint plus a full random copy of itself.
   Meta construction is not sufficient on its own -- the source also builds
   non-parameter tensors in ``__init__`` that no checkpoint contains -- so those
   are rebuilt through the source's own code and anything unaccounted for is a
   hard error (see :func:`_rebuild_meta_tensors`).
8. **The native side reads only the converted artifact**, through the shipping
   pipeline loader (``pipeline.load_model``). It never opens a Torch
   checkpoint, so the load path being validated is the one the product uses --
   its prevalidation, its bounded shard loop, its strict key and shape checks --
   rather than a second reader that could drift from it.
9. **Nothing is downloaded, inferred or truncated.** The two artefacts per
   variant -- the original FunASR directory and the converted safetensors -- are
   named explicitly in the spec, and audio longer than the variant bound is
   refused after measuring the decoded array, not trusted from the manifest.

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

#: Reference-side parameter initialisation. ``"meta"`` allocates no weight
#: storage at all; ``"cpu"`` is the documented escape hatch that materialises a
#: real random model before the same strict load, for an architecture that turns
#: out not to be constructible on meta. It costs a full-size copy in RSS, so it
#: is opt-in and recorded in the run metadata.
REFERENCE_INIT_DEVICES = ("meta", "cpu")
DEFAULT_REFERENCE_INIT = "meta"

#: Cap on a rebuilt non-parameter table, in rows. The source's own
#: ``PositionalEncoding`` default is 5000; anything larger is refused rather
#: than allocated, so a hostile or mistaken ``max_len`` cannot become an
#: unbounded allocation hidden inside "reconstruct the buffers".
MAX_RECONSTRUCTED_ROWS = 8_192

#: SenseVoice conditioning, shared by both backends so the two cannot be
#: configured differently by accident. ``inference`` resolves the textnorm from
#: ``use_itn`` when no explicit ``text_norm`` is given.
SENSEVOICE_LANGUAGE = "auto"
SENSEVOICE_USE_ITN = False

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
    #: FunASR modules that must be imported so ``tables.encoder_classes`` /
    #: ``decoder_classes`` can resolve the config's submodule NAME strings.
    registry_imports: tuple[str, ...] = ()
    #: The config keys whose values must be resolved from configuration.json
    #: rather than config.yaml, which ships them as ``null``.
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
        checkpoint="models/paraformer/model.pt",
        config="models/paraformer/config.yaml",
        max_seconds=30.0,
        frontend=dict(_ASR_FRONTEND),
        reference_import="funasr.models.paraformer.model",
        reference_class="Paraformer",
        registry_imports=(
            "funasr.models.sanm.encoder",
            "funasr.models.paraformer.decoder",
            "funasr.models.paraformer.cif_predictor",
        ),
        notes=(
            "config.yaml has ctc_weight=0.0, so no CTC head is built. Final "
            "logits are the DECODER log-softmax from cal_decoder_with_predictor."
        ),
    ),
    Variant(
        key="sensevoice-small",
        family="sensevoice",
        checkpoint="models/sensevoice/model.pt",
        config="models/sensevoice/config.yaml",
        max_seconds=30.0,
        # The official config is 80 mels with lfr_m=7, so the frontend's
        # output_size() -- which is what becomes the model's input_size -- is
        # 80 * 7 = 560.
        frontend={**_ASR_FRONTEND, "n_mels": 80, "lfr_m": 7, "lfr_n": 6},
        reference_import="funasr.models.sense_voice.model",
        reference_class="SenseVoiceSmall",
        # SenseVoiceEncoderSmall lives in the same module as SenseVoiceSmall, so
        # importing the reference module registers it.
        registry_imports=("funasr.models.sense_voice.model",),
        notes=(
            "input_size=560 comes from WavFrontend.output_size(), not the config. "
            "CMVN and the BPE model come from configuration.json; config.yaml ships "
            "cmvn_file: null. Conditioning ids (lid/textnorm/event-emo) recorded."
        ),
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
    native_checkpoint: str | Path | None = None,
    reference_init: str = DEFAULT_REFERENCE_INIT,
    canonical_whisper_source: str | Path = CANONICAL_WHISPER_SOURCE,
) -> dict[str, Any]:
    """Produce the JSON-serialisable plan both backend runs execute.

    Free of heavy imports on purpose: the spec is cheap to build, reviewable in a
    diff, and identical for both backends by construction.

    Two artefacts are named per variant and neither is guessed. ``checkpoint`` is
    the original FunASR file, read by the reference through Torch's mmap loader;
    ``native_checkpoint`` is the converted safetensors the native side loads
    through the shipping pipeline. Nothing is downloaded and no path is derived
    from a naming convention.
    """
    variant = resolve_variant(variant_name)
    if max_samples <= 0:
        raise AlignmentError("--max-samples must be positive")
    if reference_init not in REFERENCE_INIT_DEVICES:
        raise AlignmentError(
            f"reference_init must be one of {REFERENCE_INIT_DEVICES}, not {reference_init!r}"
        )
    assets_root = Path(assets_root)
    native = Path(native_checkpoint) if native_checkpoint else None
    if native is not None and not native.exists():
        raise AlignmentError(
            f"converted native checkpoint not found at {native}; convert it first with "
            f"convert_checkpoint(..., model={variant.family!r}, dtype='float32') inside "
            "the memory guard, or build the spec without --native-checkpoint and add it "
            "before the native run"
        )

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
        "schema": "funasr-mlx-alignment/3",
        "variant": {
            "key": variant.key,
            "family": variant.family,
            "checkpoint": str(assets_root / variant.checkpoint),
            "config": str(assets_root / variant.config),
            # The converted artefact the NATIVE side loads through the shipping
            # pipeline loader (``pipeline.load_model``). It is a different file
            # from ``checkpoint`` and is never inferred from a naming convention.
            "native_checkpoint": str(native) if native is not None else None,
            "conversion": {
                # Named without its package on purpose: the reference half of this
                # module must not mention the native port at all, and the native
                # runner already spells out the full call.
                "tool": "convert_checkpoint",
                "model": variant.family,
                # float32 is required: the reference is float32, and a float16
                # native run would compare different numbers and say so only as a
                # diffuse per-element difference.
                "dtype": "float32",
                "source": str(assets_root / variant.checkpoint),
                "destination": str(native) if native is not None else None,
                "requires_guarded_child": True,
            },
            # CMVN and tokenizer filenames are declared only in configuration.json;
            # config.yaml ships cmvn_file: null, so they are never guessed here.
            "cmvn": None
            if variant.family == "whisper"
            else resolve_asset_paths(assets_root / variant.config).get("cmvn_file"),
            "assets": {}
            if variant.family == "whisper"
            else resolve_asset_paths(assets_root / variant.config),
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
            "max_seconds": variant.max_seconds,
            "fixed_order": [item["key"] for item in inputs],
        },
        "inputs": inputs,
        "output_root": str(output_root),
        "dtype": "float32",
        "reference": {
            # "meta" + assign=True is the memory-bounded path; "cpu" is the
            # opt-in fallback and is recorded so a run can never be mistaken for
            # one that allocated nothing.
            "init": reference_init,
        },
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
    model_shapes: dict[str, tuple[int, ...]] | None = None,
    checkpoint_shapes: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Compare two key sets, reporting every difference in both directions.

    ``model_shapes``/``checkpoint_shapes`` add the second half of "strict": a key
    that exists on both sides with the wrong shape is a failure that a name-only
    check reads as a success. ``load_state_dict`` validates shapes internally as
    well, but stating the check here means the adapter both owns it and *records*
    the difference, instead of depending on a library internal it did not write.
    """
    expected = sorted(str(key) for key in model_keys)
    provided = sorted(str(key) for key in checkpoint_keys)
    expected_set, provided_set = set(expected), set(provided)
    missing = sorted(expected_set - provided_set)
    unexpected = sorted(provided_set - expected_set)
    shape_mismatch: list[dict[str, Any]] = []
    if model_shapes is not None and checkpoint_shapes is not None:
        shared = sorted(expected_set & provided_set)
        shape_mismatch = [
            {
                "name": name,
                "model": [int(size) for size in model_shapes[name]],
                "checkpoint": [int(size) for size in checkpoint_shapes[name]],
            }
            for name in shared
            if tuple(model_shapes[name]) != tuple(checkpoint_shapes[name])
        ]
    exact = not missing and not unexpected and not shape_mismatch
    return {
        "label": label,
        "model_key_count": len(expected),
        "checkpoint_key_count": len(provided),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "exact_match": exact,
        "verdict": "ok" if exact else "mismatch",
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
                    "detail": (
                        f"reference={reference_meta.get(field_name)!r} "
                        f"native={native_meta.get(field_name)!r}"
                    ),
                }
            )

    # How each side was loaded is part of the evidence: two runs that agree
    # numerically but were loaded differently are still not comparable, and a
    # float16 native checkpoint must not be able to pass as a float32 run. A
    # schema-3 run REQUIRES both records; an older or hand-made file is left
    # alone rather than being told it is wrong.
    for side, metadata in (("reference", reference_meta), ("native", native_meta)):
        declared = str(metadata.get("schema", ""))
        inventory = metadata.get("inventory")
        if inventory is None:
            if declared.endswith("/3"):
                findings.append(
                    {
                        "kind": "inventory_missing",
                        "detail": f"{side} declares {declared} but records no key inventory",
                    }
                )
        elif inventory.get("verdict") != "ok":
            findings.append(
                {
                    "kind": "inventory_not_exact",
                    "detail": (
                        f"{side} inventory={inventory.get('verdict')} "
                        f"missing={len(inventory.get('missing', []))} "
                        f"unexpected={len(inventory.get('unexpected', []))} "
                        f"shape_mismatch={len(inventory.get('shape_mismatch', []))}"
                    ),
                }
            )
        if declared.endswith("/3") and not metadata.get("loading"):
            findings.append(
                {
                    "kind": "loading_disposition_missing",
                    "detail": f"{side} records how it was loaded nowhere in its metadata",
                }
            )
    reference_loading = reference_meta.get("loading") or {}
    native_loading = native_meta.get("loading") or {}
    if reference_loading and native_loading:
        for field in ("dtype", "keys_checked"):
            if reference_loading.get(field) != native_loading.get(field):
                findings.append(
                    {
                        "kind": "loading_disposition_differ",
                        "detail": (
                            f"{field}: reference={reference_loading.get(field)!r} "
                            f"native={native_loading.get(field)!r}"
                        ),
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
        for field_name, kind in (
            ("audio_samples", "audio_length_disagreement"),
            ("fbank_frames", "fbank_frame_disagreement"),
        ):
            if reference_sample.get(field_name) != native_sample.get(field_name):
                findings.append(
                    {
                        "kind": kind,
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
        "schema": "funasr-mlx-alignment-diff/3",
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

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        # Paraformer's official checkpoint is legacy (non-zip) serialization;
        # mmap is unavailable there and the payload must load into RSS.
        if "mmap" not in str(error):
            raise
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    for container in ("model", "state_dict", "model_state_dict"):
        if isinstance(payload, dict) and container in payload:
            payload = payload[container]
            break
    if not isinstance(payload, dict):
        raise AlignmentError(f"{path} does not contain a state_dict mapping")
    return {str(key): value for key, value in payload.items()}


# --------------------------------------------------------------------------- #
# Reference weight loading: strict, shape-checked, and allocation-free
# --------------------------------------------------------------------------- #
def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _reference_parameter_shapes(model: Any) -> dict[str, tuple[int, ...]]:
    """Every parameter's shape, read before any data is loaded.

    On the meta device these are still the architecture's real shapes, which is
    what makes an explicit shape check possible even though nothing was
    allocated. Reading them from a model whose parameters already hold weights
    would only re-derive what the checkpoint says.
    """
    return {
        name: tuple(int(size) for size in tensor.shape) for name, tensor in model.named_parameters()
    }


def _meta_leftovers(
    model: Any, torch: Any, kinds: Sequence[str] = ("parameter", "buffer", "attribute")
) -> list[dict[str, Any]]:
    """Tensors still on the meta device, with their owning class and attribute.

    Parameters are included deliberately: a parameter left on meta is not an
    error at the first matmul, it is wrong numbers.
    """
    leftovers: list[dict[str, Any]] = []
    for path, module in model.named_modules():
        owner = type(module).__name__
        candidates: list[tuple[str, str, Any]] = [
            (name, "parameter", value) for name, value in module._parameters.items()
        ]
        candidates += [(name, "buffer", value) for name, value in module._buffers.items()]
        candidates += [
            (name, "attribute", value)
            for name, value in vars(module).items()
            if isinstance(value, torch.Tensor)
        ]
        for name, kind, value in candidates:
            if kind not in kinds or not isinstance(value, torch.Tensor) or not value.is_meta:
                continue
            leftovers.append(
                {
                    "path": _join(path, name),
                    "kind": kind,
                    "class": owner,
                    "shape": [int(size) for size in value.shape],
                }
            )
    return leftovers


def _rebuild_positional_encoding_pe(module: Any, tensor: Any, torch: Any) -> Any:
    """Recreate ``PositionalEncoding.pe`` with the source's own method.

    ``pe`` is a plain attribute, so it is absent from every published
    state_dict -- but it is not stateless either: ``__init__`` builds a
    ``max_len``-row table and ``forward`` extends it. A meta tensor cannot be
    brought back by ``extend_pe`` (``Tensor.to`` out of meta raises), so the table
    is rebuilt here by the source's own code, with the source's own row count
    (recoverable from the meta tensor's shape) and dtype. Rows are computed
    independently of one another, so the result is bit-identical to the table the
    reference would have built for itself.
    """
    rows = int(tensor.shape[1])
    module.pe = None
    module.extend_pe(torch.zeros(1, rows, dtype=tensor.dtype, device="cpu"))
    return module.pe


def _rebuild_stream_pe(module: Any, tensor: Any, torch: Any) -> Any:
    """``StreamPositionalEncoding.extend_pe(length, device, dtype)`` has a
    different signature; the values it produces are the same table."""
    module.pe = None
    module.extend_pe(int(tensor.shape[1]), torch.device("cpu"), torch.float32)
    return module.pe


def _rebuild_stream_tmp(module: Any, tensor: Any, torch: Any) -> Any:
    """``self.tmp = torch.tensor(0.0).expand(1, max_len)`` -- zeros are the same
    values, and the source only ever reads its size."""
    return torch.zeros(1, int(tensor.shape[1]), dtype=torch.float32, device="cpu")


def _meta_rebuild_rules(torch: Any) -> list[tuple[tuple[type, str], str, Any]]:
    """(class, attribute) -> (human source call, rebuilder), most specific first.

    The classes come from the imported FunASR module rather than from a name
    comparison, so a subclass inherits the right rule and a same-named attribute
    on an unrelated class does not.
    """
    from funasr.models.transformer import embedding as funasr_embedding

    return [
        (
            (funasr_embedding.StreamPositionalEncoding, "tmp"),
            "zeros(1, max_len)  # source: torch.tensor(0.0).expand(1, max_len)",
            _rebuild_stream_tmp,
        ),
        (
            (funasr_embedding.StreamPositionalEncoding, "pe"),
            "extend_pe(max_len, cpu, float32)  # source's own method",
            _rebuild_stream_pe,
        ),
        (
            (funasr_embedding.PositionalEncoding, "pe"),
            "extend_pe(zeros(1, max_len))  # source's own method",
            _rebuild_positional_encoding_pe,
        ),
    ]


def _rebuild_meta_tensors(model: Any, torch: Any) -> list[dict[str, Any]]:
    """Rebuild every non-parameter tensor the source creates in ``__init__``.

    No rule invents numbers: each one calls the source's own construction code,
    and each records what it did. A meta tensor with no rule is a hard error,
    because leaving it in place would not fail -- it would be read as an ordinary
    tensor and produce plausible wrong activations.
    """
    rules = _meta_rebuild_rules(torch)
    records: list[dict[str, Any]] = []
    for path, module in model.named_modules():
        targets: list[tuple[str, str, Any]] = [
            (name, "buffer", value) for name, value in module._buffers.items()
        ]
        targets += [
            (name, "attribute", value)
            for name, value in vars(module).items()
            if isinstance(value, torch.Tensor)
        ]
        for name, kind, tensor in targets:
            if not tensor.is_meta:
                continue
            rule = next(
                (
                    entry
                    for entry in rules
                    if isinstance(module, entry[0][0]) and name == entry[0][1]
                ),
                None,
            )
            where = _join(path, name)
            if rule is None:
                raise AlignmentError(
                    f"{where} is on the meta device and no source-defined "
                    f"reconstruction exists for {type(module).__name__}.{name}; "
                    "refusing to run with an uninitialised tensor instead of "
                    "inventing values"
                )
            rows = int(tensor.shape[1]) if tensor.ndim > 1 else int(tensor.shape[0])
            if rows > MAX_RECONSTRUCTED_ROWS:
                raise AlignmentError(
                    f"{where} needs {rows} rows, above the bounded "
                    f"{MAX_RECONSTRUCTED_ROWS}; refusing to allocate it"
                )
            before = [int(size) for size in tensor.shape]
            rebuilt = rule[2](module, tensor, torch)
            if not isinstance(rebuilt, torch.Tensor) or rebuilt.is_meta:
                raise AlignmentError(
                    f"rebuilding {where} did not produce a real tensor, so the model "
                    "is not fully initialised"
                )
            records.append(
                {
                    "path": where,
                    "class": type(module).__name__,
                    "attribute": name,
                    "kind": kind,
                    "source_call": rule[1],
                    "shape_before": before,
                    "shape_after": [int(size) for size in rebuilt.shape],
                    "elements": int(rebuilt.numel()),
                }
            )
    return records


def _probe_meta_positional_encoding(torch: Any) -> dict[str, Any]:
    """Prove, cheaply, that meta construction and the rebuild work together.

    ``PositionalEncoding.__init__`` builds a ``max_len``-row table in place --
    ``pe[:, 0::2] = torch.sin(...)`` -- and that is the one step of these two
    architectures most likely to lack a meta kernel. Finding out halfway through
    a 50-layer model would be expensive and vague, so a single tiny instance of
    the real class is built on meta and pushed through the very same rule that
    will later rebuild the real tables.

    Nothing here is a weight: it is 5000x8 floats, built and discarded.
    """
    from funasr.models.transformer import embedding as funasr_embedding

    rules = _meta_rebuild_rules(torch)
    rule = next(
        entry
        for entry in rules
        if entry[0][0] is funasr_embedding.PositionalEncoding and entry[0][1] == "pe"
    )
    d_model = 8
    try:
        with torch.device("meta"):
            probe = funasr_embedding.PositionalEncoding(d_model, 0.0)
    except Exception as error:  # noqa: BLE001 - reported, never swallowed
        raise AlignmentError(
            "the source PositionalEncoding cannot be constructed on the meta device "
            f"({type(error).__name__}: {error}); rebuild with --reference-init cpu and "
            "accept a full random copy in RSS, or extend the model construction itself"
        ) from error
    if not isinstance(probe.pe, torch.Tensor) or not probe.pe.is_meta:
        raise AlignmentError(
            "the meta probe produced no positional table, so the reconstruction rules "
            "would have nothing to match; refusing to continue on that assumption"
        )
    rows = int(probe.pe.shape[1])
    if rows > MAX_RECONSTRUCTED_ROWS:
        raise AlignmentError(
            f"the source positional table needs {rows} rows, above the bounded "
            f"{MAX_RECONSTRUCTED_ROWS}"
        )
    rebuilt = rule[2](probe, probe.pe, torch)
    if not isinstance(rebuilt, torch.Tensor) or rebuilt.is_meta:
        raise AlignmentError(
            "the positional table could not be rebuilt on a real device from a meta "
            "tensor, so a real model would keep an uninitialised tensor"
        )
    if tuple(rebuilt.shape) != (1, rows, d_model) or not bool(torch.isfinite(rebuilt).all()):
        raise AlignmentError(
            f"the rebuilt positional table is unusable (shape={tuple(rebuilt.shape)}); "
            "refusing to load a model that would attend to uninitialised positions"
        )
    return {
        "class": funasr_embedding.PositionalEncoding.__name__,
        "rows": rows,
        "d_model_probe": d_model,
        "rebuild": rule[1],
        "shape": [int(size) for size in rebuilt.shape],
    }


def _load_reference_weights(
    model: Any, state: dict[str, Any], *, torch: Any, device: str = "meta"
) -> dict[str, Any]:
    """Bind official weights into ``model``; return how that was done.

    Names are compared in both directions and every shape is checked before a
    single value is bound, so a renamed head, a dropped head or a re-ordered
    weight is a failure rather than a silently missing parameter.
    """
    model_shapes = _reference_parameter_shapes(model)
    checkpoint_shapes = {
        str(name): tuple(int(size) for size in tensor.shape) for name, tensor in state.items()
    }
    inventory = inventory_report(
        model_shapes,
        checkpoint_shapes,
        label="reference",
        model_shapes=model_shapes,
        checkpoint_shapes=checkpoint_shapes,
    )
    if inventory["missing"] or inventory["unexpected"]:
        raise AlignmentError(
            f"reference key inventory mismatch: missing={inventory['missing']} "
            f"unexpected={inventory['unexpected']}"
        )
    if inventory["shape_mismatch"]:
        raise AlignmentError(
            f"reference shape mismatch for {len(inventory['shape_mismatch'])} keys: "
            f"{inventory['shape_mismatch'][:5]}"
        )
    inexact = sorted(
        name
        for name, tensor in state.items()
        if tensor.is_floating_point() and tensor.dtype != torch.float32
    )
    if inexact:
        raise AlignmentError(
            f"{len(inexact)} checkpoint tensors are not float32 (for example "
            f"{inexact[:5]}); convert explicitly instead of letting each backend "
            "pick a precision and comparing the difference as if it were a port bug"
        )
    if device == "meta":
        # The only way a meta parameter receives data: the values are bound in
        # place, so no tensor is allocated and nothing is copied. Torch still
        # validates shapes here; the checks above are why this does not depend on
        # that staying true.
        model.load_state_dict(state, strict=True, assign=True)
    else:
        model.load_state_dict(state, strict=True)
    model.requires_grad_(False)

    pending = _meta_leftovers(model, torch, kinds=("buffer", "attribute"))
    rebuilt = _rebuild_meta_tensors(model, torch) if device == "meta" else []
    leftovers = _meta_leftovers(model, torch)
    if leftovers:
        raise AlignmentError(
            f"{len(leftovers)} tensors are still on the meta device after loading, "
            f"so the model is not fully initialised: {leftovers[:5]}"
        )
    return {
        "device": device,
        "assignment": "assign=True" if device == "meta" else "copy_",
        "dtype": "float32",
        "keys_checked": len(model_shapes),
        "shapes_checked": len(checkpoint_shapes),
        "meta_non_parameters_before": pending,
        "rebuilt_non_parameters": rebuilt,
        "meta_after": leftovers,
        "inventory": inventory,
    }


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge, matching FunASR's ``deep_update`` semantics."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def build_model_conf(config: dict[str, Any]) -> dict[str, Any]:
    """Reproduce AutoModel's construction kwargs exactly.

    ``auto_model.py:779-781`` does::

        model_conf = {}
        deep_update(model_conf, kwargs.get("model_conf", {}))
        deep_update(model_conf, kwargs)
        model = model_class(**model_conf)

    So the WHOLE config is merged over ``model_conf``, which is how the
    top-level ``encoder:``/``decoder:``/``predictor:`` NAME strings and their
    ``*_conf`` dicts reach the constructor. Forwarding only ``model_conf`` -- or
    only the ``*_conf`` keys -- silently builds a model with no encoder.
    """
    merged: dict[str, Any] = {}
    _deep_update(merged, config.get("model_conf") or {})
    _deep_update(merged, config)
    return merged


def resolve_asset_paths(config_path: str | Path) -> dict[str, Any]:
    """Resolve asset filenames from ``configuration.json``, not config.yaml.

    The official ``config.yaml`` ships ``cmvn_file: null`` and ``bpemodel: null``;
    the real filenames live in ``configuration.json`` under ``file_path_metas``.
    Guessing ``am.mvn`` by convention happens to work for these two models and
    would break silently for any model that differs, so it is read from the
    authoritative source.
    """
    config_path = Path(config_path)
    directory = config_path.parent
    metas_path = directory / "configuration.json"
    if not metas_path.is_file():
        raise AlignmentError(
            f"{directory}/configuration.json is missing; CMVN and tokenizer "
            "filenames are only declared there, not in config.yaml"
        )
    metas = json.loads(metas_path.read_text()).get("file_path_metas", {})
    frontend = metas.get("frontend_conf") or {}
    tokenizer = metas.get("tokenizer_conf") or {}

    def _resolve(name: Any) -> str | None:
        return str(directory / name) if name else None

    return {
        "directory": str(directory),
        "init_param": _resolve(metas.get("init_param")),
        "cmvn_file": _resolve(frontend.get("cmvn_file")),
        "bpemodel": _resolve(tokenizer.get("bpemodel")),
        "token_list": _resolve(tokenizer.get("token_list")),
        "seg_dict_file": _resolve(tokenizer.get("seg_dict_file")),
    }


# --------------------------------------------------------------------------- #
# Reference backend (Torch). Imports no MLX, ever.
# --------------------------------------------------------------------------- #
def run_reference(spec: dict[str, Any], output_root: str | Path | None = None) -> dict[str, Any]:
    """Run the PyTorch source: real config, real constructors, real frontend."""
    import torch

    variant = spec["variant"]
    reference_init = (spec.get("reference") or {}).get("init", DEFAULT_REFERENCE_INIT)
    if reference_init not in REFERENCE_INIT_DEVICES:
        raise AlignmentError(
            f"unknown reference init device {reference_init!r}; expected one of "
            f"{REFERENCE_INIT_DEVICES}"
        )
    output_root = Path(output_root or spec["output_root"]) / "reference"
    output_root.mkdir(parents=True, exist_ok=True)

    if variant["family"] == "whisper":
        if spec.get("inputs"):
            raise AlignmentError(
                "whisper-tiny has a loadable canonical reference and a checked key "
                "inventory, but this adapter has no per-sample stage pipeline for it; "
                "the families that can be aligned end to end are paraformer-zh and "
                "sensevoice-small. Refusing rather than emitting an empty run that "
                "reads like a pass."
            )
        # Canonical Whisper is a separate AST-loaded definition set with its own
        # state_dict; it is loaded on CPU (no meta path) and that is recorded.
        model, extras = _build_canonical_whisper(variant)
        frontend, tokenizer = None, None
        loading = _load_reference_weights(
            model,
            _load_checkpoint_state(Path(variant["checkpoint"])),
            torch=torch,
            device="cpu",
        )
        loading["mode"] = "canonical AST-loaded Whisper state_dict"
    else:
        assets = resolve_asset_paths(variant["config"])
        # configuration.json is authoritative for the weights filename too.
        variant = {**variant, "checkpoint": assets["init_param"] or variant["checkpoint"]}
        model, extras, frontend, tokenizer = _build_funasr_reference(
            variant, assets, torch, reference_init
        )
        loading = _load_reference_weights(
            model,
            _load_checkpoint_state(Path(variant["checkpoint"])),
            torch=torch,
            device=reference_init,
        )
        loading["mode"] = "funasr real constructor"

    inventory = loading["inventory"]
    model.eval()
    extras["loading"] = {k: v for k, v in loading.items() if k != "inventory"}

    samples = [
        _run_reference_sample(model, frontend, variant, item, output_root, torch, tokenizer)
        for item in spec["inputs"]
    ]

    metadata = {
        "schema": "funasr-mlx-alignment-run/3",
        "backend": "reference",
        "variant": variant,
        "dtype": "float32",
        "frontend_config": variant["frontend"],
        "cmvn": variant["cmvn"],
        "inventory": inventory,
        "loading": loading,
        "extras": extras,
        "samples": samples,
        "torch_version": torch.__version__,
        "device": "cpu",
    }
    _write_metadata(output_root / "metadata.json", metadata)
    return metadata


def _build_funasr_reference(
    variant: dict[str, Any], assets: dict[str, Any], torch: Any, init_device: str = "meta"
) -> tuple[Any, Any, Any, Any]:
    """Construct the real FunASR model exactly as ``AutoModel`` does.

    Order matters and follows ``auto_model.py:755-781``: the tokenizer supplies
    ``vocab_size``, the built frontend supplies ``input_size`` via
    ``output_size()``, and only then is the model constructed from
    ``build_model_conf``. Both of those values are DERIVED, not configured --
    the official config.yaml contains neither, so omitting either silently builds
    a model with the wrong widths (and the strict shape check then rejects the
    checkpoint rather than loading it).

    ``init_device="meta"`` wraps only the model construction: no weight storage
    is allocated, and the official values are bound afterwards by
    :func:`_load_reference_weights`. The frontend is deliberately built outside
    that context -- it is real computation, not parameters.
    """
    import importlib

    import yaml

    for module_path in variant["registry_imports"]:
        importlib.import_module(module_path)

    config = yaml.safe_load(Path(variant["config"]).read_text())
    frontend = _build_reference_frontend(config, assets)
    vocab_size, tokenizer = _reference_vocab_size(config, assets)

    model_conf = build_model_conf(config)
    model_conf["input_size"] = int(frontend.output_size())
    model_conf["vocab_size"] = int(vocab_size["size"])

    module = importlib.import_module(variant["reference_import"])
    model_class = getattr(module, variant["reference_class"])
    probe = _probe_meta_positional_encoding(torch) if init_device == "meta" else None
    if init_device == "meta":
        try:
            with torch.device("meta"):
                model = model_class(**model_conf)
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            # No silent fallback: a source constructor that cannot run without
            # real storage is a fact the operator needs, and 'cpu' is an explicit
            # opt-in with a known memory cost rather than a hidden default.
            raise AlignmentError(
                f"{variant['reference_class']} could not be constructed on the meta "
                f"device ({type(error).__name__}: {error}). Either that source "
                "constructor genuinely needs real storage -- in which case rebuild "
                "with --reference-init cpu and accept a full random copy in RSS -- or "
                "its non-parameter construction needs a bounded source-defined "
                "reconstruction rule, which belongs in _meta_rebuild_rules. Refusing "
                "to guess which."
            ) from error
    else:
        model = model_class(**model_conf)

    extras: dict[str, Any] = {
        "model_conf_keys": sorted(model_conf),
        "input_size": int(frontend.output_size()),
        "frontend_output_size_formula": "n_mels * lfr_m",
        "vocab_size": vocab_size,
        "cmvn_file": assets.get("cmvn_file"),
        "encoder_name": config.get("encoder"),
        "decoder_name": config.get("decoder"),
        "predictor_name": config.get("predictor"),
        "init_device": init_device,
        "meta_positional_probe": probe,
    }
    # Conditioning id tables must be recorded, not assumed: they decide what the
    # model actually consumed and are impossible to infer from a tensor shape.
    for attribute in ("lid_dict", "textnorm_dict", "lid_int_dict"):
        table = getattr(model, attribute, None)
        if isinstance(table, dict):
            extras[attribute] = {str(k): int(v) for k, v in table.items()}
    extras["ctc_weight"] = float(getattr(model, "ctc_weight", 0.0) or 0.0)
    extras["has_ctc_head"] = getattr(model, "ctc", None) is not None
    return model, extras, frontend, tokenizer


def _build_reference_frontend(config: dict[str, Any], assets: dict[str, Any]) -> Any:
    """FunASR's own ``WavFrontend``, from config.yaml's own frontend_conf.

    ``frontend_conf`` is forwarded verbatim -- including ``window: hamming``,
    which differs from the library default and would otherwise be guessed wrong.
    The CMVN comes from configuration.json, because config.yaml ships it null.
    """
    frontend_conf = dict(config.get("frontend_conf") or {})
    frontend_conf.pop("cmvn_file", None)
    # The official yaml omits dither, and FunASR's WavFrontend then defaults to
    # 1.0 -- random noise on every run. The native side uses the spec's dither
    # (0.0), so pin the reference to the same value or the comparison measures
    # dither noise instead of the port.
    frontend_conf.setdefault("dither", 0.0)
    frontend_class = _resolve_wav_frontend()
    return frontend_class(cmvn_file=assets.get("cmvn_file"), **frontend_conf)


def _reference_vocab_size(
    config: dict[str, Any], assets: dict[str, Any]
) -> tuple[dict[str, Any], Any]:
    """Derive ``vocab_size`` from the official tokenizer, and record its order.

    Token ids are only comparable between backends once the vocabulary ORDER is
    pinned, so the id list is fingerprinted here rather than assumed to match.
    The tokenizer instance is returned as well: the reference can decode the ids
    it produced, which is provenance the native side cannot produce (it has no
    tokenizer) and therefore is recorded rather than compared.
    """
    import importlib
    import json as _json

    tokenizer_name = config.get("tokenizer")
    tokenizer_conf = dict(config.get("tokenizer_conf") or {})
    token_list_path = assets.get("token_list")
    bpemodel = assets.get("bpemodel")

    if tokenizer_name == "CharTokenizer" and token_list_path:
        tokens = _json.loads(Path(token_list_path).read_text())
        return (
            {
                "class": tokenizer_name,
                "size": len(tokens),
                "token_list": token_list_path,
                "head": [str(t) for t in tokens[:8]],
                "first_ids": list(range(min(8, len(tokens)))),
            },
            None,
        )

    # configuration.json's tokenizer_conf is authoritative where config.yaml
    # ships nulls (bpemodel: None would reach sentencepiece as a literal None).
    if bpemodel:
        tokenizer_conf = {k: v for k, v in tokenizer_conf.items() if v is not None}
        tokenizer_conf.setdefault("bpemodel", bpemodel)

    for module_path, attribute in (
        ("funasr.tokenizer.sentencepiece_tokenizer", "SentencepiecesTokenizer"),
        ("funasr.tokenizer.char_tokenizer", "CharTokenizer"),
    ):
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        if not hasattr(module, attribute):
            continue
        if attribute != tokenizer_name:
            continue
        if bpemodel:
            tokenizer_conf.setdefault("bpemodel", bpemodel)
        tokenizer = getattr(module, attribute)(**tokenizer_conf)
        size = getattr(tokenizer, "vocab_size", None)
        if size is None:
            # SentencepiecesTokenizer spells it get_vocab_size() over the
            # underlying piece table.
            getter = getattr(tokenizer, "get_vocab_size", None)
            size = getter() if callable(getter) else None
        if size is None:
            raise AlignmentError(
                f"{tokenizer_name} exposes no vocab_size; refusing to guess the "
                "vocabulary size the checkpoint was trained with"
            )
        return (
            {
                "class": attribute,
                "size": int(size),
                "bpemodel": bpemodel,
                "module": module_path,
            },
            tokenizer,
        )

    raise AlignmentError(
        f"cannot derive vocab_size for tokenizer {tokenizer_name!r}; the constructor "
        "requires it and config.yaml does not contain it"
    )


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


def _reject_overlength(key: str, samples: int, variant: dict[str, Any]) -> None:
    """Refuse audio longer than the variant bound, measured rather than declared.

    ``build_alignment_spec`` bounds what the manifest *claims*; this re-checks
    the array that was actually decoded. A manifest that lies must fail here
    rather than be truncated by one frontend and not the other, because only one
    side truncating invalidates the whole comparison while still producing
    numbers that look plausible.
    """
    seconds = samples / float(SAMPLE_RATE)
    if seconds > float(variant["max_seconds"]):
        raise AlignmentError(
            f"{key}: audio is {seconds:.2f}s, above the {variant['max_seconds']:.2f}s "
            f"bound of {variant['key']}; refused rather than truncated"
        )


def _paraformer_greedy_ids(model: Any, decoder_out: Any, token_length: Any) -> list[int]:
    """``Paraformer.inference``'s greedy token path, without beam search.

    ``model.py:632-664``: the argmax is taken over the first
    ``pre_token_length[i]`` columns of the log-softmax, then ``sos``/``eos``/
    ``blank_id`` are dropped. The same three ids the native port drops, because
    otherwise the two sides would report different id lists for identical
    posteriors.
    """
    dropped = {int(model.eos), int(model.sos), int(model.blank_id)}
    limit = min(int(token_length[0]), int(decoder_out.shape[1]))
    row = [int(v) for v in decoder_out[0, :limit].argmax(dim=-1).tolist()]
    return [value for value in row if value not in dropped]


def _ctc_greedy_ids(logits_row: Any, length: int, blank_id: int) -> list[int]:
    """``SenseVoiceSmall.inference``'s CTC greedy path.

    ``model.py:1013-1021``: argmax over the valid frames, then
    ``unique_consecutive``, then blanks removed. Order matters -- collapsing
    duplicates BEFORE removing blanks turns ``[a, blank, a]`` into ``[a, a]``.
    """
    ids = [int(v) for v in logits_row[:length].argmax(dim=-1).tolist()]
    collapsed: list[int] = []
    previous: int | None = None
    for value in ids:
        if value != previous:
            collapsed.append(value)
            previous = value
    return [value for value in collapsed if value != blank_id]


def _reference_text(tokenizer: Any, token_ids: Sequence[int], family: str) -> str | None:
    """Decode the ids the reference produced, when its own tokenizer can.

    Recorded as provenance only. The native port ships no tokenizer, so this is
    never compared: inventing a text decoder to make the two sides look
    comparable would defeat the purpose of comparing ids.
    """
    if tokenizer is None:
        return None
    try:
        if hasattr(tokenizer, "decode"):
            return str(tokenizer.decode(list(token_ids)))
        if family == "paraformer" and hasattr(tokenizer, "ids2tokens"):
            return " ".join(str(token) for token in tokenizer.ids2tokens(list(token_ids)))
    except Exception as error:  # noqa: BLE001 - provenance must never fail a run
        return f"<decode failed: {type(error).__name__}>"
    return None


def _run_reference_sample(
    model: Any,
    frontend: Any,
    variant: dict[str, Any],
    item: dict[str, Any],
    output_root: Path,
    torch: Any,
    tokenizer: Any = None,
) -> dict[str, Any]:
    import numpy as np

    audio = _load_audio(item["audio"])
    _reject_overlength(item["key"], int(audio.shape[0]), variant)
    with torch.no_grad():
        # WavFrontend.forward(input(B,T), input_lengths) returns (feats,
        # feats_len) and already applies LFR + CMVN internally.
        fbank, _fbank_len = frontend(
            torch.from_numpy(audio)[None, :],
            torch.tensor([audio.shape[0]], dtype=torch.int32),
        )
        fbank = np.asarray(fbank, dtype=np.float32)
        # frontend returns (B=1, T, D); keep batch dim for the model but save
        # the (T, D) row so the comparison geometry matches the native side.
        speech = torch.from_numpy(fbank)
        lengths = torch.tensor([fbank.shape[1]], dtype=torch.int32)

        finals: dict[str, Any] = {}
        token_ids: list[int] | None = None
        conditioning: dict[str, Any] = {}

        if variant["family"] == "paraformer":
            encoder_out, encoder_out_lens = model.encode(speech, lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            sematic_embeds, pre_token_length, _, _ = model.calc_predictor(
                encoder_out, encoder_out_lens
            )
            # ``inference`` rounds and casts the token length BEFORE the decoder
            # and before slicing the argmax (model.py:617-620); the native port
            # rounds the same value, so an unrounded one would desynchronise the
            # two sides by a whole column.
            token_length = pre_token_length.round().long()
            decoder_out, _ = model.cal_decoder_with_predictor(
                encoder_out, encoder_out_lens, sematic_embeds, token_length
            )
            finals = {
                "encoder": encoder_out,
                "pre_decoder": sematic_embeds,
                "final_logits": decoder_out,
            }
            token_ids = _paraformer_greedy_ids(model, decoder_out, token_length)
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
            token_ids = _ctc_greedy_ids(
                ctc_logits[0], int(encoder_out_lens[0]), int(model.blank_id)
            )
        else:  # pragma: no cover - dispatched earlier
            raise AlignmentError(f"reference path for {variant['family']} is not wired")

    fbank_row = fbank[0] if fbank.ndim == 3 else fbank
    stages = [
        _stage_record("frontend", fbank_row, path=output_root / f"{item['key']}_frontend.npz")
    ]
    saved: dict[str, Any] = {"frontend": fbank_row}
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
        # Measured, not declared: the manifest's own claim is compared against
        # these, and a frontend that silently truncates shows up as a frame
        # disagreement rather than as a mysterious numeric difference.
        "audio_samples": int(audio.shape[0]),
        "audio_seconds": round(int(audio.shape[0]) / float(SAMPLE_RATE), 6),
        "fbank_frames": int(fbank.shape[0] if fbank.ndim == 2 else fbank.shape[1]),
        "stages": stages,
        "token_ids": token_ids,
        "conditioning_ids": conditioning or None,
        "text": _reference_text(tokenizer, token_ids or [], variant["family"]),
    }


def _apply_sensevoice_conditioning(
    model: Any, speech: Any, lengths: Any, torch: Any
) -> tuple[Any, Any, dict[str, Any]]:
    """Prepend SenseVoice's conditioning tokens exactly as ``inference`` does.

    The conditioning is a fixed id sequence built from ``lid_dict`` and
    ``textnorm_dict`` plus the hard-coded event/emotion pair ``[1, 2]``. It is
    1 (textnorm) + 3 (language + event + emo) = 4 frames of prefix, and the
    lengths must grow to match or the encoder masks real audio away. The
    resulting id ORDER is
    ``[language, event, emo, textnorm, audio...]``, which is the same sequence
    the native port builds as one embedded row; the ids are recorded so a
    disagreement is visible rather than buried in a tensor shape.
    """
    language = SENSEVOICE_LANGUAGE
    textnorm = "withitn" if SENSEVOICE_USE_ITN else "woitn"
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
    # CTC.log_softmax() itself applies ctc_lo then log-softmax
    # (ctc/ctc.py:201). Pre-projecting here would double-apply it.
    return head.log_softmax(encoder_out)


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
    """Run the native MLX port from the CONVERTED artefact, through the loader.

    Deliberately not through Torch. The native process reads only the
    safetensors the converter produced, via ``funasr_mlx.pipeline.load_model``,
    so the path being validated is the one the product ships -- its
    prevalidation, its bounded shard loop, its strict key and shape checks --
    rather than a second reader inside this tool that could quietly disagree with
    it.
    Must run inside the memory guard; the loader itself also enforces that, and
    this wrapper says so in terms an operator can act on.
    """
    import mlx.core as mx

    from funasr_mlx import pipeline
    from funasr_mlx.checkpoints import checkpoint_files, safetensor_header
    from funasr_mlx.frontend import FbankConfig, WavFrontend
    from funasr_mlx.memory import guarded_child_active
    from funasr_mlx.models.sanm import flatten_state_dict

    if not guarded_child_active():
        raise AlignmentError(
            "the native backend must run inside the memory guard: "
            "python -m funasr_mlx.runtime -- <repo>/tools/alignment_asr.py run "
            "--backend native --spec <spec.json> --output-root <dir>"
        )

    variant = spec["variant"]
    native_checkpoint = variant.get("native_checkpoint")
    if not native_checkpoint:
        raise AlignmentError(
            f"{variant['key']} has no converted native checkpoint; build the spec with "
            "--native-checkpoint <safetensors path|dir>, produced by "
            f"funasr_mlx.checkpoints.convert_checkpoint(..., model={variant['family']!r}, "
            "dtype='float32') in a guarded child"
        )
    if variant["family"] == "whisper" and spec.get("inputs"):
        raise AlignmentError(
            "whisper-tiny has no native per-sample stage pipeline in this adapter; "
            "the families that can be aligned end to end are paraformer-zh and "
            "sensevoice-small"
        )
    output_root = Path(output_root or spec["output_root"]) / "native"
    output_root.mkdir(parents=True, exist_ok=True)

    # The native side must be sized by the CONVERTED artifact's config (which
    # carries the tokenizer-derived vocab_size), not the original yaml whose
    # nulls the native defaults would silently mis-size (55465 vs 25055).
    _native_config_path = Path(native_checkpoint) / "config.json"
    model, _ = pipeline.load_model(
        native_checkpoint, family=variant["family"], config=_native_config_path
    )

    parameters = flatten_state_dict(model)
    # Read the headers once, through the loader's own reader, so the reported
    # inventory describes exactly the files that were loaded.
    headers: dict[str, Any] = {}
    for path in checkpoint_files(native_checkpoint):
        header = safetensor_header(path)
        header.pop("__metadata__", None)
        headers.update(header)
    inventory = inventory_report(
        parameters,
        headers,
        label="native",
        model_shapes={name: tuple(int(s) for s in a.shape) for name, a in parameters.items()},
        checkpoint_shapes={
            name: tuple(int(size) for size in entry["shape"]) for name, entry in headers.items()
        },
    )
    if inventory["verdict"] != "ok":
        raise AlignmentError(
            f"native key inventory mismatch: missing={inventory['missing']} "
            f"unexpected={inventory['unexpected']} shapes={len(inventory['shape_mismatch'])}"
        )
    inexact = sorted(name for name, array in parameters.items() if array.dtype != mx.float32)
    if inexact:
        raise AlignmentError(
            f"{len(inexact)} native parameters are not float32 (for example {inexact[:5]}); "
            "reconvert with dtype='float32' rather than comparing a float16 run against a "
            "float32 reference and reading the difference as a port defect"
        )
    mx.eval(parameters)

    frontend = WavFrontend(_native_frontend_config(variant, FbankConfig), cmvn=variant["cmvn"])
    samples = [
        _run_native_sample(model, frontend, variant, item, output_root, mx)
        for item in spec["inputs"]
    ]

    metadata = {
        "schema": "funasr-mlx-alignment-run/3",
        "backend": "native",
        "variant": variant,
        "dtype": "float32",
        "frontend_config": variant["frontend"],
        "cmvn": variant["cmvn"],
        "inventory": inventory,
        "loading": {
            "loader": "funasr_mlx.pipeline.load_model",
            "device": str(mx.default_device()),
            "dtype": "float32",
            "keys_checked": inventory["model_key_count"],
            "source_checkpoint": str(variant["checkpoint"]),
            "native_checkpoint": str(native_checkpoint),
        },
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


def _native_conditioning_ids(language: str, use_itn: bool, text_norm: str | None) -> dict[str, Any]:
    """The ids the native prefix actually embeds, read from the port's tables.

    Recorded rather than hard-coded so that a change to either vocabulary table
    shows up as a conditioning disagreement in the diff instead of hiding inside
    a prefix tensor nobody compares.
    """
    from funasr_mlx.models.sensevoice import (
        EMO_ID,
        EVENT_ID,
        LID_DICT,
        NUM_PREFIX_FRAMES,
        TEXTNORM_DICT,
    )

    resolved = text_norm or ("withitn" if use_itn else "woitn")
    return {
        "language": language,
        "language_id": int(LID_DICT.get(language, 0)),
        "textnorm": resolved,
        "textnorm_id": int(TEXTNORM_DICT.get(resolved, TEXTNORM_DICT["woitn"])),
        "event_emo_ids": [int(EVENT_ID), int(EMO_ID)],
        "prefix_frames": int(NUM_PREFIX_FRAMES),
    }


def _native_paraformer_token_ids(model: Any, outputs: Any) -> list[int]:
    """``Paraformer.greedy_decode``'s token selection, without a second forward.

    ``greedy_decode`` is the public entry point, but it recomputes the whole
    encoder; the alignment run already holds exactly the activations it would
    recompute, so the same helper it calls is used directly instead of paying
    for a duplicate encoder pass over a full model. A rename surfaces loudly
    here rather than silently changing the reported tokens.
    """
    helper = getattr(model, "_greedy_from_outputs", None)
    if helper is None:
        raise AlignmentError(
            "Paraformer no longer exposes _greedy_from_outputs; refusing to hand-roll "
            "token selection, because a second decode path is a second thing to be wrong"
        )
    return helper(outputs)[0]


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
    _reject_overlength(item["key"], int(audio.shape[0]), variant)
    # The native frontend takes rank-1 mono and returns (T, D) directly.
    fbank = np.asarray(frontend(audio), dtype=np.float32)
    speech = mx.array(fbank)[None, :, :]
    lengths = mx.array([fbank.shape[0]], dtype=mx.int32)

    fbank_row = fbank[0] if fbank.ndim == 3 else fbank
    stages = [
        _stage_record("frontend", fbank_row, path=output_root / f"{item['key']}_frontend.npz")
    ]
    saved: dict[str, Any] = {"frontend": fbank_row}
    finals: dict[str, Any] = {}
    token_ids: list[int] | None = None
    conditioning: dict[str, Any] | None = None

    if variant["family"] == "paraformer":
        outputs = model.forward(speech, lengths)
        if outputs.decoder_out is None:
            raise AlignmentError(
                "the native paraformer produced no decoder output; the published "
                "checkpoint has ctc_weight=0 and a decoder, so this means the config "
                "was not applied -- refusing to substitute the CIF embeddings for logits"
            )
        finals = {
            "encoder": outputs.encoder_out,
            # The CIF acoustic embedding IS the decoder input, so it is exported
            # under this name rather than being regenerated by a second call.
            "pre_decoder": outputs.acoustic_embeds,
            "final_logits": outputs.decoder_out,
        }
        token_ids = _native_paraformer_token_ids(model, outputs)
    elif variant["family"] == "sensevoice":
        from funasr_mlx.models.sensevoice import ctc_greedy_decode

        # One encoder pass: `forward_logits` is `encode` + this same CTC head, so
        # calling both would run a 50-layer encoder twice for no new information.
        encoder_out, encoder_out_lens = model.encode(
            speech,
            lengths,
            language=SENSEVOICE_LANGUAGE,
            use_itn=SENSEVOICE_USE_ITN,
            text_norm=None,
        )
        logits = model.ctc.log_softmax(encoder_out)
        finals = {"encoder": encoder_out, "pre_decoder": encoder_out, "final_logits": logits}
        token_ids = ctc_greedy_decode(logits, encoder_out_lens, model.blank_id)[0]
        conditioning = _native_conditioning_ids(SENSEVOICE_LANGUAGE, SENSEVOICE_USE_ITN, None)
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
        "audio_samples": int(audio.shape[0]),
        "audio_seconds": round(int(audio.shape[0]) / float(SAMPLE_RATE), 6),
        "fbank_frames": int(fbank.shape[0] if fbank.ndim == 2 else fbank.shape[1]),
        "stages": stages,
        "token_ids": token_ids,
        "conditioning_ids": conditioning,
        # The native port ships no tokenizer; the reference's text is provenance
        # and is never compared, so the field is explicitly absent here.
        "text": None,
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
    build_parser.add_argument(
        "--native-checkpoint",
        default=None,
        help="converted safetensors file or directory the native run will load",
    )
    build_parser.add_argument(
        "--reference-init",
        default=DEFAULT_REFERENCE_INIT,
        choices=REFERENCE_INIT_DEVICES,
        help="'meta' allocates no reference weights; 'cpu' opts into a full random copy",
    )
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
            print(
                f"          native artefact: convert it with "
                f"funasr_mlx.checkpoints.convert_checkpoint(source=<assets>/"
                f"{variant.checkpoint}, destination=<dir>, model={variant.family!r}, "
                "dtype='float32') inside a guarded child, then pass the result to "
                "build --native-checkpoint"
            )
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
            native_checkpoint=args.native_checkpoint,
            reference_init=args.reference_init,
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
