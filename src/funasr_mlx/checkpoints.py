"""Offline checkpoint inspection and bounded, sharded weight conversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import struct
import zipfile
from typing import Mapping

import numpy as np

SUPPORTED_MODELS = (
    "sensevoice",
    "paraformer",
    "ct-punc",
    "fsmn-vad",
    "campplus",
    "emotion2vec",
    "whisper",
    # LLM-based ASR families (safetensors sources; see SOURCE_RULES).
    "fun-asr-nano",
    "qwen3-asr",
    "glm-asr-nano",
    "moss-transcribe-diarize",
)

#: Output precisions ``convert_checkpoint`` may emit. ``bfloat16`` is only
#: reachable through the streaming writer (NumPy has no bfloat16), so it is
#: admitted for families whose canonical dtype is bf16.
SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")


@dataclass(frozen=True)
class SourceRule:
    """How one LLM family's source keys map onto the native parameter names.

    ``prefixes`` maps a source key prefix to the native prefix, longest first.
    ``renames`` are literal substring rewrites applied afterwards (used for the
    Qwen3-ASR ``conv2d`` -> ``conv`` submodule move and for MOSS's
    ``language_model`` -> ``model`` decoder nesting). ``drop`` lists source keys
    that the native model does not carry, because the port *derives* them (the
    Qwen3-ASR sinusoidal position table). A rule is strict: a key that matches
    no prefix and is not in ``drop`` raises rather than being passed through, so
    a renamed upstream checkpoint cannot silently produce a wrong key set.
    """

    prefixes: tuple[tuple[str, str], ...]
    renames: tuple[tuple[str, str], ...] = ()
    drop: tuple[str, ...] = ()

    def apply(self, name: str) -> str | None:
        if name in self.drop:
            return None
        for source, target in self.prefixes:
            if name.startswith(source):
                short = target + name[len(source) :]
                break
        else:
            raise ValueError(f"Checkpoint key outside the declared source layout: {name}")
        for old, new in self.renames:
            if old in short:
                short = short.replace(old, new)
        return short


#: The five LLM-ASR families. Fun-ASR-Nano's ``llm.*`` subtree is an
#: unmodified ``Qwen3ForCausalLM`` state dict (PROGRESS.md), so it is bound
#: under the model's ``llm.`` attribute with the prefix stripped. That family's
#: source is a torch ZIP and keeps the identity rule; the rest are safetensors.
#:
#: ``llm.lm_head.weight`` is absent from the native fun-asr-nano checkpoint
#: because the Qwen3 backbone is built with ``tie_word_embeddings=True`` and the
#: port reaches the head through ``embed_tokens.as_linear``; the source ships a
#: separate (numerically identical) copy that the strict loader would reject.
SOURCE_RULES = {
    "fun-asr-nano": SourceRule(
        prefixes=(
            ("audio_encoder.", "audio_encoder."),
            ("audio_adaptor.", "audio_adaptor."),
            ("llm.", "llm."),
        ),
        drop=("llm.lm_head.weight",),
    ),
    "qwen3-asr": SourceRule(
        # The audio tower is a top-level attribute of the native model, but the
        # decoder stays under ``thinker.`` (``Qwen3ASR.thinker`` wraps
        # ``Qwen3ForCausalLM``, whose transformer is its own ``model``).
        prefixes=(("thinker.audio_tower.", "audio_tower."), ("thinker.", "thinker.")),
        renames=tuple(
            (f"audio_tower.conv2d{index}.{suffix}", f"audio_tower.conv2d{index}.conv.{suffix}")
            for index in (1, 2, 3)
            for suffix in ("weight", "bias")
        ),
        # ``lm_head`` is dropped because the decoder is tied
        # (``tie_word_embeddings=True``) and the port reaches the head through
        # ``embed_tokens.as_linear``. The audio tower's sinusoidal table is not
        # in the checkpoint at all: the port rebuilds it from the config, so it
        # is declared in :data:`DERIVED_NATIVE_PARAMETERS` instead.
        drop=("thinker.lm_head.weight",),
    ),
    "glm-asr-nano": SourceRule(
        prefixes=(
            ("audio_tower.", "audio_tower."),
            # The port folds the projector into the encoder, so its two linears
            # sit under ``audio_tower.`` rather than beside it.
            ("multi_modal_projector.linear_1", "audio_tower.linear_1"),
            ("multi_modal_projector.linear_2", "audio_tower.linear_2"),
            ("language_model.", "language_model."),
        )
    ),
    "moss-transcribe-diarize": SourceRule(
        prefixes=(("model.", ""),),
        # ``Qwen3ForCausalLM`` nests the transformer under ``model``, so the
        # source's ``language_model.<...>`` becomes ``language_model.model.<...>``.
        renames=(("language_model.", "language_model.model."),),
    ),
}

#: Families whose source tensors are bfloat16 and whose canonical native
#: dtype is therefore bfloat16 (no lossy cast on conversion).
BFLOAT16_FAMILIES = frozenset(SOURCE_RULES)

#: Families whose source is a torch ZIP rather than safetensors.
TORCH_SOURCE_FAMILIES = frozenset({"fun-asr-nano"})

#: Native parameters a port *derives* at construction rather than loading from a
#: checkpoint, so the strict key diff must not expect them on disk. Listed here
#: as data beside the layout rules and consumed by ``pipeline.load_model``,
#: which validates that each declared name really is a model parameter (a stale
#: declaration is an error, not a silent skip).
#:
#: ``qwen3-asr`` builds its sinusoidal position table from the config in
#: ``_Sinusoids``; the source checkpoint stores no equivalent tensor.
DERIVED_NATIVE_PARAMETERS = {
    "qwen3-asr": ("audio_tower.positional_embedding.table",),
}
DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "U32": 4,
    "U64": 8,
}


@dataclass(frozen=True)
class CheckpointInfo:
    files: tuple[str, ...]
    tensor_count: int
    tensor_bytes: int
    largest_tensor_bytes: int
    dtypes: dict[str, int]


def _bounded_json(path: Path, limit: int = 16_000_000):
    if path.stat().st_size > limit:
        raise ValueError("Checkpoint metadata exceeds size bound")
    return json.loads(path.read_text())


def safetensor_header(path: str | Path) -> dict:
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if length > 16_000_000 or length > size - 8:
            raise ValueError("Invalid or oversized safetensors header")
        header = json.loads(handle.read(length))
    intervals = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        dtype = tensor["dtype"]
        shape = tensor["shape"]
        start, end = tensor["data_offsets"]
        if dtype not in DTYPE_BYTES or any(not isinstance(dim, int) or dim < 0 for dim in shape):
            raise ValueError(f"Unsupported dtype/shape for {name}")
        expected = math.prod(shape) * DTYPE_BYTES[dtype]
        if start < 0 or end < start or end - start != expected or end > size - 8 - length:
            raise ValueError(f"Invalid tensor offsets for {name}")
        intervals.append((start, end))
    previous = 0
    for start, end in sorted(intervals):
        if start != previous:
            raise ValueError("Overlapping or non-contiguous tensor data")
        previous = end
    if previous != size - 8 - length:
        raise ValueError("Unexpected trailing tensor data")
    return header


def checkpoint_files(path: str | Path) -> tuple[Path, ...]:
    path = Path(path).resolve()
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError("Native inference requires safetensors; convert .pt offline")
        return (path,)
    index = path / "model.safetensors.index.json"
    if index.exists():
        metadata = _bounded_json(index)
        weight_map = metadata.get("weight_map") if isinstance(metadata, dict) else None
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in weight_map.items()
            )
        ):
            raise ValueError("Checkpoint index needs a nonempty string weight_map")
        names = sorted(set(weight_map.values()))
        files = []
        for name in names:
            candidate = (path / name).resolve()
            if not candidate.is_relative_to(path) or candidate.suffix != ".safetensors":
                raise ValueError("Unsafe checkpoint shard path")
            files.append(candidate)
        return tuple(files)
    files = tuple(sorted(path.glob("*.safetensors")))
    if not files:
        raise ValueError("No native safetensors checkpoint found")
    return files


def inspect_checkpoint(path: str | Path) -> CheckpointInfo:
    files = checkpoint_files(path)
    keys, total, largest, dtypes = set(), 0, 0, {}
    for file in files:
        for name, tensor in safetensor_header(file).items():
            if name == "__metadata__":
                continue
            if name in keys:
                raise ValueError(f"Duplicate tensor name across shards: {name}")
            keys.add(name)
            size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
            total += size
            largest = max(largest, size)
            dtypes[tensor["dtype"]] = dtypes.get(tensor["dtype"], 0) + size
    if not keys:
        raise ValueError("Checkpoint contains no tensors")
    return CheckpointInfo(tuple(str(file) for file in files), len(keys), total, largest, dtypes)


def convert_tensor(name: str, tensor: np.ndarray, model: str) -> np.ndarray:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model family: {model}")
    if model in {"sensevoice", "paraformer", "ct-punc"}:
        if name.endswith(("fsmn_block.weight", "cif_conv1d.weight")):
            if tensor.ndim != 3:
                raise ValueError(f"Expected rank-3 Conv1d tensor for {name}")
            return tensor.transpose(0, 2, 1)
        if ".conv." in name and name.endswith("weight") and tensor.ndim == 4:
            return tensor.transpose(0, 2, 3, 1)
    elif model == "fsmn-vad" and name.endswith(("conv_left.weight", "conv_right.weight")):
        if tensor.ndim != 4 or tensor.shape[1] != 1 or tensor.shape[3] != 1:
            raise ValueError(f"Expected (D,1,K,1) tensor for {name}")
        return tensor[:, 0, :, :]
    elif model in {"campplus", "emotion2vec"} and name.endswith("weight"):
        if tensor.ndim == 3:
            return tensor.transpose(0, 2, 1)
        if tensor.ndim == 4:
            return tensor.transpose(0, 2, 3, 1)
    elif model == "whisper" and name.endswith(("conv1.weight", "conv2.weight")):
        if tensor.ndim != 3:
            raise ValueError(f"Expected rank-3 Whisper convolution for {name}")
        return tensor.transpose(0, 2, 1)
    elif model == "fun-asr-nano":
        # SANM fsmn_block: torch (D, 1, K) -> MLX Conv1d (D, K, 1).
        if name.endswith("fsmn_block.weight"):
            if tensor.ndim != 3:
                raise ValueError(f"Expected rank-3 FSMN block for {name}")
            return tensor.transpose(0, 2, 1)
    elif model == "qwen3-asr":
        # Qwen3-ASR audio tower: torch Conv2d (O, I, KH, KW) -> MLX (O, KH, KW, I).
        if ".conv." in name and name.endswith("weight") and tensor.ndim == 4:
            return tensor.transpose(0, 2, 3, 1)
    elif model in {"glm-asr-nano", "moss-transcribe-diarize"}:
        # torch Conv1d (O, I, K) -> MLX Conv1d (O, K, I).
        if name.endswith(("conv1.weight", "conv2.weight")):
            if tensor.ndim != 3:
                raise ValueError(f"Expected rank-3 convolution for {name}")
            return tensor.transpose(0, 2, 1)
    return tensor


def convert_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    model: str,
    dtype: str = "float16",
    shard_bytes: int = 64_000_000,
    tensor_limit: int = 384_000_000,
    prefix: str = "",
    config: Mapping | None = None,
    allow_legacy: bool = False,
    stream_large_tensors: bool = False,
    name_map: Mapping[str, str] | None = None,
    allowed_source_keys: tuple[str, ...] | None = None,
    source_rule: str | None = None,
) -> CheckpointInfo:
    """Convert a local checkpoint to the native layout, one tensor at a time.

    Two source formats are supported: a torch ZIP (``.pt``, mmap+``weights_only``)
    and HF/native ``.safetensors``. The safetensors path never materialises the
    whole state dict: each shard is mmapped, its tensor copied one at a time, and
    the copy released before the next (see :func:`_convert_safetensors_source`).

    ``source_rule`` selects a declared :data:`SOURCE_RULES` entry, which both
    renames keys and (via :func:`convert_tensor`) fixes tensor layout. A
    safetensors source without a rule keeps its parameter names unchanged, so
    the family's model class must expect exactly those names.

    Only call inside the guarded launcher. No network or pickle fallback.
    Strict inference loading later checks the architecture's keys and shapes.
    """
    from .memory import MemoryPolicy, guarded_child_active

    if not guarded_child_active():
        raise RuntimeError("Checkpoint conversion must run under the memory guard")
    if model not in SUPPORTED_MODELS or dtype not in SUPPORTED_DTYPES:
        raise ValueError("Unsupported model or output precision")
    if dtype == "bfloat16" and model not in BFLOAT16_FAMILIES:
        raise ValueError("bfloat16 output is only defined for the bf16-native LLM families")
    if source_rule is not None and source_rule not in SOURCE_RULES:
        raise ValueError(f"Unknown source rule: {source_rule}")
    if source_rule is not None and model != source_rule:
        raise ValueError(f"Source rule {source_rule} does not describe model {model}")
    if not 1_000_000 <= shard_bytes <= 128_000_000 or not 0 < tensor_limit <= 384_000_000:
        raise ValueError("Unsafe shard/tensor buffer size")
    source, destination = Path(source), Path(destination)
    is_safetensors = source.is_dir() or source.suffix == ".safetensors"
    # A safetensors source of a non-torch family must declare its layout rule
    # (otherwise nothing renames its keys onto the native names). A torch
    # source's layout lives in convert_tensor, but a rule may still be passed
    # for its `drop` list (fun-asr-nano's tied lm_head).
    if is_safetensors and model not in TORCH_SOURCE_FAMILIES and source_rule is None:
        raise ValueError(
            "Safetensors sources must declare their layout: pass source_rule"
        )
    source_is_file = source.is_file()
    has_index = (source / "model.safetensors.index.json").is_file()
    has_shards = bool(tuple(source.glob("*.safetensors")))
    if not source_is_file and not (has_index or has_shards):
        raise ValueError("Missing source checkpoint")
    if source_is_file and source.stat().st_size > 8_000_000_000:
        raise ValueError("Oversized source checkpoint")
    if destination.exists():
        raise FileExistsError("Conversion destination must not already exist")

    import os

    policy = MemoryPolicy(**json.loads(os.environ["FUNASR_MLX_MEMORY_POLICY"]))
    if is_safetensors:
        return _convert_safetensors_source(
            source,
            destination,
            model=model,
            dtype=dtype,
            shard_bytes=shard_bytes,
            tensor_limit=tensor_limit,
            config=config,
            source_rule=source_rule,
        )

    import torch

    is_zip = zipfile.is_zipfile(source)
    if not is_zip:
        if not allow_legacy:
            raise ValueError("Legacy checkpoint requires explicit allow_legacy admission")
        if source.stat().st_size + 500_000_000 > policy.process_limit:
            raise ValueError("Legacy checkpoint plus reader headroom exceeds job budget")
    state = torch.load(source, map_location="cpu", weights_only=True, mmap=is_zip)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty state_dict mapping")
    destination.mkdir(parents=True)
    pending, buffered, shard_number, weight_map = {}, 0, 0, {}
    skipped: list[str] = []
    dropped: list[str] = []

    # A bfloat16 source kept at bfloat16 stays a raw uint16 carrier (NumPy has
    # no bfloat16) and is written under the BF16 code, exactly as the
    # safetensors path does. Anything else is rounded on the host, in place.
    output_code = {"float16": "F16", "float32": "F32", "bfloat16": "BF16"}[dtype]
    write_shard = None
    if output_code == "BF16":
        from .checkpoint_stream import write_shard

    def metadata() -> dict:
        # Exactly the two keys every already-published native shard carries;
        # the output dtype belongs in the index metadata, not here.
        return {"format": "funasr-mlx", "model": model}

    def flush():
        nonlocal pending, buffered, shard_number
        if not pending:
            return
        shard_number += 1
        filename = f"model-{shard_number:05d}.safetensors"
        if write_shard is None:
            from safetensors.numpy import save_file

            temporary = destination / (filename + ".partial")
            save_file(pending, str(temporary), metadata=metadata())
            temporary.replace(destination / filename)
        else:
            write_shard(destination / filename, pending, metadata=metadata(), code=output_code)
        weight_map.update({name: filename for name in pending})
        pending, buffered = {}, 0

    for original_name, tensor in state.items():
        if not isinstance(original_name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Non-tensor checkpoint entries are not silently discarded")
        if source_rule is not None and original_name in SOURCE_RULES[source_rule].drop:
            dropped.append(original_name)
            continue
        if allowed_source_keys is not None and original_name not in allowed_source_keys:
            if ".decoder." not in original_name and ".ema" not in original_name:
                raise ValueError(
                    f"Checkpoint key outside declared inference subset: {original_name}"
                )
            skipped.append(original_name)
            continue
        if prefix and not original_name.startswith(prefix):
            raise ValueError(f"Tensor outside requested source prefix: {original_name}")
        name = (name_map or {}).get(original_name, original_name[len(prefix) :])
        if name in weight_map or name in pending:
            raise ValueError(f"Duplicate converted tensor name: {name}")
        large = tensor.numel() * max(tensor.element_size(), 4) > tensor_limit
        if large and not stream_large_tensors:
            raise ValueError(f"Tensor exceeds conversion buffer bound: {name}")
        is_bf16 = tensor.dtype == torch.bfloat16
        if is_bf16:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            array = tensor.view(torch.uint16).numpy()
        else:
            array = tensor.detach().numpy()
        del tensor
        array = convert_tensor(name, array, model)
        keep_bf16 = is_bf16 and dtype == "bfloat16"
        if is_bf16 and not keep_bf16:
            array = _bf16_carrier_to(array, dtype)
        elif array.dtype.kind == "f" and array.dtype != np.dtype(dtype):
            array = array.astype(np.dtype(dtype), copy=False)
        if name.endswith(("running_mean", "running_var")) and not is_bf16:
            array = array.astype(np.float32, copy=False)
        # ascontiguousarray promotes a 0-d array to shape (1,), which would
        # corrupt the rank-0 BatchNorm training counters; keep those as-is.
        if array.ndim:
            array = np.ascontiguousarray(array)
        if large:
            from .checkpoint_stream import write_tensor

            flush()
            shard_number += 1
            filename = f"model-{shard_number:05d}.safetensors"
            write_tensor(
                destination / filename,
                name,
                array,
                dtype=array.dtype,
                code="BF16" if keep_bf16 else None,
                metadata=metadata(),
            )
            weight_map[name] = filename
            del array
            continue
        size = array.nbytes
        if buffered and buffered + size > shard_bytes:
            flush()
        pending[name] = array
        buffered += size
        del array
        if buffered >= shard_bytes:
            flush()
    flush()
    index = {
        "metadata": {
            "format": "funasr-mlx",
            "model": model,
            "dtype": dtype,
            "source_format": "zip" if is_zip else "legacy",
            "stream_large_tensors": stream_large_tensors,
            "name_map": dict(name_map or {}),
            "source_rule": source_rule,
            "training_only_skipped": sorted(skipped),
            "dropped_source_keys": sorted(dropped),
        },
        "weight_map": weight_map,
    }
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    if config is not None:
        (destination / "config.json").write_text(json.dumps(dict(config), indent=2) + "\n")
    return inspect_checkpoint(destination)


def info_json(info: CheckpointInfo) -> str:
    return json.dumps(asdict(info), indent=2)


#: safetensors type code -> NumPy dtype used to carry the payload in memory.
#: ``BF16`` maps to ``uint16``: NumPy has no bfloat16, so a bfloat16 tensor is
#: moved around as its raw 16-bit words and only ever written under the
#: ``BF16`` code. Every other code maps to its exact NumPy equivalent.
_CARRIER_DTYPES = {
    "BF16": np.uint16,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def _carry(tensor, code: str) -> np.ndarray:
    """Copy one source tensor into a small owned array in its carrier dtype.

    The copy is the point: the source buffer belongs to a lazily-read shard
    handle, and converted arrays that still referenced it would pin the whole
    shard in memory. Every layout transpose therefore happens *after* this
    copy, on the owned array. bfloat16 is viewed as ``uint16`` first because
    ``torch.Tensor.numpy`` refuses the bfloat16 scalar type outright.
    """
    if code == "BF16":
        tensor = tensor.view(__import__("torch").uint16)
    return np.ascontiguousarray(tensor.numpy())


def _bf16_carrier_to(array: np.ndarray, dtype: str) -> np.ndarray:
    """Round a ``uint16``-carried bfloat16 tensor to the requested precision.

    bfloat16 -> float32 is exact (a 16-bit left shift; the shift must be done
    in 32-bit lanes or the high half is discarded). The further round to
    float16 happens on the host, exactly as an on-device ``astype`` would.
    """
    as_float32 = np.left_shift(array.astype(np.uint32), 16).view(np.float32)
    if dtype == "float32":
        return as_float32
    return as_float32.astype(np.float16)


def _safetensors_source_files(source: Path) -> tuple[Path, ...]:
    """Resolve one safetensors file or an HF-style sharded directory."""
    if source.is_file():
        return (source,)
    index = source / "model.safetensors.index.json"
    if index.exists():
        metadata = _bounded_json(index)
        weight_map = metadata.get("weight_map") if isinstance(metadata, dict) else None
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in weight_map.items()
            )
        ):
            raise ValueError("Checkpoint index needs a nonempty string weight_map")
        files = []
        for name in sorted(set(weight_map.values())):
            candidate = (source / name).resolve()
            if not candidate.is_relative_to(source.resolve()) or candidate.suffix != ".safetensors":
                raise ValueError("Unsafe checkpoint shard path")
            if not candidate.is_file():
                raise ValueError(f"Checkpoint shard listed in the index is missing: {name}")
            files.append(candidate)
        return tuple(files)
    return tuple(sorted(source.glob("*.safetensors")))


def _convert_safetensors_source(
    source: Path,
    destination: Path,
    *,
    model: str,
    dtype: str,
    shard_bytes: int,
    tensor_limit: int,
    config: Mapping | None,
    source_rule: str | None,
) -> CheckpointInfo:
    """One-tensor-at-a-time conversion of a safetensors source.

    Memory contract: each shard is mmapped by the safetensors reader, one
    tensor is copied into a small owned buffer, converted, and released before
    the next tensor is touched. Peak extra memory is one tensor, not one model.

    Key discipline: every source key is passed through the declared
    ``source_rule`` (which raises on an unrecognised key) and duplicate native
    names are rejected, so a renamed or partially-downloaded upstream
    checkpoint fails the conversion instead of silently producing an
    incomplete native checkpoint.
    """
    from safetensors import safe_open

    rule = SOURCE_RULES[source_rule] if source_rule is not None else None
    files = _safetensors_source_files(source)
    if not files:
        raise ValueError("No safetensors shards found in the source checkpoint")
    declared: dict[str, tuple[str, tuple[int, ...]]] = {}
    for path in files:
        header = safetensor_header(path)
        header.pop("__metadata__", None)
        for name, entry in header.items():
            if name in declared:
                raise ValueError(f"Duplicate source tensor across shards: {name}")
            declared[name] = (entry["dtype"], tuple(entry["shape"]))
    if not declared:
        raise ValueError("Checkpoint contains no tensors")
    # Resolve every source key to its native name up front, so a key outside
    # the declared layout fails before anything is written to disk.
    native_names = {
        name: (rule.apply(name) if rule is not None else name) for name in declared
    }
    dropped = sorted(name for name, target in native_names.items() if target is None)
    if not any(target is not None for target in native_names.values()):
        raise ValueError("The source layout maps no tensor onto a native parameter")

    # One output code per shard, so a bf16 checkpoint stays bf16 on disk.
    output_code = {"float16": "F16", "float32": "F32", "bfloat16": "BF16"}[dtype]
    write_shard = None
    if output_code == "BF16":
        from .checkpoint_stream import write_shard

    destination.mkdir(parents=True)
    pending, buffered, shard_number, weight_map = {}, 0, 0, {}

    def metadata() -> dict:
        # Two keys only, matching every already-published native shard; the
        # output dtype is recorded in the index metadata instead.
        return {"format": "funasr-mlx", "model": model}

    def next_shard() -> Path:
        nonlocal shard_number
        shard_number += 1
        return destination / f"model-{shard_number:05d}.safetensors"

    def flush() -> None:
        nonlocal pending, buffered
        if not pending:
            return
        shard = next_shard()
        if write_shard is None:
            from safetensors.numpy import save_file

            temporary = destination / (shard.name + ".partial")
            save_file(pending, str(temporary), metadata=metadata())
            temporary.replace(shard)
        else:
            write_shard(shard, pending, metadata=metadata(), code=output_code)
        weight_map.update({name: shard.name for name in pending})
        pending, buffered = {}, 0

    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                code, shape = declared[name]
                if code not in _CARRIER_DTYPES:
                    raise ValueError(f"Unsupported source dtype {code} for {name}")
                native = native_names[name]
                if native is None:
                    continue
                if native in weight_map or native in pending:
                    raise ValueError(f"Duplicate converted tensor name: {native}")
                elements = int(np.prod(shape) if shape else 1)
                large = elements * max(DTYPE_BYTES[code], 4) > tensor_limit
                array = convert_tensor(native, _carry(handle.get_tensor(name), code), model)
                # A bfloat16 source kept at bfloat16 stays a raw uint16 carrier
                # and is written under the BF16 code; anything else is rounded
                # to the requested precision here, on the host.
                keep_bf16 = code == "BF16" and dtype == "bfloat16"
                if code == "BF16" and not keep_bf16:
                    array = _bf16_carrier_to(array, dtype)
                elif array.dtype.kind == "f":
                    array = array.astype(np.dtype(dtype), copy=False)
                array = np.ascontiguousarray(array)
                if large:
                    from .checkpoint_stream import write_tensor

                    flush()
                    shard = next_shard()
                    write_tensor(
                        shard,
                        native,
                        array,
                        dtype=array.dtype,
                        code="BF16" if keep_bf16 else None,
                        metadata=metadata(),
                    )
                    weight_map[native] = shard.name
                    del array
                    continue
                size = array.nbytes
                if buffered and buffered + size > shard_bytes:
                    flush()
                pending[native] = array
                buffered += size
                del array
                if buffered >= shard_bytes:
                    flush()
    flush()
    index = {
        "metadata": {
            "format": "funasr-mlx",
            "model": model,
            "dtype": dtype,
            "source_format": "safetensors",
            "source_rule": source_rule,
            "source_shards": [path.name for path in files],
            "source_tensors": len(declared),
            "dropped_source_keys": dropped,
        },
        "weight_map": weight_map,
    }
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    if config is not None:
        (destination / "config.json").write_text(json.dumps(dict(config), indent=2) + "\n")
    return inspect_checkpoint(destination)


def info_json(info: CheckpointInfo) -> str:
    return json.dumps(asdict(info), indent=2)
