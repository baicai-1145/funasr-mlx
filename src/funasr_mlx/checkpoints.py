"""Port notes: docs/port-notes/checkpoints.py.md"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import struct
import zipfile
from typing import Any, Mapping

import numpy as np

SUPPORTED_MODELS = (
    "sensevoice",
    "paraformer",
    "ct-punc",
    "fsmn-vad",
    "campplus",
    "emotion2vec",
    "whisper",
    "fun-asr-nano",
    "qwen3-asr",
    "glm-asr-nano",
    "moss-transcribe-diarize",
)

SUPPORTED_DTYPES = ("float16", "float32", "bfloat16")


@dataclass(frozen=True)
class SourceRule:
    # see docs/port-notes/functions.md :: SourceRule

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


# see docs/port-notes/comments.md :: checkpoints.py:55
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
        prefixes=(("thinker.audio_tower.", "audio_tower."), ("thinker.", "thinker.")),
        renames=tuple(
            (f"audio_tower.conv2d{index}.{suffix}", f"audio_tower.conv2d{index}.conv.{suffix}")
            for index in (1, 2, 3)
            for suffix in ("weight", "bias")
        ),
        # see docs/port-notes/comments.md :: checkpoints.py:72
        drop=("thinker.lm_head.weight",),
    ),
    "glm-asr-nano": SourceRule(
        prefixes=(
            ("audio_tower.", "audio_tower."),
            ("multi_modal_projector.linear_1", "audio_tower.linear_1"),
            ("multi_modal_projector.linear_2", "audio_tower.linear_2"),
            ("language_model.", "language_model."),
        )
    ),
    "moss-transcribe-diarize": SourceRule(
        prefixes=(("model.", ""),),
        renames=(("language_model.", "language_model.model."),),
    ),
}

IDENTITY_SOURCE_RULE = "native"

BFLOAT16_FAMILIES = frozenset(SOURCE_RULES)

TORCH_SOURCE_FAMILIES = frozenset({"fun-asr-nano"})

# see docs/port-notes/comments.md :: checkpoints.py:95
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

QUANTIZED_SUFFIXES = (".weight", ".scales", ".biases")


def quantize_bf16_weight(
    name: str,
    array: np.ndarray,
    *,
    code: str,
    bits: int,
    group_size: int,
    mode: str = "affine",
    gptq: "GptqWeight | None" = None,
) -> dict[str, np.ndarray]:
    """Pack one 16-bit weight with MLX group quantization.

    Full notes: docs/port-notes/functions.md :: quantize_bf16_weight
    """
    import mlx.core as mx

    if code not in ("BF16", "F16"):
        raise ValueError(f"Quantization requires a 16-bit source weight, got {code}")
    if not name.endswith(".weight"):
        raise ValueError(f"Expected a Linear weight name, got {name}")
    module = name[: -len(".weight")]
    if code == "BF16":
        weight = mx.array(array).view(mx.bfloat16)
    elif code == "F16":
        weight = mx.array(array).astype(mx.float16)
    else:  # pragma: no cover - rejected above
        raise ValueError(f"Quantization requires a 16-bit source weight, got {code}")
    if gptq is None:
        packed, scales, biases = mx.quantize(
            weight, group_size=group_size, bits=bits, mode=mode
        )
    else:
        if mode != "affine":
            raise ValueError("GPTQ is only defined for affine group quantization")
        packed, scales, biases = gptq.quantize_tensor(
            name, weight, bits=bits, group_size=group_size
        )
    mx.eval(packed, scales, biases)
    del weight
    if biases is None:
        raise ValueError(f"Mode {mode} produced no group biases for {name}")
    carry = (
        (lambda tensor: np.array(tensor.view(mx.uint16)))
        if code == "BF16"
        else (lambda tensor: np.array(tensor))
    )
    result = {
        name: np.array(packed),
        module + ".scales": carry(scales),
        module + ".biases": carry(biases),
    }
    del packed, scales, biases
    return result


@dataclass(frozen=True)
class GptqWeight:
    # see docs/port-notes/functions.md :: GptqWeight

    hessians: Mapping[str, np.ndarray]
    configs: Mapping[str, Any] = field(default_factory=dict)
    default_config: Any = None
    # see docs/port-notes/comments.md :: checkpoints.py:179
    method: str = "gptq"
    outcomes: dict[str, dict] = field(default_factory=dict)

    def has(self, name: str) -> bool:
        return name in self.hessians

    def config_for(self, name: str):
        from .models.llm.gptq import GptqConfig

        if name in self.configs:
            return GptqConfig.from_dict(self.configs[name])
        if self.default_config is not None:
            return GptqConfig.from_dict(self.default_config)
        return GptqConfig()

    def quantize_tensor(
        self, name: str, weight, *, bits: int, group_size: int
    ) -> tuple:
        # see docs/port-notes/functions.md :: quantize_tensor
        import mlx.core as mx

        from .models.llm.gptq import gptq_quantize, rtn_quantize

        if self.method not in ("gptq", "rtn"):
            raise ValueError(f"Unknown quantization method: {self.method}")
        hessian = self.hessians.get(name)
        if hessian is None:
            raise ValueError(f"No GPTQ Hessian was collected for {name}")
        config = self.config_for(name)
        if config.bits != bits or config.group_size != group_size:
            raise ValueError(
                f"GPTQ config for {name} uses bits={config.bits} "
                f"group_size={config.group_size}, but the converter asked for "
                f"bits={bits} group_size={group_size}"
            )
        # see docs/port-notes/comments.md :: checkpoints.py:215
        rows, width = int(weight.shape[0]), int(weight.shape[1])
        source = np.empty((rows, width), dtype=np.float32)
        step = max(int(config.row_chunk) or rows, 1)
        for start in range(0, rows, step):
            stop = min(start + step, rows)
            source[start:stop] = np.array(weight[start:stop].astype(mx.float32))
        hessian32 = np.asarray(hessian, dtype=np.float32)
        if self.method == "rtn":
            outcome = rtn_quantize(source, config, hessian32)
        else:
            outcome = gptq_quantize(source, hessian32, config)
        del source, hessian32
        self.outcomes[name] = outcome.to_json()
        dtype = weight.dtype
        packed = mx.array(outcome.packed)
        scales = mx.array(outcome.scales).astype(dtype)
        biases = mx.array(outcome.biases).astype(dtype)
        mx.eval(packed, scales, biases)
        del outcome
        return packed, scales, biases


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
        if name.endswith("fsmn_block.weight"):
            if tensor.ndim != 3:
                raise ValueError(f"Expected rank-3 FSMN block for {name}")
            return tensor.transpose(0, 2, 1)
    elif model == "qwen3-asr":
        if ".conv." in name and name.endswith("weight") and tensor.ndim == 4:
            return tensor.transpose(0, 2, 3, 1)
    elif model in {"glm-asr-nano", "moss-transcribe-diarize"}:
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
    quantization: Mapping | None = None,
    gptq: "GptqWeight | None" = None,
) -> CheckpointInfo:
    """Convert a local checkpoint to the native layout, one tensor at a time.

    Full notes: docs/port-notes/functions.md :: convert_checkpoint
    """
    if model not in SUPPORTED_MODELS or dtype not in SUPPORTED_DTYPES:
        raise ValueError("Unsupported model or output precision")
    if dtype == "bfloat16" and model not in BFLOAT16_FAMILIES:
        raise ValueError("bfloat16 output is only defined for the bf16-native LLM families")
    if source_rule is not None and source_rule not in SOURCE_RULES:
        if source_rule != IDENTITY_SOURCE_RULE:
            raise ValueError(f"Unknown source rule: {source_rule}")
    if source_rule is not None and source_rule not in (model, IDENTITY_SOURCE_RULE):
        raise ValueError(f"Source rule {source_rule} does not describe model {model}")
    if not 1_000_000 <= shard_bytes <= 128_000_000 or not 0 < tensor_limit <= 384_000_000:
        raise ValueError("Unsafe shard/tensor buffer size")
    source, destination = Path(source), Path(destination)
    is_safetensors = source.is_dir() or source.suffix == ".safetensors"
    # see docs/port-notes/comments.md :: checkpoints.py:415
    if is_safetensors and model not in TORCH_SOURCE_FAMILIES and source_rule is None:
        raise ValueError("Safetensors sources must declare their layout: pass source_rule")
    source_is_file = source.is_file()
    has_index = (source / "model.safetensors.index.json").is_file()
    has_shards = bool(tuple(source.glob("*.safetensors")))
    if not source_is_file and not (has_index or has_shards):
        raise ValueError("Missing source checkpoint")
    if source_is_file and source.stat().st_size > 10_000_000_000:
        raise ValueError("Oversized source checkpoint")
    if destination.exists():
        raise FileExistsError("Conversion destination must not already exist")

    import os

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
            quantization=quantization,
            gptq=gptq,
        )

    import torch

    is_zip = zipfile.is_zipfile(source)
    if not is_zip:
        if not allow_legacy:
            raise ValueError("Legacy checkpoint requires explicit allow_legacy admission")
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

    output_code = {"float16": "F16", "float32": "F32", "bfloat16": "BF16"}[dtype]
    write_shard = None
    if output_code == "BF16":
        from .checkpoint_stream import write_shard

    def metadata() -> dict:
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


# see docs/port-notes/comments.md :: checkpoints.py:581
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
    # see docs/port-notes/functions.md :: _carry
    if code == "BF16":
        tensor = tensor.view(__import__("torch").uint16)
    return np.ascontiguousarray(tensor.numpy())


def _bf16_carrier_to(array: np.ndarray, dtype: str) -> np.ndarray:
    # see docs/port-notes/functions.md :: _bf16_carrier_to
    as_float32 = np.left_shift(array.astype(np.uint32), 16).view(np.float32)
    if dtype == "float32":
        return as_float32
    return as_float32.astype(np.float16)


def _safetensors_source_files(source: Path) -> tuple[Path, ...]:
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
    quantization: Mapping | None = None,
    gptq: "GptqWeight | None" = None,
) -> CheckpointInfo:
    """One-tensor-at-a-time conversion of a safetensors source.

    Full notes: docs/port-notes/functions.md :: _convert_safetensors_source
    """
    from safetensors import safe_open

    from .models.llm.quant import QuantizationSpec

    import mlx.core as mx

    spec = QuantizationSpec.from_dict(quantization) if quantization is not None else None
    if spec is not None and dtype not in ("bfloat16", "float16"):
        raise ValueError("W4A16 quantization requires a 16-bit checkpoint dtype")
    if spec is not None and not spec.targets:
        raise ValueError("A quantization section must declare at least one target prefix")
    rule = None
    if source_rule is not None and source_rule != IDENTITY_SOURCE_RULE:
        rule = SOURCE_RULES[source_rule]
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
    native_names = {name: (rule.apply(name) if rule is not None else name) for name in declared}
    dropped = sorted(name for name, target in native_names.items() if target is None)
    if not any(target is not None for target in native_names.values()):
        raise ValueError("The source layout maps no tensor onto a native parameter")

    output_code = {"float16": "F16", "float32": "F32", "bfloat16": "BF16"}[dtype]
    write_shard = None
    if output_code == "BF16" or spec is not None:
        from .checkpoint_stream import write_shard

    destination.mkdir(parents=True)
    pending, buffered, shard_number, weight_map = {}, 0, 0, {}
    quant_pending: dict[str, np.ndarray] = {}
    quant_codes: dict[str, str] = {}
    quant_buffered = 0
    quantized_names: list[str] = []
    skipped_quantization: list[str] = []

    def metadata() -> dict:
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

    def flush_quantized() -> None:
        nonlocal quant_pending, quant_codes, quant_buffered
        if not quant_pending:
            return
        shard = next_shard()
        if write_shard is None:
            raise ValueError("Mixed-width output requires the bounded writer")
        write_shard(shard, quant_pending, metadata=metadata(), codes=quant_codes)
        weight_map.update({name: shard.name for name in quant_pending})
        quant_pending, quant_codes, quant_buffered = {}, {}, 0

    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                code, shape = declared[name]
                if code not in _CARRIER_DTYPES:
                    raise ValueError(f"Unsupported source dtype {code} for {name}")
                native = native_names[name]
                if native is None:
                    continue
                if native in weight_map or native in pending or native in quant_pending:
                    raise ValueError(f"Duplicate converted tensor name: {native}")
                elements = int(np.prod(shape) if shape else 1)
                large = elements * max(DTYPE_BYTES[code], 4) > tensor_limit
                array = _carry(handle.get_tensor(name), code)
                if source_rule != IDENTITY_SOURCE_RULE:
                    array = convert_tensor(native, array, model)
                keep_bf16 = code == "BF16" and dtype == "bfloat16"
                if spec is not None and spec.quantizes_tensor(native, array.shape):
                    # see docs/port-notes/comments.md :: checkpoints.py:754
                    flush_quantized()
                    # see docs/port-notes/comments.md :: checkpoints.py:756
                    tensor_bits = spec.bits_for_weight(native)
                    tensor_group = spec.group_for(native[: -len(".weight")])
                    packed = quantize_bf16_weight(
                        native,
                        array,
                        code="BF16" if keep_bf16 else "F16",
                        bits=tensor_bits,
                        group_size=tensor_group,
                        mode=spec.mode,
                        gptq=gptq if (gptq is not None and gptq.has(native)) else None,
                    )
                    del array
                    # see docs/port-notes/comments.md :: checkpoints.py:769
                    mx.clear_cache()
                    side_code = "BF16" if keep_bf16 else "F16"
                    module = native[: -len(".weight")]
                    group_bytes = sum(int(value.nbytes) for value in packed.values())
                    shard_bound = (
                        max(256 * 1024 * 1024, int(packed[native].nbytes)) + 32 * 1024 * 1024
                    )
                    if group_bytes > shard_bound:
                        raise ValueError(
                            f"Quantized group for {native} is {group_bytes / 1e6:.1f} MB, over "
                            f"the {shard_bound / 1e6:.1f} MB per-shard bound the strict loader "
                            "enforces; use a smaller group size or a wider bit width"
                        )
                    quant_pending[native] = packed[native]
                    quant_pending[module + ".scales"] = packed[module + ".scales"]
                    quant_pending[module + ".biases"] = packed[module + ".biases"]
                    quant_codes[native] = "U32"
                    quant_codes[module + ".scales"] = side_code
                    quant_codes[module + ".biases"] = side_code
                    quant_buffered += group_bytes
                    quantized_names.append(native)
                    del packed
                    if quant_buffered >= shard_bytes:
                        flush_quantized()
                    continue
                if (
                    spec is not None
                    and not spec.quantizes_tensor(native, array.shape)
                    and spec.is_linear_weight(native, array.shape)
                ):
                    # see docs/port-notes/comments.md :: checkpoints.py:800
                    skipped_quantization.append(native)
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
    flush_quantized()
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
            "quantization": spec.to_dict() if spec is not None else None,
            "quantized_tensors": len(quantized_names),
            "quantization_skipped": sorted(skipped_quantization),
            "gptq": gptq.outcomes if gptq is not None else None,
        },
        "weight_map": weight_map,
    }
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    if config is not None:
        (destination / "config.json").write_text(json.dumps(dict(config), indent=2) + "\n")
    return inspect_checkpoint(destination)


def info_json(info: CheckpointInfo) -> str:
    return json.dumps(asdict(info), indent=2)
