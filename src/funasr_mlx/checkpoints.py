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
)
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
) -> CheckpointInfo:
    """Convert a local torch ZIP checkpoint with mmap and one tensor at a time.

    Only call inside the guarded launcher. No network or pickle fallback.
    Strict inference loading later checks the architecture's keys and shapes.
    """
    from .memory import MemoryPolicy, guarded_child_active

    if not guarded_child_active():
        raise RuntimeError("Checkpoint conversion must run under the memory guard")
    if model not in SUPPORTED_MODELS or dtype not in {"float16", "float32"}:
        raise ValueError("Unsupported model or output precision")
    if not 1_000_000 <= shard_bytes <= 128_000_000 or not 0 < tensor_limit <= 384_000_000:
        raise ValueError("Unsafe shard/tensor buffer size")
    source, destination = Path(source), Path(destination)
    if not source.is_file() or source.stat().st_size > 8_000_000_000:
        raise ValueError("Missing or oversized source checkpoint")
    if destination.exists():
        raise FileExistsError("Conversion destination must not already exist")
    import torch
    from safetensors.numpy import save_file

    import os

    policy = MemoryPolicy(**json.loads(os.environ["FUNASR_MLX_MEMORY_POLICY"]))
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

    def flush():
        nonlocal pending, buffered, shard_number
        if not pending:
            return
        shard_number += 1
        filename = f"model-{shard_number:05d}.safetensors"
        temporary = destination / (filename + ".partial")
        save_file(pending, str(temporary), metadata={"format": "funasr-mlx", "model": model})
        temporary.replace(destination / filename)
        weight_map.update({name: filename for name in pending})
        pending, buffered = {}, 0

    for original_name, tensor in state.items():
        if not isinstance(original_name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Non-tensor checkpoint entries are not silently discarded")
        if prefix and not original_name.startswith(prefix):
            raise ValueError(f"Tensor outside requested source prefix: {original_name}")
        name = (name_map or {}).get(original_name, original_name[len(prefix) :])
        if name in weight_map or name in pending:
            raise ValueError(f"Duplicate converted tensor name: {name}")
        large = tensor.numel() * max(tensor.element_size(), 4) > tensor_limit
        if large and not stream_large_tensors:
            raise ValueError(f"Tensor exceeds conversion buffer bound: {name}")
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        array = tensor.detach().numpy()
        array = convert_tensor(name, array, model)
        output_dtype = dtype if array.dtype.kind == "f" else array.dtype
        if name.endswith(("running_mean", "running_var")):
            output_dtype = "float32"
        size = array.size * np.dtype(output_dtype).itemsize
        if large:
            from .checkpoint_stream import write_tensor

            flush()
            shard_number += 1
            filename = f"model-{shard_number:05d}.safetensors"
            write_tensor(
                destination / filename,
                name,
                array,
                dtype=output_dtype,
                metadata={"format": "funasr-mlx", "model": model},
            )
            weight_map[name] = filename
            del array, tensor
            continue
        if buffered and buffered + size > shard_bytes:
            flush()
        array = np.asarray(array, dtype=output_dtype, order="C")
        pending[name] = array
        buffered += array.nbytes
        if buffered >= shard_bytes:
            flush()
        del array, tensor
    flush()
    index = {
        "metadata": {
            "format": "funasr-mlx",
            "model": model,
            "dtype": dtype,
            "source_format": "zip" if is_zip else "legacy",
            "stream_large_tensors": stream_large_tensors,
            "name_map": dict(name_map or {}),
        },
        "weight_map": weight_map,
    }
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    if config is not None:
        (destination / "config.json").write_text(json.dumps(dict(config), indent=2) + "\n")
    return inspect_checkpoint(destination)


def info_json(info: CheckpointInfo) -> str:
    return json.dumps(asdict(info), indent=2)
