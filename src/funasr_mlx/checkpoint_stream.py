"""Bounded safetensors writing for contiguous large inference parameters."""

from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np

_CODES = {
    "float32": "F32",
    "float16": "F16",
    "float64": "F64",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint8": "U8",
    "uint32": "U32",
    "uint64": "U64",
    "bool": "BOOL",
}


def write_tensor(
    path: Path, name: str, array: np.ndarray, *, dtype, metadata, chunk_bytes: int = 16_000_000
):
    """Write one tensor, converting only bounded slices rather than its full data."""
    output_dtype = np.dtype(dtype)
    if not array.flags.c_contiguous:
        raise ValueError("Large streaming tensors must be C-contiguous")
    if not output_dtype.isnative or output_dtype.name not in _CODES:
        raise ValueError("Unsupported streaming tensor dtype")
    count = int(array.size)
    header = {
        "__metadata__": dict(metadata),
        name: {
            "dtype": _CODES[output_dtype.name],
            "shape": list(array.shape),
            "data_offsets": [0, count * output_dtype.itemsize],
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    temporary = path.with_suffix(path.suffix + ".partial")
    flat = array.reshape(-1)
    stride = max(1, chunk_bytes // max(array.dtype.itemsize, output_dtype.itemsize))
    try:
        with temporary.open("wb") as output:
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            for start in range(0, count, stride):
                block = np.asarray(flat[start : start + stride], dtype=output_dtype, order="C")
                output.write(memoryview(block).cast("B"))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
