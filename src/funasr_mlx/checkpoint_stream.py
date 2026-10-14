"""Port notes: docs/port-notes/checkpoint_stream.py.md"""

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

#: safetensors type codes the bounded writer can emit directly, with their
#: bytes per element. Includes ``BF16``, which NumPy cannot name as a dtype:
#: a bfloat16 payload is carried as raw ``uint16`` words and written under the
#: ``BF16`` code, so the bits reach disk unchanged. ``U32`` is the packed-code
#: width of group quantization, which is why a W4A16 shard mixes it with the
#: 2-byte ``BF16`` scales and needs a per-tensor code map.
_CODE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U32": 4,
    "U64": 8,
    "BOOL": 1,
}


def write_shard(
    path: Path,
    tensors: dict,
    *,
    metadata: dict,
    code: str | None = None,
    codes: dict | None = None,
) -> None:
    """Write a whole shard whose tensors are all carried in one dtype.

    Full notes: docs/port-notes/functions.md :: write_shard
    """
    if (code is None) == (codes is None):
        raise ValueError("Pass exactly one of code or codes")
    if code is not None:
        if code not in _CODE_BYTES:
            raise ValueError(f"Unsupported shard type code {code}")
        codes = {name: code for name in tensors}
    header: dict = {"__metadata__": dict(metadata)}
    offset = 0
    for name, array in tensors.items():
        tensor_code = codes.get(name)
        if tensor_code not in _CODE_BYTES:
            raise ValueError(f"Unsupported shard type code {tensor_code} for {name}")
        width = _CODE_BYTES[tensor_code]
        if array.dtype.itemsize != width:
            raise ValueError(
                f"Shard tensor {name} has element width {array.dtype.itemsize}, "
                f"but code {tensor_code} needs {width}"
            )
        size = int(array.size) * width
        header[name] = {
            "dtype": tensor_code,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with temporary.open("wb") as output:
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            for array in tensors.values():
                output.write(np.ascontiguousarray(array).tobytes())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_tensor(
    path: Path,
    name: str,
    array: np.ndarray,
    *,
    dtype,
    metadata,
    chunk_bytes: int = 16_000_000,
    code: str | None = None,
):
    # see docs/port-notes/functions.md :: write_tensor
    output_dtype = np.dtype(dtype)
    if not array.flags.c_contiguous:
        raise ValueError("Large streaming tensors must be C-contiguous")
    if code is not None:
        if _CODE_BYTES.get(code) != array.dtype.itemsize:
            raise ValueError("Streaming type code disagrees with the array element width")
        output_code = code
    elif not output_dtype.isnative or output_dtype.name not in _CODES:
        raise ValueError("Unsupported streaming tensor dtype")
    else:
        output_code = _CODES[output_dtype.name]
    count = int(array.size)
    header = {
        "__metadata__": dict(metadata),
        name: {
            "dtype": output_code,
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
