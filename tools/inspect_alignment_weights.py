"""Inspect safe mmap checkpoint metadata without evaluating tensor values."""

from __future__ import annotations

import argparse
import gc
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    import torch

    catalog = json.loads((args.root / "catalog.json").read_text())
    for family, package in catalog["models"].items():
        weight_files = [f for f in package["files"] if f["path"].endswith((".pt", ".bin"))]
        if len(weight_files) != 1:
            raise ValueError(f"Expected one original checkpoint for {family}")
        path = args.root / "models" / family / weight_files[0]["path"]
        is_zip = zipfile.is_zipfile(path)
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=is_zip)
        if isinstance(payload, dict) and "dims" in payload:
            (path.parent / "dims.json").write_text(json.dumps(payload["dims"], indent=2))
        state = payload
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(payload, dict) and key in payload:
                state = payload[key]
                break
        if not isinstance(state, dict) or not state:
            raise ValueError("Checkpoint does not contain a tensor mapping")
        tensors, other = {}, {}
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                tensors[name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": value.numel() * value.element_size(),
                }
            else:
                other[name] = type(value).__name__
        report = {
            "family": family,
            "checkpoint": str(path),
            "tensors": tensors,
            "non_tensor_entries": other,
            "tensor_count": len(tensors),
            "tensor_bytes": sum(t["bytes"] for t in tensors.values()),
            "largest": sorted(tensors.items(), key=lambda item: item[1]["bytes"], reverse=True)[:5],
            "torch_version": torch.__version__,
            "serialization": "torch-zip-mmap" if is_zip else "legacy-weights-only-cpu",
        }
        (path.parent / "weight-inventory.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "tensors"}), flush=True)
        del payload, state, value
        gc.collect()


if __name__ == "__main__":
    main()
