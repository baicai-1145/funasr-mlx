"""Convert official bounded checkpoints to the shipping native format in FP32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=["fsmn-vad", "campplus", "whisper"])
    parser.add_argument("--native-root", type=Path)
    args = parser.parse_args()
    from funasr_mlx.checkpoints import convert_checkpoint

    for family in args.families:
        model_dir = args.root / "models" / family
        if family == "whisper":
            config = {"model_conf": json.loads((model_dir / "dims.json").read_text())}
            source = model_dir / "tiny.pt"
        else:
            config = yaml.safe_load((model_dir / "config.yaml").read_text())
            source = model_dir / ("campplus_cn_common.bin" if family == "campplus" else "model.pt")
            if (model_dir / "am.mvn").exists():
                config.setdefault("frontend_conf", {})["cmvn_file"] = str(model_dir / "am.mvn")
        destination = (args.native_root or args.root / "native") / family
        if destination.exists():
            raise FileExistsError(f"Native artifact already exists: {destination}")
        info = convert_checkpoint(
            source,
            destination,
            model=family,
            dtype="float32",
            shard_bytes=32_000_000,
            config=config,
        )
        print(
            json.dumps(
                {
                    "family": family,
                    "native": str(destination),
                    "tensors": info.tensor_count,
                    "bytes": info.tensor_bytes,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
