"""Convert four pinned official model states, including admitted legacy/large tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main():
    from funasr_mlx.checkpoints import convert_checkpoint
    from tools.alignment_utilities import emotion_native_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--family", choices=("sensevoice", "paraformer", "ct-punc", "emotion2vec"), required=True
    )
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.root / "models" / args.family
    raw = yaml.safe_load((model_dir / "config.yaml").read_text())
    inventory = json.loads((model_dir / "weight-inventory.json").read_text())
    names = {}
    config = raw
    disposition = {}
    if args.family == "emotion2vec":
        labels = (model_dir / "tokens.txt").read_text().splitlines()
        config, disposition = emotion_native_config(raw, labels)
        prefix = "d2v_model."
        names = {name: name.removeprefix(prefix) for name in inventory["tensors"]}
        unknown = sorted(name for name in inventory["tensors"] if not name.startswith(prefix))
        if unknown:
            raise ValueError(f"Emotion inventory contains tensors outside {prefix}: {unknown[:8]}")
        config["skip_ema"] = bool(raw.get("skip_ema", True))
    if (model_dir / "am.mvn").is_file():
        config.setdefault("frontend_conf", {})["cmvn_file"] = str(model_dir / "am.mvn")
    info = convert_checkpoint(
        model_dir / "model.pt",
        args.destination,
        model=args.family,
        dtype="float32",
        shard_bytes=32_000_000,
        allow_legacy=args.family == "paraformer",
        stream_large_tensors=True,
        name_map=names,
        config=config,
    )
    (args.destination / "source-config.yaml").write_text((model_dir / "config.yaml").read_text())
    (args.destination / "config-disposition.json").write_text(json.dumps(disposition, indent=2))
    print(
        json.dumps(
            {
                "family": args.family,
                "parameters": info.tensor_count,
                "tensor_bytes": info.tensor_bytes,
                "largest": info.largest_tensor_bytes,
            }
        )
    )


if __name__ == "__main__":
    main()
