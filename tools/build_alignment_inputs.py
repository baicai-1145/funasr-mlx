"""Build actual per-family counts and persist explicit native artifact selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILIES = ("fsmn-vad", "campplus", "emotion2vec", "ct-punc")


def write_json(path, document):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(document, ensure_ascii=True, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-override", action="append", default=[], metavar="FAMILY=PATH")
    args = parser.parse_args()
    inputs = args.root / "inputs"
    inputs.mkdir(exist_ok=True)
    selection = inputs / "selection.json"
    previous = json.loads(selection.read_text()) if selection.is_file() else {}
    overrides = dict(previous.get("checkpoint_overrides") or {})
    for argument in args.checkpoint_override:
        family, separator, path = argument.partition("=")
        if separator != "=" or family not in FAMILIES or not path:
            raise ValueError("Checkpoint override must be FAMILY=PATH for a utility model")
        overrides[family] = str(Path(path).resolve())
    corpora = {}
    for name in ("english", "chinese", "emotion"):
        entries = json.loads((args.root / f"prepared/manifest-{name}.json").read_text())["samples"]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Prepared {name} manifest must contain actual samples")
        corpora[name] = entries
    counts = {}
    plans = []
    for family in FAMILIES:
        model_dir = args.root / "models" / family
        selected = (
            corpora["emotion"]
            if family == "emotion2vec"
            else corpora["english"] + corpora["chinese"]
        )
        entries = [
            {
                **sample,
                "key": family + "_" + sample["key"],
                "family": family,
                "limit_seconds": 10 if family == "emotion2vec" else 30,
            }
            for sample in selected
        ]
        if family == "ct-punc":
            entries = [
                {**sample, "text": sample["transcript"], "limit_chars": 512} for sample in entries
            ]
        checkpoint = overrides.get(family, str(args.root / "native" / family))
        assets = {
            "model_dir": str(model_dir),
            "config": str(model_dir / "config.yaml"),
            "checkpoint": checkpoint,
            "checkpoint_selection": "recorded-override" if family in overrides else "default",
            "cmvn": str(model_dir / "am.mvn") if (model_dir / "am.mvn").exists() else None,
        }
        if family == "campplus":
            assets["weights"] = str(model_dir / "campplus_cn_common.bin")
        if family == "emotion2vec":
            assets["labels"] = str(model_dir / "tokens.txt")
        if family == "ct-punc":
            assets["tokens"] = str(model_dir / "tokens.json")
        counts[family] = len(entries)
        plans.append((family, entries, assets))
    write_json(selection, {"checkpoint_overrides": overrides, "manifest_counts": counts})
    for family, entries, assets in plans:
        write_json(inputs / f"{family}.json", entries)
        write_json(inputs / f"{family}-assets.json", assets)
    print(json.dumps({"utility_manifest_counts": counts, "checkpoint_overrides": overrides}))


if __name__ == "__main__":
    main()
