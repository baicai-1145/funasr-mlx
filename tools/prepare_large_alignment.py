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
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    args = parser.parse_args()
    model_dir = args.root / "models" / args.family
    raw = yaml.safe_load((model_dir / "config.yaml").read_text())
    inventory = json.loads((model_dir / "weight-inventory.json").read_text())
    names = {}
    config = raw
    disposition = {}
    allowed = None
    if args.family == "emotion2vec":
        labels = (model_dir / "tokens.txt").read_text().splitlines()
        config, disposition = emotion_native_config(raw, labels)
        prefix = "d2v_model."
        allowed = tuple(sorted(inventory["tensors"]))
        missing = sorted(name for name in ("proj.weight", "proj.bias") if name not in allowed)
        if missing:
            raise ValueError(f"Emotion inference head missing from inventory: {missing}")
        names = {name: name.removeprefix(prefix) for name in allowed if name.startswith(prefix)}
    if args.family == "ct-punc":
        # The official tokenizer ships as tokens.json; the model needs the
        # vocabulary size to size its embedding (471067 x 516).
        tokens = json.loads((model_dir / "tokens.json").read_text())
        config = json.loads(json.dumps(raw))  # plain dict, no yaml tags
        config.setdefault("model_conf", {})["vocab_size"] = len(tokens)
    if args.family == "sensevoice":
        # The strict loader sizes ctc_lo from vocab_size; the official yaml
        # leaves it to the tokenizer (25055 pieces in the shipped bpe model).
        import sentencepiece as _spm

        _sp = _spm.SentencePieceProcessor()
        _sp.load(str(model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"))
        config = json.loads(json.dumps(raw))
        config.setdefault("model_conf", {})["vocab_size"] = int(_sp.get_piece_size())
        config.setdefault("model_conf", {})["input_size"] = int(
            (raw.get("frontend_conf") or {}).get("n_mels", 80)
        ) * int((raw.get("frontend_conf") or {}).get("lfr_m", 7))
    if (model_dir / "am.mvn").is_file():
        config.setdefault("frontend_conf", {})["cmvn_file"] = str(model_dir / "am.mvn")
    info = convert_checkpoint(
        model_dir / "model.pt",
        args.destination,
        model=args.family,
        dtype=args.dtype,
        shard_bytes=32_000_000,
        allow_legacy=args.family == "paraformer",
        stream_large_tensors=True,
        name_map=names,
        config=config,
        allowed_source_keys=allowed or None,
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
