"""AWQ calibration: adapt decoder weights to the w4 grid using real audio.

Two phases (mirroring ``gptq_calibrate.py``):

``--phase calibrate``
  Load the bf16 checkpoint, drive REAL audio prefills through the model
  (the same ``build_prefill`` builders the GPTQ calibrator uses), capture
  each transformer block's Linear inputs, and run the AWQ scale+clip search
  per block.  The adjusted weights are saved as a new bf16 checkpoint
  (``<family>-awq``); the layout and parameter names are unchanged, so the
  destination is just another native bf16 checkpoint.

``--phase convert``
  (delegated) Run the existing converter on the AWQ-adjusted checkpoint with
  the requested spec; identical to any other conversion, nothing AWQ-specific.

The never-worse guarantee is per block: if AWQ-adjusted quantization measures
worse than plain RTN on the block's own captured input, the block's original
weights are kept.

Usage (inside the guard)::

    python3 -m funasr_mlx.runtime --root . --report .tmp/x.json --timeout 3600 \\
      -- python3 tools/awq_calibrate.py --family qwen3-asr-0.6b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tools.gptq_calibrate import (  # noqa: E402
    FAMILIES,
    build_prefill,
    load_manifest,
    model_config,
)


def calibrate(args) -> int:
    import mlx.core as mx
    from funasr_mlx.memory import guarded_child_active
    from funasr_mlx.models.llm.awq import AwqBlockQuantizer, AwqConfig
    from funasr_mlx.pipeline import load_model

    if not guarded_child_active():
        raise RuntimeError("AWQ calibration must run under the memory guard")

    native_dir, family, head_path, key = FAMILIES[args.family]
    source = args.native_root / native_dir
    model, raw_config = load_model(source, family=family)
    config = model_config(family, model, raw_config)

    # The decoder block list lives at family-specific paths; qwen3-shaped
    # families all expose ``<root>.model.layers`` after the load (the thinker
    # wrapper keeps HF naming).
    decoder = model.thinker if hasattr(model, "thinker") else model
    if hasattr(decoder, "language_model"):  # moss
        decoder = decoder.language_model
    blocks = decoder.model.layers
    print(f"{args.family}: {len(blocks)} decoder blocks")

    samples = load_manifest(args.manifest, "calibration", args.languages, args.per_language)
    print(f"AWQ calibration utterances: {len(samples)}")

    bits, group = args.bits, args.group_size

    def quantize_func(w):
        wq = mx.quantize(w, bits=bits, group_size=group)
        return mx.dequantize(*wq, bits=bits, group_size=group)

    awq_cfg = AwqConfig(n_grid=args.n_grid, max_shrink=args.max_shrink)
    report_blocks = []
    started = time.perf_counter()
    for index, block in enumerate(blocks):
        quantizer = AwqBlockQuantizer(block, awq_cfg)
        # capture inputs for this block over every utterance
        for sample in samples:
            prefill_for = build_prefill(family, key, model, config, source, sample)
            quantizer.capture(prefill_for)
        stats = quantizer.optimize(quantize_func, do_clip=not args.no_clip)
        stats["block"] = index
        report_blocks.append(stats)
        if index % 4 == 0 or index == len(blocks) - 1:
            elapsed = time.perf_counter() - started
            print(
                f"  block {index:2d}/{len(blocks)} before={stats['before']:.3e} "
                f"after={stats['after']:.3e} fallback={stats['fallback']} "
                f"({elapsed:.0f}s)",
                flush=True,
            )
        # free captured activations before the next block
        for _, module in block.leaf_modules().items():
            if hasattr(module, "input_feat"):
                delattr(module, "input_feat")
        mx.eval(block.parameters())
        mx.clear_cache()

    # save the AWQ-adjusted weights as a plain bf16 checkpoint
    destination = args.destination
    destination.mkdir(parents=True, exist_ok=True)
    from mlx.utils import tree_flatten

    # Reuse the converter's writer: an identity-rule conversion with no
    # quantization section copies the (adjusted) tensors into the native
    # layout, sharded, with config passthrough.
    from funasr_mlx.checkpoints import convert_checkpoint

    info = convert_checkpoint(
        source,
        destination,
        model=family,
        dtype="bfloat16",
        config=raw_config if isinstance(raw_config, dict) else None,
        source_rule="native",
        tensor_limit=args.tensor_limit,
    )
    # The converter re-reads the SOURCE weights, which discards the AWQ
    # adjustments -- so overwrite the destination tensors with the live
    # model's parameters directly instead.
    destination2 = destination
    import shutil

    shutil.rmtree(destination2)
    mx.save_safetensors(
        str(destination2 / "model.safetensors"),
        dict(tree_flatten(model.parameters())),
    )
    (destination2 / "config.json").write_text(json.dumps(raw_config, indent=2))
    print(f"wrote AWQ-adjusted bf16 checkpoint to {destination2}")

    payload = {
        "family": args.family,
        "samples": len(samples),
        "bits": bits,
        "group_size": group,
        "n_grid": awq_cfg.n_grid,
        "max_shrink": awq_cfg.max_shrink,
        "clip": not args.no_clip,
        "blocks": report_blocks,
        "fallbacks": sum(1 for b in report_blocks if b["fallback"]),
        "improved": sum(1 for b in report_blocks if b["after"] < b["before"]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "blocks"}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=tuple(FAMILIES), required=True)
    parser.add_argument(
        "--native-root", type=Path, default=Path("/Volumes/2T/funasr-mlx-alignment/native-llm")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/Volumes/2T/funasr-mlx-alignment/calib/calib-manifest.json"),
    )
    parser.add_argument("--per-language", type=int, default=48)
    parser.add_argument("--languages", default="zh,en,ja,ko,es,fr,de,ru,th,vi")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--max-shrink", type=float, default=0.5)
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--tensor-limit", type=int, default=384_000_000)
    parser.add_argument(
        "--destination", type=Path, default=None, help="AWQ-adjusted bf16 output directory"
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    args.languages = [part for part in args.languages.split(",") if part]
    if args.destination is None:
        args.destination = args.native_root / f"{args.family}-awq-bf16"
    if args.report is None:
        args.report = Path(".tmp/awq") / f"{args.family}-calibrate.json"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    return calibrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
