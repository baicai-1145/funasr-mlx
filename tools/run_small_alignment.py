"""Serialize complete-weight FP32 diagnostic jobs with explicit model admission.

This launcher is deliberately limited to the three prepared small models.
An admission refusal stops immediately; it never waits for memory to recover.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def model_policy(family, mode):
    from funasr_mlx.memory import MemoryPolicy

    # Full tiny: 151MB weights, bounded attention, observed native peak ~227MB.
    if family == "whisper" and mode == "native":
        return MemoryPolicy(process_limit=1_200_000_000, mlx_limit=800_000_000)
    return MemoryPolicy(process_limit=1_600_000_000, mlx_limit=950_000_000)


def model_arguments(storage, family, mode):
    if family == "whisper":
        return [
            "tools/alignment_whisper.py",
            "--mode",
            mode,
            "--root",
            str(storage),
            "--manifest",
            str(storage / "prepared/manifest-english.json"),
            str(storage / "prepared/manifest-chinese.json"),
        ]
    return [
        "tools/alignment_utilities.py",
        "--mode",
        mode,
        "--manifest",
        str(storage / f"inputs/{family}.json"),
        "--assets",
        str(storage / f"inputs/{family}-assets.json"),
        "--out",
        str(storage / f"results/{family}"),
    ]


def main():
    from funasr_mlx.memory import MemoryBudgetExceeded, MemoryPolicy, run_guarded

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Pinned asset campaign directory")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("fsmn-vad", "campplus", "whisper"),
        default=["fsmn-vad", "campplus", "whisper"],
    )
    parser.add_argument("--mode", choices=("both", "reference", "native"), default="both")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    modes = ("reference", "native") if args.mode == "both" else (args.mode,)
    try:
        for family in args.families:
            for mode in modes:
                policy = model_policy(family, mode)
                print(
                    json.dumps(
                        {
                            "role": "full_model_inference",
                            "family": family,
                            "mode": mode,
                            "policy": asdict(policy),
                        }
                    ),
                    flush=True,
                )
                report = run_guarded(
                    [sys.executable, *model_arguments(args.root, family, mode)],
                    root=project,
                    policy=policy,
                    report_path=f".tmp/final-{family}-{mode}-memory.json",
                    timeout_seconds=60,
                    configure_device=mode == "native",
                )
                print(json.dumps(asdict(report)), flush=True)
                if report.returncode:
                    return report.returncode
            if args.mode != "reference":
                report = run_guarded(
                    [
                        sys.executable,
                        "tools/compare_alignment_arrays.py",
                        "--family",
                        family,
                        "--root",
                        str(args.root / f"results/{family}"),
                        "--report",
                        str(args.root / f"results/{family}/comparison.json"),
                    ],
                    root=project,
                    policy=MemoryPolicy(process_limit=500_000_000, mlx_limit=250_000_000),
                    report_path=f".tmp/final-{family}-comparison-memory.json",
                    timeout_seconds=25,
                    configure_device=False,
                )
                print(json.dumps(asdict(report)), flush=True)
                if report.returncode:
                    return report.returncode
    except MemoryBudgetExceeded as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
