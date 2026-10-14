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
    import subprocess

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
    for family in args.families:
        for mode in modes:
            print(
                json.dumps({"role": "full_model_inference", "family": family, "mode": mode}),
                flush=True,
            )
            report = subprocess.run(
                [sys.executable, *model_arguments(args.root, family, mode)], cwd=project
            )
            print(f"rc={report.returncode}", flush=True)
            if report.returncode:
                return report.returncode
        if args.mode != "reference":
            report = subprocess.run(
                [sys.executable, *model_arguments(args.root, family, mode)], cwd=project
            )
            print(f"rc={report.returncode}", flush=True)
            if report.returncode:
                return report.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
