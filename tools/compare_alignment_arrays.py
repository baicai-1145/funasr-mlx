"""Measure complete saved neural outputs; empty stages are never agreement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_STAGES = {
    "fsmn-vad": {"frontend", "scores", "segment_bounds"},
    "campplus": {"frontend", "embedding"},
    "whisper": {"frontend", "encoder", "decoder_logits", "next_token"},
    "ct-punc": {"token_ids", "punc_ids", "logits"},
    "emotion2vec": {"frames", "pooled", "logits", "probabilities"},
}


def compare_case(reference, native, *, required=()):
    result = {}
    with (
        np.load(reference, allow_pickle=False) as original,
        np.load(native, allow_pickle=False) as rewritten,
    ):
        required = set(required)
        missing = required - set(original.files).intersection(rewritten.files)
        if missing:
            raise ValueError(f"Required stage missing from one or both backends: {sorted(missing)}")
        for stage in sorted(set(original.files) | set(rewritten.files)):
            if stage not in original or stage not in rewritten:
                result[stage] = {"status": "not_measured", "reason": "stage absent in one backend"}
                continue
            left, right = original[stage], rewritten[stage]
            if left.shape != right.shape:
                if stage in required:
                    raise ValueError(f"Required stage shape mismatch: {stage}")
                result[stage] = {
                    "status": "shape_mismatch",
                    "reference": list(left.shape),
                    "native": list(right.shape),
                }
                continue
            if left.size > 4_000_000 or right.size > 4_000_000:
                raise ValueError("Saved stage exceeds element bound")
            if left.dtype != right.dtype:
                if stage in required:
                    raise ValueError(f"Required stage dtype mismatch: {stage}")
                result[stage] = {
                    "status": "dtype_mismatch",
                    "reference": str(left.dtype),
                    "native": str(right.dtype),
                }
                continue
            if left.size == 0:
                result[stage] = {"status": "empty", "shape": list(left.shape)}
                if stage == "segment_bounds":
                    result[stage]["exact"] = True
                if stage not in {"segment_bounds", "frame_prob", "frame_state"}:
                    raise ValueError(f"Empty neural stage: {stage}")
                continue
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError(f"Non-finite neural stage: {stage}")
            if stage == "prompt_tokens":
                if not np.array_equal(left, right):
                    raise ValueError("Shared prompt input contract differs between backends")
                result[stage] = {
                    "status": "shared_input",
                    "shape": list(left.shape),
                    "origin": "reference tokenizer; copied input, not numerical evidence",
                }
                continue
            if left.dtype.kind in "iub":
                result[stage] = {
                    "status": "measured",
                    "shape": list(left.shape),
                    "exact": bool(np.array_equal(left, right)),
                }
                continue
            expected, actual = left.astype(np.float64), right.astype(np.float64)
            difference = actual - expected
            denominator = max(float(np.linalg.norm(expected)), 1e-30)
            result[stage] = {
                "status": "measured",
                "shape": list(left.shape),
                "dtype": str(left.dtype),
                "max_abs": float(np.max(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(difference**2))),
                "relative_l2": float(np.linalg.norm(difference) / denominator),
                "reference_l2": denominator,
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--family", choices=sorted(REQUIRED_STAGES))
    arguments = parser.parse_args()
    family = arguments.family or arguments.root.name
    if family not in REQUIRED_STAGES:
        raise ValueError("Declare the model family to enforce required numerical stages")
    references = {
        p.name.removesuffix(".reference.npz") for p in arguments.root.glob("*.reference.npz")
    }
    natives = {p.name.removesuffix(".native.npz") for p in arguments.root.glob("*.native.npz")}
    if references != natives:
        raise ValueError("Reference and native result key sets differ")
    cases = []
    for reference in sorted(arguments.root.glob("*.reference.npz")):
        native = reference.with_name(reference.name.replace(".reference.", ".native."))
        if not native.exists():
            raise FileNotFoundError(f"Missing matching native result: {native}")
        case = {"key": reference.name.removesuffix(".reference.npz")}
        try:
            case["stages"] = compare_case(reference, native, required=REQUIRED_STAGES[family])
        except ValueError as error:
            case.update(stages={}, error=str(error))
        cases.append(case)
    if not cases:
        raise ValueError("No matching reference/native cases")
    summary = {}
    for stage in sorted({s for case in cases for s in case["stages"]}):
        records = [case["stages"][stage] for case in cases if stage in case["stages"]]
        measured = [r for r in records if r["status"] == "measured"]
        shared = sum(r["status"] == "shared_input" for r in records)
        statuses = {
            status: sum(r["status"] == status for r in records)
            for status in sorted({r["status"] for r in records})
        }
        summary[stage] = {
            "measured": len(measured),
            "shared_input": shared,
            "not_measured": sum(r["status"] == "not_measured" for r in records),
            "empty": statuses.get("empty", 0),
            "mismatched": sum(r["status"] in {"shape_mismatch", "dtype_mismatch"} for r in records),
            "status_counts": statuses,
        }
        floating = [r for r in measured if "max_abs" in r]
        discrete = [r for r in measured if "exact" in r]
        if floating and discrete:
            raise ValueError(f"Stage switches between floating and discrete outputs: {stage}")
        if floating:
            summary[stage].update(
                {
                    "max_abs": max(r["max_abs"] for r in measured),
                    "relative_l2_max": max(r["relative_l2"] for r in measured),
                }
            )
        elif discrete:
            summary[stage]["exact_count"] = sum(r["exact"] for r in measured)
        empty_outputs = [r for r in records if r["status"] == "empty" and r.get("exact")]
        if empty_outputs:
            summary[stage]["exact_empty_outputs"] = len(empty_outputs)
    invalid = sum(
        "error" in case
        or any(
            stage["status"] not in {"measured", "shared_input", "empty"}
            for stage in case["stages"].values()
        )
        for case in cases
    )
    report = {
        "schema": "funasr-mlx-measurements/2",
        "comparator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_integrity": "valid" if not invalid else "invalid",
        "invalid_cases": invalid,
        "empty_stage_policy": {
            "segment_bounds": "Empty required endpoint arrays are valid no-speech outputs; "
            "frontend and scores must remain nonempty",
            "frame_prob": "Empty indicates uncaptured diagnostics, not numerical agreement",
            "frame_state": "Empty indicates uncaptured diagnostics, not numerical agreement",
        },
        "sample_count": len(cases),
        "family": family,
        "required_stages": sorted(REQUIRED_STAGES[family]),
        "criteria": "Measurements only; no implicit acceptance tolerance",
        "summary": summary,
        "cases": cases,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_suffix(arguments.report.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(arguments.report)
    print(
        json.dumps(
            {
                "report": str(arguments.report),
                "samples": len(cases),
                "measurement_integrity": report["measurement_integrity"],
                "summary": summary,
            },
            indent=2,
        )
    )
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
