"""Missing and shared input stages must not masquerade as neural agreement."""

import json
import sys

import numpy as np
import pytest

from tools.compare_alignment_arrays import compare_case


def pair(tmp_path, left, right):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, **left)
    np.savez(b, **right)
    return a, b


def test_missing_required_stage_is_rejected(tmp_path):
    paths = pair(
        tmp_path, {"scores": np.ones(2, np.float32)}, {"unrelated": np.ones(2, np.float32)}
    )
    with pytest.raises(ValueError, match="Required stage"):
        compare_case(*paths, required={"scores"})


def test_prompt_identity_is_an_input_contract_not_a_measurement(tmp_path):
    paths = pair(
        tmp_path,
        {"prompt_tokens": np.array([1], np.int32)},
        {"prompt_tokens": np.array([1], np.int32)},
    )
    result = compare_case(*paths)
    assert result["prompt_tokens"]["status"] == "shared_input"
    assert "exact" not in result["prompt_tokens"]


def test_empty_neural_stage_is_never_agreement(tmp_path):
    paths = pair(tmp_path, {"scores": np.zeros(0, np.float32)}, {"scores": np.zeros(0, np.float32)})
    with pytest.raises(ValueError, match="Empty neural"):
        compare_case(*paths)


def test_optional_stage_mismatch_is_reported_and_fails(tmp_path, monkeypatch, capsys):
    from tools.compare_alignment_arrays import main

    root = tmp_path / "campplus"
    root.mkdir()
    common = {"frontend": np.ones((1, 2), np.float32), "embedding": np.ones(2, np.float32)}
    np.savez(root / "a.reference.npz", **common, optional=np.ones(1, np.float32))
    np.savez(root / "a.native.npz", **common, optional=np.ones(2, np.float32))
    report_path = root / "comparison.json"
    monkeypatch.setattr(sys, "argv", ["compare", "--root", str(root), "--report", str(report_path)])
    assert main() == 1
    capsys.readouterr()
    report = json.loads(report_path.read_text())
    assert report["measurement_integrity"] == "invalid"
    assert report["summary"]["optional"]["mismatched"] == 1
    assert report["comparator_sha256"]


def test_empty_required_vad_segments_are_valid_no_speech(tmp_path, monkeypatch, capsys):
    from tools.compare_alignment_arrays import main

    root = tmp_path / "fsmn-vad"
    root.mkdir()
    arrays = {
        "frontend": np.ones((1, 2), np.float32),
        "scores": np.ones((1, 2), np.float32),
        "segment_bounds": np.zeros((0, 2), np.int32),
    }
    np.savez(root / "a.reference.npz", **arrays)
    np.savez(root / "a.native.npz", **arrays)
    report_path = root / "comparison.json"
    monkeypatch.setattr(sys, "argv", ["compare", "--root", str(root), "--report", str(report_path)])
    assert main() == 0
    capsys.readouterr()
    summary = json.loads(report_path.read_text())["summary"]["segment_bounds"]
    assert summary["empty"] == summary["exact_empty_outputs"] == 1
    assert summary["not_measured"] == 0


def test_disabled_vad_frame_stage_is_explicitly_empty(tmp_path):
    paths = pair(
        tmp_path,
        {"frame_prob": np.zeros((0, 3), np.float32)},
        {"frame_prob": np.zeros((0, 3), np.float32)},
    )
    assert compare_case(*paths)["frame_prob"]["status"] == "empty"
