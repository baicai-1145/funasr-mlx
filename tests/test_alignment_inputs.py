"""Provenance and counts of generated local diagnostic inputs."""

import json
import sys

from tools import build_alignment_inputs as inputs


def prepare(root):
    folder = root / "prepared"
    folder.mkdir()
    for corpus in ("english", "chinese", "emotion"):
        sample = {"key": corpus, "duration_seconds": 1.0, "transcript": "example"}
        (folder / f"manifest-{corpus}.json").write_text(json.dumps({"samples": [sample]}))


def test_counts_are_derived_from_written_entries(tmp_path, monkeypatch, capsys):
    prepare(tmp_path)
    monkeypatch.setattr(sys, "argv", ["inputs", "--root", str(tmp_path)])
    inputs.main()
    report = json.loads(capsys.readouterr().out)
    assert report["utility_manifest_counts"] == {
        "fsmn-vad": 2,
        "campplus": 2,
        "emotion2vec": 1,
        "ct-punc": 2,
    }


def test_explicit_checkpoint_override_survives_regeneration(tmp_path, monkeypatch, capsys):
    prepare(tmp_path)
    checkpoint = tmp_path / "fixed" / "campplus"
    monkeypatch.setattr(
        sys,
        "argv",
        ["inputs", "--root", str(tmp_path), "--checkpoint-override", f"campplus={checkpoint}"],
    )
    inputs.main()
    monkeypatch.setattr(sys, "argv", ["inputs", "--root", str(tmp_path)])
    inputs.main()
    capsys.readouterr()
    assets = json.loads((tmp_path / "inputs/campplus-assets.json").read_text())
    assert assets["checkpoint"] == str(checkpoint)
    assert assets["checkpoint_selection"] == "recorded-override"
