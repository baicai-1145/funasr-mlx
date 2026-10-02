"""Subset acquisition must preserve pinned campaign download records."""

import json
import sys

import pytest

from tools import prepare_alignment_assets as assets


def test_existing_catalog_cannot_silently_change_revision(tmp_path, monkeypatch):
    original = {"models": {"test": {"revision": "pinned"}}, "datasets": {}, "errors": []}
    (tmp_path / "catalog.json").write_text(json.dumps(original))
    changed = tmp_path / "replacement.json"
    changed.write_text(json.dumps({**original, "models": {"test": {"revision": "different"}}}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["assets", "--root", str(tmp_path), "--from-catalog", str(changed), "--scan-only"],
    )
    with pytest.raises(ValueError, match="Pinned campaign catalog"):
        assets.main()
    assert json.loads((tmp_path / "catalog.json").read_text()) == original


def test_group_run_merges_existing_download_records(tmp_path, monkeypatch, capsys):
    original = {
        "local": str(tmp_path / "models/old/model.pt"),
        "url": "https://example.test/pinned/model.pt",
        "actual_sha256": "a" * 64,
    }
    (tmp_path / "downloaded.json").write_text(json.dumps([original]))
    catalog = tmp_path / "input-catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "models": {},
                "datasets": {
                    "test": {
                        "files": [
                            {
                                "path": "metadata.json",
                                "bytes": 1,
                                "url": "https://example.test/pinned/metadata.json",
                            }
                        ]
                    }
                },
                "errors": [],
            }
        )
    )

    def download(item, path):
        return {**item, "local": str(path), "actual_sha256": "b" * 64, "actual_bytes": 1}

    monkeypatch.setattr(assets, "download", download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assets",
            "--root",
            str(tmp_path),
            "--from-catalog",
            str(catalog),
            "--group",
            "datasets",
            "--metadata-only",
        ],
    )
    assets.main()
    assets.main()
    capsys.readouterr()
    records = json.loads((tmp_path / "downloaded.json").read_text())
    assert len(records) == 2
    assert {row["local"]: row for row in records}[original["local"]] == original
