import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.prepare_alignment_data import mono_16k, safe_target


def test_stereo_downmix_is_explicit_mean():
    samples = np.array([[0.25, 0.75], [-0.5, 0.5]], dtype=np.float32)
    result = mono_16k(samples, 16000)
    np.testing.assert_array_equal(result, np.array([0.5, 0.0], dtype=np.float32))


def test_ravdess_resampling_keeps_mono_and_sample_duration():
    samples = np.ones((480, 2), dtype=np.float32)
    result = mono_16k(samples, 48000)
    assert result.shape == (160,)
    assert result.dtype == np.float32


@pytest.mark.parametrize("channels", [0, 9])
def test_invalid_channel_count_is_rejected(channels):
    with pytest.raises(ValueError, match="channels"):
        mono_16k(np.zeros((16, channels), dtype=np.float32), 16000)


def test_overlong_audio_is_not_truncated():
    with pytest.raises(ValueError, match="bound"):
        mono_16k(np.zeros((160001, 1), dtype=np.float32), 16000)


def test_nonfinite_audio_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        mono_16k(np.array([[np.nan]], dtype=np.float32), 16000)


def test_partial_preparation_retains_provenance_and_updates_real_selection(tmp_path, monkeypatch):
    from tools import prepare_alignment_data as prepare

    output = tmp_path / "prepared"
    output.mkdir()
    entries = [{"key": str(i), "duration_seconds": 2.0} for i in range(2)]
    (output / "corpus-english.json").write_text(json.dumps(entries))
    (output / "manifest-english.json").write_text(json.dumps({"samples": entries}))
    original = {
        "english": {
            "count": 2,
            "license": "CC BY 4.0",
            "archive_sha256": "a" * 64,
            "extracted_bytes": 100,
            "custom_note": "retain me",
        }
    }
    (output / "coverage.json").write_text(json.dumps(original))
    monkeypatch.setattr(
        prepare, "prepare_ravdess", lambda *_: (entries, {"count": 2, "license": "CC BY-NC-SA 4.0"})
    )
    monkeypatch.setattr(
        sys, "argv", ["prepare", "--root", str(tmp_path), "--corpus", "emotion", "--samples", "1"]
    )
    prepare.main()
    result = json.loads((output / "coverage.json").read_text())["english"]
    assert result["license"] == original["english"]["license"]
    assert result["archive_sha256"] == original["english"]["archive_sha256"]
    assert result["custom_note"] == "retain me"
    selected = json.loads((output / "manifest-english.json").read_text())["samples"]
    assert result["selected_count"] == len(selected) == 1


def test_current_source_revision_supersedes_cached_coverage(tmp_path, monkeypatch):
    from tools import prepare_alignment_data as prepare

    output = tmp_path / "prepared"
    output.mkdir()
    for name in ("english", "chinese", "emotion"):
        (output / f"corpus-{name}.json").write_text(
            json.dumps([{"key": name, "duration_seconds": 1.0}])
        )
    cached = {
        "chinese": {
            "license": "not-declared-by-selected-mirror",
            "revision": "stale",
            "custom_note": "retain",
        }
    }
    (output / "coverage.json").write_text(json.dumps(cached))
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "datasets": {
                    "librispeech-test-clean": {"repository": "OpenSLR/12"},
                    "aishell-test": {"repository": "mirror", "revision": "current"},
                    "ravdess-speech": {"record": "1188976"},
                }
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["prepare", "--root", str(tmp_path), "--corpus", "selection"])
    prepare.main()
    chinese = json.loads((output / "coverage.json").read_text())["chinese"]
    assert chinese["revision"] == "current"
    assert chinese["custom_note"] == "retain"


def test_zenodo_source_uses_doi_without_inventing_hub_repository(tmp_path):
    from tools.prepare_alignment_data import source_coverage

    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {"datasets": {"ravdess-speech": {"record": "1188976", "doi": "10.5281/zenodo.1188976"}}}
        )
    )
    metadata = source_coverage(tmp_path, "emotion")
    assert metadata["doi"] == "10.5281/zenodo.1188976"
    assert metadata["license"] == "CC BY-NC-SA 4.0"
    assert "repository" not in metadata


def test_archive_copy_rejects_more_than_declared_bytes():
    from tools.prepare_alignment_data import copy_bounded

    with pytest.raises(ValueError, match="exceeds"):
        copy_bounded(io.BytesIO(b"oversized"), io.BytesIO(), 1)


def test_archive_copy_rejects_truncated_member():
    from tools.prepare_alignment_data import copy_bounded

    with pytest.raises(ValueError, match="shorter"):
        copy_bounded(io.BytesIO(b"short"), io.BytesIO(), 100)


def test_archive_path_cannot_escape_project_data_root():
    root = Path("/project/data")
    with pytest.raises(ValueError, match="escapes"):
        safe_target(root, "../outside")
    assert safe_target(root, "Actor_01/audio.wav") == root / "Actor_01/audio.wav"
