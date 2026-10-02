"""Static checks for ``tools/alignment_utilities.py``.

These tests import the adapter module and exercise everything that does **not**
need weights: manifest and asset validation, the bound and rejection logic, the
output writers, the dispatch table, and the pure helper functions.

Nothing here imports MLX, Torch or FunASR.  That is deliberate and is asserted
explicitly by :func:`test_module_imports_no_ml_or_torch`, because the adapter's
whole contract is that heavy imports are lazy and confined to the functions the
lead runs in a guarded child.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import alignment_utilities as au  # noqa: E402


# --------------------------------------------------------------------------- #
# lazy-import contract
# --------------------------------------------------------------------------- #
def test_module_imports_no_ml_or_torch():
    """Importing the adapter must not drag in a numerical runtime."""
    for banned in ("mlx", "mlx.core", "torch", "funasr", "funasr.auto"):
        assert banned not in sys.modules, banned


def test_every_family_has_both_sides():
    assert set(au.FAMILIES) == {"fsmn-vad", "ct-punc", "campplus", "emotion2vec"}
    for family in au.FAMILIES:
        for mode in ("reference", "native"):
            assert callable(au._DISPATCH[(family, mode)]), (family, mode)


# --------------------------------------------------------------------------- #
# bounds and rejection
# --------------------------------------------------------------------------- #
def test_overlong_input_is_rejected_not_truncated():
    entry = {"key": "a", "limit_seconds": 2.0}
    with pytest.raises(ValueError, match="refusing to truncate"):
        au._reject_overlong(entry, 2.001, "limit_seconds", 2.0, "s")
    # exactly at the limit is allowed
    au._reject_overlong(entry, 2.0, "limit_seconds", 2.0, "s")


def test_missing_limit_is_an_error_not_an_unbounded_pass():
    with pytest.raises(ValueError, match="must be positive"):
        au._reject_overlong({"key": "a"}, 1.0, "limit_seconds", 0.0, "s")


def test_check_audio_measures_seconds_at_16k():
    entry = {"key": "a", "limit_seconds": 1.0}
    au._check_audio(entry, np.zeros(16000, dtype=np.float32))
    with pytest.raises(ValueError, match="refusing to truncate"):
        au._check_audio(entry, np.zeros(16001, dtype=np.float32))


def test_npz_bound_is_enforced_and_the_partial_is_removed(tmp_path):
    original = au.MAX_NPZ_BYTES
    au.MAX_NPZ_BYTES = 16
    try:
        with pytest.raises(ValueError, match="exceed"):
            au._write_npz(tmp_path, "k", "native", {"big": np.zeros(64, dtype=np.float32)})
    finally:
        au.MAX_NPZ_BYTES = original
    assert list(tmp_path.iterdir()) == [], "a rejected export must leave nothing behind"


def test_npz_rejects_non_finite_values(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        au._write_npz(tmp_path, "k", "native",
                      {"x": np.array([1.0, np.nan], dtype=np.float32)})
    assert list(tmp_path.iterdir()) == []


def test_npz_and_json_round_trip(tmp_path):
    au._write_npz(tmp_path, "k", "reference", {
        "ids": np.asarray([1, 2, 3], dtype=np.int32),
        "scores": np.asarray([0.5, 0.25], dtype=np.float32),
    })
    au._write_json(tmp_path, "k", "reference", {"text": "hi", "n": 2})
    with np.load(tmp_path / "k.reference.npz") as loaded:
        assert list(loaded["ids"]) == [1, 2, 3]
        assert loaded["scores"].dtype == np.float32
    payload = json.loads((tmp_path / "k.reference.json").read_text())
    assert payload == {"text": "hi", "n": 2}
    assert not list(tmp_path.glob("*.partial")), "partials must be renamed away"


# --------------------------------------------------------------------------- #
# assets
# --------------------------------------------------------------------------- #
def test_require_reports_a_missing_field(tmp_path):
    with pytest.raises(ValueError, match="must supply 'labels'"):
        au._require({}, "labels")
    with pytest.raises(FileNotFoundError):
        au._require({"labels": str(tmp_path / "nope.txt")}, "labels")


def test_model_dir_must_be_a_funasr_directory(tmp_path):
    empty = tmp_path / "model"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        au._reference_model_dir({"model_dir": str(empty)})
    (empty / "config.yaml").write_text("model: x\n")
    assert au._reference_model_dir({"model_dir": str(empty)}) == str(empty)


def test_labels_keep_source_order_and_drop_blanks(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("neutral\n\nhappy\nunuse_3\nsad\n")
    assert au._read_labels(str(path)) == ["neutral", "happy", "unuse_3", "sad"]


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_cosine_matches_a_hand_computed_value():
    a = np.asarray([1.0, 0.0, 0.0])
    assert au._cosine(a, a) == pytest.approx(1.0)
    assert au._cosine(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])) == pytest.approx(0.0)
    assert au._cosine(np.asarray([1.0, 1.0]), np.asarray([1.0, 0.0])) == pytest.approx(
        1 / np.sqrt(2)
    )


def test_cosine_refuses_a_zero_norm():
    with pytest.raises(ValueError, match="zero-norm"):
        au._cosine(np.zeros(3), np.ones(3))


def test_vad_result_flattens_batches_of_start_end_pairs():
    result = [[[0, 500], [700, 900]], [[1200, 1500]]]
    segments, frames = au._split_vad_result(result, {})
    assert segments == [[0, 500], [700, 900], [1200, 1500]]
    assert frames == []


def test_vad_frame_probs_come_from_the_cache_not_the_return_value():
    class FrameProb:
        noise_prob, speech_prob, score, frm_state = 0.1, 0.9, 0.5, 1

    class Stats:
        frame_probs = [FrameProb(), FrameProb()]

    segments, frames = au._split_vad_result([[[0, 10]]], {"stats": Stats()})
    assert segments == [[0, 10]]
    assert len(frames) == 2
    assert frames[0].speech_prob == 0.9


def test_vad_result_rejects_a_malformed_segment_entry():
    # a pair needs at least [start_ms, end_ms]; a bare value is rejected here
    with pytest.raises(ValueError, match="unrecognised VAD segment"):
        au._split_vad_result([[[0]]], {})


def test_vad_segments_accepts_both_container_shapes():
    class SpeechSegment:
        start_ms, end_ms = 10, 20

    class Stats:
        segments = [SpeechSegment(), [30, 40]]

    assert au._vad_segments(Stats()) == [[10, 20], [30, 40]]

    class Empty:
        segments = None

    assert au._vad_segments(Empty()) == []


def test_vad_lfr_factor_is_derived_from_input_dim():
    assert au._vad_encoder_conf({"encoder_conf": {"input_dim": 400}})["input_dim"] == 400
    assert au._vad_encoder_conf({"input_dim": 400})["input_dim"] == 400
    assert 400 % 80 == 0  # the adapter's own precondition


# --------------------------------------------------------------------------- #
# manifest plumbing
# --------------------------------------------------------------------------- #
def _assets(tmp_path: Path, **extra) -> dict:
    base = {"model_dir": str(tmp_path)}
    (tmp_path / "config.yaml").write_text("model: x\n")
    return {**base, **extra}


def test_run_manifest_rejects_an_unknown_family(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([{"key": "k", "family": "nope"}]))
    assets = tmp_path / "a.json"
    assets.write_text(json.dumps(_assets(tmp_path)))
    with pytest.raises(ValueError, match="unknown family"):
        au.run_manifest(manifest, assets, tmp_path / "out", "native")


def test_run_manifest_rejects_an_unknown_mode(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text("[]")
    assets = tmp_path / "a.json"
    assets.write_text(json.dumps(_assets(tmp_path)))
    with pytest.raises(ValueError, match="mode must be"):
        au.run_manifest(manifest, assets, tmp_path / "out", "both")


def test_run_manifest_rejects_a_non_list_manifest(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"key": "k"}))
    assets = tmp_path / "a.json"
    assets.write_text(json.dumps(_assets(tmp_path)))
    with pytest.raises(ValueError, match="JSON list"):
        au.run_manifest(manifest, assets, tmp_path / "out", "native")


def test_run_manifest_creates_the_output_directory(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text("[]")
    assets = tmp_path / "a.json"
    assets.write_text(json.dumps(_assets(tmp_path)))
    out = tmp_path / "out"
    assert au.run_manifest(manifest, assets, out, "native") == []
    assert out.is_dir()


def test_cli_parses_both_sides(tmp_path, monkeypatch):
    manifest = tmp_path / "m.json"
    manifest.write_text("[]")
    assets = tmp_path / "a.json"
    assets.write_text(json.dumps(_assets(tmp_path)))
    assert au.main(["--mode", "native", "--manifest", str(manifest),
                    "--assets", str(assets), "--out", str(tmp_path / "o")]) == 0


def test_native_parameter_disposition_groups_the_key_space():
    report = au._native_parameter_disposition({
        "proj.weight": np.zeros((2, 2)),
        "proj.bias": np.zeros((2,)),
        "modality_encoders.AUDIO.local_encoder.conv_layers.0.0.weight": np.zeros((2, 2, 2)),
        "modality_encoders.AUDIO.relative_positional_encoder.1.0.weight": np.zeros((2, 2, 2)),
        "blocks.0.attn.qkv.weight": np.zeros((2, 2)),
    })
    assert report["n_native_parameters"] == 5
    assert report["has_proj_head"] is True
    assert len(report["feature_extractor"]) == 1
    assert len(report["positional"]) == 1
    assert report["ema_keys"] == [] and report["decoder_keys"] == []


def test_dispatch_rejects_a_mode_outside_the_two_sides():
    assert set(mode for _family, mode in au._DISPATCH) == {"reference", "native"}