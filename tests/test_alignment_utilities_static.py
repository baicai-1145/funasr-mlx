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


def test_vad_stages_flattens_batches_of_start_end_pairs():
    result = [[[0, 500], [700, 900]], [[1200, 1500]]]
    segments, frames, scores = au._vad_stages(result, {"stats": _Stats(scores=np.zeros((1, 3, 2)))})
    assert segments == [[0, 500], [700, 900], [1200, 1500]]
    assert frames == []
    assert scores.shape == (1, 3, 2)


def test_vad_stages_reads_frame_probs_and_scores_from_the_cache():
    """Neither is in the return value: both live on ``cache['stats']``."""
    class FrameProb:
        noise_prob, speech_prob, score, frm_state = 0.1, 0.9, 0.5, 1

    cache = {"stats": _Stats(frame_probs=[FrameProb(), FrameProb()],
                            scores=np.arange(8, dtype=np.float32).reshape(1, 4, 2))}
    segments, frames, scores = au._vad_stages([[[0, 10]]], cache)
    assert segments == [[0, 10]]
    assert len(frames) == 2 and frames[0].speech_prob == 0.9
    assert scores.shape == (1, 4, 2)


def test_vad_stages_preserves_the_negative_streaming_end_point():
    segments, _f, _s = au._vad_stages([[[0, -1]]], {"stats": _Stats()})
    assert segments == [[0, -1]], "a dropped -1 end point would hide a streaming disagreement"


def test_vad_stages_rejects_a_malformed_segment_entry():
    with pytest.raises(ValueError, match="unrecognised VAD segment"):
        au._vad_stages([[[0]]], {"stats": _Stats()})


def test_vad_stages_refuses_an_uninitialised_cache():
    with pytest.raises(ValueError, match="no 'stats'"):
        au._vad_stages([], {})
    with pytest.raises(ValueError, match="no 'scores'"):
        au._vad_stages([], {"stats": _Stats(scores=None)})


def test_vad_stages_raises_on_an_absent_attribute_rather_than_defaulting_empty():
    """A renamed field must fail loudly, not read as an empty stage."""

    class Renamed:
        pass

    class OnlyFrameProbs:
        frame_probs = []

    with pytest.raises(ValueError, match="has no 'frame_probs'"):
        au._vad_stages([], {"stats": Renamed()})
    with pytest.raises(ValueError, match="has no 'scores'"):
        au._vad_stages([], {"stats": OnlyFrameProbs()})


def test_required_attr_does_not_invent_a_default():
    class Empty:
        pass

    assert au._required_attr(_Stats(), "frame_probs") == []
    with pytest.raises(ValueError, match="has no 'missing'"):
        au._required_attr(Empty(), "missing")


def test_vad_input_dim_is_read_from_either_config_shape():
    assert au._vad_input_dim({"encoder_conf": {"input_dim": 400}}) == 400
    assert au._vad_input_dim({"input_dim": 256}) == 256
    # falls back to the official fsmn-vad encoder_conf value, never to a guess
    assert au._vad_input_dim({}) == 400


def test_waveform_geometry_crops_only_the_trailing_incomplete_frame():
    """``needed = (T-1) * shift + length``: snipped frames leave no tail frame."""
    frames, shift, length = 10, 160, 400
    needed = (frames - 1) * shift + length
    assert needed == 9 * 160 + 400
    # the full span would be frames*shift + length = 2000, i.e. 200 samples more
    assert needed == frames * shift + length - shift


_UNSET = object()


def _Stats(frame_probs=None, scores=_UNSET):
    class Stats:
        pass

    stats = Stats()
    stats.frame_probs = frame_probs or []
    stats.scores = np.zeros((1, 1, 2), dtype=np.float32) if scores is _UNSET else scores
    return stats


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