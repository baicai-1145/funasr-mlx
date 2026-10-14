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
import subprocess
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
    program = (
        "import sys; "
        f"sys.path.insert(0, {str(_TOOLS)!r}); "
        "import alignment_utilities; "
        "banned = ('mlx', 'mlx.core', 'torch', 'funasr', 'funasr.auto'); "
        "assert not set(banned).intersection(sys.modules), set(banned).intersection(sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", program], check=True, timeout=10)



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


# --------------------------------------------------------------------------- #
# official YAML -> native emotion config
# --------------------------------------------------------------------------- #
def _official_emotion_yaml() -> dict:
    """The real emotion2vec_plus_large shape, trimmed to what the mapper reads."""
    return {
        "model": "Emotion2vec",
        "tokenizer": "CharTokenizer",
        "tokenizer_conf": {"unk_symbol": "<unk>", "split_with_space": True},
        "scope_map": ["d2v_model.", "none"],
        "model_conf": {
            "_name": "data2vec_multi",
            "embed_dim": 1024,
            "depth": 8,
            "num_heads": 16,
            "mlp_ratio": 4.0,
            "norm_eps": 1.0e-05,
            "norm_affine": True,
            "layer_norm_first": False,
            "normalize": True,
            "end_of_block_targets": False,
            "average_top_k_layers": 16,
            "ema_decay": 0.9997,
            "skip_ema": False,
            "d2v_loss": 1.0,
            "cls_loss": 1.0,
            "adversarial_training": False,
            "encoder_dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "layerdrop": 0.0,
            "supported_modality": "AUDIO",
            "modalities": {
                "_name": None,
                "audio": {
                    "type": "AUDIO",
                    "extractor_mode": "layer_norm",
                    "feature_encoder_spec":
                        "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
                    "conv_pos_depth": 5,
                    "conv_pos_width": 95,
                    "conv_pos_groups": 16,
                    "conv_pos_pre_ln": False,
                    "prenet_depth": 4,
                    "num_extra_tokens": 10,
                    "init_extra_token_zero": True,
                    "use_alibi_encoder": True,
                    "num_alibi_heads": 16,
                    "alibi_scale": 1.0,
                    "alibi_max_pos": None,
                    "learned_alibi": False,
                    "learned_alibi_scale": True,
                    "learned_alibi_scale_per_head": True,
                    "learned_alibi_scale_per_layer": False,
                    "model_depth": 8,
                    "mask_prob": 0.55,
                    "prenet_dropout": 0.1,
                    "decoder": {"decoder_dim": 768, "decoder_groups": 16},
                },
                "image": {"type": "IMAGE", "embed_dim": 768},
                "text": {"type": "TEXT", "max_source_positions": 512},
            },
        },
    }


def test_emotion_config_maps_the_official_fields():
    kwargs, report = au.emotion_native_config(_official_emotion_yaml(), ["a", "b", "c"])
    assert kwargs["embed_dim"] == 1024
    assert kwargs["depth"] == 8
    assert kwargs["num_heads"] == 16
    assert kwargs["norm_eps"] == pytest.approx(1.0e-05)
    assert kwargs["feature_encoder_spec"].startswith("[(512, 10, 5)]")
    assert kwargs["conv_pos_depth"] == 5 and kwargs["conv_pos_width"] == 95
    assert kwargs["conv_pos_groups"] == 16
    assert kwargs["prenet_depth"] == 4 and kwargs["num_extra_tokens"] == 10
    assert kwargs["use_alibi_encoder"] is True and kwargs["num_alibi_heads"] == 16
    assert kwargs["vocab_size"] == 3 and kwargs["labels"] == ("a", "b", "c")
    # keys that do not exist natively must never be passed
    assert "d2v_loss" not in kwargs and "modalities" not in kwargs
    assert "_name" not in kwargs and "supported_modality" not in kwargs


def test_emotion_config_renames_alibi_scale_to_alibi_scale_init():
    """The one place where the same value has a different name."""
    raw = _official_emotion_yaml()
    raw["model_conf"]["modalities"]["audio"]["alibi_scale"] = 3.5
    kwargs, _report = au.emotion_native_config(raw, ["a"])
    assert kwargs["alibi_scale_init"] == 3.5
    assert "alibi_scale" not in kwargs


def test_emotion_config_reports_every_unmapped_key_rather_than_dropping_it():
    _kwargs, report = au.emotion_native_config(_official_emotion_yaml(), ["a"])
    unmapped = report["unmapped"]
    assert "ema_decay" in unmapped["ema_and_optimizer"]
    assert "skip_ema" in unmapped["ema_and_optimizer"]
    assert "d2v_loss" in unmapped["pretraining_objectives"]
    assert "cls_loss" in unmapped["pretraining_objectives"]
    assert "encoder_dropout" in unmapped["regularisation"]
    assert "modalities.image" in unmapped["other_modalities_and_top_level"]
    assert "modalities.text" in unmapped["other_modalities_and_top_level"]
    assert "decoder" in unmapped["audio_modality_not_mapped"]
    assert "mask_prob" in unmapped["audio_modality_not_mapped"]
    # everything mapped is genuinely absent from the unmapped buckets
    assert "feature_encoder_spec" not in unmapped["audio_modality_not_mapped"]
    assert report["scope_map"] == ["d2v_model.", "none"]
    assert "model_conf" not in report["ignored_sections"]


def test_emotion_config_flags_a_depth_disagreement():
    raw = _official_emotion_yaml()
    raw["model_conf"]["modalities"]["audio"]["model_depth"] = 6
    _kwargs, report = au.emotion_native_config(raw, ["a"])
    check = report["consistency"]["model_depth_vs_depth"]
    assert check == {"model_depth": 6, "depth": 8, "agree": False}


def test_emotion_config_reports_modalities_beyond_the_declared_one():
    _kwargs, report = au.emotion_native_config(_official_emotion_yaml(), ["a"])
    checks = report["consistency"]
    assert checks["supported_modality"] == "AUDIO"
    assert checks["modalities_beyond_declared"] == ["image", "text"]


def test_emotion_config_defaults_conv_bias_because_the_official_omits_it():
    kwargs, report = au.emotion_native_config(_official_emotion_yaml(), ["a"])
    assert "conv_bias" not in _official_emotion_yaml()["model_conf"]["modalities"]["audio"]
    assert kwargs["conv_bias"] is False
    assert report["defaulted"] == ["model_conf.modalities.audio.conv_bias"]


def test_emotion_config_tolerates_a_missing_optional_key():
    raw = _official_emotion_yaml()
    del raw["model_conf"]["modalities"]["audio"]["alibi_max_pos"]
    kwargs, _report = au.emotion_native_config(raw, ["a"])
    assert "alibi_max_pos" not in kwargs


# --------------------------------------------------------------------------- #
# ct-punc vocabulary format
# --------------------------------------------------------------------------- #
def test_ct_punc_tokens_are_read_as_a_json_list(tmp_path):
    """The published tokens.json is a JSON list, not newline-separated text."""
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps(["<blank>", "<s>", "</s>", "，", "。"], ensure_ascii=False))
    assert au._load_ct_punc_tokens(str(path)) == ["<blank>", "<s>", "</s>", "，", "。"]


def test_ct_punc_tokens_fall_back_to_newline_text(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("a\nb\n\nc\n")
    assert au._load_ct_punc_tokens(str(path)) == ["a", "b", "c"]


def test_ct_punc_tokens_reject_a_non_string_list(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="JSON list of strings"):
        au._load_ct_punc_tokens(str(path))


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