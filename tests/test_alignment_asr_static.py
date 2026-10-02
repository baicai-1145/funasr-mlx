"""Static checks for the alignment adapter.

These run with **neither MLX nor Torch installed**, which is the point: the
planning, inventory and comparison logic must be reviewable and diffable without
either backend. AST and pure-Python only -- no numeric work, no weights.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import alignment_asr as alignment  # noqa: E402

SOURCE = (TOOLS / "alignment_asr.py").read_text()
TREE = ast.parse(SOURCE)
FUNCTIONS = {node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)}


def _dump(name: str) -> str:
    return ast.dump(FUNCTIONS[name])


def _imported_roots(name: str) -> set[str]:
    """Top-level module names actually imported anywhere inside ``name``.

    Checked from the import nodes rather than by substring-matching the dump: a
    docstring or a schema tag like ``funasr-mlx-alignment-run/2`` mentions a
    backend without importing it, and only the import node is evidence.
    """
    roots: set[str] = set()
    for node in ast.walk(FUNCTIONS[name]):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _calls(name: str) -> set[str]:
    """Plain function names called inside ``name``."""
    return {
        node.func.id
        for node in ast.walk(FUNCTIONS[name])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _uses_attribute(name: str, module: str, attribute: str) -> bool:
    """Whether ``name`` contains ``module.attribute`` anywhere."""
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == attribute
        and isinstance(node.value, ast.Name)
        and node.value.id == module
        for node in ast.walk(FUNCTIONS[name])
    )


def _manifest(root: Path, durations=(3.0, 2.0)) -> Path:
    samples = []
    for index, duration in enumerate(durations):
        audio = root / f"s{index}.wav"
        audio.write_bytes(b"placeholder")  # the spec builder only checks existence
        samples.append({"key": f"s{index}", "audio": str(audio), "duration_seconds": duration})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}))
    return manifest


# --------------------------------------------------------------------------- #
# Backend isolation -- the rule that makes the comparison meaningful
# --------------------------------------------------------------------------- #
def test_no_backend_imported_at_module_scope():
    """Importing the adapter must not drag in MLX or Torch."""
    names: set[str] = set()
    for node in TREE.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert "mlx" not in names
    assert "torch" not in names


def test_reference_process_never_touches_mlx_or_the_native_port():
    """The reference must not import the MLX frontend or funasr_mlx at all.

    Two failures in one: it wastes memory, and -- worse -- running the *native*
    frontend inside the reference process would make the frontend comparison
    tautological, since both sides would execute identical code.
    """
    helpers = (
        "run_reference",
        "_run_reference_sample",
        "_build_reference_frontend",
        "_build_funasr_reference",
        "_build_canonical_whisper",
        "_sensevoice_ctc_logits",
        "_apply_sensevoice_conditioning",
        "_resolve_wav_frontend",
    )
    for helper in helpers:
        roots = _imported_roots(helper)
        assert "mlx" not in roots, f"{helper} imports MLX"
        assert "funasr_mlx" not in roots, f"{helper} imports the native port"
    # The reference half of the file must not reach for the native frontend.
    assert "funasr_mlx" not in SOURCE.split("def run_native")[0]


def test_native_process_never_touches_torch_or_funasr():
    """The native run must not pull in the reference backend."""
    for helper in (
        "run_native",
        "_run_native_sample",
        "_build_native_model",
        "_native_frontend_config",
    ):
        roots = _imported_roots(helper)
        assert "torch" not in roots, f"{helper} imports Torch"
        assert "funasr" not in roots, f"{helper} imports the reference package"
    # The checkpoint is deliberately read through Torch's reader so both backends
    # consume the same bytes; that one shared helper is the only exception.
    assert "torch" in _imported_roots("_load_checkpoint_state")


def test_both_frontends_are_built_independently_from_one_config():
    """Each backend gets a real frontend, configured from the same spec values."""
    assert "WavFrontend" in _dump("_resolve_wav_frontend")
    assert "FbankConfig" in _dump("_run_native") or "FbankConfig" in _dump(
        "_native_frontend_config"
    )
    # The native config is derived from the spec's shared settings.
    assert "variant" in _dump("_native_frontend_config")
    # CMVN must be applied on both sides or the features cannot be compared.
    assert "cmvn" in _dump("_build_reference_frontend")
    assert "cmvn" in _dump("run_native")


# --------------------------------------------------------------------------- #
# No hand-rolled reference math
# --------------------------------------------------------------------------- #
def test_reference_uses_real_upstream_constructors():
    """FunASR models come from their real constructors, not zero-arg guesses."""
    assert "model_conf" in _dump("_split_funasr_config")
    for name in ("encoder_conf", "decoder_conf", "predictor_conf", "ctc_conf"):
        assert name in _dump("_split_funasr_config"), f"{name} must be forwarded"
    assert "importlib" in _dump("_build_funasr_reference")
    # Registry imports are required: configs name submodules as strings.
    for variant in alignment.SUPPORTED:
        if variant.family != "whisper":
            assert variant.registry_imports, f"{variant.key} needs registry imports"


def test_whisper_reference_is_ast_loaded_from_upstream():
    """Canonical Whisper must be the real module, extracted from its source."""
    assert _uses_attribute("reference_definitions", "ast", "parse")
    assert "exec" in _calls("reference_definitions")
    assert "compile" in _calls("reference_definitions")
    # A missing definition is an error, never a silent skip.
    assert "missing" in _dump("reference_definitions")
    assert "use_sdpa" in _dump("_build_canonical_whisper")


def test_paraformer_final_logits_come_from_the_decoder_not_ctc():
    """Published paraformer has ctc_weight=0, so a CTC head may not exist.

    The reference must use cal_decoder_with_predictor, whose log-softmax is the
    real final output.
    """
    body = _dump("_run_reference_sample")
    assert "cal_decoder_with_predictor" in body
    assert "model.ctc" not in body, "paraformer logits must not come from a CTC head"
    assert "_sensevoice_ctc_logits" in body, "SenseVoice still needs its real CTC head"


def test_ctc_head_absence_is_refused_not_substituted():
    """If the head is missing, say so rather than approximating it."""
    body = _dump("_sensevoice_ctc_logits")
    assert "AlignmentError" in body
    assert "refusing to substitute" in body


def test_sensevoice_conditioning_ids_are_recorded():
    """The conditioning prefix must be explicit, not implied by a shape.

    1 (language) + 1 (textnorm) + 2 (event/emo) frames of prefix, and the
    lengths must grow to match or the encoder masks real audio away.
    """
    body = _dump("_apply_sensevoice_conditioning")
    assert "lid_dict" in body
    assert "textnorm_dict" in body
    assert "event_emo_ids" in body
    assert "prefix_frames" in body
    # The metadata must carry them so a mismatch is visible in the diff.
    assert "conditioning_ids" in _dump("_run_reference_sample")
    assert "conditioning_ids" in _dump("compare_alignment")


# --------------------------------------------------------------------------- #
# Safe loading and inventory
# --------------------------------------------------------------------------- #
def test_checkpoint_loading_is_safe_and_memory_bounded():
    """weights_only=True and mmap=True are both required.

    weights_only refuses arbitrary pickle globals even for a trusted, pinned
    checkpoint; mmap avoids a second full copy of the weights in RSS.
    """
    keywords = {
        node.arg: ast.literal_eval(node.value)
        for node in ast.walk(FUNCTIONS["_load_checkpoint_state"])
        if isinstance(node, ast.keyword)
        and node.arg in {"weights_only", "mmap", "map_location"}
        and isinstance(node.value, ast.Constant)
    }
    assert keywords.get("weights_only") is True
    assert keywords.get("mmap") is True
    assert keywords.get("map_location") == "cpu"


def test_inventory_reports_both_directions():
    """Missing *and* unexpected keys, or a renamed head gets silently skipped."""
    report = alignment.inventory_report(["a", "b", "c"], {"b": 1, "c": 2}, label="x")
    assert report["missing"] == ["a"]
    assert report["unexpected"] == []
    assert report["verdict"] == "mismatch"
    both = alignment.inventory_report(["a"], {"a": 1, "z": 2})
    assert both["unexpected"] == ["z"]
    assert alignment.inventory_report(["a", "b"], {"b": 0, "a": 0})["verdict"] == "ok"


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #
def test_unsupported_variants_are_refused_with_reasons():
    for key in ("whisper-large", "whisper-large-v3-turbo"):
        assert key in alignment.UNSUPPORTED
        assert alignment.UNSUPPORTED[key].strip()
    with pytest.raises(alignment.AlignmentError) as excinfo:
        alignment.resolve_variant("whisper-large")
    assert "not aligned" in str(excinfo.value)


def test_only_bounded_whisper_is_offered():
    whisper = [v for v in alignment.SUPPORTED if v.family == "whisper"]
    assert [v.key for v in whisper] == ["whisper-tiny"]
    assert "tiny" in whisper[0].checkpoint
    assert "bounded" in whisper[0].notes.lower()


# --------------------------------------------------------------------------- #
# Spec and comparison
# --------------------------------------------------------------------------- #
def test_spec_is_json_serialisable_and_carries_shared_frontend_config(tmp_path):
    spec = alignment.build_alignment_spec(
        _manifest(tmp_path), "paraformer-zh", tmp_path, assets_root=tmp_path
    )
    assert json.loads(json.dumps(spec)) == spec
    assert spec["dtype"] == "float32"
    assert spec["audio"]["batch_size"] == 1
    assert spec["variant"]["frontend"]["lfr_m"] == 7
    # Both backends read the frontend settings from here, so they cannot diverge.
    assert "frontend" in spec["variant"]
    assert spec["variant"]["cmvn"].endswith("am.mvn")


def test_overtime_samples_are_refused_not_truncated(tmp_path):
    """One backend truncating and the other not would invalidate everything."""
    manifest = _manifest(tmp_path, durations=(900.0,))
    with pytest.raises(alignment.AlignmentError) as excinfo:
        alignment.build_alignment_spec(manifest, "paraformer-zh", tmp_path, assets_root=tmp_path)
    assert "exceeds" in str(excinfo.value)


def test_compare_flags_token_stage_dtype_and_conditioning_disagreement():
    reference = {
        "variant": {"key": "paraformer-zh"},
        "dtype": "float32",
        "frontend_config": {"lfr_m": 7},
        "cmvn": "am.mvn",
        "samples": [
            {
                "key": "a",
                "token_ids": [1, 2],
                "conditioning_ids": {"language_id": 3},
                "stages": [{"stage": "encoder", "shape": [1, 5, 8], "dtype": "float32"}],
            }
        ],
    }
    native = {
        "variant": {"key": "paraformer-zh"},
        "dtype": "float32",
        "frontend_config": {"lfr_m": 7},
        "cmvn": "am.mvn",
        "samples": [
            {
                "key": "a",
                "token_ids": [1, 3],
                "conditioning_ids": {"language_id": 4},
                "stages": [{"stage": "encoder", "shape": [1, 6, 8], "dtype": "float16"}],
            }
        ],
    }
    kinds = {f["kind"] for f in alignment.compare_alignment(reference, native)["findings"]}
    assert {
        "token_ids_differ",
        "conditioning_ids_differ",
        "stage_shape_mismatch",
        "stage_dtype_mismatch",
    } <= kinds
    assert alignment.compare_alignment(reference, reference)["clean"] is True


def test_compare_flags_a_truncated_stage():
    """A capped activation must never be compared as if it were the full tensor."""
    base = {
        "variant": {"key": "paraformer-zh"},
        "dtype": "float32",
        "frontend_config": {},
        "cmvn": None,
        "samples": [
            {
                "key": "a",
                "token_ids": [1],
                "conditioning_ids": None,
                "stages": [
                    {"stage": "encoder", "shape": [1, 5], "dtype": "float32", "truncated": True}
                ],
            }
        ],
    }
    kinds = {f["kind"] for f in alignment.compare_alignment(base, base)["findings"]}
    assert "stage_truncated" in kinds


def test_compare_flags_a_cmvn_or_frontend_mismatch():
    """Different CMVN or frontend settings invalidate the whole comparison."""
    left = {
        "variant": {"key": "paraformer-zh"},
        "dtype": "float32",
        "frontend_config": {"lfr_m": 7},
        "cmvn": "am.mvn",
        "samples": [],
    }
    right = dict(left, cmvn="other.mvn", frontend_config={"lfr_m": 1})
    kinds = {f["kind"] for f in alignment.compare_alignment(left, right)["findings"]}
    assert "cmvn_mismatch" in kinds
    assert "frontend_config_mismatch" in kinds


# --------------------------------------------------------------------------- #
# Bounded saving
# --------------------------------------------------------------------------- #
def test_saved_tensors_are_bounded_and_truncation_is_recorded():
    import numpy as np

    small = np.zeros((2, 3), np.float32)
    _, truncated = alignment._bounded(small)
    assert truncated is False
    value, truncated = alignment._bounded(np.zeros(alignment.MAX_SAVED_ELEMENTS + 10, np.float32))
    assert truncated is True
    assert value.size == alignment.MAX_SAVED_ELEMENTS
    assert alignment._stage_record("encoder", small)["truncated"] is False


def test_split_funasr_config_prefers_model_conf_and_refuses_emptiness():
    config = {
        "model_conf": {"encoder": "sanm_encoder", "input_size": 560},
        "decoder_conf": {"vocab_size": 7000},
        "ctc_conf": {"ctc_weight": 0.0},
    }
    split = alignment._split_funasr_config(config)
    assert split["encoder"] == "sanm_encoder"
    assert split["decoder_conf"] == {"vocab_size": 7000}
    assert split["ctc_conf"] == {"ctc_weight": 0.0}
    with pytest.raises(alignment.AlignmentError):
        alignment._split_funasr_config({"frontend_conf": {}})
