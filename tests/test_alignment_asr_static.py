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


def _source(name: str) -> str:
    """The real source text of a top-level function.

    Preferred over matching against ``ast.dump`` for behaviour assertions: the
    dump renders strings single-quoted and reorders nothing else usefully, so a
    check written against it describes the serialiser rather than the code.
    """
    node = FUNCTIONS[name]
    return "\n".join(SOURCE.splitlines()[node.lineno - 1 : node.end_lineno])


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
    """Only the reference may load Torch; the native run has its own loader.

    The native side reads the CONVERTED safetensors through
    ``funasr_mlx.pipeline.load_model``. Reading the original ``.pt`` there would
    put two checkpoints and two backends in one process, and the comparison would
    no longer be about the shipped load path at all.
    """
    for helper in (
        "run_native",
        "_run_native_sample",
        "_native_frontend_config",
        "_native_conditioning_ids",
        "_native_paraformer_token_ids",
    ):
        roots = _imported_roots(helper)
        assert "torch" not in roots, f"{helper} imports Torch"
        assert "funasr" not in roots, f"{helper} imports the reference package"
    # The Torch reader exists for the reference only, and the native run must not
    # reach it even indirectly.
    assert "torch" in _imported_roots("_load_checkpoint_state")
    assert "_load_checkpoint_state" not in _dump("run_native")
    assert "_load_checkpoint_state" not in _dump("_run_native_sample")


def test_native_run_uses_the_shipping_pipeline_loader():
    """The validated artefact is the one the product loads, not a private path."""
    src = _source("run_native")
    assert "load_model" in src
    # A missing guard, a missing artefact and a float16 checkpoint are all
    # refused loudly rather than silently producing numbers to compare.
    for token in ("guarded_child_active", "native_checkpoint", "float32"):
        assert token in src, token
    sample = _source("_run_native_sample")
    # The CIF embeddings are the decoder INPUT and are never promoted to being
    # the decoder logits.
    assert "outputs.decoder_out" in sample
    assert '"pre_decoder": outputs.acoustic_embeds' in sample
    assert '"final_logits": outputs.acoustic_embeds' not in sample
    assert "ctc_greedy_decode" in sample
    # One encoder pass, not two: `model(...)` would recompute it.
    assert "model.encode(" in sample
    assert "= model(" not in sample


def test_reference_builds_on_meta_and_binds_the_official_values():
    """No weight storage is allocated, and nothing is loaded unverified."""
    build = _source("_build_funasr_reference")
    assert 'with torch.device("meta"):' in build
    load = _source("_load_reference_weights")
    assert "assign=True" in load
    assert "strict=True" in load
    # The shape check is stated here rather than left to a library internal.
    assert "inventory_report(" in load
    assert "shape_mismatch" in load
    # A silent precision change is refused, not performed.
    assert "torch.float32" in load
    assert "model.float()" not in load
    # Inference only: no autograd graph is retained over mmap'd weights.
    assert "requires_grad_" in load


def test_meta_non_parameters_are_rebuilt_by_source_code():
    """Tensors no checkpoint contains are rebuilt, never invented or skipped."""
    rules = _source("_meta_rebuild_rules")
    # Rules come from the imported FunASR classes, so a subclass inherits the
    # right rule and an unrelated same-named attribute does not borrow one.
    assert "funasr.models.transformer" in rules
    assert "isinstance" in _source("_rebuild_meta_tensors")
    for attribute in ('"pe"', '"tmp"'):
        assert attribute in rules
    # The rebuild calls the source's own method rather than recomputing the table
    # by hand.
    assert "extend_pe" in _source("_rebuild_positional_encoding_pe")
    assert "MAX_RECONSTRUCTED_ROWS" in _source("_rebuild_meta_tensors")
    # Anything still on meta stops the run instead of reading as a real tensor.
    assert "_meta_leftovers" in _source("_load_reference_weights")
    assert "still on the meta device" in _source("_load_reference_weights")
    assert "no source-defined" in _source("_rebuild_meta_tensors")
    # The one construction step most likely to lack a meta kernel is proven on a
    # 5000x8 instance of the real class before a 50-layer model is half-built,
    # and through the same rule the real tables will use.
    probe = _source("_probe_meta_positional_encoding")
    assert "funasr_embedding.PositionalEncoding(" in probe
    assert "_meta_rebuild_rules" in probe
    assert "isfinite" in probe
    assert "_probe_meta_positional_encoding" in _source("_build_funasr_reference")


class _FakeIds:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class _FakeRow:
    """Minimal stand-in for a 1-D tensor row: slice -> argmax -> tolist."""

    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return _FakeRow(self.values[index])

    def argmax(self, dim=-1):
        assert dim == -1
        return _FakeIds(self.values)


class _FakeMatrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, index):
        if isinstance(index, tuple):
            batch, window = index
            return _FakeRow(self.rows[batch][window])
        return _FakeMatrix(self.rows[index])


def test_ctc_greedy_ids_collapses_duplicates_before_dropping_blanks():
    """``inference`` order: argmax -> unique_consecutive -> drop blanks.

    Duplicates collapse BEFORE blanks are removed, so a blank between two equal
    ids does not merge them. Getting that order wrong changes the text without
    changing a single logit.
    """
    row = _FakeRow([5, 5, 0, 0, 7, 7, 0, 3])
    assert alignment._ctc_greedy_ids(row, 8, 0) == [5, 7, 3]
    # The valid length truncates first: the trailing [0, 3] is not audio.
    assert alignment._ctc_greedy_ids(row, 6, 0) == [5, 7]
    assert alignment._ctc_greedy_ids(_FakeRow([0, 0, 0]), 3, 0) == []


def test_paraformer_greedy_ids_drops_sos_eos_and_blank():
    """``inference`` slices by the rounded token length, then filters the ids."""
    from types import SimpleNamespace

    model = SimpleNamespace(eos=2, sos=2, blank_id=0)
    matrix = _FakeMatrix([[9, 2, 5, 5, 0]])
    assert alignment._paraformer_greedy_ids(model, matrix, [3]) == [9, 5]
    # The length is the rounded CIF token count, so it decides how much of the
    # argmax is real output -- and only sos/eos/blank are ever dropped.
    assert alignment._paraformer_greedy_ids(model, matrix, [5]) == [9, 5, 5]
    # A token length beyond the row is clamped, not an index error.
    assert alignment._paraformer_greedy_ids(model, matrix, [9]) == [9, 5, 5]


def test_overlength_is_measured_and_refused_on_both_sides():
    """The bound is checked against the decoded array, on both backends."""
    for name in ("_run_reference_sample", "_run_native_sample"):
        src = _source(name)
        assert "_reject_overlength(" in src, name
        assert '"audio_samples"' in src, name
        assert '"fbank_frames"' in src, name
    reject = _source("_reject_overlength")
    assert "truncated" in reject
    assert "SAMPLE_RATE" in reject
    variant = {"max_seconds": 20.0, "key": "paraformer-zh"}
    alignment._reject_overlength("a", 16_000 * 20, variant)
    for samples in (16_000 * 21, 16_000 * 60):
        try:
            alignment._reject_overlength("a", samples, variant)
        except alignment.AlignmentError as error:
            assert "refused" in str(error)
        else:  # pragma: no cover
            raise AssertionError("over-length audio was accepted")


def test_native_artefact_is_named_explicitly_and_must_exist(tmp_path):
    """Two artefacts, both explicit; nothing downloaded or inferred."""
    assets = tmp_path / "models" / "paraformer"
    assets.mkdir(parents=True)
    (assets / "config.yaml").write_text(
        "init_param: model.pt\n"
        "model: Paraformer\n"
        "frontend: WavFrontend\n"
        "tokenizer: CharTokenizer\n"
        "token_list: tokens.json\n"
    )
    (assets / "configuration.json").write_text(
        json.dumps({"init_param": "model.pt", "cmvn_file": "am.mvn"})
    )
    manifest = _manifest(tmp_path, durations=(3.0,))
    native = tmp_path / "converted" / "model.safetensors"
    native.parent.mkdir()
    native.write_bytes(b"")

    spec = alignment.build_alignment_spec(
        manifest,
        "paraformer-zh",
        tmp_path,
        assets_root=tmp_path,
        native_checkpoint=native,
        max_samples=1,
    )
    assert spec["variant"]["native_checkpoint"] == str(native)
    assert spec["variant"]["checkpoint"] == str(assets / "model.pt")
    conversion = spec["variant"]["conversion"]
    assert conversion["model"] == "paraformer"
    assert conversion["dtype"] == "float32"
    assert conversion["requires_guarded_child"] is True
    assert conversion["source"] == str(assets / "model.pt")
    assert spec["reference"]["init"] == "meta"
    # json.dumps of the whole spec: the plan has to be reviewable in a diff.
    assert json.loads(json.dumps(spec)) == spec

    # A path that does not exist is refused at build time, with the fix named.
    try:
        alignment.build_alignment_spec(
            manifest,
            "paraformer-zh",
            tmp_path,
            assets_root=tmp_path,
            native_checkpoint=tmp_path / "absent.safetensors",
            max_samples=1,
        )
    except alignment.AlignmentError as error:
        assert "not found" in str(error)
        assert "convert_checkpoint" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a missing native checkpoint was accepted")

    # An unknown init device is refused rather than silently defaulted.
    try:
        alignment.build_alignment_spec(
            manifest,
            "paraformer-zh",
            tmp_path,
            assets_root=tmp_path,
            reference_init="cuda",
            max_samples=1,
        )
    except alignment.AlignmentError as error:
        assert "reference_init" in str(error)
    else:  # pragma: no cover
        raise AssertionError("an unknown reference init device was accepted")


def test_inventory_report_checks_shapes_as_well_as_names():
    shapes_model = {"a": (2, 3), "b": (4,)}
    shapes_checkpoint = {"a": (2, 3), "b": (5,)}
    report = alignment.inventory_report(
        shapes_model,
        shapes_checkpoint,
        label="t",
        model_shapes=shapes_model,
        checkpoint_shapes=shapes_checkpoint,
    )
    assert report["verdict"] == "mismatch"
    assert report["missing"] == [] and report["unexpected"] == []
    assert report["shape_mismatch"] == [{"name": "b", "model": [4], "checkpoint": [5]}]
    # A name-only call stays usable and simply has no shape opinion.
    plain = alignment.inventory_report(shapes_model, shapes_checkpoint, label="t")
    assert plain["shape_mismatch"] == []


def test_compare_flags_inventory_and_loading_problems():
    """A run that cannot show how it loaded is not a clean comparison."""
    base = {
        "schema": "funasr-mlx-alignment-run/3",
        "variant": {"key": "paraformer-zh"},
        "inventory": {"verdict": "ok", "missing": [], "unexpected": [], "shape_mismatch": []},
        "loading": {"dtype": "float32", "keys_checked": 4},
        "samples": [],
    }
    assert alignment.compare_alignment(base, base)["clean"] is True

    missing_inventory = {k: v for k, v in base.items() if k != "inventory"}
    kinds = {f["kind"] for f in alignment.compare_alignment(missing_inventory, base)["findings"]}
    assert "inventory_missing" in kinds

    no_loading = {k: v for k, v in base.items() if k != "loading"}
    kinds = {f["kind"] for f in alignment.compare_alignment(no_loading, base)["findings"]}
    assert "loading_disposition_missing" in kinds

    bad = {
        **base,
        "inventory": {
            "verdict": "mismatch",
            "missing": ["x"],
            "unexpected": [],
            "shape_mismatch": [],
        },
    }
    kinds = {f["kind"] for f in alignment.compare_alignment(bad, base)["findings"]}
    assert "inventory_not_exact" in kinds

    float16_native = {**base, "loading": {"dtype": "float16", "keys_checked": 4}}
    kinds = {f["kind"] for f in alignment.compare_alignment(base, float16_native)["findings"]}
    assert "loading_disposition_differ" in kinds

    # A mismatched feature count is a geometry problem, not a numeric one.
    left = {**base, "samples": [{"key": "a", "audio_samples": 10, "fbank_frames": 3}]}
    right = {**base, "samples": [{"key": "a", "audio_samples": 12, "fbank_frames": 4}]}
    kinds = {f["kind"] for f in alignment.compare_alignment(left, right)["findings"]}
    assert kinds == {"audio_length_disagreement", "fbank_frame_disagreement"}


def test_whisper_cannot_be_aligned_end_to_end_and_says_so():
    """Refused with a reason, not an empty run that reads like a pass."""
    for name in ("run_reference", "run_native"):
        src = _source(name)
        assert "per-sample stage pipeline" in src, name
        assert "paraformer-zh" in src, name


def test_both_frontends_are_built_independently_from_one_config():
    """Each backend gets a real frontend, configured from the same spec values."""
    assert "WavFrontend" in _dump("_resolve_wav_frontend")
    assert "FbankConfig" in _dump("run_native") or "FbankConfig" in _dump("_native_frontend_config")
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
    assert "model_conf" in _dump("build_model_conf")
    for name in ("encoder_conf", "decoder_conf", "predictor_conf", "ctc_conf"):
        configuration = {"model_conf": {"ctc_weight": 0.0}, name: {"size": 4}}
        assert alignment.build_model_conf(configuration)[name] == {"size": 4}
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
    directory = tmp_path / "models" / "paraformer"
    directory.mkdir(parents=True)
    (directory / "configuration.json").write_text(
        json.dumps(
            {
                "file_path_metas": {
                    "init_param": "model.pt",
                    "frontend_conf": {"cmvn_file": "am.mvn"},
                }
            }
        )
    )
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


def test_build_model_conf_reproduces_automodel_merge():
    """The whole config must be merged over model_conf, as AutoModel does.

    Forwarding only model_conf -- or only the *_conf keys -- drops the top-level
    encoder/decoder/predictor NAME strings and builds a model with no encoder.
    """
    config = {
        "model": "Paraformer",
        "model_conf": {"ctc_weight": 0.0, "lsm_weight": 0.1},
        "encoder": "SANMEncoder",
        "decoder": "ParaformerSANMDecoder",
        "predictor": "CifPredictorV2",
        "encoder_conf": {"output_size": 512},
        "frontend_conf": {"n_mels": 80},
    }
    merged = alignment.build_model_conf(config)
    assert merged["encoder"] == "SANMEncoder"
    assert merged["decoder"] == "ParaformerSANMDecoder"
    assert merged["predictor"] == "CifPredictorV2"
    assert merged["ctc_weight"] == 0.0
    # Whole-config values win over model_conf, exactly like deep_update order.
    assert merged["encoder_conf"] == {"output_size": 512}


def test_resolve_asset_paths_reads_configuration_json_not_config_yaml(tmp_path):
    """config.yaml ships cmvn_file: null; only configuration.json names it."""
    (tmp_path / "am.mvn").write_text("cmvn")
    (tmp_path / "config.yaml").write_text("frontend_conf:\n  cmvn_file: null\n")
    (tmp_path / "tokens.json").write_text('["<blank>", "<s>"]')
    (tmp_path / "model.pt").write_bytes(b"x")
    (tmp_path / "configuration.json").write_text(
        json.dumps(
            {
                "file_path_metas": {
                    "init_param": "model.pt",
                    "config": "config.yaml",
                    "frontend_conf": {"cmvn_file": "am.mvn"},
                    "tokenizer_conf": {"token_list": "tokens.json"},
                }
            }
        )
    )
    assets = alignment.resolve_asset_paths(tmp_path / "config.yaml")
    assert assets["cmvn_file"].endswith("am.mvn")
    assert assets["token_list"].endswith("tokens.json")
    assert assets["init_param"].endswith("model.pt")
    assert assets["bpemodel"] is None

    with pytest.raises(alignment.AlignmentError):
        alignment.resolve_asset_paths(tmp_path)  # a directory is not a config file


def test_sensevoice_and_paraformer_derive_input_size_from_the_frontend():
    """input_size=560 is derived from WavFrontend.output_size(), not configured.

    WavFrontend.output_size() is n_mels * lfr_m; the Mel23 variant registered
    elsewhere uses a different formula, so the wrong class would silently give a
    different width.
    """
    variant = {v.key: v for v in alignment.SUPPORTED}
    assert (
        variant["sensevoice-small"].frontend["n_mels"]
        * variant["sensevoice-small"].frontend["lfr_m"]
        == 560
    )
    assert variant["sensevoice-small"].frontend["n_mels"] == 80
    # The registry module must actually host the configured encoder.
    assert (
        variant["sensevoice-small"].reference_import in variant["sensevoice-small"].registry_imports
    )


def test_paraformer_registry_imports_cover_its_configured_names():
    """encoder/decoder/predictor are name strings resolved through the registry."""
    paraformer = {v.key: v for v in alignment.SUPPORTED}["paraformer-zh"]
    joined = " ".join(paraformer.registry_imports)
    assert "sanm.encoder" in joined
    assert "paraformer.decoder" in joined
    assert "paraformer.cif_predictor" in joined
