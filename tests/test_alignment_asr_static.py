"""Static checks for the alignment adapter.

These run on a machine with **neither MLX nor Torch installed**, which is the
point: the adapter's planning and comparison logic must be usable, reviewable and
diffable without pulling in either backend. They are AST and pure-Python checks
only -- no numeric work, no weights, no network.
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


def _module_source() -> ast.Module:
    return ast.parse((TOOLS / "alignment_asr.py").read_text())


def _top_level_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_backend_is_imported_at_module_scope():
    """Importing the adapter must not drag in MLX or Torch.

    ``build`` and ``compare`` run on machines that have neither backend, and the
    reference/native isolation only holds if neither is resident before its
    backend is explicitly selected.
    """
    imports = _top_level_imports(_module_source())
    assert "mlx" not in imports, "MLX must only be imported inside a backend function"
    assert "torch" not in imports, "Torch must only be imported inside a backend function"


def test_backends_are_isolated_into_separate_entry_points():
    """Reference and native must not share a runner.

    They have to run in separate processes so neither backend is resident while
    the other is loading weights.
    """
    source = (TOOLS / "alignment_asr.py").read_text()
    tree = _module_source()
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("run_reference", "run_native"):
        assert name in functions, f"{name} must exist as its own entry point"

    reference = functions["run_reference"]
    native = functions["run_native"]
    reference_body = ast.dump(reference)
    native_body = ast.dump(native)
    assert "mlx" not in reference_body, "the reference runner must never touch MLX"
    assert "torch" not in native_body, "the native runner must never touch Torch"
    assert "run_native" not in reference_body
    assert "run_reference" not in native_body
    assert "def run_reference" in source and "def run_native" in source


def test_reference_path_never_reimplements_source_math():
    """The reference must instantiate the real upstream class.

    A hand-rolled reference would be a second implementation to be wrong, and
    agreement between two reimplementations proves nothing about either.
    """
    for variant in alignment.SUPPORTED:
        if variant.family == "whisper":
            continue
        assert variant.reference_import.startswith("funasr."), variant.key
        assert variant.reference_class, variant.key

    tree = _module_source()
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    importer = functions["_import_reference"]
    dumped = ast.dump(importer)
    assert "importlib" in dumped, "the reference must be reached through importlib"
    assert (TOOLS / "alignment_asr.py").read_text().count("def _import_reference") == 1


def test_unsupported_variants_are_listed_with_reasons():
    """A refusal must explain itself, and large/turbo must not be quietly aligned."""
    assert alignment.UNSUPPORTED, "expected an explicit unsupported list"
    for key, reason in alignment.UNSUPPORTED.items():
        assert reason.strip(), f"{key} needs a stated reason"
    for key in ("whisper-large", "whisper-large-v3-turbo"):
        assert key in alignment.UNSUPPORTED

    with pytest.raises(alignment.ManifestError) as excinfo:
        alignment.resolve_variant("whisper-large")
    assert "not aligned" in str(excinfo.value)

    with pytest.raises(alignment.ManifestError):
        alignment.resolve_variant("not-a-real-variant")


def test_only_bounded_whisper_is_offered():
    """The tiny checkpoint is a bounded proof; it must not stand in for larger."""
    whisper_variants = [v for v in alignment.SUPPORTED if v.family == "whisper"]
    assert [v.key for v in whisper_variants] == ["whisper-tiny"]
    assert "tiny" in whisper_variants[0].checkpoint
    assert "bounded" in whisper_variants[0].notes.lower()


def test_spec_is_json_serialisable_and_free_of_backend_objects(tmp_path):
    spec = alignment.build_alignment_spec(
        _manifest(tmp_path),
        "paraformer-zh",
        tmp_path,
        assets_root=tmp_path,
    )
    # Round-trips through JSON with nothing exotic in it.
    assert json.loads(json.dumps(spec)) == spec
    assert spec["dtype"] == "float32"
    assert spec["audio"]["batch_size"] == 1
    assert spec["audio"]["fixed_order"] == ["a", "b"]
    assert spec["variant"]["key"] == "paraformer-zh"
    assert spec["unsupported_variants"]["whisper-large"]


def _manifest(root: Path) -> Path:
    first = root / "a.wav"
    second = root / "b.wav"
    for path in (first, second):
        # The spec builder only checks existence; no backend ever reads these.
        path.write_bytes(b"placeholder")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {"key": "a", "audio": str(first), "duration_seconds": 3.0, "transcript": "hi"},
                    {"key": "b", "audio": str(second), "duration_seconds": 2.0},
                ]
            }
        )
    )
    return manifest


def test_samples_over_the_variant_bound_are_refused_not_truncated(tmp_path):
    """One backend truncating and the other not would invalidate the comparison."""
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"placeholder")
    manifest = tmp_path / "long_manifest.json"
    manifest.write_text(
        json.dumps({"samples": [{"key": "long", "audio": str(audio), "duration_seconds": 900.0}]})
    )
    with pytest.raises(alignment.ManifestError) as excinfo:
        alignment.build_alignment_spec(manifest, "paraformer-zh", tmp_path, assets_root=tmp_path)
    assert "exceeds" in str(excinfo.value)


def test_inventory_reports_both_directions():
    """Missing *and* unexpected keys must both surface.

    Checking only the missing direction is how a renamed or dropped head gets
    silently skipped.
    """
    report = alignment.inventory_report(["a", "b", "c"], {"b": 1, "c": 2}, label="x")
    assert report["missing"] == ["a"]
    assert report["unexpected"] == []
    assert report["verdict"] == "mismatch"
    assert report["exact_match"] is False

    both = alignment.inventory_report(["a"], {"a": 1, "z": 2})
    assert both["missing"] == []
    assert both["unexpected"] == ["z"]

    clean = alignment.inventory_report(["a", "b"], {"b": 0, "a": 0})
    assert clean["verdict"] == "ok"
    assert clean["model_key_count"] == 2
    assert clean["checkpoint_key_count"] == 2


def test_compare_flags_token_stage_and_dtype_disagreement():
    reference = {
        "variant": {"key": "paraformer-zh"},
        "dtype": "float32",
        "samples": [
            {
                "key": "a",
                "token_ids": [1, 2],
                "stages": [{"stage": "encoder", "shape": [1, 5, 8], "dtype": "float32"}],
            }
        ],
    }
    native = {
        "variant": {"key": "paraformer-zh"},
        "dtype": "float32",
        "samples": [
            {
                "key": "a",
                "token_ids": [1, 3],
                "stages": [{"stage": "encoder", "shape": [1, 6, 8], "dtype": "float16"}],
            }
        ],
    }
    report = alignment.compare_alignment(reference, native)
    kinds = {finding["kind"] for finding in report["findings"]}
    assert "token_ids_differ" in kinds
    assert "stage_shape_mismatch" in kinds
    assert "stage_dtype_mismatch" in kinds
    assert report["clean"] is False

    clean = alignment.compare_alignment(reference, reference)
    assert clean["clean"] is True
    assert clean["finding_count"] == 0
    assert "no tolerance" in clean["note"]


def test_compare_flags_a_mismatched_variant_or_sample_set():
    left = {"variant": {"key": "paraformer-zh"}, "dtype": "float32", "samples": [{"key": "a"}]}
    right = {"variant": {"key": "sensevoice-small"}, "dtype": "float32", "samples": [{"key": "b"}]}
    kinds = {f["kind"] for f in alignment.compare_alignment(left, right)["findings"]}
    assert "variant_mismatch" in kinds
    assert "sample_set_mismatch" in kinds


def test_saved_tensors_are_bounded_and_truncation_is_recorded():
    """A partial activation must never be mistaken for a full one."""
    import numpy as np

    small = np.zeros((2, 3), np.float32)
    value, truncated = alignment._bounded(small)
    assert truncated is False
    assert value.size == small.size

    oversized = np.zeros(alignment.MAX_SAVED_ELEMENTS + 10, np.float32)
    value, truncated = alignment._bounded(oversized)
    assert truncated is True
    assert value.size == alignment.MAX_SAVED_ELEMENTS

    record = alignment._stage_record("encoder", small)
    assert record["shape"] == [2, 3]
    assert record["dtype"] == "float32"
    assert record["truncated"] is False
