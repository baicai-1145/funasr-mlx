"""Pure manifest and backend-import contracts, not numerical Whisper acceptance."""

import ast
import json
from pathlib import Path

import pytest

from tools.alignment_whisper import load_inputs


def manifest(tmp_path, **overrides):
    sample = {
        "key": "clip",
        "audio": "clip.wav",
        "duration_seconds": 1.0,
        "corpus": "librispeech-test-clean",
        "language": "en",
        "sha256": "a" * 64,
    }
    sample.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"samples": [sample]}))
    return path


def test_manifest_requires_audio_hash_and_corpus(tmp_path):
    with pytest.raises(ValueError, match="provenance"):
        load_inputs([manifest(tmp_path, sha256="")])


def test_overlong_input_is_rejected_before_backend_import(tmp_path):
    with pytest.raises(ValueError, match="bound"):
        load_inputs([manifest(tmp_path, duration_seconds=30.01)])


def test_duplicate_input_keys_are_rejected(tmp_path):
    path = manifest(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        load_inputs([path, path])


def test_shared_provenance_is_preserved(tmp_path):
    (sample,) = load_inputs([manifest(tmp_path)])
    assert sample["sha256"] == "a" * 64
    assert sample["corpus"] == "librispeech-test-clean"


def test_native_uses_public_causal_logits_entry_point():
    path = Path(__file__).resolve().parents[1] / "tools/alignment_whisper.py"
    tree = ast.parse(path.read_text())
    native = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_native"
    )
    calls = [
        node.func.attr
        for node in ast.walk(native)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "model"
    ]
    assert "logits" in calls
    assert "decoder" not in calls


def test_backends_have_separate_runtime_imports():
    path = Path(__file__).resolve().parents[1] / "tools/alignment_whisper.py"
    tree = ast.parse(path.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, required, banned in (
        ("run_reference", "torch", "mlx"),
        ("run_native", "mlx", "torch"),
    ):
        imports = set()
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add((node.module or "").split(".")[0])
        assert required in imports
        assert banned not in imports
