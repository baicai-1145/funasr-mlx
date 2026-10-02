"""Small differential tests execute selected exact reference definitions only."""

import ast
import math
import os
from pathlib import Path

import numpy as np
import pytest

REFERENCE = Path(
    os.environ.get("FUNASR_REFERENCE_ROOT", str(Path(__file__).resolve().parents[2] / "funasr"))
)


def reference_definitions(relative_path, names, extra_globals=None):
    import torch

    filename = REFERENCE / relative_path
    if not filename.is_file():
        pytest.skip("FunASR reference checkout is unavailable")
    tree = ast.parse(filename.read_text(), filename=str(filename))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in definitions} == set(names)
    from dataclasses import dataclass
    from typing import Dict, Iterable, List, Optional, Tuple

    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "Tensor": torch.Tensor,
        "Dict": Dict,
        "Iterable": Iterable,
        "Optional": Optional,
        "Tuple": Tuple,
        "List": List,
        "dataclass": dataclass,
        "math": math,
        "np": np,
        "decode_function": None,
        "detect_language_function": None,
        "transcribe_function": None,
        "SDPA_AVAILABLE": True,
        "scaled_dot_product_attention": torch.nn.functional.scaled_dot_product_attention,
    }
    namespace.update(extra_globals or {})
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_cif_matches_exact_reference(batch):
    import mlx.core as mx
    import torch
    from funasr_mlx.models.paraformer import cif_v1

    reference = reference_definitions(
        "funasr/models/paraformer/cif_predictor.py", ["cif_wo_hidden_v1", "cif_v1"]
    )
    hidden = np.arange(batch * 6 * 3, dtype=np.float32).reshape(batch, 6, 3) / 7
    alphas = np.broadcast_to(
        np.array([0.5, 0.25, 0.5, 0.25, 0.5, 0.125], np.float32), (batch, 6)
    ).copy()
    expected, fires = reference["cif_v1"](torch.from_numpy(hidden), torch.from_numpy(alphas), 1.0)
    actual, actual_fires = cif_v1(mx.array(hidden), mx.array(alphas), 1.0)
    mx.eval(actual, actual_fires)
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual_fires), fires.numpy(), atol=1e-6)


@pytest.mark.parametrize("lengths", [[7, 7], [7, 4], [3, 1]])
def test_sanm_attention_matches_exact_reference(lengths):
    import mlx.core as mx
    import torch
    from funasr_mlx.models.sanm import MultiHeadedAttentionSANM, load_strict

    reference = reference_definitions(
        "funasr/models/sanm/attention.py", ["MultiHeadedAttentionSANM"]
    )
    torch.manual_seed(123)
    golden = reference["MultiHeadedAttentionSANM"](2, 8, 8, 0.0, 3).eval()
    actual_model = MultiHeadedAttentionSANM(2, 8, 8, 0.0, 3)
    state = {name: value.detach().numpy() for name, value in golden.state_dict().items()}
    load_strict(actual_model, state, convert=True)
    samples = np.random.default_rng(31).normal(0, 0.1, (2, 7, 8)).astype(np.float32)
    mask = (np.arange(7)[None, None, :] < np.asarray(lengths)[:, None, None]).astype(np.float32)
    with torch.no_grad():
        expected = golden(torch.from_numpy(samples), torch.from_numpy(mask)).numpy()
    actual = actual_model(mx.array(samples), mx.array(mask))
    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-4, atol=1e-5)
