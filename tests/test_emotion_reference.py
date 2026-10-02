"""Official fairseq/AudioEncoder construction comparisons, tiny shapes only."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_reference_parity import reference_definitions

SOURCES = Path(__file__).resolve().parents[1] / ".tmp/emotion2vec-reference"


def source_globals():
    import torch

    namespace = reference_definitions(str(SOURCES / "transpose_last.py"), ["TransposeLast"])
    namespace.update(reference_definitions(str(SOURCES / "same_pad.py"), ["SamePad"]))
    namespace.update(reference_definitions(str(SOURCES / "layer_norm.py"), ["Fp32LayerNorm"]))
    namespace.update({"torch": torch, "nn": torch.nn, "LayerNorm": torch.nn.LayerNorm})
    return namespace


def assign_source_weights(native, golden):
    import mlx.core as mx

    values = []
    for name, tensor in golden.state_dict().items():
        value = tensor.detach().numpy()
        if value.ndim == 3:
            value = value.transpose(0, 2, 1)
        values.append((name, mx.array(value)))
    native.load_weights(values, strict=True)


@pytest.mark.parametrize("batch", [1, 2])
def test_feature_cnn_matches_exact_fairseq_source(batch):
    import mlx.core as mx
    import torch
    from funasr_mlx.models.emotion2vec import ConvFeatureExtraction

    namespace = reference_definitions(
        str(SOURCES / "wav2vec2.py"), ["ConvFeatureExtractionModel"], source_globals()
    )
    specification = [(8, 5, 2), (8, 3, 2), (8, 3, 2)]
    torch.manual_seed(42)
    golden = namespace["ConvFeatureExtractionModel"](specification, mode="layer_norm").eval()
    native = ConvFeatureExtraction(specification, eps=1e-5, conv_bias=False)
    assign_source_weights(native, golden)
    samples = np.random.default_rng(51).normal(0, 0.1, (batch, 400)).astype(np.float32)
    with torch.no_grad():
        expected = golden(torch.from_numpy(samples)).numpy()
    actual = native(mx.array(samples))
    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-4, atol=1e-5)


def official_positional_encoder(depth, width):
    filename = SOURCES / "audio.py"
    if not filename.exists():
        pytest.skip("Official emotion2vec sources are unavailable")
    tree = ast.parse(filename.read_text())
    encoder = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AudioEncoder"
    )
    constructor = next(
        node
        for node in encoder.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    names = {"num_pos_layers", "k", "positional_encoder"}
    statements = [
        node
        for node in constructor.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
    ]
    assert len(statements) == 3
    namespace = source_globals()
    namespace.update(
        {
            "modality_cfg": SimpleNamespace(
                conv_pos_depth=depth, conv_pos_width=width, conv_pos_groups=4
            ),
            "embed_dim": 8,
        }
    )
    module = ast.Module(body=statements, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)
    return namespace["positional_encoder"].eval()


@pytest.mark.parametrize("width,length", [(12, 5), (12, 9), (14, 7)])
def test_positional_encoder_matches_official_construction(width, length):
    import mlx.core as mx
    import torch
    from funasr_mlx.models.emotion2vec import PositionalEncoder

    torch.manual_seed(12)
    golden = official_positional_encoder(2, width)
    native = PositionalEncoder(8, 2, max(3, width // 2), 4, 1e-5)
    assign_source_weights(native, golden)
    inputs = np.random.default_rng(25).normal(0, 0.1, (2, length, 8)).astype(np.float32)
    with torch.no_grad():
        expected = golden(torch.from_numpy(inputs)).numpy()
    actual = native(mx.array(inputs))
    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-4, atol=1e-5)
