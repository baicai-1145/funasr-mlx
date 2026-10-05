"""W4A16 quantization: converter layout, loader wiring and round-trip numerics.

Everything here runs on tiny synthetic tensors -- no checkpoint, no network.
The point of the suite is that the two halves of the quantization contract
(the converter's tensor-name predicate and the model's module predicate) agree,
and that a quantized checkpoint round-trips through ``pipeline.load_model``
bit-for-bit against the in-process ``mx.quantize`` result.
"""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from funasr_mlx.checkpoints import (
    IDENTITY_SOURCE_RULE,
    QUANTIZED_SUFFIXES,
    convert_checkpoint,
    quantize_bf16_weight,
)
from funasr_mlx.models.llm.quant import (
    DEFAULT_BITS,
    DEFAULT_GROUP_SIZE,
    NON_LINEAR_WEIGHT_NAMES,
    QuantizationSpec,
    apply_quantization,
)
from funasr_mlx.pipeline import create_model


def _spec(**kwargs) -> dict:
    base = {
        "bits": DEFAULT_BITS,
        "group_size": DEFAULT_GROUP_SIZE,
        "mode": "affine",
        "targets": ["model.layers."],
    }
    base.update(kwargs)
    return base


# --------------------------------------------------------------------------- #
# tensor-name predicate
# --------------------------------------------------------------------------- #
def test_quantizes_only_selected_linear_weights():
    spec = QuantizationSpec.from_dict(_spec())
    assert spec.quantizes_tensor("model.layers.0.mlp.down_proj.weight", (3072, 1024))
    assert spec.quantizes_tensor("model.layers.27.self_attn.q_proj.weight", (2048, 1024))
    # Outside the target prefix: the audio tower is untouched by default.
    assert not spec.quantizes_tensor("audio_tower.layers.0.fc1.weight", (3584, 896))
    # Never the embeddings or the head, even inside a selected prefix.
    for name in ("model.layers.0.embed_tokens.weight", "model.lm_head.weight"):
        assert not spec.quantizes_tensor(name, (1024, 1024))
    # Rank-2 only: a conv or norm weight is not a Linear.
    assert not spec.quantizes_tensor("model.layers.0.mlp.conv.weight", (8, 3, 3, 4))
    assert not spec.quantizes_tensor("model.layers.0.norm.weight", (1024,))
    # Not a weight at all.
    assert not spec.quantizes_tensor("model.layers.0.mlp.down_proj.bias", (1024,))


def test_group_size_must_divide_the_input_dimension():
    spec = QuantizationSpec.from_dict(_spec(group_size=64))
    assert spec.quantizes_tensor("model.layers.0.mlp.up_proj.weight", (3072, 1024))
    assert not spec.quantizes_tensor("model.layers.0.mlp.up_proj.weight", (3072, 1056))
    # 32 divides what 64 does not; the declaration drives the decision.
    assert QuantizationSpec.from_dict(_spec(group_size=32)).quantizes_tensor(
        "model.layers.0.mlp.up_proj.weight", (3072, 1056)
    )


@pytest.mark.parametrize(
    "data,match",
    [
        ({"bits": 7, "group_size": 64, "targets": ["a."]}, "bit width"),
        ({"bits": 4, "group_size": 16, "targets": ["a."]}, "group size"),
        ({"bits": 4, "group_size": 64, "mode": "int4", "targets": ["a."]}, "mode"),
        ({"bits": 4, "group_size": 64, "targets": []}, "targets"),
        ("4", "mapping"),
    ],
)
def test_invalid_quantization_config_is_rejected(data, match):
    with pytest.raises(ValueError, match=match):
        QuantizationSpec.from_dict(data)


# --------------------------------------------------------------------------- #
# module predicate -- must mirror the tensor-name predicate exactly
# --------------------------------------------------------------------------- #
class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(128, 128, bias=False)
        self.mlp = nn.Linear(128, 256, bias=False)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(64, 128)
        self.layers = [_Block() for _ in range(2)]
        self.norm = nn.RMSNorm(128)
        self.lm_head = nn.Linear(128, 64, bias=False)


def test_apply_quantization_matches_the_declared_targets():
    model = _Tiny()
    result = apply_quantization(model, _spec(targets=["layers."]))
    assert result is not None
    assert isinstance(model.layers[0].mlp, nn.QuantizedLinear)
    assert isinstance(model.layers[0].self_attn, nn.QuantizedLinear)
    # Not a Linear -> left alone, and never reported as "skipped Linear".
    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.norm, nn.RMSNorm)
    assert isinstance(model.lm_head, nn.Linear)
    assert result.summary()["linear_layers_quantized"] == 4
    assert model.lm_head.weight.dtype == mx.float32


def test_apply_quantization_is_none_without_a_section():
    model = _Tiny()
    assert apply_quantization(model, None) is None
    assert isinstance(model.layers[0].mlp, nn.Linear)


def test_quantized_module_parameters_are_the_three_expected_names():
    from mlx.utils import tree_flatten

    model = _Tiny()
    apply_quantization(model, _spec(targets=["layers."]))
    names = {name for name, _ in tree_flatten(model.parameters())}
    for layer in (0, 1):
        for leaf in ("self_attn", "mlp"):
            for suffix in QUANTIZED_SUFFIXES:
                assert f"layers.{layer}.{leaf}{suffix}" in names
    assert "lm_head.weight" in names
    assert "embed_tokens.weight" in names


def test_non_linear_weight_names_stay_unquantized():
    """The converter's exclusion list and the model predicate must agree."""
    spec = QuantizationSpec.from_dict(_spec(targets=["layers."]))
    for marker in NON_LINEAR_WEIGHT_NAMES:
        name = f"layers.0.{marker}.weight"
        assert not spec.quantizes_tensor(name, (128, 128))
    model = _Tiny()
    model.layers[0].lm_head = nn.Linear(128, 64, bias=False)
    apply_quantization(model, _spec(targets=["layers."]))
    assert isinstance(model.layers[0].lm_head, nn.Linear)


# --------------------------------------------------------------------------- #
# packing round-trip
# --------------------------------------------------------------------------- #
def _bf16_carrier(weight: mx.array) -> np.ndarray:
    return np.array(weight.view(mx.uint16))


@pytest.mark.parametrize("group_size", [32, 64])
def test_quantize_bf16_weight_round_trips_through_mlx(group_size):
    weight = (mx.random.normal((64, 128)) * 0.05).astype(mx.bfloat16)
    packed = quantize_bf16_weight(
        "layers.0.mlp.weight",
        _bf16_carrier(weight),
        code="BF16",
        bits=4,
        group_size=group_size,
    )
    assert set(packed) == {
        "layers.0.mlp.weight",
        "layers.0.mlp.scales",
        "layers.0.mlp.biases",
    }
    assert packed["layers.0.mlp.weight"].dtype == np.uint32
    assert packed["layers.0.mlp.scales"].dtype == np.uint16
    # Group scales live in the source dtype: 128 inputs / group_size groups.
    assert packed["layers.0.mlp.scales"].shape == (64, 128 // group_size)
    x = mx.random.normal((2, 128)).astype(mx.bfloat16)
    got = mx.quantized_matmul(
        x,
        mx.array(packed["layers.0.mlp.weight"]),
        scales=mx.array(packed["layers.0.mlp.scales"]).view(mx.bfloat16),
        biases=mx.array(packed["layers.0.mlp.biases"]).view(mx.bfloat16),
        transpose=True,
        group_size=group_size,
        bits=4,
    )
    reference = mx.quantized_matmul(
        x,
        *mx.quantize(weight, group_size=group_size, bits=4),
        transpose=True,
        group_size=group_size,
        bits=4,
    )
    mx.eval(got, reference)
    assert mx.array_equal(got, reference)


def test_quantize_rejects_a_wide_source_dtype():
    with pytest.raises(ValueError, match="16-bit source weight"):
        quantize_bf16_weight("w", np.zeros((4, 64), np.float32), code="F32", bits=4, group_size=64)


# --------------------------------------------------------------------------- #
# end-to-end: convert -> load_model -> strict parameter binding
# --------------------------------------------------------------------------- #
TINY_QWEN3_ASR = {
    "thinker_config": {
        "audio_config": {
            "d_model": 64,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 128,
            "downsample_hidden_size": 4,
            "output_dim": 64,
            "n_mels": 8,
            "n_window": 2,
            "max_source_positions": 64,
        },
        "text_config": {
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 64,
            "tie_word_embeddings": True,
        },
    }
}


def _write_native_bf16_source(directory, family: str, config: dict) -> dict:
    """Build a tiny model, write its bf16 parameters as a native checkpoint."""
    from mlx.utils import tree_flatten

    from funasr_mlx.checkpoint_stream import write_shard
    from funasr_mlx.checkpoints import DERIVED_NATIVE_PARAMETERS

    directory.mkdir(parents=True, exist_ok=True)
    model = create_model(family, config)
    parameters = dict(tree_flatten(model.parameters()))
    # Parameters the port derives at construction are not on disk (the strict
    # loader deletes them from its expectation set), so the fixture must too.
    for name in DERIVED_NATIVE_PARAMETERS.get(family, ()):
        parameters.pop(name)
    assert parameters and sum(value.size for value in parameters.values()) < 1_000_000
    tensors = {name: _bf16_carrier(value.astype(mx.bfloat16)) for name, value in parameters.items()}
    write_shard(
        directory / "model-00001.safetensors",
        tensors,
        metadata={"format": "funasr-mlx", "model": family},
        code="BF16",
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"format": "funasr-mlx", "model": family, "dtype": "bfloat16"},
                "weight_map": {name: "model-00001.safetensors" for name in tensors},
            }
        )
    )
    return parameters


def _bf16_words(array: mx.array) -> np.ndarray:
    """Raw 16-bit words of an array rounded to bfloat16.

    NumPy cannot name bfloat16, so bit-exact comparison goes through the raw
    words. The rounding is part of the check: the checkpoint stores bf16, while
    the fixture's model parameters are whatever dtype the constructor produced.
    """
    return np.asarray(array.astype(mx.bfloat16).view(mx.uint16))


def test_convert_and_load_w4a16_checkpoint(tmp_path):
    """The converter's tensor predicate and the loader's module predicate agree.

    The whole point of the design: ``convert_checkpoint`` packs exactly the
    tensors the model-side ``nn.quantize`` produces, so the strict loader finds
    every ``.scales``/``.biases`` and nothing is silently skipped.
    """
    from mlx.utils import tree_flatten

    from funasr_mlx.pipeline import load_model

    source = tmp_path / "source"
    parameters = _write_native_bf16_source(source, "qwen3-asr", TINY_QWEN3_ASR)
    quantization = _spec(targets=["thinker.model.layers."])
    destination = tmp_path / "quantized"
    convert_checkpoint(
        source,
        destination,
        model="qwen3-asr",
        dtype="bfloat16",
        config={**TINY_QWEN3_ASR, "quantization": quantization},
        source_rule=IDENTITY_SOURCE_RULE,
        quantization=quantization,
    )
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert index["metadata"]["quantization"]["bits"] == 4
    assert index["metadata"]["quantized_tensors"] == 14  # 2 layers x 7 Linears
    # Nothing under the decoder prefix is skipped: every Linear there has an
    # input dim divisible by 64, and LayerNorm weights are not Linears at all.
    assert index["metadata"]["quantization_skipped"] == []
    # The embedding and the audio tower are still full precision.
    assert "thinker.model.embed_tokens.scales" not in index["weight_map"]
    assert "audio_tower.proj1.scales" not in index["weight_map"]

    loaded, _ = load_model(destination, family="qwen3-asr")
    assert isinstance(loaded.thinker.model.layers[0].mlp.down_proj, nn.QuantizedLinear)
    assert isinstance(loaded.thinker.model.embed_tokens, nn.Embedding)
    assert isinstance(loaded.audio_tower.proj1, nn.Linear)
    assert loaded._quantization.summary()["linear_layers_quantized"] == 14

    # Every expected parameter round-trips exactly, including the packed codes.
    actual = dict(tree_flatten(loaded.parameters()))
    for name, expected in parameters.items():
        module = name[: -len(".weight")]
        if f"{module}.scales" in actual:  # quantized: weight is now uint32 codes
            assert actual[name].dtype == mx.uint32, name
            assert actual[f"{module}.scales"].dtype == mx.bfloat16, name
            continue
        assert name in actual, name
        np.testing.assert_array_equal(
            _bf16_words(actual[name]), _bf16_words(expected), err_msg=name
        )


def test_converter_skips_layers_the_group_size_cannot_divide(tmp_path):
    """A tensor the converter cannot pack must be reported, not silently dropped.

    The loader would then reject the checkpoint (the model predicate skipped
    the same Linear), which is the loud failure the design intends.
    """
    source = tmp_path / "source"
    _write_native_bf16_source(source, "qwen3-asr", TINY_QWEN3_ASR)
    quantization = _spec(targets=["thinker.model.layers."], group_size=128)
    destination = tmp_path / "quantized"
    convert_checkpoint(
        source,
        destination,
        model="qwen3-asr",
        dtype="bfloat16",
        config={**TINY_QWEN3_ASR, "quantization": quantization},
        source_rule=IDENTITY_SOURCE_RULE,
        quantization=quantization,
    )
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    # Only down_proj (input dim 128) is divisible by 128; the other 12 decoder
    # Linears are recorded as skipped, and the load then succeeds because the
    # model-side predicate skips exactly the same modules.
    assert index["metadata"]["quantized_tensors"] == 2
    skipped = index["metadata"]["quantization_skipped"]
    assert len(skipped) == 12 and all(name.endswith(".weight") for name in skipped)
    assert not any("down_proj" in name for name in skipped)
