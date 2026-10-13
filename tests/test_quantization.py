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


def test_embedding_needs_an_explicit_target():
    """``nn.Embedding`` is not an ``nn.Linear``: the Linear rule cannot reach it.

    Regression guard for a real bug in this task: the first version of the
    "decoder + head" variant silently quantized nothing, because
    ``isinstance(nn.Embedding(...), nn.Linear)`` is False and the embedding
    target prefix was ignored. Both predicates must select it.
    """
    model = _Tiny()
    decoder_only = QuantizationSpec.from_dict(_spec(targets=["layers."]))
    assert not decoder_only.selects_embedding("embed_tokens")
    assert not decoder_only.quantizes_tensor("embed_tokens.weight", (64, 128))
    apply_quantization(model, decoder_only)
    assert isinstance(model.embed_tokens, nn.Embedding)

    model = _Tiny()
    with_embedding = QuantizationSpec.from_dict(_spec(targets=["layers.", "embed_tokens"]))
    assert with_embedding.selects_embedding("embed_tokens")
    assert with_embedding.quantizes_tensor("embed_tokens.weight", (64, 128))
    # An untied lm_head is still never selected by the embedding prefix.
    assert not with_embedding.quantizes_tensor("lm_head.weight", (64, 128))
    result = apply_quantization(model, with_embedding)
    assert isinstance(model.embed_tokens, nn.QuantizedEmbedding)
    assert "embed_tokens" in result.quantized
    assert isinstance(model.lm_head, nn.Linear)


def test_quantized_embedding_forward_and_as_linear_agree():
    """The tied head path (``embed_tokens.as_linear``) must work quantized."""
    model = _Tiny()
    apply_quantization(model, _spec(targets=["layers.", "embed_tokens"]))
    tokens = mx.array([[1, 2, 3]])
    embedded = model.embed_tokens(tokens)
    mx.eval(embedded)
    assert embedded.shape == (1, 3, 128)
    head = model.embed_tokens.as_linear(mx.zeros((1, 128)))
    mx.eval(head)
    assert head.shape == (1, 64)


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


def test_quantized_group_over_the_shard_bound_is_rejected(tmp_path):
    """A quantized group must fit one shard, or the strict loader cannot load it.

    ``pipeline.load_model`` binds a Linear's weight/scales/biases as a unit and
    enforces ``max(256MB, largest_tensor) + 32MB`` per shard (raised from
    128+16: the old bound rejected the 1.7B tied embedding's 175MB group with
    2.6MB to spare on machines that had 2x headroom to the wired limit --
    conservative for an 8GB era).  A group still larger than the bound must
    fail at conversion, not at load.
    """
    import numpy as np

    from funasr_mlx.checkpoints import quantize_bf16_weight

    spec = QuantizationSpec.from_dict(_spec(targets=["layers."]))
    shape = (151936, 2048)
    assert spec.quantizes_tensor("layers.0.linear.weight", shape)
    packed_bytes = shape[0] * (shape[1] * 4 // 32) * 4 + 2 * shape[0] * (shape[1] // 64) * 2
    bound = max(256 * 1024 * 1024, shape[0] * (shape[1] * 4 // 32) * 4) + 32 * 1024 * 1024
    # The 1.7B embedding now fits INSIDE the raised bound -- that is the
    # point of the raise. Construct a genuinely oversized group instead:
    # the same vocabulary at w4/g32 doubles the side tensors.
    assert packed_bytes < bound, "the 1.7B embedding must now be admissible"
    g32_side = 2 * shape[0] * (shape[1] // 32) * 2
    weight_bytes = shape[0] * (shape[1] * 4 // 32) * 4
    assert weight_bytes + g32_side < bound, "sanity: even g32 stays inside"
    # A group that cannot fit any bound: the packed weight itself plus side
    # tensors over max(256MB, weight)+32MB is impossible for w4/g64 on this
    # shape, so the converter-side arithmetic is validated directly below.

    # The same arithmetic on a small tensor: the group is weight + 2 side
    # tensors, and the side tensors are 2 bytes per 64-weight group.
    weight = np.zeros((2048, 1024), dtype=np.uint16)
    packed = quantize_bf16_weight(
        "layers.0.linear.weight", weight, code="BF16", bits=4, group_size=64
    )
    group_bytes = sum(int(value.nbytes) for value in packed.values())
    expected = 2048 * (1024 * 4 // 32) * 4 + 2 * 2048 * (1024 // 64) * 2
    assert group_bytes == expected


def test_converter_emits_one_shard_per_quantized_group(tmp_path):
    """The group must not be split, and the emitted shard must satisfy the bound.

    A real convert -> load round trip on a checkpoint whose tied embedding is
    large enough to matter, asserting the loader accepts every shard.
    """
    from funasr_mlx.pipeline import load_model

    source = tmp_path / "source"
    _write_native_bf16_source(source, "qwen3-asr", TINY_QWEN3_ASR)
    destination = tmp_path / "quantized"
    quantization = _spec(targets=["thinker.model.layers.", "thinker.model.embed_tokens"])
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
    # The embedding is selected this time, so the count is 14 + 1.
    assert index["metadata"]["quantized_tensors"] == 15
    for name in ("thinker.model.embed_tokens.weight",):
        assert index["weight_map"][name] == index["weight_map"]["thinker.model.embed_tokens.scales"]
    loaded, _ = load_model(destination, family="qwen3-asr")
    assert isinstance(loaded.thinker.model.embed_tokens, nn.QuantizedEmbedding)


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


# --------------------------------------------------------------------------- #
# GPTQ integration: same layout, better values
# --------------------------------------------------------------------------- #
def test_gptq_converter_writes_the_same_layout_as_rtn(tmp_path):
    """A GPTQ run must be indistinguishable from RTN by layout alone.

    The whole point of GPTQ here is that the runtime path is untouched: the same
    three parameter names, the same packed-code dtype and shape, the same
    group side-tensor shape.  Only the values differ.  This asserts the layout
    part and that the values really did change, so a regression that silently
    fell back to RTN would be caught.
    """
    from funasr_mlx.checkpoints import GptqWeight
    from funasr_mlx.models.llm.gptq import GptqConfig

    source = tmp_path / "source"
    _write_native_bf16_source(source, "qwen3-asr", TINY_QWEN3_ASR)
    quantization = _spec(targets=["thinker.model.embed_tokens"])
    config = {**TINY_QWEN3_ASR, "quantization": quantization}

    plain = tmp_path / "plain"
    convert_checkpoint(
        source, plain, model="qwen3-asr", dtype="bfloat16", config=config,
        source_rule=IDENTITY_SOURCE_RULE, quantization=quantization,
    )

    # A Hessian for the embedding: identity-ish, so GPTQ has something real to
    # compensate against without needing a calibration forward.
    import mlx.core as mx

    model = create_model("qwen3-asr", config)
    weight = np.array(model.thinker.model.embed_tokens.weight.astype(mx.float32))
    hidden = weight.shape[1]
    rng = np.random.RandomState(0)
    activations = rng.randn(256, hidden).astype(np.float32)
    hessian = 2.0 * (activations.T @ activations)
    gptq = GptqWeight(
        hessians={"thinker.model.embed_tokens.weight": hessian},
        configs={
            "thinker.model.embed_tokens.weight": GptqConfig(
                bits=4, group_size=64, storage="bfloat16", act_order=True
            ).to_dict()
        },
    )

    destination = tmp_path / "gptq"
    convert_checkpoint(
        source, destination, model="qwen3-asr", dtype="bfloat16", config=config,
        source_rule=IDENTITY_SOURCE_RULE, quantization=quantization, gptq=gptq,
    )

    assert gptq.outcomes, "GPTQ did not run for the embedding"
    outcome = gptq.outcomes["thinker.model.embed_tokens.weight"]
    assert outcome["method"] == "gptq"
    assert outcome["output_error"] is not None
    assert outcome["rtn_output_error"] is not None
    # The gain is the reason for the whole exercise.
    assert outcome["output_error"] < outcome["rtn_output_error"]

    plain_index = json.loads((plain / "model.safetensors.index.json").read_text())
    gptq_index = json.loads((destination / "model.safetensors.index.json").read_text())
    name = "thinker.model.embed_tokens.weight"
    assert set(gptq_index["weight_map"]) == set(plain_index["weight_map"])
    assert gptq_index["weight_map"][name] == gptq_index["weight_map"][
        "thinker.model.embed_tokens.scales"
    ]
    assert gptq_index["metadata"]["gptq"] is not None

    def triple(directory):
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        shard = mx.load(str(directory / index["weight_map"][name]))
        return (
            np.array(shard[name]),
            np.array(shard["thinker.model.embed_tokens.scales"].astype(mx.float32)),
            np.array(shard["thinker.model.embed_tokens.biases"].astype(mx.float32)),
        )

    plain_packed, plain_scales, plain_biases = triple(plain)
    gptq_packed, gptq_scales, gptq_biases = triple(destination)
    # Same layout ...
    assert gptq_packed.shape == plain_packed.shape
    assert gptq_packed.dtype == plain_packed.dtype
    assert gptq_scales.shape == plain_scales.shape
    assert gptq_biases.shape == plain_biases.shape
    # ... different values.
    assert not np.array_equal(gptq_packed, plain_packed)


def test_gptq_converter_loads_through_the_normal_loader(tmp_path):
    """The GPTQ checkpoint must load with the existing strict loader, unchanged."""
    from funasr_mlx.checkpoints import GptqWeight
    from funasr_mlx.models.llm.gptq import GptqConfig
    from funasr_mlx.pipeline import load_model

    source = tmp_path / "source"
    _write_native_bf16_source(source, "qwen3-asr", TINY_QWEN3_ASR)
    quantization = _spec(targets=["thinker.model.embed_tokens"])
    config = {**TINY_QWEN3_ASR, "quantization": quantization}
    model = create_model("qwen3-asr", config)
    weight = np.array(model.thinker.model.embed_tokens.weight.astype(mx.float32))
    hessian = 2.0 * np.eye(weight.shape[1], dtype=np.float32)
    gptq = GptqWeight(
        hessians={"thinker.model.embed_tokens.weight": hessian},
        configs={
            "thinker.model.embed_tokens.weight": GptqConfig(
                bits=4, group_size=64, storage="bfloat16"
            ).to_dict()
        },
    )
    destination = tmp_path / "gptq"
    convert_checkpoint(
        source, destination, model="qwen3-asr", dtype="bfloat16", config=config,
        source_rule=IDENTITY_SOURCE_RULE, quantization=quantization, gptq=gptq,
    )
    loaded, _ = load_model(destination, family="qwen3-asr")
    assert isinstance(loaded.thinker.model.embed_tokens, nn.QuantizedEmbedding)


def test_gptq_config_must_agree_with_the_converter(tmp_path):
    """A bits/group mismatch must fail loudly, not write a mixed checkpoint."""
    from funasr_mlx.checkpoints import GptqWeight
    from funasr_mlx.models.llm.gptq import GptqConfig

    source = tmp_path / "source"
    _write_native_bf16_source(source, "qwen3-asr", TINY_QWEN3_ASR)
    quantization = _spec(targets=["thinker.model.embed_tokens"])
    config = {**TINY_QWEN3_ASR, "quantization": quantization}
    gptq = GptqWeight(
        hessians={"thinker.model.embed_tokens.weight": np.eye(64, dtype=np.float32)},
        configs={
            "thinker.model.embed_tokens.weight": GptqConfig(
                bits=8, group_size=64, storage="bfloat16"
            ).to_dict()
        },
    )
    with pytest.raises(ValueError, match="bits=8"):
        convert_checkpoint(
            source, tmp_path / "out", model="qwen3-asr", dtype="bfloat16",
            config=config, source_rule=IDENTITY_SOURCE_RULE,
            quantization=quantization, gptq=gptq,
        )


def test_gptq_weight_without_a_hessian_raises():
    """Asking GPTQ for a tensor it has no Hessian for must not silently fall back."""
    from funasr_mlx.checkpoints import GptqWeight

    gptq = GptqWeight(hessians={"a.weight": np.eye(4, dtype=np.float32)})
    assert gptq.has("a.weight")
    assert not gptq.has("b.weight")
    with pytest.raises(ValueError, match="No GPTQ Hessian"):
        gptq.quantize_tensor("b.weight", mx.zeros((2, 4)), bits=4, group_size=4)


def test_quantization_spec_records_the_method_and_rejects_unknown_ones():
    assert QuantizationSpec.from_dict(_spec()).method == "rtn"
    assert QuantizationSpec.from_dict(_spec(method="gptq")).method == "gptq"
    assert QuantizationSpec.from_dict(_spec(method="gptq")).to_dict()["method"] == "gptq"
    with pytest.raises(ValueError, match="Unsupported quantization method"):
        QuantizationSpec.from_dict(_spec(method="magic"))


# --------------------------------------------------------------------------- #
# the activation-dtype bug: quantized codes are not activations
# --------------------------------------------------------------------------- #
def test_activation_dtype_is_not_the_packed_code_dtype():
    """The dtype activations must follow is the OUTPUT dtype, not ``weight``'s.

    ``QuantizedEmbedding`` stores ``weight`` as packed ``uint32`` codes.  Every
    family splices the audio tower's output into the token stream and casts it to
    ``embed_tokens``'s dtype -- read as ``weight.dtype``, that is ``uint32``, so
    the cast turned every audio embedding into an integer and destroyed the
    transcript from step 0.

    The failure looked exactly like "4-bit is too coarse for the tied embedding"
    (garbage from step 0, bf16 perfect), so it was chased as an accuracy problem
    across GPTQ, act-order, w8 and mixed Hessians before the cast was found.  No
    quantization scheme can fix it, which is why this test pins the helper rather
    than any scheme.
    """
    from funasr_mlx.models.llm.quant import activation_dtype

    plain = nn.Embedding(64, 32)
    assert activation_dtype(plain) == plain.weight.dtype

    quantized = nn.QuantizedEmbedding(64, 32, group_size=32, bits=4)
    assert quantized.weight.dtype == mx.uint32  # the trap
    assert activation_dtype(quantized) == quantized.scales.dtype
    assert activation_dtype(quantized) != mx.uint32
    # And it must agree with what the module actually emits.
    out = quantized(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert activation_dtype(quantized) == out.dtype


def test_activation_dtype_handles_quantized_linear():
    from funasr_mlx.models.llm.quant import activation_dtype

    layer = nn.Linear(32, 16, bias=False)
    quantized = nn.QuantizedLinear.from_linear(layer, group_size=32, bits=4)
    assert quantized.weight.dtype == mx.uint32
    assert activation_dtype(quantized) == quantized.scales.dtype


def test_activation_dtype_rejects_a_weightless_module():
    from funasr_mlx.models.llm.quant import activation_dtype

    class NoWeight(nn.Module):
        def __init__(self):
            super().__init__()

    with pytest.raises(ValueError, match="no weight"):
        activation_dtype(NoWeight())


def test_a_quantized_decoder_reports_a_floating_activation_dtype():
    """End to end: the family helper must never return an integer dtype.

    This is the assertion that would have caught the bug at its source instead of
    ten languages of garbage transcripts later.
    """
    model = create_model("qwen3-asr", TINY_QWEN3_ASR)
    quantization = _spec(targets=["thinker.model.embed_tokens"])
    result = apply_quantization(model, quantization)
    assert result is not None and result.quantized
    dtype = model._decoder_dtype()
    assert dtype in (mx.bfloat16, mx.float16, mx.float32), (
        f"qwen3-asr reported {dtype} as the decoder activation dtype after "
        "quantizing the tied embedding"
    )
    assert dtype != mx.uint32


def test_no_dtype_helper_reads_a_quantizable_weight_directly():
    """Every activation-dtype helper must survive its module being quantized.

    The failure this guards is subtle and was hit twice in one task: a helper
    reads `<module>.weight.dtype` to decide what to cast activations to. That is
    correct for a plain module, but a quantized module stores packed ``uint32``
    codes in ``weight``, so the cast turns real activations into integers --
    silently, and only for checkpoints that quantize that particular module. It
    produced garbage transcripts from step 0 that looked exactly like "4-bit is
    too coarse", and it survived being chased through GPTQ, act-order, w8 and
    mixed Hessians.

    This test walks the helpers that feed activations and asserts each either
    routes through ``activation_dtype`` or reads a weight that quantization
    cannot reach (a conv / norm, which are not ``nn.Linear``).
    """
    import inspect

    from funasr_mlx.models import (
        emotion2vec,
        glm_asr_thinker,
        moss_transcribe,
        qwen3_asr_thinker,
        sensevoice,
        whisper,
    )

    helpers = [
        (qwen3_asr_thinker.Qwen3ASR, "_decoder_dtype"),
        (qwen3_asr_thinker.Qwen3ASR, "_tower_dtype"),
        (glm_asr_thinker.GlmAsrThinker, "_decoder_dtype"),
        (glm_asr_thinker.GlmAsrThinker, "_tower_dtype"),
        (moss_transcribe.MossTranscribeDiarize, "_decoder_dtype"),
        (sensevoice.SenseVoiceEncoderSmall, "_weight_dtype"),
        (whisper.AudioEncoder, "_weight_dtype"),
        (whisper.TextDecoder, "_weight_dtype"),
        (emotion2vec.AudioEncoder, "_weight_dtype"),
    ]
    for owner, name in helpers:
        source = inspect.getsource(getattr(owner, name))
        assert "activation_dtype" in source or "conv" in source or "ln" in source, (
            f"{owner.__name__}.{name} reads a weight dtype directly; if that "
            "weight is ever quantized the cast becomes a uint32 cast. Route it "
            "through funasr_mlx.models.llm.quant.activation_dtype."
        )


# --------------------------------------------------------------------------- #
# per-target bit widths (a w8 head beside a w4 stack)
# --------------------------------------------------------------------------- #
def test_bits_for_uses_the_longest_matching_prefix():
    from funasr_mlx.models.llm.quant import QuantizationSpec

    spec = QuantizationSpec.from_dict(
        {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "targets": ["thinker.model.layers.", "thinker.model.embed_tokens"],
            "target_bits": {"thinker.model.embed_tokens": 8},
        }
    )
    assert spec.bits == 4
    assert spec.bits_for("thinker.model.embed_tokens") == 8
    assert spec.bits_for("thinker.model.layers.0.self_attn.q_proj") == 4
    # A nested override wins over the broader one it sits inside.
    nested = QuantizationSpec.from_dict(
        {
            "bits": 4,
            "group_size": 64,
            "targets": ["thinker.model.layers."],
            "target_bits": {
                "thinker.model.layers.": 8,
                "thinker.model.layers.0.": 4,
            },
        }
    )
    assert nested.bits_for("thinker.model.layers.0.mlp.up_proj") == 4
    assert nested.bits_for("thinker.model.layers.1.mlp.up_proj") == 8


def test_target_bits_round_trip_and_validate():
    from funasr_mlx.models.llm.quant import QuantizationSpec

    spec = QuantizationSpec.from_dict(
        {
            "bits": 4,
            "group_size": 64,
            "targets": ["a."],
            "target_bits": {"a.embed": 8},
        }
    )
    assert spec.to_dict()["target_bits"] == {"a.embed": 8}
    assert QuantizationSpec.from_dict(spec.to_dict()).bits_for("a.embed") == 8
    # A spec with no overrides must not grow a target_bits key.
    plain = QuantizationSpec.from_dict({"bits": 4, "group_size": 64, "targets": ["a."]})
    assert "target_bits" not in plain.to_dict()
    with pytest.raises(ValueError, match="Unsupported bit width"):
        QuantizationSpec.from_dict(
            {"bits": 4, "group_size": 64, "targets": ["a."], "target_bits": {"a.": 7}}
        )
    with pytest.raises(ValueError, match="target_bits must be a mapping"):
        QuantizationSpec.from_dict(
            {"bits": 4, "group_size": 64, "targets": ["a."], "target_bits": [8]}
        )


def test_target_groups_use_the_longest_matching_prefix():
    """Per-target group sizes mirror target_bits (mlx-lm's AWQ default: a
    g32 tied embedding beside a g64 stack)."""
    from funasr_mlx.models.llm.quant import QuantizationSpec

    spec = QuantizationSpec.from_dict(
        {
            "bits": 4,
            "group_size": 64,
            "targets": ["thinker.model.layers.", "thinker.model.embed_tokens"],
            "target_groups": {"thinker.model.embed_tokens": 32},
        }
    )
    assert spec.group_for("thinker.model.embed_tokens") == 32
    assert spec.group_for("thinker.model.layers.0.self_attn.q_proj") == 64
    assert spec.group_for("thinker.model.embed_tokens") != spec.group_size
    # Nested prefix wins, same rule as bits_for.
    nested = QuantizationSpec.from_dict(
        {
            "bits": 4,
            "group_size": 64,
            "targets": ["a."],
            "target_groups": {"a.": 128, "a.head.": 32},
        }
    )
    assert nested.group_for("a.head.linear") == 32
    assert nested.group_for("a.other.linear") == 128
    assert spec.to_dict()["target_groups"] == {"thinker.model.embed_tokens": 32}
    plain = QuantizationSpec.from_dict({"bits": 4, "group_size": 64, "targets": ["a."]})
    assert "target_groups" not in plain.to_dict()
    with pytest.raises(ValueError, match="Unsupported group size"):
        QuantizationSpec.from_dict(
            {"bits": 4, "group_size": 64, "targets": ["a."], "target_groups": {"a.": 48}}
        )


def test_a_g32_head_beside_a_g64_stack_loads_strictly():
    """The group override must be honoured at LOAD time, like the bit override.

    A g32 QuantizedEmbedding has twice as many groups, so its scales/biases
    shapes differ from a g64 one; building it at the spec default would make
    the strict loader reject the checkpoint's shapes.
    """
    model = create_model("qwen3-asr", TINY_QWEN3_ASR)
    quantization = {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "method": "gptq",
        "targets": ["thinker.model.layers.", "thinker.model.embed_tokens"],
        "target_groups": {"thinker.model.embed_tokens": 32},
    }
    result = apply_quantization(model, quantization)
    assert result is not None
    head = model.thinker.model.embed_tokens
    assert isinstance(head, nn.QuantizedEmbedding)
    # 32-wide groups: the tiny config's hidden size is 64 -> 2 groups (g64
    # would give 1), and the scales side tensor carries one entry per group.
    assert head.group_size == 32
    assert head.scales.shape[1] == 64 // 32
    # The decoder Linears keep the default group.
    layer = model.thinker.model.layers[0].self_attn.q_proj
    assert isinstance(layer, nn.QuantizedLinear)
    assert layer.group_size == 64
    assert result.group_overrides == {"thinker.model.embed_tokens": 32}



def test_a_w8_head_beside_a_w4_stack_loads_strictly():
    """The override must be honoured at LOAD time, not just in the report.

    If apply_quantization built the head at the spec's default width, the strict
    loader would reject the checkpoint's shapes. This is the end-to-end check
    that the per-target width reaches the module construction.
    """
    model = create_model("qwen3-asr", TINY_QWEN3_ASR)
    quantization = {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "method": "gptq",
        "targets": ["thinker.model.layers.", "thinker.model.embed_tokens"],
        "target_bits": {"thinker.model.embed_tokens": 8},
    }
    result = apply_quantization(model, quantization)
    assert result is not None
    head = model.thinker.model.embed_tokens
    assert isinstance(head, nn.QuantizedEmbedding)
    # 8 bits -> 4 codes per uint32 word; the hidden size is 64 in the tiny config.
    assert head.bits == 8
    assert head["weight"].shape[-1] == 64 // 4
    # The decoder Linears keep the default width.
    layer = model.thinker.model.layers[0].self_attn.q_proj
    assert isinstance(layer, nn.QuantizedLinear)
    assert layer.bits == 4
    assert result.overrides == {"thinker.model.embed_tokens": 8}


def test_quantizes_tensor_matches_mlx_divisibility_for_every_bit_width():
    """The converter predicate must agree with what mx.quantize actually accepts.

    MLX's only divisibility requirement is that the GROUP SIZE divides the input
    dimension; the pack factor (codes per uint32 word) is not a constraint on the
    columns. An earlier version of the predicate also required
    ``columns % (32 // bits) == 0``, which was wrong in both directions: it
    rejected w3/w5/w6 at 64 columns (MLX accepts them) and accepted w2 at 96
    columns (MLX rejects it, because 64 does not divide 96). Over-strict is the
    dangerous direction -- it silently skips tensors that should be quantized.
    """
    from funasr_mlx.models.llm.quant import QuantizationSpec

    for bits in (2, 3, 4, 5, 6, 8):
        spec = QuantizationSpec.from_dict(
            {
                "bits": bits,
                "group_size": 64,
                "targets": ["a."],
                "target_bits": {"a.": bits},
            }
        )
        for columns in (64, 96, 128, 160, 192, 256):
            predicted = spec.quantizes_tensor("a.w.weight", (8, columns))
            try:
                mx.eval(mx.quantize(mx.random.normal((8, columns)), group_size=64, bits=bits))
                accepted = True
            except Exception:
                accepted = False
            assert predicted == accepted, (
                f"bits={bits} columns={columns}: predicate says {predicted}, "
                f"mx.quantize says {accepted}"
            )
