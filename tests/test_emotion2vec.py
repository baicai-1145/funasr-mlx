"""Tiny deterministic tests for the native MLX ``emotion2vec`` port.

These check *source formulas, key names and config validation*.  They are not
real-checkpoint acceptance: no weights were downloaded and no numerical parity
against PyTorch was executed.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from funasr_mlx.models.emotion2vec import (
    INFERENCE_ONLY_PREFIXES,
    AltBlock,
    AudioEncoder,
    CheckpointKeyError,
    ConvFeatureExtraction,
    Emotion2VecConfig,
    Emotion2VecModel,
    OrderedModules,
    PositionalEncoder,
    SamePad,
    SequentialStack,
    TransposeLast,
    conv_feature_output_length,
    convert_checkpoint_tensors,
    flat_parameter_names,
    get_alibi,
    get_slopes,
    is_conv_weight,
    load_state_dict_strict,
    parse_feature_encoder_spec,
    preprocess_waveform,
)

LARGE_SPEC = "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)]"


def assert_allclose(actual, expected, *, atol: float = 1e-5, rtol: float = 0.0) -> None:
    """``pytest.approx`` rejects nested Python lists; go through numpy."""
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=rtol,
        atol=atol,
    )


def tiny_config(**overrides) -> Emotion2VecConfig:
    """A miniature that keeps every *structural* ratio of the large config."""
    spec = "[(8, 5, 2)] + [(8, 3, 2)] * 2"
    payload = dict(
        embed_dim=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        feature_encoder_spec=spec,
        conv_pos_depth=2,
        conv_pos_width=12,
        conv_pos_groups=4,
        prenet_depth=2,
        num_alibi_heads=2,
        num_extra_tokens=3,
        max_samples=4000,
        max_frames=64,
    )
    payload.update(overrides)
    return Emotion2VecConfig(**payload)


# --------------------------------------------------------------------------- #
# feature_encoder_spec: ast parsing, never eval
# --------------------------------------------------------------------------- #
def test_large_spec_resolves_to_seven_layers():
    assert parse_feature_encoder_spec(LARGE_SPEC) == [
        (512, 10, 5),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 2, 2),
        (512, 2, 2),
    ]


def test_spec_accepts_only_int_literal_arithmetic():
    assert parse_feature_encoder_spec("[(4, 3, 2)]") == [(4, 3, 2)]
    assert parse_feature_encoder_spec("[(4, 3, 2)] + [(4, 1, 1)] * 3") == [
        (4, 3, 2), (4, 1, 1), (4, 1, 1), (4, 1, 1)
    ]
    # eval() would happily run these; ast must not.
    for hostile in (
        "__import__('os').getcwd()",
        "[(2**64, 3, 1)]",
        "[(len('ab'), 3, 1)]",
        "open('/etc/passwd')",
        "dir()",
    ):
        with pytest.raises(ValueError):
            parse_feature_encoder_spec(hostile)


def test_spec_rejects_malformed_layers():
    for bad in ("[(4, 3)]", "[(4, 3, 1, 1)]", "[]", "", "   ", "not python at all (("):
        with pytest.raises(ValueError):
            parse_feature_encoder_spec(bad)
    with pytest.raises(ValueError):
        parse_feature_encoder_spec("[(0, 3, 1)]")
    with pytest.raises(ValueError):
        parse_feature_encoder_spec("[(4, 3, 0)]")


def test_conv_output_length_matches_the_fairseq_formula():
    spec = parse_feature_encoder_spec(LARGE_SPEC)
    length = 48000
    for _, kernel, stride in spec:
        length = math.floor((length - kernel) / stride + 1)
    assert conv_feature_output_length(48000, spec) == length
    assert conv_feature_output_length(48000, spec) == 149
    # a stack that runs out of input must raise rather than return a bad length
    with pytest.raises(ValueError, match="too short"):
        conv_feature_output_length(10, spec)


# --------------------------------------------------------------------------- #
# ALiBi
# --------------------------------------------------------------------------- #
def test_slopes_power_of_two_branch():
    # get_slopes_power_of_2: start = 2 ** (-(2 ** -(log2(n) - 3)))
    slopes = get_slopes(16)
    assert len(slopes) == 16
    start = 2 ** (-(2 ** -(math.log2(16) - 3)))
    assert slopes[0] == pytest.approx(start)
    assert slopes[1] == pytest.approx(start * start)
    assert slopes == sorted(slopes, reverse=True)


def test_slopes_non_power_of_two_branch():
    slopes = get_slopes(6)
    power = get_slopes(4)
    assert slopes[:4] == pytest.approx(power)
    assert len(slopes) == 6


def test_alibi_is_zero_on_the_diagonal_and_linear_in_distance():
    bias = get_alibi(5, 3)
    mx.eval(bias)
    assert bias.shape == (3, 5, 5)
    assert_allclose(np.diagonal(bias[0]), np.zeros(5), atol=1e-6)
    # bias[h][i, j] == -|i - j| * slope[h]
    slope = get_slopes(3)[0]
    rows = np.asarray(bias[0])
    for i in range(5):
        for j in range(5):
            assert rows[i, j] == pytest.approx(-abs(i - j) * slope, abs=1e-6)


# --------------------------------------------------------------------------- #
# SamePad
# --------------------------------------------------------------------------- #
def test_same_pad_for_even_kernels_crops_the_last_time_slot():
    """fairseq SamePad *crops*: ``x[:, :, :-remove]`` -- it never pads."""
    pad = SamePad(6)
    assert pad.remove == 1
    out = pad(mx.arange(32, dtype=mx.float32).reshape(1, 4, 8))  # 1*4*8 == 32
    mx.eval(out)
    assert out.shape == (1, 4, 7)  # the final time slot is dropped
    # the values that survive are the FIRST seven, not a shifted or padded copy
    assert_allclose(np.asarray(out)[0, 0], np.arange(7, dtype=np.float64))


def test_same_pad_for_the_large_kernel_is_the_identity():
    # k = 19 is odd -> remove = 0, so SamePad is the identity for every length.
    pad = SamePad(19)
    assert pad.remove == 0
    for length in (4, 7, 8, 9):
        out = pad(mx.zeros((1, 4, length)))
        mx.eval(out)
        assert out.shape[-1] == length


def test_positional_encoder_preserves_the_time_axis():
    """conv padding=k//2 plus SamePad's crop/identity must return ``L`` exactly."""
    for kernel_size in (4, 5, 6, 19):
        module = PositionalEncoder(8, depth=2, kernel_size=kernel_size, groups=4, eps=1e-5)
        for length in (5, 7, 8):
            out = module(mx.zeros((1, length, 8)))
            mx.eval(out)
            assert out.shape == (1, length, 8), (kernel_size, length)


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_large_config_decodes_the_published_architecture():
    config = Emotion2VecConfig()
    assert config.embed_dim == 1024
    assert config.depth == 8
    assert config.num_heads == 16
    assert config.mlp_ratio == 4.0
    assert config.norm_eps == 1e-5
    assert config.layer_norm_first is False  # post-norm
    assert config.feature_embed_dim == 512
    assert config.pos_kernel_size == 19  # max(3, 95 // 5)
    assert config.prenet_depth == 4
    assert config.num_extra_tokens == 10
    assert config.learned_alibi is False
    assert config.learned_alibi_scale is True
    assert config.learned_alibi_scale_per_head is True
    assert config.learned_alibi_scale_per_layer is False
    assert config.ffn_targets is True  # not end_of_block_targets


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        Emotion2VecConfig(embed_dim=15, num_heads=16)
    with pytest.raises(ValueError):
        Emotion2VecConfig(extractor_mode="group_norm")
    with pytest.raises(ValueError):
        Emotion2VecConfig(num_alibi_heads=8, num_heads=16)
    with pytest.raises(ValueError):
        Emotion2VecConfig(embed_dim=18, conv_pos_groups=16)
    with pytest.raises(ValueError):
        Emotion2VecConfig(depth=0)
    with pytest.raises(ValueError):
        Emotion2VecConfig(vocab_size=3, labels=("a", "b"))
    with pytest.raises(ValueError):
        Emotion2VecConfig(learned_alibi=True, alibi_max_pos=None)
    with pytest.raises(TypeError):
        Emotion2VecConfig(embed_dim="1024")
    with pytest.raises(TypeError):
        Emotion2VecConfig(mlp_ratio="4.0")


def test_frames_for_enforces_the_memory_bounds():
    config = Emotion2VecConfig(max_samples=3200, max_frames=64)
    frames = config.frames_for(3200)
    assert frames == 9
    assert frames + config.num_extra_tokens <= config.max_frames
    with pytest.raises(ValueError, match="max_samples"):
        config.frames_for(3201)
    tight = Emotion2VecConfig(max_samples=160000, max_frames=5)
    with pytest.raises(ValueError, match="max_frames"):
        tight.frames_for(16000)


def test_frames_for_rejects_an_input_too_short_for_the_stack():
    config = Emotion2VecConfig()
    with pytest.raises(ValueError, match="too short"):
        config.frames_for(4)


# --------------------------------------------------------------------------- #
# parameter names
# --------------------------------------------------------------------------- #
def test_parameter_names_match_the_reference_prefixes():
    config = tiny_config(vocab_size=5, labels=("a", "b", "c", "d", "e"))
    names = set(flat_parameter_names(Emotion2VecModel(config)))

    # audio modality: local feature encoder, one conv weight + one affine LN per layer
    for index in range(3):
        assert f"modality_encoders.AUDIO.local_encoder.conv_layers.{index}.0.weight" in names
        assert f"modality_encoders.AUDIO.local_encoder.conv_layers.{index}.2.1.weight" in names
        assert f"modality_encoders.AUDIO.local_encoder.conv_layers.{index}.2.1.bias" in names
    # conv_bias=False -> no conv bias
    assert "modality_encoders.AUDIO.local_encoder.conv_layers.0.0.bias" not in names

    # projection is Sequential(TransposeLast, LayerNorm(512), Linear)
    assert "modality_encoders.AUDIO.project_features.1.weight" in names
    assert "modality_encoders.AUDIO.project_features.2.weight" in names
    # project_features.0 is TransposeLast -> no parameters
    assert not any(name.endswith("project_features.0.weight") for name in names)

    # relative positional encoder: Sequential(TransposeLast, 2 blocks, TransposeLast)
    assert "modality_encoders.AUDIO.relative_positional_encoder.0.weight" not in names
    for index in (1, 2):
        assert f"modality_encoders.AUDIO.relative_positional_encoder.{index}.0.weight" in names
        assert f"modality_encoders.AUDIO.relative_positional_encoder.{index}.0.bias" in names
        # elementwise_affine=False -> the block norm has no parameters
        assert (
            f"modality_encoders.AUDIO.relative_positional_encoder.{index}.3.weight" not in names
        )

    # context encoder norm (layer_norm_first=False -> prenet_norm exists)
    assert "modality_encoders.AUDIO.context_encoder.norm.weight" in names

    # extra tokens and the learned ALiBi scale
    assert "modality_encoders.AUDIO.extra_tokens" in names
    assert "modality_encoders.AUDIO.alibi_scale" in names

    # top level blocks and the classifier
    assert "blocks.0.norm1.weight" in names
    assert "blocks.0.attn.qkv.weight" in names
    assert "blocks.0.attn.proj.weight" in names
    assert "blocks.0.mlp.fc1.weight" in names
    assert "blocks.0.mlp.fc2.weight" in names
    assert "blocks.1.norm2.weight" in names
    assert "proj.weight" in names
    assert "proj.bias" in names

    # nothing under the training-only decoder
    assert not any(name.startswith("modality_encoders.AUDIO.decoder.") for name in names)
    # and no self.norm at top level (layer_norm_first=False)
    assert "norm.weight" not in names


def test_parameter_count_matches_the_reference_module_inventory():
    config = tiny_config(vocab_size=5, labels=("a", "b", "c", "d", "e"))
    names = flat_parameter_names(Emotion2VecModel(config))
    per_block = 12  # norm1 2, attn.qkv 2, attn.proj 2, norm2 2, fc1 2, fc2 2
    expected = (
        3 * (1 + 2)            # 3 conv feature blocks: conv weight + affine LN
        + 4                    # project_features LayerNorm(2) + Linear(2)
        + 2 * 2                # 2 positional blocks: conv weight + bias
        + 2                    # context_encoder.norm
        + config.prenet_depth * per_block
        + 1                    # extra_tokens
        + 1                    # alibi_scale
        + config.depth * per_block
        + 2                    # proj
    )
    assert len(names) == expected


def test_parameter_names_are_seen_at_all():
    """Regression guard: MLX nn.Module backs children in its dict, not __dict__."""
    names = flat_parameter_names(Emotion2VecModel(tiny_config()))
    assert len(names) > 50
    assert all(isinstance(value, mx.array) for value in names.values())


def test_alibi_scale_shape_follows_the_per_layer_and_per_head_flags():
    per_head = AudioEncoder(tiny_config()).alibi_scale
    assert per_head.shape == (1, 1, 2, 1, 1)  # per_layer=False, per_head=True
    per_layer = AudioEncoder(
        tiny_config(learned_alibi_scale_per_layer=True)
    ).alibi_scale
    assert per_layer.shape == (2 + 2, 1, 2, 1, 1)  # prenet_depth + depth
    shared = AudioEncoder(
        tiny_config(learned_alibi_scale_per_head=False)
    ).alibi_scale
    assert shared.shape == (1, 1, 1, 1, 1)


def test_no_head_means_no_projection():
    model = Emotion2VecModel(tiny_config(vocab_size=-1))
    assert model.proj is None
    assert "proj.weight" not in flat_parameter_names(model)
    with pytest.raises(RuntimeError, match="vocab_size"):
        model.classify(mx.zeros((1, 400)))


def test_audio_encoder_enforces_the_input_bounds_directly():
    encoder = AudioEncoder(tiny_config(max_samples=600, max_frames=64))
    with pytest.raises(ValueError, match=r"\(B, samples\)"):
        encoder.contextualized_features(mx.zeros((2, 2, 600)))
    with pytest.raises(ValueError, match="max_samples"):
        encoder.contextualized_features(mx.zeros((1, 601)))


def test_learned_alibi_scale_per_layer_is_applied_per_block():
    """H1: the prenet must honour one scale row per layer, not ignore the flag."""
    config = tiny_config(learned_alibi_scale_per_layer=True)
    encoder = AudioEncoder(config)
    assert encoder.alibi_scale.shape[0] == config.prenet_depth + config.depth
    x, alibi_bias, remaining = encoder.contextualized_features(mx.zeros((1, 400)))
    mx.eval(x, alibi_bias, remaining)
    # the tail handed to the shared blocks has exactly `depth` rows left
    assert remaining.shape[0] == config.depth
    assert remaining.shape == (config.depth, 1, config.num_alibi_heads, 1, 1)


def test_single_row_alibi_scale_is_folded_and_returns_none():
    config = tiny_config(learned_alibi_scale_per_layer=False)
    encoder = AudioEncoder(config)
    assert encoder.alibi_scale.shape[0] == 1
    _x, _bias, remaining = encoder.contextualized_features(mx.zeros((1, 400)))
    assert remaining is None


def test_emotion_scores_runs_the_encoder_exactly_once(monkeypatch):
    """H2: the previous implementation re-encoded to build ``feats``."""
    model = Emotion2VecModel(tiny_config(vocab_size=3, labels=("a", "b", "c")))
    calls = {"n": 0}
    original = model.extract_features

    def counted(samples, *, remove_extra_tokens=True):
        calls["n"] += 1
        return original(samples, remove_extra_tokens=remove_extra_tokens)

    monkeypatch.setattr(model, "extract_features", counted)
    result = model.emotion_scores(mx.zeros((1, 400)))
    mx.eval(result.feats)
    assert calls["n"] == 1
    assert result.labels == ("a", "b", "c")
    assert result.feats.shape == (1, 16)


def test_emotion_scores_requires_batch_one_and_a_head():
    headless = Emotion2VecModel(tiny_config(vocab_size=-1))
    with pytest.raises(RuntimeError, match="vocab_size"):
        headless.emotion_scores(mx.zeros((1, 400)))
    model = Emotion2VecModel(tiny_config(vocab_size=3, labels=("a", "b", "c")))
    with pytest.raises(ValueError, match="single EmotionResult"):
        model.emotion_scores(mx.zeros((2, 400)))


def test_unuse_labels_are_suppressed():
    model = Emotion2VecModel(
        tiny_config(vocab_size=4, labels=("a", "unuse_1", "b", "unuse_2"))
    )
    result = model.emotion_scores(mx.zeros((1, 400)))
    assert result.labels == ("a", "b")
    assert len(result.scores) == 2
    assert abs(sum(result.scores) - 1.0) < 1e-4


def test_classify_returns_frames_so_callers_do_not_re_encode():
    model = Emotion2VecModel(tiny_config(vocab_size=3, labels=("a", "b", "c")))
    pooled, logits, frames = model.classify(mx.zeros((1, 400)))
    mx.eval(pooled, logits, frames)
    expected = tiny_config().frames_for(400)
    assert frames.shape == (1, expected, 16)
    assert pooled.shape == (1, 16)
    assert logits.shape == (1, 3)


# --------------------------------------------------------------------------- #
# ordered containers
# --------------------------------------------------------------------------- #
def test_ordered_modules_indexes_positionally_and_by_key():
    inner = AltBlock(8, 2, 1.0, 1e-5, True, False, True)
    stack = OrderedModules({"0": inner, "1": TransposeLast()})
    assert stack[0] is inner
    assert stack["0"] is inner
    assert len(stack) == 2
    # SequentialStack is the same container with automatic 0..n-1 indices
    auto = SequentialStack({"0": inner, "1": TransposeLast()})
    assert auto["0"] is inner and auto[1] is not None


def test_conv_feature_extraction_rejects_a_non_2d_input():
    module = ConvFeatureExtraction(tiny_config().conv_layers, eps=1e-5, conv_bias=False)
    with pytest.raises(ValueError, match=r"\(B, samples\)"):
        module(mx.zeros((1, 2, 4)))


def test_alt_block_is_post_norm():
    """Post-norm branch: ``attn`` sees the raw ``x``, ``norm1`` comes after."""
    block = AltBlock(8, 2, 1.0, 1e-5, True, layer_norm_first=False, ffn_targets=True)
    assert block.layer_norm_first is False
    x = mx.arange(24, dtype=mx.float32).reshape(1, 3, 8) / 100.0
    out, target = block(x, None)
    mx.eval(out, target)
    assert out.shape == (1, 3, 8)
    assert target.shape == (1, 3, 8)
    # ffn_targets=True -> t is the *pre*-norm2 mlp output, so t != x.
    assert not np.allclose(np.asarray(target), np.asarray(out), atol=1e-6)
    # ffn_targets=False -> t is reassigned to the final x, so t == x.
    other = AltBlock(8, 2, 1.0, 1e-5, True, layer_norm_first=False, ffn_targets=False)
    out2, target2 = other(x, None)
    mx.eval(out2, target2)
    assert_allclose(target2, out2)


def test_alt_block_rejects_a_mismatched_alibi_bias():
    block = AltBlock(8, 2, 1.0, 1e-5, True, False, True)
    with pytest.raises(ValueError, match="alibi_bias"):
        block(mx.zeros((1, 3, 8)), mx.zeros((1, 2, 4, 4)))


# --------------------------------------------------------------------------- #
# checkpoint conversion / strict loading
# --------------------------------------------------------------------------- #
def _fake_pytorch_state(model: Emotion2VecModel) -> dict[str, mx.array]:
    """A PyTorch-shaped state dict for the same key set.

    The predicate here is deliberately **independent** of the production
    ``is_conv_weight``: it is a rank check (``(out, in//g, K)`` has a trailing
    extent equal to an odd kernel and is not square in the naive sense) so the
    round-trip test cannot confirm a wrong namespace by construction.
    """
    state: dict[str, mx.array] = {}
    for name, value in flat_parameter_names(model).items():
        if name.endswith(".0.weight") and value.ndim == 3 and "conv" in name:
            state[name] = mx.transpose(value, (0, 2, 1))  # MLX -> PyTorch
        else:
            state[name] = value
    return state


def test_no_extra_block_namespace_in_the_checkpoint_keys():
    """C3: children must be registered directly, not under an alias attribute."""
    config = tiny_config(vocab_size=5, labels=("a", "b", "c", "d", "e"))
    names = flat_parameter_names(Emotion2VecModel(config))
    for forbidden in (".block.", ".layers.", ".encoder."):
        assert not any(forbidden in name for name in names), forbidden
    assert any(name.startswith("modality_encoders.AUDIO.local_encoder.conv_layers.0.0.")
               for name in names)
    assert any(name.startswith("modality_encoders.AUDIO.relative_positional_encoder.1.0.")
               for name in names)


def test_conv_weight_predicate_matches_the_exact_source_keys():
    """C2: exact documented keys, not a positional ``parts[-3]`` guess."""
    assert is_conv_weight("modality_encoders.AUDIO.local_encoder.conv_layers.0.0.weight")
    assert is_conv_weight("modality_encoders.AUDIO.local_encoder.conv_layers.6.0.weight")
    assert is_conv_weight("modality_encoders.AUDIO.relative_positional_encoder.4.0.weight")
    # negatives: Linear weights, LayerNorms and the indexing that fooled the old check
    for name in (
        "modality_encoders.AUDIO.project_features.2.weight",
        "modality_encoders.AUDIO.local_encoder.conv_layers.0.2.1.weight",
        "modality_encoders.AUDIO.local_encoder.conv_layers.0.2.1.bias",
        "modality_encoders.AUDIO.relative_positional_encoder.1.3.weight",
        "blocks.0.norm1.weight",
        "proj.weight",
    ):
        assert not is_conv_weight(name), name


def test_tree_flatten_returns_a_dict_only_with_an_explicit_destination():
    """C1: the default return is a list of tuples, so ``.items()`` would raise."""
    from mlx.utils import tree_flatten

    assert isinstance(tree_flatten({"a": {"b": mx.zeros((1,))}}), list)
    assert isinstance(tree_flatten({"a": {"b": mx.zeros((1,))}}, destination={}), dict)
    names = flat_parameter_names(Emotion2VecModel(tiny_config()))
    assert names  # would raise AttributeError if the dict form were not requested


def test_conv_weights_are_transposed_back_from_the_pytorch_layout():
    model = Emotion2VecModel(tiny_config(vocab_size=5, labels=("a", "b", "c", "d", "e")))
    state = _fake_pytorch_state(model)
    key = "modality_encoders.AUDIO.local_encoder.conv_layers.0.0.weight"
    # the PyTorch-side fixture is (out, in//g, K) = (8, 1, 5)
    assert state[key].shape == (8, 1, 5)
    converted, discarded = convert_checkpoint_tensors(state)
    assert discarded == ()
    # after conversion the tensor is in MLX layout (out, K, in//g) = (8, 5, 1)
    assert converted[key].shape == (8, 5, 1)
    assert flat_parameter_names(model)[key].shape == (8, 5, 1)
    pos = "modality_encoders.AUDIO.relative_positional_encoder.1.0.weight"
    assert state[pos].shape == (16, 4, 5)
    assert converted[pos].shape == (16, 5, 4)


def test_module_list_stack_indexes_by_position_and_by_string_key():
    """``Module.update`` hands back the string keys that were registered."""
    from funasr_mlx.models.emotion2vec import ModuleListStack

    stack = ModuleListStack([TransposeLast(), TransposeLast(), TransposeLast()])
    assert len(stack) == 3
    assert stack[0] is not None
    assert stack["0"] is not None
    assert stack[-1] is not None
    assert stack["2"] is not None
    with pytest.raises(KeyError):
        stack["not-a-key"]
    with pytest.raises(IndexError):
        stack[3]


def test_conv_weight_conversion_rejects_a_wrong_rank():
    model = Emotion2VecModel(tiny_config(vocab_size=5, labels=("a", "b", "c", "d", "e")))
    state = _fake_pytorch_state(model)
    state["modality_encoders.AUDIO.local_encoder.conv_layers.0.0.weight"] = mx.zeros((2, 2))
    with pytest.raises(ValueError, match="3-D Conv1d weight"):
        convert_checkpoint_tensors(state)


def test_decoder_keys_are_discarded_and_reported_not_dropped_silently():
    model = Emotion2VecModel(tiny_config(vocab_size=5, labels=("a", "b", "c", "d", "e")))
    state = _fake_pytorch_state(model)
    state["modality_encoders.AUDIO.decoder.conv1.weight"] = mx.zeros((2, 2))
    state["modality_encoders.AUDIO.decoder.conv2.bias"] = mx.zeros((2,))
    converted, discarded = convert_checkpoint_tensors(state)
    assert discarded == (
        "modality_encoders.AUDIO.decoder.conv1.weight",
        "modality_encoders.AUDIO.decoder.conv2.bias",
    )
    assert not any(name.startswith("modality_encoders.AUDIO.decoder.") for name in converted)


def test_unknown_prefix_is_an_error_not_a_silent_drop():
    model = Emotion2VecModel(tiny_config(vocab_size=5, labels=("a", "b", "c", "d", "e")))
    state = _fake_pytorch_state(model)
    state["modality_encoders.AUDIO.mystery_layer.weight"] = mx.zeros((4, 4))
    with pytest.raises(CheckpointKeyError, match="unexpected"):
        load_state_dict_strict(model, state)


def test_strict_loading_accepts_a_matching_state_dict():
    model = Emotion2VecModel(tiny_config(vocab_size=5, labels=("a", "b", "c", "d", "e")))
    discarded, unexpected = load_state_dict_strict(model, _fake_pytorch_state(model))
    assert discarded == ()
    assert unexpected == ()


def test_strict_loading_rejects_missing_unexpected_and_mismatched():
    model = Emotion2VecModel(tiny_config(vocab_size=5, labels=("a", "b", "c", "d", "e")))
    key = "blocks.0.norm1.weight"

    missing = _fake_pytorch_state(model)
    del missing[key]
    with pytest.raises(CheckpointKeyError, match="missing"):
        load_state_dict_strict(model, missing)

    extra = _fake_pytorch_state(model)
    extra["blocks.9.norm1.weight"] = mx.zeros((16,))
    with pytest.raises(CheckpointKeyError, match="unexpected"):
        load_state_dict_strict(model, extra)

    wrong_shape = _fake_pytorch_state(model)
    wrong_shape[key] = mx.zeros((8,))
    with pytest.raises(CheckpointKeyError, match="mismatched"):
        load_state_dict_strict(model, wrong_shape)


def test_strict_loading_can_refuse_training_only_keys():
    model = Emotion2VecModel(tiny_config(vocab_size=5, labels=("a", "b", "c", "d", "e")))
    state = _fake_pytorch_state(model)
    state["modality_encoders.AUDIO.decoder.conv1.weight"] = mx.zeros((2, 2))
    load_state_dict_strict(model, state)  # allowed by default and reported
    with pytest.raises(CheckpointKeyError, match="training-only"):
        load_state_dict_strict(model, state, allow_training_only=False)


# --------------------------------------------------------------------------- #
# waveform preprocessing
# --------------------------------------------------------------------------- #
def test_preprocess_matches_f_layer_norm_over_the_whole_waveform():
    wave = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    out = preprocess_waveform(wave, normalize=True, eps=0.0)
    values = np.asarray(out[0])
    expected = (np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0) / np.sqrt(2.0)
    assert_allclose(values, expected, atol=1e-6)
    assert abs(values.mean()) < 1e-6


def test_preprocess_promotes_1d_input_and_can_skip_normalisation():
    out = preprocess_waveform(mx.array([1.0, 2.0, 3.0]), normalize=False)
    mx.eval(out)
    assert out.shape == (1, 3)
    assert_allclose(np.asarray(out[0]), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"\(samples,\)"):
        preprocess_waveform(mx.zeros((2, 2, 2)))


def test_inference_only_prefixes_are_the_documented_ones():
    assert INFERENCE_ONLY_PREFIXES == ("modality_encoders.AUDIO.decoder.",)