"""Tiny architecture / parity specifications for the MLX ASR ports.

These tests are deliberately small (single-digit dimensions, < 20 frames) so a
full run stays in the low-megabyte range.  They check **structure and
checkpoint-key parity**, not model accuracy: passing them is *not* acceptance of
a real FunASR checkpoint.  Numerical parity against PyTorch weights is the lead's
serialized job under the memory guard.

Run with ``pytest tests/test_asr.py`` from the repository root.  The whole module
skips itself when MLX is not installed, so collection never fails.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="mlx is required for the model ports")

from mlx import nn as mxnn  # noqa: E402

from funasr_mlx.models.paraformer import (  # noqa: E402
    CifPredictorConfig,
    CifPredictorV2,
    Paraformer,
    ParaformerConfig,
    ParaformerDecoderConfig,
    ParaformerSANMDecoder,
    SANMEncoder,
    SANMEncoderConfig,
    CIF_DEVICE,
    cif_v1,
    cif_wo_hidden_v1,
    paraformer_config_from_dict,
)
from funasr_mlx.models.sanm import (  # noqa: E402
    CONV1D_LAYOUT,
    CONV2D_LAYOUT,
    IDENTITY_LAYOUT,
    CheckpointKeyError,
    Conv2dSubsampling,
    ModuleStack,
    PositionalEncoding,
    SinusoidalPositionEncoder,
    attention_blocked,
    check_short_utt,
    constant_pad_1d,
    convert_checkpoint_tensor,
    flatten_state_dict,
    flatten_state_dict_with_rules,
    load_strict,
    make_pad_mask,
    sequence_mask,
)
from funasr_mlx.models.sensevoice import (  # noqa: E402
    EMO_DICT,
    LID_DICT,
    NUM_METADATA_EMBEDDINGS,
    NUM_PREFIX_FRAMES,
    TEXTNORM_DICT,
    SenseVoiceEncoderSmall,
    SenseVoiceEncoderSmallConfig,
    SenseVoiceSmall,
    SenseVoiceSmallConfig,
    ctc_greedy_decode,
    sensevoice_config_from_dict,
)


# --------------------------------------------------------------------------- #
# tiny configurations (single-digit dims everywhere)
# --------------------------------------------------------------------------- #
def _sanm_encoder_config() -> SANMEncoderConfig:
    return SANMEncoderConfig(
        output_size=8,
        attention_heads=2,
        linear_units=6,
        num_blocks=3,
        kernel_size=3,
        input_layer="pe",
        pos_enc_class="SinusoidalPositionEncoder",
    )


def _sensevoice_encoder_config() -> SenseVoiceEncoderSmallConfig:
    return SenseVoiceEncoderSmallConfig(
        output_size=8,
        attention_heads=2,
        linear_units=6,
        num_blocks=2,
        tp_blocks=1,
        kernel_size=3,
    )


def _paraformer_config() -> ParaformerConfig:
    return ParaformerConfig(
        input_size=8,
        vocab_size=13,
        encoder_conf=_sanm_encoder_config(),
        decoder_conf=ParaformerDecoderConfig(
            attention_heads=2,
            linear_units=6,
            num_blocks=3,
            att_layer_num=2,
            kernel_size=3,
        ),
    )


def _paraformer(**kwargs) -> Paraformer:
    cfg = _paraformer_config()
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return Paraformer(config=cfg)


def _encoder_layer_keys(prefix: str) -> set[str]:
    """Parameter names of one SANM encoder layer, in PyTorch checkpoint order."""
    return {
        f"{prefix}.self_attn.linear_q_k_v.weight",
        f"{prefix}.self_attn.linear_q_k_v.bias",
        f"{prefix}.self_attn.linear_out.weight",
        f"{prefix}.self_attn.linear_out.bias",
        f"{prefix}.self_attn.fsmn_block.weight",
        f"{prefix}.feed_forward.w_1.weight",
        f"{prefix}.feed_forward.w_1.bias",
        f"{prefix}.feed_forward.w_2.weight",
        f"{prefix}.feed_forward.w_2.bias",
        f"{prefix}.norm1.weight",
        f"{prefix}.norm1.bias",
        f"{prefix}.norm2.weight",
        f"{prefix}.norm2.bias",
    }


def _decoder_layer_keys(prefix: str, with_src_attn: bool, with_self_attn: bool) -> set[str]:
    keys: set[str] = set()
    if with_self_attn:
        keys.add(f"{prefix}.self_attn.fsmn_block.weight")
        keys |= {f"{prefix}.norm2.weight", f"{prefix}.norm2.bias"}
    if with_src_attn:
        keys |= {
            f"{prefix}.src_attn.linear_q.weight",
            f"{prefix}.src_attn.linear_q.bias",
            f"{prefix}.src_attn.linear_k_v.weight",
            f"{prefix}.src_attn.linear_k_v.bias",
            f"{prefix}.src_attn.linear_out.weight",
            f"{prefix}.src_attn.linear_out.bias",
        }
        keys |= {f"{prefix}.norm3.weight", f"{prefix}.norm3.bias"}
    # PositionwiseFeedForwardDecoderSANM: w_2 has no bias, norm is eps=1e-12.
    keys |= {
        f"{prefix}.feed_forward.w_1.weight",
        f"{prefix}.feed_forward.w_1.bias",
        f"{prefix}.feed_forward.w_2.weight",
        f"{prefix}.feed_forward.norm.weight",
        f"{prefix}.feed_forward.norm.bias",
        f"{prefix}.norm1.weight",
        f"{prefix}.norm1.bias",
    }
    return keys


# --------------------------------------------------------------------------- #
# masks / padding
# --------------------------------------------------------------------------- #
def test_sequence_mask_and_make_pad_mask_are_inverses():
    lengths = mx.array([5, 3, 1])
    keep = sequence_mask(lengths, maxlen=6)
    pad = make_pad_mask(lengths, maxlen=6)
    assert keep.shape == (3, 6)
    # mx arrays already flatten to nested Python lists, so no per-row .tolist().
    assert keep.tolist() == [
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ]
    # make_pad_mask must be the exact logical inverse of sequence_mask.
    assert (keep.astype(mx.bool_) == ~pad).all().item()


def test_sequence_mask_accepts_batched_lengths():
    keep = sequence_mask(mx.array([[5], [3], [1]]), maxlen=6)
    assert keep.shape == (3, 6)
    assert keep[0].tolist() == [1, 1, 1, 1, 1, 0]


def test_constant_pad_1d_pads_the_time_axis():
    x = mx.array([[[1.0], [2.0]]])  # (B=1, T=2, C=1), MLX NLC
    padded = constant_pad_1d(x, 2, 1, axis=1)
    assert padded.shape == (1, 5, 1)
    assert padded[0, :, 0].tolist() == [0.0, 0.0, 1.0, 2.0, 0.0]
    assert constant_pad_1d(x, 0, 0, axis=1) is x


# --------------------------------------------------------------------------- #
# containers / checkpoint naming
# --------------------------------------------------------------------------- #
def test_module_stack_has_no_layers_level_in_state_dict():
    stack = ModuleStack([mxnn.Linear(3, 2) for _ in range(12)])
    keys = set(flatten_state_dict(stack))
    assert keys == {f"{i}.{leaf}" for i in range(12) for leaf in ("weight", "bias")}
    assert not any(k.startswith("layers") for k in keys)
    assert len(stack) == 12
    assert stack[0] is stack[0]


def test_flatten_state_dict_reads_the_mapping_storage_not_dunder_dict():
    """Regression: ``mlx.nn.Module`` subclasses ``dict``.

    Arrays and sub-modules are stored via ``module[key] = value`` (the mapping),
    never in ``__dict__``, so a ``vars()``-based traversal silently returns an
    empty dict.  This pins the traversal for a bare leaf module, a nested module
    and a container.
    """
    leaf = mxnn.Linear(3, 2)
    assert set(flatten_state_dict(leaf)) == {"weight", "bias"}
    # __dict__ really is empty of tensors -- this is the trap being guarded.
    assert not any(isinstance(v, mx.array) for v in vars(leaf).values())
    # Both storage paths must agree.
    assert set(dict.keys(leaf)) == {"weight", "bias"}

    nested = ModuleStack([mxnn.Linear(3, 2), mxnn.Linear(3, 2)])
    assert set(flatten_state_dict(nested)) == {
        "0.weight",
        "0.bias",
        "1.weight",
        "1.bias",
    }


def test_flatten_state_dict_skips_private_and_non_array_config_attributes():
    """``_``-prefixed config and plain scalars must never become weights."""
    encoder = SANMEncoder(input_size=8, config=_sanm_encoder_config())
    keys = flatten_state_dict(encoder)
    assert keys, "encoder must expose parameters"
    assert not any(k.startswith("_") for k in keys)
    assert not any(part.startswith("_") for k in keys for part in k.split("."))
    # out_channels / output_level / config are configuration, not tensors.
    assert not {"out_channels", "output_level", "config"} & set(keys)


def test_layout_rules_are_derived_from_module_type_not_name():
    """A renamed conv parameter must still get the conv1d transpose."""
    tensors, rules = flatten_state_dict_with_rules(_paraformer())
    conv_key = "encoder.encoders.0.self_attn.fsmn_block.weight"
    assert rules[conv_key] == CONV1D_LAYOUT
    assert rules["encoder.after_norm.bias"] == IDENTITY_LAYOUT
    torch_shape = (8, 1, 3)
    converted = convert_checkpoint_tensor(
        "some.renamed.weight", mx.zeros(torch_shape), CONV1D_LAYOUT
    )
    assert converted.shape == (8, 3, 1)
    assert set(tensors) == set(flatten_state_dict(_paraformer()))


def test_module_stack_preserves_dict_semantics():
    """MLX calls ``dst[key]`` / ``key in dst`` internally; those must still work."""
    stack = ModuleStack([mxnn.Linear(3, 2), mxnn.Linear(3, 2)])
    assert "0" in dict.keys(stack)
    assert isinstance(dict.__getitem__(stack, "0"), mxnn.Linear)
    assert stack["0"] is dict.__getitem__(stack, "0")
    assert stack[-1] is dict.__getitem__(stack, "1")
    assert len(dict.items(stack)) == 2


def test_conv1d_weight_layout_is_mlx_native():
    model = _paraformer()
    key = "encoder.encoders.0.self_attn.fsmn_block.weight"
    weight = flatten_state_dict(model)[key]
    # depth-wise: (out, kernel, in/groups=1)
    assert weight.shape == (8, 3, 1)


def test_convert_checkpoint_tensor_transposes_only_conv_kernels():
    torch_conv1d = mx.zeros((8, 1, 3))  # torch layout (out, in/groups, k)
    converted = convert_checkpoint_tensor(
        "encoder.encoders.0.self_attn.fsmn_block.weight", torch_conv1d
    )
    assert converted.shape == (8, 3, 1)
    assert CONV1D_LAYOUT == "conv1d_torch_out_ink"

    torch_conv2d = mx.zeros((6, 1, 3, 3))  # torch layout (out, in, kh, kw)
    converted2d = convert_checkpoint_tensor("encoder.embed.conv.0.weight", torch_conv2d)
    assert converted2d.shape == (6, 3, 3, 1)
    assert CONV2D_LAYOUT == "conv2d_torch_out_ink"

    linear = mx.zeros((5, 7))
    assert convert_checkpoint_tensor("ctc.ctc_lo.weight", linear).shape == (5, 7)


# --------------------------------------------------------------------------- #
# Paraformer encoder
# --------------------------------------------------------------------------- #
def test_sanm_encoder_state_dict_matches_pytorch_naming():
    encoder = SANMEncoder(input_size=8, config=_sanm_encoder_config())
    keys = set(flatten_state_dict(encoder))
    expected = set()
    expected |= _encoder_layer_keys("encoders0.0")
    for i in range(2):  # num_blocks - 1
        expected |= _encoder_layer_keys(f"encoders.{i}")
    expected |= {"after_norm.weight", "after_norm.bias"}
    assert keys == expected
    # "pe" input layer has no parameters at all.
    assert not any(k.startswith("embed") for k in keys)


def test_sanm_encoder_forward_shapes_and_lengths():
    encoder = SANMEncoder(input_size=8, config=_sanm_encoder_config())
    speech = mx.zeros((2, 9, 8))
    lengths = mx.array([9, 5])
    out, olens = encoder(speech, lengths)
    assert out.shape == (2, 9, 8)
    assert olens.tolist() == [9, 5]


def test_sanm_encoder_rejects_unsupported_input_layer():
    cfg = _sanm_encoder_config()
    cfg.input_layer = "pe_online"
    with pytest.raises(NotImplementedError):
        SANMEncoder(input_size=8, config=cfg)


def test_check_short_utt_ignores_non_conv_embedders():
    encoder = SANMEncoder(input_size=8, config=_sanm_encoder_config())
    assert check_short_utt(encoder.embed, 9) == (False, -1)


def test_sanm_encoder_conv2d_input_layer_shapes_and_keys():
    """Covers the NHWC conv front-end: 12 frames / 8 bins -> 2 frames / 6 bins."""
    cfg = _sanm_encoder_config()
    cfg.output_size = 6
    cfg.input_layer = "conv2d"
    encoder = SANMEncoder(input_size=8, config=cfg)
    keys = set(flatten_state_dict(encoder))
    assert {
        "embed.conv.0.weight",
        "embed.conv.0.bias",
        "embed.conv.2.weight",
        "embed.conv.2.bias",
        "embed.out.0.weight",
        "embed.out.0.bias",
    } <= keys
    # Conv2d weights use the MLX (out, kh, kw, in) layout.
    assert flatten_state_dict(encoder)["embed.conv.0.weight"].shape == (6, 3, 3, 1)

    out, masks = encoder.embed(mx.zeros((1, 12, 8)), mx.ones((1, 1, 12)))
    assert out.shape == (1, 2, 6)
    assert masks.shape == (1, 1, 2)


def test_sanm_encoder_linear_input_layer_keys():
    cfg = _sanm_encoder_config()
    cfg.input_layer = "linear"
    encoder = SANMEncoder(input_size=8, config=cfg)
    keys = set(flatten_state_dict(encoder))
    assert {"embed.0.weight", "embed.0.bias", "embed.1.weight", "embed.1.bias"} <= keys


# --------------------------------------------------------------------------- #
# CIF predictor
# --------------------------------------------------------------------------- #
def test_cif_predictor_state_dict_keys_and_shapes():
    predictor = CifPredictorV2(idim=8, l_order=1, r_order=1, tail_threshold=0.45)
    state = flatten_state_dict(predictor)
    assert set(state) == {
        "cif_conv1d.weight",
        "cif_conv1d.bias",
        "cif_output.weight",
        "cif_output.bias",
    }
    # CifPredictorV2 uses a *dense* Conv1d (groups=1), unlike V1.
    assert state["cif_conv1d.weight"].shape == (8, 3, 8)
    assert state["cif_output.weight"].shape == (1, 8)


def test_cif_reorder_gather_preserves_the_feature_axis():
    """Regression: ``mx.take`` flattens its input when ``axis`` is omitted.

    ``cif_v1`` gathers whole time slices, so it must pass ``axis=0``; without it
    the feature dimension is dropped and the following reshape blows up.
    """
    hidden = mx.arange(3 * 4, dtype=mx.float32).reshape(3, 4)
    flat = mx.array([2, 0], dtype=mx.int32)
    # Without axis the gather is over a flattened view -> 2 scalars, not 2 rows.
    flattened = mx.take(hidden, flat)
    assert flattened.shape == (2,)
    with_axis = mx.take(hidden, flat, axis=0)
    assert with_axis.shape == (2, 4)
    assert with_axis[0].tolist() == hidden[2].tolist()
    assert with_axis[1].tolist() == hidden[0].tolist()


def test_cif_v1_matches_reference_prefix_sum_formula():
    """Source-derived parity case (adjudicated against cif_predictor.py:818-905).

    alphas   = [0.5, 0.25, 0.5, 0.25, 0.5, 0.125]  (sum 2.125 -> max_label_len 2)
    prefix_sum        = [0.5, 0.75, 1.25, 1.5, 2.0, 2.125]
    prefix_sum_floor  = [0,   0,    1,    1,   2,   2   ]
    rolled_floor      = [0,   0,    0,    1,   1,   2   ]  (roll, then [:,0]=0)
    dislocation_diff  = [0,   0,    1,    0,   1,   0   ]  -> fires at t=2 and t=4
    hidden[t]         = [t + 1, t + 1]
    prefix_sum_hidden = [0.5, 1.0, 2.5, 3.5, 6.0, 6.75]

    frames[j] = P[t_j] - P[t_{j-1}] + rem[t_{j-1}] * h[t_{j-1}] - rem[t_j] * h[t_j]
    frame 0   = [2.5, 2.5] - 0           + 0            - 0.25*[3,3] = [1.75, 1.75]
    frame 1   = [6.0, 6.0] - [2.5, 2.5]  + 0.25*[3,3] - 0.0*[5,5] = [4.25, 4.25]
    """
    alphas = mx.array([[0.5, 0.25, 0.5, 0.25, 0.5, 0.125]])
    hidden = mx.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
    frames, fires = cif_v1(hidden, alphas, 1.0)
    assert frames.shape == (1, 2, 2)
    assert [round(v, 6) for v in frames[0, 0].tolist()] == [1.75, 1.75]
    assert [round(v, 6) for v in frames[0, 1].tolist()] == [4.25, 4.25]
    assert [round(v, 6) for v in fires[0].tolist()] == [0.5, 0.75, 1.25, 0.5, 1.0, 0.125]


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_cif_v1_is_row_independent_for_any_batch(batch):
    """Regression: the flatten-and-gather needs a per-batch row offset.

    The reference adds ``batch_indexes = arange(B).unsqueeze(-1) * len_time`` to
    ``frame_idxs`` before flattening, because ``prefix_sum_hidden[fire_idxs]``
    over a ``(B, T, D)`` tensor indexes row-major.  Without that offset every row
    re-reads slice 0, which is invisible at B=1 and wrong for B>=2.
    """
    hidden = mx.arange(batch * 6 * 3, dtype=mx.float32).reshape(batch, 6, 3) / 7.0
    alphas = mx.broadcast_to(mx.array([[0.5, 0.25, 0.5, 0.25, 0.5, 0.125]], mx.float32), (batch, 6))
    frames, fires = cif_v1(hidden, alphas, 1.0)
    assert frames.shape == (batch, 2, 3)

    for b in range(batch):
        one_frames, one_fires = cif_v1(
            mx.expand_dims(hidden[b], 0), mx.expand_dims(alphas[b], 0), 1.0
        )
        assert mx.allclose(frames[b : b + 1], one_frames, atol=1e-6).item()
        assert mx.allclose(fires[b : b + 1], one_fires, atol=1e-6).item()

    if batch > 1:
        assert not mx.allclose(frames[0], frames[1], atol=1e-6).item()


def test_cif_v1_boundary_form_matches_the_longhand_expression():
    """The simplification must be algebraically identical to the roll/zero form.

    Recomputes the reference expression independently, straight from
    ``cif_wo_hidden_v1``'s source formulas:
        fire_idxs = floor(cumsum(a)) - floor(roll(cumsum(a))) > 0   (shifted)
        rem[t]    = fires[t] - floor(fires[t])
        frame[j]  = P[t_j] - P[t_{j-1}] + rem[t_{j-1}]*h[t_{j-1}] - rem[t_j]*h[t_j]
    with every ``previous`` term zero at j = 0.
    """
    hidden_np = np.arange(2 * 6 * 3, dtype=np.float32).reshape(2, 6, 3) / 7.0
    alphas_np = np.tile(np.array([0.5, 0.25, 0.5, 0.25, 0.5, 0.125], np.float32), (2, 1))
    hidden = mx.array(hidden_np)
    alphas = mx.array(alphas_np)

    frames, fires = cif_v1(hidden, alphas, 1.0)
    mx.eval(frames, fires)
    got = np.asarray(frames)
    fires_np = np.asarray(fires)

    # The reference takes the cumulative sum in float64 before narrowing.
    # Two DIFFERENT prefixes are needed: the scalar alphas drive the fire
    # decisions, while the boundary algebra uses ``prefix_sum_hidden`` -- the
    # cumulative sum of alphas *weighted by* the hidden features, i.e. a
    # (B, T, D) vector.  Conflating them makes the expectation a scalar.
    alpha_prefix = np.cumsum(alphas_np.astype(np.float64), axis=1)
    prefix_hidden = np.cumsum(
        alphas_np.astype(np.float64)[..., None] * hidden_np.astype(np.float64), axis=1
    )
    rolled = np.roll(alpha_prefix, 1, axis=1)
    rolled[:, 0] = 0.0
    fire_idxs = (np.floor(alpha_prefix) - np.floor(rolled)) > 0
    rem = fires_np - np.floor(fires_np)

    for row in range(2):
        times = list(np.flatnonzero(fire_idxs[row]))
        assert times, "fixture must fire"
        for j, t in enumerate(times):
            prev = times[j - 1] if j else None
            expected = prefix_hidden[row, t] - rem[row, t] * hidden_np[row, t]
            if prev is not None:
                expected -= prefix_hidden[row, prev]
                expected += rem[row, prev] * hidden_np[row, prev]
            np.testing.assert_allclose(
                got[row, j], expected, rtol=1e-5, atol=1e-6, err_msg=f"row {row} frame {j}"
            )


def _cold_probe_case(d: int = 3):
    """hidden[t] = [(d*t + j) / 7 for j in range(d)]; the probe's alphas."""
    hidden = mx.arange(6 * d, dtype=mx.float32).reshape(6, d) / 7.0
    alphas = mx.broadcast_to(mx.array([[0.5, 0.25, 0.5, 0.25, 0.5, 0.125]], mx.float32), (2, 6))
    return mx.broadcast_to(hidden[None], (2, 6, d)), alphas


# Hand-derived from the source formula for d=3 (see the derivation below).
# Fire times are t=2 and t=4 because the running sum is
# [0.5, 0.75, 1.25, 1.5, 2.0, 2.125] and floor jumps at 2 and 4, with
# rem[2] = 0.25 and rem[4] = 0.0.  Then boundary[j] = P[t_j] - rem[t_j]*h[t_j]:
#   P[2] = [15/28, 5/7, 25/28], h[2] = [6/7, 1, 8/7]
#     -> boundary[0] = [9/28, 13/28, 17/28]
#   P[4] = [12/7, 2, 16/7], rem[4] = 0
#     -> boundary[1] = [12/7, 2, 16/7]
# and frames[j] = boundary[j] - boundary[j-1] with boundary[-1] = 0, so
#   frames = [[9/28, 13/28, 17/28], [39/28, 43/28, 47/28]]
# Both rows use the same alphas and the same broadcast hidden, so row 1 repeats
# row 0; the frame axis is max_label_len = round(sum(alphas)) = round(2.125) = 2,
# giving a (B=2, 2, D=3) expectation rather than a (2, 3) one.
_COLD_PROBE_EXPECTED_D3 = (np.array([[9, 13, 17], [39, 43, 47]], np.float32) / 28.0)[None].repeat(
    2, axis=0
)


def test_cif_predictor_defaults_to_the_deterministic_cpu_device():
    """The default device is CPU, on both public entry points.

    Metal has been observed to return float bit patterns in the index arrays
    this code depends on, so the default is pinned to CPU rather than left to
    whatever stream the caller happens to be on.
    """
    from funasr_mlx.models import paraformer as paraformer_module

    assert CIF_DEVICE is mx.cpu
    assert paraformer_module.CIF_DEVICE is mx.cpu
    assert paraformer_module._cif_stream(None) is mx.cpu
    for function in (paraformer_module.cif_v1, paraformer_module.cif_wo_hidden_v1):
        assert function.__kwdefaults__["device"] is None


def test_cif_wo_hidden_accumulates_the_prefix_sum_in_float64():
    """Non-dyadic alphas whose running sum reaches 1.0 only in float32.

    The reference is ``torch.cumsum(..., dtype=torch.float64).to(float32)``, so
    the final partial sum stays *below* 1.0 and no integration boundary fires.
    Accumulating in float32 instead rounds it up to exactly 1.0, which would
    emit a token the reference never emits -- so this fixture fails outright on
    the float32 accumulation and passes only with the float64 one.
    """
    # float32 running sums: [0.08405613, 0.4631718, 1.0]           -> fires at t=2
    # float64 running sums: [0.08405613, 0.4631718, 0.99999994]    -> never fires
    alphas = mx.array([[0.08405613154172897, 0.37911567091941833, 0.5368281602859497]], mx.float32)
    fires, fire_idxs = cif_wo_hidden_v1(alphas, 1.0, return_fire_idxs=True)
    mx.eval(fires, fire_idxs)
    assert not bool(mx.any(fire_idxs).item()), "float32 accumulation would fire here"
    assert bool(mx.all(fires < 1.0).item())

    # The predictor agrees: no boundary means no emitted frame, and the
    # zero-fire path still reserves round(sum(alphas)) == 1 column of zeros.
    hidden = mx.arange(1 * 3 * 2, dtype=mx.float32).reshape(1, 3, 2)
    frames, _ = cif_v1(hidden, alphas, 1.0)
    mx.eval(frames)
    assert frames.shape == (1, 1, 2)
    assert bool(mx.all(frames == 0).item())


def test_cif_v1_device_override_is_honoured():
    """Passing the device explicitly must take the same path as the default."""
    hidden, alphas = _cold_probe_case(3)
    default_frames, default_fires = cif_v1(hidden, alphas, 1.0)
    explicit_frames, explicit_fires = cif_v1(hidden, alphas, 1.0, device=mx.cpu)
    mx.eval(default_frames, default_fires, explicit_frames, explicit_fires)
    assert mx.array_equal(default_frames, explicit_frames).item()
    assert mx.array_equal(default_fires, explicit_fires).item()


@pytest.mark.parametrize("seed,b,d", [(1, 1, 2), (2, 2, 3), (3, 3, 8), (4, 2, 5)])
def test_cif_v1_repeated_fresh_rows_match_the_reference_formula(seed, b, d):
    """Fresh per-row inputs, each recomputed by the reference formula in numpy.

    Every call is compared against the formula, never against a previous call,
    so a consistently wrong result cannot pass.  Runs on the default (CPU)
    path three times per case to cover the evaluation this pins.
    """
    rng = np.random.default_rng(seed)
    hidden_np = rng.normal(0.0, 1.0, (b, 6, d)).astype(np.float32)
    alphas_np = rng.uniform(0.15, 0.9, (b, 6)).astype(np.float32)

    alpha_prefix = np.cumsum(alphas_np.astype(np.float64), axis=1)
    rolled = np.roll(alpha_prefix, 1, axis=1)
    rolled[:, 0] = 0.0
    fire_idxs = (np.floor(alpha_prefix) - np.floor(rolled)) > 0
    assert fire_idxs.any(), "fixture must fire at least once"
    fires = (np.where(fire_idxs, 1.0, 0.0) + alpha_prefix - np.floor(alpha_prefix)).astype(
        np.float32
    )
    rem = fires - np.floor(fires)
    prefix = np.cumsum(
        alphas_np.astype(np.float64)[..., None] * hidden_np.astype(np.float64), axis=1
    )

    # The source emits (B, max_label_len, D) with
    #   max_label_len = round(max_b sum_t alphas[b, t])
    # and zero-fills every column past that row's own fire count.  Building the
    # expectation by concatenating all rows' frames loses the batch dimension
    # entirely, so the row's frames must be written into their own row of a
    # zero-filled (B, max_label_len, D) array -- the padding comes from the
    # zeros, which is exactly what the source's `valid` mask produces.
    label_lens = np.round(alphas_np.astype(np.float64).sum(axis=1)).astype(int)
    max_label_len = int(label_lens.max())
    expected = np.zeros((b, max_label_len, d), np.float64)
    for row in range(b):
        times = list(np.flatnonzero(fire_idxs[row]))[:max_label_len]
        previous = np.zeros(d, np.float64)  # boundary[-1] = 0
        for j, t in enumerate(times):
            boundary = prefix[row, t] - rem[row, t] * hidden_np[row, t]
            expected[row, j] = boundary - previous
            previous = boundary

    hidden = mx.array(hidden_np)
    alphas = mx.array(alphas_np)
    for call in range(3):
        frames, got_fires = cif_v1(hidden, alphas, 1.0)
        mx.eval(frames, got_fires)
        np.testing.assert_allclose(
            np.asarray(frames),
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"seed={seed} b={b} d={d} call={call}",
        )
        np.testing.assert_allclose(
            np.asarray(got_fires), fires, rtol=1e-6, atol=1e-6, err_msg=f"seed={seed} call={call}"
        )


def test_cif_wo_hidden_v1_default_device_matches_the_device_override():
    alphas = mx.array([[0.5, 0.25, 0.5, 0.25, 0.5, 0.125]], mx.float32)
    default_fires, default_idxs = cif_wo_hidden_v1(alphas, 1.0, return_fire_idxs=True)
    explicit_fires, explicit_idxs = cif_wo_hidden_v1(
        alphas, 1.0, return_fire_idxs=True, device=mx.cpu
    )
    mx.eval(default_fires, default_idxs, explicit_fires, explicit_idxs)
    assert mx.array_equal(default_fires, explicit_fires).item()
    assert mx.array_equal(default_idxs, explicit_idxs).item()


def test_cif_v1_cold_probe_case_matches_hand_derived_values():
    """Pin the probe's own input against literals derived from the source.

    The previous repeat spec compared every call against the *first* call, which
    cannot detect a cold output that is consistently wrong.  This one compares
    against independently derived constants.
    """
    hidden, alphas = _cold_probe_case(3)
    frames, fires = cif_v1(hidden, alphas, 1.0)
    mx.eval(frames, fires)
    np.testing.assert_allclose(np.asarray(frames), _COLD_PROBE_EXPECTED_D3, rtol=1e-6, atol=1e-6)
    # fires is the cif_peak signal: marks the two boundaries.
    np.testing.assert_allclose(
        np.asarray(fires),
        np.tile(np.array([0.5, 0.75, 1.25, 0.5, 1.0, 0.125], np.float32), (2, 1)),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("d", [3, 8])
def test_cif_v1_cold_and_warm_calls_agree_with_the_reference_formula(d):
    """Every call -- the cold first one included -- must match the formula.

    Deliberately does NOT compare calls to each other: six identical wrong
    answers would pass.  The expectation is recomputed in numpy from
    ``cif_wo_hidden_v1``'s own formulas each time.
    """
    hidden, alphas = _cold_probe_case(d)
    hidden_np = np.asarray(hidden, dtype=np.float64)
    alphas_np = np.asarray(alphas, dtype=np.float64)

    prefix = np.cumsum(alphas_np[..., None] * hidden_np, axis=1)
    alpha_prefix = np.cumsum(alphas_np, axis=1)
    rolled = np.roll(alpha_prefix, 1, axis=1)
    rolled[:, 0] = 0.0
    fire_idxs = (np.floor(alpha_prefix) - np.floor(rolled)) > 0
    fires = (np.where(fire_idxs, 1.0, 0.0) + alpha_prefix - np.floor(alpha_prefix)).astype(
        np.float32
    )
    rem = fires - np.floor(fires)

    for call in range(6):
        frames, got_fires = cif_v1(hidden, alphas, 1.0)
        mx.eval(frames, got_fires)
        expected = np.zeros_like(np.asarray(frames, dtype=np.float64))
        for row in range(2):
            times = list(np.flatnonzero(fire_idxs[row]))
            assert times, "fixture must fire"
            previous = np.zeros(d, np.float64)  # boundary[-1] = 0
            for j, t in enumerate(times):
                boundary = prefix[row, t] - rem[row, t] * hidden_np[row, t]
                expected[row, j] = boundary - previous
                previous = boundary
        np.testing.assert_allclose(
            np.asarray(frames), expected, rtol=1e-5, atol=1e-6, err_msg=f"call {call}"
        )
        np.testing.assert_allclose(
            np.asarray(got_fires),
            fires,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"call {call}",
        )


def test_cif_v1_is_bit_identical_across_repeated_calls():
    """A cold first call must not differ from later ones.

    The fire ordering comes from an argsort; keying every non-fire slot with a
    single sentinel left the tail of the permutation backend- and
    compilation-dependent.  The sort key is now tie-free, so repeated calls --
    and per-utterance calls -- must agree exactly.
    """
    hidden = mx.arange(2 * 6 * 3, dtype=mx.float32).reshape(2, 6, 3) / 7.0
    alphas = mx.broadcast_to(mx.array([[0.5, 0.25, 0.5, 0.25, 0.5, 0.125]], mx.float32), (2, 6))
    first_frames, first_fires = cif_v1(hidden, alphas, 1.0)
    mx.eval(first_frames, first_fires)
    for _ in range(5):
        frames, fires = cif_v1(hidden, alphas, 1.0)
        mx.eval(frames, fires)
        assert mx.array_equal(frames, first_frames).item()
        assert mx.array_equal(fires, first_fires).item()


def test_cif_v1_preserves_source_fire_ordering_in_a_batch():
    """Fires must be emitted per row in increasing time order, not interleaved."""
    alphas = mx.array([[0.5, 0.25, 0.5, 0.25, 0.5, 0.125], [0.25, 0.25, 0.5, 0.5, 0.25, 0.25]])
    hidden = mx.arange(2 * 6 * 2, dtype=mx.float32).reshape(2, 6, 2) / 11.0
    frames, fires = cif_v1(hidden, alphas, 1.0)

    assert mx.allclose(fires[0], mx.array([0.5, 0.75, 1.25, 0.5, 1.0, 0.125]), atol=1e-6).item()
    for b in range(2):
        one, _ = cif_v1(mx.expand_dims(hidden[b], 0), mx.expand_dims(alphas[b], 0), 1.0)
        kept = min(one.shape[1], frames.shape[1])
        assert mx.allclose(frames[b, :kept], one[0, :kept], atol=1e-6).item()


def test_cif_v1_handles_the_empty_fire_case():
    alphas = mx.zeros((1, 5))
    hidden = mx.zeros((1, 5, 3))
    frames, fires = cif_v1(hidden, alphas, 1.0)
    assert frames.shape == (1, 0, 3)
    assert fires.sum().item() == 0.0


def test_cif_predictor_forward_shapes():
    predictor = CifPredictorV2(idim=8, l_order=1, r_order=1, tail_threshold=0.45)
    hidden = mx.random.normal((2, 9, 8)) * 0.5
    mask = mx.expand_dims(~make_pad_mask(mx.array([9, 6]), 9), 1)
    embeds, token_num, alphas, peak = predictor(hidden, mask)
    assert embeds.shape[0] == 2 and embeds.shape[2] == 8
    assert token_num.shape == (2,)
    assert alphas.shape == (2, 10)  # tail_threshold appends one alpha column
    assert peak.shape == (2, 10)


def test_cif_tail_lands_on_the_first_padded_frame_or_the_appended_slot():
    """Regression pin for the exact reference shift trick in ``tail_process_fn``.

    ``mask_2 - mask_1`` is 1.0 at index ``i`` when ``mask[i-1] != mask[i]`` and at
    the appended slot when ``mask[T-1] == 1``; the tail threshold is placed
    there.  So a *full* mask puts it on the appended column, and a *padded* mask
    puts it on the first padded frame (never on the appended column).
    """
    predictor = CifPredictorV2(idim=4, tail_threshold=0.45)
    hidden = mx.zeros((1, 6, 4))
    alphas = mx.zeros((1, 6))

    _, tailed_full, num_full = predictor.tail_process_fn(hidden, alphas, mx.ones((1, 6)))
    assert tailed_full.shape == (1, 7)
    assert [round(v, 6) for v in tailed_full[0].tolist()] == [0, 0, 0, 0, 0, 0, 0.45]
    assert num_full.tolist() == [0.0]

    padded = mx.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])  # ilen == 3
    _, tailed_pad, num_pad = predictor.tail_process_fn(hidden, alphas, padded)
    assert tailed_pad.shape == (1, 7)
    assert [round(v, 6) for v in tailed_pad[0].tolist()] == [0, 0, 0, 0.45, 0, 0, 0]
    assert num_pad.tolist() == [0.0]


def test_cif_tail_without_mask_appends_the_threshold():
    predictor = CifPredictorV2(idim=4, tail_threshold=0.45)
    hidden = mx.zeros((2, 5, 4))
    alphas = mx.zeros((2, 5))
    _, tailed, token_num = predictor.tail_process_fn(hidden, alphas, None)
    assert tailed.shape == (2, 6)
    assert [round(v, 6) for v in tailed[0].tolist()] == [0, 0, 0, 0, 0, 0.45]
    assert [round(v, 6) for v in tailed[1].tolist()] == [0, 0, 0, 0, 0, 0.45]
    assert token_num.tolist() == [0.0, 0.0]


def _deterministic_cif_predictor(tail_threshold: float) -> CifPredictorV2:
    """A CIF predictor whose alphas are a known constant.

    Zeroing both conv/linear weights and setting ``cif_output.bias = 2.0`` makes
    every frame emit ``relu(sigmoid(2.0) - 0.0) == sigmoid(2.0)``, so the alpha
    sum is exactly ``frames * sigmoid(2.0)``.  No random initialisation is
    involved, so the assertions below are exact rather than merely plausible.
    """
    predictor = CifPredictorV2(idim=4, tail_threshold=tail_threshold)
    predictor.cif_conv1d.weight = mx.zeros((4, 3, 4))
    predictor.cif_conv1d.bias = mx.zeros((4,))
    predictor.cif_output.weight = mx.zeros((1, 4))
    predictor.cif_output.bias = mx.array([[2.0]])
    return predictor


def test_cif_token_num_is_not_floored_when_the_tail_is_disabled():
    """Regression: the ``tail_threshold <= 0`` path must return the raw sum.

    Reference ``cif_predictor.py:293`` is ``token_num = alphas.sum(-1)`` with no
    floor; only ``tail_process_fn`` floors (``:461``).  Flooring here would change
    the downstream ``round(token_num)`` applied by ``Paraformer.forward``.
    """
    predictor = _deterministic_cif_predictor(tail_threshold=0.0)
    hidden = mx.zeros((1, 6, 4))
    _, token_num, alphas, _ = predictor(hidden, None)

    expected = 6.0 * (1.0 / (1.0 + math.exp(-2.0)))
    assert mx.allclose(alphas, mx.full((1, 6), 1.0 / (1.0 + math.exp(-2.0))), atol=1e-6).item()
    assert mx.allclose(token_num, mx.array([expected]), atol=1e-6).item()
    # The exact value is fractional, and it must NOT have been floored to 5.0.
    assert expected > math.floor(expected)
    assert not mx.allclose(token_num, mx.array([float(math.floor(expected))]), atol=1e-6).item()


def test_cif_token_num_is_floored_only_on_the_tail_path():
    """With ``tail_threshold > 0`` the floor from ``tail_process_fn`` applies."""
    predictor = _deterministic_cif_predictor(tail_threshold=0.45)
    hidden = mx.zeros((1, 6, 4))
    mask = mx.ones((1, 1, 6))
    _, token_num, alphas, _ = predictor(hidden, mask)

    raw = 6.0 * (1.0 / (1.0 + math.exp(-2.0))) + 0.45  # + the appended tail
    assert mx.allclose(token_num, mx.array([float(math.floor(raw))]), atol=1e-6).item()


# --------------------------------------------------------------------------- #
# review regressions: positional encoding, mask normalisation, containers
# --------------------------------------------------------------------------- #
def test_positional_encoding_interleaves_sin_and_cos():
    """Regression: the reference writes ``pe[:, 0::2]=sin`` / ``pe[:, 1::2]=cos``.

    Concatenating ``[sin, cos]`` instead silently corrupts every column from
    ``d_model // 2`` onward.  Position 0 has phase 0 everywhere, so an
    interleaved row is ``[0, 1, 0, 1, ...]`` while a concatenated row would be
    ``[0] * 4 + [1] * 4`` -- an unambiguous discriminator.
    """
    d_model = 8
    pe = PositionalEncoding(d_model=d_model)
    table = pe._table(3, mx.float32)
    assert table.shape == (3, d_model)
    assert [round(v, 6) for v in table[0].tolist()] == [0.0, 1.0] * (d_model // 2)

    # Even columns are sin, odd columns are cos, at a non-trivial position too.
    div_term = [math.exp(-math.log(10000.0) / d_model * (2 * i)) for i in range(d_model // 2)]
    expected_row1 = []
    for phase in div_term:
        expected_row1 += [math.sin(1.0 * phase), math.cos(1.0 * phase)]
    assert [round(v, 6) for v in table[1].tolist()] == [round(v, 6) for v in expected_row1]


def test_sinusoidal_position_encoder_stays_concatenated():
    """The opposite convention: FunASR's ``SinusoidalPositionEncoder`` truly
    concatenates sin-then-cos.  The two must not be made to match."""
    d_model = 8
    encoder = SinusoidalPositionEncoder(d_model=d_model)
    encoded = encoder.encode(mx.array([[0]]), depth=d_model).squeeze(0).squeeze(0)
    assert [round(v, 6) for v in encoded.tolist()] == [0.0] * 4 + [1.0] * 4
    # ... and it genuinely differs from the interleaved PositionalEncoding table.
    interleaved = PositionalEncoding(d_model=d_model)._table(1, mx.float32).squeeze(0)
    assert [round(v, 6) for v in interleaved.tolist()] == [0.0, 1.0] * 4


def test_attention_blocked_is_dtype_explicit():
    """Regression: boolean masks must block padding too.

    ``mask == 0`` is only meaningful for numeric masks; on a boolean array its
    answer depends on MLX's cross-type comparison promotion, which is not
    something to assume.  ``attention_blocked`` branches on the dtype, so both
    conventions agree without relying on that promotion.
    """
    numeric = mx.array([[1.0, 0.0, 0.0]])
    boolean = mx.array([[True, False, False]])
    for mask in (numeric, boolean):
        blocked = attention_blocked(mask)
        assert blocked.dtype == mx.bool_
        assert blocked.shape == (1, 1, 3)
        assert blocked.tolist() == [[[False, True, True]]]


def test_conv2d_subsampling_is_batch_size_independent():
    """Regression: ``ModuleStack`` must not unpack the batch axis.

    ``conv = [Conv2d, ReLU, Conv2d, ReLU]`` returns a single array, so feeding a
    block's return value back as ``*args`` would hand the next block ``B``
    arguments.  That only worked at batch size 1.
    """
    sub = Conv2dSubsampling(idim=8, odim=6)
    x1 = mx.arange(1 * 12 * 8, dtype=mx.float32).reshape(1, 12, 8) / 97.0
    x2 = mx.concatenate([x1, x1], axis=0)

    out1, mask1 = sub(x1, mx.ones((1, 1, 12)))
    out2, mask2 = sub(x2, mx.ones((2, 1, 12)))

    assert out1.shape == (1, 2, 6)
    assert out2.shape == (2, 2, 6)
    assert mask1.shape == (1, 1, 2)
    assert mask2.shape == (2, 1, 2)
    assert mx.allclose(out2[0], out1[0], atol=1e-5).item()
    assert mx.allclose(out2[1], out1[0], atol=1e-5).item()


def test_module_stack_threads_the_leading_argument_only():
    """Regression: the chain keeps trailing args/keywords fixed across blocks."""
    seen = []

    class _Spy(mxnn.Module):
        def __init__(self, tag: str) -> None:
            super().__init__()
            self._tag = tag

        def __call__(self, x: mx.array, extra=None):
            seen.append((self._tag, tuple(x.shape), extra))
            return x + 1

    stack = ModuleStack([_Spy("a"), _Spy("b")])
    out = stack(mx.zeros((4, 3)), extra="k")
    assert out.shape == (4, 3)
    assert mx.allclose(out, mx.full((4, 3), 2.0)).item()
    assert [tag for tag, _, _ in seen] == ["a", "b"]
    # Each block received the whole batch, not B separate scalars.
    assert all(shape == (4, 3) for _, shape, _ in seen)
    assert all(extra == "k" for _, _, extra in seen)
    with pytest.raises(TypeError):
        stack()


def _valid_output_rows(lens, width):
    """``(B, 1, 1)`` mask selecting the output rows that correspond to real tokens.

    Masking is only defined over *valid* output positions. A padded output row
    has no defined value, so requiring it to match would be claiming a property
    the reference never had. Corrupting a padded **input** frame cannot change
    a valid output row; nothing constrains the padded output rows themselves.
    """
    return mx.expand_dims(mx.arange(width)[None, :] < lens[:, None], -1)


def test_decoder_output_is_invariant_to_the_padded_memory_tail():
    """Regression: a padded memory frame must not influence the logits.

    The offline Paraformer decoder consumes the **CIF acoustic embedding**, not
    token ids: the reference ``ParaformerSANMDecoder.forward`` does ``x = tgt``
    with ``tgt = ys_in_pad`` and never calls ``self.embed`` (only
    ``forward_one_step`` does, at decoder.py:604).  So both ``hs_pad`` and
    ``ys_in_pad`` must be ``(B, T, attention_dim)`` here.
    """
    decoder = ParaformerSANMDecoder(
        vocab_size=13, encoder_output_size=8, config=_paraformer_config().decoder_conf
    )
    decoder.eval()
    tgt = mx.arange(2 * 4 * 8, dtype=mx.float32).reshape(2, 4, 8) / 41.0
    tgt_lens = mx.array([4, 2])
    memory = mx.arange(2 * 8 * 8, dtype=mx.float32).reshape(2, 8, 8) / 53.0
    hlens = mx.array([5, 3])

    clean, _ = decoder(memory, hlens, tgt, tgt_lens)

    # Ragged batch: hlens = [5, 3].  Corrupt *each row's own* padded frames only
    # (frames >= hlens[b]) and leave every valid frame untouched.  A correctly
    # masked decoder must return identical logits.
    padded = mx.logical_not(mx.arange(8)[None, :] < hlens[:, None])  # (2, 8)
    corrupted = mx.where(mx.expand_dims(padded, -1), memory + 1000.0, memory)
    dirty, _ = decoder(corrupted, hlens, tgt, tgt_lens)
    # Valid output rows only; the tolerance is unchanged, not loosened.
    valid = _valid_output_rows(tgt_lens, clean.shape[1])
    assert mx.allclose(
        mx.where(valid, clean, mx.array(0.0)), mx.where(valid, dirty, mx.array(0.0)), atol=1e-4
    ).item()
    # Sanity: the corruption really did reach the padded region.
    assert not mx.allclose(mx.sum(corrupted, axis=-1), mx.sum(memory, axis=-1)).item()


def test_decoder_output_is_invariant_to_the_padded_target_tail():
    """The same property for the self-attention (``tgt_mask``) side of the decoder."""
    decoder = ParaformerSANMDecoder(
        vocab_size=13, encoder_output_size=8, config=_paraformer_config().decoder_conf
    )
    decoder.eval()
    tgt = mx.arange(2 * 4 * 8, dtype=mx.float32).reshape(2, 4, 8) / 41.0
    tgt_lens = mx.array([4, 2])
    memory = mx.arange(2 * 8 * 8, dtype=mx.float32).reshape(2, 8, 8) / 53.0
    hlens = mx.array([5, 3])

    clean, _ = decoder(memory, hlens, tgt, tgt_lens)
    padded = mx.logical_not(mx.arange(4)[None, :] < tgt_lens[:, None])  # (2, 4)
    corrupted = mx.where(mx.expand_dims(padded, -1), tgt + 1000.0, tgt)
    dirty, _ = decoder(memory, hlens, corrupted, tgt_lens)
    valid = _valid_output_rows(tgt_lens, clean.shape[1])
    assert mx.allclose(
        mx.where(valid, clean, mx.array(0.0)), mx.where(valid, dirty, mx.array(0.0)), atol=1e-4
    ).item()


def test_cif_predictor_config_defaults_follow_paraformer():
    cfg = CifPredictorConfig()
    assert (cfg.threshold, cfg.l_order, cfg.r_order, cfg.tail_threshold) == (
        1.0,
        1,
        1,
        0.45,
    )


# --------------------------------------------------------------------------- #
# Paraformer decoder
# --------------------------------------------------------------------------- #
def test_paraformer_decoder_state_dict_matches_pytorch_naming():
    decoder = ParaformerSANMDecoder(
        vocab_size=13, encoder_output_size=8, config=_paraformer_config().decoder_conf
    )
    keys = set(flatten_state_dict(decoder))
    expected = {
        "embed.0.weight",
        "after_norm.weight",
        "after_norm.bias",
        "output_layer.weight",
        "output_layer.bias",
    }
    for i in range(2):
        expected |= _decoder_layer_keys(f"decoders.{i}", with_src_attn=True, with_self_attn=True)
    expected |= _decoder_layer_keys("decoders2.0", with_src_attn=False, with_self_attn=True)
    expected |= _decoder_layer_keys("decoders3.0", with_src_attn=False, with_self_attn=False)
    assert keys == expected
    # ``w_2`` of PositionwiseFeedForwardDecoderSANM has no bias.
    assert "decoders.0.feed_forward.w_2.bias" not in keys


def test_paraformer_decoder_has_no_decoders2_when_blocks_match():
    cfg = _paraformer_config()
    cfg.decoder_conf.num_blocks = cfg.decoder_conf.att_layer_num
    decoder = ParaformerSANMDecoder(vocab_size=13, encoder_output_size=8, config=cfg.decoder_conf)
    assert decoder.decoders2 is None
    assert not any(k.startswith("decoders2") for k in flatten_state_dict(decoder))


def test_paraformer_decoder_forward_shapes():
    decoder = ParaformerSANMDecoder(
        vocab_size=13, encoder_output_size=8, config=_paraformer_config().decoder_conf
    )
    memory = mx.zeros((2, 9, 8))
    memory_lens = mx.array([9, 6])
    acoustic = mx.zeros((2, 4, 8))
    token_lens = mx.array([4, 3])
    logits, olens = decoder(memory, memory_lens, acoustic, token_lens)
    assert logits.shape == (2, 4, 13)
    assert olens.tolist() == [4, 3]


# --------------------------------------------------------------------------- #
# Paraformer model
# --------------------------------------------------------------------------- #
def test_paraformer_state_dict_top_level_layout():
    model = _paraformer()
    keys = set(flatten_state_dict(model))
    prefixes = {k.split(".")[0] for k in keys}
    assert prefixes == {"encoder", "decoder", "predictor"}
    assert "predictor.cif_conv1d.weight" in keys
    assert "decoder.output_layer.weight" in keys
    assert "encoder.after_norm.bias" in keys


def test_paraformer_full_forward_shapes():
    model = _paraformer()
    speech = mx.random.normal((2, 11, 8)) * 0.3
    lengths = mx.array([11, 8])
    outs = model(speech, lengths)
    assert outs.encoder_out.shape == (2, 11, 8)
    assert outs.encoder_out_lens.tolist() == [11, 8]
    assert outs.acoustic_embeds.shape[0] == 2 and outs.acoustic_embeds.shape[2] == 8
    assert outs.token_length.shape == (2,)
    if outs.decoder_out is not None:
        assert outs.decoder_out.shape[0] == 2
        assert outs.decoder_out.shape[2] == 13


def test_paraformer_greedy_decode_shape_is_bounded_by_token_length():
    model = _paraformer()
    speech = mx.random.normal((1, 10, 8)) * 0.3
    tokens = model.greedy_decode(speech, mx.array([10]))
    assert len(tokens) == 1
    assert all(isinstance(t, int) and 0 <= t < 13 for t in tokens[0])


def test_paraformer_config_from_funasr_document():
    cfg = paraformer_config_from_dict(
        {
            "model": "Paraformer",
            "model_conf": {"input_size": 560, "vocab_size": 8408, "ctc_weight": 0.0},
            "encoder_conf": {"output_size": 512, "num_blocks": 50, "input_layer": "pe"},
            "decoder_conf": {"att_layer_num": 16, "num_blocks": 16, "kernel_size": 11},
            "predictor_conf": {"idim": 512, "tail_threshold": 0.45},
        }
    )
    assert cfg.input_size == 560 and cfg.vocab_size == 8408
    assert cfg.encoder_conf.output_size == 512 and cfg.encoder_conf.num_blocks == 50
    assert cfg.decoder_conf.att_layer_num == 16
    assert cfg.predictor_conf.idim == 512 and cfg.predictor_conf.tail_threshold == 0.45


def test_paraformer_predictor_idim_defaults_to_encoder_width():
    cfg = paraformer_config_from_dict({"encoder_conf": {"output_size": 64}})
    assert cfg.predictor_conf.idim == 64


# --------------------------------------------------------------------------- #
# SenseVoice
# --------------------------------------------------------------------------- #
def test_sensevoice_metadata_tables_match_the_reference():
    assert NUM_METADATA_EMBEDDINGS == 16  # 7 + len(lid_dict) + len(textnorm_dict)
    assert NUM_PREFIX_FRAMES == 4  # language, event, emotion, textnorm
    assert LID_DICT == {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
    assert TEXTNORM_DICT == {"withitn": 14, "woitn": 15}
    assert EMO_DICT["unk"] == 25009
    assert EMO_DICT["neutral"] == 25004


def test_sensevoice_encoder_state_dict_matches_pytorch_naming():
    encoder = SenseVoiceEncoderSmall(input_size=12, config=_sensevoice_encoder_config())
    keys = set(flatten_state_dict(encoder))
    expected = set()
    expected |= _encoder_layer_keys("encoders0.0")
    expected |= _encoder_layer_keys("encoders.0")  # num_blocks - 1 == 1
    expected |= {"after_norm.weight", "after_norm.bias"}
    expected |= _encoder_layer_keys("tp_encoders.0")
    expected |= {"tp_norm.weight", "tp_norm.bias"}
    assert keys == expected


def test_sensevoice_model_state_dict_top_level_layout():
    model = SenseVoiceSmall(
        config=SenseVoiceSmallConfig(
            input_size=12,
            vocab_size=32,
            encoder_conf=_sensevoice_encoder_config(),
        )
    )
    keys = set(flatten_state_dict(model))
    assert {k.split(".")[0] for k in keys} == {"embed", "encoder", "ctc"}
    assert keys & {"embed.weight", "ctc.ctc_lo.weight", "ctc.ctc_lo.bias", "encoder.tp_norm.weight"}


def test_sensevoice_prefix_is_four_frames_then_speech():
    model = SenseVoiceSmall(
        config=SenseVoiceSmallConfig(
            input_size=12,
            vocab_size=32,
            encoder_conf=_sensevoice_encoder_config(),
        )
    )
    assert model.embed.weight.shape == (NUM_METADATA_EMBEDDINGS, 12)
    prefix = model._metadata_prefix(batch_size=3, language="zh", use_itn=False, text_norm=None)
    assert prefix.shape == (3, NUM_PREFIX_FRAMES, 12)
    # [language=3, event=1, emotion=2, textnorm=woitn=15]
    expected = model.embed(mx.array([[3, 1, 2, 15]]))
    for b in range(3):
        assert mx.allclose(prefix[b], expected[0]).item()


def test_sensevoice_forward_and_greedy_decode_shapes():
    model = SenseVoiceSmall(
        config=SenseVoiceSmallConfig(
            input_size=12,
            vocab_size=32,
            encoder_conf=_sensevoice_encoder_config(),
        )
    )
    speech = mx.random.normal((2, 9, 12)) * 0.3
    lengths = mx.array([9, 6])
    logits = model(speech, lengths, language="auto", use_itn=True)
    assert logits.shape == (2, 9 + NUM_PREFIX_FRAMES, 32)
    tokens = model.greedy_decode(speech, lengths)
    assert len(tokens) == 2
    assert all(0 <= t < 32 for row in tokens for t in row)


def test_sensevoice_config_from_funasr_document():
    cfg = sensevoice_config_from_dict(
        {
            "model": "SenseVoiceSmall",
            "model_conf": {"input_size": 560, "vocab_size": 55465, "length_normalized_loss": True},
            "encoder_conf": {
                "output_size": 512,
                "num_blocks": 50,
                "tp_blocks": 20,
                "attention_heads": 4,
                "linear_units": 2048,
                "kernel_size": 11,
            },
        }
    )
    assert cfg.input_size == 560 and cfg.vocab_size == 55465
    assert cfg.encoder_conf.num_blocks == 50 and cfg.encoder_conf.tp_blocks == 20


def test_ctc_greedy_decode_collapses_before_dropping_blanks():
    # 1 -> 0 (blank) -> 1 must decode to two ``1`` tokens, matching
    # ``unique_consecutive`` followed by a blank filter.
    log_probs = mx.log(
        mx.array(
            [
                [
                    [0.1, 0.9],  # -> 1
                    [0.9, 0.1],  # -> 0 (blank)
                    [0.1, 0.9],  # -> 1
                    [0.9, 0.1],  # -> 0 (blank)
                    [0.1, 0.9],  # -> 1
                ]
            ]
        )
    )
    assert ctc_greedy_decode(log_probs, mx.array([5]), blank_id=0) == [[1, 1, 1]]


def test_ctc_greedy_decode_respects_lengths():
    # Deterministic known logits, so the assertion has teeth: both rows have
    # *non-blank* argmax tokens inside their valid frames and deliberately
    # different spurious tokens in the padded tail.  Lengths must truncate them.
    #   row 0 (ilen=3): argmax [1, 2, 1, 3]  -> tail's 3 dropped -> [1, 2, 1]
    #   row 1 (ilen=1): argmax [2, 3, 3, 1]  -> tail dropped, and the repeated
    #                                 3,3 collapses to one via unique_consecutive
    log_probs = mx.log(
        mx.array(
            [
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
            ]
        )
    )
    assert mx.argmax(log_probs, axis=-1).tolist() == [[1, 2, 1, 3], [2, 3, 3, 1]]
    assert ctc_greedy_decode(log_probs, mx.array([3, 1]), blank_id=0) == [[1, 2, 1], [2]]
    # Decoding the full padded batch instead leaks the tails (and collapses 3,3).
    assert ctc_greedy_decode(log_probs, mx.array([4, 4]), blank_id=0) == [
        [1, 2, 1, 3],
        [2, 3, 1],
    ]


# --------------------------------------------------------------------------- #
# strict loading
# --------------------------------------------------------------------------- #
def test_load_strict_round_trips_and_rejects_mismatch():
    model = _paraformer()
    state = flatten_state_dict(model)

    clone = _paraformer()
    loaded = load_strict(clone, state)
    assert len(loaded) == len(state)

    missing = dict(state)
    missing.pop("decoder.output_layer.weight")
    with pytest.raises(CheckpointKeyError):
        load_strict(_paraformer(), missing)

    unexpected = dict(state)
    unexpected["decoder.not_a_real_weight"] = mx.zeros((1,))
    with pytest.raises(CheckpointKeyError):
        load_strict(_paraformer(), unexpected)


def test_load_strict_accepts_a_shared_prefix_and_rejects_shape_mismatch():
    model = _paraformer()
    state = {f"model.{k}": v for k, v in flatten_state_dict(model).items()}
    assert len(load_strict(_paraformer(), state, prefix="model.")) == len(state)

    bad = dict(state)
    bad["model.decoder.output_layer.weight"] = mx.zeros((2, 2))
    with pytest.raises(ValueError):
        load_strict(_paraformer(), bad, prefix="model.")


def _torch_layout_state(model):
    """Full PyTorch-layout state derived from the module-derived layout rules.

    ``convert=True`` applies a layout rule to *every* conv a module owns, so a
    fixture that transposes only one conv is a mixed layout and cannot load.
    Two details matter:

    * only ``.weight`` is transposed. The rule is derived per owning module, so
      it also labels the matching **bias**, which is 1-D and must be left alone.
    * MLX Conv2d stores ``(out, KH, KW, in)`` and torch Conv2d stores
      ``(out, in, KH, KW)``, so the round-trip is ``(0, 3, 1, 2)`` -- not
      ``(0, 2, 3, 1)``, which reorders to ``(out, KW, in, KH)``.
    """
    state = dict(flatten_state_dict(model))
    _, rules = flatten_state_dict_with_rules(model)
    torch_state = {}
    for name, tensor in state.items():
        rule = rules[name]
        if not name.endswith(".weight") or tensor.ndim not in (3, 4):
            torch_state[name] = tensor
        elif rule == CONV1D_LAYOUT:
            torch_state[name] = mx.transpose(tensor, (0, 2, 1))
        elif rule == CONV2D_LAYOUT:
            torch_state[name] = mx.transpose(tensor, (0, 3, 1, 2))
        else:
            torch_state[name] = tensor
    return state, torch_state, rules


class _MixedConvHolder(mxnn.Module):
    """A holder mixing both conv kinds so one module exercises both rules."""

    def __init__(self) -> None:
        super().__init__()
        # mxnn.Conv1d takes a SCALAR kernel_size (convolution.py:39,
        # ``kernel_size: int``); only Conv2d accepts a tuple.
        self.conv1 = mxnn.Conv1d(3, 5, 3, bias=True)
        self.conv2 = mxnn.Conv2d(2, 4, (3, 5), bias=True)  # deliberately non-square


def _mixed_conv_holder():
    """Distinct, non-uniform parameters so any axis mix-up is visible.

    The Conv2d kernel is 3x5 rather than square, which is what makes the
    permutations distinguishable at all: on a square kernel with equal
    channels, ``(0,3,1,2)`` and a wrong ordering can coincide.
    """
    holder = _MixedConvHolder()
    holder.conv1.weight = mx.arange(5 * 3 * 3, dtype=mx.float32).reshape(5, 3, 3) / 17.0
    holder.conv1.bias = mx.arange(5, dtype=mx.float32) / 3.0 + 1.0
    holder.conv2.weight = mx.arange(4 * 2 * 3 * 5, dtype=mx.float32).reshape(4, 3, 5, 2) / 23.0
    holder.conv2.bias = mx.arange(4, dtype=mx.float32) / 5.0 - 1.0
    return holder


@pytest.mark.parametrize(
    "name,rule,torch_shape",
    [
        ("conv1.weight", CONV1D_LAYOUT, (5, 3, 3)),
        ("conv1.bias", IDENTITY_LAYOUT, (5,)),
        ("conv2.weight", CONV2D_LAYOUT, (4, 2, 3, 5)),
        ("conv2.bias", IDENTITY_LAYOUT, (4,)),
    ],
)
def test_conv_layout_rule_applies_to_the_weight_only(name, rule, torch_shape):
    """A conv's 1-D bias must be IDENTITY in both layouts.

    The rule follows the *owning module*, so before this was pinned the bias
    inherited the kernel rule and ``load_strict(convert=True)`` transposed a
    1-D array with three axes -- a ValueError on every conv that has a bias,
    which is essentially every conv in a real checkpoint.
    """
    tensors, rules = flatten_state_dict_with_rules(_mixed_conv_holder())
    assert rules[name] == rule
    tensor = tensors[name]
    assert tuple(tensor.shape) != tuple(torch_shape) or rule == IDENTITY_LAYOUT


def test_standalone_conv_weights_and_biases_round_trip_through_the_loader():
    """One end-to-end load of a full torch-layout conv state, weights and biases.

    Built from the module-derived rules rather than by hand, so it exercises the
    same public path a real checkpoint takes: ``flatten_state_dict_with_rules``
    -> torch state -> ``load_strict(convert=True)`` -> exact round trip.
    """
    source = _mixed_conv_holder()
    tensors, rules = flatten_state_dict_with_rules(source)
    torch_state = {}
    for name, tensor in tensors.items():
        rule = rules[name]
        if rule == CONV1D_LAYOUT:
            torch_state[name] = mx.transpose(tensor, (0, 2, 1))
        elif rule == CONV2D_LAYOUT:
            torch_state[name] = mx.transpose(tensor, (0, 3, 1, 2))
        else:
            torch_state[name] = tensor

    # The torch layouts must genuinely differ, or the round trip proves nothing.
    assert tuple(torch_state["conv1.weight"].shape) == (5, 3, 3)
    assert tuple(torch_state["conv2.weight"].shape) == (4, 2, 3, 5)
    assert tuple(torch_state["conv1.bias"].shape) == (5,)
    assert tuple(torch_state["conv2.bias"].shape) == (4,)

    clone = _MixedConvHolder()
    load_strict(clone, torch_state, convert=True)
    restored = flatten_state_dict(clone)
    for name, tensor in tensors.items():
        assert mx.array_equal(restored[name], tensor).item(), name


def test_load_strict_converts_torch_conv_layout_on_load():
    model = _paraformer()
    state, torch_state, rules = _torch_layout_state(model)
    key = "encoder.encoders.0.self_attn.fsmn_block.weight"
    assert rules[key] == CONV1D_LAYOUT
    assert torch_state[key].shape == (8, 1, 3)
    clone = _paraformer()
    load_strict(clone, torch_state, convert=True)
    assert clone.encoder.encoders[0].self_attn.fsmn_block.weight.shape == (8, 3, 1)
    for name, tensor in state.items():
        assert mx.array_equal(flatten_state_dict(clone)[name], tensor).item()


@pytest.mark.parametrize("layer", ["pe", "conv2d"])
def test_load_strict_converts_every_conv_owner(layer):
    """``convert=True`` transposes **every** conv, so the fixture must as well.

    A fixture that flips one conv and leaves the rest in MLX layout is a mixed
    layout: the untouched convs then fail their shape check, because
    ``convert=True`` applies the rule globally.  This builds a fully torch-layout
    state from the module-derived rules and round-trips all of it.
    """
    config = _paraformer_config()
    config.encoder_conf.input_layer = layer
    model = Paraformer(config=config)
    state, torch_state, rules = _torch_layout_state(model)

    # Weights only: the rule also labels the 1-D bias, which is never transposed.
    conv_keys = sorted(
        name
        for name in state
        if rules[name] != IDENTITY_LAYOUT
        and name.endswith(".weight")
        and state[name].ndim in (3, 4)
    )
    assert conv_keys, "fixture must exercise at least one conv"
    if layer == "conv2d":
        # The conv2d branch only exists for a conv-subsampling encoder.
        assert "encoder.embed.conv.0.weight" in conv_keys

    clone_config = _paraformer_config()
    clone_config.encoder_conf.input_layer = layer
    clone = Paraformer(config=clone_config)
    load_strict(clone, torch_state, convert=True)

    loaded = flatten_state_dict(clone)
    for name in conv_keys:
        assert mx.array_equal(loaded[name], state[name]).item()


def test_load_strict_converts_the_decoder_depthwise_fsmn_conv():
    """The decoder's depthwise FSMN conv must be converted like the encoder's.

    Both are plain ``mxnn.Conv1d``; a name-regex or single-site fix would leave
    the decoder key in torch layout and fail with (8, 1, 3) vs (8, 3, 1).
    """
    model = _paraformer()
    state = dict(flatten_state_dict(model))
    key = "decoder.decoders.0.self_attn.fsmn_block.weight"
    assert key in state
    torch_state = dict(state)
    torch_state[key] = mx.transpose(state[key], (0, 2, 1))
    assert tuple(torch_state[key].shape) == (8, 1, 3)
    # Mixed layout is rejected: convert=True also transposes the other convs.
    with pytest.raises(ValueError, match="shape mismatch"):
        load_strict(_paraformer(), torch_state, convert=True)


def test_load_strict_does_not_convert_by_default():
    """MLX-layout tensors must pass through untouched (no double transpose)."""
    model = _paraformer()
    state = dict(flatten_state_dict(model))
    key = "encoder.encoders.0.self_attn.fsmn_block.weight"
    clone = _paraformer()
    load_strict(clone, state)
    loaded = flatten_state_dict(clone)
    assert loaded[key].shape == state[key].shape == (8, 3, 1)
