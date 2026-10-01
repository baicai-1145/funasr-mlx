"""Tiny, offline, network-free tests for the CAM++ MLX port.

Deliberately tiny: single-digit feature dims and a handful of frames, so the
whole file stays far below the project memory ceiling.  These are structural /
algebraic checks -- checkpoint naming, layer ordering, layout rules, mask and
pooling arithmetic -- **not** real-checkpoint acceptance.
"""

from __future__ import annotations

import math

import pytest

mx = pytest.importorskip("mlx.core", reason="mlx is required for the model ports")

from mlx import nn as mxnn  # noqa: E402

from funasr_mlx.models.campplus import (  # noqa: E402
    CAMPlusConfig,
    CAMPPlus,
    BatchNorm1d,
    CAMLayer,
    CheckpointKeyError,
    LoadReport,
    NamedModuleDict,
    convert_checkpoint_tensor,
    flatten_state_dict,
    load_strict,
    statistics_pooling,
)


def _tiny_config(**overrides) -> CAMPlusConfig:
    base = dict(
        feat_dim=8,
        embedding_size=6,
        growth_rate=2,
        bn_size=2,
        init_channels=4,
        m_channels=4,
        num_blocks=(2, 2),
        block_spec=((2, 3, 1), (2, 3, 2)),
        output_level="segment",
    )
    base.update(overrides)
    return CAMPlusConfig.from_dict(base)


def _campplus(**overrides) -> CAMPPlus:
    return CAMPPlus(config=_tiny_config(**overrides))


# --------------------------------------------------------------------------- #
# checkpoint plumbing
# --------------------------------------------------------------------------- #
def test_flatten_state_dict_reads_the_mapping_storage_not_dunder_dict():
    """Regression: ``mlx.nn.Module`` subclasses ``dict``.

    Arrays and sub-modules live in the mapping storage, so a ``vars()``-based
    traversal silently yields an empty dict.  Pin it for a bare leaf, a nested
    container and a full model.
    """
    leaf = mxnn.Linear(3, 2)
    assert not any(isinstance(v, mx.array) for v in vars(leaf).values())
    assert set(flatten_state_dict(leaf)) == {"weight", "bias"}
    assert set(dict.keys(leaf)) == {"weight", "bias"}

    nested = NamedModuleDict()
    nested.add("first", mxnn.Linear(3, 2))
    nested.add("second", mxnn.Linear(2, 2))
    assert set(flatten_state_dict(nested)) == {
        "first.weight",
        "first.bias",
        "second.weight",
        "second.bias",
    }

    state = flatten_state_dict(_campplus())
    assert state, "the model must expose parameters"
    assert "head.conv1.weight" in state
    assert "xvector.dense.linear.weight" in state


def test_state_dict_keys_carry_the_xvector_prefix():
    """The reference wraps everything after ``head`` in ``self.xvector``."""
    keys = set(flatten_state_dict(_campplus()))
    assert "xvector.tdnn.linear.weight" in keys
    assert "xvector.block1.tdnnd1.linear1.weight" in keys
    assert "xvector.block1.tdnnd2.cam_layer.linear_local.weight" in keys
    assert "xvector.transit1.linear.weight" in keys
    assert "xvector.out_nonlinear.0.weight" in keys  # BN weight, affine
    assert "xvector.dense.linear.weight" in keys
    # batchnorm_ is non-affine, so it contributes only the running buffers.
    assert "xvector.dense.nonlinear.0.weight" not in keys
    assert "xvector.dense.nonlinear.0.running_mean" in keys
    assert not any(key.startswith("_") for key in keys)


def test_convert_checkpoint_tensor_transposes_only_conv_kernels():
    conv1d = mx.zeros((4, 1, 5))  # torch (out, in/groups, k)
    assert convert_checkpoint_tensor("x.tdnn.linear.weight", conv1d).shape == (4, 5, 1)
    conv2d = mx.zeros((4, 1, 3, 3))  # torch (out, in, kh, kw)
    assert convert_checkpoint_tensor("head.conv1.weight", conv2d).shape == (4, 3, 3, 1)
    linear = mx.zeros((6, 9))
    assert convert_checkpoint_tensor("xvector.dense.linear.weight", linear).shape == (6, 9)


def test_load_strict_round_trips_and_reports_torch_only_buffers():
    model = _campplus()
    state = flatten_state_dict(model)

    clone = _campplus()
    loaded = load_strict(clone, {k: v for k, v in state.items()})
    assert loaded == sorted(state)
    for key, value in state.items():
        assert mx.array_equal(flatten_state_dict(clone)[key], value).item()

    # A raw torch state_dict carries BatchNorm's num_batches_tracked, which MLX
    # does not model; it must be reported rather than silently swallowed.
    torch_state = {k: v for k, v in state.items()}
    torch_state["head.bn1.num_batches_tracked"] = mx.array(0.0)
    report = load_strict(_campplus(), torch_state)
    assert isinstance(report, LoadReport)
    assert report.dropped == ["head.bn1.num_batches_tracked"]

    bad = dict(state)
    bad.pop("xvector.dense.linear.weight")
    bad["not.a.real.key"] = mx.zeros((1,))
    with pytest.raises(CheckpointKeyError):
        load_strict(_campplus(), bad)


# --------------------------------------------------------------------------- #
# architecture
# --------------------------------------------------------------------------- #
def test_xvector_stage_order_is_interleaved():
    """Regression: the reference is block1, transit1, block2, transit2, ...

    Iterating all blocks before all transits would desynchronise the channel
    arithmetic, since each transit halves the channels feeding the next block.
    """
    model = _campplus()
    assert model.xvector.names() == [
        "tdnn",
        "block1",
        "transit1",
        "block2",
        "transit2",
        "out_nonlinear",
        "stats",
        "dense",
    ]


def test_fcm_divides_the_mel_axis_by_eight_and_keeps_time():
    """FCM treats mel as height and time as width; only mel is decimated."""
    model = _campplus()
    assert model.head.out_channels == model.config.m_channels * (model.config.feat_dim // 8)
    fbank = mx.arange(8 * 2, dtype=mx.float32).reshape(1, 2, 8) / 17.0
    out = model.head(mx.transpose(fbank, (0, 2, 1)))  # (B, F, T)
    assert out.shape == (1, 2, model.head.out_channels)  # time preserved


def test_forward_shapes_for_segment_and_frame_output():
    fbank = mx.arange(1 * 8 * 8, dtype=mx.float32).reshape(1, 8, 8) / 23.0
    segment = _campplus()
    segment.eval()
    assert segment(fbank).shape == (1, 6)  # embedding_size

    frame = _campplus(output_level="frame")
    frame.eval()
    frames = frame(fbank)
    assert frames.ndim == 3
    assert frames.shape[0] == 1
    assert frames.shape[2] == frame._frame_channels


def test_embed_matches_forward_for_a_single_utterance():
    fbank = mx.arange(1 * 8 * 8, dtype=mx.float32).reshape(1, 8, 8) / 29.0
    model = _campplus()
    model.eval()
    lengths = mx.array([8])
    assert mx.allclose(model.embed(fbank, lengths), model(fbank), atol=1e-6).item()


def test_embed_is_invariant_to_the_padded_tail():
    """Each row's padded frames must not leak into its segment statistics."""
    model = _campplus()
    model.eval()
    fbank = mx.arange(2 * 8 * 8, dtype=mx.float32).reshape(2, 8, 8) / 31.0
    lengths = mx.array([8, 5])
    clean = model.embed(fbank, lengths)

    padded = mx.logical_not(mx.arange(8)[None, :] < lengths[:, None])
    corrupted = mx.where(mx.expand_dims(padded, -1), fbank + 1000.0, fbank)
    dirty = model.embed(corrupted, lengths)
    assert mx.allclose(clean, dirty, atol=1e-4).item()


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def test_batchnorm_eval_uses_running_statistics():
    bn = BatchNorm1d(3)
    bn.weight = mx.array([1.0, 2.0, 3.0])
    bn.bias = mx.array([0.5, 0.0, -0.5])
    bn.running_mean = mx.array([1.0, 2.0, 3.0])
    bn.running_var = mx.array([4.0, 5.0, 6.0])
    bn.eval()

    x = mx.array([[[2.0, 4.0, 9.0]]])  # (B=1, L=1, C=3)
    y = bn(x)
    eps = 1e-5
    for c, (value, weight, bias, mean, var) in enumerate(
        zip([2.0, 4.0, 9.0], [1.0, 2.0, 3.0], [0.5, 0.0, -0.5], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    ):
        expected = (value - mean) / math.sqrt(var + eps) * weight + bias
        assert y[0, 0, c].item() == pytest.approx(expected, abs=1e-5)


def test_non_affine_batchnorm_exposes_only_running_buffers():
    bn = BatchNorm1d(2, affine=False)
    state = flatten_state_dict(bn)
    assert set(state) == {"running_mean", "running_var"}


def test_statistics_pooling_reduces_time_and_concatenates():
    x = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]])  # (1, 3, 2)
    pooled = statistics_pooling(x)
    assert pooled.shape == (1, 1, 4)
    mean = [1.0 + 3.0 + 5.0, 2.0 + 4.0 + 7.0]
    mean = [m / 3.0 for m in mean]
    var = [
        sum((v - m) ** 2 for v, m in zip(col, mean)) / 2.0
        for col in ([1.0, 3.0, 5.0], [2.0, 4.0, 7.0])
    ]
    expected = mean + [math.sqrt(v) for v in var]
    assert [round(v, 5) for v in pooled[0, 0].tolist()] == [round(v, 5) for v in expected]


def test_seg_pooling_uses_ceil_mode_and_excludes_padding():
    """``avg_pool1d(..., ceil_mode=True)`` averages only the *valid* positions."""
    layer = CAMLayer(
        bn_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False
    )
    x = mx.array([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [9.0]]])  # (1, 7, 1)
    pooled = layer.seg_pooling(x, seg_len=4)
    assert pooled.shape == x.shape
    # segment 0 = frames 0..3 -> mean 2.5 ; segment 1 = frames 4..6 only -> mean 6.0
    first = [v[0] for v in pooled[0, 0:4].tolist()]
    second = [v[0] for v in pooled[0, 4:7].tolist()]
    assert first == pytest.approx([2.5] * 4)
    assert second == pytest.approx([20.0 / 3.0] * 3)


def test_module_container_applies_children_in_insertion_order():
    container = NamedModuleDict()
    container.add("a", mxnn.Linear(2, 2))
    container.add("b", mxnn.Linear(2, 2))
    assert container.names() == ["a", "b"]
    assert len(container) == 2
    assert isinstance(container["a"], mxnn.Linear)
