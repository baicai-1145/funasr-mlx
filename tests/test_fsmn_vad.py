"""Tiny offline tests for the native MLX ``fsmn-vad`` port.

All dimensions are toy-sized (sample rate 160 Hz, 25 ms / 10 ms frames), nothing
touches the network and no real checkpoint is loaded.  These are architecture /
cache / decision-logic tests, not real-checkpoint acceptance.

Run with: ``python -m pytest tests/test_fsmn_vad.py``
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from funasr_mlx.models.fsmn_vad import (
    AudioChangeState,
    FSMNBlock,
    FrameState,
    FsmnEncoderConfig,
    FsmnVadModel,
    LinearTransform,
    SpeechSegment,
    VadDetectMode,
    VadOptions,
    VadStateMachine,
    WindowDetector,
    convert_checkpoint_tensors,
    flat_parameter_names,
    load_state_dict_strict,
)

# A 100x-downscaled sample rate keeps the tensors tiny while preserving the
# 25 ms / 10 ms frame geometry (25 Hz / 10 Hz -> 4-sample frames, 1-sample shift).
TINY_SAMPLE_RATE = 160
TINY_OPTS = dict(
    sample_rate=TINY_SAMPLE_RATE,
    window_size_ms=200,
    sil_to_speech_time_thres=150,
    speech_to_sil_time_thres=150,
    lookback_time_start_point=200,
    lookahead_time_end_point=100,
    max_end_silence_time=800,
    max_start_silence_time=3000,
    max_single_segment_time=60000,
    frame_in_ms=10,
    frame_length_ms=25,
    speech_noise_thres=0.6,
)


def tiny_encoder_config(**overrides) -> FsmnEncoderConfig:
    payload = dict(
        input_dim=6,
        input_affine_dim=4,
        fsmn_layers=2,
        linear_dim=6,
        proj_dim=4,
        lorder=5,
        rorder=0,
        lstride=1,
        rstride=0,
        output_affine_dim=4,
        output_dim=3,
        use_softmax=True,
    )
    payload.update(overrides)
    return FsmnEncoderConfig(**payload)


def tiny_options(**overrides) -> VadOptions:
    payload = dict(TINY_OPTS)
    payload.update(overrides)
    return VadOptions(**payload)


def make_model(encoder_overrides: dict | None = None, **kwargs) -> FsmnVadModel:
    config = tiny_encoder_config(**(encoder_overrides or {}))
    return FsmnVadModel(config, tiny_options(), **kwargs)


def force_output_bias(model: FsmnVadModel, bias: list[float]) -> None:
    """Zero ``out_linear2``'s weight and pin its bias to make frames deterministic.

    ``score[0] ~= 1`` drives every frame to silence; a strongly negative bias for
    class 0 drives every frame to speech.
    """
    model.update(
        {
            "encoder": {
                "out_linear2": {
                    "linear": {
                        "weight": mx.zeros(
                            (model.config.output_dim, model.config.output_affine_dim)
                        ),
                        "bias": mx.array(bias, dtype=mx.float32),
                    }
                }
            }
        }
    )


def fake_waveform(num_frames: int, options: VadOptions | None = None, seed: int = 0):
    """Deterministic waveform exactly aligned with ``num_frames`` VAD frames."""
    options = options or tiny_options()
    total = (num_frames - 1) * options.frame_shift_length + options.frame_sample_length
    base = mx.arange(total, dtype=mx.float32)
    signal = mx.sin(base * 0.01 + seed) + 0.05 * mx.cos(base * 0.37 + seed)
    return mx.reshape(signal, (1, total))


def assert_allclose(actual, expected, *, atol: float = 1e-5, rtol: float = 0.0) -> None:
    """``pytest.approx`` rejects nested Python lists; go through numpy instead."""
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=rtol,
        atol=atol,
    )


def fake_feats(num_frames: int, dim: int, seed: int = 0) -> mx.array:
    values = mx.arange(num_frames * dim, dtype=mx.float32) * 0.01 + seed
    return mx.reshape(values, (1, num_frames, dim))


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_encoder_config_validation():
    with pytest.raises(ValueError, match="fsmn_layers"):
        tiny_encoder_config(fsmn_layers=1)
    with pytest.raises(ValueError, match="rstride"):
        tiny_encoder_config(rorder=4, rstride=0)
    assert tiny_encoder_config(lorder=5, lstride=2).cache_frames == 8


def test_vad_options_validation():
    with pytest.raises(ValueError, match="detect_mode"):
        VadOptions(**{**TINY_OPTS, "detect_mode": 7})
    with pytest.raises(ValueError, match="sil_pdf_ids"):
        VadOptions(**{**TINY_OPTS, "silence_pdf_num": 2, "sil_pdf_ids": [0]})
    with pytest.raises(ValueError, match="max_end_silence_time"):
        VadOptions(
            **{**TINY_OPTS, "max_end_silence_time": 100, "speech_to_sil_time_thres": 150}
        )
    with pytest.raises(ValueError, match="frame_length_ms"):
        VadOptions(**{**TINY_OPTS, "frame_in_ms": 30, "frame_length_ms": 25})


def test_from_dict_builds_the_template_config():
    model = FsmnVadModel.from_dict(
        {
            "model": "FsmnVADStreaming",
            "model_conf": dict(TINY_OPTS),
            "encoder": "FSMN",
            "encoder_conf": {
                "input_dim": 6,
                "input_affine_dim": 4,
                "fsmn_layers": 2,
                "linear_dim": 6,
                "proj_dim": 4,
                "lorder": 5,
                "rorder": 0,
                "lstride": 1,
                "rstride": 0,
                "output_affine_dim": 4,
                "output_dim": 3,
            },
        },
        max_chunk_frames=64,
    )
    assert model.config.input_dim == 6
    assert model.vad_opts.detect_mode == VadDetectMode.MULTIPLE_UTTERANCE.value
    assert model.max_chunk_frames == 64


def test_derived_frame_geometry():
    options = tiny_options()
    assert options.frame_sample_length == 4  # 25 ms * 160 Hz
    assert options.frame_shift_length == 1  # 10 ms * 160 Hz
    assert options.max_end_sil_frame_cnt_thresh == 650


# --------------------------------------------------------------------------- #
# encoder architecture
# --------------------------------------------------------------------------- #
def test_flat_parameter_names_sees_every_tensor():
    """Regression guard: MLX ``nn.Module`` stores children in its *dict* part.

    A loader that walked ``vars(module)`` instead of ``module.parameters()`` would
    silently see zero tensors.  Pin the exact tensor count so that can never
    regress.
    """
    params = flat_parameter_names(make_model())
    # in_linear1 2 + in_linear2 2 + 2 blocks x 4 (linear 1, conv_left 1, affine 2)
    # + out_linear1 2 + out_linear2 2
    assert len(params) == 16
    assert all(isinstance(value, mx.array) for value in params.values())
    assert "encoder.in_linear1.linear.weight" in params


def test_linear_transform_is_the_nn_linear_itself():
    """``BasicBlock.linear`` is a ``LinearTransform`` whose own parameter is ``weight``.

    The reference ``LinearTransform`` holds ``self.linear = nn.Linear(..., bias=False)``
    and ``BasicBlock`` registers it under ``linear``, so the checkpoint path is
    ``fsmn.<i>.linear.weight`` -- not ``fsmn.<i>.linear.linear.weight``.
    """
    params = flat_parameter_names(LinearTransform(6, 4))
    assert set(params) == {"weight"}
    assert params["weight"].shape == (4, 6)


def test_fsmn_block_only_owns_its_depthwise_conv():
    block = FSMNBlock(dim=2, lorder=3, rorder=0, lstride=1, rstride=0)
    assert set(flat_parameter_names(block)) == {"conv_left.weight"}


def test_parameter_names_match_funasr_checkpoint_layout():
    names = set(flat_parameter_names(make_model()))
    expected_subset = {
        "encoder.in_linear1.linear.weight",
        "encoder.in_linear1.linear.bias",
        "encoder.in_linear2.linear.weight",
        "encoder.fsmn.0.linear.weight",
        "encoder.fsmn.0.fsmn_block.conv_left.weight",
        "encoder.fsmn.0.affine.linear.weight",
        "encoder.fsmn.0.affine.linear.bias",
        "encoder.fsmn.1.linear.weight",
        "encoder.fsmn.1.fsmn_block.conv_left.weight",
        "encoder.out_linear1.linear.weight",
        "encoder.out_linear2.linear.bias",
    }
    assert expected_subset <= names
    assert not any(".layers." in name for name in names)
    # LinearTransform has no bias; the depthwise conv has no bias either.
    assert "encoder.fsmn.0.linear.bias" not in names
    assert "encoder.fsmn.0.fsmn_block.conv_left.bias" not in names
    # rorder=0 means FunASR does not create conv_right at all.
    assert "encoder.fsmn.0.fsmn_block.conv_right.weight" not in names


def test_parameter_shapes_match_pytorch_convention():
    params = flat_parameter_names(make_model())
    assert params["encoder.in_linear1.linear.weight"].shape == (4, 6)
    assert params["encoder.fsmn.0.linear.weight"].shape == (4, 6)
    assert params["encoder.fsmn.0.fsmn_block.conv_left.weight"].shape == (4, 5, 1)
    assert params["encoder.out_linear2.linear.weight"].shape == (3, 4)
    assert params["encoder.out_linear2.linear.bias"].shape == (3,)


def test_encoder_preserves_time_and_normalises_probabilities():
    model = make_model()
    feats = fake_feats(9, model.config.input_dim)
    scores = model.encoder(feats, cache={})
    mx.eval(scores)
    assert scores.shape == (1, 9, model.config.output_dim)
    sums = [sum(row) for row in scores[0].tolist()]
    assert all(value == pytest.approx(1.0, abs=1e-5) for value in sums)


def test_encoder_without_softmax_returns_logits():
    model = make_model({"use_softmax": False})
    scores = model.encoder(fake_feats(5, model.config.input_dim), cache={})
    mx.eval(scores)
    assert scores.shape == (1, 5, 3)


# --------------------------------------------------------------------------- #
# FSMN cache / chunk behaviour
# --------------------------------------------------------------------------- #
def test_block_left_cache_matches_zero_padding():
    """cache=None zero-pads (lorder-1)*lstride frames; the cache is equivalent."""
    block = FSMNBlock(dim=2, lorder=4, rorder=0, lstride=1, rstride=0)
    x = mx.arange(3 * 2, dtype=mx.float32).reshape(1, 3, 2)
    offline, cache = block(x, None)
    assert cache is None

    stream_cache = None
    streamed = None
    for step in range(3):
        frame = x[:, step : step + 1, :]
        streamed, stream_cache = block(frame, stream_cache)
    assert stream_cache.shape == (1, 2, 3)  # (lorder - 1) * lstride
    mx.eval(offline, streamed)
    assert_allclose(streamed, offline)


def test_cache_length_is_stable():
    """Streaming keeps exactly ``(lorder - 1) * lstride`` frames of left context.

    ``FSMNBlock`` returns ``cache=None`` when it is called with ``cache=None`` --
    that is the reference behaviour (``fsmn_vad_streaming/encoder.py`` only assigns
    ``cache`` inside the ``cache is not None`` branch).  A streaming caller must
    therefore start from an explicit zero buffer, which is what
    ``BasicBlock.forward`` does when it seeds ``cache_layer_<i>``.
    """
    block = FSMNBlock(dim=3, lorder=5, rorder=0, lstride=2, rstride=0)
    x = fake_feats(6, 3)
    cache = mx.zeros((1, 3, block.keep))
    for step in range(6):
        _, cache = block(x[:, step : step + 1, :], cache)
    assert cache.shape == (1, 3, (5 - 1) * 2)


def test_streaming_chunks_equal_single_pass():
    """With rorder=0 the streaming FSMN is comparable to the offline pass."""
    source = make_model()
    feats = fake_feats(12, source.config.input_dim, seed=1)

    offline = source.encoder(feats, cache=None)
    stream_cache: dict = {}
    chunks = []
    for start in range(0, 12, 5):
        chunk = source.encoder(feats[:, start : start + 5, :], cache=stream_cache)
        mx.eval(chunk)
        chunks.append(chunk)
    streamed = mx.concatenate(chunks, axis=1)
    mx.eval(offline, streamed)
    assert_allclose(streamed, offline)


def test_rorder_creates_a_right_kernel_and_keeps_the_time_axis():
    block = FSMNBlock(dim=2, lorder=4, rorder=3, lstride=1, rstride=1)
    assert block.conv_right is not None
    assert block.conv_right.weight.shape == (2, 3, 1)
    out, _ = block(fake_feats(6, 2), None)
    mx.eval(out)
    assert out.shape == (1, 6, 2)


def test_rorder_right_context_uses_future_frames_only():
    """A right kernel of ``[1]`` adds exactly x[t + 1], zero-padded at the end."""
    block = FSMNBlock(dim=1, lorder=1, rorder=1, lstride=1, rstride=1)
    block.conv_left.weight = mx.zeros((1, 1, 1))  # keep only the residual
    block.conv_right.weight = mx.array([[[1.0]]])
    x = mx.arange(4, dtype=mx.float32).reshape(1, 4, 1)
    out, _ = block(x, None)
    mx.eval(out)
    assert_allclose(out.reshape(-1), [1.0, 3.0, 5.0, 3.0])


def test_encoder_rejects_bad_cache_shape():
    block = FSMNBlock(dim=3, lorder=5, rorder=0, lstride=1, rstride=0)
    with pytest.raises(ValueError, match="cache must be"):
        block(fake_feats(2, 3), mx.zeros((1, 5, 4)))


# --------------------------------------------------------------------------- #
# strict checkpoint loading
# --------------------------------------------------------------------------- #
def _fake_pytorch_state(model: FsmnVadModel) -> dict[str, mx.array]:
    state: dict[str, mx.array] = {}
    for name, value in flat_parameter_names(model).items():
        if name.endswith(".fsmn_block.conv_left.weight"):
            d, k, _ = value.shape  # (D, K, 1) -> (D, 1, K, 1)
            value = mx.transpose(value, (0, 2, 1)).reshape(d, 1, k, 1)
        state[name] = value
    return state


def test_checkpoint_conv_layout_conversion():
    state = {
        "encoder.fsmn.0.fsmn_block.conv_left.weight": mx.arange(
            2 * 1 * 3 * 1, dtype=mx.float32
        ).reshape(2, 1, 3, 1)
    }
    converted = convert_checkpoint_tensors(state)
    key = "encoder.fsmn.0.fsmn_block.conv_left.weight"
    assert converted[key].shape == (2, 3, 1)
    np.testing.assert_array_equal(
        np.asarray(converted[key]), np.asarray(state[key].reshape(2, 3, 1))
    )


def test_load_state_dict_strict_roundtrip():
    source = make_model()
    target = make_model()
    load_state_dict_strict(target, _fake_pytorch_state(source))

    feats = fake_feats(6, source.config.input_dim, seed=2)
    a = source.encoder(feats, cache={})
    b = target.encoder(feats, cache={})
    mx.eval(a, b)
    assert_allclose(a, b)


def test_load_state_dict_strict_rejects_mismatch():
    model = make_model()
    state = _fake_pytorch_state(model)

    broken = dict(state)
    broken.pop("encoder.in_linear1.linear.bias")
    with pytest.raises(KeyError, match="missing"):
        load_state_dict_strict(model, broken)

    extra = dict(state)
    extra["encoder.bogus"] = mx.zeros((1,))
    with pytest.raises(KeyError, match="unexpected"):
        load_state_dict_strict(model, extra)

    wrong_shape = dict(state)
    wrong_shape["encoder.out_linear2.linear.weight"] = mx.zeros((9, 4))
    with pytest.raises(ValueError, match="shapes do not match"):
        load_state_dict_strict(model, wrong_shape)


# --------------------------------------------------------------------------- #
# decibel / frame state
# --------------------------------------------------------------------------- #
def test_decibel_matches_closed_form():
    model = make_model()
    cache = model.init_cache()
    options = model.vad_opts
    frames = 5
    waveform = fake_waveform(frames, options)
    model.compute_decibel(waveform, cache=cache, frame_count=frames)

    stats = cache["stats"]
    assert len(stats.decibel) == frames
    signal = waveform.reshape(-1).tolist()
    for index in range(frames):
        start = index * options.frame_shift_length
        window = signal[start : start + options.frame_sample_length]
        expected = 10.0 * math.log10(sum(value * value for value in window) + 1e-6)
        assert stats.decibel[index] == pytest.approx(expected, abs=1e-3)


def test_decibel_rejects_misaligned_waveform():
    model = make_model()
    cache = model.init_cache()
    with pytest.raises(RuntimeError, match="not aligned"):
        model.compute_decibel(mx.zeros((1, 11)), cache=cache, frame_count=5)


def test_frame_state_threshold_uses_speech_noise_thres():
    model = make_model()
    cache = model.init_cache()
    stats = cache["stats"]
    # noise prob 0.3 -> 0.7 >= 0.3 + 0.6 is false -> silence.
    stats.decibel = [-30.0]
    stats.noise_scores = [0.3]
    assert model.get_frame_state(0, cache) == FrameState.SIL

    # noise prob 0.1 -> 0.9 >= 0.1 + 0.6 is true, SNR/decibel gates pass.
    stats.noise_scores = [0.1]
    assert model.get_frame_state(0, cache) == FrameState.SPEECH

    # A decibel below decibel_thres always reports silence.
    stats.decibel = [-200.0]
    assert model.get_frame_state(0, cache) == FrameState.SIL

    # Out-of-range frames report silence.
    stats.decibel = []
    assert model.get_frame_state(0, cache) == FrameState.SIL


def test_noise_average_decibel_ema():
    model = make_model()
    cache = model.init_cache()
    stats = cache["stats"]
    stats.decibel = [-50.0, -40.0]
    stats.noise_scores = [0.9, 0.9]  # silence -> EMA update
    model.get_frame_state(0, cache)
    assert stats.noise_average_decibel == pytest.approx(-50.0)
    model.get_frame_state(1, cache)
    window = model.vad_opts.noise_frame_num_used_for_snr
    assert stats.noise_average_decibel == pytest.approx((-40.0 + -50.0 * (window - 1)) / window)


# --------------------------------------------------------------------------- #
# window detector state machine
# --------------------------------------------------------------------------- #
def test_window_detector_sil_to_speech_and_back():
    detector = WindowDetector(
        window_size_ms=200, sil_to_speech_time=150, speech_to_sil_time=150, frame_size_ms=10
    )
    detector.reset()
    assert detector.get_window_size() == 20
    assert detector.frame_size_ms == 10  # plain int attribute, as in the reference
    transitions = [detector.detect_one_frame(FrameState.SPEECH) for _ in range(20)]
    assert transitions[14] == AudioChangeState.SIL_TO_SPEECH  # 15th frame fills the window
    assert all(t == AudioChangeState.SPEECH_TO_SPEECH for t in transitions[15:])
    back = [detector.detect_one_frame(FrameState.SIL) for _ in range(20)]
    assert back[4] == AudioChangeState.SPEECH_TO_SIL  # window drops back to 15
    assert all(t == AudioChangeState.SIL_TO_SIL for t in back[5:])


def test_window_detector_reset_and_invalid_state():
    detector = WindowDetector(200, 150, 150, 10)
    for _ in range(20):
        detector.detect_one_frame(FrameState.SPEECH)
    assert detector.pre_frame_state == FrameState.SPEECH
    detector.reset()
    assert detector.pre_frame_state == FrameState.SIL
    assert detector.win_sum == 0
    assert detector.detect_one_frame(FrameState.INVALID) == AudioChangeState.INVALID


# --------------------------------------------------------------------------- #
# cache lifecycle
# --------------------------------------------------------------------------- #
def test_init_cache_resets_everything():
    model = make_model()
    cache = model.init_cache()
    stats = cache["stats"]
    assert stats.vad_state_machine == VadStateMachine.START_POINT_NOT_DETECTED
    assert stats.confirmed_start_frame == -1
    assert stats.lastest_confirmed_silence_frame == -1
    assert cache["encoder"] == {}
    assert stats.max_end_sil_frame_cnt_thresh == model.vad_opts.max_end_sil_frame_cnt_thresh
    assert stats.speech_noise_thres == model.vad_opts.speech_noise_thres


def test_init_cache_honours_overrides():
    model = make_model()
    cache = model.init_cache(max_end_silence_time=400, speech_noise_thres=0.25)
    assert model.vad_opts.max_end_silence_time == 400
    assert cache["stats"].max_end_sil_frame_cnt_thresh == 400 - 150
    assert cache["stats"].speech_noise_thres == 0.25


def test_drop_cached_frames_trims_history():
    model = make_model()
    cache = model.init_cache()
    stats = cache["stats"]
    stats.frm_cnt = 10
    stats.decibel = [float(i) for i in range(10)]
    stats.noise_scores = [float(i) for i in range(10)]
    stats.scores = mx.zeros((1, 10, 3))
    stats.data_buf_all_init = True
    stats.data_buf_all_len = 10
    model.drop_cached_frames(4, cache=cache)
    assert stats.decibel == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert stats.noise_scores == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert stats.scores.shape == (1, 6, 3)
    assert stats.data_buf_all_len == 6
    assert stats.last_drop_frames == 4
    # Re-dropping the same frame is a no-op.
    model.drop_cached_frames(4, cache=cache)
    assert len(stats.decibel) == 6


def test_drop_cached_frames_rejects_future_frames():
    model = make_model()
    cache = model.init_cache()
    cache["stats"].frm_cnt = 3
    with pytest.raises(RuntimeError, match="only 3 frames exist"):
        model.drop_cached_frames(9, cache=cache)


def test_reset_detection_requires_closed_segment():
    model = make_model()
    cache = model.init_cache()
    stats = cache["stats"]
    stats.frm_cnt = 20
    stats.decibel = [0.0] * 20
    stats.noise_scores = [0.0] * 20
    stats.output_data_buf.append(SpeechSegment(start_ms=0, end_ms=100))
    with pytest.raises(RuntimeError, match="last segment to be closed"):
        model.reset_detection(cache)
    stats.output_data_buf[0].contain_seg_end_point = True
    model.reset_detection(cache)
    assert stats.vad_state_machine == VadStateMachine.START_POINT_NOT_DETECTED
    assert stats.confirmed_start_frame == -1
    assert stats.last_drop_frames == 10  # int(100 / frame_in_ms)


# --------------------------------------------------------------------------- #
# end-to-end (synthetic features only)
# --------------------------------------------------------------------------- #
def test_forward_requires_initialised_cache_and_bounded_chunks():
    model = make_model(max_chunk_frames=8)
    with pytest.raises(ValueError, match="init_cache"):
        model(fake_feats(4, 6), fake_waveform(4), {})
    cache = model.init_cache()
    with pytest.raises(ValueError, match="max_chunk_frames"):
        model(fake_feats(9, 6), fake_waveform(9), cache, is_final=True)
    with pytest.raises(ValueError, match="batch_size=1"):
        model(mx.zeros((2, 4, 6)), fake_waveform(4), cache, is_final=True)


def test_forward_on_pure_silence_reports_no_segments():
    model = make_model()
    force_output_bias(model, [10.0, 0.0, 0.0])  # score[0] ~= 1 -> always silence
    frames = 12
    cache = model.init_cache()
    segments = model(
        fake_feats(frames, model.config.input_dim),
        fake_waveform(frames),
        cache,
        is_final=True,
        is_streaming_input=False,
    )
    assert segments == []
    assert cache["stats"].vad_state_machine == VadStateMachine.START_POINT_NOT_DETECTED


def test_forward_on_pure_speech_closes_one_segment():
    model = make_model()
    force_output_bias(model, [-10.0, 0.0, 0.0])  # score[0] ~= 0 -> always speech
    frames = 20  # the 200 ms window needs 15 speech frames to leave silence
    cache = model.init_cache()
    segments = model(
        fake_feats(frames, model.config.input_dim),
        fake_waveform(frames),
        cache,
        is_final=True,
        is_streaming_input=False,
    )
    assert segments == [[[0, frames * 10]]]


def test_streaming_output_marks_open_segments():
    model = make_model()
    force_output_bias(model, [-10.0, 0.0, 0.0])
    cache = model.init_cache()
    first = model(
        fake_feats(20, model.config.input_dim),
        fake_waveform(20),
        cache,
        is_final=False,
        is_streaming_input=True,
    )
    # The window opens the segment inside this very block.
    assert first == [[[0, -1]]]


def test_reset_restores_initial_state():
    model = make_model()
    force_output_bias(model, [10.0, 0.0, 0.0])
    cache = model.init_cache()
    frames = 8
    model(
        fake_feats(frames, model.config.input_dim),
        fake_waveform(frames),
        cache,
        is_final=True,
        is_streaming_input=False,
    )
    model.reset(cache)
    fresh = model.init_cache()
    assert cache["stats"].frm_cnt == fresh["stats"].frm_cnt == 0
    assert cache["stats"].vad_state_machine == VadStateMachine.START_POINT_NOT_DETECTED
    assert cache["encoder"] == {}
    assert cache["stats"].decibel == []
