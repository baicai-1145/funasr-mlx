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
    cif_v1,
    paraformer_config_from_dict,
)
from funasr_mlx.models.sanm import (  # noqa: E402
    CONV1D_LAYOUT,
    CONV2D_LAYOUT,
    CheckpointKeyError,
    ModuleStack,
    check_short_utt,
    constant_pad_1d,
    convert_checkpoint_tensor,
    flatten_state_dict,
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
    assert [row.tolist() for row in keep.tolist()] == [
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


def test_cif_v1_matches_reference_prefix_sum_formula():
    """Hand-computed parity case for ``cif_v1`` (float32-exact values).

    alphas   = [0.5, 0.25, 0.5, 0.25, 0.5, 0.125]
    hidden[t] = [t + 1, t + 1]
    fires at t = 2 and t = 4, emitted frames follow
    ``P[t] - P[t-1] + rem[t-1] * h[t-1] - rem[t] * h[t]``.
    """
    alphas = mx.array([[0.5, 0.25, 0.5, 0.25, 0.5, 0.125]])
    hidden = mx.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
    frames, fires = cif_v1(hidden, alphas, 1.0)
    assert frames.shape == (1, 2, 2)
    assert [round(v, 6) for v in frames[0, 0].tolist()] == [2.25, 2.25]
    assert [round(v, 6) for v in frames[0, 1].tolist()] == [4.5, 4.5]
    assert [round(v, 6) for v in fires[0].tolist()] == [0.5, 0.75, 1.25, 0.5, 1.0, 0.125]


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


def test_cif_tail_is_zero_for_unpadded_utterance():
    """Parity quirk: the reference's shift trick yields a zero tail here."""
    predictor = CifPredictorV2(idim=4, tail_threshold=0.45)
    hidden = mx.zeros((1, 6, 4))
    full_mask = mx.ones((1, 1, 6))
    alphas, mask_2d = predictor._alphas(hidden, full_mask)
    _, alphas_tailed, token_num = predictor.tail_process_fn(hidden, alphas, mask_2d)
    assert alphas_tailed.shape == (1, 7)
    # alphas were exactly zero (sigmoid(0)=0.5 -> relu(0.5) is not zero, so only
    # the *added* tail column is compared here).
    assert alphas_tailed[0, 6].item() == 0.0
    assert token_num.shape == (1,)


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
    log_probs = mx.zeros((2, 4, 3))
    ids = mx.argmax(log_probs, axis=-1)  # all zeros -> token 0 == blank
    assert ctc_greedy_decode(log_probs, mx.array([4, 2]), blank_id=0) == [[], []]
    assert ids.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]


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


def test_load_strict_converts_torch_conv_layout_on_load():
    model = _paraformer()
    state = dict(flatten_state_dict(model))
    key = "encoder.encoders.0.self_attn.fsmn_block.weight"
    state[key] = mx.transpose(state[key], (0, 2, 1))  # emulate the torch layout
    assert state[key].shape == (8, 1, 3)
    clone = _paraformer()
    load_strict(clone, state, convert=True)
    assert clone.encoder.encoders[0].self_attn.fsmn_block.weight.shape == (8, 3, 1)


def test_load_strict_does_not_convert_by_default():
    """MLX-layout tensors must pass through untouched (no double transpose)."""
    model = _paraformer()
    state = dict(flatten_state_dict(model))
    key = "encoder.encoders.0.self_attn.fsmn_block.weight"
    clone = _paraformer()
    load_strict(clone, state)
    loaded = flatten_state_dict(clone)
    assert loaded[key].shape == state[key].shape == (8, 3, 1)
