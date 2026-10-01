"""Tiny offline tests for the native MLX ``ct-punc`` port.

All dimensions here are toy-sized; nothing touches the network and no real
checkpoint is loaded.  These are *architecture / post-processing* tests, not
real-checkpoint acceptance (numerical parity against FunASR has to be run
separately by the lead under the memory guard).

Run with: ``python -m pytest tests/test_ct_punc.py``
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from funasr_mlx.models.ct_punc import (
    ASCII_PUNC_MAP,
    CharVocab,
    CtPuncConfig,
    CtPuncModel,
    CtPuncPunctuator,
    PuncResult,
    SanmAttention,
    SanmEncoder,
    SanmEncoderConfig,
    expand_word_level_punc,
    flat_parameter_names,
    is_first_byte_ascii,
    load_state_dict_strict,
    make_valid_mask,
    sinusoidal_position_encoding,
    split_to_mini_sentence,
    split_words,
)

PUNC_LIST = ("_", "，", "。", "？", "、")


def tiny_encoder_config(**overrides) -> SanmEncoderConfig:
    payload = dict(
        input_size=8,
        output_size=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=2,
        kernel_size=3,
        sanm_shfit=0,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )
    payload.update(overrides)
    return SanmEncoderConfig(**payload)


def tiny_config(**overrides) -> CtPuncConfig:
    payload = dict(
        vocab_size=16,
        embed_unit=8,
        att_unit=8,
        punc_list=PUNC_LIST,
        sentence_end_id=3,
        encoder=tiny_encoder_config(),
        max_input_tokens=256,
    )
    payload.update(overrides)
    return CtPuncConfig(**payload)


def tiny_vocab(vocab_size: int = 16) -> CharVocab:
    tokens = ["<unk>"] + [chr(0x4E00 + i) for i in range(vocab_size - 1)]
    return CharVocab(tokens)


def assert_allclose(actual, expected, *, atol: float = 1e-5, rtol: float = 0.0) -> None:
    """``pytest.approx`` rejects nested Python lists; go through numpy instead."""
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=rtol,
        atol=atol,
    )


def cjk_text(count: int) -> str:
    return "".join(chr(0x4E00 + i) for i in range(count))


def test_tiny_vocab_is_the_cjk_fixture():
    """The CJK fixture maps the first ``vocab_size - 1`` CJK chars to ids 1..n."""
    vocab = tiny_vocab()
    assert len(vocab) == 16
    assert vocab.encode(list(cjk_text(3))) == [1, 2, 3]
    assert vocab.encode(["definitely-not-in-vocab"]) == [0]


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_config_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        tiny_encoder_config(output_size=9, attention_heads=2)
    with pytest.raises(ValueError, match="input_layer"):
        tiny_encoder_config(input_layer="conv2d")
    with pytest.raises(ValueError, match="selfattention_layer_type"):
        tiny_encoder_config(selfattention_layer_type="selfattn")
    with pytest.raises(ValueError):
        tiny_config(sentence_end_id=99)
    with pytest.raises(ValueError, match="embed_unit == att_unit"):
        tiny_config(att_unit=16)


def test_config_from_funasr_yaml_shape():
    config = CtPuncConfig.from_dict(
        {
            "model": "CTTransformer",
            "model_conf": {
                "vocab_size": 16,
                "embed_unit": 8,
                "att_unit": 8,
                "punc_list": list(PUNC_LIST),
                "sentence_end_id": 3,
            },
            "encoder": "SANMEncoder",
            "encoder_conf": {
                "input_size": 8,
                "output_size": 8,
                "attention_heads": 2,
                "linear_units": 16,
                "num_blocks": 2,
                "input_layer": "pe",
                "kernel_size": 3,
            },
        }
    )
    assert config.punc_size == 5
    assert config.encoder.conv_left_pad == 1
    assert config.encoder.conv_right_pad == 1
    assert config.encoder.d_k == 4


def test_config_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown model_conf keys"):
        CtPuncConfig.from_dict({"model_conf": {"vocab_size": 16, "nope": 1}})
    with pytest.raises(ValueError, match="unknown encoder_conf keys"):
        SanmEncoderConfig.from_dict({"nope": 1})


def test_config_requires_vocab_size():
    with pytest.raises(ValueError, match="vocab_size is required"):
        CtPuncConfig.from_dict({"model_conf": {"embed_unit": 8, "att_unit": 8}})


def test_sanm_shift_changes_padding_split():
    shifted = tiny_encoder_config(kernel_size=3, sanm_shfit=1)
    assert shifted.conv_left_pad == 2
    assert shifted.conv_right_pad == 0
    with pytest.raises(ValueError, match="padding"):
        tiny_encoder_config(kernel_size=3, sanm_shfit=5)


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def test_split_words_matches_funasr_rules():
    assert split_words("hello world") == ["hello", "world"]
    assert split_words("a中b") == ["a", "中", "b"]
    assert split_words("  ") == []


def test_split_to_mini_sentence():
    assert split_to_mini_sentence([1, 2, 3], 20) == [[1, 2, 3]]
    assert split_to_mini_sentence(list(range(20)), 20) == [list(range(20))]
    chunks = split_to_mini_sentence(list(range(45)), 20)
    assert [len(c) for c in chunks] == [20, 20, 5]
    with pytest.raises(ValueError):
        split_to_mini_sentence([1, 2, 3], 1)


def test_is_first_byte_ascii():
    assert is_first_byte_ascii("hello")
    assert not is_first_byte_ascii("中")
    assert not is_first_byte_ascii("")


def test_ascii_punc_map_covers_reference_marks():
    assert ASCII_PUNC_MAP["，"] == ","
    assert ASCII_PUNC_MAP["。"] == "."
    assert ASCII_PUNC_MAP["？"] == "?"
    assert ASCII_PUNC_MAP["、"] == ","


def test_expand_word_level_punc_inserts_no_punc_ids():
    # The reference inserts at index ``i`` of the token list while walking tokens
    # in reverse, and hard-codes the inserted value as ``1``.
    assert expand_word_level_punc([2, 3, 0], ["中", "国", "人"]) == [2, 3, 0]
    assert expand_word_level_punc([3, 0], ["中国", "人"]) == [1, 3, 0]
    # Non-CJK words are left alone.
    assert expand_word_level_punc([3, 0], ["ab", "中"]) == [3, 0]
    # ``1`` is only "no punctuation" when punc_list[1] is the blank mark; callers
    # with a different label table must pass the right id explicitly.
    assert expand_word_level_punc([3, 0], ["中国", "人"], no_punc_id=0) == [0, 3, 0]


# --------------------------------------------------------------------------- #
# tensor building blocks
# --------------------------------------------------------------------------- #
def test_sinusoidal_position_encoding_matches_funasr_formula():
    depth = 8
    x = mx.zeros((1, 3, depth))
    y = sinusoidal_position_encoding(x)
    mx.eval(y)
    log_increment = math.log(10000.0) / (depth // 2 - 1)
    for t in range(3):
        for d in range(depth):
            offset = d if d < depth // 2 else d - depth // 2
            timescale = math.exp(offset * (-log_increment))
            expected = (
                math.sin((t + 1) * timescale)
                if d < depth // 2
                else math.cos((t + 1) * timescale)
            )
            assert float(y[0, t, d]) == pytest.approx(expected, abs=1e-5)


def test_make_valid_mask_marks_padding():
    mask = make_valid_mask([3, 1], 4)
    mx.eval(mask)
    assert mask.shape == (2, 1, 4)
    assert [float(v) for v in mask[0, 0].tolist()] == [1.0, 1.0, 1.0, 0.0]
    assert [float(v) for v in mask[1, 0].tolist()] == [1.0, 0.0, 0.0, 0.0]


def test_layer_norm_eps_is_funasr_value():
    model = CtPuncModel(tiny_config())
    assert model.encoder.after_norm.eps == pytest.approx(1e-12)
    assert model.encoder.encoders0[0].norm1.eps == pytest.approx(1e-12)


def test_sanm_fsmn_padding_split_is_one_frame_each_side():
    """kernel=3, sanm_shfit=0 -> ConstantPad1d((1, 1)), i.e. ``y_left = [0, x, 0]``.

    With a depthwise kernel whose tap 2 is the only non-zero entry,
    ``out[t] = y_left[t - 2]``:

    * ``left=1, right=1`` -> ``y_left = [0, x0, x1, ...]`` -> ``out = [0, 0, 0, x0, x1]``
    * ``left=2, right=0`` -> ``y_left = [0, 0, x0, ...]`` -> ``out = [0, 0, 0, 0, x0]``

    so the assertion below pins the split, not just the total padding.
    """
    channels, length = 2, 5
    attn = SanmAttention(n_head=1, in_feat=channels, n_feat=channels, kernel_size=3)
    assert (attn.left_pad, attn.right_pad) == (1, 1)

    tap = mx.zeros((channels, 3, 1))
    tap = tap + mx.array([[[0.0], [0.0], [1.0]]])  # only kernel index 2
    attn.fsmn_block.weight = tap

    base = mx.arange(channels * length, dtype=mx.float32).reshape(1, length, channels)
    memory = attn.forward_fsmn(base, None)
    expected = base + mx.concatenate(
        [mx.zeros((1, 3, channels)), base[:, 0 : length - 3, :]], axis=1
    )
    mx.eval(memory, expected)
    assert memory.shape == (1, length, channels)
    assert_allclose(memory, expected)


def test_sanm_fsmn_uses_one_frame_of_left_context():
    """Tap 1 with ``left=1`` must read ``x[t - 1]`` and nothing older."""
    channels, length = 2, 4
    attn = SanmAttention(n_head=1, in_feat=channels, n_feat=channels, kernel_size=3)
    tap = mx.zeros((channels, 3, 1))
    tap = tap + mx.array([[[0.0], [1.0], [0.0]]])  # only kernel index 1
    attn.fsmn_block.weight = tap

    base = mx.arange(channels * length, dtype=mx.float32).reshape(1, length, channels)
    memory = attn.forward_fsmn(base, None)
    expected = base + mx.pad(base[:, :-1, :], [(0, 0), (0, 1), (0, 0)])
    mx.eval(memory, expected)
    assert_allclose(memory, expected)


def test_sanm_fsmn_mask_zeroes_padding():
    attn = SanmAttention(n_head=1, in_feat=2, n_feat=2, kernel_size=3)
    attn.fsmn_block.weight = mx.ones((2, 3, 1))
    x = mx.ones((1, 4, 2))
    mask = make_valid_mask([2], 4)
    out = attn.forward_fsmn(x, mask)
    mx.eval(out)
    values = out.tolist()
    assert all(abs(v - 0.0) < 1e-6 for v in values[0][2])
    assert all(abs(v - 0.0) < 1e-6 for v in values[0][3])


# --------------------------------------------------------------------------- #
# architecture
# --------------------------------------------------------------------------- #
def test_flat_parameter_names_sees_every_tensor():
    """Regression guard: MLX ``nn.Module`` stores children in its *dict* part.

    A loader that walked ``vars(module)`` instead of ``module.parameters()`` would
    silently see zero tensors and every checkpoint load would fail open/closed
    in the wrong way.  Pin the exact tensor count so that can never regress.
    """
    model = CtPuncModel(tiny_config())
    params = flat_parameter_names(model)
    # embed 1 + decoder 2 + after_norm 2 + 2 encoder layers x 13
    #   (norm1 2, norm2 2, linear_out 2, linear_q_k_v 2, fsmn_block 1, w_1 2, w_2 2)
    assert len(params) == 31
    assert all(isinstance(value, mx.array) for value in params.values())
    assert "embed.weight" in params


def test_parameter_names_match_funasr_checkpoint_layout():
    names = set(flat_parameter_names(CtPuncModel(tiny_config())))
    expected_subset = {
        "embed.weight",
        "decoder.weight",
        "decoder.bias",
        "encoder.after_norm.weight",
        "encoder.after_norm.bias",
        "encoder.encoders0.0.self_attn.linear_q_k_v.weight",
        "encoder.encoders0.0.self_attn.linear_out.bias",
        "encoder.encoders0.0.self_attn.fsmn_block.weight",
        "encoder.encoders0.0.feed_forward.w_1.weight",
        "encoder.encoders0.0.feed_forward.w_2.bias",
        "encoder.encoders0.0.norm1.weight",
        "encoder.encoders0.0.norm2.bias",
        "encoder.encoders.0.self_attn.fsmn_block.weight",
        "encoder.encoders.0.feed_forward.w_1.bias",
    }
    assert expected_subset <= names
    # MLX's own ModuleList would have inserted an extra "layers." segment.
    assert not any(".layers." in name for name in names)
    # The depthwise SANM conv has no bias in FunASR.
    assert "encoder.encoders0.0.self_attn.fsmn_block.bias" not in names


def test_parameter_shapes_match_pytorch_convention():
    params = flat_parameter_names(CtPuncModel(tiny_config()))
    assert params["embed.weight"].shape == (16, 8)
    assert params["decoder.weight"].shape == (5, 8)
    assert params["decoder.bias"].shape == (5,)
    assert params["encoder.encoders0.0.self_attn.linear_q_k_v.weight"].shape == (24, 8)
    assert params["encoder.encoders0.0.self_attn.fsmn_block.weight"].shape == (8, 3, 1)
    assert params["encoder.encoders0.0.feed_forward.w_1.weight"].shape == (16, 8)
    assert params["encoder.after_norm.weight"].shape == (8,)


def test_forward_shapes():
    model = CtPuncModel(tiny_config())
    ids = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=mx.int32)
    logits = model.punc_forward(ids, [4, 2])
    assert logits.shape == (2, 4, 5)
    rows = model.argmax_punc_ids(logits)
    assert [len(row) for row in rows] == [4, 4]
    assert all(0 <= v < 5 for row in rows for v in row)


def test_encoder_output_lengths():
    encoder = SanmEncoder(tiny_encoder_config())
    hidden, olens = encoder(mx.zeros((2, 5, 8)), [5, 3])
    assert hidden.shape == (2, 5, 8)
    assert olens == [5, 3]


def test_padded_batch_rows_match_zero_padded_single_runs():
    """Padding must not leak into the valid positions of a batched forward.

    FunASR's depthwise convolution zero-pads *beyond the padded tensor length*, so
    the reference behaviour is "batch row == single run right-padded with
    zeros to the batch maximum", not "== shorter single run".
    """
    model = CtPuncModel(tiny_config())
    ids = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=mx.int32)
    batched = model.punc_forward(ids, [5, 2])
    full = model.punc_forward(mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32), [5])
    short = model.punc_forward(mx.array([[6, 7, 8, 9, 10]], dtype=mx.int32), [2])
    mx.eval(batched, full, short)
    assert_allclose(batched[0], full[0])
    # The short row was padded with zeros by the mask before the convolution.
    assert_allclose(batched[1, :2], short[0, :2])


# --------------------------------------------------------------------------- #
# strict checkpoint loading
# --------------------------------------------------------------------------- #
def _fake_pytorch_state(model: CtPuncModel) -> dict[str, mx.array]:
    """Round-trip the model's own parameters into a PyTorch-shaped state dict."""
    state: dict[str, mx.array] = {}
    for name, value in flat_parameter_names(model).items():
        if name.endswith(".self_attn.fsmn_block.weight"):
            value = mx.transpose(value, (0, 2, 1))  # (C, K, 1) -> (C, 1, K)
        state[name] = value
    return state


def test_load_state_dict_strict_roundtrip_and_conv_transpose():
    source = CtPuncModel(tiny_config())
    target = CtPuncModel(tiny_config())
    load_state_dict_strict(target, _fake_pytorch_state(source))

    source_params = flat_parameter_names(source)
    target_params = flat_parameter_names(target)
    conv_key = "encoder.encoders0.0.self_attn.fsmn_block.weight"
    assert target_params[conv_key].shape == (8, 3, 1)
    assert_allclose(target_params[conv_key], source_params[conv_key], atol=1e-6)
    lin_key = "encoder.encoders0.0.feed_forward.w_1.weight"
    assert_allclose(target_params[lin_key], source_params[lin_key], atol=1e-6)

    ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    a = source.punc_forward(ids, [3])
    b = target.punc_forward(ids, [3])
    mx.eval(a, b)
    assert_allclose(a[0], b[0])


def test_load_state_dict_strict_rejects_missing_and_extra_keys():
    model = CtPuncModel(tiny_config())
    state = _fake_pytorch_state(model)

    broken = dict(state)
    broken.pop("decoder.bias")
    with pytest.raises(KeyError, match="missing"):
        load_state_dict_strict(model, broken)

    extra = dict(state)
    extra["decoder.something_else"] = mx.zeros((5,))
    with pytest.raises(KeyError, match="unexpected"):
        load_state_dict_strict(model, extra)


def test_load_state_dict_strict_rejects_shape_mismatch():
    model = CtPuncModel(tiny_config())
    state = _fake_pytorch_state(model)
    state["decoder.weight"] = mx.zeros((7, 8))
    with pytest.raises(ValueError, match="shapes do not match"):
        load_state_dict_strict(model, state)


# --------------------------------------------------------------------------- #
# punctuator post-processing
# --------------------------------------------------------------------------- #
# Label ids for ``PUNC_LIST``.  ``0`` is the blank mark ("_") and therefore the
# "no punctuation" class; using ``1`` for "none" is a fixture bug that silently
# turns blanks into commas.
PUNC_NONE = 0
PUNC_COMMA = 1
PUNC_PERIOD = 2
PUNC_QUESTION = 3  # == sentence_end_id
PUNC_DUN = 4

ASCII_WORDS = ["hello", "world", "ok"]


def stub_vocab(vocab_size: int = 16) -> CharVocab:
    """Vocabulary that contains every token the stub tests feed in.

    It must be exactly ``vocab_size`` long because
    :class:`CtPuncPunctuator` rejects a mismatched vocabulary.  CJK characters
    occupy ids 1..12 and the ASCII words 13..15.
    """
    cjk_count = vocab_size - 1 - len(ASCII_WORDS)
    tokens = ["<unk>"] + [chr(0x4E00 + i) for i in range(cjk_count)] + list(ASCII_WORDS)
    assert len(tokens) == vocab_size
    return CharVocab(tokens)


class StubPunctuator(CtPuncPunctuator):
    """Punctuator whose model call is replaced by a scripted table.

    The script is keyed by the token-id tuple the punctuator will *actually*
    look up (derived from :meth:`key`), never by hand-written ids: that removes
    the whole class of fixture bugs where a token falls back to ``<unk>`` and the
    ids silently become ``0, 0, 0``.  An unexpected mini-sentence is a hard error,
    which also pins the chunking.
    """

    def __init__(self, **config_overrides):
        payload = dict(split_size=4, cache_pop_trigger_limit=6)
        payload.update(config_overrides)
        config = tiny_config(**payload)
        self._script: dict[tuple[int, ...], list[int]] = {}
        self.seen_lengths: list[int] = []
        self.model = CtPuncModel(config)
        self.vocab = stub_vocab(config.vocab_size)
        self.config = config

    def key(self, tokens) -> tuple[int, ...]:
        return tuple(self.vocab.encode(list(tokens)))

    def script(self, chunks) -> "StubPunctuator":
        """``chunks`` is an iterable of ``(token_sequence, predicted_ids)``."""
        self._script = {self.key(tokens): list(ids) for tokens, ids in chunks}
        return self

    def predict_punc_ids(self, token_ids):  # type: ignore[override]
        ids = [int(i) for i in token_ids]
        self.seen_lengths.append(len(ids))
        if tuple(ids) not in self._script:
            raise AssertionError(f"unexpected mini-sentence ({len(ids)} tokens): {ids}")
        return list(self._script[tuple(ids)])


def test_empty_text_returns_empty_result():
    punctuator = StubPunctuator()
    assert punctuator.punctuate("") == PuncResult(text="", punc_array=None)
    assert punctuator.punctuate("   ") == PuncResult(text="", punc_array=None)


def test_single_chunk_emits_tokens_and_punc_array():
    chars = list(cjk_text(4))
    punctuator = StubPunctuator().script(
        [(chars, [PUNC_COMMA, PUNC_PERIOD, PUNC_QUESTION, PUNC_NONE])]
    )
    result = punctuator.punctuate("".join(chars))
    assert punctuator.seen_lengths == [4]
    # The reference appends the mark *after* each token, and a trailing CJK char
    # without a terminator gets "。" plus a sentence_end_id rewrite of the last id.
    assert result.text == (
        chars[0] + "，" + chars[1] + "。" + chars[2] + "？" + chars[3] + "。"
    )
    assert result.punc_array == [PUNC_COMMA, PUNC_PERIOD, PUNC_QUESTION, PUNC_QUESTION]


def test_ascii_tokens_get_spaces_capitalisation_and_ascii_marks():
    punctuator = StubPunctuator().script(
        [(ASCII_WORDS, [PUNC_COMMA, PUNC_PERIOD, PUNC_NONE])]
    )
    result = punctuator.punctuate(" ".join(ASCII_WORDS))
    # CTTransformer.inference, per token i:
    #   capitalise when (i == 0 or the *previous* mark is a sentence end) and the
    #   first char is 1 byte; prefix " " when i == 0 and 1 byte, or when i > 0 and
    #   both token i-1 and token i start with a 1-byte char; then append the mark,
    #   mapped to ASCII when the token starts with a 1-byte char.
    # -> " Hello" + "," + " world" + "." + " Ok"; "ok" is capitalised because the
    # previous mark is "。"; the trailing "k" is 1 byte and is not "." or "?", so a
    # final "." is appended and the last punc id is rewritten to sentence_end_id.
    assert result.text == " Hello, world. Ok."
    assert result.punc_array == [PUNC_COMMA, PUNC_PERIOD, PUNC_QUESTION]


def test_chunk_without_terminator_is_carried_forward():
    # split_size=4 -> chunks [0:4] and [4:8].  The first has no terminator inside
    # the scan range [len-2 .. 2], so nothing is emitted and it is re-run as cache.
    chars = list(cjk_text(8))
    punctuator = StubPunctuator().script(
        [
            (chars[0:4], [PUNC_NONE] * 4),
            (chars[0:8], [PUNC_NONE] * 7 + [PUNC_QUESTION]),
        ]
    )
    result = punctuator.punctuate("".join(chars))
    assert punctuator.seen_lengths == [4, 8]
    assert result.punc_array == [PUNC_NONE] * 7 + [PUNC_QUESTION]
    # The mark follows its own token, so "？" lands *after* the last character; and
    # because "？" is already a sentence end the trailing-fixup branch adds nothing.
    assert result.text == "".join(chars) + "？"


def test_over_long_chunk_is_cut_at_the_last_comma():
    # 12 tokens / split_size=4 -> 3 chunks.  Chunk 1 (8 tokens with cache) exceeds
    # cache_pop_trigger_limit=6 without a terminator, so the comma inside the scan
    # range is promoted to sentence_end_id and the rest becomes the next cache.
    chars = list(cjk_text(12))
    punctuator = StubPunctuator().script(
        [
            (chars[0:4], [PUNC_NONE] * 4),
            (chars[0:8], [PUNC_NONE, PUNC_NONE, PUNC_COMMA] + [PUNC_NONE] * 5),
            (chars[3:12], [PUNC_PERIOD] + [PUNC_NONE] * 8),
        ]
    )
    result = punctuator.punctuate("".join(chars))
    assert punctuator.seen_lengths == [4, 8, 9]
    # Emitted chunk 1 is chars[0:3] (up to and including the promoted comma) and
    # chunk 2 is the 5 carried tokens plus the 4 new ones.
    assert result.punc_array == [
        PUNC_NONE, PUNC_NONE, PUNC_QUESTION,
        PUNC_PERIOD, PUNC_NONE, PUNC_NONE, PUNC_NONE, PUNC_NONE, PUNC_NONE, PUNC_NONE,
        PUNC_NONE, PUNC_QUESTION,
    ]
    # sentence_end_id is 3 -> punc_list[3] == "？", so the promoted mark renders as
    # a question mark, not a period; only chunk 2's own "。" is a period.
    assert result.text == (
        "".join(chars[0:3]) + "？" + chars[3] + "。" + "".join(chars[4:12]) + "。"
    )


def test_sentence_end_id_must_map_to_a_terminator():
    """``sentence_end_id`` 3 -> "？" is consistent with ``sentence_end_puncs``.

    Promoted commas are rewritten to ``sentence_end_id``, so if that id were not a
    sentence end the promoted chunk would still never close a sentence.
    """
    config = CtPuncConfig()
    assert config.punc_list[config.sentence_end_id] in config.sentence_end_puncs
    # "。" is index 2, "？" is index 3; sentence_end_id must be the latter.
    assert config.punc_list.index("。") == PUNC_PERIOD
    assert config.punc_list.index("？") == config.sentence_end_id


def test_punctuator_rejects_oversized_mini_sentence():
    """The bound must live in the real predictor, not in a test stub."""
    config = tiny_config(max_input_tokens=2)
    punctuator = CtPuncPunctuator(CtPuncModel(config), stub_vocab(), config)
    with pytest.raises(ValueError, match="max_input_tokens"):
        punctuator.predict_punc_ids([1, 2, 3])


def test_punctuator_requires_matching_vocab_size():
    model = CtPuncModel(tiny_config())
    with pytest.raises(ValueError, match="vocabulary size"):
        CtPuncPunctuator(model, CharVocab(["<unk>", "a"]))


def test_reset_is_a_noop_for_offline_ct_punc():
    assert StubPunctuator().reset() is None


# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #
def test_char_vocab_encode_and_unknown_fallback():
    vocab = CharVocab(["<unk>", "a", "中"])
    assert vocab.encode(["a", "中", "zzz"]) == [1, 2, 0]
    assert vocab.unk_id == 0
    assert len(vocab) == 3
    with pytest.raises(ValueError, match="duplicated"):
        CharVocab(["<unk>", "a", "a"])
    with pytest.raises(ValueError, match="absent"):
        CharVocab(["a", "b"])
