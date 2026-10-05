"""Tests for the 10-language acceptance metrics.

The acceptance decision rests on three measurements that are easy to get subtly
wrong and impossible to notice by eye:

* the token edit distance (a plain Levenshtein, but on *answer spans*, not full
  sequences -- post-EOS hallucination is genuine bf16 behaviour and must not be
  charged to the quantized arm);
* the transcript comparison, which has to tolerate the punctuation and spacing
  differences CJK/Thai references carry;
* the ``language X`` guide-span canary, which is what catches
  language-direction damage.

All synthetic; no audio, no model.
"""

from __future__ import annotations

import pytest

from tools.gptq_acceptance import (
    EOS_TOKENS,
    answer_span,
    edit_distance,
    guide_span_ok,
    normalize,
    token_edit_distance,
)


# --------------------------------------------------------------------------- #
# edit distance
# --------------------------------------------------------------------------- #
def test_edit_distance_identical_sequences_is_zero():
    assert edit_distance([1, 2, 3], [1, 2, 3]) == 0


def test_edit_distance_counts_substitutions():
    assert edit_distance([1, 2, 3], [1, 9, 3]) == 1


def test_edit_distance_counts_insertions_and_deletions():
    assert edit_distance([1, 2, 3], [1, 2, 3, 4]) == 1
    assert edit_distance([1, 2, 3, 4], [1, 2, 3]) == 1


def test_edit_distance_handles_empty_sequences():
    assert edit_distance([], []) == 0
    assert edit_distance([], [1, 2]) == 2
    assert edit_distance([1, 2], []) == 2


def test_edit_distance_is_symmetric():
    a, b = [1, 2, 3, 4, 5], [1, 9, 3]
    assert edit_distance(a, b) == edit_distance(b, a)


def test_edit_distance_matches_a_known_example():
    # kitten -> sitting is the textbook distance-3 example.
    assert edit_distance(list("kitten"), list("sitting")) == 3


def test_token_edit_distance_rate_is_relative_to_the_baseline():
    distance, rate = token_edit_distance([1, 2, 3], [1, 2, 3, 4])
    assert distance == 1
    assert rate == pytest.approx(0.25)


def test_token_edit_distance_does_not_divide_by_zero():
    distance, rate = token_edit_distance([], [])
    assert distance == 0
    assert rate == 0.0


# --------------------------------------------------------------------------- #
# answer span
# --------------------------------------------------------------------------- #
def test_answer_span_cuts_at_the_first_end_of_turn_token():
    tokens = [10, 11, 151645, 12, 13]
    assert answer_span(tokens) == [10, 11, 151645]


def test_answer_span_includes_the_end_of_turn_token():
    """The EOS token itself is part of the answer: its absence is a failure."""
    assert answer_span([1, 151643]) == [1, 151643]


@pytest.mark.parametrize("eos", EOS_TOKENS)
def test_answer_span_recognizes_every_declared_eos(eos):
    assert answer_span([7, eos, 8]) == [7, eos]


def test_answer_span_keeps_everything_when_no_eos_is_present():
    assert answer_span([1, 2, 3]) == [1, 2, 3]


def test_answer_span_ignores_post_eos_divergence():
    """Two sequences differing only after EOS must have identical spans.

    This is the exact situation task-5 documented as genuine bf16 behaviour:
    the answer is complete and the model then rambles differently.
    """
    baseline = [1, 2, 3, 151645, 10, 11, 12]
    quantized = [1, 2, 3, 151645, 99, 98, 97, 96]
    assert answer_span(baseline) == answer_span(quantized)
    assert token_edit_distance(answer_span(quantized), answer_span(baseline))[1] == 0.0


# --------------------------------------------------------------------------- #
# transcript normalization
# --------------------------------------------------------------------------- #
def test_normalize_strips_the_asr_text_marker():
    assert "<asr_text>" not in normalize("<asr_text>你好", "zh")


def test_normalize_strips_end_of_turn_markers():
    result = normalize("你好<|im_end|>", "zh")
    assert "im_end" not in result
    assert "你好" in result


def test_normalize_removes_punctuation_for_cjk():
    assert normalize("甚至出现交易，几乎停滞。", "zh") == normalize("甚至出现交易几乎停滞", "zh")


def test_normalize_ignores_case_for_latin_scripts():
    assert normalize("HELLO WORLD", "en") == normalize("hello world", "en")


def test_normalize_ignores_spacing_for_latin_scripts():
    assert normalize("hello  world", "en") == normalize("hello world", "en")


@pytest.mark.parametrize("language", ["zh", "ja", "ko", "th"])
def test_normalize_strips_whitespace_entirely_for_cjk_and_thai(language):
    assert normalize("a b c", language) == "abc"


def test_normalize_keeps_distinct_words_distinct():
    """Normalization must not make different transcripts look equal."""
    assert normalize("hello world", "en") != normalize("hello there", "en")


def test_normalize_handles_none():
    assert normalize(None, "en") == ""


def test_normalize_is_idempotent():
    once = normalize("Hello, world! <|im_end|>", "en")
    assert normalize(once, "en") == once


# --------------------------------------------------------------------------- #
# the guide-span canary
# --------------------------------------------------------------------------- #
def test_guide_span_detects_the_verbatim_prefix():
    assert guide_span_ok("language Chinese<asr_text>你好", "Chinese")


def test_guide_span_rejects_a_missing_prefix():
    assert not guide_span_ok("你好", "Chinese")


def test_guide_span_rejects_a_wrong_language():
    """The canary is language-specific: the wrong name is damage."""
    assert not guide_span_ok("language English<asr_text>你好", "Chinese")


def test_guide_span_rejects_a_corrupted_prefix():
    assert not guide_span_ok("language Chin ese<asr_text>你好", "Chinese")


def test_guide_span_rejects_none():
    assert not guide_span_ok(None, "Chinese")


@pytest.mark.parametrize(
    "language,prompt",
    [("zh", "Chinese"), ("ja", "Japanese"), ("th", "Thai"), ("vi", "Vietnamese")],
)
def test_guide_span_works_for_every_language_name(language, prompt):
    assert guide_span_ok(f"language {prompt}<asr_text>x", prompt)


# --------------------------------------------------------------------------- #
# the acceptance criterion, end to end on synthetic rows
# --------------------------------------------------------------------------- #
def _row(quant_tokens, baseline_tokens):
    """The acceptance metric on a synthetic pair: span token edit rate."""
    from tools.gptq_acceptance import token_edit_distance as ted

    quant_span = answer_span(quant_tokens)
    base_span = answer_span(baseline_tokens)
    _, rate = ted(quant_span, base_span)
    return rate


def _tokens(count: int, eos: int = 151645) -> list[int]:
    """``count`` distinct non-EOS tokens plus the end-of-turn token.

    Token ids start at 1000 because ``EOS_TOKENS`` includes 0 (a real pad/EOS
    id in these vocabularies), so a sequence built from ``range`` would be cut
    at its first element by :func:`answer_span`.
    """
    return list(range(1000, 1000 + count)) + [eos]


def test_a_two_percent_budget_tolerates_one_token_in_fifty():
    baseline = _tokens(50)
    quantized = _tokens(50)
    quantized[10] = 9999
    # One substitution over a 51-token baseline.
    assert _row(quantized, baseline) == pytest.approx(1 / 51)
    assert _row(quantized, baseline) <= 0.02


def test_a_two_percent_budget_rejects_a_two_token_error_in_fifty():
    baseline = _tokens(50)
    quantized = _tokens(50)
    quantized[10] = 9999
    quantized[20] = 9999
    assert _row(quantized, baseline) > 0.02


def test_a_sequence_starting_with_a_zero_id_is_cut_immediately():
    """Guards the fixture above: 0 is a real EOS id, not a safe filler token."""
    assert answer_span([0, 1, 2]) == [0]


def test_garbage_output_fails_the_budget_immediately():
    """The qwen3-asr-0.6b RTN-embedding failure mode: nonsense from step 0."""
    baseline = [11528, 8453, 151704, 100636, 151645]
    quantized = [1048, 271, 271, 271, 271, 271]
    assert _row(quantized, baseline) > 0.02
