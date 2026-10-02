"""Tiny, offline, network-free tests for the native MLX Whisper port.

Deliberately tiny (4 mel bins, 8 channels, 2 layers) so the file stays far below
the project memory ceiling.  These are structural / algebraic checks -- parameter
naming, mask geometry, KV-cache equivalence, log-mel frame arithmetic -- **not**
real-checkpoint acceptance, and there is no tokenizer and no transcription here.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core", reason="mlx is required for the model ports")

from mlx import nn as mxnn  # noqa: E402

from funasr_mlx.models.whisper import (  # noqa: E402
    CONV1D_LAYOUT,
    IDENTITY_LAYOUT,
    AudioEncoder,
    CheckpointKeyError,
    GreedyDecodeConfig,
    ModelDimensions,
    MultiHeadAttention,
    ResidualAttentionBlock,
    TextDecoder,
    Whisper,
    convert_checkpoint_tensor,
    detect_language,
    flatten_state_dict,
    flatten_state_dict_with_rules,
    greedy_decode,
    load_strict,
    log_mel_spectrogram,
    mel_filterbank,
    pad_or_trim,
    reflect_pad_1d,
    sinusoids,
)

N_MELS = 4
N_FFT = 8
HOP = 2
N_STATE = 8
N_HEAD = 2


def _dims(**overrides) -> ModelDimensions:
    base = dict(
        n_mels=N_MELS,
        n_audio_ctx=20,
        n_audio_state=N_STATE,
        n_audio_head=N_HEAD,
        n_audio_layer=2,
        n_vocab=16,
        n_text_ctx=12,
        n_text_state=N_STATE,
        n_text_head=N_HEAD,
        n_text_layer=2,
    )
    base.update(overrides)
    return ModelDimensions.from_dict(base)


def _whisper(**overrides) -> Whisper:
    return Whisper(_dims(**overrides))


# --------------------------------------------------------------------------- #
# checkpoint plumbing
# --------------------------------------------------------------------------- #
def test_flatten_state_dict_reads_the_mapping_storage_not_dunder_dict():
    """Regression: ``mlx.nn.Module`` subclasses ``dict``; ``vars()`` finds no
    tensors, so a ``vars()``-based traversal silently yields an empty dict."""
    leaf = mxnn.Linear(3, 2)
    assert not any(isinstance(v, mx.array) for v in vars(leaf).values())
    assert set(flatten_state_dict(leaf)) == {"weight", "bias"}

    state = flatten_state_dict(_whisper())
    assert state, "the model must expose parameters"
    for key in (
        "encoder.conv1.weight",
        "encoder.conv2.weight",
        "encoder.blocks.0.attn.query.weight",
        "encoder.blocks.0.mlp.0.weight",
        "encoder.blocks.0.mlp.2.weight",
        "encoder.ln_post.weight",
        "decoder.token_embedding.weight",
        "decoder.positional_embedding",
        "decoder.blocks.0.cross_attn.query.weight",
        "decoder.ln.weight",
    ):
        assert key in state, key


def test_conv1d_weights_use_the_mlx_layout_and_convert_back():
    model = _whisper()
    weight = flatten_state_dict(model)["encoder.conv1.weight"]
    assert weight.shape == (N_STATE, 3, N_MELS)  # MLX (out, k, in)
    torch_weight = mx.zeros((N_STATE, N_MELS, 3))
    converted = convert_checkpoint_tensor("encoder.conv1.weight", torch_weight, CONV1D_LAYOUT)
    assert converted.shape == (N_STATE, 3, N_MELS)
    assert convert_checkpoint_tensor("decoder.token_embedding.weight", mx.zeros((4, 4))).shape == (
        4,
        4,
    )


def test_decoder_causal_mask_is_lower_triangular_neg_inf():
    decoder = TextDecoder(n_vocab=16, n_ctx=12, n_state=N_STATE, n_head=N_HEAD, n_layer=2)
    mask = decoder.mask
    assert mask.shape == (12, 12)
    for i in range(12):
        for j in range(12):
            value = mask[i, j].item()
            assert math.isinf(value) and value < 0 if j > i else value == 0.0
    # The mask is NOT a checkpoint parameter: torch registers it persistent=False.
    # Assert on the real forms. The previous version checked "decoder.mask" and
    # "decoder._mask" against a STANDALONE decoder, whose keys carry no "decoder."
    # prefix, so both assertions were trivially true and the prefixed key the
    # canonical golden load actually sees was never exercised.
    assert "_mask" not in flatten_state_dict(decoder)
    assert "decoder._mask" not in flatten_state_dict(_whisper())


def test_mask_is_excluded_under_every_prefix_form():
    """One leaf-name rule must cover the full model, a submodule and a prefix."""
    keys = set(flatten_state_dict(_whisper()))
    assert keys, "the fixture must expose parameters"
    assert not any(key.rsplit(".", 1)[-1] == "_mask" for key in keys)
    prefixed = set(flatten_state_dict(_whisper(), "model"))
    assert not any(key.rsplit(".", 1)[-1] == "_mask" for key in prefixed)
    assert "model.decoder.blocks.0.attn.query.weight" in prefixed


def test_persistent_buffers_are_not_over_excluded():
    """``positional_embedding`` is persistent and must survive the exclusion."""
    keys = set(flatten_state_dict(_whisper()))
    assert "encoder.positional_embedding" in keys
    # Sanity: the rule really is filtering, not filtering everything.
    assert len(keys) > 10


def test_alignment_heads_is_excluded_once_it_is_an_array():
    """``alignment_heads`` is persistent=False in canonical too (model.py:276)."""
    model = _whisper()
    assert model.alignment_heads is None
    model.alignment_heads = mx.zeros((2, N_HEAD, N_STATE // 2), mx.float32)
    keys = set(flatten_state_dict(model))
    assert "alignment_heads" not in keys


def test_published_key_set_loads_strictly_without_a_fabricated_mask():
    """The inventory IS the published key set: nothing missing, nothing invented.

    A strict load of exactly the flattened names must succeed, which is only
    true if the non-persistent buffers are excluded from *both* sides. Fabricating
    a ``decoder._mask`` entry to satisfy the loader would defeat the point.
    """
    model = _whisper()
    state = dict(flatten_state_dict(model))
    assert not any(key.rsplit(".", 1)[-1] in {"_mask", "alignment_heads"} for key in state)
    clone = _whisper()
    assert load_strict(clone, state).loaded == sorted(state)


def _module_path(dotted: str):
    """Resolve ``funasr_mlx.a.b`` to its file, or None."""
    root = Path(__file__).resolve().parents[1] / "src" / "funasr_mlx"
    candidate = root.joinpath(*dotted.split(".")[1:])
    if candidate.with_suffix(".py").exists():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").exists():
        return candidate / "__init__.py"
    return None


def _module_level_names(path: Path) -> set[str]:
    """Names a module defines at top level, without importing it."""
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def test_every_model_symbol_imported_by_the_test_suite_exists():
    """Guard the whole suite against unresolvable imports.

    A missing symbol aborts collection for the entire file rather than failing
    one test, and that has now happened twice (a ``flatten_state_dict_with_rules``
    imported from this module before it was defined, and the same name missing
    from CAM++ earlier). A single unresolvable import takes down the whole
    guarded run, so it is worth pinning statically.

    Pure ``ast``: no model module is imported and nothing is executed, so this
    costs no memory and needs no device.
    """
    tests_dir = Path(__file__).resolve().parent
    models_dir = Path(__file__).resolve().parents[1] / "src" / "funasr_mlx" / "models"
    cache: dict[Path, set[str]] = {}
    problems: list[str] = []
    checked = 0

    for test_file in sorted(tests_dir.glob("test_*.py")):
        tree = ast.parse(test_file.read_text())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("funasr_mlx.models")
            ):
                continue
            path = _module_path(node.module)
            if path is None:
                problems.append(f"{test_file.name}:{node.lineno} no module {node.module}")
                continue
            if path not in cache:
                cache[path] = _module_level_names(path)
            for alias in node.names:
                if alias.name == "*":
                    continue
                checked += 1
                if alias.name in cache[path]:
                    continue
                # A submodule import from the package root is legitimate.
                if path.name == "__init__.py" and (models_dir / f"{alias.name}.py").exists():
                    continue
                problems.append(
                    f"{test_file.name}:{node.lineno} {node.module}.{alias.name} is not defined"
                )

    assert problems == [], "unresolvable imports: " + "; ".join(problems)
    assert checked > 0, "the audit found no imports, so it is not actually checking"


def test_flatten_state_dict_with_rules_returns_the_same_maps_as_the_report():
    """The tuple form must be the report's own dictionaries, not copies.

    Delegating to :func:`flatten_state_dict` means one walk and one set of
    dictionaries, so the two views cannot disagree about a rule.
    """
    model = _whisper()
    report = flatten_state_dict(model)
    tensors, rules = flatten_state_dict_with_rules(model)
    assert tensors == report.tensors
    assert rules == report.rules
    assert set(tensors) == set(rules)
    assert tensors, "the fixture must expose some parameters"


def test_no_one_dimensional_tensor_carries_a_conv_layout_rule():
    """Invariant: only kernels may carry a conv rule.

    ``_rule_for`` keys off the *owning module*, so a conv's 1-D bias inherited
    the kernel rule and was then transposed with three axes -- a hard ValueError
    on every conv that has one, which is every conv in the published encoder
    (``nn.Conv1d`` defaults to ``bias=True``).
    """
    tensors, rules = flatten_state_dict_with_rules(_whisper())
    biases = [name for name in tensors if name.endswith("bias") and rules[name] != IDENTITY_LAYOUT]
    assert biases == [], f"conv biases must be IDENTITY, got {biases}"


def test_load_strict_round_trips_and_rejects_mismatch():
    model = _whisper()
    state = flatten_state_dict(model)
    clone = _whisper()
    assert load_strict(clone, state).loaded == sorted(state)
    with pytest.raises(CheckpointKeyError):
        load_strict(_whisper(), {k: v for k, v in state.items() if k != "decoder.ln.bias"})


# --------------------------------------------------------------------------- #
# positional encoding
# --------------------------------------------------------------------------- #
def test_sinusoids_is_zero_based_and_negatively_incremented():
    table = sinusoids(4, N_STATE)
    assert table.shape == (4, N_STATE)
    # sin(0) = 0 in the first half, cos(0) = 1 in the second -> concatenated.
    assert [round(v, 6) for v in table[0].tolist()] == [0.0] * 4 + [1.0] * 4
    # inv_timescales = exp(-log(10000) / (channels // 2 - 1) * arange(channels // 2))
    increment = math.log(10000.0) / (N_STATE // 2 - 1)
    expected = [math.sin(1.0 * math.exp(-increment * i)) for i in range(N_STATE // 2)] + [
        math.cos(1.0 * math.exp(-increment * i)) for i in range(N_STATE // 2)
    ]
    assert [round(v, 5) for v in table[1].tolist()] == [round(v, 5) for v in expected]


# --------------------------------------------------------------------------- #
# front-end
# --------------------------------------------------------------------------- #
def test_mel_filterbank_shape_and_slaney_normalisation():
    filters = mel_filterbank(n_mels=4, n_fft=N_FFT, sample_rate=16000)
    assert filters.shape == (4, N_FFT // 2 + 1)
    assert bool((filters >= 0).all().item())
    # With a deliberately tiny n_fft the lowest filters are legitimately all-zero
    # (their triangle falls between FFT bins), so only require that *some* rows
    # carry energy and that no row is negative or non-finite.
    energies = [sum(row) for row in filters.tolist()]
    assert max(energies) > 0.0
    assert all(e >= 0.0 for e in energies)


def test_log_mel_spectrogram_frame_count_and_layout():
    audio = mx.arange(160, dtype=mx.float32).reshape(1, 160) / 1000.0
    mel = log_mel_spectrogram(audio, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
    # centre=True pads by n_fft // 2 -> 1 + (160 + 8 - 8) // 2 = 81 frames, minus the
    # final frame torch drops with [..., :-1] -> 80.
    assert mel.shape == (1, 160 // HOP, N_MELS)
    assert bool(mx.isfinite(mel).all().item())


def test_log_mel_spectrogram_enforces_the_frame_guard():
    audio = mx.zeros((1, 4000), mx.float32)
    with pytest.raises(ValueError, match="max_frames"):
        log_mel_spectrogram(audio, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, max_frames=10)


def test_reflect_pad_excludes_the_edge_sample():
    """torch F.pad(x, (2, 2), mode='reflect') on [1, 2, 3, 4].

    The edge value is not repeated, and both fillers are *descending* index runs:
    left = x[2], x[1] and right = x[2], x[1] for n=4, pad=2.
    """
    x = mx.array([[1.0, 2.0, 3.0, 4.0]])
    padded = reflect_pad_1d(x, 2)
    assert padded.shape == (1, 8)
    assert padded[0].tolist() == [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
    # Single-sided pads must match too.
    assert reflect_pad_1d(x, 1).shape == (1, 6)
    assert reflect_pad_1d(x, 1)[0].tolist() == [2.0, 1.0, 2.0, 3.0, 4.0, 3.0]
    assert reflect_pad_1d(x, 0).tolist() == x.tolist()


def test_pad_or_trim_pads_and_truncates():
    x = mx.arange(5, dtype=mx.float32)
    assert pad_or_trim(x, 8).shape == (8,)
    assert pad_or_trim(x, 8)[5:].tolist() == [0.0, 0.0, 0.0]
    assert pad_or_trim(x, 3).tolist() == [0.0, 1.0, 2.0]


# --------------------------------------------------------------------------- #
# attention / KV cache
# --------------------------------------------------------------------------- #
def test_kv_cache_matches_a_single_full_forward():
    """The cached incremental decode must equal one full forward pass."""
    model = _whisper().eval()
    mel = mx.arange(1 * 40 * N_MELS, dtype=mx.float32).reshape(1, 40, N_MELS) / 97.0
    audio_features = model.embed_audio(mel)
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    full_tokens = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)

    reference = model.logits(full_tokens, audio_features)

    cache: dict[str, mx.array] = {}
    model.logits(prompt, audio_features, cache)
    assert cache, "the cache must be populated after the prompt pass"
    assert cache["blocks.0.attn.key"].shape[1] == 4  # self-attention covers the prompt
    assert cache["blocks.0.cross_attn.key"].shape[1] == audio_features.shape[1]

    step = model.logits(mx.array([[5]], dtype=mx.int32), audio_features, cache)
    assert cache["blocks.0.attn.key"].shape[1] == 5  # appended, not replaced
    # Token 5 sits at position 4 of full_tokens = [1, 2, 3, 4, 5, 6], so the
    # matching reference row is index 4.  Index -1 is position 5, which is the
    # prediction for token 6 given one more token -- a different prediction, not
    # a cached-vs-uncached difference.
    assert mx.allclose(step[0, -1, :], reference[0, 4, :], atol=1e-4).item()


def test_kv_cache_matches_a_full_forward_at_every_position():
    """Decoding one token at a time must reproduce the full pass row for row.

    Stronger than comparing a single row: it exercises the positional offset and
    the self-attention cache append at every step, and it is where a positional
    or mask-extent error would show up.
    """
    model = _whisper().eval()
    mel = mx.arange(1 * 40 * N_MELS, dtype=mx.float32).reshape(1, 40, N_MELS) / 97.0
    audio_features = model.embed_audio(mel)
    full_tokens = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)

    reference = model.logits(full_tokens, audio_features)

    cache: dict[str, mx.array] = {}
    for position in range(full_tokens.shape[1]):
        step = model.logits(
            mx.array([[full_tokens[0, position]]], dtype=mx.int32), audio_features, cache
        )
        mx.eval(step)
        assert mx.allclose(step[0, -1, :], reference[0, position, :], atol=1e-4).item(), (
            f"cached step {position} differs from the full forward at that position"
        )
        assert cache["blocks.0.attn.key"].shape[1] == position + 1


def test_greedy_decode_bound_ignores_the_cross_attention_cache():
    """Only self-attention grows, so only it may trip the cache bound.

    The bound is ``n_text_ctx + max_tokens``, but cross-attention is keyed once
    at ``n_audio_ctx``.  Here ``n_audio_ctx`` (20) exceeds the bound (16), and on
    every published Whisper it is 1500 against 448, so taking the max over *all*
    cache entries made decoding raise unconditionally.
    """
    model = _whisper().eval()
    assert model.dims.n_audio_ctx > model.dims.n_text_ctx + 4
    audio_features = model.embed_audio(mx.zeros((1, 40, N_MELS), mx.float32))
    result = greedy_decode(
        model,
        audio_features,
        mx.array([[1, 2]], dtype=mx.int32),
        GreedyDecodeConfig(max_tokens=4, eot_token_id=15),
    )
    assert result.tokens and all(len(row) <= 4 for row in result.tokens)


def test_suppress_masks_several_token_ids():
    """Regression: comparing (1, V) against (1, N) only broadcasts when N is 1
    or N == V, so two or more suppressed ids used to raise."""
    model = _whisper().eval()
    mel = mx.zeros((1, 40, N_MELS), mx.float32)
    audio_features = model.embed_audio(mel)
    prompt = mx.array([[1, 2]], dtype=mx.int32)

    plain = greedy_decode(model, audio_features, prompt, GreedyDecodeConfig(max_tokens=4))
    banned = greedy_decode(
        model,
        audio_features,
        prompt,
        GreedyDecodeConfig(max_tokens=4, suppress_tokens=(3, 7, 11, 13), eot_token_id=15),
    )
    assert all(tok not in (3, 7, 11, 13) for row in banned.tokens for tok in row)
    assert plain.tokens is not None and banned.tokens is not None


def test_greedy_decode_stops_at_eot_and_returns_ids_only():
    model = _whisper().eval()
    mel = mx.zeros((2, 40, N_MELS), mx.float32)
    audio_features = model.embed_audio(mel)
    prompt = mx.array([[1, 2], [3, 4]], dtype=mx.int32)
    result = greedy_decode(
        model,
        audio_features,
        prompt,
        GreedyDecodeConfig(max_tokens=5, eot_token_id=15, space_token_id=0),
    )
    assert len(result.tokens) == 2
    assert len(result.tokens_including_prompt[0]) == 2 + len(result.tokens[0])
    assert all(isinstance(tok, int) for row in result.tokens for tok in row)
    # suppress_blank forbids eot at the very first generated position.
    assert 15 not in result.tokens[0] or not result.tokens[0]
    assert all(tok != 0 for row in result.tokens for tok in row)


def test_detect_language_returns_sorted_logprobs():
    model = _whisper().eval()
    mel = mx.zeros((1, 40, N_MELS), mx.float32)
    audio_features = model.embed_audio(mel)
    scored = detect_language(model, audio_features, [5, 6, 7], [1, 9])
    assert len(scored) == 3
    assert {tok for tok, _ in scored} == {5, 6, 7}
    assert all(math.isfinite(lp) for _, lp in scored)
    logprobs = [lp for _, lp in scored]
    assert logprobs == sorted(logprobs, reverse=True)


def _stride1d(conv):
    """Normalise a Conv1d stride to a tuple (MLX stores the value verbatim)."""
    value = conv.stride
    return tuple(value) if isinstance(value, (tuple, list)) else (int(value),)


def test_encoder_halves_the_time_axis_exactly_once():
    """Canonical OpenAI Whisper: conv1 has no stride, only conv2 stride=2.

    The SenseVoice vendored fork strides conv1 as well (2+2 = 4x reduction); that
    is NOT what funasr/models/whisper/model.py runs.
    """
    encoder = AudioEncoder(n_mels=N_MELS, n_ctx=20, n_state=N_STATE, n_head=N_HEAD, n_layer=2)
    out = encoder(mx.zeros((1, 2 * 20, N_MELS), mx.float32))
    assert out.shape == (1, 20, N_STATE)
    assert _stride1d(encoder.conv1) == (1,)
    assert _stride1d(encoder.conv2) == (2,)


@pytest.mark.parametrize("frames", [4, 38, 41, 42, 60])
def test_encoder_rejects_a_mis_sized_mel_input(frames):
    """The canonical encoder asserts an exact length; it must not slice to fit.

    Slicing the positional table to whatever length the conv stack produced is
    what previously hid the conv1 stride bug from this whole suite.

    Canonical ``conv2`` is ``Conv1d(..., kernel_size=3, stride=2, padding=1)``,
    so the output length is ``floor((L + 2 - 3) / 2) + 1 == ceil(L / 2)``.  The
    check is on the **output** matching ``n_ctx``, not on the input being exactly
    ``2 * n_ctx``: an odd input is accepted whenever it halves to ``n_ctx``.
    """
    encoder = AudioEncoder(n_mels=N_MELS, n_ctx=20, n_state=N_STATE, n_head=N_HEAD, n_layer=2)
    with pytest.raises(ValueError, match="incorrect audio shape"):
        encoder(mx.zeros((1, frames, N_MELS), mx.float32))


def test_encoder_accepts_an_odd_length_that_halves_to_n_ctx():
    """39 mel frames reduce to exactly 20, which canonical accepts.

    The contract is "the encoder output must equal ``n_ctx``", so rejecting this
    would be stricter than the reference; 38 (->19) and 42 (->21) are the
    genuinely wrong neighbours.
    """
    encoder = AudioEncoder(n_mels=N_MELS, n_ctx=20, n_state=N_STATE, n_head=N_HEAD, n_layer=2)
    assert encoder(mx.zeros((1, 39, N_MELS), mx.float32)).shape == (1, 20, N_STATE)


def test_residual_block_applies_cross_attention_before_the_mlp():
    """Canonical order is attn -> cross_attn -> mlp, with no attn_pct gates.

    The vendored SenseVoice fork gates the branches by attn_pct, where
    attn_pct == 0.0 for cross-attention blocks -- that discards self-attention
    in the decoder entirely.
    """
    block = ResidualAttentionBlock(
        N_STATE, N_HEAD, cross_attention=True, n_text_ctx=12, prefix="blocks.0"
    )
    assert not hasattr(block, "_attn_pct")
    assert not hasattr(block, "_resid_pct")

    x = mx.arange(1 * 3 * N_STATE, dtype=mx.float32).reshape(1, 3, N_STATE)
    audio = mx.arange(1 * 5 * N_STATE, dtype=mx.float32).reshape(1, 5, N_STATE)
    out = block(x, xa=audio)
    assert out.shape == (1, 3, N_STATE)

    # The MLP is last: its residual must dominate once it is the only branch fed.
    only_mlp = x + block.mlp(block.mlp_ln(x))
    assert not mx.allclose(out, only_mlp, atol=1e-6).item()


def test_attention_cache_prefix_names_are_namespaced():
    attn = MultiHeadAttention(N_STATE, N_HEAD, prefix="blocks.3.cross_attn")
    assert attn.cache_prefix == "blocks.3.cross_attn"
    out = attn(mx.zeros((1, 5, N_STATE)), xa=mx.zeros((1, 9, N_STATE)))
    assert out.shape == (1, 5, N_STATE)
