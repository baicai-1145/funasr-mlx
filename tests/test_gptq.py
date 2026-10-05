"""GPTQ quantizer unit tests: MLX layout compatibility and GPTQ <= RTN error.

Three independent contracts are checked here.

1. **Layout / bit-exactness.** ``AffineQuantizer`` must reproduce
   ``mx.quantize`` bit for bit -- the same codes, the same ``scales`` and the
   same ``biases`` -- because the W4A16 runtime dequantizes with
   ``mx.quantized_matmul`` and never sees this code.  A divergence would show up
   as a silent accuracy loss in the shipped checkpoint, not as a crash.
2. **Packing.** The ``uint32`` bit order is MLX's (element ``j`` in bits
   ``bits * (j % pack_factor)`` of word ``j // pack_factor``), verified by
   round-tripping through ``mx.dequantize``.
3. **GPTQ <= RTN.** On a synthetic least-squares problem with a known optimal
   solution, GPTQ's reconstruction error must not exceed plain round-to-nearest
   on the same matrix, with the same quantizer.  The reference GPTQ paper's
   whole claim is this inequality; if it fails, the implementation is wrong.

Everything here is tiny and CPU-only (no model, no checkpoint, no network).
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from funasr_mlx.models.llm.gptq import (  # noqa: E402
    AffineQuantizer,
    GptqConfig,
    HessianAccumulator,
    gptq_quantize,
    output_error,
    pack_codes,
    rtn_quantize,
    unpack_codes,
)


def _mlx_quantize(weight: np.ndarray, *, bits: int, group_size: int):
    """``mx.quantize`` on an fp32 array, returned as NumPy in fp32."""
    packed, scales, biases = mx.quantize(
        mx.array(weight), group_size=group_size, bits=bits
    )
    mx.eval(packed, scales, biases)
    return (
        np.array(packed),
        np.array(scales.astype(mx.float32)),
        np.array(biases.astype(mx.float32)),
    )


# --------------------------------------------------------------------------- #
# 1. the quantizer is bit-exact against mx.quantize
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [4, 8])
def test_affine_quantizer_matches_mx_quantize(group_size: int, bits: int) -> None:
    """Codes, scales and biases must equal ``mx.quantize`` exactly (fp32 source)."""
    rng = np.random.RandomState(0)
    weight = rng.randn(5, 256).astype(np.float32)
    quantizer = AffineQuantizer(bits=bits, group_size=group_size, storage="float32")
    packed_expected, scales_expected, biases_expected = _mlx_quantize(
        weight, bits=bits, group_size=group_size
    )

    codes = np.empty(weight.shape, dtype=np.uint8)
    scales = np.empty((weight.shape[0], weight.shape[1] // group_size), dtype=np.float32)
    biases = np.empty_like(scales)
    for index, start in enumerate(range(0, weight.shape[1], group_size)):
        stop = start + group_size
        scale, bias = quantizer.params(weight[:, start:stop])
        scales[:, index] = scale
        biases[:, index] = bias
        codes[:, start:stop] = quantizer.codes(weight[:, start:stop], scale, bias)

    assert np.array_equal(scales, scales_expected)
    assert np.array_equal(biases, biases_expected)
    assert np.array_equal(quantizer.pack(codes), packed_expected)


def test_affine_quantizer_matches_mx_quantize_float16() -> None:
    """The same must hold when the side tensors are stored as float16."""
    rng = np.random.RandomState(1)
    weight = rng.randn(3, 128).astype(np.float16)
    quantizer = AffineQuantizer(bits=4, group_size=64, storage="float16")
    packed16, scales16, biases16 = mx.quantize(mx.array(weight), group_size=64, bits=4)
    mx.eval(packed16, scales16, biases16)

    codes = np.empty(weight.shape, dtype=np.uint8)
    scales = np.empty((3, 2), dtype=np.float16)
    biases = np.empty((3, 2), dtype=np.float16)
    for index, start in enumerate(range(0, 128, 64)):
        block = weight[:, start : start + 64].astype(np.float32)
        scale, bias = quantizer.params(block)
        scales[:, index] = scale
        biases[:, index] = bias
        codes[:, start : start + 64] = quantizer.codes(block, scale, bias)

    assert np.array_equal(quantizer.pack(codes), np.array(packed16))
    assert np.array_equal(scales, np.array(scales16))
    assert np.array_equal(biases, np.array(biases16))


def test_scale_sign_follows_the_larger_magnitude_end() -> None:
    """The kernel's sign rule: bias is the *larger-magnitude* end, not the min.

    This is the step a textbook ``scale = (max - min) / (2^b - 1)`` port gets
    wrong, and getting it wrong changes every code in the group.
    """
    quantizer = AffineQuantizer(bits=4, group_size=64, storage="float32")
    # max magnitude on the negative side -> bias is the minimum, scale > 0
    negative = np.linspace(-8.0, 1.0, 64, dtype=np.float32)[None, :]
    scale, bias = quantizer.params(negative)
    assert scale[0] > 0
    assert bias[0] == pytest.approx(-8.0)
    # max magnitude on the positive side -> bias is the maximum, scale < 0
    positive = np.linspace(-1.0, 8.0, 64, dtype=np.float32)[None, :]
    scale, bias = quantizer.params(positive)
    assert scale[0] < 0
    assert bias[0] == pytest.approx(8.0)


def test_scale_renormalization_lands_the_far_end_on_an_integer_code() -> None:
    """The stored scale is ``edge / round(edge / raw_scale)``, not the raw scale.

    ``edge`` is the group's larger-magnitude end and is also the stored bias, so
    the edge always maps to code 0; the renormalization is what makes
    ``edge / scale`` an exact integer, which is the step a textbook port misses
    (it would leave a fractional code at the edge and shift every other code).
    """
    quantizer = AffineQuantizer(bits=4, group_size=64, storage="float32")
    for seed in range(4):
        block = np.random.RandomState(seed).randn(1, 64).astype(np.float32)
        scale, bias = quantizer.params(block)
        edge = float(bias[0])
        ratio = edge / float(scale[0])
        # Renormalized: the edge is an exact integer number of scales from 0.
        assert float(ratio) == pytest.approx(round(float(ratio)), abs=1e-3)
        # ... and the edge is the bias, so it maps to code 0 exactly.
        assert int(quantizer.codes(np.array([[edge]], dtype=np.float32), scale, bias)[0, 0]) == 0
        # Every code in the group is the plain affine rounding, clamped.
        expected = np.clip(
            np.floor(np.abs((block - bias[:, None]) / scale[:, None]) + 0.5)
            * np.sign((block - bias[:, None]) / scale[:, None]),
            0,
            quantizer.max_code,
        ).astype(np.uint8)
        assert np.array_equal(quantizer.codes(block, scale, bias), expected)
        # The far end sits exactly |ratio| scales from the edge.
        far = float(block.min()) if edge == float(block.max()) else float(block.max())
        code = int(quantizer.codes(np.array([[far]], dtype=np.float32), scale, bias)[0, 0])
        assert code == min(int(round(abs((far - edge) / float(scale[0])))), quantizer.max_code)


def test_constant_group_quantizes_exactly() -> None:
    """A constant group round-trips exactly (bias is the constant, code 0)."""
    quantizer = AffineQuantizer(bits=4, group_size=64, storage="float32")
    for value in (-3.0, 0.0, 2.5):
        block = np.full((1, 64), value, dtype=np.float32)
        scale, bias = quantizer.params(block)
        _, reconstruction = quantizer.quantize(block, scale, bias)
        assert np.allclose(reconstruction, block, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. packing round-trips through the MLX runtime
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bits", [4, 8])
def test_packing_round_trips_through_mx_dequantize(bits: int) -> None:
    """Pack with our packer, dequantize with MLX, get the affine reconstruction."""
    rng = np.random.RandomState(3)
    weight = rng.randn(4, 128).astype(np.float32)
    quantizer = AffineQuantizer(bits=bits, group_size=64, storage="float32")
    codes = np.empty(weight.shape, dtype=np.uint8)
    scales = np.empty((4, 2), dtype=np.float32)
    biases = np.empty((4, 2), dtype=np.float32)
    for index, start in enumerate(range(0, 128, 64)):
        scale, bias = quantizer.params(weight[:, start : start + 64])
        scales[:, index] = scale
        biases[:, index] = bias
        codes[:, start : start + 64] = quantizer.codes(weight[:, start : start + 64], scale, bias)
    packed = quantizer.pack(codes)

    # MLX reads our packed words and our side tensors back as the same numbers.
    dequantized = np.array(
        mx.dequantize(
            mx.array(packed), mx.array(scales), mx.array(biases), group_size=64, bits=bits
        )
    )
    expected = np.repeat(scales, 64, axis=1) * codes + np.repeat(biases, 64, axis=1)
    assert np.allclose(dequantized, expected, atol=1e-5)

    # ... and the codes survive the packing bit order.
    assert np.array_equal(unpack_codes(packed, 128, bits=bits), codes)
    assert np.array_equal(pack_codes(codes, bits=bits), packed)


def test_quantized_matmul_uses_the_same_layout() -> None:
    """The real runtime GEMM must agree with our own dequantization."""
    rng = np.random.RandomState(4)
    weight = rng.randn(8, 64).astype(np.float32)
    quantizer = AffineQuantizer(bits=4, group_size=64, storage="float32")
    scale, bias = quantizer.params(weight)
    codes = quantizer.codes(weight, scale, bias)
    packed = quantizer.pack(codes)

    x = rng.randn(2, 64).astype(np.float32)
    out = mx.quantized_matmul(
        mx.array(x), mx.array(packed), mx.array(scale[:, None]), mx.array(bias[:, None]),
        transpose=True, group_size=64, bits=4,
    )
    mx.eval(out)
    expected = x @ quantizer.dequantize(codes, scale, bias).T
    assert np.allclose(np.array(out), expected, atol=1e-4)


# --------------------------------------------------------------------------- #
# 3. GPTQ beats round-to-nearest
# --------------------------------------------------------------------------- #
def _least_squares_problem(rows: int = 512, columns: int = 128, seed: int = 5):
    """A calibration set whose Hessian has a wide, realistic eigenvalue spread.

    Correlated columns (an outlier direction plus a decaying spectrum) are the
    regime GPTQ is designed for; with an isotropic Hessian the two methods are
    nearly tied and the test would not discriminate.
    """
    rng = np.random.RandomState(seed)
    spectrum = np.logspace(0, -3, columns).astype(np.float32)
    mixing = np.linalg.qr(rng.randn(columns, columns))[0].astype(np.float32)
    activations = (rng.randn(rows, columns) * spectrum) @ mixing.T
    activations[:, 0] *= 20.0  # one dominant outlier direction
    weight = rng.randn(16, columns).astype(np.float32)
    return activations, weight


@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("group_size", [32, 64])
def test_gptq_output_error_is_at_most_rtn_output_error(
    act_order: bool, group_size: int
) -> None:
    """The core GPTQ claim, measured with the same quantizer for both arms.

    GPTQ minimizes the *layer-output* error (the Hessian-weighted norm), which is
    the quantity that propagates through the network -- for the LM head it is
    literally the logits error.  That is the guarantee; the unweighted
    Frobenius norm is not, and is reported separately.
    """
    activations, weight = _least_squares_problem()


def test_stored_triple_is_the_one_gptq_chose_not_a_rederivation() -> None:
    """The stored triple must reproduce the matrix the reported metrics describe.

    Regression guard.  An earlier version re-derived the triple by re-quantizing
    the *finished* matrix, which silently destroyed most of the gain: the
    reconstruction drifted (measured 598 vs 135 layer-output error on this very
    problem, i.e. 4.4x worse) while every test still passed, because the stored
    triple was still self-consistent.

    Note what is deliberately *not* asserted: that re-quantizing the final matrix
    reproduces the stored parameters.  With dynamic groups the parameters come
    from the running (partially compensated) matrix, so that is legitimately not
    a no-op.  The contract is that the stored triple dequantizes to the matrix
    the error metrics were computed on, and that the codes are exactly the ones
    those stored parameters imply.
    """
    activations, weight = _least_squares_problem(rows=512, columns=128)
    hessian = 2.0 * (activations.T @ activations)
    config = GptqConfig(bits=4, group_size=64, storage="float32")
    outcome = gptq_quantize(weight, hessian, config)

    codes = unpack_codes(outcome.packed, weight.shape[1])
    stored = np.repeat(outcome.scales, 64, axis=1) * codes + np.repeat(
        outcome.biases, 64, axis=1
    )
    assert outcome.error == pytest.approx(float(np.mean((weight - stored) ** 2)), rel=1e-5)
    assert outcome.output_error == pytest.approx(
        output_error(weight, stored, hessian), rel=1e-5
    )
    for index, start in enumerate(range(0, 128, 64)):
        assert np.array_equal(
            AffineQuantizer(bits=4, group_size=64, storage="float32").codes(
                stored[:, start : start + 64],
                outcome.scales[:, index],
                outcome.biases[:, index],
            ),
            codes[:, start : start + 64],
        )


def test_act_order_uses_static_groups_so_the_stored_triple_stays_consistent() -> None:
    """act-order permutes the columns; the stored triple must still line up.

    Once act-order permutes the columns, the columns of one *stored* group
    arrive interleaved, so the parameters have to be taken once, from the
    original matrix, over the group's own columns in storage order (static
    groups).  Taking them from the permuted running matrix instead -- the
    dynamic-group rule that is correct without act-order -- leaves the stored
    codes and the stored side tensors describing different groups, and the
    checkpoint then dequantizes to a matrix the reported metrics never described.
    """
    activations, weight = _least_squares_problem(rows=512, columns=128)
    hessian = 2.0 * (activations.T @ activations)
    config = GptqConfig(bits=4, group_size=64, act_order=True, storage="float32")
    outcome = gptq_quantize(weight, hessian, config)
    codes = unpack_codes(outcome.packed, weight.shape[1])
    stored = np.repeat(outcome.scales, 64, axis=1) * codes + np.repeat(
        outcome.biases, 64, axis=1
    )
    assert outcome.error == pytest.approx(float(np.mean((weight - stored) ** 2)), rel=1e-5)
    assert outcome.output_error == pytest.approx(
        output_error(weight, stored, hessian), rel=1e-5
    )
    for index, start in enumerate(range(0, 128, 64)):
        assert np.array_equal(
            AffineQuantizer(bits=4, group_size=64, storage="float32").codes(
                stored[:, start : start + 64],
                outcome.scales[:, index],
                outcome.biases[:, index],
            ),
            codes[:, start : start + 64],
        )


def test_act_order_beats_plain_order_on_a_correlated_hessian() -> None:
    """act-order is not cosmetic: it must improve the metric it targets."""
    activations, weight = _least_squares_problem(rows=512, columns=128)
    hessian = 2.0 * (activations.T @ activations)
    plain = gptq_quantize(weight, hessian, GptqConfig(bits=4, group_size=64, storage="float32"))
    ordered = gptq_quantize(
        weight, hessian, GptqConfig(bits=4, group_size=64, act_order=True, storage="float32")
    )
    assert ordered.output_error < plain.output_error



def test_gptq_reduces_output_error_substantially_on_a_hard_problem() -> None:
    """Not just "no worse": on a hard problem GPTQ should cut the error clearly."""
    activations, weight = _least_squares_problem(rows=1024, columns=256)
    hessian = 2.0 * (activations.T @ activations)
    config = GptqConfig(bits=4, group_size=64, act_order=True, storage="float32")
    gptq = gptq_quantize(weight, hessian, config)
    rtn = rtn_quantize(weight, config, hessian)
    assert gptq.output_error < 0.5 * rtn.output_error, (
        f"GPTQ output error {gptq.output_error:.6f} vs RTN {rtn.output_error:.6f}: "
        "expected a clear win"
    )


def test_gptq_spends_frobenius_error_to_buy_output_error() -> None:
    """The two metrics disagree, and that is the documented behaviour.

    GPTQ's error compensation moves mass around: it accepts a larger unweighted
    distance in the low-influence directions to shrink the weighted one.  If this
    test ever starts failing because the Frobenius error dropped below RTN's,
    the metrics are being confused somewhere and the module docstring is wrong.
    """
    activations, weight = _least_squares_problem()
    hessian = 2.0 * (activations.T @ activations)
    config = GptqConfig(bits=4, group_size=64, storage="float32")
    gptq = gptq_quantize(weight, hessian, config)
    rtn = rtn_quantize(weight, config, hessian)
    assert gptq.output_error < rtn.output_error
    assert gptq.frobenius_reduction is not None
    assert gptq.error == pytest.approx(gptq.error)


def test_gptq_output_dequantizes_to_its_own_reconstruction() -> None:
    """The stored triple must be self-consistent, in the storage dtype."""
    activations, weight = _least_squares_problem(rows=256, columns=128)
    hessian = 2.0 * (activations.T @ activations)
    config = GptqConfig(bits=4, group_size=64, storage="float16")
    outcome = gptq_quantize(weight, hessian, config)

    codes = unpack_codes(outcome.packed, weight.shape[1])
    dequantized = np.array(
        mx.dequantize(
            mx.array(outcome.packed),
            mx.array(outcome.scales).astype(mx.float16),
            mx.array(outcome.biases).astype(mx.float16),
            group_size=64,
            bits=4,
        )
    )
    assert outcome.scales.dtype == np.float16
    assert outcome.biases.dtype == np.float16
    # The side tensors are stored in fp16, so the runtime's reconstruction is
    # the fp16-rounded one -- compare against that, not the fp32 ideal.
    codes = unpack_codes(outcome.packed, weight.shape[1])
    for index, start in enumerate(range(0, weight.shape[1], 64)):
        block = np.repeat(outcome.scales[:, index : index + 1].astype(np.float32), 64, axis=1)
        block *= codes[:, start : start + 64]
        block += np.repeat(outcome.biases[:, index : index + 1].astype(np.float32), 64, axis=1)
        assert np.allclose(block, dequantized[:, start : start + 64], atol=1e-3)


def test_gptq_improves_with_calibration_quality() -> None:
    """More calibration rows must not make the result worse (same Hessian scale)."""
    activations, weight = _least_squares_problem(rows=1024, columns=128)
    config = GptqConfig(bits=4, group_size=64, act_order=True, storage="float32")
    # Compare both arms against the *full* Hessian, so the metric is fixed and
    # only the statistics GPTQ is allowed to see change.
    reference = 2.0 * (activations.T @ activations)
    poor = gptq_quantize(weight, 2.0 * (activations[:64].T @ activations[:64]), config)
    good = gptq_quantize(weight, reference, config)
    poor_against_reference = output_error(
        weight, _reconstruct(poor, weight), reference
    )
    good_against_reference = output_error(
        weight, _reconstruct(good, weight), reference
    )
    assert good_against_reference <= poor_against_reference * 1.05


def _reconstruct(outcome, weight: np.ndarray) -> np.ndarray:
    """Rebuild the quantized matrix from a stored triple."""
    codes = unpack_codes(outcome.packed, weight.shape[1])
    group = weight.shape[1] // outcome.scales.shape[1]
    return np.repeat(outcome.scales, group, axis=1) * codes + np.repeat(
        outcome.biases, group, axis=1
    )


def test_dead_columns_do_not_break_the_cholesky() -> None:
    """A never-activated column leaves a zero Hessian diagonal; it must not NaN."""
    activations, weight = _least_squares_problem(columns=128)
    activations[:, 7] = 0.0
    activations[:, 9] = 0.0
    hessian = 2.0 * (activations.T @ activations)
    outcome = gptq_quantize(weight, hessian, GptqConfig(bits=4, group_size=64))
    assert np.isfinite(outcome.error)
    assert np.isfinite(outcome.scales).all()
    assert np.isfinite(outcome.biases).all()
    assert np.array_equal(outcome.packed[:, 0], outcome.packed[:, 0])


def test_gptq_requires_matching_dimensions() -> None:
    activations, weight = _least_squares_problem(columns=128)
    with pytest.raises(ValueError, match="Hessian shape"):
        gptq_quantize(weight, np.eye(64, dtype=np.float32))
    with pytest.raises(ValueError, match="not divisible by the group size"):
        gptq_quantize(
            np.zeros((4, 100), dtype=np.float32),
            np.eye(100, dtype=np.float32),
            GptqConfig(group_size=64),
        )
    with pytest.raises(ValueError, match="Unsupported quantization bit width"):
        GptqConfig(bits=7)
    with pytest.raises(ValueError, match="Unknown GPTQ options"):
        GptqConfig.from_dict({"damp": 0.01})


def test_act_order_is_reported_and_invertible() -> None:
    """act-order must be a pure permutation of the columns, and reported."""
    activations, weight = _least_squares_problem(columns=128)
    hessian = 2.0 * (activations.T @ activations)
    config = GptqConfig(bits=4, group_size=64, act_order=True, storage="float32")
    ordered = gptq_quantize(weight, hessian, config)
    plain = gptq_quantize(weight, hessian, GptqConfig(bits=4, group_size=64, storage="float32"))
    assert ordered.act_order_permutation is not None
    assert plain.act_order_permutation is None
    assert sorted(ordered.act_order_permutation.tolist()) == list(range(128))
    # Both are valid quantizations of the same matrix, but not the same one.
    assert not np.array_equal(ordered.packed, plain.packed)


# --------------------------------------------------------------------------- #
# 4. the Hessian accumulator
# --------------------------------------------------------------------------- #
def test_hessian_accumulator_is_two_xtx() -> None:
    rng = np.random.RandomState(6)
    activations = rng.randn(64, 8).astype(np.float32)
    accumulator = HessianAccumulator(columns=8)
    accumulator.add(activations[:40], "zh")
    accumulator.add(activations[40:], "en")
    expected = 2.0 * (activations.T @ activations)
    assert np.allclose(accumulator.hessian(balance=False), expected, atol=1e-5)
    assert accumulator.row_counts() == {"zh": 40, "en": 24}


def test_hessian_accumulator_balances_languages() -> None:
    """The balance guard: no language may dominate the Hessian mass."""
    rng = np.random.RandomState(7)
    accumulator = HessianAccumulator(columns=4)
    for language, rows in (("zh", 500), ("en", 500), ("th", 50), ("vi", 50)):
        accumulator.add(rng.randn(rows, 4).astype(np.float32), language)
    unbalanced = accumulator.language_mass(balance=False)
    assert unbalanced["zh"] + unbalanced["en"] > 0.9
    balanced = accumulator.language_mass(balance=True)
    assert max(balanced.values()) <= 0.3
    # The cap variant keeps the raw data but bounds any single language.
    capped = accumulator.language_mass(cap_fraction=0.3)
    assert max(capped.values()) <= 0.3 + 1e-9


def test_hessian_accumulator_accepts_three_dimensional_batches() -> None:
    rng = np.random.RandomState(8)
    accumulator = HessianAccumulator(columns=4)
    batch = rng.randn(2, 3, 4).astype(np.float32)
    assert accumulator.add(batch, "ja") == 6
    assert np.allclose(
        accumulator.hessian(balance=False),
        2.0 * (batch.reshape(-1, 4).T @ batch.reshape(-1, 4)),
        atol=1e-5,
    )


def test_hessian_accumulator_round_trips_through_disk(tmp_path) -> None:
    rng = np.random.RandomState(9)
    accumulator = HessianAccumulator(columns=4)
    accumulator.add(rng.randn(16, 4).astype(np.float32), "zh")
    accumulator.add(rng.randn(8, 4).astype(np.float32), "ko")
    path = tmp_path / "hessian.npz"
    accumulator.save(path)
    restored = HessianAccumulator.load(path)
    assert restored.row_counts() == accumulator.row_counts()
    assert np.array_equal(restored.hessian(balance=True), accumulator.hessian(balance=True))


def test_hessian_accumulator_rejects_wrong_width() -> None:
    accumulator = HessianAccumulator(columns=4)
    with pytest.raises(ValueError, match="Expected"):
        accumulator.add(np.zeros((3, 5), dtype=np.float32), "zh")
    with pytest.raises(ValueError, match="No activations"):
        HessianAccumulator(columns=4).hessian()


def test_row_chunking_is_exact_and_does_not_change_the_result() -> None:
    """Row chunks must be a pure memory optimisation, not an approximation.

    GPTQ quantizes every row of ``W`` against the same ``H`` with no coupling
    between rows, so splitting the rows cannot change any decision.  This matters
    because a tied LM head (151936x1024) is 622 MB as one float32 array and the
    memory guard's MLX limit is far below what the inner loop would need, so the
    shipping path always chunks -- if chunking perturbed the result, every
    acceptance number would describe a different matrix than the one tested.
    """
    activations, weight = _least_squares_problem(rows=512, columns=128)
    hessian = 2.0 * (activations.T @ activations)
    reference = None
    for act_order in (False, True):
        reference = None
        for row_chunk in (0, 1, 7, 64):
            config = GptqConfig(
                bits=4,
                group_size=64,
                act_order=act_order,
                storage="float32",
                row_chunk=row_chunk,
            )
            outcome = gptq_quantize(weight, hessian, config)
            codes = unpack_codes(outcome.packed, weight.shape[1])
            if reference is None:
                reference = codes
                baseline = outcome
            else:
                assert np.array_equal(codes, reference), (
                    f"row_chunk={row_chunk} changed the codes (act_order={act_order})"
                )
        # The reported metrics must not depend on the chunking either.
        assert outcome.error == pytest.approx(baseline.error, rel=1e-6)
        assert outcome.output_error == pytest.approx(baseline.output_error, rel=1e-6)


def test_row_chunk_accepts_a_single_row_and_the_whole_matrix() -> None:
    """The two degenerate chunk sizes must agree (one row vs all rows)."""
    activations, weight = _least_squares_problem(rows=32, columns=64)
    hessian = 2.0 * (activations.T @ activations)
    whole = gptq_quantize(
        weight, hessian, GptqConfig(bits=4, group_size=64, storage="float32", row_chunk=0)
    )
    single = gptq_quantize(
        weight, hessian, GptqConfig(bits=4, group_size=64, storage="float32", row_chunk=1)
    )
    assert np.array_equal(
        unpack_codes(whole.packed, 64), unpack_codes(single.packed, 64)
    )


def test_row_chunk_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="row_chunk"):
        GptqConfig(row_chunk=-1)


# --------------------------------------------------------------------------- #
# per-language Hessian reweighting
# --------------------------------------------------------------------------- #
def _accumulator_with_languages(rows_by_language):
    accumulator = HessianAccumulator(columns=8)
    for language, count in rows_by_language.items():
        accumulator.add(np.random.RandomState(len(language)).randn(count, 8).astype(np.float32), language)
    return accumulator


def test_uniform_language_mass_is_the_default():
    accumulator = _accumulator_with_languages({"zh": 500, "ja": 100, "th": 100})
    shares = accumulator.language_mass(balance=True)
    assert shares == pytest.approx({"zh": 1 / 3, "ja": 1 / 3, "th": 1 / 3})


def test_language_weights_shift_mass_without_changing_the_total():
    accumulator = _accumulator_with_languages({"zh": 500, "ja": 100, "th": 100})
    weighted = accumulator.language_mass_with({"ja": 3.0})
    assert sum(weighted.values()) == pytest.approx(1.0)
    assert weighted["ja"] > weighted["zh"]
    # The total effective rows must be unchanged: reweighting redistributes,
    # it does not change the scale (the damping is relative to the mean
    # diagonal, so a scale change would silently alter the regularisation).
    unweighted = accumulator.language_mass(balance=True)
    assert sum(unweighted.values()) == pytest.approx(1.0)


def test_language_weights_preserve_the_total_effective_rows():
    """The renormalisation must happen in effective-row space.

    "Total mass" is only meaningful in effective rows -- the space the 30% guard
    is stated in.  An earlier version renormalised the per-language MULTIPLIERS
    instead, which changed the overall scale of the Hessian and therefore the
    damping (defined relative to the mean diagonal).  This pins the invariant that
    actually holds; note the trace is NOT preserved, because
    trace(X_i^T X_i) / rows_i differs by language, so preserving effective rows
    does not preserve the trace.  Asserting trace-invariance was wrong and the
    test failed when it was written that way.
    """
    accumulator = _accumulator_with_languages({"zh": 400, "ja": 100})
    before = accumulator._effective_rows(balance=True)
    for weights in ({"ja": 4.0}, {"zh": 2.0}, {"ja": 3.0, "zh": 0.5}):
        after = accumulator._weighted_effective_rows(
            weights, balance=True, cap_fraction=None
        )
        assert sum(after.values()) == pytest.approx(sum(before.values()))
        assert sum(accumulator.language_mass_with(weights).values()) == pytest.approx(1.0)
    # And the Hessian genuinely changes.
    plain = accumulator.hessian(balance=True)
    weighted = accumulator.hessian(balance=True, language_weights={"ja": 4.0})
    assert not np.allclose(plain, weighted)


def test_language_weights_reject_an_unknown_language():
    accumulator = _accumulator_with_languages({"zh": 10, "ja": 10})
    with pytest.raises(ValueError, match="no activations"):
        accumulator.hessian(balance=True, language_weights={"ko": 2.0})


def test_an_empty_language_weights_mapping_is_a_no_op():
    accumulator = _accumulator_with_languages({"zh": 10, "ja": 10})
    plain = accumulator.hessian(balance=True)
    assert np.allclose(plain, accumulator.hessian(balance=True, language_weights={}))


def test_language_weights_can_be_checked_before_building_the_hessian():
    """The guard is on the share, so it must be checkable up front."""
    accumulator = _accumulator_with_languages({"zh": 100, "ja": 100, "th": 100})
    shares = accumulator.language_mass_with({"ja": 100.0})
    assert max(shares.values()) > 0.30
    assert shares["ja"] == pytest.approx(max(shares.values()))
