"""GPTQ (Hessian-based error-compensating PTQ) for the LLM-ASR weight matrices.

Why this module exists
----------------------
The tied LM head / token embedding of the qwen3-shaped families is 311-485 MB of
the bf16 checkpoint and is streamed once per decode step, so it is a hard floor
on decode speed (task-9's ledger: 44% of the quantized step).  Round-to-nearest
group quantization of that matrix (``mx.quantize``) is measured to destroy the
transcript from step 0, so it was left at bf16.

GPTQ (Frantar et al., 2022) keeps the *exact same runtime layout* -- standard MLX
affine group quantization, so the existing ``QuantizedLinear`` loader, scope
predicates and verification harness are reused unchanged -- and only computes
better *values* for the packed codes, ``scales`` and ``biases``.  The error of
each column is compensated in the not-yet-quantized columns, using the
second-order statistics of the layer's real input distribution.

Algorithm
---------
For a target matrix ``W`` of shape ``[out, in]`` and calibration activations
``X`` of shape ``[N, in]`` collected through the real (frozen) forward pass::

    H   = 2 X^T X                      (float32 accumulation, [in, in])
    H  += lambda * I,  lambda = damp_percent * mean(diag(H))
    U   = chol(inv(H))^T               (upper triangular: Hinv = U^T U)
    for each block of `block_size` columns:
        for each column i in the block:
            q_i = quantize(W[:, i])            # the MLX affine quantizer
            err = (W[:, i] - q_i) / U[i, i]
            W[:, i:] -= err * U[i, i:]

``act_order`` permutes the columns into descending ``diag(H)`` order first (and
back afterwards), which matters most for embedding-like matrices with outlier
directions: the most influential columns are quantized first, so their error is
compensated by the columns that still have room to absorb it.  Because the
permutation makes a stored group's columns non-contiguous, act-order is paired
with **static groups** -- the group parameters are taken once, from the original
matrix, over each group's own columns in storage order.  Without act-order the
columns are visited in storage order and the parameters are taken dynamically
from the running (already error-compensated) matrix, as Frantar describes.

The quantizer is :class:`AffineQuantizer`, a bit-exact port of MLX's
``affine_quantize`` kernel (``mlx/backend/metal/kernels/quantized.h``), so GPTQ
output loads through ``mx.quantized_matmul`` with no runtime change at all.  The
kernel's exact form matters and is *not* the textbook ``(max - min) / 2^b - 1``:
it picks the sign of the scale by which end of the group has the larger
magnitude and then renormalizes the scale so that end lands exactly on an
integer code, which is why a naive reimplementation disagrees with MLX on real
weights (see ``tests/test_gptq.py``).

The inner loop stores the codes and the group parameters **as it chooses them**.
Re-deriving the triple afterwards by re-quantizing the finished matrix looks
equivalent (the triple stays self-consistent) but is not: it moved this
implementation's layer-output error from 135 to 598 on the module's own test
problem, destroying most of the gain.  The stored triple is what the error
metrics are computed on.

Numeric contract
----------------
* All Hessian accumulation and the whole GPTQ inner loop run in **float32**,
  whatever the checkpoint dtype is.
* ``scales`` and ``biases`` are rounded to the **storage dtype** before the
  codes are computed from them, so the stored triple dequantizes exactly as the
  runtime sees it (bit-exact against ``mx.dequantize``).
* No MLX is required to run the core; ``mlx.core`` is imported lazily, only for
  the dtype cast of the side tensors (bf16 has no NumPy representation).

This module is deliberately free of model knowledge: it takes a weight, a
Hessian and a config.  Collecting the Hessian through the real model is the
caller's job (:class:`HessianAccumulator` provides the accumulation and the
language-balance guard).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "AffineQuantizer",
    "GptqConfig",
    "GptqOutcome",
    "HessianAccumulator",
    "gptq_quantize",
    "output_error",
    "pack_codes",
    "rtn_quantize",
    "unpack_codes",
]

#: Epsilon from MLX's ``affine_quantize`` kernel: the smallest scale it will use.
_MLX_SCALE_EPS = 1e-7

#: Storage dtypes a quantized checkpoint can carry its ``scales``/``biases`` in.
_STORAGE_DTYPES = ("bfloat16", "float16", "float32")


# --------------------------------------------------------------------------- #
# the quantizer
# --------------------------------------------------------------------------- #
def _round_half_away(values: np.ndarray) -> np.ndarray:
    """``round()`` with ties away from zero, as the MLX kernel does.

    NumPy's ``np.round`` is half-to-even, which disagrees with the kernel on
    exact .5 ties; they are rare on real weights but not on synthetic ones.
    """
    return np.floor(np.abs(values) + 0.5) * np.sign(values)


class AffineQuantizer:
    """MLX affine group quantization, bit-exact (see module docstring).

    ``bits`` and ``group_size`` must be a combination ``mx.quantize`` accepts:
    bits in (2, 3, 4, 5, 6, 8) and group size in (32, 64, 128).  ``storage``
    names the dtype the side tensors are stored in; it only affects rounding of
    ``scales``/``biases`` and therefore which codes are chosen.
    """

    def __init__(self, *, bits: int = 4, group_size: int = 64, storage: str = "bfloat16"):
        if bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"Unsupported quantization bit width: {bits}")
        if group_size not in (32, 64, 128):
            raise ValueError(f"Unsupported quantization group size: {group_size}")
        if storage not in _STORAGE_DTYPES:
            raise ValueError(f"Unsupported storage dtype: {storage}")
        self.bits = bits
        self.group_size = group_size
        self.storage = storage
        self.max_code = (1 << bits) - 1
        #: Number of codes packed into one uint32 word (8 for 4 bits, 4 for 8).
        self.pack_factor = 32 // bits
        self._casts = 0

    @property
    def storage_dtype(self) -> np.dtype:
        """NumPy dtype the side tensors are returned in.

        bfloat16 has no NumPy representation, so it is carried as float32 whose
        *values* are already bf16-rounded -- converting to bf16 later is exact.
        """
        if self.storage == "float16":
            return np.dtype(np.float16)
        return np.dtype(np.float32)

    # -- side tensors ------------------------------------------------------- #
    def cast(self, values: np.ndarray) -> np.ndarray:
        """Round ``values`` to the storage dtype, keeping a float32 carrier."""
        values = np.asarray(values, dtype=np.float32)
        if self.storage == "float32":
            return values
        if self.storage == "float16":
            return values.astype(np.float16).astype(np.float32)
        import mlx.core as mx

        self._casts += 1
        tensor = mx.array(values).astype(mx.bfloat16).astype(mx.float32)
        mx.eval(tensor)
        return np.array(tensor)

    def params(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-row ``(scale, bias)`` for one group, exactly as the kernel does.

        ``block`` is ``[rows, group_size]`` float32.  The kernel's three steps
        are reproduced verbatim:

        1. ``scale = max((max - min) / n_bins, eps)``, with ``max`` floored at
           zero (the kernel initializes ``w_max = 0``);
        2. the sign is chosen by which end has the larger magnitude -- so the
           stored bias is that end and the scale's sign matches;
        3. the scale is renormalized to ``edge / round(edge / scale)``, so the
           edge value lands exactly on an integer code.  A group whose edge
           rounds to zero keeps the raw scale and a zero bias.
        """
        block = np.asarray(block, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != self.group_size:
            raise ValueError(
                f"Expected a [rows, {self.group_size}] group, got {block.shape}"
            )
        low = block.min(axis=1)
        high = np.maximum(block.max(axis=1), 0.0)
        scale = np.maximum((high - low) / self.max_code, _MLX_SCALE_EPS)
        low_side = np.abs(low) > np.abs(high)
        scale = np.where(low_side, scale, -scale)
        edge = np.where(low_side, low, high)
        codes = _round_half_away(edge / scale)
        at_zero = codes == 0.0
        safe = np.where(at_zero, 1.0, codes)
        scale = np.where(at_zero, scale, edge / safe)
        bias = np.where(at_zero, 0.0, edge)
        return self.cast(scale), self.cast(bias)

    def codes(self, values: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Codes for ``values`` ``[rows, cols]`` under per-row ``scale``/``bias``.

        A 1-D ``values`` is treated as a single column (``[rows, 1]``), which is
        the shape the GPTQ inner loop works in.
        """
        values = np.asarray(values, dtype=np.float32)
        single = values.ndim == 1
        if single:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError(f"Expected a 1-D or 2-D array, got shape {values.shape}")
        scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        bias = np.asarray(bias, dtype=np.float32).reshape(-1)
        if scale.shape != (values.shape[0],) or bias.shape != (values.shape[0],):
            raise ValueError(
                f"Expected one scale/bias per row ({values.shape[0]}), got "
                f"{scale.shape} and {bias.shape}"
            )
        x = (values - bias[:, None]) / scale[:, None]
        clipped = np.minimum(_round_half_away(x), self.max_code)
        # A negative scale is legal in this scheme (bias is the max end), so the
        # low clamp is applied to the code, not to x.
        return np.clip(clipped, 0, self.max_code).astype(np.uint8)

    def dequantize(
        self, codes: np.ndarray, scale: np.ndarray, bias: np.ndarray
    ) -> np.ndarray:
        """``scale * code + bias`` -- the affine reconstruction the runtime does."""
        codes = np.asarray(codes, dtype=np.float32)
        single = codes.ndim == 1
        if single:
            codes = codes[:, None]
        scale = np.asarray(scale, dtype=np.float32).reshape(-1)
        bias = np.asarray(bias, dtype=np.float32).reshape(-1)
        out = codes * scale[:, None] + bias[:, None]
        return out[:, 0] if single else out

    def quantize(
        self, values: np.ndarray, scale: np.ndarray, bias: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(codes, reconstruction)`` for ``values`` under given params.

        The output rank follows the input: a 1-D column gives 1-D results, which
        is the shape the GPTQ inner loop works in.
        """
        single = np.asarray(values).ndim == 1
        codes = self.codes(values, scale, bias)
        reconstruction = self.dequantize(codes, scale, bias)
        if single:
            return codes[:, 0], reconstruction[:, 0]
        return codes, reconstruction

    # -- packing ------------------------------------------------------------ #
    def pack(self, codes: np.ndarray) -> np.ndarray:
        """Pack ``uint8`` codes into ``uint32`` words, MLX's bit order.

        Element ``j`` of a group occupies bits ``bits * (j % pack_factor)`` of
        word ``j // pack_factor`` -- little-endian, lowest code in the lowest
        bits, as documented for ``mx.quantize``.
        """
        codes = np.asarray(codes)
        if codes.ndim != 2:
            raise ValueError("Codes must be a 2-D array")
        rows, columns = codes.shape
        if columns % self.pack_factor:
            raise ValueError(
                f"{columns} codes do not fill {self.pack_factor}-per-word groups"
            )
        words = np.zeros((rows, columns // self.pack_factor), dtype=np.uint32)
        mask = np.uint32(self.max_code)
        for slot in range(self.pack_factor):
            shifted = codes[:, slot:: self.pack_factor].astype(np.uint32) & mask
            words |= shifted << np.uint32(self.bits * slot)
        return words

    def unpack(self, words: np.ndarray, columns: int) -> np.ndarray:
        """Inverse of :meth:`pack` (used by the tests and the verifier)."""
        words = np.asarray(words, dtype=np.uint32)
        codes = np.zeros((words.shape[0], columns), dtype=np.uint8)
        mask = np.uint32(self.max_code)
        for slot in range(self.pack_factor):
            codes[:, slot:: self.pack_factor] = (
                (words >> np.uint32(self.bits * slot)) & mask
            ).astype(np.uint8)
        return codes


def pack_codes(codes: np.ndarray, *, bits: int = 4) -> np.ndarray:
    """Convenience wrapper: pack ``uint8`` codes with MLX's bit order."""
    return AffineQuantizer(bits=bits).pack(codes)


def unpack_codes(words: np.ndarray, columns: int, *, bits: int = 4) -> np.ndarray:
    """Convenience wrapper: unpack ``uint32`` words with MLX's bit order."""
    return AffineQuantizer(bits=bits).unpack(words, columns)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GptqConfig:
    """GPTQ hyper-parameters, mirroring the reference implementation's knobs."""

    bits: int = 4
    group_size: int = 64
    #: Hessian dampening, as a fraction of ``mean(diag(H))``.
    damp_percent: float = 0.01
    #: Columns per lazy batch update (Frantar's act-order note).
    block_size: int = 128
    #: Quantize in descending-``diag(H)`` order (recommended for embeddings).
    #: This permutes the columns, so it is paired with **static groups**: the
    #: group parameters are taken once, from the original matrix, over each
    #: group's own columns in storage order.  (Frantar's act-order note makes the
    #: same pairing; with dynamic groups the stored codes and the stored side
    #: tensors would describe different groups.)
    act_order: bool = False
    #: Dtype the side tensors are stored in (must match the checkpoint dtype).
    storage: str = "bfloat16"
    #: Rows processed per pass.  GPTQ is row-independent -- every row of ``W`` is
    #: quantized against the same ``H`` with no coupling between rows -- so this
    #: is exact, not an approximation.  It exists because a tied LM head is large
    #: (151936x1024 is 622 MB as a single float32 array, and the inner loop wants
    #: several such arrays live at once) while the memory guard's MLX limit is far
    #: below that.  0 means "all rows in one pass".
    row_chunk: int = 8192

    def __post_init__(self) -> None:
        if self.bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"Unsupported quantization bit width: {self.bits}")
        if self.group_size not in (32, 64, 128):
            raise ValueError(f"Unsupported quantization group size: {self.group_size}")
        if not np.isfinite(self.damp_percent) or self.damp_percent < 0:
            raise ValueError("damp_percent must be finite and non-negative")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.row_chunk < 0:
            raise ValueError("row_chunk must be non-negative")
        if self.storage not in _STORAGE_DTYPES:
            raise ValueError(f"Unsupported storage dtype: {self.storage}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "GptqConfig":
        data = dict(data or {})
        known = {field_name for field_name in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown GPTQ options: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass
class GptqOutcome:
    """The three tensors a quantized checkpoint stores, plus error accounting.

    Two error metrics are reported, and they are not interchangeable:

    ``error``
        Mean squared **Frobenius** error ``mean((W - Ŵ)^2)`` -- the plain
        elementwise distance.
    ``output_error``
        The quantity GPTQ actually minimizes: ``trace(D (H/2) D^T)`` with
        ``D = W - Ŵ``, i.e. the total squared error of the layer's *outputs*,
        ``sum_n ||(W - Ŵ) x_n||^2``, over the calibration activations.  This is
        the metric that matters for the LM head (it is the logits error), and
        it is computable from the Hessian alone.

    GPTQ is guaranteed to beat round-to-nearest on ``output_error``, not on
    ``error``: it deliberately spends unweighted error to buy weighted error.
    Reporting only the Frobenius number would make a working GPTQ look worse
    than RTN, so both are always carried.
    """

    #: ``uint32`` packed codes, ``[out, in * bits / 32]``.
    packed: np.ndarray
    #: Per-group scale, ``[out, in / group_size]``, in the storage dtype.
    scales: np.ndarray
    #: Per-group bias, same shape and dtype as ``scales``.
    biases: np.ndarray
    #: Mean squared Frobenius error of the reconstructed weight.
    error: float
    #: Total squared layer-output error ``trace(D (H/2) D^T)``.
    output_error: float | None = None
    #: Mean squared error of plain round-to-nearest on the same matrix.
    rtn_error: float | None = None
    #: Output error of plain round-to-nearest on the same matrix.
    rtn_output_error: float | None = None
    #: Fraction of *output* error removed relative to round-to-nearest.
    error_reduction: float | None = None
    #: Fraction of Frobenius error removed relative to round-to-nearest (may be
    #: negative: see the class docstring).
    frobenius_reduction: float | None = None
    #: Column permutation applied by act-order (``None`` when disabled).
    act_order_permutation: np.ndarray | None = None
    method: str = "gptq"

    def to_json(self) -> dict:
        return {
            "method": self.method,
            "error": self.error,
            "output_error": self.output_error,
            "rtn_error": self.rtn_error,
            "rtn_output_error": self.rtn_output_error,
            "error_reduction": self.error_reduction,
            "frobenius_reduction": self.frobenius_reduction,
            "act_order": self.act_order_permutation is not None,
            "packed_shape": list(self.packed.shape),
            "scales_shape": list(self.scales.shape),
        }


# --------------------------------------------------------------------------- #
# GPTQ core
# --------------------------------------------------------------------------- #
def output_error(weight: np.ndarray, reconstructed: np.ndarray, hessian: np.ndarray) -> float:
    """Total squared layer-output error of a reconstruction, from ``H`` alone.

    For ``D = W - Ŵ`` and ``H = 2 X^T X``::

        sum_n ||D x_n||^2 = trace(D (H/2) D^T)

    so no activations are needed once the Hessian exists -- which is what makes
    this reportable for a matrix whose calibration set is 600 utterances long.

    Computed in row blocks.  ``trace(D M D^T)`` decomposes as a sum over the
    ROWS of ``D`` (the ``i``-th row contributes ``D[i] M D[i]^T``), so this is
    exact -- and it has to be blocked, because materialising ``D @ M @ D.T`` for
    a 151936-row head needs a 151936x151936 intermediate.
    """
    difference = np.asarray(weight, dtype=np.float32) - np.asarray(
        reconstructed, dtype=np.float32
    )
    half = 0.5 * np.asarray(hessian, dtype=np.float32)
    total = 0.0
    block = 4096
    for start in range(0, difference.shape[0], block):
        piece = difference[start : start + block]
        total += float(np.sum((piece @ half) * piece))
    return total


def _upper_cholesky_inverse(hessian: np.ndarray) -> np.ndarray:
    """``U`` with ``Hinv = U^T U`` and ``U`` upper triangular (float32 out).

    The reference implementation goes ``chol(H) -> cholesky_inverse -> chol(·,
    upper=True)``; ``inv(H)`` then an upper Cholesky is the same matrix with one
    fewer decomposition, and the dampening has already made ``H`` positive
    definite.
    """
    inverse = np.linalg.inv(hessian)
    # Symmetrize: the inverse of a symmetric matrix is symmetric only up to
    # rounding, and Cholesky reads the lower triangle.
    inverse = 0.5 * (inverse + inverse.T)
    return np.linalg.cholesky(inverse).T.astype(np.float32)


def _check_inputs(weight: np.ndarray, hessian: np.ndarray, config: GptqConfig) -> None:
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight, got shape {weight.shape}")
    if hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError(
            f"Hessian shape {hessian.shape} does not match the weight's input "
            f"dimension {weight.shape[1]}"
        )
    if weight.shape[1] % config.group_size:
        raise ValueError(
            f"Input dimension {weight.shape[1]} is not divisible by the group "
            f"size {config.group_size}"
        )


def gptq_quantize(
    weight: np.ndarray,
    hessian: np.ndarray,
    config: GptqConfig | None = None,
    *,
    with_rtn_baseline: bool = True,
) -> GptqOutcome:
    """Quantize ``weight`` ``[out, in]`` with GPTQ using ``hessian`` ``[in, in]``.

    ``hessian`` is the second-order statistic of the layer input,
    ``H = 2 X^T X`` accumulated in float32 (:class:`HessianAccumulator` does
    exactly that).  The returned triple is what a W4A16 checkpoint stores, in
    the standard MLX affine group layout.
    """
    config = config or GptqConfig()
    weight = np.asarray(weight, dtype=np.float32)
    hessian = np.asarray(hessian, dtype=np.float32)
    _check_inputs(weight, hessian, config)

    # Keep the caller's basis: act-order permutes ``weight``/``hessian`` below,
    # and the reported error must be measured in the original coordinates.
    original_weight = weight
    original_hessian = hessian

    quantizer = AffineQuantizer(
        bits=config.bits, group_size=config.group_size, storage=config.storage
    )
    columns = weight.shape[1]

    rtn_error = None
    rtn_output_error = None
    if with_rtn_baseline:
        rtn_reconstruction = _rtn_reconstruct(original_weight, quantizer)
        rtn_error = float(np.mean((original_weight - rtn_reconstruction) ** 2))
        rtn_output_error = output_error(
            original_weight, rtn_reconstruction, original_hessian
        )

    # A column the calibration never exercised has a zero Hessian diagonal; it
    # carries no second-order information, so it must not be allowed to blow up
    # the inverse.  Zero the column and give the Hessian a unit diagonal there,
    # exactly as the reference does.
    dead = np.diag(hessian) == 0
    if dead.any():
        hessian = hessian.copy()
        hessian[dead, dead] = 1.0
        weight = weight.copy()
        weight[:, dead] = 0.0

    permutation = None
    inverse_permutation = None
    if config.act_order:
        permutation = np.argsort(-np.diag(hessian)).astype(np.int64)
        inverse_permutation = np.argsort(permutation)
        weight = weight[:, permutation]
        hessian = hessian[np.ix_(permutation, permutation)]
    damp = config.damp_percent * float(np.mean(np.diag(hessian)))
    damped = hessian.copy()
    diagonal = np.arange(columns)
    damped[diagonal, diagonal] += damp
    upper = _upper_cholesky_inverse(damped)

    # Group parameters, keyed by the group index in the *stored* layout.
    #
    # Without act-order the columns are visited in storage order, so a group's
    # parameters are simply taken when the loop reaches its first column, from
    # the running (already error-compensated) matrix -- Frantar's "dynamic
    # groups".  With act-order the columns arrive in permuted order, so the
    # columns of one stored group are no longer contiguous, and taking the
    # parameters from the permuted matrix would leave the stored codes and the
    # stored side tensors describing different groups (the checkpoint would
    # dequantize to something the metrics never described).  act-order is
    # therefore paired with **static groups**, as in the reference
    # implementation: the parameters are chosen once, from the original matrix,
    # over the group's own columns in storage order.
    group_count = columns // config.group_size
    codes = np.empty(weight.shape, dtype=np.uint8)
    scales = np.empty((weight.shape[0], group_count), dtype=np.float32)
    biases = np.empty_like(scales)
    chunk = config.row_chunk or weight.shape[0]

    # GPTQ is row-independent: each row is quantized against the same ``H`` with
    # no coupling between rows, so processing row chunks is exact.  It is what
    # keeps a 151936x1024 tied head inside the guard's MLX budget -- one full
    # float32 copy of that matrix is already 622 MB, and the inner loop wants
    # several live at once.
    for row_start in range(0, weight.shape[0], chunk):
        row_stop = min(row_start + chunk, weight.shape[0])
        running = weight[row_start:row_stop].copy()
        quantized = np.empty_like(running)

        # Group parameters are keyed by the group index in the *STORED* layout,
        # because that is what the checkpoint carries.
        #
        # Without act-order the columns are visited in storage order, so a
        # group's parameters are simply taken when the loop reaches its first
        # column, read off the running (already error-compensated) matrix --
        # Frantar's "dynamic groups".  With act-order the columns arrive in
        # permuted order, so a stored group's columns are no longer contiguous
        # and taking the parameters from the permuted matrix would leave the
        # stored codes and the stored side tensors describing different groups
        # (the checkpoint would dequantize to something the metrics never
        # described).  act-order is therefore paired with **static groups**, as
        # in the reference implementation: the parameters are chosen once, from
        # the original matrix, over the group's own columns in storage order.
        #
        # Either way the parameters are per-chunk: they are read off this chunk's
        # rows, so they are recomputed here rather than shared across chunks.
        static_params: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if config.act_order:
            for index in range(group_count):
                start = index * config.group_size
                static_params[index] = quantizer.params(
                    original_weight[row_start:row_stop, start : start + config.group_size]
                )
        dynamic_params: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def stored_params(
            index: int, current: np.ndarray = running
        ) -> tuple[np.ndarray, np.ndarray]:
            """Parameters for stored group ``index`` on this chunk's rows."""
            if index in static_params:
                return static_params[index]
            if index not in dynamic_params:
                stop = (index + 1) * config.group_size
                dynamic_params[index] = quantizer.params(
                    current[:, index * config.group_size : stop]
                )
            return dynamic_params[index]

        def params_for(column: int) -> tuple[np.ndarray, np.ndarray]:
            """Parameters for the stored group that column ``column`` lands in."""
            stored_column = column if permutation is None else int(permutation[column])
            return stored_params(stored_column // config.group_size)

        for start in range(0, columns, config.block_size):
            stop = min(start + config.block_size, columns)
            # ``block`` is this block's slice of the running (error-compensated)
            # matrix.  The compensation must be applied to it as the loop
            # proceeds, so it is kept separate from the values chosen.
            block = running[:, start:stop].copy()
            block_upper = upper[start:stop, start:stop]
            block_error = np.empty_like(block)
            for offset in range(stop - start):
                column = start + offset
                scale, bias = params_for(column)
                block_codes, reconstruction = quantizer.quantize(
                    block[:, offset], scale, bias
                )
                codes[row_start:row_stop, column] = block_codes
                quantized[:, offset] = reconstruction
                pivot = float(block_upper[offset, offset])
                error = (block[:, offset] - reconstruction) / pivot
                block_error[:, offset] = error
                if offset + 1 < block.shape[1]:
                    block[:, offset + 1 :] -= np.outer(
                        error, block_upper[offset, offset + 1 :]
                    )
            if stop < columns:
                # Lazy batch update (Frantar's act-order note): every column in
                # the block contributes ``error_i * Hinv[i, stop:]`` to the
                # remaining columns, so the whole block's compensation is applied
                # at once.  ``error_i`` is the *scaled* error
                # ``(w - q) / Hinv[i, i]`` -- the reference's ``Err1`` -- and NOT
                # ``block - block_quantized``, which differs by a factor of
                # ``Hinv[i, i]`` per column because the running block has already
                # absorbed the earlier columns' updates.
                running[:, stop:] -= block_error @ upper[start:stop, stop:]

        for index in range(group_count):
            scale, bias = stored_params(index)
            scales[row_start:row_stop, index] = scale
            biases[row_start:row_stop, index] = bias

        if inverse_permutation is not None:
            codes[row_start:row_stop] = codes[row_start:row_stop][:, inverse_permutation]

        del running, quantized, block, block_error

    dtype = quantizer.storage_dtype
    packed = quantizer.pack(codes)
    scales = scales.astype(dtype)
    biases = biases.astype(dtype)
    # The metrics are computed from the STORED triple, not accumulated from the
    # inner loop's own arrays.  Those are not the same object -- the loop's
    # per-chunk reconstruction is built from the running (error-compensated)
    # matrix, while the checkpoint stores codes and side tensors that are read
    # back through a different path -- and accumulating chunk-local traces of it
    # did not reproduce the stored triple's metric (measured 1112899.59 against
    # the true 392.65 on the module's own test problem).  Rebuilding the
    # reconstruction from the stored triple, in row blocks so no full float32
    # copy is materialised, makes the reported numbers describe exactly what a
    # reader of the checkpoint would get.
    #
    # ``trace(D (H/2) D^T)`` does decompose over the rows of ``D`` (row ``i``
    # contributes ``D[i] (H/2) D[i]^T``), so blocking it is exact, not an
    # approximation.
    error = 0.0
    result_output_error = 0.0
    half = 0.5 * original_hessian
    for start in range(0, weight.shape[0], chunk):
        stop = min(start + chunk, weight.shape[0])
        piece = (
            np.repeat(scales[start:stop], config.group_size, axis=1)
            * codes[start:stop]
            + np.repeat(biases[start:stop], config.group_size, axis=1)
        )
        difference = original_weight[start:stop] - piece
        error += float(np.sum(difference**2))
        result_output_error += float(np.sum((difference @ half) * difference))
        del piece, difference
    error /= weight.shape[0] * columns
    del codes
    reduction = None
    frobenius_reduction = None
    if rtn_output_error:
        reduction = 1.0 - result_output_error / rtn_output_error
    if rtn_error:
        frobenius_reduction = 1.0 - error / rtn_error
    return GptqOutcome(
        packed=packed,
        scales=scales,
        biases=biases,
        error=error,
        output_error=result_output_error,
        rtn_error=rtn_error,
        rtn_output_error=rtn_output_error,
        error_reduction=reduction,
        frobenius_reduction=frobenius_reduction,
        act_order_permutation=permutation,
    )


def rtn_quantize(
    weight: np.ndarray,
    config: GptqConfig | None = None,
    hessian: np.ndarray | None = None,
) -> GptqOutcome:
    """Plain round-to-nearest group quantization (``mx.quantize``'s own rule).

    Provided as the baseline GPTQ has to beat, computed with the *same*
    quantizer so the comparison isolates the algorithm and not the rounding.
    Pass ``hessian`` to also get the layer-output error, which is the metric the
    head-to-head claim is made on.
    """
    config = config or GptqConfig()
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight, got shape {weight.shape}")
    if weight.shape[1] % config.group_size:
        raise ValueError(
            f"Input dimension {weight.shape[1]} is not divisible by the group "
            f"size {config.group_size}"
        )
    quantizer = AffineQuantizer(
        bits=config.bits, group_size=config.group_size, storage=config.storage
    )
    reconstructed = _rtn_reconstruct(weight, quantizer)
    error = float(np.mean((weight - reconstructed) ** 2))
    packed, scales, biases = _pack_triple(reconstructed, quantizer)
    result_output_error = None
    if hessian is not None:
        result_output_error = output_error(weight, reconstructed, hessian)
    return GptqOutcome(
        packed=packed,
        scales=scales,
        biases=biases,
        error=error,
        output_error=result_output_error,
        rtn_error=error,
        rtn_output_error=result_output_error,
        error_reduction=0.0,
        frobenius_reduction=0.0,
        method="rtn",
    )


def _rtn_reconstruct(weight: np.ndarray, quantizer: AffineQuantizer) -> np.ndarray:
    """Reconstruct ``weight`` by quantizing each group independently."""
    group = quantizer.group_size
    out = np.empty_like(weight, dtype=np.float32)
    for start in range(0, weight.shape[1], group):
        stop = min(start + group, weight.shape[1])
        scale, bias = quantizer.params(weight[:, start:stop])
        _, out[:, start:stop] = quantizer.quantize(weight[:, start:stop], scale, bias)
    return out


def _pack_triple(
    reconstructed: np.ndarray, quantizer: AffineQuantizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-derive the stored triple from a reconstructed matrix.

    Deriving the triple from the *final* reconstructed values (rather than
    keeping the per-column codes) guarantees the checkpoint is self-consistent:
    ``mx.dequantize(packed, scales, biases)`` reproduces exactly what GPTQ
    decided, with no dependency on how the inner loop was organized.
    """
    group = quantizer.group_size
    columns = reconstructed.shape[1]
    scales = np.empty((reconstructed.shape[0], columns // group), dtype=np.float32)
    biases = np.empty_like(scales)
    codes = np.empty((reconstructed.shape[0], columns), dtype=np.uint8)
    for index, start in enumerate(range(0, columns, group)):
        stop = start + group
        scale, bias = quantizer.params(reconstructed[:, start:stop])
        scales[:, index] = scale
        biases[:, index] = bias
        codes[:, start:stop] = quantizer.codes(reconstructed[:, start:stop], scale, bias)
    dtype = quantizer.storage_dtype
    return quantizer.pack(codes), scales.astype(dtype), biases.astype(dtype)


# --------------------------------------------------------------------------- #
# Hessian collection
# --------------------------------------------------------------------------- #
@dataclass
class HessianAccumulator:
    """``H = 2 X^T X`` in float32, with a per-language balance guard.

    Calibration activations are added per utterance with a ``language`` tag.
    Because Chinese and English have far more local data than the other eight
    languages, raw row counts would put ~85% of the Hessian mass on two
    languages; :meth:`hessian` therefore reweights each language's contribution
    so every language contributes the same number of effective rows (see the
    ``balance`` argument), which keeps any single language below the ~30% mass
    guard the brief sets.

    The accumulator is model-agnostic: the caller decides which activations to
    feed it.  ``columns`` must be the matrix's input dimension.
    """

    columns: int
    #: Per-language ``X^T X`` (float32) and row count.
    sums: dict[str, np.ndarray] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.columns <= 0:
            raise ValueError("columns must be positive")

    def add(self, activations: np.ndarray, language: str = "") -> int:
        """Accumulate one batch of activations ``[N, columns]`` (float32 math)."""
        activations = np.asarray(activations, dtype=np.float32)
        if activations.ndim == 3:
            # [batch, sequence, hidden] -> rows
            activations = activations.reshape(-1, activations.shape[-1])
        if activations.ndim != 2 or activations.shape[1] != self.columns:
            raise ValueError(
                f"Expected [N, {self.columns}] activations, got {activations.shape}"
            )
        if activations.shape[0] == 0:
            return 0
        product = activations.T @ activations
        bucket = self.sums.get(language)
        if bucket is None:
            self.sums[language] = product
        else:
            bucket += product
        self.rows[language] = self.rows.get(language, 0) + int(activations.shape[0])
        return int(activations.shape[0])

    def row_counts(self) -> dict[str, int]:
        return dict(self.rows)

    def hessian(
        self,
        *,
        balance: bool = True,
        cap_fraction: float | None = None,
        language_weights: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """The combined ``H = 2 X^T X``, optionally language-balanced.

        ``balance=True`` scales each language's contribution to the smallest
        per-language row count, so every language contributes equally and none
        can exceed ``1 / n_languages`` of the mass.  ``cap_fraction`` instead
        caps the largest language's *share* at that fraction by lowering a
        common row threshold (see :meth:`_effective_rows`), which keeps more of
        a large corpus while still bounding the dominant language.

        ``language_weights`` multiplies the balanced per-language contributions
        by explicit factors.  Equal mass is the right *default* (it is what the
        brief's 30% guard asks for), but it is not the right target when the
        languages do not quantize equally well: if one language's transcripts
        diverge far more than another's at the same bit width, equal mass is
        under-weighting exactly the directions that are failing.  Weighting by
        measured divergence spends a fixed bit budget where the error actually
        shows up.  Weights are renormalised, so this redistributes mass between
        languages without changing the total -- an overall scale would silently
        alter the damping, which is defined relative to the mean diagonal.
        """
        if not self.sums:
            raise ValueError("No activations have been accumulated")
        weights = self._row_weights(balance=balance, cap_fraction=cap_fraction)
        if language_weights:
            unknown = set(language_weights) - set(self.sums)
            if unknown:
                raise ValueError(
                    "language_weights names languages with no activations: "
                    f"{sorted(unknown)}"
                )
            weights = self._apply_language_weights(weights, language_weights)
        total = np.zeros((self.columns, self.columns), dtype=np.float32)
        for language, product in self.sums.items():
            total += product * np.float32(weights[language])
        return 2.0 * total

    @staticmethod
    def _apply_language_weights(
        weights: Mapping[str, float], language_weights: Mapping[str, float]
    ) -> dict[str, float]:
        """Multiply per-language weights and renormalise, preserving the total."""
        scaled = {
            name: value * float(language_weights.get(name, 1.0))
            for name, value in weights.items()
        }
        before = sum(weights.values())
        after = sum(scaled.values())
        if after > 0 and before > 0:
            scaled = {name: value * before / after for name, value in scaled.items()}
        return scaled

    def language_mass_with(
        self,
        language_weights: Mapping[str, float],
        *,
        balance: bool = True,
        cap_fraction: float | None = None,
    ) -> dict[str, float]:
        """Per-language mass shares that ``language_weights`` would produce.

        The brief's guard is on the *share*, so a reweighting has to be checked
        against it before the Hessian is built -- otherwise a rebalance can push
        a language past the bound and only surface as a failed run much later.
        """
        weights = self._row_weights(balance=balance, cap_fraction=cap_fraction)
        scaled = self._apply_language_weights(weights, language_weights)
        total = sum(scaled.values())
        return {name: value / total for name, value in scaled.items()}

    def _effective_rows(
        self, *, balance: bool = True, cap_fraction: float | None = None
    ) -> dict[str, float]:
        """Per-language row count the Hessian effectively sees.

        Both modes are one family -- a common row threshold ``T`` with
        ``rows_i -> min(rows_i, T)``:

        * ``balance`` picks ``T = min(rows)``, giving every language an equal
          share;
        * ``cap_fraction`` bisects for the *largest* ``T`` whose resulting
          maximum share is still within the cap, so the most calibration data is
          kept subject to the guard.
        """
        if cap_fraction is not None:
            if not 0 < cap_fraction <= 1:
                raise ValueError("cap_fraction must be in (0, 1]")
            counts = np.array([self.rows[name] for name in sorted(self.rows)], dtype=np.float64)
            floor = 1.0 / counts.size
            if cap_fraction < floor - 1e-12:
                raise ValueError(
                    f"cap_fraction {cap_fraction} is below the {floor:.4f} floor of "
                    f"{counts.size} languages"
                )

            def share(threshold: float) -> float:
                kept = np.minimum(counts, threshold)
                return float(kept.max() / kept.sum())

            low, high = 0.0, float(counts.max())
            for _ in range(200):
                middle = 0.5 * (low + high)
                if share(middle) <= cap_fraction:
                    low = middle
                else:
                    high = middle
            threshold = low
        elif balance:
            threshold = float(min(self.rows.values()))
        else:
            return {name: float(count) for name, count in self.rows.items()}
        return {name: float(min(count, threshold)) for name, count in self.rows.items()}

    def _row_weights(
        self, *, balance: bool = True, cap_fraction: float | None = None
    ) -> dict[str, float]:
        """Multiplier applied to each language's accumulated ``X^T X``.

        The stored products are already sums, so an unweighted Hessian needs a
        multiplier of **1**, not the row count.
        """
        if cap_fraction is None and not balance:
            return {name: 1.0 for name in self.rows}
        effective = self._effective_rows(balance=balance, cap_fraction=cap_fraction)
        return {name: effective[name] / self.rows[name] for name in self.rows}

    def language_mass(
        self, *, balance: bool = True, cap_fraction: float | None = None
    ) -> dict[str, float]:
        """Share of the Hessian's effective rows per language (the guard's check).

        Mirrors :meth:`hessian` exactly, so the reported shares are the weights
        actually applied to the accumulated products.
        """
        rows = self._effective_rows(balance=balance, cap_fraction=cap_fraction)
        total = sum(rows.values())
        return {name: value / total for name, value in rows.items()}

    def save(self, path) -> None:
        """Write the accumulator as an ``.npz`` (one array per language)."""
        payload = {"rows": np.array(sorted(self.rows.items()), dtype=object)}
        for language, product in self.sums.items():
            payload["sum:" + (language or "_")] = product
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path, *, columns: int | None = None) -> "HessianAccumulator":
        with np.load(path, allow_pickle=True) as data:
            rows = {str(name): int(count) for name, count in data["rows"]}
            sums = {
                key[len("sum:") :].replace("_", ""): data[key]
                for key in data.files
                if key.startswith("sum:")
            }
            if columns is None:
                columns = next(iter(sums.values())).shape[0]
            return cls(columns=columns, sums=sums, rows=rows)


def hessian_from_activations(activations: Iterable[np.ndarray], columns: int) -> np.ndarray:
    """Convenience: ``H = 2 X^T X`` over a stream of activation batches."""
    accumulator = HessianAccumulator(columns=columns)
    for batch in activations:
        accumulator.add(batch)
    return accumulator.hessian(balance=False)
