# gptq.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
GPTQ (Hessian-based error-compensating PTQ) for the LLM-ASR weight matrices.

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

```
