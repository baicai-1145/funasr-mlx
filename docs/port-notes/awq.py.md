# awq.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
AWQ (activation-aware weight quantization) for the Qwen3 LLM-ASR decoder.

Ported from mlx-lm ``mlx_lm/quant/awq.py`` (Apache 2.0, Copyright © 2025 Apple
Inc.), which itself implements Lin et al., "AWQ: Activation-aware Weight
Quantization" (https://arxiv.org/abs/2306.00978).  The port keeps the
algorithm and adapts the plumbing:

* The calibration input is REAL AUDIO driven through the family's own prefill
  path (tower -> splice -> decoder), reusing the same ``build_prefill``
  builders the GPTQ calibrator uses -- not text tokens.  The activations the
  ASR decoder sees come from spliced audio embeddings, and a text-only
  Hessian/scale would describe a distribution the model never visits.
* The scale search runs per transformer block with the block's own captured
  input, exactly as upstream: ``scales = x_max ** ratio`` over a grid, each
  candidate folded in, the block quantized+dequantized, and the output MSE
  against the unquantized block output compared.  The best scales are then
  FOLDED into the neighbouring modules (RMSNorm weight divided, successor
  Linear weights multiplied), so the runtime layout is untouched.
* The clip search shrinks per-group weight ranges when that lowers the
  layer's output error (the outlier-channels case), also per upstream.
* ``no_clip`` keeps q/k projections untouched (their error couples through
  the attention bmm, so per-layer MSE is not a faithful loss there).
* After scale+clip, a block is re-checked: if quantizing the AWQ-adjusted
  block is WORSE than quantizing the untouched one, the original weights are
  restored (the never-worse guarantee).

Everything here produces ordinary bf16 module weights; the actual packing is
done afterwards by the existing converter (RTN or GPTQ per tensor), so the
checkpoint side of the pipeline is completely unchanged.

Memory contract: one transformer block plus its captured input is live at a
time; the captured activations for 48 utterances x 10 languages at hidden 1024
are chunk-frozen and concatenated once per block, then released.

```
