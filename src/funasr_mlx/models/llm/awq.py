"""AWQ (activation-aware weight quantization) for the Qwen3 LLM-ASR decoder.

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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

try:  # mlx.nn has no no_grad context in 0.32; weights are only ever assigned,
    # never differentiated, so the context is a no-op guard upstream added for
    # its training runtime.  Define a local fallback rather than import it.
    from contextlib import nullcontext

    def no_grad():
        return nullcontext()
except Exception:  # pragma: no cover
    raise


@dataclass(frozen=True)
class AwqConfig:
    """Search hyper-parameters (mlx-lm defaults)."""

    n_grid: int = 20
    max_shrink: float = 0.5
    #: Subsampled rows for the clip search (upstream ``n_sample_token``).
    clip_rows: int = 512
    #: Batch width for the row dimension during the clip search.
    clip_row_batch: int = 64
    #: Batch width over output channels during the clip search.
    clip_out_batch: int = 256
    #: Scale grid over ``x_max ** ratio``; upstream range.
    max_ratio: float = 1.0


@dataclass
class ScaleSite:
    """One scale search site: a norm (or Linear) feeding several Linears.

    Mirrors mlx-lm's ``ScaleConfig`` for the Qwen3 block layout:

    * ``prev`` -- the module whose OUTPUT is scaled (an RMSNorm for qwen3);
      its weight is divided by the scales.
    * ``layers`` -- the successor Linear weights are multiplied by the scales.
    * ``block`` -- optional sub-block to inspect (``self_attn``), because the
      loss must be measured on the attention output, not one Linear alone.
    """

    prev: str
    layers: tuple[str, ...]
    block: str | None = None


#: Qwen3 decoder block: identical topology to mlx-lm's llama config.
QWEN3_SCALE_SITES = (
    ScaleSite(block="self_attn", prev="input_layernorm",
              layers=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
    ScaleSite(prev="mlp.up_proj", layers=("mlp.down_proj",)),
    ScaleSite(block="mlp", prev="post_attention_layernorm",
              layers=("mlp.gate_proj", "mlp.up_proj")),
)

#: q/k projections are excluded from clipping (attention couples their error);
#: o_proj/v_proj remain clip-eligible per upstream's no_clip=["q_proj","k_proj"].
QWEN3_NO_CLIP = ("q_proj", "k_proj")


def mse(x, y):
    return ((x - y).astype(__import__("mlx").core.float32)) ** 2


def _linears(block):
    """The block's Linear modules as ``{path: module}`` (nested paths)."""
    import mlx.nn as nn

    return {
        path: module
        for path, module in block.named_modules()
        if isinstance(module, nn.Linear)
    }


class AwqBlockQuantizer:
    """Scale+clip search for one transformer block, on captured input."""

    def __init__(self, block, config: AwqConfig | None = None) -> None:
        import mlx.nn as nn

        self.block = block
        self.config = config or AwqConfig()
        self._nn = nn

    # -- activation capture -------------------------------------------------- #
    def _submodule(self, key: str):
        module = self.block
        for part in key.split("."):
            module = module[part]
        return module

    def capture(self, run: Callable[[], Any]) -> None:
        """Run ``run()`` with input hooks on this block's Linears.

        ``run`` is a callable that performs the model's real prefill for one
        utterance; the block must be part of the live model.  Each Linear's
        input rows are appended to ``input_feat`` on the module (mlx-lm's
        convention), in float32, and the block's own input is recorded on
        ``self.block_input`` for the never-worse reference.
        """
        import mlx.core as mx
        from mlx.utils import tree_map

        orig = self.block.leaf_modules()

        def catcher_factory(module):
            if not isinstance(module, (self._nn.Linear, self._nn.RMSNorm)):
                return module
            record_norm_input = isinstance(module, self._nn.RMSNorm)

            class Catcher(self._nn.Module):
                def __call__(self, x, *args, **kwargs):
                    if record_norm_input:
                        # The block's own input is what enters its first norm.
                        rows = x.reshape(-1, x.shape[-1]).astype(mx.float32)
                        if hasattr(module, "block_input"):
                            module.block_input = mx.concatenate(
                                [module.block_input, rows], axis=0
                            )
                        else:
                            module.block_input = rows
                    else:
                        rows = x.reshape(-1, x.shape[-1]).astype(mx.float32)
                        if hasattr(module, "input_feat"):
                            module.input_feat = mx.concatenate([module.input_feat, rows], axis=0)
                        else:
                            module.input_feat = rows
                    return module(x, *args, **kwargs)

            return Catcher()

        capture_leaves = tree_map(
            catcher_factory, orig, is_leaf=self._nn.Module.is_module
        )
        self.block.update_modules(capture_leaves)
        try:
            run()
        finally:
            self.block.update_modules(orig)
        mx.eval(self.block.parameters())

    # -- scale search --------------------------------------------------------- #
    def _run_block(self, block, x, mask=None):
        """Run ``block`` on ``x`` [1, rows, hidden] in row batches."""
        import mlx.core as mx

        y = []
        for start in range(0, x.shape[1], 32):
            chunk = x[:, start : start + 32]
            out = block(chunk) if mask is None else block(chunk, mask)
            y.append(out)
            mx.eval(y)
        return mx.concatenate(y, axis=1)

    def search_scale(self, site: ScaleSite, quantize_func) -> Any:
        """Grid-search channel scales for one site; returns the best scales.

        The reference output is the block (or sub-block) run on the captured
        input of the site's FIRST layer; each candidate scales that input's
        channels via the successor weights only (equivalent math, see
        upstream), quantize-dequantizes them, and the MSE of the sub-block's
        output picks the winner.
        """
        import mlx.core as mx
        from mlx.utils import tree_flatten

        layers = [self._submodule(name) for name in site.layers]
        # mlx-lm's ``module2inspect``: the sub-block when the site names one
        # (attention), else the FIRST layer itself -- never the whole block,
        # because the whole block's input is the residual stream, not the
        # successor Linear's input.
        inner = self._submodule(site.block) if site.block else layers[0]
        x = layers[0].input_feat  # [rows, hidden]
        x = x[None]  # [1, rows, hidden] -- block call convention

        with no_grad():
            out = self._run_block(inner, x)
            x_max = x.abs().mean(axis=(0, 1)).reshape(-1)
            weights = tree_flatten(self.block.parameters())
            best_error, best_scales = float("inf"), None
            for step in range(self.config.n_grid):
                ratio = step * self.config.max_ratio / self.config.n_grid
                scales = mx.maximum(x_max**ratio, 1e-4)
                scales = scales / mx.sqrt(scales.max() * scales.min())
                for layer in layers:
                    if hasattr(layer, "weight"):
                        layer.weight = quantize_func(layer.weight * scales) / scales
                out_q = self._run_block(inner, x)
                loss = mse(out, out_q).sum() / out.size
                mx.eval(loss)
                value = float(loss.item())
                if value < best_error:
                    best_error, best_scales = value, scales
                # restore
                self.block.load_weights(weights)
            if best_scales is None:
                raise RuntimeError("scale search found no candidate")
            mx.eval(best_scales)
            return best_scales

    # -- clip search ---------------------------------------------------------- #
    def search_clip(self, name: str, module, quantize_func) -> Any:
        """Per-group shrink search for one Linear; returns the clipped weight."""
        import mlx.core as mx

        cfg = self.config
        x = module.input_feat
        stride = max((x.shape[0] + cfg.clip_rows - 1) // cfg.clip_rows, 1)
        x = x[::stride]
        group_size = module.weight.shape[1] // max(
            module.weight.shape[1] // 64, 1
        )  # caller passes the packing group via quantize_func's closure
        w = module.weight
        xg = x.reshape(x.shape[0], -1, 64)  # groups of 64 (packing default)
        w_shape = w.shape
        w_all = w.reshape(-1, w.shape[-1])
        best_max_all = []
        for bstart in range(0, w_all.shape[0], cfg.clip_out_batch):
            wrows = w_all[bstart : bstart + cfg.clip_out_batch]
            group_shape = (wrows.shape[0], wrows.shape[-1] // 64)
            best_err = mx.full(group_shape, float("inf"))
            best_max = mx.zeros((*group_shape, 1), dtype=mx.float32)
            wg = wrows.reshape(wrows.shape[0], -1, 64)
            init_max = mx.maximum(wg.abs().max(axis=-1, keepdims=True), 1e-8)
            ref = mx.einsum("bdg,odg->bod", xg, wg.astype(mx.float32))
            for i in range(int(cfg.max_shrink * cfg.n_grid)):
                p = 1 - i / cfg.n_grid
                wmax = p * init_max
                clipped = mx.clip(wg.astype(mx.float32), -wmax, wmax)
                wq = quantize_func(clipped).astype(mx.float32)
                out_q = mx.einsum("bdg,odg->bod", xg, wq.reshape(wq.shape[0], -1, 64))
                loss = mse(ref, out_q).sum(axis=0).mean(axis=0)
                better = loss < best_err
                best_err = mx.where(better, loss, best_err)
                best_max = mx.where(better[..., None], wmax, best_max)
                mx.eval(best_err, best_max)
            best_max_all.append(best_max)
        best_max = mx.concatenate(best_max_all, axis=0)  # [out, in/group, 1]
        best_max = best_max[..., 0]  # [out, in/group]
        w32 = module.weight.astype(mx.float32)
        groups = w32.reshape(w32.shape[0], -1, 64)
        clipped = mx.clip(groups, -best_max[..., None], best_max[..., None])
        clipped = clipped.reshape(w32.shape)
        mx.eval(clipped)
        return clipped

    # -- folding --------------------------------------------------------------- #
    def apply_scale(self, site: ScaleSite, scales) -> None:
        """Fold the chosen scales into the site's modules (bf16 stays bf16).

        Two fold shapes, exactly as mlx-lm: a Linear predecessor divides its
        OUTPUT-channel rows (``weight / scales[:, None]`` -- the scales index
        the successor's input channels, which are the predecessor's outputs),
        while a norm predecessor divides elementwise (its weight is a vector
        over the same channels it normalises).
        """
        prev = self._submodule(site.prev)
        for layer_name in site.layers:
            layer = self._submodule(layer_name)
            layer.weight = (layer.weight * scales).astype(layer.weight.dtype)
        if isinstance(prev, self._nn.Linear):
            prev.weight = (prev.weight / scales[:, None].astype(prev.weight.dtype)).astype(
                prev.weight.dtype
            )
        elif hasattr(prev, "weight") and prev.weight is not None:
            prev.weight = (prev.weight / scales.astype(prev.weight.dtype)).astype(
                prev.weight.dtype
            )
        # keep the captured input consistent with the folded weights
        for layer_name in site.layers:
            layer = self._submodule(layer_name)
            if hasattr(layer, "input_feat"):
                layer.input_feat = layer.input_feat / scales

    # -- orchestration ---------------------------------------------------------- #
    def optimize(self, quantize_func, *, do_clip: bool = True) -> dict:
        """Run the full scale+clip pass on this block. Returns a small report."""
        import mlx.core as mx
        from mlx.utils import tree_flatten

        # reference loss: plain RTN quantization of the untouched block
        weights = tree_flatten(self.block.parameters())
        # reference output for the never-worse check uses the block's OWN
        # input: what enters its first norm (the residual stream).
        x = self._submodule("input_layernorm").block_input[None]
        linears = _linears(self.block)
        with no_grad():
            out = self._run_block(self.block, x)
            for path, module in linears.items():
                if isinstance(module, self._nn.Linear):
                    module.weight = quantize_func(module.weight).astype(module.weight.dtype)
            out_q = self._run_block(self.block, x)
            before = float((mse(out, out_q).sum() / out.size).item())
            self.block.load_weights(weights)

        for site in QWEN3_SCALE_SITES:
            scales = self.search_scale(site, quantize_func)
            self.apply_scale(site, scales)

        if do_clip:
            for path, module in _linears(self.block).items():
                if isinstance(module, self._nn.Linear) and not any(
                    key in path for key in QWEN3_NO_CLIP
                ):
                    module.weight = self.search_clip(path, module, quantize_func).astype(
                        module.weight.dtype
                    )

        with no_grad():
            out_q = self._run_block(self.block, x)
            for path, module in _linears(self.block).items():
                if isinstance(module, self._nn.Linear):
                    module.weight = quantize_func(module.weight).astype(module.weight.dtype)
            after_block = self._run_block(self.block, x)
            after = float((mse(out, after_block).sum() / out.size).item())
            self.block.load_weights(weights)

        if after > before:
            self.block.load_weights(weights)
            return {"before": before, "after": after, "fallback": True}
        return {"before": before, "after": after, "fallback": False}
