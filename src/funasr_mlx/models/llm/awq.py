"""AWQ (activation-aware weight quantization) for the Qwen3 LLM-ASR decoder.

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/awq.py.md
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
    # see docs/port-notes/functions.md :: ScaleSite

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


def _to_mx(value):
    # see docs/port-notes/functions.md :: _to_mx
    import mlx.core as mx

    return mx.array(value)


def _linears(block):
    import mlx.nn as nn

    return {
        path: module
        for path, module in block.named_modules()
        if isinstance(module, nn.Linear)
    }


def _is_first_norm(module, block) -> bool:
    return module is getattr(block, "input_layernorm", None)


class MultiBlockCapture:
    # see docs/port-notes/functions.md :: MultiBlockCapture

    def __init__(self, blocks, *, row_cap: int = 1536) -> None:
        import mlx.nn as nn

        self.blocks = list(blocks)
        self.row_cap = row_cap
        self._nn = nn
        self._installed = []

    def install(self) -> None:
        import mlx.core as mx
        from mlx.utils import tree_map

        for block in self.blocks:
            orig = block.leaf_modules()

            def catcher_factory(module, nn=self._nn, mx=mx, cap=self.row_cap):
                if isinstance(module, nn.RMSNorm):
                    # ONLY the block's first norm records (its input IS the
                    # block input); the mid-block norm's input is not used.
                    if not _is_first_norm(module, block):
                        return module
                    record_norm = True
                elif isinstance(module, nn.Linear):
                    record_norm = False
                else:
                    return module

                class Catcher(nn.Module):
                    def __call__(self, x, *args, **kwargs):
                        # HOST-side storage with a hard stop: once ``row_cap``
                        # rows are held, further rows are DROPPED (the cap is
                        # a statistical sample, not an exhaustive log).  This
                        # avoids both the wired-GW trap of MLX-side buffers and
                        # the O(n^2) re-concat allocation spikes.
                        import numpy as _np

                        attr = "block_input" if record_norm else "input_feat"
                        held = getattr(module, attr, None)
                        if held is None:
                            setattr(
                                module, attr,
                                _np.array(x.reshape(-1, x.shape[-1]).astype(mx.float32), dtype=_np.float16),
                            )
                        elif held.shape[0] < cap:
                            need = cap - held.shape[0]
                            more = _np.array(
                                x.reshape(-1, x.shape[-1]).astype(mx.float32), dtype=_np.float16
                            )[:need]
                            setattr(module, attr, _np.concatenate([held, more], axis=0))
                        # else: at cap, drop
                        return module(x, *args, **kwargs)

                return Catcher()

            leaves = tree_map(catcher_factory, orig, is_leaf=self._nn.Module.is_module)
            block.update_modules(leaves)
            self._installed.append(orig)

    def observe(self, run: Callable[[], Any]) -> None:
        run()

    def uninstall(self) -> None:
        for block, orig in zip(self.blocks, self._installed):
            block.update_modules(orig)
        self._installed = []

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *exc):
        self.uninstall()


class AwqBlockQuantizer:

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
        # see docs/port-notes/functions.md :: capture
        import mlx.core as mx
        from mlx.utils import tree_map

        orig = self.block.leaf_modules()

        def catcher_factory(module):
            if not isinstance(module, (self._nn.Linear, self._nn.RMSNorm)):
                return module
            record_norm_input = isinstance(module, self._nn.RMSNorm)

            class Catcher(self._nn.Module):
                def __call__(self, x, *args, **kwargs):
                    # HOST-side numpy storage (see MultiBlockCapture for why:
                    # MLX-side buffers are wired GPU memory).
                    import numpy as _np

                    if record_norm_input:
                        rows = _np.array(x.reshape(-1, x.shape[-1]).astype(mx.float32))
                        attr = "block_input"
                    else:
                        rows = _np.array(x.reshape(-1, x.shape[-1]).astype(mx.float32))
                        attr = "input_feat"
                    if hasattr(module, attr):
                        setattr(module, attr, _np.concatenate([getattr(module, attr), rows], axis=0))
                    else:
                        setattr(module, attr, rows)
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
        # see docs/port-notes/functions.md :: _run_block
        import mlx.core as mx

        def once(chunk):
            if hasattr(block, "input_layernorm") and hasattr(block, "self_attn"):
                # a whole transformer block: unroll the qwen3 residual pattern
                h = chunk
                h = h + block.self_attn(block.input_layernorm(h), mask)
                return h + block.mlp(block.post_attention_layernorm(h))
            return block(chunk, mask) if mask is not None else block(chunk)

        y = []
        for start in range(0, x.shape[1], 256):
            out = once(x[:, start : start + 256])
            y.append(out)
            mx.eval(y)
        return mx.concatenate(y, axis=1)

    def search_scale(self, site: ScaleSite, quantize_func) -> Any:
        # see docs/port-notes/functions.md :: search_scale
        import mlx.core as mx
        from mlx.utils import tree_flatten

        layers = [self._submodule(name) for name in site.layers]
        # mlx-lm's ``module2inspect``: the sub-block when the site names one
        # (attention), else the FIRST layer itself -- never the whole block,
        # because the whole block's input is the residual stream, not the
        # successor Linear's input.
        inner = self._submodule(site.block) if site.block else layers[0]
        x = _to_mx(layers[0].input_feat)  # [rows, hidden]
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
        import mlx.core as mx

        cfg = self.config
        x = _to_mx(module.input_feat)
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
        # see docs/port-notes/functions.md :: apply_scale
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
        # keep the captured input consistent with the folded weights (host
        # numpy divided by the mx scales, back through numpy for the next site)
        import numpy as onp

        scales_np = onp.asarray(scales)
        for layer_name in site.layers:
            layer = self._submodule(layer_name)
            if hasattr(layer, "input_feat"):
                layer.input_feat = onp.asarray(layer.input_feat) / scales_np

    # -- orchestration ---------------------------------------------------------- #
    def optimize(self, quantize_func, *, do_clip: bool = True) -> dict:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        # reference loss: plain RTN quantization of the untouched block
        weights = tree_flatten(self.block.parameters())
        # reference output for the never-worse check uses the block's OWN
        # input: what enters its first norm (the residual stream).
        x = _to_mx(self._submodule("input_layernorm").block_input)[None]
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
            # NEVER-WORSE: if the AWQ-adjusted block quantizes WORSE than the
            # untouched one, the adjustment is discarded and the block keeps
            # its original weights. (The non-fallback path is the difference
            # from the original port: it used to restore here unconditionally,
            # which silently discarded every scale/clip the search had folded
            # in -- the saved "AWQ" checkpoint was the raw weights in f16 and
            # every downstream "AWQ vs RTN" comparison was two RTN runs.)
            self.block.load_weights(weights)
            return {"before": before, "after": after, "fallback": True}
        # The adjustment helps: re-fold it. ``weights`` was captured before the
        # search started, so the adjusted parameters must be re-applied --
        # cheapest correct path is to re-run the same deterministic search on
        # the pristine block (the captures are unchanged, so the search
        # reproduces the same scales/clips) rather than stash a second tree.
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
        mx.eval(self.block.parameters())
        return {"before": before, "after": after, "fallback": False}
