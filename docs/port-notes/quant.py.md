# quant.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
W4A16 weight quantization for the LLM-ASR decoder stacks.

Scheme: MLX **affine group quantization** (``mx.quantize``), the same layout
``mlx-lm`` uses -- per-group ``scales``/``biases`` carried in the checkpoint
dtype and the weights packed into ``uint32`` words.  ``nn.quantize`` turns each
selected ``nn.Linear`` into ``nn.QuantizedLinear``, whose ``weight`` /
``scales`` / ``biases`` parameters are exactly the three tensors the converter
writes, so a quantized checkpoint strict-loads through the ordinary
``pipeline.load_model`` path with no special-casing anywhere else.

The whole configuration lives in the native ``config.json``::

    "quantization": {"bits": 4, "group_size": 64, "mode": "affine",
                     "targets": ["thinker.model.layers."]}

``targets`` are module-path prefixes.  Only ``nn.Linear`` modules under one of
them are quantized, so the token embeddings, the LM head, every convolution and
every norm keep the checkpoint dtype.  The converter in
:mod:`funasr_mlx.checkpoints` applies the *same* rule to tensor names
(``<module path>.weight``); because the two rules must agree for the strict
loader to find ``.scales``/``.biases``, any divergence fails loudly at load
time instead of silently mis-binding weights.

Why decoder-only by default: decode at bs=1 streams the decoder weights once
per step, so 4-bit decoder weights cut the step's dominant traffic ~3.5x.  The
audio tower runs once per utterance (prefill side), so quantizing it buys
memory but not decode time; it is offered as an extra target prefix for
families whose memory budget needs it.

```
