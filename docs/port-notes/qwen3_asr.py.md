# qwen3_asr.py — port notes

## module

```
Qwen3-ASR audio encoder + thinker, ported from the qwen-asr package.

Source: qwen-asr 0.0.6 ``qwen_asr/core/transformers_backend/
modeling_qwen3_asr.py`` (Apache 2.0, Qwen team), inference subset.

Architecture (``thinker_config.audio_config`` of Qwen3-ASR-0.6B):
``conv2d1/2/3`` (stride 2 + GELU, 8x total downsample) -> ``conv_out``
-> sinusoidal positions -> 18 pre-norm encoder layers (LayerNorm, GELU
FFN, windowed varlen attention) -> ``ln_post`` -> ``proj1``/GELU/
``proj2`` (d_model 896 -> output_dim 1024, the Qwen3-0.6B hidden size).

The reference runs attention as flash-varlen over per-chunk windows of
``n_window * 2`` mel frames; here the same windowing is expressed as a
block-diagonal mask over one batched attention, which is numerically
identical (each window only ever attends within itself).

```

## module

```
Qwen3-ASR audio encoder + thinker, ported from the qwen-asr package.

Source: qwen-asr 0.0.6 ``qwen_asr/core/transformers_backend/
modeling_qwen3_asr.py`` (Apache 2.0, Qwen team), inference subset.

Architecture (``thinker_config.audio_config`` of Qwen3-ASR-0.6B):
``conv2d1/2/3`` (stride 2 + GELU, 8x total downsample) -> ``conv_out``
-> sinusoidal positions -> 18 pre-norm encoder layers (LayerNorm, GELU
FFN, windowed varlen attention) -> ``ln_post`` -> ``proj1``/GELU/
``proj2`` (d_model 896 -> output_dim 1024, the Qwen3-0.6B hidden size).

The reference runs attention as flash-varlen over per-chunk windows of
``n_window * 2`` mel frames; here the same windowing is expressed as a
block-diagonal mask over one batched attention, which is numerically
identical (each window only ever attends within itself).

```

## module

```
Qwen3-ASR audio encoder + thinker, ported from the qwen-asr package.

Source: qwen-asr 0.0.6 ``qwen_asr/core/transformers_backend/
modeling_qwen3_asr.py`` (Apache 2.0, Qwen team), inference subset.

Architecture (``thinker_config.audio_config`` of Qwen3-ASR-0.6B):
``conv2d1/2/3`` (stride 2 + GELU, 8x total downsample) -> ``conv_out``
-> sinusoidal positions -> 18 pre-norm encoder layers (LayerNorm, GELU
FFN, windowed varlen attention) -> ``ln_post`` -> ``proj1``/GELU/
``proj2`` (d_model 896 -> output_dim 1024, the Qwen3-0.6B hidden size).

The reference runs attention as flash-varlen over per-chunk windows of
``n_window * 2`` mel frames; here the same windowing is expressed as a
block-diagonal mask over one batched attention, which is numerically
identical (each window only ever attends within itself).

```
