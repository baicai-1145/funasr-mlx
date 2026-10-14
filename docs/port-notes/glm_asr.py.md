# glm_asr.py — port notes

## module

```
GLM-ASR-Nano audio encoder + projector, ported from the checkpoint's code.

Structure verified from ``model.safetensors`` (zai-org/GLM-ASR-Nano-2512,
MIT): ``conv1`` (1280, 128, 3) stride 1 + ``conv2`` (1280, 1280, 3) stride 2
=> a single 2x downsample; 32 pre-norm blocks with Llama-style names
(``input_layernorm``/``post_attention_layernorm``), attention q/k/v/o where
q, v, o carry biases and k does not, ``partial_rotary_factor`` 0.5; final
``norm``; projector ``linear_1`` 1280->4096 + gelu + ``linear_2`` 4096->2048.

Text side is a Llama-shaped decoder (see :mod:`.llm.llama`).

```

## module

```
GLM-ASR-Nano audio encoder + projector, ported from the checkpoint's code.

Structure verified from ``model.safetensors`` (zai-org/GLM-ASR-Nano-2512,
MIT): ``conv1`` (1280, 128, 3) stride 1 + ``conv2`` (1280, 1280, 3) stride 2
=> a single 2x downsample; 32 pre-norm blocks with Llama-style names
(``input_layernorm``/``post_attention_layernorm``), attention q/k/v/o where
q, v, o carry biases and k does not, ``partial_rotary_factor`` 0.5; final
``norm``; projector ``linear_1`` 1280->4096 + gelu + ``linear_2`` 4096->2048.

Text side is a Llama-shaped decoder (see :mod:`.llm.llama`).

```

## module

```
GLM-ASR-Nano audio encoder + projector, ported from the checkpoint's code.

Structure verified from ``model.safetensors`` (zai-org/GLM-ASR-Nano-2512,
MIT): ``conv1`` (1280, 128, 3) stride 1 + ``conv2`` (1280, 1280, 3) stride 2
=> a single 2x downsample; 32 pre-norm blocks with Llama-style names
(``input_layernorm``/``post_attention_layernorm``), attention q/k/v/o where
q, v, o carry biases and k does not, ``partial_rotary_factor`` 0.5; final
``norm``; projector ``linear_1`` 1280->4096 + gelu + ``linear_2`` 4096->2048.

Text side is a Llama-shaped decoder (see :mod:`.llm.llama`).

```
