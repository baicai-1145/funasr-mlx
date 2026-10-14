# moss_transcribe.py — port notes

## module

```
MOSS-Transcribe-Diarize: Whisper-Medium encoder + VQAdaptor + Qwen3-0.6B.

Reference: OpenMOSS-Team/MOSS-Transcribe-Diarize ``modeling_moss_transcribe_diarize.py``
(ships with the checkpoint; MIT). The audio tower is the **HF transformers**
``WhisperEncoder`` layout (NOT openai/whisper): conv1 stride 2 + conv2
stride 1 (one 2x downsample total), a LEARNED ``embed_positions`` table
(1500, 1024), 24 pre-norm blocks with ``self_attn_layer_norm`` /
``final_layer_norm`` (LayerNorm with bias), MHA with q/k/v/out biases,
fc1/fc2 GELU, no top-level norm. Text side reuses :mod:`.llm.qwen3`
(q_norm/k_norm, tie_word_embeddings).

```

## module

```
MOSS-Transcribe-Diarize: Whisper-Medium encoder + VQAdaptor + Qwen3-0.6B.

Reference: OpenMOSS-Team/MOSS-Transcribe-Diarize ``modeling_moss_transcribe_diarize.py``
(ships with the checkpoint; MIT). The audio tower is the **HF transformers**
``WhisperEncoder`` layout (NOT openai/whisper): conv1 stride 2 + conv2
stride 1 (one 2x downsample total), a LEARNED ``embed_positions`` table
(1500, 1024), 24 pre-norm blocks with ``self_attn_layer_norm`` /
``final_layer_norm`` (LayerNorm with bias), MHA with q/k/v/out biases,
fc1/fc2 GELU, no top-level norm. Text side reuses :mod:`.llm.qwen3`
(q_norm/k_norm, tie_word_embeddings).

```

## module

```
MOSS-Transcribe-Diarize: Whisper-Medium encoder + VQAdaptor + Qwen3-0.6B.

Reference: OpenMOSS-Team/MOSS-Transcribe-Diarize ``modeling_moss_transcribe_diarize.py``
(ships with the checkpoint; MIT). The audio tower is the **HF transformers**
``WhisperEncoder`` layout (NOT openai/whisper): conv1 stride 2 + conv2
stride 1 (one 2x downsample total), a LEARNED ``embed_positions`` table
(1500, 1024), 24 pre-norm blocks with ``self_attn_layer_norm`` /
``final_layer_norm`` (LayerNorm with bias), MHA with q/k/v/out biases,
fc1/fc2 GELU, no top-level norm. Text side reuses :mod:`.llm.qwen3`
(q_norm/k_norm, tie_word_embeddings).

```
