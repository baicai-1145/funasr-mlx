# qwen3_asr_thinker.py — port notes

## module

```
Qwen3-ASR thinker: audio tower + Qwen3 text decoder assembly (inference).

Model structure (``thinker.*`` of the official checkpoint):
``thinker.audio_tower`` (the conv encoder in :mod:`funasr_mlx.models.qwen3_asr`)
+ ``thinker.model`` / ``thinker.lm_head`` -- a Qwen3-0.6B-shaped decoder that
REUSES :class:`funasr_mlx.models.llm.Qwen3ForCausalLM` (the thinker's
text_config matches Qwen3-0.6B; rope/mask/qk-norm semantics are identical;
ASR never needs the MRoPE grid variant because all three position streams
carry the same ids for pure-audio turns).

Prompt (``get_generation_prompt`` of the qwen-asr package):
``<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n<|im_start|>assistant\n``
where N = audio-tower frames; the ``<|audio_pad|>`` span's embeddings are
replaced by the tower output. With ``language`` forced:
``...assistant\nlanguage {lang}<asr_text>``.

```

## module

```
Qwen3-ASR thinker: audio tower + Qwen3 text decoder assembly (inference).

Model structure (``thinker.*`` of the official checkpoint):
``thinker.audio_tower`` (the conv encoder in :mod:`funasr_mlx.models.qwen3_asr`)
+ ``thinker.model`` / ``thinker.lm_head`` -- a Qwen3-0.6B-shaped decoder that
REUSES :class:`funasr_mlx.models.llm.Qwen3ForCausalLM` (the thinker's
text_config matches Qwen3-0.6B; rope/mask/qk-norm semantics are identical;
ASR never needs the MRoPE grid variant because all three position streams
carry the same ids for pure-audio turns).

Prompt (``get_generation_prompt`` of the qwen-asr package):
``<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n<|im_start|>assistant\n``
where N = audio-tower frames; the ``<|audio_pad|>`` span's embeddings are
replaced by the tower output. With ``language`` forced:
``...assistant\nlanguage {lang}<asr_text>``.

```

## module

```
Qwen3-ASR thinker: audio tower + Qwen3 text decoder assembly (inference).

Model structure (``thinker.*`` of the official checkpoint):
``thinker.audio_tower`` (the conv encoder in :mod:`funasr_mlx.models.qwen3_asr`)
+ ``thinker.model`` / ``thinker.lm_head`` -- a Qwen3-0.6B-shaped decoder that
REUSES :class:`funasr_mlx.models.llm.Qwen3ForCausalLM` (the thinker's
text_config matches Qwen3-0.6B; rope/mask/qk-norm semantics are identical;
ASR never needs the MRoPE grid variant because all three position streams
carry the same ids for pure-audio turns).

Prompt (``get_generation_prompt`` of the qwen-asr package):
``<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n<|im_start|>assistant\n``
where N = audio-tower frames; the ``<|audio_pad|>`` span's embeddings are
replaced by the tower output. With ``language`` forced:
``...assistant\nlanguage {lang}<asr_text>``.

```
