# fun_asr_nano.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
FunASRNano: audio encoder + adaptor + Qwen3 LLM, MLX port.

Port of ``funasr/models/fun_asr_nano/model.py`` (inference path):

* ``audio_encoder``: ``SenseVoiceEncoderSmall`` -- REUSED unchanged from
  :mod:`funasr_mlx.models.sensevoice` (this exact encoder is already
  FP32-parity-verified for the SenseVoice model; Fun-ASR-Nano's config
  builds the same module with ``tp_blocks=20``).
* ``audio_adaptor``: the ``Transformer`` adaptor from
  ``funasr/models/llm_asr/adaptor.py`` -- a k-frame stacking projection
  (here k=1) followed by standard pre/post-norm encoder layers.
* ``llm``: ``Qwen3ForCausalLM`` from :mod:`funasr_mlx.models.llm`
  (mlx-lm port, HF-validated on Qwen3-0.6B).
* ``ctc_decoder``: timestamp/forced-alignment head -- not part of the
  transcription path and not ported (raising explicitly).

The inference protocol mirrors ``inference_prepare``: build the chat
template, replace the ``<|startofspeech|>..<|endofspeech|>`` span with the
adaptor embeddings via ``input_embeddings``, then decode step-by-step with
the KV cache.

```
