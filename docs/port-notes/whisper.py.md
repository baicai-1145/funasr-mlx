# whisper.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
Native MLX inference port of **canonical OpenAI Whisper** as used by FunASR.

Authoritative source
--------------------
``funasr/models/whisper/model.py`` (FunASR wrapper) does ``import whisper``
(line 11) and builds ``whisper.model.Whisper`` (line 60), i.e. the **external
``openai-whisper`` package**, not the vendored SenseVoice copy.  The encoder,
decoder, attention, residual block and cache semantics here are therefore ported
from the canonical ``whisper/model.py`` (OpenAI release v20250625, pinned at
``/Users/baicai1145/repos/funasr-mlx/.tmp/openai-whisper-model.py``).

The vendored ``funasr/models/sense_voice/whisper_lib/`` tree is a **different
encoder** and must not be used as the reference for the Whisper path.  The two
differ in ways that silently corrupt a canonical checkpoint:

* ``whisper_lib``'s ``conv1`` has ``stride=2``; canonical's has **no stride**.
  Canonical halves the time axis exactly once, so the input contract is
  ``2 * n_audio_ctx`` mel frames.
* ``whisper_lib``'s ``ResidualAttentionBlock`` scales the branches by
  ``attn_pct`` / ``1 - attn_pct`` with ``attn_pct = 0.0`` on cross-attention
  blocks, discarding self-attention in the decoder.  Canonical has no gates and
  orders the branches ``attn -> cross_attn -> mlp``.

``funasr/models/sense_voice/whisper_lib/audio.py`` remains the reference for the
log-mel front-end arithmetic only, since it is numerically the same STFT/Slaney
computation the canonical package performs.

There is **no** dependency on ``mlx_whisper`` / ``openai-whisper`` / any torch
package, and nothing is ever downloaded: the model is a pure
``mlx.core``/``mlx.nn`` module tree loaded from a converted safetensors file.

Scope and honesty boundary
--------------------------
The tokenizer, feature file loading, beam search, temperature fallback,
timestamp rules and the ``ct-punc`` post-processor are **not** implemented here.
What this module does expose is the raw material the lead needs to integrate
them:

* :func:`log_mel_spectrogram` -- waveform ``(B, T)`` -> log-mel ``(B, T', 80)``
* :func:`pad_or_trim` -- waveform -> exactly ``n_samples``
* :func:`greedy_decode` -- **token ids only**, never text
* :func:`detect_language` -- raw per-language-token log-probabilities, given the
  language token ids supplied by the caller

Layout conventions (documented, not accidental)
----------------------------------------------
* Mel features are **channels-last**: ``(B, T', n_mels)``.  The reference uses
  ``(B, n_mels, T')`` and a ``permute(0, 2, 1)`` after ``conv2``; MLX's Conv1d
  is already channels-last, so that permute disappears.
* MLX convolutions are channels-last, so ``encoder.conv{1,2}.weight`` is stored
  as ``(n_state, kernel, in_channels)`` and converted from torch's
  ``(n_state, in_channels, kernel)`` by :func:`convert_checkpoint_tensor`.
* The causal mask is registered with ``persistent=False`` in the reference, so it
  is regenerated here and is **not** part of the expected parameter set.
* ``encoder.positional_embedding`` *is* a persistent buffer in the reference, so
  it is materialised as a real array and round-trips through strict loading
  (its regenerated value is identical).

```
