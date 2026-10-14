# pseudo_streaming.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
Pseudo-streaming layer for ALL model families.

One protocol, every model: audio arrives in a growing buffer; each
``push()`` transcribes ``buffer[0:current]`` from scratch (cheap: classic
models are single-pass; LLM towers are bounded), ``stream()`` yields the
transient best-effort text. The final ``close()`` output is BIT-IDENTICAL
to offline transcription of the full audio (same computation by
construction).

Family support matrix:
- sensevoice / paraformer / whisper: single forward on the prefix; the
  text just stops early (no premature punctuation machinery -- non-AR
  or fixed-vocab decoding does not invent sentence endings).
- qwen3-asr / glm-asr-nano / moss / fun-asr-nano: re-run tower+prefill+
  greedy per push (the tail-roll behavior measured at 94.6% char survival,
  .tmp/opt/LOG.md).
- ct-punc / campplus / emotion2vec / fsmn-vad: not audio-transcription
  models; excluded by construction (Transcriber raises).

This module is display-layer glue; it adds no numeric behavior of its own.

```
