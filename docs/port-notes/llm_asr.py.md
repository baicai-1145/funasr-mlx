# llm_asr.py — port notes

## module

```
LLM-ASR transcription drivers for the CLI / API / server layers.

One module, three consumers:
* ``funasr-mlx transcribe`` (CLI) -- single-shot, guarded child.
* ``funasr_mlx.transcribe()`` (Python API) -- resident worker with KV
  reset-reuse across calls (validated bit-stable on M4; the RTF numbers in
  the README come from this pattern).
* ``funasr_mlx serve`` (HTTP) -- wraps the same driver.

The per-family prompt/decode protocols are the acceptance-validated ones
(tools/gptq_acceptance.py); token-level behavior is identical.

```

## module

```
LLM-ASR transcription drivers for the CLI / API / server layers.

One module, three consumers:
* ``funasr-mlx transcribe`` (CLI) -- single-shot, guarded child.
* ``funasr_mlx.transcribe()`` (Python API) -- resident worker with KV
  reset-reuse across calls (validated bit-stable on M4; the RTF numbers in
  the README come from this pattern).
* ``funasr_mlx serve`` (HTTP) -- wraps the same driver.

The per-family prompt/decode protocols are the acceptance-validated ones
(tools/gptq_acceptance.py); token-level behavior is identical.

```

## module

```
LLM-ASR transcription drivers for the CLI / API / server layers.

One module, three consumers:
* ``funasr-mlx transcribe`` (CLI) -- single-shot, guarded child.
* ``funasr_mlx.transcribe()`` (Python API) -- resident worker with KV
  reset-reuse across calls (validated bit-stable on M4; the RTF numbers in
  the README come from this pattern).
* ``funasr_mlx serve`` (HTTP) -- wraps the same driver.

The per-family prompt/decode protocols are the acceptance-validated ones
(tools/gptq_acceptance.py); token-level behavior is identical.

```
