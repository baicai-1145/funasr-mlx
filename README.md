# funasr-mlx

Native [MLX](https://github.com/ml-explore/mlx) inference ports of the
FunASR model zoo for Apple Silicon — memory-bounded, numerically verified
against the official references, with no PyTorch/runtime dependency at
inference time.

All 14 checkpoints covering 11 architectures from the FunASR README model
zoo are ported. Every port is validated layer-by-layer against its
official implementation (fp32, shared-input parity) and end-to-end on real
audio before it is marked done.

## Coverage & Verification

| Model | Family | Layer parity (fp32) | Real-audio e2e |
| --- | --- | --- | --- |
| SenseVoiceSmall | `sensevoice` | source-exact diffs | ✅ |
| Paraformer (offline) | `paraformer` | rel 1e-6 (GEMM noise) | ✅ CER ~0 |
| Paraformer streaming | `paraformer` | 4.0e-7 tail path | ⚠️ CER 0.156* |
| Whisper tiny | `whisper` | canonical full-forward | ✅ |
| Whisper large-v3 | `whisper` | enc 1.5e-4 / dec 2.3e-5 | ✅ RTF 0.058 |
| Whisper large-v3-turbo | `whisper` | enc 5.3e-4 / dec 2.6e-4 | ✅ RTF 0.061 |
| emotion2vec+large | `emotion2vec` | fairseq-exact CNN/pos | ✅ |
| ct-punc | `ct-punc` | conv-tap exact | ✅ |
| fsmn-vad | `fsmn-vad` | conv/cache exact | ✅ |
| CAM++ | `campplus` | source-exact | ✅ |
| Fun-ASR-Nano | llm | audio 4.6e-3 / LLM 3.0e-6 | ✅ 8/8 |
| Qwen3-ASR-0.6B | llm | audio 1.2e-5 / dec 1.4e-4 | ✅ 6/6 |
| GLM-ASR-Nano-2512 | llm | tower 2.0e-5 / llama 3.3e-6 | ✅ 7/7 |
| MOSS-Transcribe-Diarize | llm | enc 1.3e-3 / dec 4.7e-5 | ✅ 8/8 +timestamps+speakers |

\* Streaming Paraformer: single-layer diffs are 1e-6, but the chunked
streaming state machine amplifies fp32 noise until CIF fire positions
drift (MLX has no fp64). Use streaming for partial results and the
offline model for final transcripts.

The four LLM-based models (Fun-ASR-Nano, Qwen3-ASR, GLM-ASR-Nano,
MOSS-Transcribe-Diarize) are ported and numerically verified
(`src/funasr_mlx/models/` + `models/llm/`), with e2e validated through
bf16 scripts; their pipeline-factory registration is pending. The seven
speech families above run through the guarded CLI today.

## Local Use

Install on an Apple Silicon Mac with Python 3.11+:

```sh
python3 -m pip install -e .
funasr-mlx --help
```

Inspect a local checkpoint or extract bounded features:

```sh
funasr-mlx inspect /path/to/model.safetensors
funasr-mlx features recording.wav --cmvn none
```

Inference requires a local converted checkpoint directory and a matching
native JSON or YAML configuration:

```sh
funasr-mlx infer /path/to/checkpoint-dir --model sensevoice --audio recording.wav
funasr-mlx infer /path/to/checkpoint-dir --model ct-punc --text "hello world" --tokens /path/to/tokens.txt
```

## W4A16 quantization

4-bit weights with 16-bit activations for the LLM-ASR decoders, using MLX
affine group quantization (group size 64). At batch size 1 the decode step
streams the weights once per step, so reading 4x fewer bytes is the largest
remaining lever; measured on an M4 it cuts peak inference memory 41-59% and
decode time up to 2.7x across the five families.

Convert a native checkpoint, then load it exactly as an unquantized one:

```sh
python3 tools/convert_llm_checkpoints.py --family qwen3-asr-0.6b \
    --source /path/to/qwen3-asr-0.6b --destination /path/to/qwen3-asr-0.6b-w4a16 \
    --native-source --quantize tower
```

`--quantize` picks how much is quantized, and the choice is not cosmetic:

| scope | reaches | buys |
|---|---|---|
| `decoder` | decoder Linear layers | decode time |
| `tower` | + the audio tower | peak memory; the tower runs once per utterance, so no decode cost |
| `all` | + the tied token embedding | removes the per-step LM-head read, but only some families survive it |

The token embedding is the tied LM head, so quantizing it injects 4-bit error
directly into every step's logits. It is safe for `fun-asr-nano` (tokens fully
identical) and destroys the transcript for `qwen3-asr-0.6b` and `glm-asr-nano`,
so `all` is opt-in per family rather than a default. `lm_head`, position tables,
convolutions and norms are never quantized. Loading is strict: the config
records `quantized_tensors` and `quantization_skipped`, and a key that should
have been quantized but was not is an error, not a silent fallback.

Per-family scope, checkpoint sizes, measured speedups and peak-memory drops are
in `.tmp/t9/RESULTS.md`; the scheme and its failure modes are in
`.tmp/t9/LEDGER.md`.

## Memory Policy

The hard ceiling is 10 GB (decimal) for project processes. Numerical jobs
are serialized with a git-common-directory lock. Default job budget 2 GB,
project stop threshold 8 GB, minimum host reserve 3 GB. Bounded
input/shards, admission checks, MLX allocation/cache guidelines, an
external RSS watchdog and supervisor-liveness pipe work together. No
automatic downloads or simultaneous model loading.

## Development

Optional `test`, `reference` and `convert` dependency groups. Source
differential tests use a sibling FunASR checkout; unavailable references
skip. Guarded verification:

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m funasr_mlx.runtime \
  --root . --report .tmp/check.json -- \
  python3 -m pytest -q --tb=short --basetemp=.tmp/pytest-check
```

503 tests pass in the guarded suite. Real-checkpoint accuracy, quantized
variants, and latency/RTF figures are documented in `PROGRESS.md`; the W4A16
per-family numbers are in `.tmp/t9/RESULTS.md`.

## Attribution

Qwen3/Llama decoder layouts and the KV cache are vendored (trimmed) from
[mlx-lm](https://github.com/ml-explore/mlx-lm) (Apache 2.0, Copyright ©
2023-2024 Apple Inc.); source and license notes are in the module
docstrings. Model architectures follow their official references
(FunASR, OpenAI Whisper, Qwen, GLM, OpenMOSS).

## Status

Experimental: numerically verified ports, not a production service. No
fastest / lowest-memory / production-quality claim is made.
