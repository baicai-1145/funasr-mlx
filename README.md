# FunASR MLX

Experimental native MLX inference ports for Apple Silicon, developed against the FunASR reference repository. Seven model implementations are integrated, but none is certified against an official checkpoint or real audio yet.

Whisper currently returns greedy token IDs, not decoded text, timestamps or beam-search results. Paraformer's CIF uses MLX CPU for correctness; its experimental Metal path is not validated.

## Verification Status

The merged main branch passes 350 tests: bounded runtime and conversion, exact-source frontend and model comparisons, strict synthetic checkpoint loading, and CLI inference for all seven families. These are small synthetic configurations, not production acceptance or performance measurements.

| Model | Integrated Verification |
| --- | --- |
| SenseVoiceSmall | SANM/source comparisons and native checkpoint/CLI regressions |
| Paraformer | Source attention/CIF comparisons; CPU CIF default; checkpoint/CLI regressions |
| ct-punc | Actual PyTorch convolution-tap comparisons and checkpoint/text CLI |
| fsmn-vad | Exact-source convolution/cache comparisons and checkpoint/audio CLI |
| CAM++ | Architecture, layout, strict checkpoint and embedding CLI regressions |
| emotion2vec+large | Official fairseq CNN/positional comparisons and synthetic classifier CLI |
| Whisper | Canonical OpenAI v20250625 full-forward comparisons, batch sizes 1 and 2 |

## Local Use

Install on an Apple Silicon Mac with Python 3.11 or newer:

```sh
python3 -m pip install -e .
funasr-mlx --help
```

Inspect a local checkpoint or extract bounded features without downloading a model:

```sh
funasr-mlx inspect /path/to/model.safetensors
funasr-mlx features recording.wav --cmvn none
```

Inference requires a local converted checkpoint directory and a matching native JSON or YAML configuration:

```sh
funasr-mlx infer /path/to/checkpoint-dir --model sensevoice --audio recording.wav
funasr-mlx infer /path/to/checkpoint-dir --model ct-punc --text "hello world" --tokens /path/to/tokens.txt
```

Family names: `sensevoice`, `paraformer`, `ct-punc`, `fsmn-vad`, `campplus`, `emotion2vec` and `whisper`. Emotion configuration must supply the exact checkpoint labels; unknown labels are never invented. Whisper requires explicit prompt and end-of-text token IDs. See `funasr-mlx infer --help`.

The converter and loader are experimental. Published checkpoints may require explicit prefix removal and architecture configuration. Successful synthetic loading is not proof that a published checkpoint has been accepted.

## Memory Policy

The hard ceiling is 10 GB (decimal) for project processes, not 10 GiB. Numerical jobs are serialized with a git-common-directory lock. Default job budget is 2 GB, the project stop threshold is 8 GB, and minimum available host reserve is 3 GB. Bounded input/shards, admission checks, MLX allocation/cache guidelines, an external RSS watchdog and supervisor-liveness pipe work together. Other applications remain outside the project's control. No automatic downloads or simultaneous model loading.

Audio bounds reject overlong input before reading it; they do not silently truncate. The ordinary limit is 30 seconds, while emotion is limited by its native max_samples (10 seconds at the default 16 kHz settings). Long-form inference is not implemented. LFR stacking width and temporal stride are separate; the VAD default stacks five frames with stride one.

Latest full-suite sampled RSS: 318,111,744 bytes for the test process and 1,115,897,856 bytes across project processes. This is test-run evidence, not a full-model memory bound.

## Development

Optional `test`, `reference` and `convert` dependency groups are available. Source differential tests use the sibling FunASR checkout and small pinned reference files under `.tmp/`; unavailable references are skipped.


Guarded verification:
```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m funasr_mlx.runtime --root . --report .tmp/check.json -- python3 -m pytest -q --tb=short --basetemp=.tmp/pytest-check
```

Real checkpoints, tokenizer/label assets, real-audio accuracy, quantization quality and measured latency/RTF/peak memory remain acceptance work. No fastest, lowest-memory or production-quality claim is made.
