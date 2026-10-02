# Development Progress

All seven native MLX prototypes are integrated into main. This is unfinished: no official checkpoint or real-audio acceptance has been completed.

## Boundaries

- Work directory: /Users/baicai1145/repos/funasr-mlx. FunASR reference code remains read-only.
- Hard project ceiling: 10,000,000,000 bytes; stop threshold 8 GB; default job budget 2 GB; host reserve 3 GB.
- Maximum two implementation agents; workers are static-only, and the lead serializes numerical verification under the canonical git-common-dir lock.
- MLX memory limits are guidelines. Bounded input/shards, admission and external RSS monitoring are required.
- No full model weights downloaded or loaded; all model numerical work uses tiny dimensions.

## Latest Verified Main

Fresh guarded complete suite: **350 passed**, zero failures and skips, in 1.43 seconds. Report: .tmp/final-main-memory.json; JUnit: .tmp/final-main.xml.

Sampled child peak RSS: 318,111,744 bytes. Sampled project peak RSS: 1,115,897,856 bytes. Lowest available host memory: 6,449,119,232 bytes. No memory-policy violation. These measurements apply to synthetic tests, not real models.

Core infrastructure has 64 focused tests. Numerical frontend evidence includes six exact torchaudio Kaldi cases, six FunASR LFR cases, and two torch-STFT/librosa Slaney cases. Real watchdog and supervisor-exit probes verified owned-child termination and cleanup; simulated tests cover denial, timeout, ownership and both descriptor paths.

## Integrated Models

| Model | Evidence And Limitations |
| --- | --- |
| SenseVoiceSmall | Architecture, strict loader, source SANM attention and synthetic audio CLI |
| Paraformer | Exact-source attention/CIF, strict loader and synthetic CLI; CPU CIF default |
| ct-punc | All tiny regressions, actual PyTorch convolution taps and local text CLI |
| fsmn-vad | All tiny regressions, exact source/cache and local audio CLI |
| CAM++ | Architecture/layout regressions, all-parameter native loading and embedding CLI |
| Whisper | Canonical OpenAI v20250625 full forward and weight transfer for B=1/B=2; raw greedy IDs only |
| emotion2vec+large | Official fairseq CNN and positional construction goldens, strict namespace and synthetic head CLI |

CPU CIF encloses the entire private mathematical subtree in one MLX CPU stream and evaluates its result before returning. Alpha accumulation uses float64 then narrows to float32 exactly as the reference. The installed Metal path remains unverified after reproducible corruption; no warmup workaround or GPU correctness claim is accepted.

Twenty-one integration tests create bounded synthetic checkpoints, load every actual native parameter through the public factory/loader, and exercise all seven CLI workflows. They also reject family/shape mismatches. Integration fixed CAM++ string-key container access, Whisper mel layout, VAD snipped-frame tail alignment and duplicate emotion batch axes. The emotion factory accepts its validated dataclass schema and requires exact labels for scoring. Edge cases explicitly reject unknown configuration keys and overlong input without silent truncation. A default LFR-width-five/stride-one VAD test verifies unchanged temporal cadence.

## Integration Record

- Core infrastructure commit: d734024.
- ASR branch integrated through e7b3155 after 136 focused/source tests passed.
- Utility branch integrated through 406916c after 129 focused/source tests passed.
- Independent static cross-reviews: Herdr tasks 35 and 36. Final core/CLI integration review: task 38. Its configuration-error consistency finding was addressed; claims that admission must truncate and LFR width equals stride were rejected against explicit policy/source formulas and new regression evidence.
- Team: pm4f14. Reports: /Users/baicai1145/repos/funasr/.pi/herdr-team/pm4f14/reports.

Differential tests: tests/test_reference_parity.py, tests/test_model_reference.py and tests/test_emotion_reference.py. Small pinned references are under .tmp/emotion2vec-reference and .tmp/openai-whisper-model.py. The cold CPU/Metal diagnostic is .tmp/cif_probe.py. All temporary artifacts and reports remain under .tmp/.

## Remaining Acceptance

Official checkpoint conversion/loading for every model, canonical tokenizer and label assets, full-model real-audio accuracy, streaming workflow acceptance, quantization quality, then measured steady-state latency, real-time factor and memory. CAM++ full-source model parity and emotion full-encoder parity remain broader checks than the validated components. Small tests do not establish production support, fastest inference or lowest memory.
