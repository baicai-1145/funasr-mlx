# Development Progress

This is an unfinished native MLX port. No model is certified against a real checkpoint or real audio yet.

## Constraints

- Work directory: /Users/baicai1145/repos/funasr-mlx. FunASR reference code is read-only.
- Hard project ceiling: 10,000,000,000 bytes. Stop project jobs at 8 GB; default job budget 2 GB, minimum host available-memory reserve 3 GB.
- Maximum two implementation agents; workers perform static checks only. Lead serializes all numerical verification through the canonical git-common-dir lock.
- MLX set_memory_limit is a guideline, not a hardware-enforced cap. Admission, bounded input/shards and an external RSS watchdog are necessary.
- No full model weights were downloaded or loaded during this stage. All numerical work so far uses tiny dimensions.

## Verified Infrastructure

64 tests passed in the fresh guarded core run: memory policies, authenticated inherited guard descriptor/nonce, lock contention, simulated watchdog/timeout, lazy imports, strict safetensors metadata, tiny mmap checkpoint conversion, CLI inspection, Kaldi fbank, LFR/CMVN and Whisper log-mel.

Numerical frontend evidence includes six exact torchaudio Kaldi cases (three window types, two edge modes), six exact FunASR LFR cases and two torch-STFT/librosa Slaney cases.

Latest core test process sampled peak RSS: 289,456,128 bytes. Project sampled peak: 1,554,792,448 bytes.

A separate bounded watchdog probe allocated 64 MB under a 48 MB process policy. The child was terminated by SIGTERM in approximately 0.15 seconds; sampled child RSS was 87,506,944 bytes. This verifies stop behavior, not a guarantee for arbitrary unbounded allocations.

Supervisor liveness is protected by a parent-only write pipe. A real bounded probe killed its own supervisor and verified that the orphan child stopped. Shard headroom, guarded model construction, explicit CMVN disable, relative CMVN paths and both pipe-descriptor cleanup paths have regression coverage.

## Model Work

All seven first implementations are committed in isolated worktrees, not yet integrated into main.

| Model | Current Verification State |
| --- | --- |
| SenseVoiceSmall | Architecture and strict-loader regressions iterated; real checkpoint/audio untested |
| Paraformer | SANM exact-source attention cases pass; CIF GPU mismatch remains open |
| ct-punc | All tiny regressions and actual PyTorch convolution tap goldens pass; real checkpoint untested |
| fsmn-vad | All tiny regressions and exact PyTorch source/cache goldens pass; real checkpoint untested |
| CAM++ | Runtime fixes committed; test import corrected; fresh suite/source parity pending |
| Whisper | Corrected to canonical OpenAI Whisper v20250625, not the SenseVoice fork; numerical parity pending |
| emotion2vec+large | Official/fairseq architecture and classifier sources located; checkpoint namespace/API/SamePad issues being repaired |

CIF defaults to an entirely MLX CPU computation with float64 alpha accumulation, matching the exact reference. Six small ASR source differential cases pass. The installed Metal execution path remains unverified after reproducible corruption; no warmup workaround or GPU-correctness claim is accepted.

## Team And Artifacts

Herdr team: pm4f14. Reports: /Users/baicai1145/repos/funasr/.pi/herdr-team/pm4f14/reports.

- asr worktree: /Users/baicai1145/repos/funasr/.pi/herdr-team/pm4f14/worktrees/asr. Current tasks: task-19 CIF gather repair (after task-18 CAM API + emotion review).
- speech-utils worktree: /Users/baicai1145/repos/funasr/.pi/herdr-team/pm4f14/worktrees/speech-utils. Current task: task-20 emotion checkpoint/review repairs.
- Latest relevant ASR commits: 73512f2 canonical Whisper, 733ba39 CIF algebra, 03b308a CAM checkpoint-rule API.
- Latest relevant utilities commit: 63ba959 emotion API/SamePad/utility fixtures (needs further review repairs).

Guarded launcher: PYTHONPATH=src python3 -m funasr_mlx.runtime --root . --report .tmp/check.json -- python3 -m pytest tests/... -q --tb=short --basetemp=.tmp/pytest-check.

Unmerged worktree verification uses .tmp/verify_worktree.py. Small source references are in .tmp/emotion2vec-reference and .tmp/openai-whisper-model.py. Exact-source differential tests are in tests/test_reference_parity.py. The standalone cold GPU/CPU reproduction is .tmp/cif_probe.py. Reports and JUnit XML are under .tmp/.

## Remaining Acceptance

Independent reviews, numerical parity, strict real-checkpoint conversion/loading, tokenizer/labels and audio workflows, real-audio regression, quantization quality, then measured latency/RTF and peak memory. Tiny architecture tests do not establish production support or fastest/lowest-memory claims.
