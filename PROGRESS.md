# Development Progress

All seven native MLX prototypes are integrated into main. Official assets have now been acquired, and full-weight real-audio diagnosis has started. Seven-model acceptance and corpus-wide validation remain unfinished.

## Official Assets And Diagnostic Evidence

- All 39 selected files downloaded with pinned revisions and checksums; 6,661,591,343 bytes at `/Volumes/2T/funasr-mlx-alignment`. Source records: `catalog.json` and `downloaded.json`.
- Prepared complete acquired corpora: LibriSpeech test-clean 2,620 clips (CC BY 4.0), AISHELL-derived mirror 6,920 clips (not the canonical 7,176-clip split), and RAVDESS speech 1,440 clips (CC BY-NC-SA 4.0). The AISHELL mirror does not declare a license; redistribution and commercial use remain unverified. Five RAVDESS stereo clips are explicitly downmixed. All selected waveform inputs are shared between backends. Coverage, archive/shard hashes, revisions, language and license status survive partial reruns; numerical parity does not resolve legal limitations.
- Initial diagnostic manifests select 8 English, 8 Chinese and 8 emotional clips, each no longer than 8 seconds. These are not full-corpus acceptance results.
- Safe official weight inventories read all seven checkpoints. Paraformer uses legacy serialization, which cannot mmap. ct-punc has a 972,282,288-byte embedding tensor: the bounded converter must not silently reduce this vocabulary. Emotion2vec inference tensors total 648,560,740 bytes despite its larger training-checkpoint file.
- Converted complete FSMN-VAD, CAM++ and Whisper tiny native artifacts in FP32. Converting Whisper FP16 storage to FP32 does not recreate information absent from the original weights.
- FSMN-VAD: official PyTorch CPU and native MLX/Metal ran separately and serially on all 16 diagnostic speech clips. All 16 segment endpoint arrays match exactly. Maximum posterior absolute difference is `9.179115295410156e-06`; maximum relative L2 error is `3.718324821923118e-06`. Frontend maximum absolute difference is `2.816915512084961e-04`, relative L2 maximum `3.5863460108421554e-06`. These are measurements, not an implicit acceptance tolerance. Detailed report: `/Volumes/2T/funasr-mlx-alignment/results/fsmn-vad/comparison.json`. Frame probability/state diagnostics were not measured because the official configuration disables them.
- Real VAD loading exposed an incorrect bias-free wrapper namespace. Restored `fsmn.<i>.linear.linear.weight`, corrected the previously misleading test expectations, and added an actual-source regression that failed before the fix and passes after it. Focused VAD/source/pipeline suite: 66 passed.
- CAM++: complete official FP32 weights were run separately on PyTorch CPU and MLX/Metal for all 16 selected speech clips. Embedding maximum absolute difference is `4.887580871582031e-05`, relative L2 maximum `7.275810778835532e-06`. Frontend maximum absolute difference is `0.0016393661499023438`, relative L2 maximum `5.713358277099895e-06`. Final post-review rerun and required-stage comparison completed. Report: `/Volumes/2T/funasr-mlx-alignment/results/campplus/comparison.json`.
- CAM++ real loading exposed numeric nonlinear names instead of reference names and a converter that widened scalar training counters to shape `[1]`. Named BatchNorm/PReLU modules now match the source. Only owned scalar int64 BatchNorm training counters may be ignored, with names reported. The converter retains zero-dimensional integers. CAM++ frontends now share model-owned povey/no-waveform-upscaling defaults, preserving explicit overrides; this resolved a large real-audio mismatch.
- Whisper tiny: all 16 selected speech clips completed full encoder and prompt-conditioned decoder forward on pinned `openai-whisper==20250625` FP32 CPU and native FP32 MLX/Metal in separate processes. Frontend maximum absolute difference `4.6253204345703125e-05`; encoder `0.0034027099609375` with relative L2 maximum `1.2471155891987754e-05`; decoder logits `0.0002512931823730469` with relative L2 maximum `1.4754031243654753e-05`. All 16 next-token argmaxes agree. This does not measure transcription accuracy or full autoregressive decoding. Report: `/Volumes/2T/funasr-mlx-alignment/results/whisper/comparison.json`.
- Whisper diagnosis isolated two adapter errors: mel axes and bypassing the causal `model.logits()` entry point. With identical stored reference encoder output, a missing mask produced max absolute error `26.4568`; using the canonical mask reduced it to `2.9564e-05`. The adapter uses public logits, and direct native decoder calls now default to the canonical causal mask. B=1/B=2 actual-source regressions failed before that core fix and pass after it.
- Comparisons require family-specific stages and matching case keys. Empty VAD diagnostics are distinct from measured agreement; empty segment arrays are legitimate no-speech outputs. Missing/structurally mismatched stages invalidate reports and return a nonzero exit. Shared prompt IDs are marked `shared_input`, never numerical agreement. Reports include schema `funasr-mlx-measurements/2`, comparator SHA256 and an integrity verdict, not an acceptance tolerance.
- Final guarded repository suite: **457 passed**, zero failures/skips; sampled child peak RSS `384,581,632` bytes, project peak `735,035,392`, lowest available `3,902,930,944`, no violation. JUnit `.tmp/official-alignment-complete.xml`; memory `.tmp/official-alignment-complete-memory.json`. These tests use synthetic models, not full-corpus acceptance.
- Independent read-only reviews: `review-alignment-pending-changes.md` and `review-alignment-fix-pass.md` in the team reports directory. No backend imports/numerics were performed by the reviewer. High findings and follow-up integrity/provenance findings were addressed and regression-tested; numerical evidence is from lead executions only.
- The evaluation venv is `/Volumes/2T/funasr-mlx-alignment/env`; it explicitly reuses existing base framework packages. An initial supervision invocation installed small optional dependencies into the base venv unintentionally; subsequent jobs launch the supervisor from the evaluation venv. Torch and MLX were not upgraded.

## Boundaries

- Work directory: /Users/baicai1145/repos/funasr-mlx. FunASR reference code remains read-only.
- Hard project ceiling: 10,000,000,000 bytes; stop threshold 8 GB; default job budget 2 GB; host reserve 3 GB.
- Maximum two implementation agents; workers are static-only, and the lead serializes numerical verification under the canonical git-common-dir lock.
- MLX memory limits are guidelines. Bounded input/shards, admission and external RSS monitoring are required.
- Full-model allocation needs its own memory admission. Asset I/O budgets never authorize full-model inference. The 3 GB host reserve and project ceilings remain unchanged.
- Small-model launcher `tools/run_small_alignment.py` uses 1.6 GB model jobs; native Whisper tiny has a separately calibrated 1.2 GB model job / 800 MB MLX guideline (151 MB full weights; measured native peak about 227 MB). The original 1.6 GB tiny-native reservation was denied at 4.44 GB available, before allocation. The calibrated public native command then completed all 16 cases and comparison, with sampled peak `214,614,016` bytes, lowest available `3,595,665,408`, no violation. This is fixed full-model inference admission, not an asset-I/O budget.
- Resume one backend explicitly, without waiting/polling: `tools/run_small_alignment.py --root /Volumes/2T/funasr-mlx-alignment --families whisper --mode native`. Run it with the evaluation Python and `PYTHONPATH` containing the native `src` and reference repository. The default runs all three small families serially. Reference-only mode does not compare against a potentially stale native capture.

## Prototype Baseline Before Official Alignment

Previously verified guarded complete suite: **350 passed**, zero failures and skips, in 1.43 seconds. Report: .tmp/final-main-memory.json; JUnit: .tmp/final-main.xml.

Sampled child peak RSS: 318,111,744 bytes. Sampled project peak RSS: 1,115,897,856 bytes. Lowest available host memory: 6,449,119,232 bytes. No memory-policy violation. These measurements apply to synthetic tests, not real models.

Core infrastructure has 64 focused tests. Numerical frontend evidence includes six exact torchaudio Kaldi cases, six FunASR LFR cases, and two torch-STFT/librosa Slaney cases. Real watchdog and supervisor-exit probes verified owned-child termination and cleanup; simulated tests cover denial, timeout, ownership and both descriptor paths.

## Integrated Models

| Model | Evidence And Limitations |
| --- | --- |
| SenseVoiceSmall | Architecture, strict loader, source SANM attention and synthetic audio CLI |
| Paraformer | Exact-source attention/CIF, strict loader and synthetic CLI; CPU CIF default |
| ct-punc | All tiny regressions, actual PyTorch convolution taps and local text CLI |
| fsmn-vad | Complete official FP32 weights, 16 real clips; exact endpoints, measured frontend and offline encoder posteriors; frame diagnostics not captured |
| CAM++ | Complete official FP32 weights, 16 real clips; measured frontends and 192-dimensional embeddings; no speaker EER acceptance |
| Whisper | Complete official tiny weights, 16 real clips; full encoder and prompt-conditioned decoder parity, 16 next tokens exact; no transcription acceptance |
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

Seven-model acceptance remains incomplete. Remaining families: SenseVoiceSmall, Paraformer, ct-punc and emotion2vec+large. Full-weight numerical inference has not been executed for those four. Paraformer needs bounded legacy-serialization handling; ct-punc needs a bounded solution for its 972 MB embedding without reducing vocabulary; emotion needs safe inference-state extraction from its larger training payload and explicit namespace disposition. Larger models require fresh admission under the unchanged project limits. Repeated model construction per utility entry and original-versus-derived conversion config provenance/resumption remain recorded tooling limitations.

Next acceptance stages are complete canonical checkpoints/tokenizers/labels for every model, full-model and multi-batch accuracy, streaming workflow tests, full-corpus WER/CER or task metrics, quantization quality, then steady-state latency, real-time factor and memory. VAD offline posterior parity does not prove streaming-cache equivalence; Whisper next-token agreement is not a transcript result; small numerical errors do not establish classifier accuracy. No quantization, speedup, fastest-inference or lowest-memory acceptance is claimed.

## 2026-10-03: All seven models validated on complete official FP32 weights + FP16 inference

### FP32 numerical alignment (reference PyTorch vs native MLX, full weights, real inputs)
| family | inputs | key measurements | decisions |
|---|---|---|---|
| emotion2vec+large | 8 RAVDESS clips | frames 2.62e-5 / pooled 1.96e-5 / probs 4.77e-7 (max abs) | - |
| ct-punc | 8 zh (in-vocab) + 8 en (all-unk fixtures) | zh logits 2.86e-5 | punc ids 16/16 |
| SenseVoiceSmall | 4 en + 4 zh clips | frontend 2.82e-4 / encoder 1.05e-4 / logits 3.70e-4 | token ids 8/8 |
| Paraformer | 4 en + 4 zh clips | frontend 2.82e-4 / encoder 2.65e-6 / decoder logits 3.46e-1 | token ids 8/8 |

(fsmn-vad, campplus, whisper-tiny were validated earlier; unchanged.)

### FP16 inference validation (native fp16 vs fp32 reference)
- emotion2vec+large: official checkpoint is fp16-upcast-to-fp32 (100% exact f16
  codes) -> fp16 conversion lossless, outputs bit-identical.
- ct-punc: zh logits drift 1.77e-2; decisions 16/16.
- SenseVoiceSmall: encoder drift 2.53e-2; token ids 8/8.
- Paraformer: encoder drift 1.28e-3; token ids 8/8.

### Speed / MFU analysis (M4, tools/benchmark_speed.py)
- fp32 is the fastest config for these model sizes: sensevoice 25.3ms,
  paraformer 24.7ms per sample; fp16 is ~1.5x SLOWER (mixed-dtype promotion),
  uniform-f16 casts made it 2.6x slower (reverted); mx.compile +0.8% (noise);
  batch=8 amortization worse (43.9ms/sample).
- MFU 0.2-0.7% of M4 peak is inherent: 70 sequential 560-dim GEMM layers are
  launch/memory-latency bound. Within the MFU-ceiling constraint there is no
  further speed optimization that does not trade validated numerics.
- FP16's benefit is memory: ct-punc weights 1.13 GB -> 0.57 GB.

### Port defects found and fixed this phase
- LFR left padding repeated the first frame once too often (every fused row
  shifted by one frame) - frontend.py apply_lfr rewritten to mirror upstream.
- SenseVoice CTC logits double-applied ctc_lo (log_softmax projects internally).
- emotion2vec classify() return order; reference stripped extra tokens twice.
- Reference ASR frontend inherited FunASR dither=1.0 default vs native 0.0.
- read_config parsed JSON as YAML (1e-05 -> string); native dtype gate compared
  str(dtype) never matching; converters now inject tokenizer-derived
  vocab/input sizes; legacy torch.load degrades explicitly.

Commits: 2eb4668, 9d5be6a, 48b4b5b, 4c327d4, 64dedf6.

## 2026-10-03 (cont.): SDPA fusion across all attention models + speed envelope

Per-model outcome of "operator fusion where it exists":
| family | attention | fused path | speed |
|---|---|---|---|
| sensevoice | SANM x70, T=99 | SDPA (mask folded to None) | 25.3 -> 23.2 ms (-8.3%) |
| paraformer | SANM 50+16, T=99 | SDPA (same) | 24.7 -> 24.2 ms (-1.9%) |
| whisper-tiny | MHA, T=1500 | SDPA (encoder self + decoder cross) | 38.4 -> 30.0 ms (-22%) |
| emotion2vec | Alt x24 + ALiBi | SDPA (bias as additive mask) | ~unchanged (81 ms; MLP-bound) |
| ct-punc | SANM copy x6, T~13 | not applied (T too short; 6.0 ms is GEMM-bound) | - |
| fsmn-vad | none | n/a | - |
| campplus | none | n/a | - |

Whisper parity on saved artifacts: encoder 1.4e-4 relative vs torch reference,
decoder one-step 4.96e-5 vs old native (which was 1.66e-4 vs reference),
next token identical.

Where the remaining time goes (measured, not estimated):
- emotion2vec 75 ms = MLP GEMM (174x1024@1024x4096 x2 x12 = 29 ms at ~1.7
  TFLOPS, the M4's realistic fp32 GEMM rate) + attention 1.5 ms/layer + LN.
  f16 blocks save only 10% and cost 8.3e-4 relative - not adopted for speed.
- SANM stacks are launch-latency bound (0.55 ms/layer, ~15 kernels); the
  un-fusable remainder (GEMM, depthwise conv, mx.fast.layer_norm) has no
  MLX fused equivalents.
- mx.compile: +0.8% (noise). astype to same dtype is near-zero-copy, so
  Fp32LayerNorm carries no cast tax in f32.

Commits: 05ea904 (SANM SDPA), 7999814 (emotion2vec ALiBi-as-mask), d5a72d0
(whisper SDPA).

## 2026-10-03 (cont.2): README model-coverage expansion

Goal: adapt every model in the FunASR README model zoo.

### Done this round
- **paraformer-zh-streaming** (commit ae71fe3): full streaming protocol
  ported (pe_online positions, overlap stitching, KV caches, streaming CIF
  with carried state, chunk decoder). Official 881 MB online checkpoint
  loads unchanged. Numerics: per-layer RELATIVE diff 1e-6 (pure fp32 GEMM
  reduction noise), tail-chunk path 4.0e-7 -- no semantic divergence. The
  streaming state machine amplifies fp32 noise across chunks: cumulative
  alpha drift ~0.1 flips end-of-chunk CIF fires; mean CER vs the torch
  reference 0.156 on 8 clips. Precision finding, not a porting bug (MLX
  has no fp64 to prove bitwise equivalence like the offline ports).
  Deployment: streaming for partial results; final transcript from the
  offline model.
- **whisper-large-v3 / v3-turbo** (commit 34ba232): registered as alignment
  variants; checkpoints downloaded (sha256 from official _MODELS URLs);
  conversion shards 250 MB with funasr-mlx metadata and conv1d permute.
  Dims verified from the checkpoints: n_mels=128, n_vocab=51866, turbo
  decoder = 4 layers. Both load and run through the guarded pipeline.
  RTF (encoder, 30 s window, FP32): large-v3 0.058 (17x realtime),
  turbo 0.061 (16x); fp16 large-v3 0.046 (22x). Component-level budget
  confirms this is the M4 fp32 compute envelope, not a regression:
  attention 16.3 ms + MLP 19.5 ms per layer x 32.

### LLM-based ASR models (Fun-ASR-Nano / Qwen3-ASR / GLM-ASR-Nano / MOSS)
These are thin wrappers over external LLM stacks: Qwen3-ASR delegates
entirely to the `qwen-asr` pip package; GLM uses HF AutoModel with
trust_remote_code; Fun-ASR-Nano = FunASR audio encoder + HF
AutoModelForCausalLM (Qwen3-0.6B backbone); MOSS is third-party. Adapting
them means porting an LLM inference stack. Recommended path: reuse
mlx-lm's Qwen3/GLM implementations and port only the audio encoder +
adaptor. Not started; each is a project of the size of this whole port.
