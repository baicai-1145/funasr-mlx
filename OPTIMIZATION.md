# Optimization Backlog — 17 Items by Distance-to-Roofline

**Date:** 2026-10-04 · **Host:** MacBook Air M4 (16 GB) · **Batch size:** 1 everywhere

> **Post-mortem addendum (2026-10-04, whisper sprint).** Two items below were
> closed as MEASURED-AND-REJECTED, and the lead's original attribution for both
> was wrong twice. Recorded so the same traps are not re-run:
>
> 1. **Encoder elementwise fusion (#13 tiny / large enc rows)**: NOT real. The
>    1.65x came from (a) a timing-protocol artifact — `for KK: r=fn()` + one
>    `mx.eval(r)` evaluates only the last lazy graph (wall time FLAT in KK while
>    "per-iter" falls 1/KK; real slice cost ~3.0-3.4 ms, not 210 us), and (b) a
>    hand-rolled 3-kernel GELU, while the model uses fused `mxnn.GELU()`.
>    Honest paired A/B: tiny -1.3% p=0.29, large-v3 -0.06% p=0.90. Encoder is
>    already at 73% of machine fp16 peak and FASTER than the sum of its parts.
>    The 13-15 ms target was below the hardware floor (36.7 GFLOP / 2.44 TFLOPS
>    = 15.0 ms). Evidence: reports/whisper-encoder-fusion.md.
> 2. **Decode KV-append replacement (whisper decode rows)**: NOT worth it. The
>    "52% of layer" claim was a sum-of-parts fallacy (isolated concat 0.17-0.30
>    ms summed into block attribution; inside a real block the append is ~0.1 ms
>    of ~1.1 ms). A numerically exact prealloc/slice-assign buffer was built and
>    reverted: 300-pair A/B -0.6%, CI95 [-0.158,+0.087], p=0.603. Ceiling for ANY
>    append mechanism: 0.116/0.079 ms (10%/7% of block); slice-assign captured
>    negative. Remaining decode cost is per-op dispatch count + the lm_head
>    bandwidth floor — no cache representation removes it.
>    Evidence: reports/task-3-*.md.
>
> **Methodology rules that fell out** (apply to every future probe):
> - eval INSIDE the timed region, every iteration; never rebind-and-drop.
> - Probe with the model's REAL ops (`mxnn.GELU`), never hand-rolled stand-ins.
> - Never sum isolated op costs into a stage attribution — inter-op overlap
>   hides/shows costs (real chain beat parts-sum in both directions here).
> - Paired in-process A/B is the only valid comparison; absolutes drift with
>   thermal (observed: same probe 4.73 vs 5.97 ms across sessions).
> - A 2.5%/p=0.055 result that fails to replicate at n=300 is noise; demand
>   >=5% with CI excluding zero before shipping.

## Measurement basis

- **Ceilings are measured on this machine, not spec numbers**
  (`.tmp/probe_hardware.py`, best-of-10):
  - fp16 GEMM **2.58 TFLOPS** @ 4096³ (2.30 @ 8192³)
  - fp32 GEMM **2.27 TFLOPS** → fp16 has only a **1.14×** compute edge on this host
  - DRAM bandwidth **92 GB/s**
  - Ridge point ≈ 28 FLOP/byte: decode (intensity ~1.3) is memory-bound → judge
    by **MBU**; towers/prefill are compute-bound → judge by **MFU**.
- Latency: `tools/benchmark_speed.py` + `.tmp/bench_fp16_llm.py` /
  `.tmp/bench_fp16_small.py`, current post-SDPA code, warmup + 20-30 repeats.
- FLOPs/bytes are analytic (2·params·tokens + attention T² terms; weights read
  once per call), not kernel-traced — the *gap*, not the percentage, is the signal.
- Full tables: `.tmp/ceiling_analysis.json`, `.tmp/ceiling_analysis_small.json`.

## The 17 items, worst gap first

| # | Model | Stage | dtype | t (ms) | MFU% | MBU% | **Gap** | Bound | RTF context (fp16) |
|---|-------|-------|-------|--------|------|------|---------|-------|---------------------|
| 1 | fsmn-vad | forward | fp16 | 3.4 | 3.2 | 0.6 | **31.2×** | launch latency | 0.0010 (968×) |
| 2 | whisper-tiny | decode/step | fp16 | 5.2 | 0.3 | 7.2 | **13.9×** | bandwidth | 3.0 ms/step effective |
| 3 | moss | adaptor | fp16 | 2.0 | 10.7 | 5.7 | **9.4×** | mixed | (tower unmeasured, see #17) |
| 4 | campplus | forward | fp16 | 15.0 | 11.7 | 1.1 | **8.5×** | launch latency | 0.0046 (219×) |
| 5 | glm-asr-nano | decode/step | fp16 | 244.4 | 0.5 | 14.2 | **7.1×** | bandwidth | RTF 1.53 — not realtime |
| 6 | glm-asr-nano | prefill | fp16 | 306.0 | 15.0 | 11.4 | **6.7×** | compute | ↑ |
| 7 | qwen3-asr-0.6b | decode/step | fp16 | 98.9 | 0.6 | 16.5 | **6.1×** | bandwidth | RTF 0.605 (1.7×) |
| 8 | whisper-tiny | decode/step | fp32 | 4.0 | 0.4 | 18.7 | **5.3×** | bandwidth | fp32 faster than fp16 |
| 9 | emotion2vec | forward | fp16 | 60.3 | 23.7 | 6.0 | **4.2×** | compute | — |
| 10 | qwen3-asr-0.6b | prefill | fp16 | 127.0 | 30.5 | 13.0 | **3.3×** | compute | ↑ #7 |
| 11 | glm-asr-nano | tower | fp16 | 243.0 | 41.8 | 5.7 | **2.4×** | compute | ↑ #5 |
| 12 | moss | prefill | fp16 | 101.0 | 42.4 | 13.1 | **2.4×** | compute | RTF 0.132 (7.6×) |
| 13 | sensevoice | enc+ctc | fp16 | 42.5 | 35.4 | 12.5 | **2.8×** | compute | fp32 is 28.8 ms |
| 14 | paraformer | enc+dec | fp16 | 43.2 | 37.8 | 11.7 | **2.6×** | compute | fp32 is 27.8 ms |
| 15 | emotion2vec | forward | fp32 | 47.1 | 30.4 | 7.7 | **3.3×** | compute | fp32 faster |
| 16 | qwen3-asr-0.6b | tower | fp16 | 41.0 | 77.1 | 9.9 | **1.3×** | compute | near ceiling ✅ |
| 17 | fun-asr-nano | decode+prefill | fp16 | 16.7/31.0 | 3.5 | **98.0** | **1.0×** | bandwidth | AT the roofline ✅ |

*(whisper-tiny encoder 1.6×, sensevoice/paraformer fp32 1.7-1.9×, ct-punc 1.5×
are already within 2× of their roofs and are not listed as work items.)*

## Grouped by fix, highest leverage first

### A. LLM decode path — fixes #5, #7 (and lifts #2's relatives)
**The single biggest win.** fun-asr-nano's decode runs at **98% of memory
bandwidth** (1.0×, item #17) on a **Qwen3-0.6B backbone with the identical
751.6 M params and identical 1.50 GB/step weight traffic as qwen3-asr**, yet
qwen3-asr takes 98.9 ms/step vs nano's 16.7 ms — a **6× gap that is pure call
path, not model size**.
- Suspect 1: `KVCache.update_and_fetch` does `self.keys[:, :, offset:n] = k`
  slice-assign per step; if not in-place this rewrites the whole buffer
  (28 layers × K/V × every step).
- Suspect 2: `.item()` / `mx.argmax(...).item()` forces a sync every token.
- Payoff if brought to nano's efficiency: qwen3-asr RTF 0.605 → ~0.35;
  glm-asr-nano decode 244 → ~40 ms/step, RTF 1.53 → ~0.85 (**crosses into
  realtime**). Verify with `mx.metal.set_wire_per_frame`-style profiling or by
  timing a bare `model(x, cache=...)` loop.

### B. Drop fp16 for mixed-dtype models — fixes #2, #8, #13, #14, #15
On this host fp16 GEMM is only 1.14× fp32, so fp16 **cannot** pay for its
mixed-dtype promotion. Measured regressions vs native precision:
sensevoice −48%, paraformer −55%, emotion2vec −28%, whisper-tiny e2e −43%,
glm decode −18%, qwen3 decode −10%, moss −8%.
**Action:** default these families to fp32 (or bf16) weights; keep fp16 only
where bandwidth-bound (ct-punc: **+28% faster**, the sole winner, because its
971 MB embedding read halves).
Also note SDPA fusion is a **net negative under fp16** here: sensevoice −13%,
paraformer −24%, emotion2vec −11% vs the pre-SDPA fp16 numbers, while it helped
fp32 (−8%/−2%). Gate SDPA on dtype.

### C. Launch-latency-bound small models — #1, #3, #4
fsmn-vad (3.2% MFU, 31×) and campplus (11.7%, 8.5×) are far from any roof, but
their absolute cost (3.4 / 15 ms) is already 219-968× realtime — **the gap is
real yet the payoff is nil unless used in a tight loop** (streaming VAD at
100 fps × chunks). If ever needed: fuse the 24-70 sequential small GEMMs
(`mx.compile` measured +0.8% = noise; needs graph-level batching instead).
fsmn-vad's whole forward is smaller than one Metal dispatch batch.

### D. Remaining compute-bound stages — #6, #9, #10, #11, #12
All transformer forwards at 15-42% MFU. Causes are the documented M4 envelope
(sequential 560-1024-dim GEMMs below the compute-bound threshold) plus, for
GLM tower (#11), attention over T=201 with d=1280. Expected recoverable headroom
is ~2× at best; these are genuine hardware limits, not bugs — deprioritize
behind A and B.

### E. Measurement debt — #3 (moss tower), caveat on #5/#7
- MOSS's whisper encoder ran **outside the timed region** on a padded 30 s
  window; only the 2 ms adaptor was timed. Its published RTF 0.132 is therefore
  **optimistic**. Re-bench with the encoder inside.
- GLM prefill/decode were measured with `lowest_available` down to 1.76 GB
  (below the 3 GB reserve; policy is advisory there) — numbers are valid but
  were collected under memory pressure.
- ct-punc fp32 needs `process_limit ≥ 4 GB` (971 MB largest tensor vs
  `process_limit/4` admission); first-ever fp32 run of that family.

## Non-items (already at/near the roof)

fun-asr-nano decode **1.0×** (98% MBU), its prefill 1.2× (84% MFU), qwen3 tower
1.3× (77%), whisper-tiny encoder 1.6× (61%), ct-punc fp16 1.5× (69% MBU).
**No optimization is possible on these without different hardware.**

## Reproduction

```sh
# ceilings
PYTHONPATH=src python3 -m funasr_mlx.runtime --root . --report .tmp/ph.json -- \
  python3 .tmp/probe_hardware.py
# small families (fsmn-vad/campplus/whisper-tiny)
PYTHONPATH=src python3 -m funasr_mlx.runtime --root . --report .tmp/b.json -- \
  python3 .tmp/bench_fp16_small.py --dtype float16 --repeats 20 --warmup 3 \
  --out /Volumes/2T/funasr-mlx-alignment/results/bench-small-float16.json
# sensevoice/paraformer/emotion2vec/ct-punc (ct-punc fp32 needs ≥4GB process_limit)
python3 .tmp/run_small2.py
# LLM families (serialized)
python3 .tmp/run_bench_matrix.py
# this table
python3 .tmp/ceiling_analysis.py && python3 .tmp/ceiling_analysis_small.py
```
