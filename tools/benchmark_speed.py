"""Benchmark native MLX inference: latency + achieved fraction of GPU peak.

Runs each validated family through its real adapter code path, timing the
per-sample forward (model already loaded). Reports:
  - mean/p50/p95 latency per sample
  - achieved FLOP/s against the M4 GPU's theoretical peak (MFU)

The MFU ceiling: optimization must not change numerics beyond the validated
FP16 drift. FLOP estimates are documented order-of-magnitude (2*params MACs
per forward per second of audio), so MFU here is a comparative metric across
configurations of the same model, not an absolute claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

REPO = Path("/Users/baicai1145/repos/funasr-mlx")
ROOT = Path("/Volumes/2T/funasr-mlx-alignment")

# Apple M4 10-core GPU: ~5.4 TFLOPS FP32 vector, ~10.9 TFLOPS FP16.
PEAK_TFLOPS = {"float32": 5.4, "float16": 10.9}

# Parameter counts (from the validated conversions' tensor inventories).
PARAMS = {
    "fsmn-vad": 1_103_000,
    "campplus": 7_163_000,
    "emotion2vec": 162_140_000,
    "ct-punc": 281_300_000,
    "sensevoice": 233_999_000,
    "paraformer": 220_085_000,
}

FAMILIES = tuple(PARAMS)


def bench_utilities(family: str, dtype: str, repeats: int, warmup: int) -> dict:
    """fsmn-vad / campplus / emotion2vec / ct-punc via alignment_utilities."""
    import sys

    sys.path.insert(0, str(REPO))
    import json as _json

    manifest = _json.loads((ROOT / "inputs" / f"{family}.json").read_text())
    assets = _json.loads((ROOT / "inputs" / f"{family}-assets.json").read_text())
    native = ROOT / ("native-fp16" if dtype == "float16" else "native-full") / family
    if native.is_dir():
        assets = {**assets, "checkpoint": str(native)}

    if family == "ct-punc":
        import mlx.core as mx

        from funasr_mlx.models.ct_punc import CharVocab, split_words
        from funasr_mlx.pipeline import load_model
        import yaml

        raw = yaml.safe_load(Path(assets["config"]).read_text())
        tokens = _json.loads(Path(assets["tokens"]).read_text())
        conf = {**raw, "model_conf": {**(raw.get("model_conf") or {}), "vocab_size": len(tokens)}}
        model, _ = load_model(native, family="ct-punc", config=conf)
        text = manifest[0]["text"]
        ids = list(CharVocab(tokens).encode(list(split_words(text))))

        def run():
            logits = model.punc_forward(mx.array([ids], dtype=mx.int32))
            mx.eval(logits)
            return logits

        flops_units = "run"
    else:
        import mlx.core as mx
        import numpy as np
        import yaml

        from funasr_mlx.frontend import read_audio
        from funasr_mlx.pipeline import load_model

        if family == "emotion2vec":
            from funasr_mlx.models.emotion2vec import preprocess_waveform
        cfg_path = Path(assets["config"])
        kwargs = None
        if family == "emotion2vec":
            from tools.alignment_utilities import emotion_native_config

            raw = yaml.safe_load(cfg_path.read_text())
            labels = Path(assets["labels"]).read_text().splitlines()
            kwargs, _ = emotion_native_config(raw, labels)
        model, config = load_model(
            native, family=family, config=kwargs if kwargs is not None else cfg_path
        )
        samples = np.asarray(read_audio(manifest[0]["audio"]), dtype=np.float32)

        if family == "emotion2vec":
            wave = preprocess_waveform(mx.array(samples[None, :]), normalize=True)

            def run():
                pooled, logits, frames = model.classify(wave)
                mx.eval(pooled, logits)
                return logits

        elif family == "campplus":
            import librosa

            feat = librosa.feature.mfcc(
                y=samples, sr=16000, n_mfcc=80, **{"n_fft": 512, "hop_length": 160}
            )
            au = np.stack([feat.T])
            from funasr_mlx.models.campplus import project_features

            feats = project_features(np.ascontiguousarray(au))

            def run():
                emb = model(mx.array(feats))
                mx.eval(emb)
                return emb

        else:  # fsmn-vad
            feats = np.zeros((1, 200, 80), dtype=np.float32)

            def run():
                out = model.detect_speech(mx.array(feats))
                mx.eval(out)
                return out

        flops_units = "second-of-audio"
    times = []
    for _ in range(warmup):
        run()
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    return {
        "times": times,
        "flops_units": flops_units,
    }


def bench_asr(family: str, dtype: str, repeats: int, warmup: int) -> dict:
    """sensevoice / paraformer via the alignment adapter's native path."""
    import sys

    sys.path.insert(0, str(REPO))
    import json as _json

    import mlx.core as mx
    import numpy as np
    import soundfile as sf

    from funasr_mlx.frontend import FbankConfig, WavFrontend
    from funasr_mlx.pipeline import load_model

    spec = _json.loads((ROOT / "results" / f"{family}-full" / "spec.json").read_text())
    native = ROOT / ("native-fp16" if dtype == "float16" else "native-full") / family
    model, _ = load_model(native, family=family, config=native / "config.json")
    item = spec["inputs"][0]
    audio, _ = sf.read(item["audio"], dtype="float32")
    cfg = FbankConfig(
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        window="hamming",
        n_mels=80,
        lfr_m=7,
        lfr_n=6,
    )
    frontend = WavFrontend(cfg, cmvn=spec["variant"]["cmvn"])
    fbank = np.asarray(frontend(audio), dtype=np.float32)
    speech = mx.array(fbank)[None, :, :]
    lengths = mx.array([fbank.shape[0]], dtype=mx.int32)

    if family == "sensevoice":

        def run():
            out, lens = model.encode(speech, lengths, language="zh", use_itn=False)
            logits = model.ctc.log_softmax(out)
            mx.eval(logits)
            return logits

    else:

        def run():
            outputs = model.forward(speech, lengths)
            mx.eval(outputs.decoder_out)
            return outputs.decoder_out

    times = []
    for _ in range(warmup):
        run()
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    return {"times": times, "flops_units": "second-of-audio"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=FAMILIES, action="append", required=True)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = {"dtype": args.dtype, "peak_tflops": PEAK_TFLOPS[args.dtype], "families": {}}
    for family in args.family:
        bench = bench_asr if family in ("sensevoice", "paraformer") else bench_utilities
        data = bench(family, args.dtype, args.repeats, args.warmup)
        times = data["times"]
        mean = statistics.mean(times)
        p50 = statistics.median(times)
        p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
        flops = 2 * 2 * PARAMS[family]
        achieved = flops / mean / 1e12
        report["families"][family] = {
            "mean_s": round(mean, 6),
            "p50_s": round(p50, 6),
            "p95_s": round(p95, 6),
            "flops_per_unit": flops,
            "flops_units": data["flops_units"],
            "achieved_tflops": round(achieved, 3),
            "mfu_vs_peak_pct": round(100 * achieved / PEAK_TFLOPS[args.dtype], 2),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
