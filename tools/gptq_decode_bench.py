"""Paired decode A/B: bf16 vs a W4A16 checkpoint, n >= 100 interleaved samples.

The 2x question needs a measured number, not bandwidth arithmetic, and the
measurement has to be paired: this project's own notes record thermal drift large
enough to swamp a single-arm comparison on an M4 Air.  So both arms are loaded
once, and the per-utterance decode is run **alternating** arm by arm -- sample i
runs bf16 then quant, sample i+1 runs quant then bf16 -- which cancels drift
between arms and leaves a paired difference with a bootstrap CI.

What is timed is the decode loop only (one token per step, greedy, no sampling,
no re-encode): the prefill/encode is excluded because it is not what the
quantized head changes, and it is dominated by the audio tower, which is the same
in both arms for this scope.

The transcript is not re-checked here -- that is the acceptance tool's job.  This
tool measures time and reports the paired statistics.

Usage::

    python3 tools/gptq_decode_bench.py --family qwen3-asr-0.6b \
        --bf16-dir <bf16> --quant-dir <w4a16-gptq> \
        --manifest <manifest.json> --out <json> [--samples 100] [--steps 32]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: Job name -> model class family (what load_model dispatches on).
MODEL_FAMILY = {
    "fun-asr-nano": "fun-asr-nano",
    "qwen3-asr-0.6b": "qwen3-asr",
    "qwen3-asr-1.7b": "qwen3-asr",
    "glm-asr-nano": "glm-asr-nano",
    "moss-transcribe-diarize": "moss-transcribe-diarize",
}


def bootstrap_ci(values: list[float], *, draws: int = 10_000, seed: int = 0):
    """Percentile bootstrap CI of the mean (the paired delta's CI)."""
    import numpy as np

    if not values:
        return (None, None)
    rng = np.random.RandomState(seed)
    array = np.asarray(values, dtype=np.float64)
    means = array[rng.randint(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def load_arm(family: str, directory: Path):
    """Load one arm plus its tokenizer directory."""
    from funasr_mlx.pipeline import load_model

    model, config = load_model(directory, family=family)
    if family == "qwen3-asr":
        existing = getattr(model, "config", None)
        if existing is not None and hasattr(existing, "audio_pad_token_id"):
            return model, existing
        from funasr_mlx.models.qwen3_asr_thinker import qwen3asr_config_from_dict

        return model, qwen3asr_config_from_dict(config)
    return model, getattr(model, "config", config)


def _audio_dtype(dtype, *, what: str = "audio input"):
    """Refuse to cast audio to an integer dtype (see gptq_calibrate.audio_dtype)."""
    import mlx.core as mx

    if not mx.issubdtype(dtype, mx.floating):
        raise ValueError(
            f"Refusing to cast {what} to {dtype}: an integer dtype here means a "
            "quantized weight's packed codes were read as an activation dtype."
        )
    return dtype


def build_decode(family: str, model, config, tokenizer_dir: Path, *, steps: int):
    """Return ``(prepare(sample), decode(prepared, steps))`` for one arm.

    ``prepare`` runs the audio tower and the prefill, and returns everything the
    decode loop needs.  Splitting it out is what lets the timing exclude the
    encode: both arms share the tower and the prompt, so only the decode loop is
    worth comparing for a head-quantization change.
    """
    import mlx.core as mx
    import soundfile as sf

    from funasr_mlx.models.llm import KVCache
    from funasr_mlx.models.whisper import log_mel_spectrogram

    from transformers import AutoTokenizer

    if family == "qwen3-asr":
        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=False)
        prefix_ids = tok(
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>",
            add_special_tokens=False,
        )["input_ids"]
        tail_ids = tok(
            "<|audio_end|><|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]
        pad_id = config.audio_pad_token_id
        n_layers = config.text.num_hidden_layers
        # The mel feeds the audio TOWER; follow the tower's dtype, not the
        # decoder embedding's (for a quantized embedding that is the packed-code uint32).
        dtype = _audio_dtype(model._tower_dtype(), what="the mel (qwen3-asr)")

        def prepare(sample):
            raw, _ = sf.read(sample["audio_path"], dtype="float32")
            mel = log_mel_spectrogram(mx.array(raw[None, :]), n_mels=128)
            mel = mx.transpose(mel[0], (1, 0)).astype(dtype)
            mx.eval(mel)
            embeds = model.encode_audio(mel)
            mx.eval(embeds)
            span = int(embeds.shape[1])
            ids = mx.array([prefix_ids + [pad_id] * span + tail_ids], dtype=mx.int32)
            caches = [KVCache() for _ in range(n_layers)]
            logits = model.forward_with_audio(ids, embeds, span, caches)
            mx.eval(logits)
            return caches, int(mx.argmax(logits[0, -1]).item())

        def decode(prepared, count):
            caches, token = prepared
            for _ in range(count):
                logits = model.thinker(inputs=mx.array([[token]], dtype=mx.int32), cache=caches)
                mx.eval(logits)
                token = int(mx.argmax(logits[0, -1]).item())
            return token

        return prepare, decode, steps

    if family == "glm-asr-nano":
        # Same prompt structure the acceptance driver uses for this family
        # (t9-validated ids; no tokenizer needed for a paired timing bench).
        prefix_ids = [59253, 10, 59261]
        tail_ids = [59262, 10, 59254, 10]
        pad_id = 59260
        n_layers = config.text["num_hidden_layers"]
        dtype = _audio_dtype(model._tower_dtype(), what="the mel (glm-asr-nano)")

        def prepare(sample):
            raw, _ = sf.read(sample["audio_path"], dtype="float32")
            mel = log_mel_spectrogram(mx.array(raw[None, :]), n_mels=128)
            mel = mx.transpose(mel[0], (1, 0)).astype(dtype)
            mx.eval(mel)
            embeds = model.encode_audio(mel)
            mx.eval(embeds)
            span = int(embeds.shape[1])
            ids = mx.array([prefix_ids + [pad_id] * span + tail_ids], dtype=mx.int32)
            caches = [KVCache() for _ in range(n_layers)]
            logits = model.forward_with_audio(ids, embeds, span, caches)
            mx.eval(logits)
            return caches, int(mx.argmax(logits[0, -1]).item())

        def decode(prepared, count):
            caches, token = prepared
            for _ in range(count):
                logits = model.language_model(
                    mx.array([[token]], dtype=mx.int32), cache=caches
                )
                mx.eval(logits)
                token = int(mx.argmax(logits[0, -1]).item())
            return token

        return prepare, decode, steps

    if family == "moss-transcribe-diarize":
        # Same protocol as the acceptance driver (t9-validated): the jinja
        # template rendered once around an <|audio_pad|> hole, the chunked
        # whisper tower in prepare (EXCLUDED from timing), greedy decode only.
        from transformers import AutoTokenizer
        import numpy as np

        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=False)
        template = (tokenizer_dir / "chat_template.jinja").read_text()
        prompt = (
            "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
            "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
            "并在段末标注结束时间戳，以清晰标明该段语音范围。"
        )
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": None},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, chat_template=template
        )
        marker = "<|audio_pad|>"
        pre, rest = rendered.split(marker, 1)
        pre_ids = tok.encode(pre)
        tail_ids = tok.encode(rest)
        audio_token_id = tok.convert_tokens_to_ids(marker)
        n_layers = config["text_config"]["num_hidden_layers"]
        tower_weight = model.whisper_encoder.conv1.weight
        mel_dtype = getattr(tower_weight, "dtype", __import__("mlx").core.bfloat16)

        def prepare(sample):
            raw, _ = sf.read(sample["audio_path"], dtype="float32")
            n_samples, hop, stride, merge = 480000, 160, 2, 4
            parts = []
            for start in range(0, len(raw), n_samples):
                chunk = raw[start : start + n_samples]
                tok_len = (len(chunk) - 1) // (hop * stride * merge) + 1
                pad = np.zeros(n_samples, dtype=np.float32)
                pad[: len(chunk)] = chunk
                mel_c = log_mel_spectrogram(mx.array(pad)[None], n_mels=80)[0]
                h = model.whisper_encoder(mx.transpose(mel_c, (1, 0)).astype(mel_dtype))
                parts.append(h[0, : tok_len * 4 * stride, :])
            feat = mx.concatenate(parts, axis=0)[None]
            mx.eval(feat)
            embeds = model.encode_audio_from_hidden(feat)
            mx.eval(embeds)
            n = int(embeds.shape[1])
            ids = mx.array([pre_ids + [audio_token_id] * n + tail_ids], dtype=mx.int32)
            caches = [KVCache() for _ in range(n_layers)]
            logits = model.forward_with_audio(ids, embeds, n, caches)
            mx.eval(logits)
            return caches, int(mx.argmax(logits[0, -1]).item())

        def decode(prepared, count):
            caches, token = prepared
            for _ in range(count):
                logits = model.language_model(
                    mx.array([[token]], dtype=mx.int32), cache=caches
                )
                mx.eval(logits)
                token = int(mx.argmax(logits[0, -1]).item())
            return token

        return prepare, decode, steps

    raise ValueError(f"No decode bench driver for {family}")

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=False)
    prefix_ids = tok(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>",
        add_special_tokens=False,
    )["input_ids"]
    tail_ids = tok(
        "<|audio_end|><|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
    )["input_ids"]
    pad_id = config.audio_pad_token_id
    n_layers = config.text.num_hidden_layers
    # The mel feeds the audio TOWER; follow the tower's dtype, not the decoder
    # embedding's (for a quantized embedding that is the packed-code uint32).
    dtype = _audio_dtype(model._tower_dtype(), what="the mel (qwen3-asr)")

    def prepare(sample):
        raw, _ = sf.read(sample["audio_path"], dtype="float32")
        mel = log_mel_spectrogram(mx.array(raw[None, :]), n_mels=128)
        mel = mx.transpose(mel[0], (1, 0)).astype(dtype)
        mx.eval(mel)
        embeds = model.encode_audio(mel)
        mx.eval(embeds)
        span = int(embeds.shape[1])
        ids = mx.array([prefix_ids + [pad_id] * span + tail_ids], dtype=mx.int32)
        caches = [KVCache() for _ in range(n_layers)]
        logits = model.forward_with_audio(ids, embeds, span, caches)
        mx.eval(logits)
        return caches, int(mx.argmax(logits[0, -1]).item())

    def decode(prepared, count):
        caches, token = prepared
        for _ in range(count):
            logits = model.thinker(inputs=mx.array([[token]], dtype=mx.int32), cache=caches)
            mx.eval(logits)
            token = int(mx.argmax(logits[0, -1]).item())
        return token

    return prepare, decode, steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=sorted(MODEL_FAMILY))
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--quant-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--languages", default="zh,en,ja,ko,es,fr,de,ru,th,vi")
    args = parser.parse_args()



    family = MODEL_FAMILY[args.family]
    languages = [part for part in args.languages.split(",") if part]
    manifest = json.loads(args.manifest.read_text())
    pool = [s for s in manifest["samples"] if s["lang"] in languages]
    # Round-robin over languages so the sample is language-balanced and not just
    # "whatever order the manifest happened to be written in".
    by_language: dict[str, list[dict]] = {language: [] for language in languages}
    for sample in pool:
        by_language[sample["lang"]].append(sample)
    samples = []
    index = 0
    while len(samples) < args.samples:
        added = False
        for language in languages:
            rows = by_language[language]
            if index < len(rows):
                samples.append(rows[index])
                added = True
                if len(samples) == args.samples:
                    break
        if not added:
            break
        index += 1
    if len(samples) < args.samples:
        raise SystemExit(
            f"Only {len(samples)} samples available across {languages}, "
            f"need {args.samples}"
        )
    print(f"paired decode samples: {len(samples)} (steps={args.steps})", flush=True)

    import mlx.core as mx

    bf16_model, bf16_config = load_arm(family, args.bf16_dir)
    quant_model, quant_config = load_arm(family, args.quant_dir)
    bf16_prepare, bf16_decode, steps = build_decode(
        family, bf16_model, bf16_config, args.bf16_dir, steps=args.steps
    )
    quant_prepare, quant_decode, _ = build_decode(
        family, quant_model, quant_config, args.bf16_dir, steps=args.steps
    )
    arms = {
        "bf16": (bf16_prepare, bf16_decode),
        "quant": (quant_prepare, quant_decode),
    }

    # Warm both arms so the first sample is not measuring cold caches / lazy
    # compilation, then drop those timings.
    for name in ("bf16", "quant"):
        prepare, decode = arms[name]
        for sample in samples[: args.warmup]:
            prepared = prepare(sample)
            decode(prepared, steps)
            del prepared
        mx.clear_cache()
    print("warmup done", flush=True)

    times: dict[str, list[float]] = {"bf16": [], "quant": []}
    tokens_agree = 0
    for index, sample in enumerate(samples):
        # Alternate the arm order every sample: drift then cancels in the paired
        # difference instead of being charged to whichever arm always ran second.
        order = ("bf16", "quant") if index % 2 == 0 else ("quant", "bf16")
        produced = {}
        for name in order:
            prepare, decode = arms[name]
            prepared = prepare(sample)
            started = time.perf_counter()
            produced[name] = decode(prepared, steps)
            times[name].append(time.perf_counter() - started)
            del prepared
        if produced["bf16"] == produced["quant"]:
            tokens_agree += 1
        if (index + 1) % 20 == 0:
            print(f"  [{index + 1}/{len(samples)}]", flush=True)

    deltas = [b - q for b, q in zip(times["bf16"], times["quant"])]
    lo, hi = bootstrap_ci(deltas)

    def stats(values: list[float]) -> dict:
        ordered = sorted(values)
        return {
            "n": len(values),
            "mean_s": statistics.fmean(values),
            "p50_s": statistics.median(values),
            "min_s": ordered[0],
            "max_s": ordered[-1],
            "ms_per_step": statistics.fmean(values) / steps * 1000.0,
        }

    payload = {
        "family": args.family,
        "model_family": family,
        "bf16_dir": str(args.bf16_dir),
        "quant_dir": str(args.quant_dir),
        "protocol": (
            "both arms resident in ONE process; per-sample alternating arm order "
            "(bf16-first on even samples, quant-first on odd); encode/prefill "
            "excluded from the timed region; greedy decode only"
        ),
        "decode_steps": steps,
        "samples": len(samples),
        "warmup_samples": args.warmup,
        "languages": languages,
        "timing": {
            "bf16": stats(times["bf16"]),
            "quant": stats(times["quant"]),
            "paired_delta_mean_s": statistics.fmean(deltas),
            "paired_delta_median_s": statistics.median(deltas),
            "paired_delta_ci95_s": [lo, hi],
            "quant_faster_samples": sum(1 for d in deltas if d > 0),
            "speedup_mean": statistics.fmean(times["bf16"]) / statistics.fmean(times["quant"]),
            "speedup_p50": statistics.median(times["bf16"]) / statistics.median(times["quant"]),
        },
        "greedy_token_agreement": {
            "samples": tokens_agree,
            "fraction": tokens_agree / len(samples),
            "note": "final greedy token after the timed window; the acceptance tool owns correctness",
        },
        "per_sample_delta_s": deltas,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["timing"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
