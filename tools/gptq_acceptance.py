"""Ten-language acceptance for a W4A16 (GPTQ) head checkpoint.

The acceptance criterion, from the task brief:

1. **Semantic equality** of the transcript against the bf16 baseline.  For CJK
   and Thai, punctuation and whitespace differences are tolerated (the tokenizer
   and the model's own formatting vary); the comparison strips them.
2. **Token-level edit distance <= 2%** of the greedy token sequence.  Measured
   on the *answer span* (up to and including the first end-of-turn token): a
   quantized embedding may shift the post-answer hallucination that task-5
   documented as genuine bf16 behaviour, and that is not an ASR failure.
3. The qwen3 **``language X`` guide span preserved verbatim** in every one of
   the 80 samples -- the canary for language-direction damage, since it is
   emitted from the first decoded tokens before any audio content.

Both arms run in ONE process on the SAME audio, alternating per sample, which is
the only valid paired protocol given the thermal drift documented in the
project's own measurement notes.  Each arm is exercised through
``pipeline.load_model`` and the family's own forward path.

Usage (inside the guard)::

    python3 tools/gptq_acceptance.py --family qwen3-asr-0.6b \
        --quant-dir /path/to/native-llm/qwen3-asr-0.6b-w4a16-gptq \
        --out .tmp/gptq/acceptance-qwen3-asr-0.6b.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: End-of-turn / end-of-text tokens per family, used to cut the answer span.
EOS_TOKENS = (151645, 151643, 59262, 151671, 0)

#: job name -> model class family (what ``load_model``/``create_model`` dispatch on)
MODEL_FAMILY = {
    "fun-asr-nano": "fun-asr-nano",
    "qwen3-asr-0.6b": "qwen3-asr",
    "qwen3-asr-1.7b": "qwen3-asr",
    "glm-asr-nano": "glm-asr-nano",
    "moss-transcribe-diarize": "moss-transcribe-diarize",
}

#: CJK + Thai scripts, where the reference transcripts carry no reliable
#: word spacing, so punctuation and whitespace are stripped before comparison.
SCRIPT_STRIP_LANGUAGES = {"zh", "ja", "ko", "th"}

_PUNCTUATION = re.compile(r"[\s。，、！？；：「」『』（）【】《》…—～·,.!?;:'\"()\[\]{}<>|/\\-]+")


#: The guide prefix qwen3-asr emits before the transcript.  It is checked on its
#: own by :func:`guide_span_ok` and must NOT be part of the transcript
#: comparison -- otherwise a perfectly correct transcript never equals the
#: reference, which is what made ``baseline_matches_expected`` read 0/80 while
#: the baseline transcripts were visibly correct.
_GUIDE = re.compile(r"^\s*language\s+[A-Za-z]+\s*")


def normalize(text: str, language: str) -> str:
    """Comparison form of a transcript: guide prefix removed, punctuation stripped."""
    if text is None:
        return ""
    text = text.replace("<asr_text>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "")
    text = _GUIDE.sub("", text)
    text = text.casefold()
    if language in SCRIPT_STRIP_LANGUAGES:
        return _PUNCTUATION.sub("", text)
    return _PUNCTUATION.sub(" ", text).strip()


def answer_span(tokens: list[int]) -> list[int]:
    """Tokens up to and including the first end-of-turn token."""
    for index, token in enumerate(tokens):
        if token in EOS_TOKENS:
            return tokens[: index + 1]
    return tokens


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two token sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (left != right),  # substitution
                )
            )
        previous = current
    return previous[-1]


def token_edit_distance(tokens_a: list[int], tokens_b: list[int]) -> tuple[int, float]:
    """``(distance, distance / len(baseline))`` over the token sequences."""
    distance = edit_distance(tokens_a, tokens_b)
    denominator = max(len(tokens_b), 1)
    return distance, distance / denominator


def guide_span_ok(transcript: str, prompt_language: str) -> bool:
    """Whether the ``language X`` guide span is present verbatim.

    qwen3-asr emits ``language {Name}<asr_text>`` as the first decoded tokens.
    A damaged head shows up here first, so it is checked on its own.
    """
    if transcript is None:
        return False
    return f"language {prompt_language}" in transcript


# --------------------------------------------------------------------------- #
# per-family drivers (mirroring tools/gptq_calibrate.py's prefill builders)
# --------------------------------------------------------------------------- #
def build_driver(
    family: str, model, config, directory: Path, *, tokenizer_dir: Path, decode_steps: int = 96
):
    """Return ``decode(sample) -> (tokens, transcript)`` for one arm.

    ``directory`` is the arm's checkpoint (where the weights come from) and
    ``tokenizer_dir`` is where the tokenizer lives.  They are separate because a
    converted checkpoint carries weights and ``config.json`` only -- the
    converter does not copy the tokenizer -- while the two arms of an A/B are by
    construction the same model and therefore the same tokenizer.  Reading the
    tokenizer from the bf16 source keeps the two arms on one identical tokenizer
    rather than letting a missing file become a silent difference.
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
        # The mel feeds the audio TOWER, so follow the tower's dtype.  Reading
        # ``embed_tokens.weight.dtype`` was wrong twice over: it is the decoder's
        # dtype, and for a quantized embedding it is the packed-code uint32, so
        # ``mel.astype(uint32)`` destroyed the audio before the tower ever saw
        # it.  This produced correct-looking runs whose transcripts were garbage.
        dtype = model._tower_dtype()

        def decode(sample):
            raw, _ = sf.read(sample["audio_path"], dtype="float32")
            mel = log_mel_spectrogram(mx.array(raw[None, :]), n_mels=128)
            mel = mx.transpose(mel[0], (1, 0)).astype(dtype)
            mx.eval(mel)
            embeds = model.encode_audio(mel)
            mx.eval(embeds)
            n = int(embeds.shape[1])
            ids = mx.array([prefix_ids + [pad_id] * n + tail_ids], dtype=mx.int32)
            caches = [KVCache() for _ in range(n_layers)]
            logits = model.forward_with_audio(ids, embeds, n, caches)
            mx.eval(logits)
            token = int(mx.argmax(logits[0, -1]).item())
            tokens = [token]
            for _ in range(decode_steps):
                lg = model.thinker(inputs=mx.array([[token]], dtype=mx.int32), cache=caches)
                mx.eval(lg)
                token = int(mx.argmax(lg[0, -1]).item())
                tokens.append(token)
                if token in (151643, 151645):
                    break
            return tokens, tok.decode(tokens).replace("<asr_text>", "")

        return decode

    raise ValueError(f"No acceptance driver for {family}")


def load_arm(family: str, directory: Path):
    """Load one arm and return ``(model, structured_config)``.

    ``load_model`` returns the raw ``config.json`` mapping; the drivers need the
    parsed config (``config.audio_pad_token_id``, ``config.text``), so it is
    normalized through the same reader ``create_model`` uses.
    """
    from funasr_mlx.pipeline import load_model

    model, config = load_model(directory, family=family)
    if family == "qwen3-asr":
        existing = getattr(model, "config", None)
        if existing is not None and hasattr(existing, "audio_pad_token_id"):
            return model, existing
        from funasr_mlx.models.qwen3_asr_thinker import qwen3asr_config_from_dict

        return model, qwen3asr_config_from_dict(config)
    return model, getattr(model, "config", config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=sorted(MODEL_FAMILY))
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--quant-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-token-edit-rate", type=float, default=0.02)
    parser.add_argument("--languages", default="zh,en,ja,ko,es,fr,de,ru,th,vi")
    parser.add_argument("--per-language", type=int, default=8)
    args = parser.parse_args()

    from funasr_mlx.memory import guarded_child_active

    if not guarded_child_active():
        raise RuntimeError("Acceptance must run under the memory guard")

    languages = [part for part in args.languages.split(",") if part]
    model_family = MODEL_FAMILY[args.family]
    manifest = json.loads(args.manifest.read_text())
    samples = []
    for language in languages:
        rows = [
            sample
            for sample in manifest["samples"]
            if sample["lang"] == language and sample["split"] == "acceptance"
        ]
        if len(rows) < args.per_language:
            raise SystemExit(f"{language}: only {len(rows)} acceptance samples")
        samples.extend(rows[: args.per_language])
    print(f"acceptance samples: {len(samples)} across {len(languages)} languages", flush=True)

    results = []
    # Load both arms once; the pairing is per sample (bf16 then quant, the
    # order alternated) so thermal drift hits both arms equally.
    bf16_model, bf16_config = load_arm(model_family, args.bf16_dir)
    quant_model, quant_config = load_arm(model_family, args.quant_dir)
    bf16_decode = build_driver(
        model_family, bf16_model, bf16_config, args.bf16_dir, tokenizer_dir=args.bf16_dir
    )
    quant_decode = build_driver(
        model_family, quant_model, quant_config, args.quant_dir, tokenizer_dir=args.bf16_dir
    )

    for index, sample in enumerate(samples):
        language = sample["lang"]
        prompt_language = sample["prompt_language"]
        order = (bf16_decode, quant_decode) if index % 2 == 0 else (quant_decode, bf16_decode)
        outputs = {}
        for position, decode in enumerate(order):
            arm = "bf16" if decode is bf16_decode else "quant"
            started = time.perf_counter()
            tokens, transcript = decode(sample)
            outputs[arm] = {
                "tokens": tokens,
                "transcript": transcript,
                "seconds": time.perf_counter() - started,
                "answer_span": answer_span(tokens),
            }
        base, quant = outputs["bf16"], outputs["quant"]
        distance, rate = token_edit_distance(quant["answer_span"], base["answer_span"])
        full_distance, full_rate = token_edit_distance(quant["tokens"], base["tokens"])
        row = {
            "language": language,
            "prompt_language": prompt_language,
            "audio_path": sample["audio_path"],
            "expected_transcript": sample["text"],
            "baseline_transcript": base["transcript"],
            "quant_transcript": quant["transcript"],
            "baseline_tokens": base["tokens"],
            "quant_tokens": quant["tokens"],
            "span_token_edit_distance": distance,
            "span_token_edit_rate": rate,
            "full_token_edit_distance": full_distance,
            "full_token_edit_rate": full_rate,
            "span_identical": quant["answer_span"] == base["answer_span"],
            "semantic_equal": normalize(quant["transcript"], language)
            == normalize(base["transcript"], language),
            "expected_matches_baseline": normalize(base["transcript"], language)
            == normalize(sample["text"], language),
            "guide_span_ok": guide_span_ok(quant["transcript"], prompt_language),
            "baseline_guide_span_ok": guide_span_ok(base["transcript"], prompt_language),
            "seconds": {"bf16": base["seconds"], "quant": quant["seconds"]},
        }
        row["pass"] = bool(
            row["semantic_equal"]
            and rate <= args.max_token_edit_rate
            and row["guide_span_ok"]
        )
        results.append(row)
        print(
            f"[{index + 1}/{len(samples)}] {language:3s} pass={row['pass']} "
            f"semantic={row['semantic_equal']} span_rate={rate:.4f} "
            f"guide={row['guide_span_ok']}",
            flush=True,
        )

    per_language = {}
    for language in languages:
        rows = [row for row in results if row["language"] == language]
        per_language[language] = {
            "samples": len(rows),
            "passed": sum(row["pass"] for row in rows),
            "semantic_equal": sum(row["semantic_equal"] for row in rows),
            "guide_span_ok": sum(row["guide_span_ok"] for row in rows),
            "max_span_token_edit_rate": max(row["span_token_edit_rate"] for row in rows),
            "mean_span_token_edit_rate": sum(row["span_token_edit_rate"] for row in rows)
            / len(rows),
            "span_identical": sum(row["span_identical"] for row in rows),
        }
    summary = {
        "family": args.family,
        "bf16_dir": str(args.bf16_dir),
        "quant_dir": str(args.quant_dir),
        "criterion": {
            "semantic_equal": "punctuation/whitespace-insensitive transcript match vs bf16",
            "max_span_token_edit_rate": args.max_token_edit_rate,
            "guide_span": "the 'language X' prefix preserved verbatim",
        },
        "totals": {
            "samples": len(results),
            "passed": sum(row["pass"] for row in results),
            "semantic_equal": sum(row["semantic_equal"] for row in results),
            "guide_span_ok": sum(row["guide_span_ok"] for row in results),
            "span_identical": sum(row["span_identical"] for row in results),
            "mean_span_token_edit_rate": sum(row["span_token_edit_rate"] for row in results)
            / len(results),
            "max_span_token_edit_rate": max(row["span_token_edit_rate"] for row in results),
            "baseline_matches_expected": sum(row["expected_matches_baseline"] for row in results),
        },
        "per_language": per_language,
        "results": results,
    }
    summary["accepted"] = summary["totals"]["passed"] == len(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"totals": summary["totals"], "accepted": summary["accepted"]}, indent=2))
    print(json.dumps(per_language, indent=2))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
