"""Build the 10-language GPTQ calibration + acceptance manifest.

Step 1 of the GPTQ-on-head task: a fixed, auditable, licence-clean audio set
for the ten qwen3-asr prompt languages (Chinese, English, Japanese, Korean,
Spanish, French, German, Russian, Thai, Vietnamese).

Sources
-------
* ``zh`` -- the local AISHELL-derived test mirror already prepared for the
  alignment work (``prepared/aishell-test``, 6920 wav, 16 kHz mono).
* ``en`` -- the local LibriSpeech ``test-clean`` mirror (2620 flac).
* the other eight -- **FLEURS** (``google/fleurs``, CC-BY-4.0), ``dev`` split
  for calibration and ``test`` split for acceptance.  FLEURS is the brief's
  documented fallback: Common Voice 17.0's HuggingFace repository publishes no
  audio (its tree holds only ``README.md``/``.gitattributes``; the CC0 audio
  lives behind Mozilla's own gated site), so it cannot be fetched
  programmatically.  FLEURS is per-language, permissive and small.

Both FLEURS splits are used on purpose: it keeps calibration and acceptance
disjoint **by construction** for those eight languages rather than by
subsampling one pool, which is the stronger guarantee.

Selection
---------
Per language ``N_CALIB`` calibration + ``N_ACCEPT`` acceptance + ``N_SPARE``
spare utterances.  Candidates are filtered to 16 kHz mono, 3-15 s, and picked
round-robin over speaker groups so no single speaker dominates; the eight
FLEURS languages use the ``client_id`` column as the speaker key.

Every selected file is opened with ``soundfile`` and its header checked before
it enters the manifest, and the calibration/acceptance audio paths are asserted
disjoint.  Nothing is copied or re-encoded: the manifest points at the verified
original files.

Usage::

    python3 tools/build_calib_manifest.py --root /Volumes/1T/funasr-mlx-models/calib \
        --alignment-root /Volumes/2T/funasr-mlx-alignment --out calib-manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

#: qwen3-asr prompt language -> (FLEURS config, manifest language tag)
FLEURS_LANGUAGES = {
    "ja": "ja_jp",
    "ko": "ko_kr",
    "es": "es_419",
    "fr": "fr_fr",
    "de": "de_de",
    "ru": "ru_ru",
    "th": "th_th",
    "vi": "vi_vn",
}

#: The ten languages, in the brief's order.
LANGUAGES = ("zh", "en", "ja", "ko", "es", "fr", "de", "ru", "th", "vi")

#: Human-readable source label per language, recorded in the manifest.
SOURCES = {
    "zh": "aishell-test-mirror",
    "en": "librispeech-test-clean",
    "ja": "fleurs-dev+test",
    "ko": "fleurs-dev+test",
    "es": "fleurs-dev+test",
    "fr": "fleurs-dev+test",
    "de": "fleurs-dev+test",
    "ru": "fleurs-dev+test",
    "th": "fleurs-dev+test",
    "vi": "fleurs-dev+test",
}

N_CALIB = 48
N_ACCEPT = 8
N_SPARE = 4

MIN_SECONDS = 3.0
MAX_SECONDS = 15.0
SAMPLE_RATE = 16000

#: The qwen3-asr prompt language name per tag -- the acceptance canary string.
PROMPT_LANGUAGE = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            value.update(chunk)
    return value.hexdigest()


def audio_info(path: Path):
    """Verified ``(seconds, sample_rate, channels, subtype)`` or ``None``."""
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception:
        return None
    return (float(info.duration), int(info.samplerate), int(info.channels), info.subtype)


def eligible(path: Path) -> tuple[float, int, int, str] | None:
    info = audio_info(path)
    if info is None:
        return None
    seconds, rate, channels, subtype = info
    if rate != SAMPLE_RATE or channels != 1:
        return None
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        return None
    return info


def round_robin(candidates: list[dict], count: int) -> list[dict]:
    """Pick ``count`` entries spreading across speaker groups.

    Candidates are bucketed by ``speaker`` (round-robin over the buckets, one
    per bucket per pass), so a talker-heavy corpus cannot fill the pool from a
    handful of voices.  Deterministic: the input order decides ties.
    """
    buckets: dict[str, list[dict]] = {}
    for entry in candidates:
        buckets.setdefault(entry["speaker"], []).append(entry)
    order = sorted(buckets)
    picked: list[dict] = []
    index = 0
    while len(picked) < count:
        progressed = False
        for speaker in order:
            bucket = buckets[speaker]
            if index < len(bucket):
                picked.append(bucket[index])
                progressed = True
                if len(picked) == count:
                    break
        if not progressed:
            break
        index += 1
    return picked


# --------------------------------------------------------------------------- #
# local corpora
# --------------------------------------------------------------------------- #
def local_candidates(corpus: list[dict], language: str) -> list[dict]:
    """Filter an already-prepared alignment corpus into manifest candidates.

    The AISHELL mirror carries no speaker column, so its speaker key is the
    stride bucket over file order -- which keeps the pick spread over the whole
    test set instead of the first N files.
    """
    rows = []
    for position, sample in enumerate(corpus):
        path = Path(sample["audio"])
        info = eligible(path)
        if info is None:
            continue
        speaker = sample.get("speaker_id") or ""
        if not speaker:
            speaker = "stride-%d" % (position % 64)
        rows.append(
            {
                "lang": language,
                "audio_path": str(path),
                "text": sample["transcript"],
                "speaker": speaker,
                "duration_seconds": info[0],
            }
        )
    return rows


def english_speaker(sample: dict) -> str:
    """LibriSpeech speaker id from the ``<speaker>/<chapter>/<file>`` path."""
    parts = Path(sample["audio"]).parts
    return "/".join(parts[-3:-1]) if len(parts) >= 3 else ""


# --------------------------------------------------------------------------- #
# FLEURS
# --------------------------------------------------------------------------- #
def fleurs_candidates(root: Path, language: str, split: str) -> list[dict]:
    """Read one FLEURS split's tsv and keep the rows whose audio verifies.

    Each split is extracted into its own directory (``fleurs/`` for ``dev``,
    ``fleurs-test/`` for ``test``) because both archives unpack their wavs
    flat, so one directory per split is what keeps calibration and acceptance
    disjoint by construction.
    """
    config = FLEURS_LANGUAGES[language]
    tsv = root / "raw" / f"fleurs-{config}-{split}.tsv"
    audio_dir = root / ("fleurs" if split == "dev" else "fleurs-test") / config
    if not tsv.is_file():
        raise SystemExit(f"Missing FLEURS transcript file: {tsv}")
    rows = []
    with tsv.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) < 3:
                continue
            client, name, text = row[0], row[1], row[2]
            path = audio_dir / name
            if not path.is_file():
                continue
            info = eligible(path)
            if info is None:
                continue
            rows.append(
                {
                    "lang": language,
                    "audio_path": str(path),
                    "text": text.strip(),
                    "speaker": client or name,
                    "duration_seconds": info[0],
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/Volumes/2T/funasr-mlx-alignment/calib"))
    parser.add_argument(
        "--alignment-root", type=Path, default=Path("/Volumes/2T/funasr-mlx-alignment")
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--calib", type=int, default=N_CALIB)
    parser.add_argument("--accept", type=int, default=N_ACCEPT)
    parser.add_argument("--spare", type=int, default=N_SPARE)
    args = parser.parse_args()

    root = args.root
    alignment = args.alignment_root
    out = args.out or (root / "calib-manifest.json")
    need = args.calib + args.accept + args.spare

    samples: list[dict] = []
    report: dict[str, dict] = {}

    # ---- Chinese / English: one local pool, sliced deterministically -------
    for language, corpus_name in (("zh", "corpus-chinese.json"), ("en", "corpus-english.json")):
        corpus = json.loads((alignment / "prepared" / corpus_name).read_text())
        rows = local_candidates(corpus, language)
        if language == "en":
            for row, sample in zip(rows, corpus):
                if row["audio_path"] == str(Path(sample["audio"])):
                    row["speaker"] = english_speaker(sample) or row["speaker"]
        if len(rows) < need:
            raise SystemExit(f"{language}: only {len(rows)} eligible utterances, need {need}")
        picked = round_robin(rows, need)
        for offset, row in enumerate(picked):
            row["split"] = (
                "calibration"
                if offset < args.calib
                else ("acceptance" if offset < args.calib + args.accept else "spare")
            )
            samples.append(row)
        report[language] = {
            "source": SOURCES[language],
            "pool": len(rows),
            "pool_speakers": len({row["speaker"] for row in rows}),
            "selected": len(picked),
        }

    # ---- the eight FLEURS languages: dev = calibration, test = acceptance --
    for language in FLEURS_LANGUAGES:
        dev = fleurs_candidates(root, language, "dev")
        test = fleurs_candidates(root, language, "test")
        if len(dev) < args.calib + args.spare:
            raise SystemExit(
                f"{language}: only {len(dev)} eligible dev utterances, need "
                f"{args.calib + args.spare}"
            )
        if len(test) < args.accept:
            raise SystemExit(f"{language}: only {len(test)} eligible test utterances")
        picked = round_robin(dev, args.calib + args.spare)
        for offset, row in enumerate(picked):
            row["split"] = "calibration" if offset < args.calib else "spare"
            samples.append(row)
        for row in round_robin(test, args.accept):
            row["split"] = "acceptance"
            samples.append(row)
        report[language] = {
            "source": SOURCES[language],
            "pool": len(dev) + len(test),
            "pool_speakers": len({row["speaker"] for row in dev + test}),
            "selected": args.calib + args.spare + args.accept,
            "dev_pool": len(dev),
            "test_pool": len(test),
        }

    # ---- verify, digest, and assert the split is disjoint ------------------
    seen: dict[str, str] = {}
    for sample in samples:
        path = Path(sample["audio_path"])
        info = eligible(path)
        if info is None:
            raise SystemExit(f"Selected audio failed verification: {path}")
        sample["duration_seconds"] = info[0]
        sample["sample_rate"] = info[1]
        sample["sha256"] = digest(path)
        previous = seen.setdefault(sample["audio_path"], sample["split"])
        if previous != sample["split"]:
            raise SystemExit(f"Audio appears in two splits: {sample['audio_path']}")
        sample["prompt_language"] = PROMPT_LANGUAGE[sample["lang"]]

    by_split: dict[str, list[dict]] = {}
    for sample in samples:
        by_split.setdefault(sample["split"], []).append(sample)
    calib_paths = {s["audio_path"] for s in by_split.get("calibration", [])}
    accept_paths = {s["audio_path"] for s in by_split.get("acceptance", [])}
    if calib_paths & accept_paths:
        raise SystemExit("Calibration and acceptance sets overlap")

    for language in LANGUAGES:
        entry = report[language]
        for split, expected in (
            ("calibration", args.calib),
            ("acceptance", args.accept),
            ("spare", args.spare),
        ):
            got = sum(1 for s in samples if s["lang"] == language and s["split"] == split)
            entry[f"{split}_count"] = got
            if got != expected:
                raise SystemExit(f"{language}/{split}: {got} samples, expected {expected}")

    total_seconds = sum(s["duration_seconds"] for s in samples)
    manifest = {
        "version": 1,
        "purpose": "GPTQ-on-head W4A16 calibration (48/lang) and 10-language acceptance (8/lang)",
        "languages": list(LANGUAGES),
        "prompt_language": PROMPT_LANGUAGE,
        "splits": {
            "calibration": args.calib,
            "acceptance": args.accept,
            "spare": args.spare,
        },
        "audio_contract": {
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "min_seconds": MIN_SECONDS,
            "max_seconds": MAX_SECONDS,
            "verified_with": "soundfile",
        },
        "sources": {
            "zh": "aishell-derived-test-mirror (local, 16 kHz wav)",
            "en": "librispeech-test-clean (local, flac)",
            "ja/ko/es/fr/de/ru/th/vi": (
                "google/fleurs CC-BY-4.0: dev split = calibration, test split = acceptance"
            ),
        },
        "totals": {
            "samples": len(samples),
            "calibration": len(calib_paths),
            "acceptance": len(accept_paths),
            "audio_seconds": round(total_seconds, 1),
        },
        "per_language": report,
        "samples": samples,
    }
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out}")
    print(json.dumps(manifest["totals"], indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
