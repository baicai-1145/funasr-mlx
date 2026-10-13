"""Build a 200-sample zh (AISHELL mirror) extended acceptance manifest.

Disjoint from every zh row the 10-language manifest already uses (60 rows),
same eligibility (16 kHz mono, 3-15 s -- MIN/MAX from the corpus builder) and
the same round-robin speaker spread.  Used to give the zh CER/WER number
statistical force and comparability with AISHELL-1 test conventions.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_calib_manifest import MIN_SECONDS, MAX_SECONDS, round_robin  # noqa: E402

ALIGNMENT = Path("/Volumes/2T/funasr-mlx-alignment")
OUT = ALIGNMENT / "calib" / "zh-extended-manifest.json"
N = 200

manifest = json.loads((ALIGNMENT / "calib" / "calib-manifest.json").read_text())
used = {s["audio_path"] for s in manifest["samples"] if s["lang"] == "zh"}
corpus = json.loads((ALIGNMENT / "prepared" / "corpus-chinese.json").read_text())

rows = []
for position, sample in enumerate(corpus):
    path = Path(sample["audio"])
    if str(path) in used:
        continue
    seconds = sample.get("duration_seconds") or 0
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        continue
    speaker = sample.get("speaker_id") or f"stride-{position % 64}"
    rows.append(
        {
            "lang": "zh",
            "audio_path": str(path),
            "text": sample["transcript"],
            "speaker": speaker,
            "duration_seconds": seconds,
            "split": "acceptance",
        }
    )

print(f"eligible unused pool: {len(rows)}")
if len(rows) < N:
    raise SystemExit("pool too small")

picked = round_robin(rows, N)
from collections import Counter

print(f"picked {len(picked)}, speakers: {len({r['speaker'] for r in picked})}")
print("duration: min %.1fs mean %.1fs max %.1fs" % (
    min(r["duration_seconds"] for r in picked),
    sum(r["duration_seconds"] for r in picked) / len(picked),
    max(r["duration_seconds"] for r in picked),
))
OUT.write_text(json.dumps({"samples": picked}, ensure_ascii=False, indent=1) + "\n")
print(f"wrote {OUT}")
