#!/usr/bin/env python3
"""Build the 12 publish-ready model repositories.

7 classic fp16 conversions (single-variant repos) + 5 LLM-ASR repos that each
carry BOTH the fp16 conversion and the final quantized checkpoint as
subdirectories (fp16/ and w4a16/).

Each staging dir gets: the safetensors + config, tokenizer files copied from
the source when present, an Apache-2.0 LICENSE + NOTICE (stating the MLX
conversion and, where applicable, quantization), and a README with the
measured acceptance/RTF numbers.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ALIGN = Path("/Volumes/2T/funasr-mlx-alignment")
NFP16 = ALIGN / "native-fp16"
NLLM = ALIGN / "native-llm"
STAGE = Path("/Volumes/2T/funasr-mlx-alignment/release-stage")

LICENSE_APACHE = "https://www.apache.org/licenses/LICENSE-2.0"

CLASSIC = {
    "sensevoice": "sensevoice",
    "fsmn-vad": "fsmn-vad",
    "campplus": "campplus",
    "paraformer": "paraformer",
    "whisper": "whisper-tiny",
    "whisper-large-v3": "whisper-large-v3",
    "whisper-large-v3-turbo": "whisper-large-v3-turbo",
    "ct-punc": "ct-punc",
    "emotion2vec": "emotion2vec",
}

LLM = [
    ("qwen3-asr-0.6b-mlx",
     [("fp16", NLLM / "qwen3-asr-0.6b", "qwen3-asr-0.6b"),
      ("w4a16", NLLM / "qwen3-asr-0.6b-w4a16-awq-w8tower-g32", "qwen3-asr-0.6b")],
     "Balanced: w4 stack+head (g32 tied head), w8 audio tower. "
     "RTF 0.079 (12.6x realtime, M4 Air); zh-200 CER vs bf16 1.1%."),
    ("qwen3-asr-1.7b-mlx",
     [("fp16", NLLM / "qwen3-asr-1.7b", "qwen3-asr-1.7b"),
      ("w4a16", NLLM / "qwen3-asr-1.7b-w4a16-awq-all", "qwen3-asr-1.7b")],
     "All-scope w4 (g64 head). Decode 45.9->19.4 ms/step (2.37x); RTF 0.146 "
     "(6.9x realtime); spot CER 1.75% over 4 languages."),
    ("glm-asr-nano-mlx",
     [("fp16", NLLM / "glm-asr-nano", "glm-asr-nano"),
      ("w4a16", NLLM / "glm-asr-nano-mixed-v2", "glm-asr-nano")],
     "Mixed: tower w4 + decoder stack w8 + embed w4/g32 + lm_head w8 "
     "(untied head cannot survive w4). 1.62x decode; RTF 0.70."),
    ("moss-transcribe-diarize-mlx",
     [("fp16", NLLM / "moss-transcribe-diarize", "moss-transcribe-diarize"),
      ("w4a16", NLLM / "moss-w4a16-awq-g32", "moss-transcribe-diarize")],
     "All-scope w4 with g32 tied head. Timestamps: 86% within +/-0.2s of "
     "bf16; RTF 0.33 (0.26 with the speaker-only prompt)."),
    ("fun-asr-nano-mlx",
     [("fp16", NLLM / "fun-asr-nano", "fun-asr-nano"),
      ("w4a16", NLLM / "fun-asr-nano-w4a16-all", "fun-asr-nano")],
     "All-scope w4 (g64). RTF 0.104 (9.6x realtime); SenseVoice-small tower "
     "144 ms + Qwen3-0.6B-shaped decoder."),
]


def tokenizer_sources(name: str) -> list[Path]:
    """Tokenizer-ish files that ride along with the weights."""
    srcs = {
        "qwen3-asr-0.6b": [NLLM / "qwen3-asr-0.6b"],
        "qwen3-asr-1.7b": [NLLM / "qwen3-asr-1.7b"],
        "glm-asr-nano": [ALIGN / "models" / "glm-asr-nano", NLLM / "glm-asr-nano"],
        "moss-transcribe-diarize": [NLLM / "moss-transcribe-diarize"],
        "fun-asr-nano": [NLLM / "fun-asr-nano"],
    }
    out = []
    for d in srcs.get(name, []):
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file() and any(
                    k in f.name.lower()
                    for k in ("tokenizer", "vocab", "merges", "chat_template", "preprocessor", "special_tokens", "added_tokens", "generation_config")
                ):
                    out.append(f)
    return out


def write_readme(path: Path, title: str, body: str, layout: str) -> None:
    text = f"""---
license: apache-2.0
---

# {title}

MLX (Apple silicon) conversion from the original release. Runs with
[funasr-mlx](https://github.com/baicai-1145/funasr-mlx) (`pip install`-free
repo; see `src/funasr_mlx`).

{body}

## Layout

{layout}

## Notices

- Upstream weights © their respective authors (Apache-2.0); this repository
  is a numerical conversion and/or quantization derivative. See `NOTICE`.
- Conversion/quantization performed with funasr-mlx @ commit
  {subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=Path(__file__).parent.parent).stdout.strip()[:12]}.
"""
    path.write_text(text, encoding="utf-8")


def write_notice(path: Path, quantized: bool) -> None:
    q = ""
    if quantized:
        q = (
            "\nThe w4a16 variant additionally applies affine weight "
            "quantization (4-bit; group sizes and per-module scopes described "
            "in the README) on top of this conversion.\n"
        )
    path.write_text(
        f"""PRODUCT: MLX-format conversion of an upstream speech model
This product contains derivative works of weights licensed Apache-2.0 by
their upstream authors (Qwen/Alibaba, Zhipu AI (GLM-ASR), OpenMOSS, or
FunAudioLLM as applicable per repository).

Changes made: tensor layout conversion to MLX (renamed tensors, dtype
conversion fp16) so the model loads with the funasr-mlx runtime.{q}
The upstream LICENSE (Apache-2.0) continues to apply:
{LICENSE_APACHE}
""",
        encoding="utf-8",
    )


def stage_dir(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.iterdir()):
        if f.is_file() and not f.name.startswith(".") and f.suffix in (".json", ".safetensors", ".txt", ".jinja", ".yaml"):
            shutil.copy2(f, dst / f.name)
            n += 1
    return n


def main() -> int:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    total = 0
    for src_name, repo_name in CLASSIC.items():
        src = NFP16 / src_name
        dst = STAGE / repo_name
        n = stage_dir(src, dst)
        # license/readme
        write_notice(dst / "NOTICE", quantized=False)
        write_readme(
            dst / "README.md",
            f"{repo_name} (MLX fp16)",
            "Float16 MLX conversion; numerically equivalent to the upstream "
            "release within fp16 rounding.",
            "Flat: config + safetensors at the repository root.",
        )
        total += n
        print(f"[classic] {repo_name}: {n} files")
    for repo_name, variants, blurb in LLM:
        dst = STAGE / repo_name
        n = 0
        for sub, src, tok_name in variants:
            n += stage_dir(src, dst / sub)
            for tf in tokenizer_sources(tok_name):
                shutil.copy2(tf, dst / sub / tf.name)
        write_notice(dst / "NOTICE", quantized=True)
        layout = "\n".join(
            f"- `{sub}/` — {'fp16 conversion' if sub == 'fp16' else 'quantized (w4a16)'} "
            f"(tokenizer included)" for sub, _, _ in variants
        )
        write_readme(
            dst / "README.md",
            repo_name.replace("-mlx", " (MLX)"),
            blurb,
            layout,
        )
        total += n
        print(f"[llm] {repo_name}: {n} files")
    print(f"STAGED {total} files into {STAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
