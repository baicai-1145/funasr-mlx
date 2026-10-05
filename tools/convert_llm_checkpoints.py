"""Convert the five LLM-ASR source checkpoints into the repo native format.

Run under the guarded launcher (see ``tools/prepare_large_alignment.py`` for the
pattern); conversion refuses to run outside a guarded child.

    python3 -m funasr_mlx.runtime --root . --report .tmp/x.json -- \
        python3 tools/convert_llm_checkpoints.py --family qwen3-asr --dtype bfloat16 \
        --source /path/to/source --destination /path/to/native

Each family declares its canonical output dtype (all five sources are
bfloat16, so ``bfloat16`` keeps every weight bit-exact; ``float16`` is available
where a smaller artifact matters more than exactness) and its source layout via
``funasr_mlx.checkpoints.SOURCE_RULES``. Tokenizer assets are copied alongside
the weights so the converted directory is self-contained for
``AutoTokenizer.from_pretrained``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from funasr_mlx.checkpoints import (
    IDENTITY_SOURCE_RULE,
    TORCH_SOURCE_FAMILIES,
    convert_checkpoint,
    info_json,
)

#: job name -> (native model family, canonical dtype). ``qwen3-asr-1.7b`` is the
#: same architecture as ``qwen3-asr`` at hidden 2048. ``qwen3-asr-0.6b`` is an
#: alias for the same architecture: the published native directory is named for
#: the model size, so the job name has to match it.
FAMILIES = {
    "fun-asr-nano": ("fun-asr-nano", "bfloat16"),
    "qwen3-asr": ("qwen3-asr", "bfloat16"),
    "qwen3-asr-0.6b": ("qwen3-asr", "bfloat16"),
    "qwen3-asr-1.7b": ("qwen3-asr", "bfloat16"),
    "glm-asr-nano": ("glm-asr-nano", "bfloat16"),
    "moss-transcribe-diarize": ("moss-transcribe-diarize", "bfloat16"),
}

#: Tokenizer/prompt assets copied so the native directory stands alone.
TOKENIZER_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "added_tokens.json",
    "special_tokens_map.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
)


def native_config(job: str, source: Path) -> dict:
    """The config slice ``create_model`` needs for ``job``.

    Each family's model class reads a different part of the upstream config, so
    the native ``config.json`` carries exactly that part.
    """
    if job == "fun-asr-nano":
        import yaml

        raw = yaml.safe_load((source / "config.yaml").read_text())
        llm = raw.get("llm_conf") or {}
        return {
            "model": "FunASRNano",
            "audio_encoder_conf": raw.get("audio_encoder_conf") or {},
            "audio_adaptor_conf": raw.get("audio_adaptor_conf") or {},
            "input_size": 560,
            "llm_conf": llm if "args" in llm else {"args": llm},
        }
    full = json.loads((source / "config.json").read_text())
    if job in {"qwen3-asr", "qwen3-asr-0.6b", "qwen3-asr-1.7b"}:
        return {"thinker_config": full["thinker_config"]}
    if job == "glm-asr-nano":
        return {
            "audio_config": full["audio_config"],
            "text_config": full["text_config"],
            "audio_token_id": full.get("audio_token_id", 59260),
        }
    if job == "moss-transcribe-diarize":
        return {
            "audio_config": full["audio_config"],
            "text_config": full["text_config"],
            "audio_merge_size": full.get("audio_merge_size", 4),
            "audio_token_id": full.get("audio_token_id", 151671),
        }
    raise ValueError(f"No config recipe for {job}")


#: Module-path prefixes of each family's **decoder** stack. A W4A16 conversion
#: quantizes exactly these by default: at bs=1 decode streams the decoder
#: weights once per step, so this is where 4 bits buys time as well as memory.
#: The audio towers are offered as a documented second target for families that
#: need the extra memory (they run once per utterance, so they buy no decode
#: time). ``embed_tokens``/``lm_head`` are never quantized -- the tensor-name
#: predicate in :class:`QuantizationSpec` excludes them by construction.
DECODER_QUANT_TARGETS = {
    "fun-asr-nano": ("llm.model.layers.",),
    "qwen3-asr": ("thinker.model.layers.",),
    "qwen3-asr-0.6b": ("thinker.model.layers.",),
    "qwen3-asr-1.7b": ("thinker.model.layers.",),
    "glm-asr-nano": ("language_model.model.layers.",),
    "moss-transcribe-diarize": ("language_model.model.layers.",),
}

#: Audio-tower prefixes, for the (memory-only) full-stack variant.
AUDIO_QUANT_TARGETS = {
    "fun-asr-nano": ("audio_encoder.", "audio_adaptor."),
    "qwen3-asr": ("audio_tower.",),
    "qwen3-asr-0.6b": ("audio_tower.",),
    "qwen3-asr-1.7b": ("audio_tower.",),
    "glm-asr-nano": ("audio_tower.",),
    "moss-transcribe-diarize": ("whisper_encoder.", "vq_adaptor."),
}


def quantization_config(job: str, scope: str, *, bits: int = 4, group_size: int = 64) -> dict:
    """The ``quantization`` section written into the native ``config.json``.

    ``scope`` is ``decoder`` (the default: decode-time win) or ``all``
    (decoder + audio tower: also the memory win for the audio stack).
    """
    if scope not in ("decoder", "all"):
        raise ValueError(f"Unknown quantization scope: {scope}")
    targets = list(DECODER_QUANT_TARGETS[job])
    if scope == "all":
        targets += list(AUDIO_QUANT_TARGETS[job])
    return {"bits": bits, "group_size": group_size, "mode": "affine", "targets": targets}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=None)
    parser.add_argument("--tensor-limit", type=int, default=384_000_000)
    parser.add_argument(
        "--quantize",
        choices=("none", "decoder", "all"),
        default="none",
        help="W4A16 group quantization: decoder only (default scope) or the whole stack",
    )
    parser.add_argument("--quant-bits", type=int, default=4)
    parser.add_argument("--quant-group-size", type=int, default=64)
    parser.add_argument(
        "--native-source",
        action="store_true",
        help="source is already in the native layout (identity key rule)",
    )
    args = parser.parse_args()

    model, default_dtype = FAMILIES[args.family]
    dtype = args.dtype or default_dtype
    source = args.source
    if not source.exists():
        raise SystemExit(f"Source checkpoint not found: {source}")
    config = native_config(args.family, source)
    quantization = None
    if args.quantize != "none":
        quantization = quantization_config(
            args.family, args.quantize, bits=args.quant_bits, group_size=args.quant_group_size
        )
        config = dict(config)
        config["quantization"] = quantization
    info = convert_checkpoint(
        source,
        args.destination,
        model=model,
        dtype=dtype,
        config=config,
        tensor_limit=args.tensor_limit,
        source_rule=IDENTITY_SOURCE_RULE if args.native_source else model,
        # The embedding matrix alone exceeds the single-tensor buffer bound in
        # every family (622 MB in fun-asr-nano, 1244 MB in the 1.7B), so the
        # chunked writer is always required. Harmless for a torch source, whose
        # layout lives in convert_tensor.
        stream_large_tensors=True,
        quantization=quantization,
    )
    print(info_json(info), flush=True)
    if model not in TORCH_SOURCE_FAMILIES or args.native_source:
        copied = []
        for name in TOKENIZER_FILES:
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, args.destination / name)
                copied.append(name)
        print(f"tokenizer assets copied: {copied}", flush=True)
    else:
        print(
            "note: copy the tokenizer assets the runtime needs into "
            f"{args.destination} (the torch source keeps them elsewhere)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
