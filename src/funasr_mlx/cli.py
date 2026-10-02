"""Offline CLI. Model and device imports happen only in a guarded worker."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from .checkpoints import SUPPORTED_MODELS, info_json, inspect_checkpoint
from .memory import GB, MB, MemoryPolicy, guarded_child_active, run_guarded


def _parser():
    parser = argparse.ArgumentParser(prog="funasr-mlx")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--memory-gb", type=int, choices=[1, 2, 3, 4], default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect safetensors without loading weights")
    inspect.add_argument("checkpoint", type=Path)
    convert = commands.add_parser(
        "convert", help="Convert a local torch ZIP checkpoint in bounded shards"
    )
    convert.add_argument("source", type=Path)
    convert.add_argument("destination", type=Path)
    convert.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    convert.add_argument("--config", type=Path)
    convert.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    convert.add_argument("--prefix", default="")
    features = commands.add_parser("features", help="Extract bounded MLX features to a local NPZ")
    features.add_argument("audio", type=Path)
    features.add_argument("output", type=Path)
    features.add_argument("--kind", choices=["kaldi", "whisper"], default="kaldi")
    features.add_argument("--config", type=Path)
    features.add_argument("--cmvn", type=Path)
    infer = commands.add_parser(
        "infer", help="Run one local native model, returning structured results"
    )
    infer.add_argument("checkpoint", type=Path)
    infer.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    infer.add_argument("--config", type=Path)
    infer.add_argument("--audio", type=Path)
    infer.add_argument("--text")
    infer.add_argument("--tokens", type=Path)
    infer.add_argument("--prompt-tokens", help="Whisper prefix token ids, comma-separated")
    infer.add_argument("--eot-token", type=int, default=50257)
    infer.add_argument("--max-tokens", type=int, default=128)
    return parser


def _infer(args):
    import mlx.core as mx
    from .frontend import FbankConfig, WavFrontend, read_audio, whisper_log_mel
    from .pipeline import load_model

    model, config = load_model(args.checkpoint, family=args.model, config=args.config)
    if args.model == "ct-punc":
        if args.text is None or args.tokens is None:
            raise ValueError("ct-punc needs --text and the matching --tokens vocabulary")
        from .models.ct_punc import CharVocab, CtPuncConfig, CtPuncPunctuator

        vocabulary = CharVocab.from_file(args.tokens)
        result = CtPuncPunctuator(model, vocabulary, CtPuncConfig.from_dict(config)).punctuate(
            args.text
        )
        return {"text": result.text, "punctuation_ids": result.punc_array}
    if args.audio is None:
        raise ValueError("Audio models require --audio")
    if args.model == "emotion2vec":
        from .models.emotion2vec import preprocess_waveform

        labels = model.config.labels
        if not labels or all(label.startswith("unuse") for label in labels):
            raise ValueError("emotion2vec inference needs the checkpoint's exact labels")
        samples = read_audio(args.audio, max_seconds=min(30.0, model.config.max_samples / 16000))
        waveform = preprocess_waveform(mx.array(samples), normalize=model.config.normalize)
        result = model.emotion_scores(waveform)
        mx.eval(result.feats)
        return {
            "key": args.audio.stem,
            "labels": list(result.labels),
            "scores": list(result.scores),
            "embedding": result.feats.tolist(),
        }
    samples = read_audio(args.audio)
    if args.model == "whisper":
        if args.prompt_tokens is None:
            raise ValueError("Whisper needs explicit tokenizer-correct --prompt-tokens")
        if not 1 <= args.max_tokens <= 448:
            raise ValueError("max_tokens must be in [1, 448]")
        from .models.whisper import GreedyDecodeConfig, greedy_decode

        prefix = [int(value) for value in args.prompt_tokens.split(",")]
        mel = whisper_log_mel(
            samples, n_mels=model.dims.n_mels, pad_to_samples=2 * model.dims.n_audio_ctx * 160
        )
        audio = model.embed_audio(mx.swapaxes(mel, 0, 1)[None])
        mx.eval(audio)
        result = greedy_decode(
            model,
            audio,
            mx.array([prefix]),
            GreedyDecodeConfig(eot_token_id=args.eot_token, max_tokens=args.max_tokens),
        )
        return {"token_ids": result.tokens, "average_logprobs": result.avg_logprobs}
    defaults = {"sensevoice": (7, 6), "paraformer": (7, 6), "fsmn-vad": (5, 1), "campplus": (1, 1)}
    m, n = defaults[args.model]
    options = dict(config.get("frontend_conf") or {})
    cmvn = options.pop("cmvn_file", None)
    options.setdefault("lfr_m", m)
    options.setdefault("lfr_n", n)
    if cmvn and not Path(cmvn).is_absolute():
        base = args.checkpoint if args.checkpoint.is_dir() else args.checkpoint.parent
        cmvn = base / cmvn
    frontend = WavFrontend(FbankConfig.from_dict(options), cmvn=cmvn)
    features = frontend(samples)
    if features.shape[0] == 0:
        return (
            {"token_ids": [[]]}
            if args.model in {"sensevoice", "paraformer"}
            else {"empty_audio": True}
        )
    lengths = mx.array([features.shape[0]], dtype=mx.int32)
    if args.model in {"sensevoice", "paraformer"}:
        return {"token_ids": model.greedy_decode(features[None], lengths)}
    if args.model == "campplus":
        features = features - mx.mean(features, axis=0, keepdims=True)
        embedding = model(features[None])
        mx.eval(embedding)
        return {"embedding": embedding.tolist()}
    cache = model.init_cache()
    needed = (
        features.shape[0] - 1
    ) * model.vad_opts.frame_shift_length + model.vad_opts.frame_sample_length
    if samples.size < needed:
        raise ValueError("VAD feature frames exceed the supplied waveform")
    # Snipped fbank frames do not consume a trailing incomplete frame.
    segments = model(
        features[None],
        mx.array(samples[:needed])[None],
        cache,
        is_final=True,
        is_streaming_input=False,
    )
    return {"segments_ms": segments}


def _execute(args):
    if args.command == "convert":
        from .checkpoints import convert_checkpoint
        from .pipeline import read_config

        info = convert_checkpoint(
            args.source,
            args.destination,
            model=args.model,
            dtype=args.dtype,
            prefix=args.prefix,
            config=read_config(args.config) if args.config else None,
        )
        print(info_json(info))
    elif args.command == "features":
        import mlx.core as mx
        import numpy as np
        from .frontend import FbankConfig, WavFrontend, read_audio, whisper_log_mel
        from .pipeline import read_config

        if args.kind == "whisper":
            features = whisper_log_mel(read_audio(args.audio))
        else:
            options = read_config(args.config) if args.config else {}
            options = dict(options.get("frontend_conf", options))
            configured_cmvn = options.get("cmvn_file")
            if configured_cmvn and args.config and not Path(configured_cmvn).is_absolute():
                options["cmvn_file"] = str(args.config.parent / configured_cmvn)
            config = FbankConfig.from_dict(options)
            samples = read_audio(args.audio, sample_rate=config.fs, max_seconds=config.max_seconds)
            kwargs = {} if args.cmvn is None else {"cmvn": args.cmvn}
            features = WavFrontend(config, **kwargs)(samples)
        mx.eval(features)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as output:
            np.savez(output, features=np.asarray(features))
        print(json.dumps({"output": str(args.output), "shape": features.shape}))
    elif args.command == "infer":
        print(json.dumps(_infer(args), ensure_ascii=False))
    else:
        raise ValueError("Unsupported worker command")


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    parser = _parser()
    args = parser.parse_args(arguments)
    if args.command == "inspect":
        print(info_json(inspect_checkpoint(args.checkpoint)))
        return 0
    if "FUNASR_MLX_MEMORY_POLICY" in os.environ:
        if not guarded_child_active():
            raise RuntimeError("Memory-policy environment is not an authenticated guarded child")
        _execute(args)
        return 0
    policy = MemoryPolicy(process_limit=args.memory_gb * GB, mlx_limit=args.memory_gb * 625 * MB)
    report = run_guarded(
        [sys.executable, "-m", "funasr_mlx.cli", *arguments],
        root=args.root,
        policy=policy,
        report_path=".tmp/last-job-memory.json",
        configure_device=args.command != "convert",
    )
    print(json.dumps({"memory": asdict(report)}), file=sys.stderr)
    return report.returncode


if __name__ == "__main__":
    raise SystemExit(main())
