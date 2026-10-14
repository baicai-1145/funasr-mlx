"""Calibrate a family's LM head and emit a GPTQ-quantized W4A16 checkpoint.

Two phases, each runnable on its own (the calibration is the expensive part and
the conversion is what has to be repeatable):

``--phase calibrate``
    Load the **bf16** native checkpoint, resolve the family's LM head, run the
    real prefill forward over the 48-per-language calibration split of
    ``calib-manifest.json`` with :class:`HeadActivationCollector` installed, and
    write the language-tagged ``H = 2 X^T X`` to an ``.npz``.  The bf16
    checkpoint is only read.

``--phase convert``
    Load that ``.npz``, build a :class:`~funasr_mlx.checkpoints.GptqWeight`, and
    convert the bf16 checkpoint into ``native-llm/<family>-w4a16-gptq/`` with the
    declared scope.  Round-to-nearest is the default for every tensor without a
    Hessian, so a scope can mix GPTQ and RTN (the decoder stays RTN; only the
    head needs the help).

Both phases run directly.

    python3 tools/gptq_calibrate.py \
        -- python3 tools/gptq_calibrate.py --family qwen3-asr-0.6b --phase calibrate

The per-family prefill builders mirror the ones task-9 used for its paired A/B
(``.tmp/t9/verify_w4a16.py``): the head input is the output of the real model
forward, never a synthetic tensor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: family -> (bf16 native directory name, model class family, head module path,
#:             audio-tower-driven prefill builder key)
FAMILIES = {
    "fun-asr-nano": ("fun-asr-nano", "fun-asr-nano", "llm.model.embed_tokens", "nano"),
    "qwen3-asr-0.6b": ("qwen3-asr-0.6b", "qwen3-asr", "thinker.model.embed_tokens", "qwen3"),
    "qwen3-asr-1.7b": ("qwen3-asr-1.7b", "qwen3-asr", "thinker.model.embed_tokens", "qwen3"),
    "glm-asr-nano": ("glm-asr-nano", "glm-asr-nano", "language_model.lm_head", "glm"),
    "moss-transcribe-diarize": (
        "moss-transcribe-diarize",
        "moss-transcribe-diarize",
        "language_model.model.embed_tokens",
        "moss",
    ),
}


def audio_dtype(dtype, *, what: str = "audio input"):
    """The dtype to cast an audio tensor to, refusing an integer dtype.

    Every tool here casts the mel/fbank before it reaches the tower, and the
    dtype used to be read as ``embed_tokens.weight.dtype``.  That is wrong twice
    over: it is the DECODER's dtype while the mel feeds the audio tower (a
    different subgraph), and once the embedding is quantized it is the packed
    uint32 codes, so ``mel.astype(uint32)`` silently destroyed the audio before
    the tower saw it.  The failure mode is the dangerous kind: the run completes,
    the transcripts are garbage, and it looks exactly like "4-bit is too coarse".

    So the dtype is passed through this helper, which refuses anything that is
    not a floating type.  A future edit that reintroduces a ``weight.dtype`` read
    fails loudly at driver-construction time instead of producing a
    plausible-looking wrong answer.
    """
    import mlx.core as mx

    if not mx.issubdtype(dtype, mx.floating):
        raise ValueError(
            f"Refusing to cast {what} to {dtype}: an integer dtype here means a "
            "quantized weight's packed codes were read as an activation dtype. "
            "Use activation_dtype()/_tower_dtype()."
        )
    return dtype


def _tokenizer(directory: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(directory), use_fast=False)


def model_config(family: str, model, config):
    """The structured config the prefill builders need.

    ``family`` here is the *model class* family (``qwen3-asr``), not the job
    name, because that is what ``load_model`` was called with.  ``load_model``
    returns the raw ``config.json`` mapping; some families hand the parsed
    dataclass back on the model object, so prefer that when it exists, and
    otherwise parse the mapping through the family's own config reader -- the
    same path ``create_model`` uses.
    """
    if family == "qwen3-asr":
        existing = getattr(model, "config", None)
        if existing is not None and hasattr(existing, "audio_pad_token_id"):
            return existing
        from funasr_mlx.models.qwen3_asr_thinker import qwen3asr_config_from_dict

        return qwen3asr_config_from_dict(config)
    if family == "fun-asr-nano":
        existing = getattr(model, "config", None)
        if existing is not None and hasattr(existing, "llm_conf"):
            return existing
        from funasr_mlx.models.fun_asr_nano import nano_config_from_dict

        return nano_config_from_dict(config)
    if family == "glm-asr-nano":
        existing = getattr(model, "config", None)
        if existing is not None and hasattr(existing, "text"):
            return existing
        from funasr_mlx.models.glm_asr_thinker import glmasr_thinker_config_from_dict

        return glmasr_thinker_config_from_dict(config)
    if family == "moss-transcribe-diarize":
        # Moss reads plain dicts (``config["text_config"]``), so the mapping is
        # already the right shape.
        return config
    raise ValueError(f"Unknown family: {family}")


def build_prefill(family: str, key: str, model, config, directory: Path, sample: dict):
    """Return ``prefill(sample)``: one real forward pass, head input captured.

    The returned callable takes a manifest sample and runs the family's own
    prefill path (tower -> splice -> decoder), which is what the head actually
    sees at inference time.
    """
    import mlx.core as mx
    import numpy as np
    import soundfile as sf

    from funasr_mlx.models.llm import KVCache
    from funasr_mlx.models.whisper import log_mel_spectrogram

    raw, _ = sf.read(sample["audio_path"], dtype="float32")

    if key == "qwen3":
        # The mel feeds the audio TOWER, so it must follow the tower's dtype.
        # Reading ``embed_tokens.weight.dtype`` here was wrong twice over: it is
        # the decoder's dtype (a different subgraph), and for a quantized
        # embedding it is the packed-code uint32, which turns the mel into
        # integers.  ``encode_audio`` casts to the tower dtype itself, so this
        # cast is only here to keep the front-end and tower consistent.
        dtype = audio_dtype(model._tower_dtype(), what="the mel (qwen3-asr)")
        mel = log_mel_spectrogram(mx.array(raw[None, :]), n_mels=128)
        mel = mx.transpose(mel[0], (1, 0)).astype(dtype)
        mx.eval(mel)
        tok = _tokenizer(directory)
        prefix_ids = tok(
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>",
            add_special_tokens=False,
        )["input_ids"]
        tail_ids = tok(
            "<|audio_end|><|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]
        pad_id = config.audio_pad_token_id
        n_layers = config.text.num_hidden_layers

        def prefill():
            embeds = model.encode_audio(mel)
            mx.eval(embeds)
            n = int(embeds.shape[1])
            ids = mx.array([prefix_ids + [pad_id] * n + tail_ids], dtype=mx.int32)
            caches = [KVCache() for _ in range(n_layers)]
            logits = model.forward_with_audio(ids, embeds, n, caches)
            mx.eval(logits)

        return prefill

    if key == "nano":
        from funasr_mlx.frontend import FbankConfig, WavFrontend

        # The fbank features feed the audio encoder; follow ITS dtype, not the
        # decoder embedding's (see the qwen3 branch for why).
        dtype = audio_dtype(model.audio_encoder._weight_dtype(), what="the fbank (nano)")
        fcfg = FbankConfig(
            frame_length=25.0,
            frame_shift=10.0,
            dither=0.0,
            window="hamming",
            n_mels=80,
            lfr_m=7,
            lfr_n=6,
        )
        fb = np.asarray(WavFrontend(fcfg, cmvn=None)(raw), dtype=np.float32)
        t = fb.shape[0]
        olens = 1 + (t - 3 + 2) // 2
        olens = 1 + (olens - 3 + 2) // 2
        fake_len = (olens - 1) // 2 + 1
        speech = mx.array(fb)[None].astype(dtype)
        mx.eval(speech)
        lengths = mx.array([t], dtype=mx.int32)
        tok = _tokenizer(directory)
        prefix_ids = tok(
            "<|im_start|>system\nYou are a speech recognition model.<|im_end|>\n"
            "<|im_start|>user\n语音转写：<|startofspeech|>",
            add_special_tokens=False,
        )["input_ids"]
        tail_ids = tok(
            "<|endofspeech|><|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]
        source_ids = prefix_ids + [0] * fake_len + tail_ids
        beg = len(prefix_ids)
        n_layers = 28

        def prefill():
            enc, _ = model.audio_encoder(speech, lengths)
            ad = model.audio_adaptor(enc)
            mx.eval(ad)
            n = min(int(ad.shape[1]), fake_len)
            ids = mx.array([source_ids], dtype=mx.int32)
            emb = model.llm.model.embed_tokens(ids)
            spliced = mx.concatenate(
                [emb[:, :beg, :], ad[:, :n, :], emb[:, beg + fake_len :, :]], axis=1
            )
            caches = [KVCache() for _ in range(n_layers)]
            h = model.llm.model(input_embeddings=spliced, cache=caches, mask="causal")
            mx.eval(model.llm.model.embed_tokens.as_linear(h[:, -1:]))

        return prefill

    if key == "glm":
        # The mel feeds the audio tower; follow the tower's dtype, not the
        # decoder embedding's (see the qwen3 branch for why).
        dtype = audio_dtype(model._tower_dtype(), what="the mel (glm)")
        mel = log_mel_spectrogram(mx.array(raw)[None], n_mels=128)[0]
        mel = mx.transpose(mel.astype(dtype), (1, 0))
        mx.eval(mel)
        prefix, tail = [59253, 10, 59261], [59262, 10, 59254, 10]
        n_layers = config.text["num_hidden_layers"]

        def prefill():
            embeds = model.encode_audio(mel)
            mx.eval(embeds)
            n = int(embeds.shape[1])
            ids = mx.array([prefix + [59260] * n + tail])
            caches = [KVCache() for _ in range(n_layers)]
            logits = model.forward_with_audio(ids, embeds, n, caches)
            mx.eval(logits)

        return prefill

    if key == "moss":
        tok = _tokenizer(directory)
        prompt = (
            "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
            "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
            "并在段末标注结束时间戳，以清晰标明该段语音范围。"
        )
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": raw},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        template = (directory / "chat_template.jinja").read_text()
        tmpl = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, chat_template=template
        )
        marker = "<|audio_pad|>"
        pre, rest = tmpl.split(marker, 1)
        n_layers = config["text_config"]["num_hidden_layers"]
        holder = {}

        def prepare():
            n_samples, hop, stride, merge = 480000, 160, 2, 4
            parts = []
            for start in range(0, len(raw), n_samples):
                chunk = raw[start : start + n_samples]
                tok_len = (len(chunk) - 1) // (hop * stride * merge) + 1
                pad = np.zeros(n_samples, dtype=np.float32)
                pad[: len(chunk)] = chunk
                mel_c = log_mel_spectrogram(mx.array(pad)[None], n_mels=80)[0]
                h = model.whisper_encoder(mx.transpose(mel_c, (1, 0)))
                parts.append(h[0, : tok_len * 4 * stride, :])
            feat = mx.concatenate(parts, axis=0)[None]
            mx.eval(feat)
            holder["feat"] = feat

        def prefill():
            if "feat" not in holder:
                prepare()
            embeds = model.encode_audio_from_hidden(holder["feat"])
            mx.eval(embeds)
            n = int(embeds.shape[1])
            ids = mx.array([tok.encode(pre + marker * n + rest)])
            caches = [KVCache() for _ in range(n_layers)]
            logits = model.forward_with_audio(ids, embeds, n, caches)
            mx.eval(logits)
            # TEACHER-FORCED DECODE REPLAY: moss emits timestamps / speaker
            # tags / segment decisions in the DECODE loop, whose activation
            # distribution differs from prefill. Calibration that never sees
            # decode steps tunes scale/clip on the wrong distribution -- the
            # out-of-tolerance timestamps clustered exactly here. The greedy
            # run below (on this bf16 source) feeds the capture the tokens the
            # model itself would emit, timestamps included.
            token = int(mx.argmax(logits[0, -1]).item())
            for _ in range(64):
                lg = model.language_model(mx.array([[token]]), cache=caches)
                mx.eval(lg)
                token = int(mx.argmax(lg[0, -1]).item())
                if token in (151643, 151645):
                    break

        return prefill

    raise ValueError(f"No prefill builder for {key}")


def load_manifest(path: Path, split: str, languages: list[str], per_language: int):
    """Pick ``per_language`` samples of ``split`` for each requested language."""
    manifest = json.loads(path.read_text())
    chosen = []
    for language in languages:
        rows = [
            sample
            for sample in manifest["samples"]
            if sample["lang"] == language and sample["split"] == split
        ]
        if len(rows) < per_language:
            raise SystemExit(
                f"{language}: manifest has {len(rows)} {split} samples, need {per_language}"
            )
        chosen.extend(rows[:per_language])
    return chosen


def parse_language_weights(spec: str | None) -> dict[str, float]:
    """Parse ``--language-weights zh=1,ja=3`` into a mapping.

    Kept as a plain function so the format is testable without a model.
    """
    if not spec:
        return {}
    weights: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--language-weights entries must be lang=weight, got {part!r}")
        name, _, value = part.partition("=")
        try:
            weight = float(value)
        except ValueError as error:
            raise SystemExit(f"Invalid weight for {name!r}: {value!r}") from error
        if not weight > 0:
            raise SystemExit(f"Weight for {name!r} must be positive, got {weight}")
        weights[name.strip()] = weight
    return weights


def calibrate(args) -> int:
    from funasr_mlx.models.llm.calibrate import HeadActivationCollector, HeadTarget
    from funasr_mlx.pipeline import load_model


    native_dir, family, head_path, key = FAMILIES[args.family]
    source = args.native_root / native_dir
    raw_model, raw_config = load_model(source, family=family)
    model = raw_model
    # ``FAMILIES`` maps a job name to the *model class* family, which is what
    # load_model/create_model dispatch on and therefore what the config reader
    # has to be asked for.
    config = model_config(family, raw_model, raw_config)
    target = HeadTarget.resolve(model, head_path=head_path)
    print(f"{args.family}: head {head_path} ({target.kind}), hidden {target.hidden_size}")

    samples = load_manifest(args.manifest, "calibration", args.languages, args.per_language)
    print(f"calibration utterances: {len(samples)}")

    collector = HeadActivationCollector(target)
    for index, sample in enumerate(samples):
        prefill_for = build_prefill(family, key, model, config, source, sample)
        rows = collector.observe(language=sample["lang"], run=prefill_for)
        if rows == 0:
            raise SystemExit(f"No head activations captured for {sample['audio_path']}")
        if index % 10 == 0 or index == len(samples) - 1:
            print(f"  [{index + 1}/{len(samples)}] {sample['lang']} rows={rows}", flush=True)

    report = collector.report()
    report["family"] = args.family
    report["manifest"] = str(args.manifest)
    report["samples"] = len(samples)
    language_weights = parse_language_weights(args.language_weights)
    if language_weights:
        # Equal per-language mass is the default, but it is not the right target
        # when the languages do not quantize equally well.  The weights are
        # reported with their resulting shares so the run is reproducible from
        # its own report, and the guard is re-checked against the SHARES the
        # reweighting actually produces.
        mass = collector.accumulator.language_mass_with(language_weights, balance=True)
        report["language_weights"] = language_weights
    else:
        mass = report["language_mass_balanced"]
    report["language_mass_effective"] = mass
    worst = max(mass.values())
    report["max_language_mass"] = worst
    print(f"language mass: max {worst:.3f} ({max(mass, key=mass.get)})")
    if worst > args.max_language_mass:
        raise SystemExit(
            f"Language balance guard failed: {max(mass, key=mass.get)} holds "
            f"{worst:.3f} of the Hessian mass, over the {args.max_language_mass} bound"
        )

    hessian = collector.hessian(balance=True, language_weights=language_weights or None)
    if not np_all_finite(hessian):
        raise SystemExit("Calibration Hessian is not finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    collector.accumulator.save(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output} and {args.report}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def np_all_finite(array) -> bool:
    import numpy as np

    return bool(np.isfinite(array).all())


def convert(args) -> int:
    from funasr_mlx.checkpoints import GptqWeight, convert_checkpoint
    from funasr_mlx.models.llm.gptq import HessianAccumulator, GptqConfig
    from funasr_mlx.models.llm.quant import QuantizationSpec


    native_dir, family, head_path, _key = FAMILIES[args.family]
    source = args.source_dir if args.source_dir is not None else args.native_root / native_dir
    spec = QuantizationSpec.from_dict(args.quantization)
    # The converter writes the destination's ``config.json`` from what it is
    # handed, and ``pipeline.load_model`` refuses to guess a configuration, so
    # the source's own config is what has to be passed through -- with the
    # quantization section added so the loader knows to rebuild the quantized
    # modules.  Without this the emitted checkpoint has no config.json at all
    # and cannot be loaded.
    if args.config is not None:
        checkpoint_config = json.loads(Path(args.config).read_text())
    else:
        checkpoint_config = json.loads((source / "config.json").read_text())
    checkpoint_config["quantization"] = spec.to_dict()
    # ``--method`` is the authority on how the values were computed; the spec's
    # own ``method`` field only records what was asked for on the command line.
    # Without this an rtn run shipped a config.json claiming method=gptq, which is
    # exactly the kind of provenance error that makes a later comparison
    # untrustworthy.
    checkpoint_config["quantization"]["method"] = args.method

    # RTN never reads the Hessian: its values are round-to-nearest on the
    # raw weights.  Loading (and requiring) the .npz for an rtn conversion
    # made an rtn-only pipeline fail on a machine that never calibrated.
    if args.method == "gptq":
        accumulator = HessianAccumulator.load(args.hessian)
    else:
        accumulator = None
    # The file stores RAW per-language ``X^T X``, so reweighting is a pure
    # post-processing step: changing the balance does not require re-running the
    # (expensive) calibration forward passes at all, only the conversion.
    language_weights = parse_language_weights(args.language_weights)
    if accumulator is not None:
        if language_weights:
            shares = accumulator.language_mass_with(language_weights, balance=True)
            worst = max(shares.values())
            print(
                f"language weights {language_weights} -> max share {worst:.3f} "
                f"({max(shares, key=shares.get)})"
            )
            if worst > args.max_language_mass:
                raise SystemExit(
                    f"Reweighting pushes {max(shares, key=shares.get)} to {worst:.3f} "
                    f"of the Hessian mass, over the {args.max_language_mass} bound"
                )
        # The Hessian file is keyed by language; rebuild the (weighted) Hessian.
        hessian = accumulator.hessian(
            balance=True, language_weights=language_weights or None
        )
        # The per-language sums are dead once the balanced Hessian exists, and a
        # 1024x1024 float32 matrix is only 4 MB but the accumulator holds ten of
        # them; drop the source so the peak is the balanced copy alone.
        del accumulator
    else:
        hessian = None
    # NB: named ``gptq_config``, not ``config`` -- the model's own config is
    # ``checkpoint_config`` above, and an earlier version let the GPTQ
    # hyper-parameters overwrite it, which wrote a GptqConfig object into the
    # destination's config.json (and crashed on the way).
    #
    # The bit width comes from the HEAD's own override, not from the spec default:
    # a w8 head beside a w4 stack is exactly the case this supports, and the
    # converter cross-checks this config against the width it packs the tensor at.
    head_bits = spec.bits_for_weight(head_path + ".weight")
    head_group = spec.group_for(head_path)
    gptq_config = GptqConfig(
        bits=head_bits,
        group_size=head_group,
        damp_percent=args.damp_percent,
        block_size=args.block_size,
        act_order=not args.no_act_order,
        storage=args.storage,
    )
    # The head's native tensor name is the head path plus ".weight".
    head_weight = head_path + ".weight"
    gptq = GptqWeight(
        hessians={head_weight: hessian} if hessian is not None else {},
        configs={head_weight: gptq_config.to_dict()},
        method=args.method,
    )
    print(
        f"{args.method.upper()} target: {head_weight} "
        f"{hessian.shape if hessian is not None else 'no-hessian(rtn)'} "
        f"{gptq_config.to_dict()}"
    )

    info = convert_checkpoint(
        source,
        args.destination,
        model=family,
        dtype=args.dtype,
        config=checkpoint_config,
        source_rule="native",
        quantization=spec.to_dict(),
        gptq=gptq,
        tensor_limit=args.tensor_limit,
    )
    if not gptq.outcomes:
        # Not an error when the head is deliberately out of scope: the tower/
        # decoder-only configuration is a supported arm (t9's known-safe scope),
        # and it quantizes only Linears, which go through mx.quantize rather than
        # the GPTQ path.  It IS an error if the head was selected and nothing
        # happened -- but "nothing" must be measured on what the CONVERTER
        # packed, not on the GPTQ outcomes: an rtn conversion legitimately has
        # zero GPTQ outcomes while the head packed fine through mx.quantize
        # (its .scales landed in the emitted weight_map).
        index_path = args.destination / "model.safetensors.index.json"
        index = json.loads(index_path.read_text()) if index_path.is_file() else {}
        weight_map = index.get("weight_map", {})
        head_quantized = (head_weight[:-len(".weight")] + ".scales") in weight_map
        if head_quantized:
            print(
                f"note: {head_path} packed via mx.quantize (rtn path), "
                "no GPTQ outcomes to report"
            )
        elif any(
            head_path.startswith(target) or target.startswith(head_path)
            for target in spec.targets
        ):
            raise SystemExit(
                f"The head was not quantized -- is {head_weight} selected by "
                f"targets {spec.targets}?"
            )
        else:
            print(
                f"note: {head_path} is out of scope for targets {list(spec.targets)}; "
                "the head stays at the checkpoint dtype (decoder/tower-only arm)"
            )
    print(json.dumps({"gptq": gptq.outcomes, "info": info.__dict__}, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "gptq": gptq.outcomes,
                    "info": info.__dict__,
                    "gptq_config": gptq_config.to_dict(),
                    "targets": list(spec.targets),
                    "language_weights": language_weights or None,
                },
                indent=2,
            )
            + "\n"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--phase", choices=("calibrate", "convert"), required=True)
    parser.add_argument(
        "--native-root", type=Path, default=Path("/Volumes/2T/funasr-mlx-alignment/native-llm")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/Volumes/2T/funasr-mlx-alignment/calib/calib-manifest.json"),
    )
    parser.add_argument("--per-language", type=int, default=48)
    parser.add_argument(
        "--languages",
        default="zh,en,ja,ko,es,fr,de,ru,th,vi",
        help="comma-separated language tags to calibrate on",
    )
    parser.add_argument("--max-language-mass", type=float, default=0.30)
    parser.add_argument(
        "--language-weights",
        default=None,
        help=(
            "Optional per-language Hessian weights, e.g. 'ja=3,th=3,vi=2'. "
            "Equal mass is the default; use this when measured per-language "
            "divergence shows some languages need more of the bit budget. "
            "Weights are renormalised so the total mass is unchanged."
        ),
    )
    parser.add_argument("--hessian", type=Path, default=None)
    parser.add_argument("--destination", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--quantization", type=json.loads, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--method",
        choices=("gptq", "rtn"),
        default="gptq",
        help=(
            "How to compute the stored values for the selected matrices. Both "
            "go through the identical packing and layout, so this isolates the "
            "algorithm: it is what makes an rtn-vs-gptq comparison honest."
        ),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help=(
            "Override the conversion source directory (e.g. an AWQ-adjusted "
            "bf16 checkpoint) instead of <native-root>/<family dir>"
        ),
    )
    parser.add_argument("--storage", default="bfloat16")
    parser.add_argument("--damp-percent", type=float, default=0.01)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--no-act-order", action="store_true")
    parser.add_argument("--tensor-limit", type=int, default=384_000_000)
    args = parser.parse_args()
    args.languages = [part for part in args.languages.split(",") if part]

    # Calibrated artifacts live on the 2T disk since the 1T USB drive proved
    # unstable (repeated disconnects mid-copy). New runs write here; the old
    # 1T location is read via --hessian/--output overrides if ever needed.
    work = Path("/Volumes/2T/funasr-mlx-alignment/calib/gptq")
    if args.output is None:
        args.output = work / f"{args.family}-hessian.npz"
    if args.report is None:
        args.report = work / f"{args.family}-{args.phase}.json"
    if args.hessian is None:
        args.hessian = work / f"{args.family}-hessian.npz"
    if args.destination is None:
        args.destination = (
            Path("/Volumes/2T/funasr-mlx-alignment/native-llm") / f"{args.family}-w4a16-gptq"
        )
    if args.quantization is None:
        # Default scope: the whole stack plus the tied head -- the point of GPTQ
        # is that the head is now safe to quantize.
        args.quantization = {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "targets": ["thinker.model.layers.", "audio_tower.", "thinker.model.embed_tokens"],
            "method": "gptq",
        }
    return calibrate(args) if args.phase == "calibrate" else convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
