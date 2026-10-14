#!/usr/bin/env python3
"""Full-weight reference/native alignment adapters for four model families.

Purpose
-------
Export **real forward-pass outputs** from the official FunASR models and from the
native ``funasr_mlx`` ports so a separate numerical comparator can check them
against each other.  This module produces *measurements*; it holds no tolerances
and draws no conclusions.

Contract
--------
* The reference side is driven by :class:`funasr.auto.auto_model.AutoModel`,
  the officially supported loader, so the model class, its frontend and its
  tokenizer are built by FunASR's own registry rather than hand-wired here.
* The native side goes through :func:`funasr_mlx.pipeline.load_model`, the same
  entry the CLI uses, so the guarded child is required and the checkpoint path
  exercised is the shipping one.
* The two sides run as **separate processes** (``--mode reference`` /
  ``--mode native``); torch and MLX never share an interpreter.
* Each entry point writes two files and returns nothing:

  ``<key>.<mode>.npz``   bounded ``float32`` / integer arrays
  ``<key>.<mode>.json``  text, ids, labels, config provenance, disposition

* Nothing is derived from the other side.  A "golden" generated from the native
  model and compared against the native model would be circular, so both sides
  always run their own forward pass.
* Inputs longer than the manifest limit are **rejected**, never truncated: a
  silently cropped utterance would compare two different signals.
* FP32 baseline -- arrays are cast on the way out so the comparator is not
  chasing dtype noise.

Manifest
--------
A JSON list of entries.  ``key`` and ``family`` are required; the rest depend on
the family.

``fsmn-vad``
    ``audio`` (path), ``limit_seconds``
``ct-punc``
    ``text``, ``limit_chars``
``campplus``
    ``audio`` (path), ``limit_seconds``
``emotion2vec``
    ``audio`` (path), ``limit_seconds``

Assets
------
A JSON object of filesystem paths supplied by the lead.  Nothing is downloaded
and no model is fetched implicitly: every path must already exist on disk, and a
missing one is a hard error.  The names below are the ones the published
FunASR repositories actually use, so they are not guesses:

============================  ===========================================
field                         published artefact
============================  ===========================================
``model_dir``                 FunASR model dir: ``config.yaml`` (+ weights)
``weights``                   campplus only: ``campplus_cn_common.bin``
``checkpoint``                **converted native** safetensors (native side)
``config``                    the family's ``config.yaml`` (always needed)
``labels``                    emotion2vec ``tokens.txt`` (8 lines)
``tokens``                    ct-punc ``tokens.json`` (a **JSON list**)
``frontend_conf``             optional per-family frontend overrides
``speaker_a`` / ``speaker_b`` two utterances for a campplus cosine pair
============================  ===========================================

``model_dir``
    The **original** FunASR model directory, used only by ``--mode reference``.
    ``AutoModel`` expects the standard directory layout; synthesising one by
    copying weights would be worse than asking for it.
``weights``
    campplus only.  The published checkpoint is ``campplus_cn_common.bin``,
    which ``AutoModel`` does not look for, so that family is built directly from
    ``model_conf`` and this file is read with ``weights_only=True``.
``checkpoint``
    The **converted native** checkpoint (safetensors) consumed by
    ``funasr_mlx.pipeline.load_model``, used only by ``--mode native``.  This is
    a different artefact from ``model_dir``, not an alias for it.
``config``
    The family's ``config.yaml``.  Always read: the emotion2vec native dataclass
    cannot accept the official fairseq schema, so the mapping is explicit.
``labels``
    emotion2vec ``tokens.txt``, read verbatim and in order -- the list is
    positional against the classifier's output rows.
``tokens``
    ct-punc ``tokens.json``.  It is a **JSON list** of 471067 entries, not a
    newline-separated file; reading it line by line would silently produce a
    garbage vocabulary.
``frontend_conf``
    FSMN-VAD's official ``frontend_conf`` already carries ``lfr_m: 5`` and
    ``lfr_n: 1`` alongside ``n_mels: 80`` and ``dither: 0.0``; those are used
    as given and only defaulted from ``input_dim`` when absent.
"""

from __future__ import annotations

import argparse
import json
from functools import wraps
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Hard ceiling on a single exported ``.npz``.  The comparator only needs
#: bounded tensors; a runaway intermediate must fail loudly, not fill a volume.
MAX_NPZ_BYTES = 64 * 1024 * 1024

#: Sample rate every one of these official frontends runs at.
SAMPLE_RATE = 16000

#: Families this module can align, in CLI order.
FAMILIES = ("fsmn-vad", "ct-punc", "campplus", "emotion2vec")


# --------------------------------------------------------------------------- #
# private output helpers
# --------------------------------------------------------------------------- #
def _write_npz(out_dir: Path, key: str, mode: str, arrays: Mapping[str, Any]) -> Path:
    import numpy as np

    materialised = {name: np.asarray(value) for name, value in arrays.items()}
    for name, value in materialised.items():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"{key}: array {name!r} contains non-finite values")
    target = out_dir / f"{key}.{mode}.npz"
    temporary = out_dir / f"{key}.{mode}.npz.partial"
    with temporary.open("wb") as handle:
        np.savez(handle, **materialised)
    if temporary.stat().st_size > MAX_NPZ_BYTES:
        temporary.unlink()
        raise ValueError(f"{key}: exported arrays exceed the {MAX_NPZ_BYTES} byte bound")
    temporary.replace(target)
    return target


def _write_json(out_dir: Path, key: str, mode: str, payload: Mapping[str, Any]) -> Path:
    target = out_dir / f"{key}.{mode}.json"
    temporary = out_dir / f"{key}.{mode}.json.partial"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def _reject_overlong(
    entry: Mapping[str, Any], actual: float, limit_field: str, limit: float, unit: str
) -> None:
    """Refuse an over-limit input instead of truncating it."""
    if limit <= 0:
        raise ValueError(f"{entry.get('key')}: {limit_field} must be positive")
    if actual > limit:
        raise ValueError(
            f"{entry.get('key')}: {actual:.3f} {unit} exceeds the {limit_field} "
            f"limit of {limit}; refusing to truncate a comparison input"
        )


def _require(assets: Mapping[str, Any], field: str) -> str:
    value = assets.get(field)
    if not value:
        raise ValueError(f"assets must supply {field!r}")
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"{field}: {path}")
    return str(path)


def _reference_model_dir(assets: Mapping[str, Any]) -> str:
    """The FunASR local model directory, validated as such."""
    directory = Path(_require(assets, "model_dir"))
    if not (directory / "config.yaml").is_file():
        raise FileNotFoundError(
            f"model_dir {directory} has no config.yaml; AutoModel needs the standard "
            "FunASR directory layout"
        )
    return str(directory)


def _load_reference(assets: Mapping[str, Any]) -> Any:
    """An ``AutoModel`` on CPU with the official frontend and tokenizer."""
    from funasr.auto.auto_model import AutoModel

    return AutoModel(
        model=_reference_model_dir(assets),
        device="cpu",
        disable_pbar=True,
        disable_update=True,
        ncpu=1,
    )


def _reference_frontend(auto: Any) -> Any:
    frontend = auto.kwargs.get("frontend")
    if frontend is None:
        raise ValueError("AutoModel did not build a frontend; cannot reproduce features")
    return frontend


def _reference_tokenizer(auto: Any) -> Any:
    tokenizer = auto.kwargs.get("tokenizer")
    if tokenizer is None:
        raise ValueError("AutoModel did not build a tokenizer; cannot reproduce ids")
    return tokenizer


def _reference_waveform(path: str) -> Any:
    """Load one utterance as a 1-D float32 numpy array at 16 kHz."""
    import numpy as np
    from funasr.utils.load_utils import load_audio_text_image_video

    audio = load_audio_text_image_video(
        [path], fs=SAMPLE_RATE, audio_fs=SAMPLE_RATE, data_type="sound"
    )[0]
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _check_audio(entry: Mapping[str, Any], samples: Any) -> None:
    _reject_overlong(
        entry,
        samples.size / SAMPLE_RATE,
        "limit_seconds",
        float(entry.get("limit_seconds", 0.0)),
        "s",
    )


# --------------------------------------------------------------------------- #
# fsmn-vad
# --------------------------------------------------------------------------- #
def run_fsmn_vad_reference(
    entry: Mapping[str, Any], assets: Mapping[str, Any], out_dir: Path
) -> None:
    """Official ``FsmnVADStreaming`` forward: frame posterior + segment endpoints."""
    import numpy as np
    import torch

    auto = _load_reference(assets)
    frontend = _reference_frontend(auto)
    module = auto.model

    samples = _reference_waveform(entry["audio"])
    _check_audio(entry, samples)
    frontend_cache: dict[str, Any] = {}
    feats, feat_lengths = frontend(
        torch.from_numpy(samples[None, :]),
        torch.tensor([samples.size], dtype=torch.int64),
        cache=frontend_cache,
        is_final=True,
        return_waveform=True,
    )
    waveform = frontend_cache.get("aligned_waveforms")
    if waveform is None or waveform.numel() == 0:
        raise ValueError("Official online frontend emitted no aligned waveform")

    cache: dict[str, Any] = {}
    module.init_cache(cache)
    # Final segment cleanup drops stats.scores; evaluate the identical fresh encoder
    # call separately so posterior measurements cannot become empty.
    with torch.no_grad():
        posterior = module.encoder(feats, cache={})
    # The official fsmn-vad config.yaml ships ``output_frame_probs: False``; the
    # frame stage is required for this comparison, so it is overridden here
    # explicitly rather than left to the config.
    result = module.forward(
        feats,
        waveform,
        cache=cache,
        is_final=True,
        is_streaming_input=False,
        output_frame_probs=True,
    )
    segments, frame_probs, _ = _vad_stages(result, cache)
    scores = posterior

    _write_npz(
        out_dir,
        entry["key"],
        "reference",
        {
            # raw encoder posterior -- the numeric stage the two sides can actually
            # be compared on, ahead of any thresholded state
            "frontend": _numpy(feats).astype(np.float32),
            "scores": _numpy(scores).reshape(-1, _numpy(scores).shape[-1]).astype(np.float32),
            "frame_prob": np.asarray(
                [[p.noise_prob, p.speech_prob, p.score] for p in frame_probs],
                dtype=np.float32,
            ).reshape(-1, 3),
            "frame_state": np.asarray([p.frm_state for p in frame_probs], dtype=np.int32),
            "segment_bounds": np.asarray(segments, dtype=np.int32).reshape(-1, 2),
        },
    )
    _write_json(
        out_dir,
        entry["key"],
        "reference",
        {
            "family": "fsmn-vad",
            "segments_ms": [[int(pair[0]), int(pair[1])] for pair in segments],
            "n_frames": int(_numpy(scores).reshape(-1, _numpy(scores).shape[-1]).shape[0]),
            "scores_source": "fresh_encoder_call_with_empty_fsmn_cache",
            "n_frame_probs": len(frame_probs),
            "duration_seconds": samples.size / SAMPLE_RATE,
            "frontend": type(frontend).__name__,
        },
    )


def _numpy(value: Any) -> Any:
    """Detach either a torch tensor or an MLX array to numpy."""
    import numpy as np

    if hasattr(value, "detach"):  # torch
        return value.detach().to("cpu").numpy()
    return np.asarray(value)


def _vad_stages(result: Any, cache: Mapping[str, Any]) -> tuple[list[Any], list[Any], Any]:
    """``(segments, frame_probs, raw_scores)`` from a VAD forward result.

    Three facts about the FSMN-VAD forward that are not visible from its
    signature, all confirmed by reading the body rather than guessing:

    * it returns ``list[list[list[int]]]`` -- a *batch* of ``[start_ms, end_ms]``
      pairs per call -- so the caller must flatten it;
    * it **never returns** frame probabilities.  ``output_frame_probs=True`` makes
      it append ``E2EVadFrameProb`` objects to ``cache["stats"].frame_probs``;
    * the raw encoder posterior is accumulated onto ``cache["stats"].scores``
      across chunks.  That is the meaningful numeric stage to compare, because
      ``frame_probs`` are already thresholded into states.

    ``-1`` end points (``[start, -1]``) are preserved as the reference emits
    them; a comparator that drops them would hide a streaming disagreement.
    """
    segments: list[Any] = []
    for batch in result or []:
        for pair in batch:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                raise ValueError(f"unrecognised VAD segment entry: {pair!r}")
            segments.append([int(pair[0]), int(pair[1])])
    stats = cache.get("stats")
    if stats is None:
        raise ValueError("VAD cache has no 'stats'; init_cache() was not run")
    # Both attributes are verified on the reference ``Stats`` and the native
    # ``VadStats``.  They are read strictly: a missing one is a wiring error and
    # silently yielding an empty list would turn it into a fake agreement.
    frame_probs = _required_attr(stats, "frame_probs")
    scores = _required_attr(stats, "scores")
    if scores is None:
        raise ValueError("VAD stats carry no 'scores'; the encoder did not run")
    return segments, list(frame_probs), scores


def _required_attr(obj: Any, name: str) -> Any:
    """``getattr`` that refuses to invent a default."""
    if not hasattr(obj, name):
        raise ValueError(
            f"{type(obj).__name__} has no {name!r}; read the source rather than "
            "guessing a container name"
        )
    return getattr(obj, name)


def run_fsmn_vad_native(entry: Mapping[str, Any], assets: Mapping[str, Any], out_dir: Path) -> None:
    """Native ``funasr_mlx.models.fsmn_vad.FsmnVadModel`` forward.

    Input geometry follows ``funasr_mlx/cli.py`` exactly:

    * features are ``(1, T, input_dim)`` and the waveform is **not** the whole
      signal -- it is the ``(1, needed)`` span those ``T`` snipped frames cover,
      ``needed = (T - 1) * frame_shift_length + frame_sample_length``.  Snipped
      fbank frames do not consume a trailing incomplete frame, so passing the
      full waveform would hand the decibel/SNR bookkeeping a span the features do
      not describe.
    * FSMN-VAD consumes **LFR-stacked** fbank: the official ``input_dim: 400``
      over an 80-bin mel is ``lfr_m = 5``.  That factor is read back out of
      ``input_dim`` rather than assumed.
    """
    import mlx.core as mx
    import numpy as np
    from funasr_mlx.frontend import FbankConfig, WavFrontend, read_audio
    from funasr_mlx.pipeline import load_model

    model, config = load_model(
        _require(assets, "checkpoint"), family="fsmn-vad", config=assets.get("config")
    )
    samples = np.asarray(read_audio(entry["audio"]), dtype=np.float32)
    _check_audio(entry, samples)

    input_dim = _vad_input_dim(config)
    n_mels = 80
    if input_dim % n_mels:
        raise ValueError(
            f"input_dim {input_dim} is not a multiple of the {n_mels}-bin mel filterbank; "
            "the LFR stack factor cannot be derived"
        )
    options = dict((config or {}).get("frontend_conf") or {}) if isinstance(config, Mapping) else {}
    options.pop("cmvn_file", None)
    options.setdefault("lfr_m", input_dim // n_mels)
    options.setdefault("lfr_n", 1)
    features = WavFrontend(FbankConfig.from_dict(options), cmvn=assets.get("cmvn"))(samples)
    if features.shape[0] == 0:
        raise ValueError(f"{entry['key']}: frontend produced no frames")

    needed = (
        features.shape[0] - 1
    ) * model.vad_opts.frame_shift_length + model.vad_opts.frame_sample_length
    if samples.size < needed:
        raise ValueError(f"{entry['key']}: VAD feature frames exceed the waveform")
    waveform = mx.array(samples[:needed])[None, ...]  # (1, needed)

    cache = model.init_cache()
    encoder_scores = model.encoder(features[None, ...], cache={})
    mx.eval(encoder_scores)
    result = model(features[None, ...], waveform, cache, is_final=True, is_streaming_input=False)
    segments, frame_probs, _ = _vad_stages(result, cache)
    posterior = np.asarray(encoder_scores).reshape(-1, encoder_scores.shape[-1])

    arrays: dict[str, Any] = {
        "frontend": np.asarray(features[None], dtype=np.float32),
        "scores": posterior.astype(np.float32),
        "segment_bounds": np.asarray(segments, dtype=np.int32).reshape(-1, 2),
    }
    arrays["frame_prob"] = np.asarray(
        [[p.noise_prob, p.speech_prob, p.score] for p in frame_probs], dtype=np.float32
    ).reshape(-1, 3)
    arrays["frame_state"] = np.asarray([p.frm_state for p in frame_probs], dtype=np.int32)
    _write_npz(out_dir, entry["key"], "native", arrays)
    _write_json(
        out_dir,
        entry["key"],
        "native",
        {
            "family": "fsmn-vad",
            "segments_ms": [[int(pair[0]), int(pair[1])] for pair in segments],
            "n_frames": int(posterior.shape[0]),
            "scores_source": "fresh_encoder_call_with_empty_fsmn_cache",
            "n_frame_probs": len(frame_probs),
            "duration_seconds": samples.size / SAMPLE_RATE,
            "lfr_m": input_dim // n_mels,
            "waveform_samples": int(needed),
        },
    )


def _vad_input_dim(config: Any) -> int:
    """The FSMN encoder ``input_dim``, wherever ``load_model`` surfaced it."""
    if isinstance(config, Mapping):
        encoder_conf = config.get("encoder_conf") or {}
        if isinstance(encoder_conf, Mapping) and "input_dim" in encoder_conf:
            return int(encoder_conf["input_dim"])
        if "input_dim" in config:
            return int(config["input_dim"])
    return 400  # the official fsmn-vad encoder_conf value


# --------------------------------------------------------------------------- #
# ct-punc
# --------------------------------------------------------------------------- #
def run_ct_punc_reference(
    entry: Mapping[str, Any], assets: Mapping[str, Any], out_dir: Path
) -> None:
    """Official ``CTTransformer.punc_forward`` over jieba-split mini-sentences.

    The word split and the id stream both come from FunASR's own helpers and
    tokenizer, because the classifier is compared on ids: re-deriving them here
    would make the comparison circular.
    """
    import numpy as np
    import torch
    from funasr.models.ct_transformer.utils import split_to_mini_sentence, split_words

    auto = _load_reference(assets)
    module = auto.model
    tokenizer = _reference_tokenizer(auto)

    text = str(entry["text"])
    _reject_overlong(
        entry, float(len(text)), "limit_chars", float(entry.get("limit_chars", 0.0)), "chars"
    )

    tokens = list(split_words(text, jieba_usr_dict=getattr(module, "jieba_usr_dict", None)))
    token_ids = list(tokenizer.encode(tokens))
    mini_words = split_to_mini_sentence(tokens, 20)
    mini_ids = split_to_mini_sentence(token_ids, 20)
    if len(mini_words) != len(mini_ids):
        raise ValueError("word and id mini-sentences diverged")

    punc_ids: list[int] = []
    blocks: list[Any] = []
    for words, ids in zip(mini_words, mini_ids):
        batch = torch.from_numpy(np.asarray([ids], dtype="int64"))
        lengths = torch.from_numpy(np.asarray([len(ids)], dtype="int32"))
        # punc_forward(text, text_lengths) -- two positional tensors, not a dict.
        logits, _ = module.punc_forward(batch, lengths)
        block = logits.detach().to(torch.float32).cpu().numpy()[0]
        blocks.append(block)
        punc_ids.extend(int(value) for value in block.argmax(axis=-1))

    _write_npz(
        out_dir,
        entry["key"],
        "reference",
        {
            "token_ids": np.asarray(token_ids, dtype=np.int32),
            "punc_ids": np.asarray(punc_ids, dtype=np.int32),
            "logits": np.concatenate(blocks, axis=0).astype(np.float32),
        },
    )
    _write_json(
        out_dir,
        entry["key"],
        "reference",
        {
            "family": "ct-punc",
            "text": text,
            "tokens": tokens,
            "punc_array": punc_ids,
            "n_mini_sentences": len(mini_words),
        },
    )


def run_ct_punc_native(entry: Mapping[str, Any], assets: Mapping[str, Any], out_dir: Path) -> None:
    """Native ``funasr_mlx.models.ct_punc.CtPuncModel.punc_forward``."""
    import mlx.core as mx
    import numpy as np
    from funasr_mlx.models.ct_punc import CharVocab, split_words
    from funasr_mlx.pipeline import load_model

    model, _config = load_model(
        _require(assets, "checkpoint"),
        family="ct-punc",
        config={
            **_load_yaml(_require(assets, "config")),
            "model_conf": {
                **_load_yaml(_require(assets, "config")).get("model_conf", {}),
                "vocab_size": len(_load_ct_punc_tokens(_require(assets, "tokens"))),
            },
        },
    )
    text = str(entry["text"])
    _reject_overlong(
        entry, float(len(text)), "limit_chars", float(entry.get("limit_chars", 0.0)), "chars"
    )

    token_ids = list(
        CharVocab(_load_ct_punc_tokens(_require(assets, "tokens"))).encode(list(split_words(text)))
    )
    logits = model.punc_forward(mx.array([token_ids], dtype=mx.int32))
    punc_ids = model.argmax_punc_ids(logits)[0]
    _write_npz(
        out_dir,
        entry["key"],
        "native",
        {
            "token_ids": np.asarray(token_ids, dtype=np.int32),
            "punc_ids": np.asarray(punc_ids, dtype=np.int32),
            "logits": np.asarray(logits[0]).astype(np.float32),
        },
    )
    _write_json(
        out_dir,
        entry["key"],
        "native",
        {
            "family": "ct-punc",
            "text": text,
            "punc_array": [int(value) for value in punc_ids],
        },
    )


# --------------------------------------------------------------------------- #
# campplus
# --------------------------------------------------------------------------- #
def run_campplus_reference(
    entry: Mapping[str, Any], assets: Mapping[str, Any], out_dir: Path
) -> None:
    """Official ``CAMPPlus.forward`` plus the reference's own ``extract_feature``.

    ``extract_feature`` applies ``num_mel_bins=80`` fbank and then **subtracts the
    per-utterance mean**.  That subtraction is part of the reference frontend, so
    it is reproduced on the native side too rather than delegated.

    The published campplus checkpoint is ``campplus_cn_common.bin``, not
    ``model.pt``, so ``AutoModel`` -- which looks for the conventional name --
    cannot load it.  When a ``weights`` asset is given the module is built
    directly from ``model_conf`` and that file is loaded with ``weights_only``.
    """
    import numpy as np
    import torch
    from funasr.models.campplus.utils import extract_feature

    raw = _load_yaml(_require(assets, "config"))
    module, backend = _build_campplus(assets, raw)
    utterance = str(assets.get("speaker_a") or entry["audio"])
    samples = _reference_waveform(utterance)
    _check_audio(entry, samples)

    speech, _lengths, _times = extract_feature([torch.from_numpy(samples)])
    with torch.no_grad():
        embedding = module.forward(speech.to(torch.float32))

    arrays: dict[str, Any] = {
        "frontend": speech.detach().cpu().numpy().astype(np.float32),
        "embedding": np.asarray(embedding[0].detach().cpu().numpy(), dtype=np.float32),
    }
    other = assets.get("speaker_b")
    if other:
        other_samples = _reference_waveform(str(other))
        second_speech, _l, _t = extract_feature([torch.from_numpy(other_samples)])
        with torch.no_grad():
            second = module.forward(second_speech.to(torch.float32))
        arrays["embedding_b"] = np.asarray(second[0].detach().cpu().numpy(), dtype=np.float32)
        arrays["cosine"] = np.asarray([_cosine(embedding[0], second[0])], dtype=np.float32)
    _write_npz(out_dir, entry["key"], "reference", arrays)
    _write_json(
        out_dir,
        entry["key"],
        "reference",
        {
            "family": "campplus",
            "backend": backend,
            "utterance": utterance,
            "output_level": getattr(module, "output_level", None),
            "feature_frames": int(speech.shape[1]),
            "cosine_source": "explicit dot/norm; funasr.models.campplus.utils has no "
            "cosine helper, so this is plain arithmetic on the two "
            "official embeddings rather than a reimplementation",
        },
    )


def _build_campplus(assets: Mapping[str, Any], raw: Mapping[str, Any]) -> tuple[Any, str]:
    """The official ``CAMPPlus``, loaded from an explicitly named weights file."""
    from funasr.models.campplus.model import CAMPPlus

    weights = assets.get("weights")
    if not weights:
        raise ValueError(
            "campplus reference needs a 'weights' asset: the published checkpoint is "
            "campplus_cn_common.bin, which AutoModel does not look for"
        )
    module = CAMPPlus(**raw["model_conf"])
    state = _load_state_dict(_require(assets, "weights"))
    module.load_state_dict(state, strict=True)
    module.float().eval()
    return module, f"CAMPPlus(**model_conf) + weights_only load of {Path(str(weights)).name}"


def _cosine(first: Any, second: Any) -> float:
    """Cosine of two 1-D embeddings."""
    import numpy as np

    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denominator == 0.0:
        raise ValueError("zero-norm embedding cannot be scored by cosine")
    return float((a * b).sum() / denominator)


def run_campplus_native(entry: Mapping[str, Any], assets: Mapping[str, Any], out_dir: Path) -> None:
    """Native ``CAMPPlus.embed`` with the reference's mean-subtracted features."""
    import mlx.core as mx
    import numpy as np
    from funasr_mlx.frontend import FbankConfig, WavFrontend, read_audio
    from funasr_mlx.pipeline import load_model

    model, config = load_model(
        _require(assets, "checkpoint"), family="campplus", config=assets.get("config")
    )
    options = dict(config.get("frontend_conf") or {}) if isinstance(config, Mapping) else {}
    from funasr_mlx.models.campplus import CAMPPLUS_FRONTEND_DEFAULTS

    for key, value in CAMPPLUS_FRONTEND_DEFAULTS.items():
        options.setdefault(key, value)
    frontend = WavFrontend(FbankConfig.from_dict(options))
    utterance = str(assets.get("speaker_a") or entry["audio"])
    samples = np.asarray(read_audio(utterance), dtype=np.float32)
    _check_audio(entry, samples)

    # WavFrontend is callable and returns (T, n_mels); the reference's
    # extract_feature subtracts the mean over the time axis.
    features = frontend(samples)
    features = features - mx.mean(features, axis=0, keepdims=True)
    embedding = model(features[None, ...])

    arrays: dict[str, Any] = {
        "frontend": np.asarray(features[None], dtype=np.float32),
        "embedding": np.asarray(embedding[0]).astype(np.float32),
    }
    other = assets.get("speaker_b")
    if other:
        other_features = frontend(np.asarray(read_audio(str(other)), dtype=np.float32))
        other_features = other_features - mx.mean(other_features, axis=0, keepdims=True)
        second = model(other_features[None, ...])
        arrays["embedding_b"] = np.asarray(second[0]).astype(np.float32)
        arrays["cosine"] = np.asarray([_cosine(embedding[0], second[0])], dtype=np.float32)
    _write_npz(out_dir, entry["key"], "native", arrays)
    _write_json(
        out_dir,
        entry["key"],
        "native",
        {
            "family": "campplus",
            "ignored_buffers": list(getattr(model, "_checkpoint_ignored_buffers", ())),
            "utterance": utterance,
            "output_level": getattr(model, "output_level", None),
        },
    )


# --------------------------------------------------------------------------- #
# emotion2vec
# --------------------------------------------------------------------------- #
def run_emotion2vec_reference(
    entry: Mapping[str, Any], assets: Mapping[str, Any], out_dir: Path
) -> None:
    """Official ``Emotion2vec``: frame features, pooled embedding, logits, probs.

    The config is **not** inferred.  It is whatever ``AutoModel`` loaded, because
    the released ``emotion2vec_plus_large`` differs from the template in depth,
    head count, alibi width and the feature spec, and guessing any of those would
    silently compare different models.
    """
    import numpy as np
    import torch

    auto = _load_reference(assets)
    module = auto.model
    cfg = module.cfg
    labels = _read_labels(_require(assets, "labels"))

    samples = _reference_waveform(entry["audio"])
    _check_audio(entry, samples)
    source = torch.from_numpy(samples)
    if bool(cfg.get("normalize", True)):
        source = torch.nn.functional.layer_norm(source, source.shape)
    source = source.view(1, -1)

    with torch.no_grad():
        # remove_extra_tokens=False: the adapter strips them itself below, from
        # the config it recorded. extract_features' own default (True) would
        # already have stripped them, and stripping twice loses 10 frames.
        feats = module.extract_features(source, padding_mask=None, remove_extra_tokens=False)["x"]
        num_extra = int(cfg.get("modalities", {}).get("audio", {}).get("num_extra_tokens", 0))
        frames = feats[:, num_extra:, :]
        pooled = frames.mean(dim=1)
        logits = module.proj(pooled) if getattr(module, "proj", None) is not None else None

    arrays: dict[str, Any] = {
        "frames": np.asarray(frames[0].detach().cpu().numpy(), dtype=np.float32),
        "pooled": np.asarray(pooled[0].detach().cpu().numpy(), dtype=np.float32),
    }
    report: dict[str, Any] = {
        "family": "emotion2vec",
        "duration_seconds": samples.size / SAMPLE_RATE,
        "n_frames": int(frames.shape[1]),
        "num_extra_tokens": num_extra,
        "labels": labels,
        "model_dir": _reference_model_dir(assets),
    }
    if logits is not None:
        probabilities = torch.softmax(logits.float(), dim=-1)
        arrays["logits"] = np.asarray(logits[0].detach().cpu().numpy(), dtype=np.float32)
        arrays["probabilities"] = np.asarray(
            probabilities[0].detach().cpu().numpy(), dtype=np.float32
        )
        keep = [i for i, label in enumerate(labels) if not label.startswith("unuse")]
        report["selected_labels"] = [labels[i] for i in keep]
        report["selected_scores"] = [float(probabilities[0, i]) for i in keep]
    else:
        report["selected_labels"] = []
        report["selected_scores"] = []
        report["note"] = "checkpoint has no proj head; frame/pooled features only"
    _write_npz(out_dir, entry["key"], "reference", arrays)
    _write_json(out_dir, entry["key"], "reference", report)


def _read_labels(path: str) -> list[str]:
    """One label per line, blank lines dropped, order preserved.

    Order is the whole point: the label list is positional against the
    classifier's output rows, so it is read verbatim rather than sorted.
    """
    return [line.rstrip("\n") for line in Path(path).read_text().splitlines() if line]


# --------------------------------------------------------------------------- #
# official YAML -> native config mapping
# --------------------------------------------------------------------------- #
#: ``official yaml path`` -> ``native Emotion2VecConfig field``.  Written out in
#: full rather than splatted with ``**`` because the two schemas share almost no
#: names: the official file is a fairseq/dat2v2 config nested under
#: ``model_conf``/``modalities.*`` and carries dozens of training-only keys the
#: native dataclass would reject outright.
_EMOTION_FIELD_MAP: Mapping[str, str] = {
    "model_conf.embed_dim": "embed_dim",
    "model_conf.depth": "depth",
    "model_conf.num_heads": "num_heads",
    "model_conf.mlp_ratio": "mlp_ratio",
    "model_conf.norm_eps": "norm_eps",
    "model_conf.norm_affine": "norm_affine",
    "model_conf.layer_norm_first": "layer_norm_first",
    "model_conf.end_of_block_targets": "end_of_block_targets",
    "model_conf.normalize": "normalize",
    "model_conf.modalities.audio.extractor_mode": "extractor_mode",
    "model_conf.modalities.audio.feature_encoder_spec": "feature_encoder_spec",
    "model_conf.modalities.audio.conv_pos_depth": "conv_pos_depth",
    "model_conf.modalities.audio.conv_pos_width": "conv_pos_width",
    "model_conf.modalities.audio.conv_pos_groups": "conv_pos_groups",
    "model_conf.modalities.audio.conv_pos_pre_ln": "conv_pos_pre_ln",
    "model_conf.modalities.audio.prenet_depth": "prenet_depth",
    "model_conf.modalities.audio.num_extra_tokens": "num_extra_tokens",
    "model_conf.modalities.audio.init_extra_token_zero": "init_extra_token_zero",
    "model_conf.modalities.audio.use_alibi_encoder": "use_alibi_encoder",
    "model_conf.modalities.audio.num_alibi_heads": "num_alibi_heads",
    # the only name that differs while meaning the same thing
    "model_conf.modalities.audio.alibi_scale": "alibi_scale_init",
    "model_conf.modalities.audio.alibi_max_pos": "alibi_max_pos",
    "model_conf.modalities.audio.learned_alibi": "learned_alibi",
    "model_conf.modalities.audio.learned_alibi_scale": "learned_alibi_scale",
    "model_conf.modalities.audio.learned_alibi_scale_per_head": "learned_alibi_scale_per_head",
    "model_conf.modalities.audio.learned_alibi_scale_per_layer": "learned_alibi_scale_per_layer",
}

#: Top-level yaml sections that are legitimately not architecture.
_EMOTION_SECTIONS = {"model", "tokenizer", "tokenizer_conf", "scope_map", "model_conf"}

#: Keys the official config leaves to the native default.  Reported, not silent.
_EMOTION_DEFAULTED = {
    "model_conf.modalities.audio.conv_bias": False,
}


def _dig(mapping: Mapping[str, Any], dotted: str) -> Any:
    node: Any = mapping
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def emotion_native_config(
    raw: Mapping[str, Any], labels: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map the official ``Emotion2vec`` yaml onto native config kwargs.

    Returns ``(kwargs, report)``.  Every official key is placed in exactly one
    of ``mapped``/``defaulted``/``unmapped``/``ignored_sections`` -- **nothing is
    dropped silently**, because a config key that quietly goes nowhere is how a
    model ends up comparing against different weights than anyone thinks.
    """
    kwargs: dict[str, Any] = {}
    mapped: list[str] = []
    for source, target in _EMOTION_FIELD_MAP.items():
        try:
            kwargs[target] = _dig(raw, source)
        except KeyError:
            continue
        mapped.append(source)

    defaulted = {
        source: value
        for source, value in _EMOTION_DEFAULTED.items()
        if _dig_or_none(raw, source) is None
    }
    for source, value in defaulted.items():
        kwargs[_EMOTION_FIELD_MAP.get(source, source.rsplit(".", 1)[-1])] = value

    kwargs["labels"] = tuple(labels)
    kwargs["vocab_size"] = int(len(labels))

    consumed = {source.split(".")[0] for source in mapped}
    consumed |= {source.split(".")[1] for source in mapped if source.startswith("model_conf.")}
    ignored_sections = sorted(set(raw) & _EMOTION_SECTIONS - {"model_conf"})

    unmapped = _emotion_unmapped_keys(raw, mapped, defaulted)

    report = {
        "mapped": sorted(mapped),
        "defaulted": sorted(defaulted),
        "unmapped": unmapped,
        "ignored_sections": ignored_sections,
        "scope_map": list(raw.get("scope_map") or []),
        "n_native_fields_set": len(kwargs),
    }
    _emotion_consistency_checks(raw, report)
    return kwargs, report


def _dig_or_none(mapping: Mapping[str, Any], dotted: str) -> Any:
    try:
        return _dig(mapping, dotted)
    except KeyError:
        return None


def _emotion_unmapped_keys(
    raw: Mapping[str, Any], mapped: Sequence[str], defaulted: Mapping[str, Any]
) -> dict[str, Any]:
    """Official leaf keys that reach no native field, grouped by why.

    They are not an error -- most are fairseq training knobs -- but an
    unclassified leftover would be indistinguishable from a bug, so each is
    bucketed.
    """
    covered = set(mapped) | set(defaulted)
    audio = (raw.get("model_conf") or {}).get("modalities", {}).get("audio") or {}
    top = raw.get("model_conf") or {}

    ema, objective, optimisation, other_modalities, audio_unmapped = [], [], [], [], []
    for key, value in top.items():
        if key == "modalities" or key in {
            "embed_dim",
            "depth",
            "num_heads",
            "mlp_ratio",
            "norm_eps",
            "norm_affine",
            "layer_norm_first",
            "end_of_block_targets",
            "normalize",
            "_name",
        }:
            continue
        if key.startswith("ema") or key in {"skip_ema", "max_update", "log_norms"}:
            ema.append(key)
        elif key.startswith("loss_") or key in {
            "d2v_loss",
            "recon_loss",
            "cls_loss",
            "cls_type",
            "shared_decoder",
            "adversarial_training",
            "adversarial_hidden_dim",
            "adversarial_weight",
            "decoder_group",
            "min_target_var",
            "min_pred_var",
            "mae_init",
            "layer_norm_target_layer",
            "batch_norm_target_layer",
            "instance_norm_target_layer",
            "instance_norm_targets",
            "layer_norm_targets",
        }:
            objective.append(key)
        elif "dropout" in key or "drop_path" in key or key == "layerdrop":
            optimisation.append(key)
        else:
            other_modalities.append(key)
    for modality in sorted((raw.get("model_conf") or {}).get("modalities", {})):
        if modality not in {"audio", "_name"}:
            other_modalities.append(f"modalities.{modality}")
    for key in sorted(audio):
        if f"model_conf.modalities.audio.{key}" not in covered:
            audio_unmapped.append(key)

    return {
        "ema_and_optimizer": sorted(ema),
        "pretraining_objectives": sorted(objective),
        "regularisation": sorted(optimisation),
        "other_modalities_and_top_level": sorted(other_modalities),
        "audio_modality_not_mapped": audio_unmapped,
    }


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a yaml mapping, got {type(data).__name__}")
    return data


def _load_state_dict(path: str) -> dict[str, Any]:
    """Safe torch state-dict read.

    ``weights_only=True`` blocks arbitrary unpickling and ``mmap=True`` keeps a
    multi-GB checkpoint off the heap.  ``mmap`` is only available on recent
    torch builds for zip-serialised files, so its absence degrades to a normal
    load rather than failing the run.
    """
    import torch

    try:
        return dict(torch.load(path, map_location="cpu", weights_only=True, mmap=True))
    except TypeError:  # older torch without mmap
        return dict(torch.load(path, map_location="cpu", weights_only=True))


def _load_ct_punc_tokens(path: str) -> list[str]:
    """The ct-punc character vocabulary.

    The published ``tokens.json`` is a **JSON list** (471067 entries), not a
    newline-separated text file.  Reading it line by line would silently produce
    a garbage vocabulary with commas and quotes as tokens, so the format is
    detected rather than assumed.
    """
    import json

    text = Path(path).read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list) or not all(isinstance(v, str) for v in data):
            raise ValueError(f"{path}: expected a JSON list of strings")
        return data
    return [line.rstrip("\n") for line in text.splitlines() if line]


def _emotion_consistency_checks(raw: Mapping[str, Any], report: dict[str, Any]) -> None:
    """Cross-check redundant official fields and record any disagreement."""
    audio = (raw.get("model_conf") or {}).get("modalities", {}).get("audio") or {}
    checks: dict[str, Any] = {}
    model_depth = audio.get("model_depth")
    depth = (raw.get("model_conf") or {}).get("depth")
    if model_depth is not None and depth is not None:
        checks["model_depth_vs_depth"] = {
            "model_depth": model_depth,
            "depth": depth,
            "agree": model_depth == depth,
        }
    declared = (raw.get("model_conf") or {}).get("supported_modality")
    if declared is not None:
        checks["supported_modality"] = declared
        present = sorted(
            m for m in (raw.get("model_conf") or {}).get("modalities", {}) if m not in {"_name"}
        )
        checks["modalities_present"] = present
        checks["modalities_beyond_declared"] = [m for m in present if m != declared.lower()]
    report["consistency"] = checks


def run_emotion2vec_native(
    entry: Mapping[str, Any], assets: Mapping[str, Any], out_dir: Path
) -> None:
    """Native emotion forward via the shipping loader, with the disposition report.

    The official yaml is mapped field by field rather than splatted into the
    native dataclass: it is a fairseq/dat2v2 config whose ``model_conf`` holds
    dozens of training-only keys (ema, losses, mask probabilities) and whose
    ``modalities`` section carries unused ``image`` and ``text`` branches.
    ``Emotion2VecConfig(**model_conf)`` would reject all of them, and silently
    filtering them first would hide a key that should have mapped.
    """
    import mlx.core as mx
    import numpy as np
    from funasr_mlx.frontend import read_audio
    from funasr_mlx.models.emotion2vec import flat_parameter_names, preprocess_waveform
    from funasr_mlx.pipeline import load_model

    raw = _load_yaml(_require(assets, "config"))
    labels = _read_labels(_require(assets, "labels"))
    kwargs, config_report = emotion_native_config(raw, labels)

    model, config = load_model(_require(assets, "checkpoint"), family="emotion2vec", config=kwargs)

    samples = np.asarray(read_audio(entry["audio"]), dtype=np.float32)
    _check_audio(entry, samples)
    waveform = preprocess_waveform(
        mx.array(samples[None, :]), normalize=bool(getattr(config, "normalize", True))
    )
    # classify() returns (pooled features, logits, frames) in that order.
    pooled, logits, frames = model.classify(waveform)
    probabilities = mx.softmax(logits, axis=-1)

    _write_npz(
        out_dir,
        entry["key"],
        "native",
        {
            "frames": np.asarray(frames[0]).astype(np.float32),
            "pooled": np.asarray(pooled[0]).astype(np.float32),
            "logits": np.asarray(logits[0]).astype(np.float32),
            "probabilities": np.asarray(probabilities[0]).astype(np.float32),
        },
    )
    keep = [i for i, label in enumerate(labels) if not label.startswith("unuse")]
    _write_json(
        out_dir,
        entry["key"],
        "native",
        {
            "family": "emotion2vec",
            "duration_seconds": samples.size / SAMPLE_RATE,
            "n_frames": int(frames.shape[1]),
            "labels": labels,
            "selected_labels": [labels[i] for i in keep],
            "selected_scores": [float(probabilities[0, i]) for i in keep],
            "config_map": config_report,
            "disposition": _native_parameter_disposition(flat_parameter_names(model)),
        },
    )


def _native_parameter_disposition(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Which native parameter groups exist, so the EMA story is checkable.

    ``ema_decay`` keeps an EMA copy of the encoder in the official checkpoint and
    the d2v2 decoder is a pretraining head unused at inference.  Neither exists
    natively, which is correct; this records the native key space so the absence
    is visible rather than looking like a silent mismatch.
    """
    keys = sorted(parameters)
    return {
        "n_native_parameters": len(keys),
        "has_proj_head": any(name.startswith("proj.") for name in keys),
        "feature_extractor": [n for n in keys if ".local_encoder." in n],
        "positional": [n for n in keys if ".relative_positional_encoder." in n],
        "blocks": [n for n in keys if n.startswith("blocks.")],
        "ema_keys": [n for n in keys if "ema" in n.lower()],
        "decoder_keys": [n for n in keys if "decoder" in n],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_DISPATCH: dict[tuple[str, str], Any] = {
    ("fsmn-vad", "reference"): run_fsmn_vad_reference,
    ("fsmn-vad", "native"): run_fsmn_vad_native,
    ("ct-punc", "reference"): run_ct_punc_reference,
    ("ct-punc", "native"): run_ct_punc_native,
    ("campplus", "reference"): run_campplus_reference,
    ("campplus", "native"): run_campplus_native,
    ("emotion2vec", "reference"): run_emotion2vec_reference,
    ("emotion2vec", "native"): run_emotion2vec_native,
}


def run_manifest(
    manifest: str | Path, assets: str | Path, out_dir: str | Path, mode: str
) -> list[Path]:
    """Run every manifest entry in ``mode``; return the JSON paths written."""
    if mode not in {"reference", "native"}:
        raise ValueError(f"mode must be reference or native, got {mode!r}")
    entries = json.loads(Path(manifest).read_text())
    if not isinstance(entries, list):
        raise ValueError("manifest must be a JSON list of entries")
    asset_map = json.loads(Path(assets).read_text())
    if not isinstance(asset_map, dict):
        raise ValueError("assets must be a JSON object of paths")

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for entry in entries:
        family = str(entry.get("family", ""))
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
        key = str(entry["key"])
        _DISPATCH[(family, mode)](entry, asset_map, destination)
        written.append(destination / f"{key}.{mode}.json")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("reference", "native"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    for path in run_manifest(arguments.manifest, arguments.assets, arguments.out, arguments.mode):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
