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
A JSON object of filesystem paths supplied by the lead.

``model_dir``
    A local FunASR model directory (``config.yaml`` plus the weights file).
    Required for ``--mode reference``: ``AutoModel`` expects the standard
    directory layout and copying weights around to fake one would be worse than
    asking for it.
``labels``
    Label list, one per line, read verbatim and in order (emotion2vec).
``tokens``
    ct-punc character vocabulary, only needed by the native side.
``speaker_a`` / ``speaker_b``
    Two utterances for a campplus cosine pair.
"""
from __future__ import annotations

import argparse
import json
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
# private output helpers (this module deliberately exposes no shared API)
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


def _reject_overlong(entry: Mapping[str, Any], actual: float, limit_field: str,
                     limit: float, unit: str) -> None:
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

    return AutoModel(model=_reference_model_dir(assets), device="cpu", disable_pbar=True)


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
    _reject_overlong(entry, samples.size / SAMPLE_RATE, "limit_seconds",
                     float(entry.get("limit_seconds", 0.0)), "s")


# --------------------------------------------------------------------------- #
# fsmn-vad
# --------------------------------------------------------------------------- #
def run_fsmn_vad_reference(entry: Mapping[str, Any], assets: Mapping[str, Any],
                           out_dir: Path) -> None:
    """Official ``FsmnVADStreaming`` forward: frame posterior + segment endpoints."""
    import numpy as np
    import torch

    auto = _load_reference(assets)
    frontend = _reference_frontend(auto)
    module = auto.model

    samples = _reference_waveform(entry["audio"])
    _check_audio(entry, samples)
    feats = frontend(torch.from_numpy(samples[None, :])).squeeze(0)
    waveform = torch.from_numpy(samples)

    cache: dict[str, Any] = {}
    module.init_cache(cache)
    result = module.forward(
        feats, waveform, cache=cache, is_final=True, is_streaming_input=False,
        output_frame_probs=True,
    )
    segments, frame_probs = _split_vad_result(result, cache)

    _write_npz(out_dir, entry["key"], "reference", {
        "frame_prob": np.asarray(
            [[p.noise_prob, p.speech_prob, p.score] for p in frame_probs],
            dtype=np.float32,
        ).reshape(-1, 3),
        "frame_state": np.asarray([p.frm_state for p in frame_probs], dtype=np.int32),
        "segment_bounds": np.asarray(
            [[s.start_ms, s.end_ms] for s in segments], dtype=np.int32
        ).reshape(-1, 2),
    })
    _write_json(out_dir, entry["key"], "reference", {
        "family": "fsmn-vad",
        "segments_ms": [[int(pair[0]), int(pair[1])] for pair in segments],
        "n_frames": len(frame_probs),
        "duration_seconds": samples.size / SAMPLE_RATE,
        "frontend": type(frontend).__name__,
    })


def _split_vad_result(result: Any, cache: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    """Flatten the reference's segments and lift its frame probs out of the cache.

    Two facts about ``FsmnVADStreaming.forward`` that are easy to get wrong and
    are **not** visible from its signature:

    * it returns ``segments`` as ``list[list[list[int]]]`` -- a *batch* of
      ``[start_ms, end_ms]`` pairs per call -- not speech-segment objects;
    * it never returns the frame probs.  ``output_frame_probs=True`` makes it
      append ``E2EVadFrameProb`` objects to ``cache["stats"].frame_probs``, so
      that is where they have to be read from.
    """
    segments: list[Any] = []
    for batch in result or []:
        for pair in batch:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                raise ValueError(f"unrecognised VAD segment entry: {pair!r}")
            segments.append([int(pair[0]), int(pair[1])])
    stats = cache.get("stats")
    frame_probs = list(getattr(stats, "frame_probs", None) or []) if stats else []
    return segments, frame_probs


def run_fsmn_vad_native(entry: Mapping[str, Any], assets: Mapping[str, Any],
                        out_dir: Path) -> None:
    """Native ``funasr_mlx.models.fsmn_vad.FsmnVadModel`` forward.

    FSMN-VAD consumes **LFR-stacked** fbank: the official ``encoder_conf`` has
    ``input_dim: 400`` against an 80-bin mel, i.e. ``lfr_m = 5``.  That factor is
    read back out of ``input_dim`` rather than assumed, because a wrong stack
    factor changes the encoder input width and would fail deep inside the model
    instead of here.
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

    encoder_conf = _vad_encoder_conf(config)
    input_dim = int(encoder_conf["input_dim"])
    n_mels = 80
    if input_dim % n_mels:
        raise ValueError(
            f"input_dim {input_dim} is not a multiple of the {n_mels}-bin mel filterbank; "
            "the LFR stack factor cannot be derived"
        )
    frontend = WavFrontend(FbankConfig(
        fs=int(encoder_conf.get("sample_rate", SAMPLE_RATE)),
        n_mels=n_mels,
        lfr_m=input_dim // n_mels,
        lfr_n=1,
    ))
    feats = frontend.fbank(samples)[None, ...]  # (1, T, input_dim)

    cache = model.init_cache()
    result = model(
        mx.array(feats), mx.array(samples), cache, is_final=True, is_streaming_input=False
    )
    stats = result[0] if isinstance(result, (list, tuple)) else result
    segments = list(_vad_segments(stats))
    frame_probs = list(getattr(stats, "frame_probs", None) or [])

    arrays: dict[str, Any] = {
        "segment_bounds": np.asarray(
            [[int(s[0]), int(s[1])] for s in segments], dtype=np.int32
        ).reshape(-1, 2),
    }
    if frame_probs:
        arrays["frame_prob"] = np.asarray(
            [[p.noise_prob, p.speech_prob, p.score] for p in frame_probs], dtype=np.float32
        ).reshape(-1, 3)
        arrays["frame_state"] = np.asarray([p.frm_state for p in frame_probs],
                                           dtype=np.int32)
    _write_npz(out_dir, entry["key"], "native", arrays)
    _write_json(out_dir, entry["key"], "native", {
        "family": "fsmn-vad",
        "segments_ms": [[int(s[0]), int(s[1])] for s in segments],
        "n_frames": len(frame_probs),
        "duration_seconds": samples.size / SAMPLE_RATE,
        "lfr_m": input_dim // n_mels,
    })


def _vad_encoder_conf(config: Any) -> Mapping[str, Any]:
    """The FSMN ``encoder_conf`` mapping from whatever ``load_model`` returned."""
    if isinstance(config, Mapping):
        merged = dict(config)
        merged.update(config.get("encoder_conf") or {})
        return merged
    return {}


def _vad_segments(stats: Any) -> list[Any]:
    """Segment ``[start_ms, end_ms]`` pairs, tolerating either container shape.

    The native model returns segments either as ``SpeechSegment`` dataclasses or
    as plain ``[start, end]`` pairs depending on the call path, so both are
    normalised here rather than assumed at the call site.
    """
    raw = getattr(stats, "segments", None)
    if raw is None:
        return []
    out: list[Any] = []
    for item in raw:
        if hasattr(item, "start_ms"):
            out.append([item.start_ms, item.end_ms])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append([item[0], item[1]])
        else:
            raise ValueError(f"unrecognised VAD segment entry: {item!r}")
    return out


# --------------------------------------------------------------------------- #
# ct-punc
# --------------------------------------------------------------------------- #
def run_ct_punc_reference(entry: Mapping[str, Any], assets: Mapping[str, Any],
                          out_dir: Path) -> None:
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
    _reject_overlong(entry, float(len(text)), "limit_chars",
                     float(entry.get("limit_chars", 0.0)), "chars")

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

    _write_npz(out_dir, entry["key"], "reference", {
        "token_ids": np.asarray(token_ids, dtype=np.int32),
        "punc_ids": np.asarray(punc_ids, dtype=np.int32),
        "logits": np.concatenate(blocks, axis=0).astype(np.float32),
    })
    _write_json(out_dir, entry["key"], "reference", {
        "family": "ct-punc",
        "text": text,
        "tokens": tokens,
        "punc_array": punc_ids,
        "n_mini_sentences": len(mini_words),
    })


def run_ct_punc_native(entry: Mapping[str, Any], assets: Mapping[str, Any],
                       out_dir: Path) -> None:
    """Native ``funasr_mlx.models.ct_punc.CtPuncModel.punc_forward``."""
    import mlx.core as mx
    import numpy as np
    from funasr_mlx.models.ct_punc import CharVocab, split_words
    from funasr_mlx.pipeline import load_model

    model, _config = load_model(
        _require(assets, "checkpoint"), family="ct-punc", config=assets.get("config")
    )
    text = str(entry["text"])
    _reject_overlong(entry, float(len(text)), "limit_chars",
                     float(entry.get("limit_chars", 0.0)), "chars")

    token_ids = list(CharVocab.from_file(_require(assets, "tokens")).encode(
        list(split_words(text))
    ))
    logits = model.punc_forward(mx.array([token_ids], dtype=mx.int32))
    punc_ids = model.argmax_punc_ids(logits)[0]
    _write_npz(out_dir, entry["key"], "native", {
        "token_ids": np.asarray(token_ids, dtype=np.int32),
        "punc_ids": np.asarray(punc_ids, dtype=np.int32),
        "logits": np.asarray(logits[0]).astype(np.float32),
    })
    _write_json(out_dir, entry["key"], "native", {
        "family": "ct-punc",
        "text": text,
        "punc_array": [int(value) for value in punc_ids],
    })


# --------------------------------------------------------------------------- #
# campplus
# --------------------------------------------------------------------------- #
def run_campplus_reference(entry: Mapping[str, Any], assets: Mapping[str, Any],
                           out_dir: Path) -> None:
    """Official ``CAMPPlus.forward`` plus the reference's own ``extract_feature``.

    ``extract_feature`` applies ``num_mel_bins=80`` fbank and then **subtracts the
    per-utterance mean**.  That subtraction is part of the reference frontend, so
    it is reproduced on the native side too rather than delegated.
    """
    import numpy as np
    import torch
    from funasr.models.campplus.utils import extract_feature

    auto = _load_reference(assets)
    module = auto.model
    utterance = str(assets.get("speaker_a") or entry["audio"])
    samples = _reference_waveform(utterance)
    _check_audio(entry, samples)

    speech, _lengths, _times = extract_feature([torch.from_numpy(samples)])
    with torch.no_grad():
        embedding = module.forward(speech.to(torch.float32))

    arrays: dict[str, Any] = {
        "embedding": np.asarray(embedding[0].detach().cpu().numpy(), dtype=np.float32),
    }
    other = assets.get("speaker_b")
    if other:
        other_samples = _reference_waveform(str(other))
        second_speech, _l, _t = extract_feature([torch.from_numpy(other_samples)])
        with torch.no_grad():
            second = module.forward(second_speech.to(torch.float32))
        arrays["embedding_b"] = np.asarray(
            second[0].detach().cpu().numpy(), dtype=np.float32
        )
        arrays["cosine"] = np.asarray(
            [_cosine(embedding[0], second[0])], dtype=np.float32
        )
    _write_npz(out_dir, entry["key"], "reference", arrays)
    _write_json(out_dir, entry["key"], "reference", {
        "family": "campplus",
        "utterance": utterance,
        "output_level": getattr(module, "output_level", None),
        "feature_frames": int(speech.shape[1]),
        "cosine_source": "explicit dot/norm; funasr.models.campplus.utils has no "
                         "cosine helper, so this is plain arithmetic on the two "
                         "official embeddings rather than a reimplementation",
    })


def _cosine(first: Any, second: Any) -> float:
    """Cosine of two 1-D embeddings."""
    import numpy as np

    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denominator == 0.0:
        raise ValueError("zero-norm embedding cannot be scored by cosine")
    return float((a * b).sum() / denominator)


def run_campplus_native(entry: Mapping[str, Any], assets: Mapping[str, Any],
                        out_dir: Path) -> None:
    """Native ``CAMPPlus.embed`` with the reference's mean-subtracted features."""
    import mlx.core as mx
    import numpy as np
    from funasr_mlx.frontend import FbankConfig, WavFrontend, read_audio
    from funasr_mlx.pipeline import load_model

    model, config = load_model(
        _require(assets, "checkpoint"), family="campplus", config=assets.get("config")
    )
    frontend = WavFrontend(
        FbankConfig.from_dict(config) if isinstance(config, Mapping) else FbankConfig()
    )
    utterance = str(assets.get("speaker_a") or entry["audio"])
    samples = np.asarray(read_audio(utterance), dtype=np.float32)
    _check_audio(entry, samples)

    # fbank() takes a rank-1 waveform and returns (T, n_mels); add the batch axis.
    feats = frontend.fbank(samples)[None, ...]
    feats = feats - mx.mean(feats, axis=1, keepdims=True)  # extract_feature parity
    embedding = model.embed(feats)

    arrays: dict[str, Any] = {
        "embedding": np.asarray(embedding[0]).astype(np.float32),
    }
    other = assets.get("speaker_b")
    if other:
        other_samples = np.asarray(read_audio(str(other)), dtype=np.float32)
        other_feats = frontend.fbank(other_samples)[None, ...]
        other_feats = other_feats - mx.mean(other_feats, axis=1, keepdims=True)
        second = model.embed(other_feats)
        arrays["embedding_b"] = np.asarray(second[0]).astype(np.float32)
        arrays["cosine"] = np.asarray([_cosine(embedding[0], second[0])],
                                      dtype=np.float32)
    _write_npz(out_dir, entry["key"], "native", arrays)
    _write_json(out_dir, entry["key"], "native", {
        "family": "campplus",
        "utterance": utterance,
        "output_level": getattr(model, "output_level", None),
    })


# --------------------------------------------------------------------------- #
# emotion2vec
# --------------------------------------------------------------------------- #
def run_emotion2vec_reference(entry: Mapping[str, Any], assets: Mapping[str, Any],
                              out_dir: Path) -> None:
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
        feats = module.extract_features(source, padding_mask=None)["x"]
        num_extra = int(cfg.get("modalities", {}).get("audio", {}).get(
            "num_extra_tokens", 0
        ))
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
        report["selected_scores"] = [
            float(probabilities[0, i]) for i in keep
        ]
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


def run_emotion2vec_native(entry: Mapping[str, Any], assets: Mapping[str, Any],
                           out_dir: Path) -> None:
    """Native emotion forward via the shipping loader, with the disposition report."""
    import mlx.core as mx
    import numpy as np
    from funasr_mlx.frontend import read_audio
    from funasr_mlx.models.emotion2vec import flat_parameter_names, preprocess_waveform
    from funasr_mlx.pipeline import load_model

    model, config = load_model(
        _require(assets, "checkpoint"), family="emotion2vec",
        config=assets.get("config"),
    )
    labels = list(getattr(config, "labels", ()) or ())

    samples = np.asarray(read_audio(entry["audio"]), dtype=np.float32)
    _check_audio(entry, samples)
    waveform = preprocess_waveform(
        mx.array(samples[None, :]), normalize=bool(getattr(config, "normalize", True))
    )
    frames, logits, pooled = model.classify(waveform)
    probabilities = mx.softmax(logits, axis=-1)

    _write_npz(out_dir, entry["key"], "native", {
        "frames": np.asarray(frames[0]).astype(np.float32),
        "pooled": np.asarray(pooled[0]).astype(np.float32),
        "logits": np.asarray(logits[0]).astype(np.float32),
        "probabilities": np.asarray(probabilities[0]).astype(np.float32),
    })
    keep = [i for i, label in enumerate(labels) if not label.startswith("unuse")]
    _write_json(out_dir, entry["key"], "native", {
        "family": "emotion2vec",
        "duration_seconds": samples.size / SAMPLE_RATE,
        "n_frames": int(frames.shape[1]),
        "labels": labels,
        "selected_labels": [labels[i] for i in keep],
        "selected_scores": [float(probabilities[0, i]) for i in keep],
        "disposition": _native_parameter_disposition(flat_parameter_names(model)),
    })


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


def run_manifest(manifest: str | Path, assets: str | Path, out_dir: str | Path,
                 mode: str) -> list[Path]:
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
    for path in run_manifest(arguments.manifest, arguments.assets, arguments.out,
                             arguments.mode):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())