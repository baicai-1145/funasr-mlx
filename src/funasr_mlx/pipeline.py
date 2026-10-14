"""Port notes: docs/port-notes/pipeline.py.md"""

from __future__ import annotations

import importlib
import json
import os
import warnings
from pathlib import Path

from .checkpoints import (
    DERIVED_NATIVE_PARAMETERS,
    checkpoint_files,
    inspect_checkpoint,
    safetensor_header,
)
MB = 1_000_000


def read_config(path: str | Path) -> dict:
    import json

    import yaml

    path = Path(path)
    if path.stat().st_size > 1_000_000:
        raise ValueError("Model configuration exceeds the metadata bound")
    text = path.read_text()
    try:
        config = json.loads(text)
    except ValueError:
        config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("Model configuration must be a mapping")
    return config


def create_model(family: str, config: dict):
    if not isinstance(config, dict):
        raise ValueError("Model configuration must be a mapping")
    if family == "sensevoice":
        module = importlib.import_module(".models.sensevoice", __package__)
        return module.SenseVoiceSmall(module.sensevoice_config_from_dict(config))
    if family == "paraformer":
        module = importlib.import_module(".models.paraformer", __package__)
        return module.Paraformer(module.paraformer_config_from_dict(config))
    if family == "ct-punc":
        module = importlib.import_module(".models.ct_punc", __package__)
        return module.CtPuncModel(module.CtPuncConfig.from_dict(config))
    if family == "fsmn-vad":
        module = importlib.import_module(".models.fsmn_vad", __package__)
        return module.FsmnVadModel.from_dict(config)
    if family == "campplus":
        module = importlib.import_module(".models.campplus", __package__)
        return module.CAMPPlus(module.campplus_config_from_dict(config))
    if family == "whisper":
        module = importlib.import_module(".models.whisper", __package__)
        return module.Whisper(module.whisper_config_from_dict(config).dims)
    if family == "emotion2vec":
        module = importlib.import_module(".models.emotion2vec", __package__)
        options = config.get("model_conf", config)
        if not isinstance(options, dict):
            raise ValueError("emotion2vec configuration must be a mapping")
        try:
            parsed = module.Emotion2VecConfig(**options)
        except TypeError as error:
            raise ValueError(f"Invalid emotion2vec configuration: {error}") from error
        return module.Emotion2VecModel(parsed)
    if family == "fun-asr-nano":
        module = importlib.import_module(".models.fun_asr_nano", __package__)
        return module.FunASRNano(module.nano_config_from_dict(config))
    if family == "qwen3-asr":
        module = importlib.import_module(".models.qwen3_asr_thinker", __package__)
        return module.Qwen3ASR(module.qwen3asr_config_from_dict(config))
    if family == "glm-asr-nano":
        module = importlib.import_module(".models.glm_asr_thinker", __package__)
        return module.GlmAsrThinker(module.glmasr_thinker_config_from_dict(config))
    if family == "moss-transcribe-diarize":
        module = importlib.import_module(".models.moss_transcribe", __package__)
        text = importlib.import_module(".models.llm.qwen3", __package__)
        return module.MossTranscribeDiarize(
            config["audio_config"],
            text.Qwen3ModelArgs.from_dict(config["text_config"]),
            audio_merge_size=int(config.get("audio_merge_size", 4)),
            audio_token_id=int(config.get("audio_token_id", 151671)),
        )
    raise ValueError(f"Unknown model family: {family}")


def load_model(checkpoint: str | Path, *, family: str, config: str | Path | dict | None = None):
    # see docs/port-notes/functions.md :: load_model
    checkpoint = Path(checkpoint)
    info = inspect_checkpoint(checkpoint)
    if config is None:
        directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
        candidates = [directory / "config.json", directory / "config.yaml"]
        config = next((path for path in candidates if path.is_file()), None)
        if config is None:
            raise ValueError("Supply a matching model configuration; defaults are not guessed")
    config_dict = config if isinstance(config, dict) else read_config(config)
    headers = {}
    files = tuple(checkpoint_files(checkpoint))
    # see docs/port-notes/comments.md :: pipeline.py:113
    shard_bound = max(256 * MB, info.largest_tensor_bytes) + 32 * MB
    for path in files:
        if path.stat().st_size > shard_bound:
            raise ValueError(
                "Checkpoint shard is too large; re-shard with bounded buffers"
            )
        header = safetensor_header(path)
        metadata = header.pop("__metadata__", {})
        if metadata.get("format") != "funasr-mlx" or metadata.get("model") != family:
            raise ValueError(
                "Checkpoint format/model metadata does not match requested native model"
            )
        headers.update(header)
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    model = create_model(family, config_dict)
    # see docs/port-notes/comments.md :: pipeline.py:132
    from .models.llm.quant import apply_quantization

    quantization = apply_quantization(model, config_dict.get("quantization"))
    if quantization is not None:
        model._quantization = quantization
    expected = dict(tree_flatten(model.parameters()))
    # see docs/port-notes/comments.md :: pipeline.py:139
    derived = set(DERIVED_NATIVE_PARAMETERS.get(family, ()))
    unknown_derived = sorted(derived - set(expected))
    if unknown_derived:
        raise KeyError(f"Declared derived parameters are not model parameters: {unknown_derived}")
    for name in derived:
        del expected[name]
    ignored = set()
    if family == "campplus":
        from .models.campplus import TORCH_ONLY_BUFFERS, BatchNorm

        ignored = {
            f"{name}.{buffer}"
            for name, child in model.named_modules()
            if isinstance(child, BatchNorm)
            for buffer in TORCH_ONLY_BUFFERS
            if f"{name}.{buffer}" in headers
        }
        for name in ignored:
            if headers[name]["shape"] != [] or headers[name]["dtype"] != "I64":
                raise ValueError(f"Invalid BatchNorm training counter: {name}")
    missing = sorted(set(expected) - set(headers))
    unexpected = sorted(set(headers) - set(expected) - ignored)
    if missing or unexpected:
        raise KeyError(f"Checkpoint keys differ: missing={missing}, unexpected={unexpected}")
    for name, tensor in expected.items():
        if tuple(tensor.shape) != tuple(headers[name]["shape"]):
            raise ValueError(
                f"Checkpoint shape mismatch for {name}: {headers[name]['shape']} vs {tensor.shape}"
            )
    del expected
    applied = set()
    for path in files:
        tensors = mx.load(str(path))
        if not isinstance(tensors, dict):
            raise ValueError("Expected a named safetensors shard")
        if applied.intersection(tensors):
            raise ValueError("Duplicate tensors while loading")
        for name in ignored.intersection(tensors):
            del tensors[name]
        model.update(tree_unflatten(list(tensors.items())))
        applied.update(tensors)
        del tensors
    if applied != set(headers) - ignored:
        raise KeyError("Incomplete checkpoint loading")
    model.eval()
    mx.eval(model.parameters())
    mx.clear_cache()
    if derived:
        model._derived_parameters = tuple(sorted(derived))
    if ignored:
        model._checkpoint_ignored_buffers = tuple(sorted(ignored))
        warnings.warn(
            f"Ignored {len(ignored)} CAM++ BatchNorm training counters; "
            "names recorded in model._checkpoint_ignored_buffers",
            stacklevel=2,
        )
    return model, config_dict
