"""Strict, local-only model construction and shard loading."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from .checkpoints import checkpoint_files, inspect_checkpoint, safetensor_header
from .memory import MB, MemoryBudgetExceeded, MemoryPolicy, guarded_child_active


def read_config(path: str | Path) -> dict:
    import yaml

    path = Path(path)
    if path.stat().st_size > 1_000_000:
        raise ValueError("Model configuration exceeds the metadata bound")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Model configuration must be a mapping")
    return config


def create_model(family: str, config: dict):
    if not guarded_child_active():
        raise RuntimeError("Model construction requires the guarded launcher")
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
    raise ValueError(f"Unknown model family: {family}")


def load_model(checkpoint: str | Path, *, family: str, config: str | Path | dict | None = None):
    """Prevalidate every key/shape, then load one bounded shard at a time.

    Only native-layout, explicitly tagged checkpoints are admitted. Running in
    a guarded child is mandatory; importing this module does not load MLX.
    """
    if not guarded_child_active():
        raise RuntimeError("Use the guarded CLI to load models")
    policy = MemoryPolicy(**json.loads(os.environ["FUNASR_MLX_MEMORY_POLICY"]))
    checkpoint = Path(checkpoint)
    info = inspect_checkpoint(checkpoint)
    if info.tensor_bytes > int(policy.mlx_limit * 0.65):
        raise MemoryBudgetExceeded(
            "Weights exceed device budget after reserving activation headroom"
        )
    if info.largest_tensor_bytes > policy.process_limit // 4:
        raise MemoryBudgetExceeded("Largest checkpoint tensor exceeds load buffer budget")
    if config is None:
        directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
        candidates = [directory / "config.json", directory / "config.yaml"]
        config = next((path for path in candidates if path.is_file()), None)
        if config is None:
            raise ValueError("Supply a matching model configuration; defaults are not guessed")
    config_dict = config if isinstance(config, dict) else read_config(config)
    headers = {}
    files = tuple(checkpoint_files(checkpoint))
    shard_bound = max(128 * MB, info.largest_tensor_bytes) + 16 * MB
    for path in files:
        if path.stat().st_size > min(shard_bound, policy.mlx_limit - info.tensor_bytes):
            raise MemoryBudgetExceeded(
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
    expected = dict(tree_flatten(model.parameters()))
    missing, unexpected = sorted(set(expected) - set(headers)), sorted(set(headers) - set(expected))
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
        model.update(tree_unflatten(list(tensors.items())))
        applied.update(tensors)
        del tensors
    if applied != set(headers):
        raise KeyError("Incomplete checkpoint loading")
    model.eval()
    mx.eval(model.parameters())
    mx.clear_cache()
    return model, config_dict
