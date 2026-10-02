"""Real local checkpoint/factory/CLI integration with bounded synthetic weights."""

import copy
import json

import numpy as np
import pytest

from funasr_mlx.cli import main
from funasr_mlx.pipeline import create_model, load_model

FRONTEND = {"n_mels": 8, "lfr_m": 1, "lfr_n": 1}
ENCODER = {
    "output_size": 8,
    "attention_heads": 2,
    "linear_units": 16,
    "num_blocks": 2,
    "kernel_size": 3,
    "input_layer": "pe",
}
CONFIGS = {
    "sensevoice": {
        "input_size": 8,
        "vocab_size": 16,
        "encoder_conf": {**ENCODER, "tp_blocks": 1},
        "frontend_conf": FRONTEND,
    },
    "paraformer": {
        "input_size": 8,
        "vocab_size": 16,
        "encoder_conf": ENCODER,
        "decoder_conf": {
            "attention_heads": 2,
            "linear_units": 16,
            "num_blocks": 2,
            "att_layer_num": 1,
            "kernel_size": 3,
        },
        "predictor_conf": {"idim": 8},
        "frontend_conf": FRONTEND,
    },
    "ct-punc": {
        "vocab_size": 16,
        "embed_unit": 8,
        "att_unit": 8,
        "punc_list": ["_", ",", ".", "?"],
        "sentence_end_id": 2,
        "encoder_conf": {**ENCODER, "input_size": 8},
    },
    "fsmn-vad": {
        "encoder_conf": {
            "input_dim": 6,
            "input_affine_dim": 4,
            "fsmn_layers": 2,
            "linear_dim": 6,
            "proj_dim": 4,
            "lorder": 5,
            "rorder": 0,
            "lstride": 1,
            "rstride": 0,
            "output_affine_dim": 4,
            "output_dim": 3,
        },
        "frontend_conf": {"n_mels": 6, "lfr_m": 1, "lfr_n": 1},
    },
    "campplus": {
        "feat_dim": 8,
        "embedding_size": 6,
        "growth_rate": 2,
        "bn_size": 2,
        "init_channels": 4,
        "m_channels": 4,
        "num_blocks": [2, 2],
        "block_spec": [[2, 3, 1], [2, 3, 2]],
        "frontend_conf": FRONTEND,
    },
    "whisper": {
        "dims": {
            "n_mels": 80,
            "n_audio_ctx": 20,
            "n_audio_state": 8,
            "n_audio_head": 2,
            "n_audio_layer": 2,
            "n_vocab": 16,
            "n_text_ctx": 12,
            "n_text_state": 8,
            "n_text_head": 2,
            "n_text_layer": 2,
        }
    },
    "emotion2vec": {
        "embed_dim": 16,
        "depth": 2,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "feature_encoder_spec": "[(8,5,2)] + [(8,3,2)] * 2",
        "conv_pos_depth": 2,
        "conv_pos_width": 12,
        "conv_pos_groups": 4,
        "prenet_depth": 2,
        "num_alibi_heads": 2,
        "num_extra_tokens": 3,
        "max_samples": 4000,
        "max_frames": 64,
        "vocab_size": 3,
        "labels": ["angry", "neutral", "unuse0"],
    },
}


def tiny_checkpoint(root, family, configuration=None):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file

    config = copy.deepcopy(CONFIGS[family] if configuration is None else configuration)
    model = create_model(family, config)
    model.eval()
    parameters = dict(tree_flatten(model.parameters()))
    assert parameters
    assert sum(value.size for value in parameters.values()) < 1_000_000, "Fixture is not tiny"
    mx.eval(parameters)
    arrays = {name: np.asarray(value) for name, value in parameters.items()}
    directory = root / family
    directory.mkdir()
    save_file(
        arrays,
        str(directory / "weights.safetensors"),
        metadata={"format": "funasr-mlx", "model": family},
    )
    (directory / "config.json").write_text(json.dumps(config))
    return directory, arrays


@pytest.mark.parametrize("family", list(CONFIGS))
def test_factory_loads_every_real_native_parameter(family, tmp_path):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    directory, expected = tiny_checkpoint(tmp_path, family)
    loaded, configuration = load_model(directory, family=family)
    actual = dict(tree_flatten(loaded.parameters()))
    assert actual.keys() == expected.keys()
    mx.eval(actual)
    for name, value in actual.items():
        np.testing.assert_array_equal(np.asarray(value), expected[name], err_msg=name)
    assert configuration == CONFIGS[family]


@pytest.mark.parametrize("family", list(CONFIGS))
def test_cli_runs_every_model_with_local_weights(family, tmp_path, capsys):
    import soundfile as sf

    directory, _ = tiny_checkpoint(tmp_path, family)
    arguments = ["infer", str(directory), "--model", family]
    if family == "ct-punc":
        tokens = tmp_path / "tokens.txt"
        tokens.write_text("\n".join(["<unk>", "hello", "world"] + [f"t{i}" for i in range(13)]))
        arguments += ["--text", "hello world", "--tokens", str(tokens)]
    else:
        audio = tmp_path / "audio.wav"
        count = 400 if family == "emotion2vec" else 1600
        samples = 0.1 * np.sin(np.arange(count) * 2 * np.pi * 440 / 16000)
        sf.write(audio, samples, 16000)
        arguments += ["--audio", str(audio)]
        if family == "whisper":
            arguments += ["--prompt-tokens", "1", "--eot-token", "2", "--max-tokens", "1"]
    assert main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    if family in {"sensevoice", "paraformer", "whisper"}:
        assert isinstance(result["token_ids"], list)
    elif family == "ct-punc":
        assert isinstance(result["text"], str)
        assert len(result["punctuation_ids"]) == 2
    elif family == "fsmn-vad":
        assert isinstance(result["segments_ms"], list)
    else:
        assert np.isfinite(np.asarray(result["embedding"])).all()
        if family == "emotion2vec":
            assert result["labels"] == ["angry", "neutral"]
            np.testing.assert_allclose(sum(result["scores"]), 1.0, atol=1e-6)


def test_vad_cli_preserves_frame_cadence_with_default_lfr_window(tmp_path, capsys):
    import mlx.core as mx
    import soundfile as sf
    from funasr_mlx.frontend import apply_lfr

    configuration = copy.deepcopy(CONFIGS["fsmn-vad"])
    configuration["encoder_conf"]["input_dim"] = 30
    configuration["frontend_conf"].pop("lfr_m")
    directory, _ = tiny_checkpoint(tmp_path, "fsmn-vad", configuration)
    assert apply_lfr(mx.zeros((8, 6)), m=5, n=1).shape == (8, 30)
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(1600), 16000)
    assert main(["infer", str(directory), "--model", "fsmn-vad", "--audio", str(audio)]) == 0
    assert isinstance(json.loads(capsys.readouterr().out)["segments_ms"], list)


def test_emotion_cli_rejects_overlong_input_without_truncation(tmp_path):
    import soundfile as sf

    directory, _ = tiny_checkpoint(tmp_path, "emotion2vec")
    audio = tmp_path / "too-long.wav"
    sf.write(audio, np.zeros(4001), 16000)
    with pytest.raises(ValueError, match="input bound"):
        main(["infer", str(directory), "--model", "emotion2vec", "--audio", str(audio)])


@pytest.mark.parametrize("configuration", [None, {"model_conf": []}, {"unknown_key": 1}])
def test_emotion_factory_reports_invalid_configuration(configuration):
    with pytest.raises(ValueError, match="configuration"):
        create_model("emotion2vec", configuration)


def test_emotion_cli_rejects_missing_checkpoint_labels(tmp_path):
    directory, _ = tiny_checkpoint(tmp_path, "emotion2vec")
    configuration = copy.deepcopy(CONFIGS["emotion2vec"])
    configuration["labels"] = []
    (directory / "config.json").write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="exact labels"):
        main(
            [
                "infer",
                str(directory),
                "--model",
                "emotion2vec",
                "--audio",
                str(tmp_path / "not-needed.wav"),
            ]
        )


def test_loader_rejects_metadata_and_shape_mismatch_before_update(tmp_path):
    from safetensors.numpy import save_file

    directory, arrays = tiny_checkpoint(tmp_path, "whisper")
    weights = directory / "weights.safetensors"
    save_file(arrays, str(weights), metadata={"format": "funasr-mlx", "model": "other"})
    with pytest.raises(ValueError, match="metadata"):
        load_model(directory, family="whisper")
    bad = dict(arrays)
    bad["encoder.conv1.weight"] = np.zeros((1, 1, 1), np.float32)
    save_file(bad, str(weights), metadata={"format": "funasr-mlx", "model": "whisper"})
    with pytest.raises(ValueError, match="shape mismatch"):
        load_model(directory, family="whisper")
