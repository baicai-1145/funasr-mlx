import json
import sys

import numpy as np
import pytest
from safetensors.numpy import save_file

from funasr_mlx.cli import main
from funasr_mlx.checkpoints import convert_checkpoint, inspect_checkpoint


def test_cli_inspects_without_loading_models(tmp_path, capsys):
    path = tmp_path / "weights.safetensors"
    save_file({"weight": np.zeros((3, 4), np.float16)}, str(path))
    assert main(["inspect", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["tensor_count"] == 1
    assert result["tensor_bytes"] == 24


def test_small_conversion_matches_layout_dtype_and_metadata(tmp_path):
    import torch
    from safetensors import safe_open

    source = tmp_path / "source.pt"
    destination = tmp_path / "converted"
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 1, 3)
    torch.save({"state_dict": {"encoder.a.fsmn_block.weight": weight}}, source)
    info = convert_checkpoint(source, destination, model="sensevoice")
    assert info.tensor_bytes == 24
    assert info.tensor_count == 1
    with safe_open(info.files[0], framework="numpy") as handle:
        actual = handle.get_tensor("encoder.a.fsmn_block.weight")
        assert actual.shape == (4, 3, 1)
        assert actual.dtype == np.float16
        assert handle.metadata() == {"format": "funasr-mlx", "model": "sensevoice"}
        np.testing.assert_array_equal(actual, weight.numpy().transpose(0, 2, 1))
    assert inspect_checkpoint(destination).tensor_bytes == 24


def test_conversion_accepts_canonical_whisper_wrapper(tmp_path):
    import torch
    from safetensors import safe_open

    source = tmp_path / "tiny-whisper.pt"
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    torch.save(
        {"dims": {"n_mels": 2}, "model_state_dict": {"encoder.conv1.weight": weight}}, source
    )
    info = convert_checkpoint(source, tmp_path / "whisper", model="whisper", dtype="float32")
    with safe_open(info.files[0], framework="numpy") as handle:
        np.testing.assert_array_equal(
            handle.get_tensor("encoder.conv1.weight"), weight.numpy().transpose(0, 2, 1)
        )


def test_features_resolves_relative_cmvn_and_uses_configured_sample_rate(tmp_path):
    import mlx.core as mx
    import soundfile as sf
    from funasr_mlx.frontend import FbankConfig, WavFrontend, read_audio

    assets = tmp_path / "assets"
    assets.mkdir()
    config_path = assets / "config.json"
    config_path.write_text(
        json.dumps(
            {"frontend_conf": {"fs": 8000, "n_mels": 4, "max_seconds": 0.2, "cmvn_file": "am.mvn"}}
        )
    )
    (assets / "am.mvn").write_text(
        "<AddShift> 4 4\n<LearnRateCoef> 0 [ 1 2 3 4 ]\n"
        "<Rescale> 4 4\n<LearnRateCoef> 0 [ 2 3 4 5 ]\n"
    )
    audio = tmp_path / "short.wav"
    samples = 0.1 * np.sin(np.arange(800) * 2 * np.pi * 440 / 8000)
    sf.write(audio, samples, 8000)
    destination = tmp_path / "features_without_suffix"
    assert main(["features", str(audio), str(destination), "--config", str(config_path)]) == 0
    plain = WavFrontend(FbankConfig(fs=8000, n_mels=4, max_seconds=0.2), cmvn=None)(
        read_audio(audio, sample_rate=8000)
    )
    mx.eval(plain)
    expected = (np.asarray(plain) + [1, 2, 3, 4]) * [2, 3, 4, 5]
    with np.load(destination) as result:
        np.testing.assert_allclose(result["features"], expected, rtol=1e-6, atol=1e-5)
    assert not destination.with_suffix(".npz").exists()


def test_features_honors_configured_duration_admission(tmp_path):
    import soundfile as sf

    audio = tmp_path / "too-long.wav"
    sf.write(audio, np.zeros(800, np.float32), 8000)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"fs": 8000, "n_mels": 4, "max_seconds": 0.05}))
    with pytest.raises(ValueError):
        main(["features", str(audio), str(tmp_path / "unused"), "--config", str(config)])
    assert not (tmp_path / "unused").exists()


def test_malformed_index_is_rejected(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    with pytest.raises(ValueError, match="weight_map"):
        inspect_checkpoint(tmp_path)


def test_cli_import_does_not_load_mlx_or_torch():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,funasr_mlx.cli; assert 'mlx.core' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
