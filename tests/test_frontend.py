from pathlib import Path

import numpy as np
import pytest

from funasr_mlx.frontend import FbankConfig, WavFrontend, apply_lfr, load_cmvn, whisper_log_mel


def test_lfr_repeats_edge_frames():
    import mlx.core as mx

    features = mx.array([[1.0], [2.0], [3.0], [4.0]])
    assert apply_lfr(features, 3, 2).tolist() == [[1.0, 1.0, 2.0], [2.0, 3.0, 4.0]]


def test_cmvn_parser_and_transform(tmp_path):
    import mlx.core as mx

    path = tmp_path / "am.mvn"
    path.write_text(
        "<AddShift> 4 4\n<LearnRateCoef> 0 [ 1 2 3 4 ]\n"
        "<Rescale> 4 4\n<LearnRateCoef> 0 [ 2 3 4 5 ]\n"
    )
    result = load_cmvn(path)
    assert np.array_equal(result, [[1, 2, 3, 4], [2, 3, 4, 5]])
    frontend = WavFrontend(FbankConfig(n_mels=4), cmvn=result)
    frontend.fbank = lambda samples: mx.zeros((2, 4))
    assert frontend([]).tolist() == [[2.0, 6.0, 12.0, 20.0]] * 2


@pytest.mark.parametrize("count", [1, 2, 5, 6, 7, 13])
def test_lfr_matches_exact_funasr_reference(count):
    import ast
    import mlx.core as mx
    import torch

    filename = Path(__file__).resolve().parents[2] / "funasr/funasr/frontends/wav_frontend.py"
    tree = ast.parse(filename.read_text())
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_lfr"
    )
    namespace = {"torch": torch, "np": np}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)
    values = np.arange(count * 4, dtype=np.float32).reshape(count, 4)
    expected = namespace["apply_lfr"](torch.from_numpy(values), 7, 6).numpy()
    actual = apply_lfr(mx.array(values), 7, 6)
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_frontend_config_preserves_cmvn_file(tmp_path):
    path = tmp_path / "am.mvn"
    path.write_text(
        "<AddShift> 4 4\n<LearnRateCoef> 0 [ 1 2 3 4 ]\n"
        "<Rescale> 4 4\n<LearnRateCoef> 0 [ 2 3 4 5 ]\n"
    )
    config = FbankConfig.from_dict({"n_mels": 4, "cmvn_file": str(path)})
    assert config.cmvn_file == str(path)
    assert np.array_equal(WavFrontend(config).cmvn, [[1, 2, 3, 4], [2, 3, 4, 5]])
    assert WavFrontend(config, cmvn=None).cmvn is None


def test_empty_and_short_features():
    frontend = WavFrontend(FbankConfig(n_mels=8))
    assert frontend([]).shape == (0, 8)
    assert frontend(np.zeros(200, dtype=np.float32)).shape == (1, 8)
    assert frontend(np.zeros(1600, dtype=np.float32)).shape == (8, 8)


def test_oversized_frontend_refuses_before_conversion():
    frontend = WavFrontend(FbankConfig(max_seconds=0.01))
    with pytest.raises(ValueError, match="max_seconds"):
        frontend(np.zeros(161))


@pytest.mark.parametrize("n_mels", [80, 128])
def test_whisper_matches_reference_stft_and_slaney_filters(n_mels):
    import mlx.core as mx
    import torch
    import librosa

    samples = np.random.default_rng(41).normal(0, 0.1, 1600).astype(np.float32)
    actual = whisper_log_mel(samples, n_mels=n_mels, pad_to_samples=1600)
    waveform = torch.from_numpy(samples)
    spectrum = (
        torch.stft(waveform, 400, 160, window=torch.hann_window(400), return_complex=True).abs()[
            :, :-1
        ]
        ** 2
    )
    filters = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels))
    log = torch.clamp(filters @ spectrum, min=1e-10).log10()
    expected = (torch.maximum(log, log.max() - 8) + 4) / 4
    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=2e-5, atol=2e-5)


def test_whisper_silence():
    import mlx.core as mx

    features = whisper_log_mel(np.zeros(400), pad_to_samples=1600)
    assert features.shape == (80, 10)
    assert bool(mx.all(features == -1.5).item())


@pytest.mark.parametrize("window", ["hamming", "povey", "hanning"])
@pytest.mark.parametrize("snip_edges", [True, False])
def test_fbank_matches_exact_torchaudio_reference(window, snip_edges):
    import mlx.core as mx
    import torch
    import importlib.util

    # Load only the standalone reference file, not FunASR's package registries.
    spec = importlib.util.find_spec("torchaudio")
    assert spec is not None, "Reference torchaudio is required for numerical acceptance"
    filename = Path(spec.origin).parent / "compliance/kaldi.py"
    reference_spec = importlib.util.spec_from_file_location("kaldi_reference", filename)
    reference = importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference)
    samples = np.random.default_rng(123).normal(0, 0.1, 1600).astype(np.float32)
    config = FbankConfig(n_mels=8, window=window, snip_edges=snip_edges)
    actual = WavFrontend(config).fbank(samples)
    mx.eval(actual)
    expected = reference.fbank(
        torch.from_numpy(samples * 32768)[None],
        num_mel_bins=8,
        dither=0,
        energy_floor=0,
        window_type=window,
        sample_frequency=16000,
        snip_edges=snip_edges,
    )
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=2e-5, atol=2e-4)
