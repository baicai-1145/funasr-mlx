"""Tiny numerical comparisons against actual PyTorch operators/source bodies."""

from pathlib import Path

import numpy as np
import pytest

from test_reference_parity import reference_definitions


@pytest.mark.parametrize("tap", [0, 1, 2])
def test_punctuation_memory_matches_actual_torch_convolution(tap):
    import mlx.core as mx
    import torch
    from funasr_mlx.models.ct_punc import SanmAttention

    inputs = np.arange(5, dtype=np.float32).reshape(1, 5, 1)
    kernel = np.zeros((1, 1, 3), np.float32)
    kernel[0, 0, tap] = 1.0
    convolution = torch.nn.Conv1d(1, 1, 3, bias=False)
    with torch.no_grad():
        convolution.weight.copy_(torch.from_numpy(kernel))
        padded = torch.nn.functional.pad(torch.from_numpy(inputs).transpose(1, 2), (1, 1))
        expected = convolution(padded).transpose(1, 2) + torch.from_numpy(inputs)
    native = SanmAttention(1, 1, 1, 3)
    native.fsmn_block.weight = mx.array(kernel.transpose(0, 2, 1))
    actual = native.forward_fsmn(mx.array(inputs), None)
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())
    if tap == 2:
        assert expected[0, 0, 0].item() == inputs[0, 1, 0]


@pytest.mark.parametrize("right,left_stride,right_stride", [(0, 1, 1), (2, 2, 2)])
def test_vad_memory_matches_exact_torch_source(right, left_stride, right_stride):
    import mlx.core as mx
    import torch
    from funasr_mlx.models.fsmn_vad import FSMNBlock

    reference = reference_definitions("funasr/models/fsmn_vad_streaming/encoder.py", ["FSMNBlock"])
    torch.manual_seed(91)
    golden = reference["FSMNBlock"](4, 4, 3, right, left_stride, right_stride).eval()
    native = FSMNBlock(4, 3, right, left_stride, right_stride)
    for name, weight in golden.state_dict().items():
        converted = weight.detach().numpy().squeeze(-1).transpose(0, 2, 1)
        native.load_weights([(name, mx.array(converted))], strict=False)
    inputs = np.random.default_rng(8).normal(0, 0.1, (2, 11, 4)).astype(np.float32)
    with torch.no_grad():
        expected, expected_cache = golden(torch.from_numpy(inputs))
    actual, actual_cache = native(mx.array(inputs))
    mx.eval(actual)
    assert expected_cache is actual_cache is None
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=1e-5, atol=1e-6)


def test_vad_streaming_cache_matches_exact_torch_source():
    import mlx.core as mx
    import torch
    from funasr_mlx.models.fsmn_vad import FSMNBlock

    reference = reference_definitions("funasr/models/fsmn_vad_streaming/encoder.py", ["FSMNBlock"])
    torch.manual_seed(31)
    golden = reference["FSMNBlock"](4, 4, 3, 0, 2, 1).eval()
    native = FSMNBlock(4, 3, 0, 2, 1)
    native.conv_left.weight = mx.array(
        golden.conv_left.weight.detach().numpy().squeeze(-1).transpose(0, 2, 1)
    )
    inputs = np.random.default_rng(12).normal(0, 0.1, (2, 12, 4)).astype(np.float32)
    torch_cache = torch.zeros(2, 4, 4, 1)
    native_cache = mx.zeros((2, 4, 4))
    for start, stop in [(0, 3), (3, 8), (8, 12)]:
        with torch.no_grad():
            expected, torch_cache = golden(torch.from_numpy(inputs[:, start:stop]), torch_cache)
        actual, native_cache = native(mx.array(inputs[:, start:stop]), native_cache)
        mx.eval(actual, native_cache)
        np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(native_cache), torch_cache.squeeze(-1).numpy())


@pytest.mark.parametrize("batch", [1, 2])
def test_whisper_full_forward_matches_canonical_openai_source(batch):
    import mlx.core as mx
    import torch
    from funasr_mlx.models.whisper import ModelDimensions, Whisper, load_strict

    source = Path(__file__).resolve().parents[1] / ".tmp/openai-whisper-model.py"
    reference = reference_definitions(
        str(source),
        [
            "ModelDimensions",
            "LayerNorm",
            "Linear",
            "Conv1d",
            "sinusoids",
            "MultiHeadAttention",
            "ResidualAttentionBlock",
            "AudioEncoder",
            "TextDecoder",
            "Whisper",
        ],
    )
    reference["MultiHeadAttention"].use_sdpa = False
    dimensions = dict(
        n_mels=4,
        n_audio_ctx=20,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=2,
        n_vocab=16,
        n_text_ctx=12,
        n_text_state=8,
        n_text_head=2,
        n_text_layer=2,
    )
    torch.manual_seed(43)
    golden = reference["Whisper"](reference["ModelDimensions"](**dimensions)).eval()
    with torch.no_grad():
        golden.decoder.positional_embedding.copy_(
            torch.randn_like(golden.decoder.positional_embedding) * 0.02
        )
    native = Whisper(ModelDimensions(**dimensions))
    native.eval()
    state = {name: value.detach().numpy() for name, value in golden.state_dict().items()}
    load_strict(native, state, convert=True)
    mel = np.random.default_rng(6).normal(0, 0.1, (batch, 4, 40)).astype(np.float32)
    tokens = np.tile(np.array([1, 2, 3, 4], np.int64), (batch, 1))
    with torch.no_grad():
        expected_audio = golden.embed_audio(torch.from_numpy(mel))
        expected = golden.logits(torch.from_numpy(tokens), expected_audio)
    # The native encoder exposes channels-last mel frames.
    actual_audio = native.embed_audio(mx.array(mel).swapaxes(1, 2))
    actual = native.logits(mx.array(tokens.astype(np.int32)), actual_audio)
    mx.eval(actual_audio, actual)
    np.testing.assert_allclose(
        np.asarray(actual_audio), expected_audio.numpy(), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=1e-4, atol=1e-5)
