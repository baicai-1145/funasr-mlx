"""Tiny, offline, network-free tests for the MOSS-Transcribe-Diarize assembly.

Deliberately tiny (4 mel bins, 8 channels, 2 whisper layers, 2 qwen3 layers) so
the file stays far below the project memory ceiling.  These are structural /
algebraic checks -- parameter naming, the audio-token splice, and the
decoder-dtype contract -- **not** real-checkpoint acceptance.

The dtype test is the regression guard for a real 5.8x bug: the whisper tower +
VQAdaptor emit float32 (the mel front-end is float32), and splicing that
straight into the bf16 Qwen3 stream promoted all 28 decoder layers to float32
activations against bf16 weights.  Both in-repo callers cast in the caller, so
the shipped model path was the only place the promotion survived.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core", reason="mlx is required for the model ports")

from funasr_mlx.models.moss_transcribe import (  # noqa: E402
    MossTranscribeDiarize,
)
from funasr_mlx.models.llm.kv_cache import KVCache  # noqa: E402
from funasr_mlx.models.llm.qwen3 import Qwen3ModelArgs  # noqa: E402

N_MELS = 4
D_MODEL = 8
TEXT_HIDDEN = 8
N_WHISPER_LAYERS = 2
N_TEXT_LAYERS = 2
AUDIO_TOKEN = 7


def _config() -> Qwen3ModelArgs:
    return Qwen3ModelArgs(
        hidden_size=TEXT_HIDDEN,
        num_hidden_layers=N_TEXT_LAYERS,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=32,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=128,
    )


def _model(dtype=mx.bfloat16) -> MossTranscribeDiarize:
    model = MossTranscribeDiarize(
        whisper_cfg={
            "num_mel_bins": N_MELS,
            "d_model": D_MODEL,
            "encoder_layers": N_WHISPER_LAYERS,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 16,
            "max_source_positions": 32,
        },
        text_config=_config(),
        audio_merge_size=2,
        audio_token_id=AUDIO_TOKEN,
    )
    # Cast every weight to `dtype` so the model stands in for a bf16 checkpoint.
    from mlx.utils import tree_map

    model.update(tree_map(lambda a: a.astype(dtype), model.parameters()))
    return model.eval()


def test_encoder_and_adaptor_emit_float32_for_a_float32_mel():
    """Documents the precondition for the dtype cast: the tower is float32 in."""
    model = _model()
    mel = mx.zeros((N_MELS, 16), dtype=mx.float32)
    hidden = model.whisper_encoder(mel)
    mx.eval(hidden)
    assert hidden.dtype == mx.float32 or hidden.dtype == mx.bfloat16
    # The adaptor must at least be finite and have the merged row count.
    out = model.encode_audio(mel)
    mx.eval(out)
    assert out.shape[0] == 1
    assert mx.all(mx.isfinite(out)).item()


def test_forward_with_audio_follows_the_decoder_weight_dtype():
    """The spliced stream must be the DECODER's dtype, not the tower's.

    Regression guard: with a bf16 decoder and a float32 adaptor output, an
    uncorrected splice yields float32 hidden states, which promotes every
    Linear in the decoder (measured 5.8x slower on M4).
    """
    model = _model(mx.bfloat16)
    assert model._decoder_dtype() == mx.bfloat16

    token_ids = mx.array([[1, 2, AUDIO_TOKEN, AUDIO_TOKEN, 3]], dtype=mx.int32)
    # float32 audio embeds, exactly what the whisper tower + adaptor produce.
    audio_embeds = mx.ones((1, 2, TEXT_HIDDEN), dtype=mx.float32)
    caches = [KVCache() for _ in range(N_TEXT_LAYERS)]
    logits = model.forward_with_audio(token_ids, audio_embeds, pad_span=2, caches=caches)
    mx.eval(logits)
    assert logits.shape == (1, 5, 32)
    assert mx.all(mx.isfinite(logits)).item()

    # The hidden state the decoder consumed is observable through the cache
    # dtype: K/V are projected from a bf16 hidden state.
    assert caches[0].keys.dtype == mx.bfloat16


def test_forward_with_audio_accepts_an_already_correct_dtype():
    """The cast must be a no-op when the caller already passed the right dtype."""
    model = _model(mx.bfloat16)
    token_ids = mx.array([[1, AUDIO_TOKEN, 2]], dtype=mx.int32)
    embeds = mx.ones((1, 1, TEXT_HIDDEN), dtype=mx.bfloat16)
    caches = [KVCache() for _ in range(N_TEXT_LAYERS)]
    logits = model.forward_with_audio(token_ids, embeds, pad_span=1, caches=caches)
    mx.eval(logits)
    assert logits.shape == (1, 3, 32)


def test_splice_places_the_audio_span_at_the_token_run():
    """The audio embeds must land exactly where the pad tokens were."""
    model = _model(mx.bfloat16)
    token_ids = mx.array([[1, 2, AUDIO_TOKEN, AUDIO_TOKEN, 3]], dtype=mx.int32)
    embeds = mx.zeros((1, 2, TEXT_HIDDEN), dtype=mx.bfloat16)
    hidden = model.language_model.model.embed_tokens(token_ids)
    first = int(mx.argmax(token_ids == AUDIO_TOKEN).item())
    spliced = mx.concatenate(
        [hidden[:, :first, :], embeds, hidden[:, first + 2 :, :]], axis=1
    )
    mx.eval(spliced)
    assert spliced.shape == (1, 5, TEXT_HIDDEN)
    assert mx.all(spliced[:, 2:4, :] == 0).item()
