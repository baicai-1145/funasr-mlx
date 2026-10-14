"""ParaformerStreaming: chunk-based streaming inference, MLX port.

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/paraformer_streaming.py.md
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as mxnn
from mlx.nn import log_softmax

from .paraformer import (
    CifPredictorV2,
    ParaformerConfig,
    SANMEncoder,
)
from .sanm import constant_pad_1d

__all__ = [
    "StreamSinusoidalPositionEncoder",
    "ParaformerStreaming",
    "ParaformerStreamingConfig",
    "streaming_config_from_dict",
]


class StreamSinusoidalPositionEncoder(mxnn.Module):
    """``transformer/embedding.py::StreamSinusoidalPositionEncoder``.

    Stateful through ``cache['start_idx']``: each chunk's positions continue
    from where the previous chunk ended, which is what makes the streamed
    encoding equal the offline one.
    """

    def __call__(self, x: mx.array, cache: dict | None) -> mx.array:
        _, timesteps, input_dim = x.shape
        start_idx = 0
        if cache is not None:
            start_idx = int(cache.get("start_idx", 0))
            cache["start_idx"] = start_idx + timesteps
        depth = input_dim
        log_timescale_increment = mx.log(mx.array(10000.0, dtype=mx.float32)) / (depth / 2 - 1)
        inv_timescales = mx.exp(
            mx.arange(depth // 2, dtype=mx.float32) * (-log_timescale_increment)
        )
        positions = mx.arange(1, timesteps + start_idx + 1, dtype=mx.float32)
        scaled_time = positions[None, :, None] * inv_timescales[None, None, :]
        encoding = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=2)
        return x + encoding[:, start_idx : start_idx + timesteps].astype(x.dtype)


def _attn_forward_chunk(attn: Any, x: mx.array, cache: dict | None, chunk_size, look_back: int):
    """``MultiHeadedAttentionSANM.forward_chunk`` on the shared offline class.

    Mirrors ``sanm/attention.py:329-366``: compute qkv for the (stitched)
    chunk, merge cached ``k``/``v`` (look-back), commit each chunk's
    right-context-free stride into the cache, attend, add the FSMN memory.
    """
    q_h, k_h, v_h, v = attn.forward_qkv(x)
    if chunk_size is not None and (look_back > 0 or look_back == -1):
        if cache is not None:
            k_stride = k_h[:, :, : -chunk_size[2], :]
            v_stride = v_h[:, :, : -chunk_size[2], :]
            k_h = mx.concatenate([cache["k"], k_h], axis=2)
            v_h = mx.concatenate([cache["v"], v_h], axis=2)
            cache["k"] = mx.concatenate([cache["k"], k_stride], axis=2)
            cache["v"] = mx.concatenate([cache["v"], v_stride], axis=2)
            if look_back != -1:
                cache["k"] = cache["k"][:, :, -(look_back * chunk_size[1]) :, :]
                cache["v"] = cache["v"][:, :, -(look_back * chunk_size[1]) :, :]
        else:
            cache = {
                "k": k_h[:, :, : -chunk_size[2], :],
                "v": v_h[:, :, : -chunk_size[2], :],
            }
    fsmn_memory = attn.forward_fsmn(v, None)
    q_h = q_h * (attn._d_k**-0.5)
    scores = mx.matmul(q_h, mx.swapaxes(k_h, -1, -2))
    att = attn.forward_attention(v_h, scores, None)
    return att + fsmn_memory, cache


def _encoder_layer_forward_chunk(layer, x, cache, chunk_size, look_back):
    residual = x
    if layer._normalize_before:
        x = layer.norm1(x)
    x, cache = _attn_forward_chunk(layer.self_attn, x, cache, chunk_size, look_back)
    x = x + residual if layer._in_size == layer._size else x
    residual = x
    if layer._normalize_before:
        x = layer.norm2(x)
    x = layer.feed_forward(x)
    x = x + residual
    return x, cache


def _predictor_forward_chunk(pred: CifPredictorV2, hidden: mx.array, cache: dict, is_final: bool):
    """``CifPredictorV2.forward_chunk`` (``cif_predictor.py:316-412``).

    Returns ``(acoustic_embeds, token_length, alphas, fires)`` with the carry
    state written back into ``cache``.
    """
    batch_size, _, hidden_size = hidden.shape
    # MLX conv1d is NLC: hidden (B, T, D) feeds directly; the reference's two
    # transposes around the pad/conv collapse into padding on axis=1.
    queries = constant_pad_1d(hidden, pred._l_order, pred._r_order, axis=1)
    output = mx.maximum(pred.cif_conv1d(queries), 0.0)
    output = pred.cif_output(output)
    alphas = mx.sigmoid(output)
    alphas = mx.maximum(alphas * pred._smooth_factor - pred._noise_threshold, 0.0)
    alphas = alphas.squeeze(-1)

    if "chunk_size" in cache:
        alphas = mx.concatenate(
            [
                alphas[:, :0],  # chunk_size[0] left look-back is always 0 online
                alphas,
            ],
            axis=1,
        )
    if not is_final:
        right = cache["chunk_size"][2] if "chunk_size" in cache else 0
        if right > 0 and alphas.shape[1] > right:
            pad = mx.zeros((batch_size, right), alphas.dtype)
            alphas = mx.concatenate([alphas[:, :-right], pad], axis=1)
    if "cif_alphas" in cache and "cif_hidden" in cache:
        hidden = mx.concatenate([cache["cif_hidden"], hidden], axis=1)
        alphas = mx.concatenate([cache["cif_alphas"], alphas], axis=1)
    if is_final:
        tail_hidden = mx.zeros((batch_size, 1, hidden_size), hidden.dtype)
        tail = mx.array([[pred._tail_threshold]], alphas.dtype)
        if batch_size > 1:
            tail = mx.broadcast_to(tail, (batch_size, 1))
        hidden = mx.concatenate([hidden, tail_hidden], axis=1)
        alphas = mx.concatenate([alphas, tail], axis=1)

    len_time = alphas.shape[1]
    token_length = []
    cache_alphas = []
    cache_hiddens = []
    list_frames_all = []
    for b in range(batch_size):
        integrate = 0.0
        frames = mx.zeros((hidden_size,), hidden.dtype)
        list_frame = []
        for t in range(len_time):
            alpha = alphas[b, t]
            if float(alpha) + integrate < float(pred._threshold):
                integrate = integrate + float(alpha)
                frames = frames + alpha * hidden[b, t]
            else:
                frames = frames + (pred._threshold - integrate) * hidden[b, t]
                list_frame.append(frames)
                integrate = integrate + float(alpha)
                integrate = integrate - float(pred._threshold)
                frames = integrate * hidden[b, t]
        cache_alphas.append(integrate)
        if integrate > 0.0:
            cache_hiddens.append(frames / integrate)
        else:
            cache_hiddens.append(frames)
        token_length.append(len(list_frame))
        list_frames_all.append(list_frame)

    cache["cif_alphas"] = mx.reshape(mx.array(cache_alphas, alphas.dtype), (1, -1))
    cache["cif_hidden"] = mx.reshape(mx.stack(cache_hiddens, 0), (1, batch_size, hidden_size))

    max_token_len = max(token_length)
    if max_token_len == 0:
        return hidden, mx.array(token_length, alphas.dtype), None, None
    list_ls = []
    for b in range(batch_size):
        if token_length[b] == 0:
            list_ls.append(mx.zeros((max_token_len, hidden_size), hidden.dtype))
        else:
            stack = mx.stack(list_frames_all[b], 0)
            pad = mx.zeros((max_token_len - token_length[b], hidden_size), hidden.dtype)
            list_ls.append(mx.concatenate([stack, pad], 0))
    embeds = mx.stack(list_ls, 0)
    return embeds, mx.array(token_length, alphas.dtype), None, None


def _cross_attn_forward_chunk(
    attn: Any, x: mx.array, memory: mx.array, cache: dict | None, chunk_size, look_back: int
):
    """``MultiHeadedAttentionCrossAtt.forward_chunk`` (attention.py:815+ for cross).

    Cross-attention to the (growing) encoder memory with the same tail-window
    cache discipline as the decoder self-attention.
    """
    q_h, k_h, v_h = attn.forward_qkv(x, memory)
    if chunk_size is not None and look_back > 0:
        if cache is not None:
            k_h = mx.concatenate([cache["k"], k_h], axis=2)
            v_h = mx.concatenate([cache["v"], v_h], axis=2)
            cache["k"] = k_h[:, :, -(look_back * chunk_size[1]) :, :]
            cache["v"] = v_h[:, :, -(look_back * chunk_size[1]) :, :]
        else:
            cache = {
                "k": k_h[:, :, -(look_back * chunk_size[1]) :, :],
                "v": v_h[:, :, -(look_back * chunk_size[1]) :, :],
            }
    q_h = q_h * (attn._d_k**-0.5)
    scores = mx.matmul(q_h, mx.swapaxes(k_h, -1, -2))
    out = attn.forward_attention(v_h, scores, None)
    return out, cache


def _dec_layer_forward_chunk(
    layer, tgt: mx.array, memory: mx.array, fsmn_cache, opt_cache, chunk_size, look_back: int
):
    residual = tgt
    if layer._normalize_before:
        tgt = layer.norm1(tgt)
    tgt = layer.feed_forward(tgt)

    x = tgt
    if layer.self_attn is not None:
        if layer._normalize_before:
            tgt = layer.norm2(tgt)
        x, fsmn_cache = layer.self_attn(tgt, None, cache=fsmn_cache)
        x = residual + x

    if layer.src_attn is not None:
        residual = x
        if layer._normalize_before:
            x = layer.norm3(x)
        x, opt_cache = _cross_attn_forward_chunk(
            layer.src_attn, x, memory, opt_cache, chunk_size, look_back
        )
        x = residual + x

    return x, memory, fsmn_cache, opt_cache


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@dataclass
class ParaformerStreamingConfig(ParaformerConfig):

    chunk_size: Sequence[int] = (0, 10, 5)
    encoder_chunk_look_back: int = 0
    decoder_chunk_look_back: int = 0


class ParaformerStreaming(mxnn.Module):
    """``ParaformerStreaming`` -- chunk-wise streaming Paraformer (MLX).

    Reuses the offline module tree (identical parameter names); adds
    ``init_cache`` / ``forward_chunk`` implementing the reference streaming
    protocol. ``generate`` feeds fixed 600 ms chunks ([0,10,5] x 960 samples).
    """

    def __init__(self, config: ParaformerStreamingConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        cfg = config or streaming_config_from_dict(None, **kwargs)
        self.config = cfg
        self.encoder = SANMEncoder(cfg.input_size, cfg.encoder_conf)
        pc = vars(cfg.predictor_conf)
        self.predictor = CifPredictorV2(**pc)
        from .paraformer import ParaformerSANMDecoder

        self.decoder = ParaformerSANMDecoder(
            cfg.vocab_size, cfg.encoder_conf.output_size, cfg.decoder_conf
        )
        self._chunk_size = [int(v) for v in cfg.chunk_size]
        self._encoder_look_back = int(cfg.encoder_chunk_look_back)
        self._decoder_look_back = int(cfg.decoder_chunk_look_back)

    # -- streaming protocol ------------------------------------------------ #
    def init_cache(self, enc_output_size: int = 512, feats_dims: int = 560) -> dict:
        """``ParaformerStreaming.init_cache`` (model.py:511-552), MLX arrays."""
        chunk_size = self._chunk_size
        return {
            "encoder": {
                "start_idx": 0,
                "cif_hidden": mx.zeros((1, 1, enc_output_size)),
                "cif_alphas": mx.zeros((1, 1)),
                "chunk_size": chunk_size,
                "encoder_chunk_look_back": self._encoder_look_back,
                "last_chunk": False,
                "opt": None,
                "feats": mx.zeros((1, chunk_size[0] + chunk_size[2], feats_dims)),
                "tail_chunk": False,
            },
            "decoder": {
                "decode_fsmn": None,
                "decoder_chunk_look_back": self._decoder_look_back,
                "opt": None,
                "chunk_size": chunk_size,
            },
            "frontend": {},
            "prev_samples": mx.zeros((0,)),
        }

    def encode_chunk(
        self, speech: mx.array, speech_lengths: mx.array, cache: dict, is_final: bool = False
    ) -> tuple[mx.array, mx.array]:
        xs_pad = speech * (self.encoder.output_size() ** 0.5)
        xs_pad = self.encoder.embed(xs_pad, cache["encoder"])
        if cache["encoder"]["tail_chunk"]:
            xs_pad = cache["encoder"]["feats"]
        else:
            # ``_add_overlap_chunk`` (scama/encoder.py:480-494): stitch the
            # retained look-back window with the new chunk, keep a tail window.
            enc_cache = cache["encoder"]
            overlap = mx.concatenate([enc_cache["feats"], xs_pad], axis=1)
            keep = enc_cache["chunk_size"][0] + enc_cache["chunk_size"][2]
            enc_cache["feats"] = overlap[:, -keep:, :]
            xs_pad = overlap

        if cache["encoder"]["opt"] is None:
            cache_layer_num = len(self.encoder.encoders0) + len(self.encoder.encoders)
            new_cache: list = [None] * cache_layer_num
        else:
            new_cache = cache["encoder"]["opt"]

        idx = 0
        for layer in self.encoder.encoders0:
            xs_pad, new_cache[idx] = _encoder_layer_forward_chunk(
                layer,
                xs_pad,
                new_cache[idx],
                cache["encoder"]["chunk_size"],
                cache["encoder"]["encoder_chunk_look_back"],
            )
            idx += 1
        for layer in self.encoder.encoders:
            xs_pad, new_cache[idx] = _encoder_layer_forward_chunk(
                layer,
                xs_pad,
                new_cache[idx],
                cache["encoder"]["chunk_size"],
                cache["encoder"]["encoder_chunk_look_back"],
            )
            idx += 1

        if self.encoder._normalize_before:
            xs_pad = self.encoder.after_norm(xs_pad)
        if (
            cache["encoder"]["encoder_chunk_look_back"] > 0
            or cache["encoder"]["encoder_chunk_look_back"] == -1
        ):
            cache["encoder"]["opt"] = new_cache
        return xs_pad, speech_lengths

    def predictor_chunk(
        self, encoder_out: mx.array, cache: dict, is_final: bool = False
    ) -> tuple[mx.array, mx.array]:
        """``calc_predictor_chunk`` -> ``CifPredictorV2.forward_chunk``."""
        return _predictor_forward_chunk(self.predictor, encoder_out, cache["encoder"], is_final)

    def decoder_chunk(
        self, encoder_out: mx.array, sematic_embeds: mx.array, cache: dict
    ) -> mx.array:
        """``cal_decoder_with_predictor_chunk`` -> decoder ``forward_chunk``."""
        dec_cache = cache["decoder"]
        x = sematic_embeds
        memory = encoder_out
        if dec_cache["decode_fsmn"] is None:
            fsmn_cache: list = [None] * len(self.decoder.decoders)
        else:
            fsmn_cache = dec_cache["decode_fsmn"]
        if dec_cache["opt"] is None:
            opt_cache: list = [None] * len(self.decoder.decoders)
        else:
            opt_cache = dec_cache["opt"]

        for i, layer in enumerate(self.decoder.decoders):
            x, memory, fsmn_cache[i], opt_cache[i] = _dec_layer_forward_chunk(
                layer,
                x,
                memory,
                fsmn_cache[i],
                opt_cache[i],
                dec_cache["chunk_size"],
                dec_cache["decoder_chunk_look_back"],
            )
        for layer in self.decoder.decoders2 or []:
            x, memory, _, _ = _dec_layer_forward_chunk(
                layer, x, memory, None, None, dec_cache["chunk_size"], 0
            )
        for layer in self.decoder.decoders3 or []:
            x, memory, _, _ = _dec_layer_forward_chunk(
                layer, x, memory, None, None, dec_cache["chunk_size"], 0
            )
        if self.decoder._normalize_before:
            x = self.decoder.after_norm(x)
        if self.decoder.output_layer is not None:
            x = self.decoder.output_layer(x)
        dec_cache["decode_fsmn"] = fsmn_cache
        if dec_cache["decoder_chunk_look_back"] > 0 or dec_cache["decoder_chunk_look_back"] == -1:
            dec_cache["opt"] = opt_cache
        return log_softmax(x, axis=-1)

    def forward_chunk(
        self,
        speech: mx.array,
        speech_lengths: mx.array,
        cache: dict,
        is_final: bool = False,
    ) -> tuple[mx.array | None, mx.array | None]:
        """One chunk end-to-end. Returns ``(am_scores, token_length)``.

        ``None`` when the chunk fired no tokens (mirrors ``generate_chunk``
        returning ``[]`` when ``max(pre_token_length) < 1``).
        """
        encoder_out, encoder_out_lens = self.encode_chunk(speech, speech_lengths, cache, is_final)
        embeds, token_length, _, _ = self.predictor_chunk(encoder_out, cache, is_final)
        if int(mx.max(token_length).item()) < 1:
            return None, token_length
        scores = self.decoder_chunk(encoder_out, embeds, cache)
        return scores, token_length


def streaming_config_from_dict(
    data: Mapping[str, Any] | None = None, **kwargs: Any
) -> ParaformerStreamingConfig:
    """Build the streaming config from the official online yaml mapping."""
    from .paraformer import paraformer_config_from_dict

    data = dict(data or {})
    data.update(kwargs)
    chunk = data.pop("chunk_size", None)
    enc_lb = data.pop("encoder_chunk_look_back", None)
    dec_lb = data.pop("decoder_chunk_look_back", None)
    base = paraformer_config_from_dict(data)
    fields = {f.name for f in ParaformerStreamingConfig.__dataclass_fields__.values()}
    return ParaformerStreamingConfig(
        **{
            **{k: v for k, v in vars(base).items() if k in fields},
            **({"chunk_size": chunk} if chunk else {}),
            **({"encoder_chunk_look_back": enc_lb} if enc_lb is not None else {}),
            **({"decoder_chunk_look_back": dec_lb} if dec_lb is not None else {}),
        }
    )
