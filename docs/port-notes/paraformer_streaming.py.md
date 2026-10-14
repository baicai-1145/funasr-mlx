# paraformer_streaming.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
ParaformerStreaming: chunk-based streaming inference, MLX port.

Port of ``funasr/models/paraformer_streaming/model.py`` (the inference paths:
``init_cache`` / ``encode_chunk`` / ``calc_predictor_chunk`` /
``cal_decoder_with_predictor_chunk`` / ``generate_chunk``) together with the
streaming machinery they call into:

* ``scama/encoder.py::_add_overlap_chunk`` -- input stitching. At inference the
  reference concatenates the cached look-back window with the new chunk and
  retains a tail window; the elaborate ``x_add_mask`` matmuls in
  ``scama/chunk_utilis.py`` are the *training-time* batched equivalent and are
  not needed on this path.
* ``transformer/embedding.py::StreamSinusoidalPositionEncoder`` (``pe_online``)
  -- positions continue from ``cache['start_idx']``.
* ``sanm/attention.py::MultiHeadedAttentionSANM.forward_chunk`` -- KV cache
  that drops each chunk's right-context ``k``/``v`` from the *cache* while
  still attending with them for the current chunk.
* ``paraformer/cif_predictor.py::CifPredictorV2.forward_chunk`` -- carried
  ``cif_hidden``/``cif_alphas`` state and the sequential fire loop.
* ``paraformer/decoder.py::ParaformerSANMDecoder.forward_chunk`` -- per-layer
  fsmn and cross-attention opt caches.

Bit-identical building blocks (encoder layers, feed-forward, attention math,
offline CIF pieces) are REUSED from :mod:`funasr_mlx.models.sanm` and
:mod:`funasr_mlx.models.paraformer`; only what differs is implemented here.

Parameter names match the official ``speech_paraformer-large...online`` state
dict exactly (same module tree as offline Paraformer), so the offline
converter loads this model unchanged.

```
