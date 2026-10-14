# sanm.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
Shared SANM (self-attention with memory) blocks for the MLX ASR ports.

This module is the single source of truth for every block that FunASR's
``paraformer`` and ``sense_voice`` models share.  It is a direct port of the
following reference files (read-only, ``/Users/baicai1145/repos/funasr``):

* ``funasr/models/sanm/attention.py``
  (``MultiHeadedAttentionSANM``, ``MultiHeadedAttentionSANMDecoder``,
  ``MultiHeadedAttentionCrossAtt``)
* ``funasr/models/sanm/encoder.py`` (``EncoderLayerSANM``)
* ``funasr/models/sanm/positionwise_feed_forward.py``
  (``PositionwiseFeedForwardDecoderSANM``)
* ``funasr/models/transformer/positionwise_feed_forward.py``
  (``PositionwiseFeedForward``)
* ``funasr/models/transformer/layer_norm.py`` (``LayerNorm``, ``eps=1e-12``)
* ``funasr/models/transformer/embedding.py``
  (``PositionalEncoding``, ``SinusoidalPositionEncoder``)
* ``funasr/models/transformer/utils/subsampling.py``
  (``Conv2dSubsampling``, ``Conv2dSubsampling2``)
* ``funasr/models/transformer/utils/nets_utils.py`` (``make_pad_mask``)
* ``funasr/models/scama/utils.py`` (``sequence_mask``)
* ``funasr/models/transformer/utils/repeat.py`` (``repeat``/``MultiSequential``)

Design rules enforced here
--------------------------
1. **Checkpoint-compatible parameter names.**  FunASR keeps its repeated
   encoder/decoder layers in ``nn.Sequential`` (via ``repeat``) or in
   ``nn.ModuleList``.  Both produce ``"<parent>.<index>.<leaf>"`` keys in a
   PyTorch ``state_dict``.  MLX's ``nn.Sequential``/``nn.ModuleList`` would add
   an extra ``layers`` level, so :class:`ModuleStack` registers children under
   their bare numeric index instead.
2. **Explicit tensor-layout conversion rules.**  PyTorch stores conv weights as
   ``(out, in/groups, k)`` / ``(out, in, kh, kw)``; MLX uses
   ``(out, k, in/groups)`` / ``(out, kh, kw, in)``.  See
   :data:`LAYOUT_RULES` and :func:`convert_checkpoint_tensor`.
   MLX convolutions are also **channels-last** (``mx.conv1d`` consumes ``NLC``,
   ``mx.conv2d`` consumes ``NHWC``) whereas PyTorch uses ``NCL``/``NCHW``; every
   conv call site here therefore feeds a channels-last tensor and pads the time
   axis explicitly with ``axis=1``.
3. **Strict loading.**  :func:`load_strict` refuses missing/unexpected keys and
   never silently skips a mismatch.
4. **Inference only.**  Every dropout in the reference is an identity in eval
   mode and is therefore elided; stochastic depth (train-time only) is dropped.
   No training losses are ported.

Known deviations and deliberately not-ported surface
-----------------------------------------------------
These are intentional and reviewed; none of them is a silent partial
implementation.

* **Streaming / chunked paths are not ported.**  ``forward_chunk``,
  ``mask_shfit_chunk``, ``mask_chunk_predictor``, ``forward_one_step``,
  ``forward_asf2``, ``SANMVadEncoder`` / ``SANMEncoder.forward(vad_indexes=...)``,
  ``mask_att_chunk_encoder``, and the ``fa-zh`` timestamp prediction are
  training/streaming-only in the reference and are absent here.
* **``MultiHeadedAttentionSANMDecoder`` cache layout deviates (NLC, not NCL).**
  The reference keeps the streaming cache in ``(b, d, L)`` (``cache = x`` taken
  before the transpose); this port keeps ``(b, L, d)``.  Internally consistent
  and behaviourally identical for offline inference, but it is a *different
  runtime contract* from the reference's export/streaming path, so a cache
  written by one cannot be read by the other.  Two further reference branches
  are streaming-only and therefore absent: the
  ``if x.size(1) != inputs.size(1): inputs = inputs[:, -1, :]`` step-position
  guard, and populating the cache on the first (``cache is None``) call.
* **``decimate_mask`` emulates ``x_mask[:, :, :-2:2][:, :, :-2:2]``** with an
  explicit index list plus ``mx.take(..., axis=-1)``.  Behaviourally identical to
  the slice, just not as cheap.
* **CIF accumulation is float32.**  The reference uses float64 in
  ``cif_v1``/``cif_wo_hidden_v1``; the peak-index thresholding in ``cif_v1`` is
  the sensitive part and can differ by a hair near an exact tie.
* **``cif_v1`` clamps ``max_label_len = min(max_label_len, t)``**; the reference
  has no clamp (``cif_predictor.py:880``).  It is unreachable in practice since
  ``alphas <= 1`` per frame plus at most one 0.45 tail gives ``floor(sum) <= t``.
* **``PositionalEncoding(reverse=True)``** flips the position index where the
  reference reverses an already-built table of length ``max_len``.  Identical for
  ``reverse=False``, the only configured path.

```
