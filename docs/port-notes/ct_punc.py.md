# ct_punc.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
Native MLX inference for FunASR's CT-Transformer punctuation model (``ct-punc``).

FunASR reference sources (read-only, ported verbatim):

* ``funasr/models/ct_transformer/model.py`` -- ``CTTransformer``, ``punc_forward``
  and the whole ``inference`` mini-sentence chunking / caching post-processing loop.
* ``funasr/models/ct_transformer/utils.py`` -- ``split_words``, ``split_to_mini_sentence``.
* ``funasr/models/sanm/encoder.py`` -- ``SANMEncoder``, ``EncoderLayerSANM``.
* ``funasr/models/sanm/attention.py`` -- ``MultiHeadedAttentionSANM``.
* ``funasr/models/transformer/embedding.py`` -- ``SinusoidalPositionEncoder``.
* ``funasr/models/transformer/positionwise_feed_forward.py`` -- ``PositionwiseFeedForward``.
* ``funasr/models/transformer/layer_norm.py`` -- ``LayerNorm`` (``eps=1e-12``).
* ``funasr/models/transformer/utils/nets_utils.py`` -- ``make_pad_mask``.
* ``funasr/tokenizer/char_tokenizer.py`` + ``funasr/tokenizer/abs_tokenizer.py``.

The module is deliberately self-contained: it does **not** import any other
``funasr_mlx`` module, so it can land before/independently of the shared ASR
runtime.  ``mlx`` is imported at module scope only (the package root stays
MLX-free); importing this module is the single explicit opt-in.

Semantics that matter and are reproduced exactly:

* ``LayerNorm`` uses ``eps=1e-12`` (FunASR's ``funasr...layer_norm.LayerNorm``),
  **not** the ``1e-5`` MLX default.
* ``SinusoidalPositionEncoder`` uses ``inv_timescales = 10000 ** (-i / (D/2 - 1))``
  with the sin block concatenated *before* the cos block (not interleaved, and not
  the textbook ``10000 ** (-2i/D)``); positions start at 1.
* ``SANMEncoder.forward`` scales the embedding by ``output_size ** 0.5`` **before**
  the (parameter-free) positional encoding.
* ``MultiHeadedAttentionSANM`` scales queries by ``d_k ** -0.5`` only (no extra
  softmax scaling), masks padded *keys* with ``-inf`` and re-zeroes the attention
  probabilities afterwards.
* The padding mask is only ever applied through the ``(B, 1, T)`` validity mask
  derived from ``~make_pad_mask``; padded positions are multiplied out of the
  FSMN memory branch both before *and* after the depthwise convolution.

Checkpoint layout conversion (PyTorch -> MLX), applied by
:func:`convert_checkpoint_tensors`:

===============================  ==========================  ==========================
FunASR / PyTorch key             PyTorch shape               MLX shape
===============================  ==========================  ==========================
``embed.weight``                 ``(vocab, embed_unit)``     identical
``decoder.weight``               ``(punc, att_unit)``        identical
``*.linear*.weight``             ``(out, in)``               identical
``*.norm*.weight``               ``(dim,)``                  identical
``*.self_attn.fsmn_block.weight`` ``(C, 1, K)``              ``(C, K, 1)`` (transpose 0,2,1)
===============================  ==========================  ==========================

Anything that does not match is a hard error: loading is strict.

Known gaps (documented, not hidden):

* Only ``encoder_conf.input_layer="pe"`` is supported.  ``conv2d*`` subsampling
  and ``embed``/``linear`` input layers change the time axis or add parameters
  under different names and are rejected up front.
* The ``CTTransformerStreaming`` VAD-aware variant (``SANMVadEncoder``,
  ``mask_shfit_chunk``, ``vad_indexes``) is **not** implemented.
* The optional ``jieba`` word-level ``punc_array`` expansion is available as
  :func:`expand_word_level_punc` but segmentation itself is not ported (``jieba``
  is an optional dependency).
* ``AutoModel``-level post-processing (sentence timestamps, speaker diarization
  segmentation, surface-text re-alignment) lives in the shared runtime and is out
  of this module's scope.

```
