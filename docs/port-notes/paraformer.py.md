# paraformer.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
MLX inference port of FunASR's ``Paraformer``.

Reference (read-only, ``/Users/baicai1145/repos/funasr``):

* ``funasr/models/sanm/encoder.py`` -> :class:`SANMEncoder`
* ``funasr/models/paraformer/cif_predictor.py`` -> :class:`CifPredictorV2`,
  :func:`cif_v1`, :func:`cif_wo_hidden_v1`
* ``funasr/models/paraformer/decoder.py`` -> :class:`ParaformerSANMDecoder`
* ``funasr/models/paraformer/model.py`` -> :class:`Paraformer`
* ``funasr/models/ctc/ctc.py`` -> :class:`CTC`
* ``funasr/utils/timestamp_tools.py`` ->
  :func:`ts_prediction_lfr6_standard` (timestamp projection)

What is ported
--------------
``encoder -> CIF predictor -> SANM decoder -> log_softmax``, i.e. the complete
non-autoregressive inference path, plus greedy token decoding and the
CIF-peak-based character timestamps.

What is *not* ported (deliberately)
----------------------------------
Label-smoothing / CTC / MAE losses, the training-time ``sampler``
(``sampling_ratio``), beam search with CTC or n-gram scorers, SpecAug and
feature extraction.  LoRA variants of the SANM blocks are rejected explicitly
instead of being silently ignored.

Parity notes
------------
* The decoder consumes the **acoustic embeddings produced by the CIF predictor**
  as ``ys_in_pad``; ``self.embed`` is *not* applied during inference (it is only
  used by the training sampler), which is why the checkpoint still carries
  ``decoder.embed.0.weight``.
* ``CifPredictorV2.tail_process_fn`` reproduces the reference's shift-based mask
  trick exactly, including the behaviour that a full-length (unpadded) utterance
  gets a zero tail.
* ``decoder.decoders2.w_2`` has **no bias** (``PositionwiseFeedForwardDecoderSANM``).

```
