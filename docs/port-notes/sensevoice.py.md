# sensevoice.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
MLX inference port of FunASR's ``SenseVoiceSmall``.

Reference (read-only, ``/Users/baicai1145/repos/funasr``):
``funasr/models/sense_voice/model.py`` and
``funasr/models/ctc/ctc.py``.

What is ported
--------------
* :class:`SenseVoiceEncoderSmall` -- the SANM encoder (one ``encoders0`` layer
  projecting from the 560-d input, ``num_blocks - 1`` ``encoders`` layers, an
  ``after_norm``, then ``tp_blocks`` ``tp_encoders`` layers and a ``tp_norm``).
* The **four metadata prefix embeddings** prepended to the acoustic features at
  inference time, in this exact order:
  ``[language, event, emotion, textnorm]`` followed by the speech frames.
* :class:`SenseVoiceSmall` -- encoder + CTC projection head (``ctc.ctc_lo``).
* Greedy CTC decoding: ``argmax`` -> ``unique_consecutive`` -> drop ``blank_id``
  (``0``), with the optional ``ban_emo_unk`` suppression of the ``<unk>`` emotion
  token.

What is *not* ported (deliberately)
-----------------------------------
Training losses (CTC / rich-transcription CE), SpecAug, tokenizers, feature
extraction, forced alignment and timestamp prediction.  The model is
inference-only and never downloads weights.

Parity notes
------------
* SenseVoice uses its **own** ``LayerNorm`` (``sense_voice/model.py``) that
  inherits the torch default ``eps=1e-5``, *not* the ``eps=1e-12`` norm used by
  the Paraformer SANM blocks.  This module therefore uses
  :class:`funasr_mlx.models.sanm.Float32LayerNorm`.
* ``xs_pad *= output_size ** 0.5`` happens *before* the sinusoidal encoder and
  the encoder returns ``tp_norm`` output while ``olens`` is derived from the mask
  computed at the very beginning.
* Only ``attention_heads``/``linear_units``/... from ``encoder_conf`` matter;
  FunASR's ``positional_dropout_rate`` and friends are training-only knobs.

```
