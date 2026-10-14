# emotion2vec.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
Native MLX inference port of ``emotion2vec_plus_large``.

Source of truth (read-only, pinned):

* ``github.com/ddlBoJack/emotion2vec`` @ ``main``, ``upstream/models/``
  ``audio.py`` (blob ``329222b2``), ``base.py`` (``c63de01c``),
  ``emotion2vec.py`` (``fe71a344``), ``modules.py`` (``dae553af``),
  ``config.py`` (``29eae511``).
* ``huggingface/emotion2vec/emotion2vec_plus_large`` @ ``6c303ba9``,
  ``config.yaml``.
* ``ConvFeatureExtractionModel`` from ``fairseq`` v0.12.2
  ``fairseq/models/wav2vec/wav2vec2.py``, lines 819-895.  Note this file is
  *not* ``conv_feature.py`` (that path 404s and is absent from fairseq's
  history); ``wav2vec2.py`` is where the class lives.
* Classifier head from the FunASR wrapper
  ``modelscope/FunASR`` @ ``main``, ``funasr/models/emotion2vec/model.py``
  (``Emotion2vec.inference``).  The upstream ``emotion2vec.py`` contains only
  ``Data2VecMultiModel`` and has no head, and the Hub repository ships no
  ``auto_map``/modeling file, so that wrapper is the only source for it.

Architecture for ``emotion2vec_plus_large`` (from the pinned ``config.yaml``):
``embed_dim=1024``, ``depth=8``, ``num_heads=16``, ``mlp_ratio=4.0``,
``norm_eps=1e-5``, ``norm_affine=true``, ``layer_norm_first=false`` (post-norm),
``feature_encoder_spec='[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] +
[(512, 2, 2)]'``, ``extractor_mode=layer_norm``, ``conv_bias=false``,
``normalize=true``, ``conv_pos_depth=5`` / ``conv_pos_width=95`` /
``conv_pos_groups=16`` / ``conv_pos_pre_ln=false`` (so ``k = max(3, 95 // 5) =
19``; the conv's ``padding = 9`` already preserves the time axis and
``SamePad(19)`` is the identity on top of it -- for an even ``k`` fairseq's
``SamePad`` crops the final time slot instead),
``prenet_depth=4``, ``num_extra_tokens=10``, ``learned_alibi=false``,
``learned_alibi_scale=true`` with ``per_head=true`` / ``per_layer=false``,
``use_alibi_encoder=true``, ``num_alibi_heads=16``, ``alibi_scale=1.0``.

Deliberately excluded (training-only, see :data:`INFERENCE_ONLY_PREFIXES`):
``modality_encoders.AUDIO.decoder.*`` (:class:`Decoder1d`), the EMA, mask
construction (``compute_mask``/``apply_mask``/``MaskInfo``), layerdrop,
drop-path, and every dropout.  At inference they are identities.

Layout rules (checkpoint ``state_dict`` -> MLX)

=========================  =========================  =====================
key suffix                 PyTorch shape             MLX shape
=========================  =========================  =====================
``*.conv_layers.N.0.weight``  ``(out, in//g, K)``     ``(out, K, in//g)``
``*.project_features.2.weight`` ``(out, in)``       ``(out, in)``
``*.relative_positional_encoder.N.0.weight`` ``(C, C//g, K)`` ``(C, K, C//g)``
``*.norm1.weight`` / ``.bias``  ``(D,)``           ``(D,)``
``*.extra_tokens``         ``(1, N, D)``            ``(1, N, D)``
``*.alibi_scale``          ``(1, 1, H, 1, 1)``       ``(1, 1, H, 1, 1)``
=========================  =========================  =====================

Traps this file deliberately avoids: MLX ``nn.Module`` stores children in its
*dict* part (so the loader uses ``module.parameters()``, never ``vars(module)``);
``mx.transpose`` takes a single axes sequence; every ``mx.nn.Conv*`` is
channels-last so each call site transposes ``NCT -> NTC`` and back; ``mx.take``
is always given an explicit ``axis``.

No numerical test was executed for this port; see the task report for what is
and is not verified.

```
