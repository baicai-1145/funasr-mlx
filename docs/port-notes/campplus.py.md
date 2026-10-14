# campplus.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
Native MLX inference port of FunASR's CAM++ speaker-verification model.

Reference (read-only, ``/Users/baicai1145/repos/funasr``):

* ``funasr/models/campplus/components.py``
* ``funasr/models/campplus/model.py``
* ``funasr/models/campplus/template.yaml`` (published defaults)

This module is deliberately **self-contained** (it duplicates the small
container / checkpoint helpers instead of importing ``funasr_mlx.models.sanm``)
so that it stays correct while the ASR modules are under review, and so that it
adds no dependency between model files.

Layout rules that matter
------------------------
1. **Channels-last convolutions.**  MLX's ``Conv1d`` consumes ``NLC`` and
   ``Conv2d`` consumes ``NHWC``; PyTorch uses ``NCL``/``NCHW``.  This port keeps
   the network's *activation* layout channels-last and therefore transposes only
   the reference's two explicit layout changes (FCM's final reshape and
   ``output_level == "frame"``).  See :meth:`FCM.__call__`.
2. **BatchNorm buffers.**  ``nn.BatchNorm{1,2}d`` carries ``weight``/``bias``
   (only when ``affine=True``) plus the persistent buffers ``running_mean`` and
   ``running_var``, and a torch-only scalar ``num_batches_tracked`` that MLX has
   no equivalent for.  :func:`load_strict` reports the dropped torch-only buffer
   explicitly instead of skipping it silently.
3. **Statistics pooling.**  The reference reduces the *time* axis of an NCL
   tensor.  This port reduces axis 1 of an NLC tensor, which is exactly the same
   quantity; see :class:`StatsPool`.

```
