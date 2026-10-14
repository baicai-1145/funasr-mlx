# fsmn_vad.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
Native MLX inference for FunASR's FSMN voice activity detector (``fsmn-vad``).

FunASR reference sources (read-only, ported verbatim):

* ``funasr/models/fsmn_vad_streaming/model.py`` -- ``FsmnVADStreaming``,
  ``VADXOptions``, ``Stats``, ``WindowDetector``, ``E2EVadSpeechBufWithDoa``,
  ``E2EVadFrameProb``, the frame-state machine (``GetFrameState``,
  ``DetectOneFrame``, ``DetectCommonFrames``, ``DetectLastFrames``) and the
  endpoint bookkeeping (``PopDataBufTillFrame``, ``PopDataToOutputBuf``,
  ``DropCachedFrames``, ``ResetDetection``, ``init_cache``).
* ``funasr/models/fsmn_vad_streaming/encoder.py`` -- ``FSMN``, ``BasicBlock``,
  ``FSMNBlock``, ``LinearTransform``, ``AffineTransform``, ``RectifiedLinear``,
  ``FsmnStack``.
* ``funasr/models/fsmn_vad_streaming/template.yaml`` -- the reference config.

The module is self-contained (no dependency on the other ``funasr_mlx`` model
modules) and imports ``mlx`` at module scope only.

Architecture notes / exact semantics reproduced:

* ``FSMNBlock`` uses a PyTorch ``Conv2d(D, D, (lorder, 1), dilation=(lstride, 1),
  groups=D)`` applied to a ``(B, D, T, 1)`` view.  MLX has no Conv2d, so it is
  expressed as ``mx.conv1d`` over the time axis with layout ``(B, T, D)`` and a
  weight of shape ``(D, K, 1)``.  The cache is the trailing
  ``(lorder - 1) * lstride`` frames of the left-padded tensor, stored as
  ``(B, D, keep)`` (the reference's trailing singleton axis is a Conv2d artifact
  and carries no information).
* ``cache is not None`` selects streaming mode.  FunASR's ``init_cache`` always
  passes a (possibly empty) dict, so VAD inference always runs in streaming
  mode; the cache is created full-length on first use.
* Lookahead / lookback semantics live in the decision layer and are preserved:
  start point is delayed by ``win_size_frame + lookback_time_start_point/frame_in_ms``
  frames, the end point is cut back by
  ``max_end_sil_frame_cnt_thresh/frame_in_ms - lookahead_time_end_point/frame_in_ms - 1``
  frames (``-1`` only when ``do_extend``).
* ``GetFrameState`` computes ``noise_prob = log(sum(silence probs)) *
  speech_2_noise_ratio`` and ``speech_prob = log(1 - sum(...))``, then compares
  ``exp(speech_prob) >= exp(noise_prob) + speech_noise_thres``.  The literal
  ``log``/``exp`` round trip is kept (not simplified to ``1 - p >= p + thres``).
* Decibel is ``10 * log10(sum(frame ** 2) + 1e-6)`` over 25 ms frames shifted by
  10 ms, computed in float32.

Deliberate, behaviour-preserving deviations:

* ``stats.data_buf`` / ``stats.data_buf_all`` are tracked as *sample counts*
  instead of tensors.  FunASR only ever reads ``len(...)`` and slices to advance
  the cursor; the samples themselves are never consumed, so keeping the waveform
  resident would waste O(duration) memory for nothing.
* ``PopDataBufTillFrame`` gets a bounded-loop guard.  FunASR's ``while`` loop can
  spin forever when the buffer runs dry; here it breaks with a warning instead.
* Dropout (identity at inference) and the dynamic-silence schedule that lives in
  ``FsmnVADStreaming.inference``'s Python-level chunking loop are out of scope for
  the tensor model; see the module docstring of the reference.

Weight layout conversion (PyTorch -> MLX):

============================  =========================  ==========================
FunASR / PyTorch key          PyTorch shape              MLX shape
============================  =========================  ==========================
``encoder.in_linear1.linear.weight``   ``(A, D)``      identical
``encoder.fsmn.N.linear.weight``      ``(P, L)``      identical
``encoder.fsmn.N.fsmn_block.conv_left.weight``  ``(D, 1, K, 1)``  ``(D, K, 1)``
``encoder.fsmn.N.affine.linear.weight``  ``(L, P)``    identical
============================  =========================  ==========================

Loading is strict; any missing/unexpected/mismatched key is an error.

```
