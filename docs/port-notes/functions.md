# Long function/class docstrings extracted from the code

(Kept verbatim; each entry names its file and symbol.)

## pseudo_streaming.py :: PseudoStream

```
Growing-buffer pseudo-streaming over any transcription callable.

    Parameters
    ----------
    transcribe_fn:
        ``f(samples: np.ndarray) -> str`` -- OFFLINE transcription of a
        waveform prefix. For classic models use
        :func:`funasr_mlx.pseudo_streaming.classic_transcriber`; for LLM
        families use :meth:`LlmAsrTranscriber.transcribe_samples` via
        :func:`llm_transcriber`.
    sample_rate: expected rate (only for reporting seconds).
    chunk_seconds: push granularity the caller commits to (informational;
      the protocol is pull-based, any push size works).
    min_push_seconds: ignore pushes smaller than this to bound recompute.
    
```

## checkpoints.py :: _convert_safetensors_source

```
One-tensor-at-a-time conversion of a safetensors source.

    Memory contract: each shard is mmapped by the safetensors reader, one
    tensor is copied into a small owned buffer, converted, and released before
    the next tensor is touched. Peak extra memory is one tensor, not one model.

    Key discipline: every source key is passed through the declared
    ``source_rule`` (which raises on an unrecognised key) and duplicate native
    names are rejected, so a renamed or partially-downloaded upstream
    checkpoint fails the conversion instead of silently producing an
    incomplete native checkpoint.
    
```

## checkpoints.py :: convert_checkpoint

```
Convert a local checkpoint to the native layout, one tensor at a time.

    Two source formats are supported: a torch ZIP (``.pt``, mmap+``weights_only``)
    and HF/native ``.safetensors``. The safetensors path never materialises the
    whole state dict: each shard is mmapped, its tensor copied one at a time, and
    the copy released before the next (see :func:`_convert_safetensors_source`).

    ``source_rule`` selects a declared :data:`SOURCE_RULES` entry, which both
    renames keys and (via :func:`convert_tensor`) fixes tensor layout. A
    safetensors source without a rule keeps its parameter names unchanged, so
    the family's model class must expect exactly those names.

    ``gptq`` supplies per-tensor Hessians (:class:`GptqWeight`) so the selected
    matrices get Hessian-compensated values instead of round-to-nearest.  The
    output layout is unchanged either way.

    Only call inside the guarded launcher. No network or pickle fallback.
    Strict inference loading later checks the architecture's keys and shapes.
    
```

## checkpoints.py :: quantize_bf16_weight

```
Pack one 16-bit weight with MLX group quantization.

    ``array`` arrives as a NumPy carrier: ``uint16`` raw words for a bfloat16
    source (NumPy has no bfloat16) or a real ``float16`` array. The result is
    keyed by native parameter name: ``<module>.weight`` (``uint32`` packed
    codes), ``<module>.scales`` and ``<module>.biases`` (both carried in the
    source's 16-bit dtype). Those are exactly the three parameter names
    ``nn.QuantizedLinear`` owns -- note the side tensors hang off the *module*,
    not off ``weight``.

    The quantization itself runs in MLX (``mx.quantize``) on the device, which
    is the only implementation guaranteed to agree bit-for-bit with
    ``mx.quantized_matmul``'s expectations; the input is freed before the
    packed copy is materialised on the host.

    ``gptq`` switches the *values* to Hessian-based error compensation while
    keeping the layout identical.  It is a :class:`GptqWeight`: the matrix's
    ``H = 2 X^T X`` plus the GPTQ hyper-parameters, keyed by native tensor name
    so a converter can pass one mapping for every quantized tensor it meets.
    The packing still goes through the same bit order, so nothing downstream
    changes.
    
```

## checkpoint_stream.py :: write_shard

```
Write a whole shard whose tensors are all carried in one dtype.

    ``code`` is the safetensors type code written for every tensor. It exists
    because a bfloat16 shard has to be written as raw ``uint16`` words under
    the ``BF16`` code -- NumPy cannot name a bfloat16 dtype, so
    ``safetensors.numpy.save_file`` would write ``U16`` and the loader would
    then see integer weights.
    ``codes`` overrides the code per tensor name, for shards that mix element
    widths: a W4A16 shard holds 4-byte packed ``uint32`` codes beside 2-byte
    ``BF16`` group scales. Exactly one of ``code``/``codes`` must be given, and
    every per-tensor code is still checked against that tensor's element width.
    
```

## paraformer.py :: tail_process_fn

```
Port of ``CifPredictorV2.tail_process_fn`` with ``tail_mask=True``.

        The reference builds ``tail_threshold * (cat([ones, mask]) - cat([mask,
        zeros]))`` and adds it to a zero-padded ``alphas``.  Working through that
        shift-and-subtract, the coefficient is ``1.0`` at index ``i`` when
        ``mask[i-1] != mask[i]``, and at the **appended** slot when
        ``mask[T-1] == 1``.  So:

        * a **full** mask (the utterance fills the padded window) puts the
          threshold on the **appended** column -- it is *not* zeroed;
        * a **padded** mask puts it on the **first padded frame**, and the
          appended column stays zero.

        Both branches are reproduced verbatim; see
        ``tests/test_asr.py::test_cif_tail_lands_on_the_first_padded_frame_or_the_appended_slot``.

        ``token_num`` is floored here (reference ``cif_predictor.py:461``), which
        is the only place a floor belongs.
        
```

## paraformer.py :: cif_v1

```
Port of ``funasr.models.paraformer.cif_predictor.cif_v1``.

    Returns ``(frame_fires, fires)`` where ``frame_fires`` has shape
    ``(B, max_label_len, D)``.  ``fires`` is the ``cif_peak`` signal used for
    timestamps (values ``>= 1.0`` mark an integration boundary).

    The fire ordering, the boundary algebra and the lengths mask are the
    reference's; only the *device* differs, and it is pinned to
    :data:`CIF_DEVICE` because the Metal backend has been observed to corrupt
    the index arrays this depends on.  See :data:`CIF_DEVICE` for the evidence
    and for why this is not a warmup.  ``device=mx.gpu`` opts back into the GPU
    path for future verification; it is not known to be correct.

    ``mx.eval`` returns ``None`` here (unlike ``torch``), so the values are
    read back from the already-materialised arrays rather than its return value.
    
```

## paraformer.py :: cif_wo_hidden_v1

```
Port of ``funasr.models.paraformer.cif_predictor.cif_wo_hidden_v1``.

    ``threshold`` is unused in the reference implementation (it only gates the
    downstream comparison) but is kept for signature parity.

    Evaluated on :data:`CIF_DEVICE` unless ``device`` overrides it; the whole
    subtree is forced inside that stream so no intermediate escapes.

    The prefix sum accumulates in ``float64`` and is narrowed to ``float32``
    afterwards, exactly like the reference's
    ``torch.cumsum(..., dtype=torch.float64).to(torch.float32)``.
    
```

## sensevoice.py :: SenseVoiceEncoderSmall

```
``funasr.models.sense_voice.model.SenseVoiceEncoderSmall``.

    Layout of the produced parameter names matches the checkpoint exactly::

        encoder.encoders.<i>.self_attn.linear_q_k_v.weight
        encoder.encoders.<i>.self_attn.fsmn_block.weight
        encoder.encoders.<i>.feed_forward.w_1.weight
        encoder.encoders.<i>.norm1.weight
        encoder.after_norm.weight
        encoder.tp_encoders.<j>....
        encoder.tp_norm.weight
    
```

## emotion2vec.py :: flat_parameter_names

```
``{dotted.name: array}`` for every parameter leaf.

    Uses ``mlx.utils.tree_flatten`` on the module itself -- **not**
    ``vars(module)``, because ``nn.Module`` is a ``dict`` subclass and
    ``__setattr__`` stores arrays and submodules in the dict part, so ``vars()``
    reports zero tensors.

    ``mlx.utils.tree_flatten`` returns a **list of ``(key, value)`` tuples by
    default**; passing ``destination={}`` is what makes it return the dict form,
    so calling ``.items()`` on the default return raises.  ``vars(module)`` is
    not used either, because ``nn.Module`` is a ``dict`` subclass and
    ``__setattr__`` stores arrays and submodules in the dict part, so ``vars()``
    reports zero tensors.

    ``module.parameters()`` would also work here: its ``valid_parameter_filter``
    (``mlx/nn/layers/base.py:235``) is
    ``isinstance(value, (dict, list, mx.array)) and not key.startswith("_")``,
    so biases, LayerNorm weights, ``extra_tokens`` and ``alibi_scale`` are all
    retained.  ``tree_flatten`` is used directly because it needs no filter
    argument and keeps the full key space.

    Non-array leaves (ints, strings, tuples) are filtered out.
    
```

## emotion2vec.py :: __call__

```
``x`` is ``NTC`` (channels-last); returns ``NTC``.

        **Layout contract.** MLX's ``Conv1d`` is channels-*last* while the
        reference works in channels-*first* ``NCT``, so the two cannot literally
        be fed the same axes:

        ==============  ==========================  ==========================
        source step     what it needs               what this port feeds it
        ==============  ==========================  ==========================
        ``0`` conv      ``NCT`` (N, C, T)           ``NTC`` -- swapped in
                                                     before the call and back
        ``1`` SamePad   last axis must be **time**  ``NCT``, so the reference's
                                                     ``x[:, :, :-remove]``
                                                     crop applies to T
        ``3`` LayerNorm last axis must be **C**    ``NTC``, swapped in and back
        ==============  ==========================  ==========================

        Handing the raw conv an ``NCT`` tensor makes MLX read the channel count
        as the *length*, which fails loudly whenever the channel count is not
        divisible by ``groups`` -- e.g. an 8-channel tensor reaching a
        ``groups=4`` conv as ``(1, 8, 5)``.

        ``TransposeLast`` is a **real** ``swapaxes(-2, -1)``, not a marker, so the
        untouched source chain runs in ``NCT`` and only the conv boundary
        deviates:

        ==============  ==========================  ========================
        source step     what it does                what this port feeds it
        ==============  ==========================  ========================
        ``0`` conv      ``NCT`` in, ``NCT`` out     ``NTC`` boundary, swapped
                                                      in and back out
        ``1`` SamePad   crop the last axis == **T** ``NCT``
        ``2``/``4``     ``NCT`` -> ``NTC`` ->       keeps the channels on the
                       ``NCT``                    norm's last axis
        ``3`` LayerNorm reduce the channel axis    ``NTC``
        ==============  ==========================  ========================
        
```

## emotion2vec.py :: SamePad

```
``fairseq.modules.SamePad`` (non-causal), ``same_pad.py:18-22``.

    ``remove = 1 if k is even else 0`` and the module **crops**, it never pads::

        if self.remove > 0:
            x = x[:, :, : -self.remove]

    So ``k = 19`` (the large config) is the identity for every length, and an even
    ``k`` drops the **last** time element.  Combined with the conv's
    ``padding = k // 2`` the time axis is preserved exactly.

    The reference slice indexes the last axis; :class:`PositionalBlock` therefore
    presents the tensor as ``NCT`` across this call, because MLX's ``Conv1d``
    needs channels-last while the crop needs time-last.
    
```

## emotion2vec.py :: __call__

```
``x`` is ``NCT`` (PyTorch layout); returns ``NCT``.

        The children are registered **directly** under their Sequential indices,
        so the checkpoint keys are exactly ``conv_layers.<i>.0.weight`` and
        ``conv_layers.<i>.2.1.weight`` / ``.bias``.  An intermediate attribute
        would insert a spurious ``.block`` namespace level.

        **Layout contract.** The container is the source's chain and every child
        does exactly what the source does -- ``TransposeLast`` is a **real**
        ``swapaxes(-2, -1)`` here, not a marker -- so the chain runs in ``NCT``
        and only the conv boundary needs help: MLX's ``Conv1d`` is
        channels-*last*.  Swapping in and back out around it is the single
        deviation; everything after it is the untouched source order.
        
```

## ct_punc.py :: forward_fsmn

```
Depthwise memory branch; the mask is applied before *and* after the conv.

        Layout: the convolution itself is ``mx.nn.Conv1d``, which is channels-last
        ``(batch, length, channels)``.  The reference works in ``(batch, channels,
        time)`` so the time axis is moved to the middle, the reference's
        ``ConstantPad1d((left, right))`` is applied to it, and the result is moved
        back before it is combined with the input.

        ``mx.core.conv1d`` reads ``in[t * stride + k * dilation]``; with
        ``y[i] = x[i - left_pad]`` kernel tap ``k`` therefore reads
        ``x[t - left_pad + k]``.
        
```

## sanm.py :: PositionalEncoding

```
``funasr.models.transformer.embedding.PositionalEncoding``.

    Fixed sin/cos table scaled by ``sqrt(d_model)``.  The table is not a
    checkpoint parameter in FunASR (it is a plain attribute there), so it is
    regenerated on demand here instead of being stored as a buffer.

    **Interleaved, not concatenated.**  The reference writes
    ``pe[:, 0::2] = sin(...)`` / ``pe[:, 1::2] = cos(...)``, i.e. even columns
    hold sin and odd columns hold cos.  Do not "simplify" this into a
    ``concatenate([sin, cos])``: that is a different table and every column from
    ``d_model // 2`` onward would be wrong.  Note that
    :class:`SinusoidalPositionEncoder` genuinely *does* concatenate (see
    ``embedding.py``), so the two must not be made to match.
    
```

## sanm.py :: ModuleStack

```
A stack of sibling modules with PyTorch-compatible parameter names.

    FunASR's ``repeat()`` builds an ``nn.Sequential`` (``encoders.0.<leaf>``) and
    the SenseVoice encoder uses ``nn.ModuleList`` (``encoders.0.<leaf>``).
    MLX's own containers would nest everything under ``layers``, so children
    are registered directly under their numeric index.

    ``mlx.nn.Module`` subclasses ``dict``, so those children live in the mapping
    storage under the keys ``"0"``, ``"1"``, ... -- which is exactly the PyTorch
    ``state_dict`` spelling.  The dunders below therefore forward to ``dict`` and
    must not shadow it: MLX internally calls ``dst[key]``, ``key in dst`` and
    ``dst.items()``, so ``__getitem__`` accepts string keys unchanged.
    
```

## sanm.py :: load_strict

```
Load ``tensors`` into ``module``, refusing any key mismatch.

    Parameters
    ----------
    module:
        Destination module.
    tensors:
        ``{name: array}`` using FunASR/PyTorch names, e.g.
        ``encoder.encoders.0.self_attn.fsmn_block.weight``.  The *arrays* must
        already be in MLX layout unless ``convert=True``.
    prefix:
        Optional key prefix shared by ``tensors`` (e.g. ``"model."``).
    convert:
        ``True`` when ``tensors`` come straight from a PyTorch ``state_dict``;
        applies :func:`convert_checkpoint_tensor` (conv weight transposes).
    dtype:
        Optional cast applied *after* the layout conversion.

    Returns
    -------
    list[str]
        The names that were loaded, sorted.

    Raises
    ------
    CheckpointKeyError
        If any expected parameter is missing or any checkpoint tensor is
        unexpected.  Shape mismatches raise ``ValueError``.
    
```

## sanm.py :: _flatten_into

```
Collect tensors from a module tree.

    ``mlx.nn.Module`` **subclasses ``dict``** (``class Module(dict)`` in
    ``mlx/nn/layers/base.py``).  Its ``__setattr__`` routes arrays, dicts, lists,
    tuples and *sub-modules* (a ``Module`` is itself a ``dict``, so
    ``isinstance(v, dict)`` holds) into the **mapping** storage via
    ``self[key] = val``; everything else -- ints, strings, dataclasses and the
    ``_``-prefixed configuration attached here -- goes to ``__dict__``.

    Consequently ``vars(module)`` contains **no tensors at all** (only
    ``_no_grad``/``_training``), and flattening through it silently yields an
    empty dict.  The mapping storage is walked instead.
    
```

## sanm.py :: attention_blocked

```
Boolean ``True`` where attention is blocked, for any mask dtype.

    The reference does ``mask.unsqueeze(1).eq(0)`` on the **float** 1.0/0.0 mask
    that FunASR's ``sequence_mask`` produces (its ``dtype`` defaults to
    ``torch.float32``), so ``.eq(0)`` marks the padded positions.

    Rather than re-implementing ``== 0`` on a dtype-agnostic array -- whose
    answer depends on MLX's comparison type promotion between ``bool`` and
    ``int``, i.e. whether ``False == 0`` is honoured -- this branches on the
    mask's own dtype explicitly:

    * ``bool`` masks mean "True = valid", so blocked is the logical negation.
      ``logical_not`` is unambiguous and needs no promotion.
    * numeric masks reproduce the reference's ``.eq(0)`` exactly, comparing
      within a single dtype.

    Both conversions are exact for the float masks this port actually produces,
    so behaviour is unchanged while a boolean mask stops being silently ignored.
    
```

## whisper.py :: detect_language

```
Raw per-language log-probabilities for language identification.

    Tokenizer-free port of ``whisper.decoding.detect_language``.  The caller
    supplies

    * ``prompt_tokens`` -- the SOT sequence (for the multilingual models this is
      ``tokenizer.sot_sequence_including_notimestamps``), and
    * ``language_token_ids`` -- the ``<|lang|>`` token ids,

    and receives ``(token_id, logprob)`` pairs sorted by descending
    log-probability.  Only the language-token columns are sampled, as in the
    reference.  Mapping ids back to language names is the tokenizer owner's job.
    
```

## whisper.py :: greedy_decode

```
Autoregressive greedy decoding with a bounded KV cache.

    Semantics ported from ``whisper.decoding.DecodingTask`` with ``beam_size``
    and ``best_of`` both 1:

    * the full prompt is processed first (building the cache),
    * afterwards exactly **one** token per step is fed,
    * ``SuppressBlank`` forbids ``[space, eot]`` at the first generated position,
    * ``SuppressTokens`` forbids the caller-supplied ids at every step,
    * decoding stops at ``eot_token_id`` or after ``max_tokens``.

    Returns token ids only.  There is no beam search, no temperature fallback
    and no text conversion.
    
```

## whisper.py :: log_mel_spectrogram

```
Port of ``whisper.audio.log_mel_spectrogram`` (channel axis last).

    Parameters
    ----------
    audio:
        Waveform ``(B, T)`` (float).  Use :func:`pad_or_trim` first.
    center:
        ``True`` reproduces ``torch.stft``'s reflect padding by ``n_fft // 2``.
    filters:
        Precomputed ``(n_mels, n_fft // 2 + 1)`` mel filterbank.  Defaults to
        :func:`mel_filterbank`.
    max_frames:
        Optional hard cap on the number of output frames, as a guard against
        unbounded allocation.

    Returns
    -------
    ``(B, T' - 1, n_mels)`` where ``T'`` is the number of STFT frames -- i.e.
    ``n_samples // hop_length`` frames, matching the reference's ``[..., :-1]``
    drop of the final frame.
    
```

## whisper.py :: __call__

```
``x`` is ``(B, T', n_mels)`` -> ``(B, T'/2, n_state)``.

        ``conv1`` has **no** stride and only ``conv2`` halves the time axis, so
        the contract is exactly ``2 * n_ctx`` mel frames in and ``n_ctx`` frames
        out.

        The incoming mel is cast to the **weight** dtype first.  The front-end
        always produces float32 mel, so without this cast an fp16 checkpoint
        would promote the first conv to float32 and leave the whole encoder in
        mixed precision, paying a promotion at every conv and Linear (measured
        on M4: 22.67 ms vs 17.48 ms for the 30 s encoder window).  For a float32
        checkpoint the cast is a no-op, so its numerics stay bit-identical.
        
```

## whisper.py :: ResidualAttentionBlock

```
``whisper.model.ResidualAttentionBlock`` (canonical OpenAI, v20250625).

    Plain pre-norm residuals with **no** weighting: ``attn``, then
    ``cross_attn``, then ``mlp``, each added at full strength.

    The SenseVoice vendored fork (``funasr/models/sense_voice/whisper_lib``)
    scales these branches by ``attn_pct`` / ``1 - attn_pct``, where
    ``attn_pct`` is ``0.0`` for cross-attention blocks -- i.e. it *discards
    self-attention entirely* in the decoder.  That fork is **not** what
    ``funasr/models/whisper/model.py`` runs: that module does ``import whisper``
    (line 11) and builds ``whisper.model.Whisper`` (line 60).  The gates are
    therefore removed, and the cross-attention branch is applied **before** the
    MLP as in the canonical source.
    
```

## whisper.py :: MultiHeadAttention

```
``whisper.model.MultiHeadAttention`` with an explicit MLX KV cache.

    Cache semantics are a direct port of the reference's forward hooks
    (``install_kv_cache_hooks``) plus the ``self.key not in kv_cache`` branch in
    ``forward``:

    * **Self-attention** (``xa is None``) always recomputes ``k``/``v`` and
      *appends* them to the cache, so ``k`` covers ``[0, offset + t)``.
    * **Cross-attention** (``xa is not None``) computes once; later steps read the
      cached tensors unchanged.

    Cache keys are strings of the form ``"blocks.<i>.attn.key"`` /
    ``"blocks.<i>.cross_attn.value"`` so the cache is inspectable and
    deterministic, instead of the reference's module-object keys.
    
```

## whisper.py :: _flatten_into

```
Collect tensors from a module tree.

    ``mlx.nn.Module`` **subclasses ``dict``** (``class Module(dict)`` in
    ``mlx/nn/layers/base.py``).  Its ``__setattr__`` routes arrays, dicts, lists,
    tuples and *sub-modules* (a ``Module`` is itself a ``dict``, so
    ``isinstance(v, dict)`` holds) into the **mapping** storage via
    ``self[key] = val``; everything else -- ints, strings, dataclasses and the
    ``_``-prefixed configuration attached here -- goes to ``__dict__``.

    Consequently ``vars(module)`` contains **no tensors at all** (only
    ``_no_grad``/``_training``), and flattening through it silently yields an
    empty dict.  The mapping storage is walked instead.
    
```

## whisper.py :: convert_checkpoint_tensor

```
Convert one PyTorch checkpoint tensor into the MLX layout.

    Parameters
    ----------
    name:
        Parameter name, used only for diagnostics and error messages.
    array:
        The tensor to convert.
    rule:
        **Explicit** rule name, normally taken from
        :func:`flatten_state_dict_with_rules`, which derives it from the owning
        module type.  There is deliberately **no name-based inference**: here the
        same attribute name (``conv1``/``conv2``/``linear``) can belong to an
        ``nn.Linear`` or to a ``nn.Conv1d``, so guessing from the name could
        silently apply the wrong transpose.  The default is identity, and a wrong
        choice surfaces as a clear shape mismatch at load time rather than as a
        silently mis-converted weight.
    
```

## campplus.py :: embed

```
Length-aware embedding extraction for a padded batch.

        Each utterance is cropped to its true length and forwarded
        **individually**.  A padded batch is deliberately *not* forwarded as-is:
        CAM++'s convolutional trunk sees the whole time axis, including
        ``CAMLayer.seg_pooling``, which averages over **all** frames.  Masking
        only the final statistics pooling therefore cannot undo the leak, and
        the reference has exactly the same property -- it pools ``hidden`` with no
        mask at all (``CAMPPlus.forward``).  Cropping per utterance is what makes
        this equivalent to the reference's real usage, where
        ``funasr/models/campplus/utils.py::extract_feature`` produces
        per-utterance features and batching is only a convenience.

        Cost: one forward per utterance instead of one per batch.  For
        ``output_level="frame"`` the rows have different lengths, so the padded
        batch is returned unchanged (and callers must mask themselves).
        
```

## campplus.py :: FCM

```
``components.FCM`` -- the mel-axis feature-map module.

    The reference treats the mel axis as the spatial *height* (it halves it
    eight times in total via two stride-2 residual stages and a stride-2 conv)
    and the time axis as the *width*.  Its final
    ``out.reshape(B, C * H, W)`` therefore folds channels into the mel axis,
    producing ``(B, m_channels * (feat_dim // 8), T)``.

    Here the tensor is ``(B, H, W, C)`` throughout, so the equivalent result is
    ``(B, W, C * H)`` obtained by transposing to ``(B, W, C, H)`` before the
    reshape -- which keeps PyTorch's ``(C, H)`` row-major ordering.
    
```

## campplus.py :: BatchNorm

```
``torch.nn.BatchNorm{1,2}d`` for channels-last MLX activations.

    The input's **last** axis must be the channel axis, so statistics are
    reduced over every leading axis, exactly like the reference reduces over
    ``(N, L)`` / ``(N, H, W)``.

    Eval semantics (the only mode used for inference) are
    ``(x - running_mean) / sqrt(running_var + eps) * weight + bias`` with
    ``running_var`` holding the *unbiased* sample variance, as in PyTorch.

    ``torch.nn.BatchNorm*`` additionally persists a scalar
    ``num_batches_tracked`` that has no computational role and no MLX analogue;
    :func:`load_strict` reports it in ``LoadReport.dropped``.
    
```

## campplus.py :: load_strict

```
Load ``tensors`` into ``module``, refusing any key mismatch.

    Parameters
    ----------
    module:
        Destination module.
    tensors:
        ``{name: array}`` using FunASR/PyTorch names, e.g.
        ``xvector.block1.tdnnd1.cam_layer.linear1.weight``.  Arrays must
        already be in MLX layout unless ``convert=True``.
    prefix:
        Optional shared key prefix (e.g. ``"model."``).
    convert:
        ``True`` when ``tensors`` come straight from a PyTorch ``state_dict``.
    dtype:
        Optional cast applied after the layout conversion.
    allow_torch_only_buffers:
        Accept (and report) torch-only BatchNorm scalars such as
        ``num_batches_tracked``, which MLX does not model.  Set ``False`` to make
        their presence an error.

    Raises
    ------
    CheckpointKeyError
        On missing or unexpected keys (other than the allowed buffers).
    ValueError
        On any shape mismatch.
    
```

## campplus.py :: _flatten_into

```
Collect tensors from a module tree.

    ``mlx.nn.Module`` **subclasses ``dict``** (``class Module(dict)`` in
    ``mlx/nn/layers/base.py``).  Its ``__setattr__`` routes arrays, dicts, lists,
    tuples and *sub-modules* (a ``Module`` is itself a ``dict``, so
    ``isinstance(v, dict)`` holds) into the **mapping** storage via
    ``self[key] = val``; everything else -- ints, strings, dataclasses and the
    ``_``-prefixed configuration attached here -- goes to ``__dict__``.

    Consequently ``vars(module)`` contains **no tensors at all** (only
    ``_no_grad``/``_training``), and flattening through it silently yields an
    empty dict.  The mapping storage is walked instead.
    
```

## campplus.py :: convert_checkpoint_tensor

```
Convert one PyTorch checkpoint tensor into the MLX layout.

    Parameters
    ----------
    name:
        Parameter name, used only for diagnostics and error messages.
    array:
        The tensor to convert.
    rule:
        **Explicit** rule name, normally taken from
        :func:`flatten_state_dict`, which derives it from the owning module type.
        There is deliberately **no name-based inference**: in this architecture the
        same attribute name (``linear``) belongs to an ``nn.Linear`` in
        ``DenseLayer``/``TransitLayer``'s sibling and to a ``nn.Conv1d`` in
        ``TDNNLayer``, so guessing from the name could silently apply the wrong
        transpose.  The default is identity, and a wrong choice surfaces as a
        clear shape mismatch at load time rather than as a silently mis-converted
        weight.
    
```

## gptq.py :: hessian

```
The combined ``H = 2 X^T X``, optionally language-balanced.

        ``balance=True`` scales each language's contribution to the smallest
        per-language row count, so every language contributes equally and none
        can exceed ``1 / n_languages`` of the mass.  ``cap_fraction`` instead
        caps the largest language's *share* at that fraction by lowering a
        common row threshold (see :meth:`_effective_rows`), which keeps more of
        a large corpus while still bounding the dominant language.

        ``language_weights`` multiplies the balanced per-language contributions
        by explicit factors.  Equal mass is the right *default* (it is what the
        brief's 30% guard asks for), but it is not the right target when the
        languages do not quantize equally well: if one language's transcripts
        diverge far more than another's at the same bit width, equal mass is
        under-weighting exactly the directions that are failing.  Weighting by
        measured divergence spends a fixed bit budget where the error actually
        shows up.  Weights are renormalised, so this redistributes mass between
        languages without changing the total -- an overall scale would silently
        alter the damping, which is defined relative to the mean diagonal.
        
```

## gptq.py :: HessianAccumulator

```
``H = 2 X^T X`` in float32, with a per-language balance guard.

    Calibration activations are added per utterance with a ``language`` tag.
    Because Chinese and English have far more local data than the other eight
    languages, raw row counts would put ~85% of the Hessian mass on two
    languages; :meth:`hessian` therefore reweights each language's contribution
    so every language contributes the same number of effective rows (see the
    ``balance`` argument), which keeps any single language below the ~30% mass
    guard the brief sets.

    The accumulator is model-agnostic: the caller decides which activations to
    feed it.  ``columns`` must be the matrix's input dimension.
    
```

## gptq.py :: output_error

```
Total squared layer-output error of a reconstruction, from ``H`` alone.

    For ``D = W - Ŵ`` and ``H = 2 X^T X``::

        sum_n ||D x_n||^2 = trace(D (H/2) D^T)

    so no activations are needed once the Hessian exists -- which is what makes
    this reportable for a matrix whose calibration set is 600 utterances long.

    Computed in row blocks.  ``trace(D M D^T)`` decomposes as a sum over the
    ROWS of ``D`` (the ``i``-th row contributes ``D[i] M D[i]^T``), so this is
    exact -- and it has to be blocked, because materialising ``D @ M @ D.T`` for
    a 151936-row head needs a 151936x151936 intermediate.
    
```

## gptq.py :: GptqOutcome

```
The three tensors a quantized checkpoint stores, plus error accounting.

    Two error metrics are reported, and they are not interchangeable:

    ``error``
        Mean squared **Frobenius** error ``mean((W - Ŵ)^2)`` -- the plain
        elementwise distance.
    ``output_error``
        The quantity GPTQ actually minimizes: ``trace(D (H/2) D^T)`` with
        ``D = W - Ŵ``, i.e. the total squared error of the layer's *outputs*,
        ``sum_n ||(W - Ŵ) x_n||^2``, over the calibration activations.  This is
        the metric that matters for the LM head (it is the logits error), and
        it is computable from the Hessian alone.

    GPTQ is guaranteed to beat round-to-nearest on ``output_error``, not on
    ``error``: it deliberately spends unweighted error to buy weighted error.
    Reporting only the Frobenius number would make a working GPTQ look worse
    than RTN, so both are always carried.
    
```

## gptq.py :: params

```
Per-row ``(scale, bias)`` for one group, exactly as the kernel does.

        ``block`` is ``[rows, group_size]`` float32.  The kernel's three steps
        are reproduced verbatim:

        1. ``scale = max((max - min) / n_bins, eps)``, with ``max`` floored at
           zero (the kernel initializes ``w_max = 0``);
        2. the sign is chosen by which end has the larger magnitude -- so the
           stored bias is that end and the scale's sign matches;
        3. the scale is renormalized to ``edge / round(edge / scale)``, so the
           edge value lands exactly on an integer code.  A group whose edge
           rounds to zero keeps the raw scale and a zero bias.
        
```

## quant.py :: activation_dtype

```
The dtype a weight-bearing module emits, for activations to follow.

    A family splices the audio tower's output into the decoder's token stream,
    and has to cast it to the dtype the decoder expects; every such site reads
    it off ``embed_tokens``.  For a plain module that is ``weight.dtype``.

    That is WRONG once the module is quantized: ``QuantizedEmbedding`` (and
    ``QuantizedLinear``) store ``weight`` as packed ``uint32`` *codes*, so
    ``weight.dtype`` is ``uint32`` and the cast turns the audio embeddings into
    integers -- silently destroying every one of them before the first decoder
    layer.  The dtype that actually flows out is the side tensor's (``scales``),
    which the converter writes in the checkpoint's 16-bit dtype.

    This was a real, hard-to-see failure: a quantized checkpoint produced
    garbage from step 0 (``'language None<|im_start|><|im_end|>'``) while the
    bf16 checkpoint was perfect, and it looked exactly like "4-bit is too coarse
    for the tied embedding" -- so it was chased as an accuracy problem across
    GPTQ, act-order, w8, and mixed Hessians before the cast was found.  No
    quantization scheme can fix it; the bug is in the cast.
    
```

## calibrate.py :: HeadActivationCollector

```
Accumulates ``H = 2 X^T X`` for one head, one utterance at a time.

    Usage::

        target = HeadTarget.resolve(model, head_path="thinker.model.embed_tokens")
        collector = HeadActivationCollector(target)
        with collector:
            for sample in calibration_samples:
                collector.observe(language=sample.lang, run=prefill_callable)

    ``observe`` runs ``run()`` with the hook installed and records every hidden
    state the head is called with during that call.
    
```


---

# Second wave: shorter docstrings (moved verbatim)

## api.py :: pseudo_stream

```
Create a PseudoStream bound to a loaded LLM transcriber.

    Must run inside a guard (use with TranscribeWorker / guarded child):

        worker = TranscribeWorker(model)
        stream = worker.pseudo_stream(chunk_seconds=1.0)
        for chunk in mic_chunks:
            ev = stream.push(chunk)
            if ev: render(ev.text)
        final = stream.close()   # == offline output, bit-identical
    
```

## api.py :: download_model

```
Download a published checkpoint (see funasr_mlx.download.download).
```

## checkpoint_stream.py :: write_tensor

```
Write one tensor, converting only bounded slices rather than its full data.

    ``code`` overrides the safetensors type code written for the payload; it is
    how a bfloat16 tensor (carried as a ``uint16`` array, since NumPy has no
    bfloat16) is stored as ``BF16`` rather than ``U16``. The override must agree
    with the array's element width, so no reinterpretation can silently happen.
    
```

## checkpoints.py :: _bf16_carrier_to

```
Round a ``uint16``-carried bfloat16 tensor to the requested precision.

    bfloat16 -> float32 is exact (a 16-bit left shift; the shift must be done
    in 32-bit lanes or the high half is discarded). The further round to
    float16 happens on the host, exactly as an on-device ``astype`` would.
    
```

## checkpoints.py :: _carry

```
Copy one source tensor into a small owned array in its carrier dtype.

    The copy is the point: the source buffer belongs to a lazily-read shard
    handle, and converted arrays that still referenced it would pin the whole
    shard in memory. Every layout transpose therefore happens *after* this
    copy, on the owned array. bfloat16 is viewed as ``uint16`` first because
    ``torch.Tensor.numpy`` refuses the bfloat16 scalar type outright.
    
```

## checkpoints.py :: quantize_tensor

```
Quantize one tensor and return MLX tensors in the usual layout.
```

## checkpoints.py :: GptqWeight

```
Per-tensor GPTQ calibration state, keyed by native tensor name.

    ``hessians`` maps a native ``<module>.weight`` name to that matrix's
    ``H = 2 X^T X`` (float32).  ``configs`` maps the same name to the
    :class:`~funasr_mlx.models.llm.gptq.GptqConfig` to use, so different
    matrices can carry different hyper-parameters.  A name with a Hessian but no
    explicit config uses ``default_config``.

    Kept as a plain data holder (no MLX import at module scope) so the converter
    can accept it without pulling the runtime in.
    
```

## checkpoints.py :: SourceRule

```
How one LLM family's source keys map onto the native parameter names.

    ``prefixes`` maps a source key prefix to the native prefix, longest first.
    ``renames`` are literal substring rewrites applied afterwards (used for the
    Qwen3-ASR ``conv2d`` -> ``conv`` submodule move and for MOSS's
    ``language_model`` -> ``model`` decoder nesting). ``drop`` lists source keys
    that the native model does not carry, because the port *derives* them (the
    Qwen3-ASR sinusoidal position table). A rule is strict: a key that matches
    no prefix and is not in ``drop`` raises rather than being passed through, so
    a renamed upstream checkpoint cannot silently produce a wrong key set.
    
```

## llm_asr.py :: LlmAsrTranscriber

```
Resident transcriber: one model load, many calls.

    Construct inside a guarded child (``memory.run_guarded``); the API layer
    arranges that. KV caches are allocated once and reset (offset=0) between
    utterances -- reuse validated bit-stable in .tmp/opt/LOG.md.
    
```

## pipeline.py :: load_model

```
Prevalidate every key/shape, then load one bounded shard at a time.

    Only native-layout, explicitly tagged checkpoints are admitted. Running in
    a guarded child is mandatory; importing this module does not load MLX.
    
```

## pseudo_streaming.py :: llm_transcriber

```
Adapt an ``LlmAsrTranscriber`` into a ``samples -> text`` callable.
```

## pseudo_streaming.py :: classic_transcriber

```
Build a ``samples -> text`` callable for classic (non-LLM) families.

    Covers sensevoice / paraformer (greedy CTC/CIF). ``cmvn`` is REQUIRED
    for sensevoice-family checkpoints (``am.mvn``; found next to the source
    configuration) -- without it the fbank distribution is shifted and the
    encoder collapses to all-blanks (verified: argmax == 0 everywhere).
    
```

## pseudo_streaming.py :: StreamEvent

```
One pseudo-stream block result.

    ``text`` is the full current best-effort transcription (stable prefix +
    rolling tail). ``is_final`` marks the closing event, whose text equals
    offline transcription exactly.
    
```

## _layout_state.py :: convert_tensor

```
Apply the transpose for ``rule``; ``names`` is the family's identity set.
```

## _strict_load.py :: load_state_dict_strict

```
Strictly load ``state`` into ``model``; raise on any mismatch.
```

## _strict_load.py :: convert_checkpoint_tensors

```
Convert a FunASR/PyTorch ``state_dict`` into MLX layout.

    ``rules`` maps key suffixes to transforms; each family's depthwise-conv
    transpose lives there. Everything else is copied verbatim.
    
```

## _strict_load.py :: flat_parameter_names

```
Flatten ``module.parameters()`` into FunASR-style dotted names.
```

## campplus.py :: forward

```
``x`` is fbank ``(B, T, feat_dim)``.

        Literal port of the reference ``forward``: the batch is assumed to be
        already padded (or to contain a single utterance), because the reference
        pools without any padding mask.  See :meth:`embed` for the length-aware
        variant.
        
```

## campplus.py :: frame_features

```
Run everything up to (but excluding) pooling; returns ``(B, T, C)``.

        The stage order is taken from the ``xvector`` container's own insertion
        order, which mirrors the reference's ``nn.Sequential``: ``tdnn``, then
        ``block1, transit1, block2, transit2, ...``, then ``out_nonlinear``.
        Iterating separate block/transit name lists instead would apply **all**
        blocks before **any** transit, which both misorders the network and
        desynchronises the channel arithmetic (each transit halves the channel
        count feeding the next block).
        
```

## campplus.py :: load_checkpoint

```
Strict load; see :func:`load_strict`.
```

## campplus.py :: state_dict

```
Flat ``{pytorch_name: mlx_array}`` plus per-key layout rules.
```

## campplus.py :: CAMPPlus

```
``funasr.models.campplus.model.CAMPPlus`` (inference only).

    Input is 80-dim Kaldi fbank **without** the LFR stack and **without** CMVN;
    ``funasr/models/campplus/utils.py::extract_feature`` mean-normalises each
    utterance across all frames and zero-pads the batch.  Both steps stay
    outside this module (frontend scope).

    ``output_level="segment"`` returns ``(B, embedding_size)``;
    ``output_level="frame"`` returns ``(B, T, channels)``.
    
```

## campplus.py :: campplus_config_from_dict

```
Build a :class:`CAMPlusConfig` from a FunASR ``model.yaml`` mapping.
```

## campplus.py :: CAMPlusConfig

```
Mirror of FunASR's ``model_conf`` for ``CAMPPlus``.

    Defaults are the published ``speech_campplus_sv_zh-cn_16k-common`` values
    taken from ``funasr/models/campplus/template.yaml``.
    
```

## campplus.py :: DenseLayer

```
``components.DenseLayer``.

    The reference special-cases 2-D input by squeezing through a length-1 axis so
    that a 1x1 ``Conv1d`` can be used as a plain ``Linear``; that branch is
    reproduced (channels-last, so the squeezed axis is the time axis).
    
```

## campplus.py :: TransitLayer

```
``components.TransitLayer`` (nonlinearity **before** the linear map).
```

## campplus.py :: CAMDenseTDNNLayer

```
``components.CAMDenseTDNNLayer``.

    ``memory_efficient`` only gates ``torch.utils.checkpoint`` during *training*
    and is therefore inert here (documented, not silently dropped).
    
```

## campplus.py :: CAMLayer

```
``components.CAMLayer`` (channels attention over a dilated TDNN).

    ``context`` mixes the *global* time mean with a non-overlapping segment mean
    (``seg_len=100``, average pooling with ``ceil_mode=True``), squeezes it
    through ``linear1``/``relu``/``linear2``/``sigmoid``, and rescales the local
    branch.  Because the channel axis is last, the time axis is axis 1.
    
```

## campplus.py :: TDNNLayer

```
``components.TDNNLayer``.

    ``padding < 0`` selects the reference's same-padding rule
    ``padding = (kernel_size - 1) // 2 * dilation`` and requires an odd kernel.
    
```

## campplus.py :: __call__

```
``x`` is ``(B, F, T)`` (mel-major, as in the reference) -> ``(B, T, C*F/8)``.
```

## campplus.py :: BasicResBlock

```
``components.BasicResBlock`` (channels-last NHWC).

    The torch shortcut is an ``nn.Sequential`` (empty => identity), which puts
    its parameters at ``shortcut.0.weight`` / ``shortcut.1.weight``; a
    :class:`ModuleStack` (empty in the identity case) reproduces those names
    exactly.
    
```

## campplus.py :: statistics_pooling

```
Time-axis mean/std pooling of an ``(B, L, C)`` tensor -> ``(B, 1, 2C)``.

    The reference's ``statistics_pooling(x, dim=-1)`` reduces the *time* axis of
    an NCL tensor and concatenates along the channel axis, giving ``(B, 2C)``.
    Reducing axis 1 of the channels-last tensor and concatenating on the last
    axis gives ``(B, 1, 2C)`` with the same numbers in the same order, which is
    what the following ``DenseLayer`` consumes.  ``eps`` is accepted for
    signature parity; the reference never uses it.
    
```

## campplus.py :: get_nonlinear

```
Retain the reference Sequential's named children and per-channel PReLU.
```

## campplus.py :: BatchNorm2d

```
``torch.nn.BatchNorm2d`` (channels-last input ``(B, H, W, C)``).
```

## campplus.py :: BatchNorm1d

```
``torch.nn.BatchNorm1d`` (channels-last input ``(B, L, C)``).
```

## campplus.py :: kaiming_normal_

```
In-place-style ``torch.nn.init.kaiming_normal_`` for MLX conv/linear weights.

    The reference re-initialises every ``Conv1d``/``Linear`` weight with
    ``kaiming_normal_`` and zeroes the biases *after* building ``CAMPPlus``.
    ``fan_in`` is the number of input elements per output element, which is the
    product of every axis except the output axis in both layouts.
    
```

## campplus.py :: LoadReport

```
Outcome of a :func:`load_strict` call.

    ``dropped`` lists the torch-only buffers that were intentionally ignored.
    It is never empty silently: callers can assert on it.
    
```

## campplus.py :: CheckpointKeyError

```
Raised by :func:`load_strict` when a checkpoint does not match a module.
```

## campplus.py :: flatten_state_dict_with_rules

```
Like :func:`flatten_state_dict`, but also returns the per-key layout rule.

    Returns ``(tensors, rules)`` with the same two dictionaries that the
    :class:`LayoutReport` carries -- nothing is dropped.  This is the contract
    :func:`funasr_mlx.models.sanm.flatten_state_dict_with_rules` exposes, so the
    three model modules agree on one name for the same operation; callers that
    want the richer object should use :func:`flatten_state_dict` instead.
    
```

## campplus.py :: flatten_state_dict

```
Flatten a module tree into FunASR/PyTorch-compatible parameter names.

    Unlike a name-pattern heuristic, the layout rule is derived from the *owning
    module type*, so it is correct for any naming the architecture happens to
    use.  Attributes starting with ``_`` are configuration, never weights.
    
```

## campplus.py :: LayoutReport

```
Flat parameter tensors plus the layout rule that applies to each key.

    Behaves like a read-only ``Mapping`` of ``{parameter_name: mlx_array}``, so it
    can be iterated, indexed, ``len()``-ed and fed to ``dict(...)`` exactly like a
    plain state dict, while also carrying the per-key rule used for the
    torch -> MLX layout conversion.
    
```

## campplus.py :: NamedLayerList

```
Named children iterated in insertion order.

    Mirrors ``torch.nn.ModuleList`` with explicit names (``CAMDenseTDNNBlock``
    uses ``tdnnd1``, ``tdnnd2``, ...).
    
```

## campplus.py :: NamedModuleDict

```
Named children applied in order; keeps the ``<parent>.<name>.<leaf>`` keys.

    Used for ``CAMPPlus.xvector``, which the reference builds as an
    ``nn.Sequential`` of named sub-modules.
    
```

## campplus.py :: ModuleStack

```
Ordered children registered under their numeric index.

    ``torch.nn.Sequential`` yields ``<parent>.<i>.<leaf>`` parameter names; MLX's
    ``nn.Sequential`` would insert an extra ``layers`` level, so children are
    registered directly under ``"0"``, ``"1"``, ...
    
```

## ct_punc.py :: reset

```
No-op: ``ct-punc`` carries no state between calls.

        Provided for API symmetry with the streaming VAD; the mini-sentence cache
        is local to :meth:`punctuate`.
        
```

## ct_punc.py :: CtPuncPunctuator

```
End-to-end ``ct-punc``: tokens -> model -> punctuated text + ``punc_array``.

    The mini-sentence chunking / look-back cache behaviour of
    ``CTTransformer.inference`` is reproduced exactly, including the case where a
    chunk contains no sentence terminator: nothing is emitted for that chunk, the
    whole chunk is carried into the next one, and once it exceeds
    ``cache_pop_trigger_limit`` tokens the last comma is promoted to a sentence
    end.
    
```

## ct_punc.py :: expand_word_level_punc

```
FunASR's jieba word-level ``punc_array`` expansion.

    Mirrors the reverse walk in ``CTTransformer.inference`` for CJK word tokens
    longer than one character: insert ``len(token) - 1`` entries of ``no_punc_id``
    so the array lines up with the character-level token stream.

    The reference hard-codes the inserted value as the literal ``1``; that is kept
    as the default so behaviour is bit-identical, but note that ``1`` only means
    "no punctuation" when ``punc_list[1] == "_"``.  Pass ``no_punc_id=0`` (or
    whatever the target config uses) when the label table is ordered differently.
    
```

## ct_punc.py :: split_to_mini_sentence

```
FunASR ``split_to_mini_sentence``: fixed-size chunks plus one remainder.
```

## ct_punc.py :: split_words

```
FunASR ``split_words`` without a jieba user dictionary.

    Splits on whitespace, then keeps maximal ASCII runs as one token and emits
    every non-ASCII character as its own token.
    
```

## ct_punc.py :: argmax_punc_ids

```
FunASR's ``y.view(-1, C).topk(1, dim=1)`` followed by ``squeeze(1)``.
```

## ct_punc.py :: CtPuncModel

```
``CTTransformer``: embedding -> SANMEncoder -> linear punc classifier.
```

## ct_punc.py :: SanmEncoder

```
``SANMEncoder`` restricted to ``input_layer="pe"``.

    Parameter paths are ``encoders0.0.…`` for the first block and ``encoders.i.…``
    for the remaining ``num_blocks - 1``, matching FunASR exactly.
    
```

## ct_punc.py :: SanmAttention

```
``MultiHeadedAttentionSANM`` (funasr/models/sanm/attention.py).

    Weight layout conversion: ``self.fsmn_block.weight`` is ``(C, 1, K)`` in
    PyTorch (depthwise ``nn.Conv1d``) and must be transposed to MLX's
    ``(C, K, 1)``.  ``linear_q_k_v`` and ``linear_out`` already use MLX's layout.
    
```

## ct_punc.py :: PositionwiseFeedForward

```
``PositionwiseFeedForward`` with the ``w_1``/``w_2`` names preserved.

    Dropout is omitted: FunASR's ``nn.Dropout`` is an identity at inference and
    this port is inference-only.
    
```

## ct_punc.py :: ModuleLayerList

```
``nn.ModuleList`` variant keyed by *string* index.

    MLX's own ``ModuleList`` nests children under ``layers``, which would shift
    every checkpoint key (``encoders.0.…`` -> ``encoders.layers.0.…``).  Storing
    the children directly under ``"0"``, ``"1"``, … keeps FunASR's
    ``nn.ModuleList`` parameter paths byte-identical.

    Note ``nn.Module.update()`` can only overwrite *existing* keys, so children
    are inserted as dict items (which is what ``Module.__setattr__`` does for
    ``dict``/``list`` values too).
    
```

## ct_punc.py :: sinusoidal_position_encoding

```
FunASR ``SinusoidalPositionEncoder`` (``input_layer="pe"``), parameter-free.

    ``log_timescale_increment = log(10000) / (depth/2 - 1)``; the sin block is
    concatenated *before* the cos block; positions start at 1.
    
```

## ct_punc.py :: make_valid_mask

```
``(~make_pad_mask(lengths)[:, None, :])`` -> ``(B, 1, maxlen)`` of 1/0 floats.
```

## ct_punc.py :: is_first_byte_ascii

```
FunASR's ``len(token[0].encode()) == 1`` test (first character is ASCII).
```

## ct_punc.py :: from_config

```
Build from a FunASR ``tokenizer_conf`` mapping (or an explicit path).
```

## ct_punc.py :: from_file

```
Load a newline-separated vocabulary (``token_list.txt``).
```

## ct_punc.py :: CharVocab

```
Character vocabulary equivalent to FunASR's ``CharTokenizer``.

    ``BaseTokenizer.encode`` simply maps every token through ``token2id`` with an
    ``<unk>`` fallback, so this class exposes exactly that behaviour and nothing
    more (no segmentation: the caller already produced tokens via
    :func:`split_words`).
    
```

## ct_punc.py :: from_dict

```
Build from a full FunASR ``config.yaml`` mapping or from a flat one.

        Accepts ``{"model_conf": {...}, "encoder_conf": {...},
        "tokenizer_conf": {...}}`` (the on-disk layout) as well as a single flat
        ``model_conf`` mapping.
        
```

## ct_punc.py :: CtPuncConfig

```
Validated mirror of ``model_conf`` for FunASR's ``CTTransformer``.
```

## ct_punc.py :: from_dict

```
Build from a FunASR ``encoder_conf`` mapping.
```

## ct_punc.py :: SanmEncoderConfig

```
Validated mirror of ``encoder_conf`` for FunASR's ``SANMEncoder``.

    Only the ``ct-punc`` configuration is supported: ``input_layer="pe"`` (a
    parameter-free sinusoidal position encoder) and
    ``selfattention_layer_type="sanm"``.
    
```

## emotion2vec.py :: load_state_dict_strict

```
Load ``state`` into ``module``; return ``(discarded, unexpected)``.

    Never silently skips a key: anything outside
    :data:`INFERENCE_ONLY_PREFIXES` must match a parameter by name *and* shape.
    
```

## emotion2vec.py :: CheckpointKeyError

```
Raised when a checkpoint does not match the model exactly.
```

## emotion2vec.py :: convert_checkpoint_tensors

```
Convert a published ``state_dict`` and report what was discarded.

    Only ``Conv1d`` weights change layout: ``(out, in//g, K) -> (out, K, in//g)``.
    Linears, LayerNorms, ``extra_tokens`` and ``alibi_scale`` already match MLX.
    
```

## emotion2vec.py :: is_conv_weight

```
True for the depthwise ``Conv1d`` weights listed in :data:`CONV_WEIGHT_PATTERNS`.
```

## emotion2vec.py :: emotion_scores

```
Full FunASR-equivalent result: labels + softmax scores + features.

        Runs the encoder exactly once; ``feats`` is the utterance-level pooled
        embedding the classifier consumed, which is the same
        ``x.mean(dim=1)`` the reference uses, so no second pass is needed.
        
```

## emotion2vec.py :: classify

```
Return ``(features (B, embed_dim), logits (B, vocab_size), frames)``.

        The head is ``nn.Linear(embed_dim, vocab_size)`` applied to the mean over
        time **after** the extra tokens have been removed
        (``Emotion2vec.inference``: ``x = x.mean(dim=1); x = self.proj(x)``).  The
        frame features are returned so a caller that also wants them does not have
        to run the whole encoder a second time.
        
```

## emotion2vec.py :: extract_features

```
Waveform ``(B, samples)`` -> frame features ``(B, T', embed_dim)``.

        ``remove_extra_tokens=True`` strips the ``num_extra_tokens`` cls tokens,
        which is what FunASR does before the utterance-level pooling.  The encoder
        is evaluated exactly **once**.
        
```

## emotion2vec.py :: Emotion2VecModel

```
``Emotion2vec`` (FunASR wrapper) as native MLX modules.

    The checkpoint is a plain ``state_dict`` of the ``Emotion2vec`` module, so the
    parameter prefixes are ``modality_encoders.AUDIO.*``, ``blocks.N.*`` and
    ``proj.*``.  ``self.modality_encoders`` is a plain dict so the keys come out
    as ``modality_encoders.AUDIO.<...>`` exactly like the reference
    ``nn.ModuleDict``.
    
```

## emotion2vec.py :: preprocess_waveform

```
``source = F.layer_norm(source, source.shape)`` then ``view(1, -1)``.

    The reference normalises over the *entire* 1-D waveform (one global
    normalisation), not per frame.
    
```

## emotion2vec.py :: EmotionResult

```
Mirrors the keys FunASR's ``Emotion2vec.inference`` returns.
```

## emotion2vec.py :: contextualized_features

```
``contextualized_features(..., mask=False, remove_masked=False)``.

        Returns ``(x, alibi_bias, alibi_scale_for_model_blocks)``.  When the
        learned scale has a single row it is folded into ``alibi_bias`` here and
        the third element is ``None``; with one row **per layer** the rows stay
        live and the tail (``alibi_scale[prenet_depth:]``) is handed back so the
        shared blocks pick their own row, exactly like the reference.
        
```

## emotion2vec.py :: alibi_bias

```
``get_alibi_bias`` -> ``(B, heads, T, T)``.
```

## emotion2vec.py :: local_features

```
Waveform ``(B, samples)`` -> projected features ``(B, T, embed_dim)``.

        The conv stack returns ``NCT`` and ``project_features`` is the source's
        ``Sequential(TransposeLast, LayerNorm(512), Linear)``.  Its leading
        ``TransposeLast`` is a **real** transpose, so the ``NCT`` tensor is fed
        straight in and the norm correctly reduces the **512 channels**.
        
```

## emotion2vec.py :: _weight_dtype

```
Dtype of this encoder's checkpoint weights (first conv always exists).
```

## emotion2vec.py :: ffn_targets

```
``ffn_targets = not end_of_block_targets`` (``emotion2vec.py``).
```

## emotion2vec.py :: pos_kernel_size

```
``k = max(3, conv_pos_width // conv_pos_depth)`` (``audio.py:87``).
```

## emotion2vec.py :: Emotion2VecConfig

```
Validated mirror of the audio modality config plus the model fields.
```

## emotion2vec.py :: __call__

```
``modules.py:227-244`` with every dropout/drop-path removed.

        Note the post-norm branch feeds the **raw** ``x`` to the attention and
        only then applies ``norm1``; ``t`` is the pre-``norm2`` mlp output when
        ``ffn_targets`` and the final ``x`` otherwise.  ``t`` is training-only
        (it supervises the teacher) and is returned for API parity only.
        
```

## emotion2vec.py :: AltBlock

```
``modules.py:180-245`` ``AltBlock`` with ``layer_norm_first=false``.

    Post-norm ordering::

        x = x + attn(norm1(x))
        x = x + mlp(norm2(x))

    and the second return value is the post-norm ``x`` when ``ffn_targets`` else
    the pre-norm mlp output.  That value is training-only (it supervises the
    teacher), so the port returns it for API parity but never consumes it.
    
```

## emotion2vec.py :: AltAttention

```
``modules.py:255-300`` ``AltAttention`` (cosine attention is not used).
```

## emotion2vec.py :: __call__

```
``NTC`` in, ``NTC`` out -- the untouched source chain.

        The leading ``TransposeLast`` converts ``NTC -> NCT`` for the blocks (each
        ``NCT -> NCT``) and the trailing one restores ``NTC``.
        
```

## emotion2vec.py :: PositionalEncoder

```
``Sequential(TransposeLast, 5 x PositionalBlock, TransposeLast)``.

    With ``conv_pos_pre_ln=false`` (the large config) the conv's
    ``padding = k // 2 = 9`` preserves ``T`` and ``SamePad(19)`` is the identity
    on top of it, so the output length is always exactly ``T``.

    Time axis contract: the encoder takes ``NTC`` and returns ``NTC``; each
    :class:`PositionalBlock` takes ``NTC`` and returns ``NCT``, so the encoder
    restores ``NTC`` for the final output.  Each :class:`PositionalBlock` is
    ``NCT`` in and ``NCT`` out, so nothing is transposed *between* blocks.
    
```

## emotion2vec.py :: ConvFeatureExtraction

```
``ConvFeatureExtractionModel`` with ``mode="layer_norm"``.
```

## emotion2vec.py :: ConvFeatureBlock

```
``wav2vec2.py:839-860`` ``block(...)`` for ``is_layer_norm=True``.

    ``Conv1d -> Dropout -> Sequential(TransposeLast, Fp32LayerNorm(affine=True),
    TransposeLast) -> GELU``.  The double transpose turns the ``NCT`` conv
    output into ``NTC`` so the LayerNorm normalises the **512 channels**, and the
    norm is **affine**, i.e. its ``weight``/``bias`` are real checkpoint tensors.
    
```

## emotion2vec.py :: parse_feature_encoder_spec

```
Parse ``feature_encoder_spec`` with ``ast`` -- ``eval`` is never used.

    Accepts only literal arithmetic over tuples of ints, i.e. ``+``, ``*`` and
    ``[a, b, c]``.  Anything else (names, calls, attributes, comprehensions)
    raises.
    
```

## emotion2vec.py :: Mlp

```
``fairseq`` ``Mlp``: ``fc1 -> act -> fc2`` with dropout around.
```

## emotion2vec.py :: Gelu

```
``nn.GELU()`` with the default (exact, erf-based) approximation.
```

## emotion2vec.py :: Fp32LayerNorm

```
``fairseq.layers.fp32_group_norm.Fp32LayerNorm``.

    A ``nn.LayerNorm`` that upcasts to float32, normalises and casts back, so
    the reduction never happens in half precision.
    
```

## emotion2vec.py :: TransposeLast

```
``fairseq.modules.TransposeLast``: swap the last two axes.
```

## emotion2vec.py :: __getitem__

```
Fetch by position (``self[0]``) or by the mapping key (``self["0"]``).

        ``nn.Module.update`` / ``tree_unflatten`` hand back the *string* keys that
        were registered here, so a positional-only ``index < 0`` comparison raises
        ``TypeError: '<' not supported between instances of 'str' and 'int'``.
        
```

## emotion2vec.py :: ModuleListStack

```
``nn.ModuleList``; children registered under their decimal index.
```

## emotion2vec.py :: SequentialStack

```
``nn.Sequential`` with automatic ``0..n-1`` indices.
```

## emotion2vec.py :: __getitem__

```
Fetch by Sequential position (``self[0]``) **or** by raw key (``self["0"]``).

        Both spellings are used: ``__call__`` walks ``_order`` positionally while
        :class:`PositionalBlock` needs named access so it can reorder the
        transposes around the channel-last conv.
        
```

## emotion2vec.py :: OrderedModules

```
An ``nn.Sequential`` whose children keep their integer index in the key.

    ``nn.Module`` is a dict subclass and ``tree_flatten`` walks ``.items()``, so
    registering a child under the *string* ``"2"`` reproduces PyTorch's
    ``...2.1.weight`` naming exactly.  A plain list would also be captured by
    ``Module.__setattr__`` and would pollute the key space, hence the tuple.
    
```

## emotion2vec.py :: get_alibi

```
``get_alibi(max_positions, attention_heads, dims=1)``.

    1-D manhattan ALiBi: ``slopes[h] * -|i - j|`` with shape
    ``(heads, max_positions, max_positions)``.
    
```

## fsmn_vad.py :: detect

```
Convenience wrapper for the offline case (auto-initialised cache).
```

## fsmn_vad.py :: __call__

```
``FsmnVADStreaming.forward``.

        ``feats`` is ``(1, T, input_dim)`` fbank (the frontend is owned by the
        shared runtime); ``waveform`` is the exact ``(1, N)`` (or ``(N,)``)
        sample span those ``T`` frames cover.
        
```

## fsmn_vad.py :: drop_cached_frames

```
``DropCachedFrames``: release decibel/score history before a frame.
```

## fsmn_vad.py :: reset

```
Full reset between utterances (``init_cache``).
```

## fsmn_vad.py :: init_cache

```
``init_cache``: fresh encoder caches, window detector and stats.
```

## fsmn_vad.py :: from_dict

```
Build from a FunASR ``config.yaml`` mapping.
```

## fsmn_vad.py :: FsmnVadModel

```
FSMN-VAD: streaming/offline speech endpointing.

    ``__call__(feats, waveform, cache, is_final=…, is_streaming_input=…)`` mirrors
    ``FsmnVADStreaming.forward`` and returns ``[[[start_ms, end_ms], …], …]``
    (batch x segment; batch size 1 is the only supported case, as in FunASR).
    
```

## fsmn_vad.py :: forward_with_caches

```
ONNX-export style: return ``(scores, [cache_layer_0, …])``.
```

## fsmn_vad.py :: FSMN

```
``FSMN`` encoder (``funasr/models/fsmn_vad_streaming/encoder.py``).

    ``cache=None`` runs the non-streaming path (zero left padding); passing a dict
    runs the streaming path and updates ``cache["cache_layer_<i>"]`` in place,
    exactly like the reference.
    
```

## fsmn_vad.py :: BasicBlock

```
``BasicBlock``: linear -> FSMN -> affine -> relu, with a per-layer cache.
```

## fsmn_vad.py :: __call__

```
Return ``(output, new_cache)`` with ``output`` shaped ``(B, T, D)``.

        Layout: ``mx.nn.Conv1d`` is channels-last ``(batch, length, channels)``.
        The reference works in ``(B, D, T)``; the time axis is moved to the middle
        so the ``(lorder - 1) * lstride`` left padding (or cache concatenation) is
        expressed on the same axis FunASR pads, then moved back for the conv.

        ``mx.core.conv1d`` reads ``in[t * stride + k * dilation]``; with
        ``y[i] = x[i - keep]`` kernel tap ``k`` therefore reads ``x[t - keep + k]``.
        
```

## fsmn_vad.py :: FSMNBlock

```
``FSMNBlock``: left/right dilated depthwise memory block.

    The reference applies ``nn.Conv2d`` to a ``(B, D, T, 1)`` view; here the time
    axis is the sequence axis of ``mx.conv1d`` and the trailing singleton axis is
    dropped (it carries no data).

    Cache shapes: ``cache`` is ``(B, D, (lorder - 1) * lstride)`` or ``None``.
    
```

## fsmn_vad.py :: AffineTransform

```
``AffineTransform``: ``self.linear = nn.Linear(input_dim, output_dim)``.

    Like :class:`LinearTransform`, this retains the named ``linear`` submodule;
    unlike it, the affine transform also has a bias.
    
```

## fsmn_vad.py :: LinearTransform

```
Bias-free wrapper retaining the reference's named ``linear`` child.

    ``BasicBlock.linear`` owns this wrapper, so its child adds the second
    ``linear`` in ``fsmn.<i>.linear.linear.weight``.
    
```

## fsmn_vad.py :: ModuleLayerList

```
``nn.ModuleList`` keyed by *string* index (see ``ct_punc.py`` for why).

    MLX's own ``ModuleList`` nests children under ``layers`` and would shift every
    checkpoint key; FunASR's ``FsmnStack`` (an ``nn.Sequential``) indexes blocks
    as ``fsmn.0.…``, ``fsmn.1.…``, so children must sit directly under ``"0"``,
    ``"1"``, …  ``nn.Module.update()`` can only overwrite existing keys, so
    children are inserted as dict items.
    
```

## fsmn_vad.py :: max_end_sil_frame_cnt_thresh

```
``Stats.max_end_sil_frame_cnt_thresh`` at cache-initialisation time.
```

## fsmn_vad.py :: VadOptions

```
Mutable mirror of FunASR's ``VADXOptions``.

    Mutable because ``ComputeScores`` overwrites ``nn_eval_block_size`` with the
    number of frames in the current block, exactly like the reference.
    
```

## fsmn_vad.py :: cache_frames

```
Left-context cache length ``(lorder - 1) * lstride``.
```

## fsmn_vad.py :: FsmnEncoderConfig

```
Validated mirror of ``encoder_conf`` for FunASR's ``FSMN`` network.
```

## fsmn_vad.py :: convert_checkpoint_tensors

```
Depthwise 4-D kernels transposed; strict elsewhere (see _strict_load).
```

## fun_asr_nano.py :: nano_config_from_nano_yaml

```
Build the config from the official Fun-ASR-Nano ``config.yaml``.

    Reads ``audio_encoder_conf`` / ``audio_adaptor_conf`` and pins the
    Qwen3-0.6B ``llm_conf.args`` (the architecture of the shipped ``llm.*``
    subtree, verified against the checkpoint).
    
```

## fun_asr_nano.py :: embed_with_speech

```
Token embeddings with the speech span spliced in, then one LLM pass.
```

## fun_asr_nano.py :: encode_audio

```
``(B, T, 560)`` LFR fbank -> adaptor embeddings ``(B, T', llm_dim)``.
```

## fun_asr_nano.py :: FunASRNano

```
``FunASRNano`` (inference): encode -> adapt -> LLM decode.
```

## fun_asr_nano.py :: StandardMHA

```
ESPnet ``MultiHeadedAttention`` (transformer/attention.py): separate
    ``linear_q/k/v`` + ``linear_out`` and NO fsmn memory -- the adaptor
    checkpoint keys (``blocks.N.self_attn.linear_q...``) are exactly this
    module, distinct from the SANM attention of the speech encoders.
```

## fun_asr_nano.py :: TransformerAdaptor

```
``funasr.models.llm_asr.adaptor.Transformer`` (inference).

    Stacks ``k`` frames (k=1 here), projects encoder_dim -> llm_dim through
    an FFN, then runs standard encoder layers. The reference layers are
    ESPnet-style ``EncoderLayer`` (post-norm when normalize_before=False,
    which is the adaptor default: EncoderLayer's default normalize_before
    is False) with ``MultiHeadedAttention`` and ``PositionwiseFeedForward``.
    
```

## glm_asr.py :: __call__

```
mel: ``(F, T)``; returns ``(1, T'', output_size)`` embeddings.

        The official projector consumes the encoder output RE-GROUPED into
        ``intermediate_size``-wide rows: every 4 frames of 1280 become one
        5120-wide row (a 4-frame merge), then linear_1 5120->4096 + gelu +
        linear_2 4096->2048.
        
```

## glm_asr.py :: GlmAsrAudioEncoder

```
``glmasr_encoder`` (inference): mel (F, T) -> (1, T', output_size).
```

## glm_asr_thinker.py :: encode_audio

```
mel (F, T) -> (1, N, text_hidden) embeddings for the pad run.

        The mel is cast to the tower's weight dtype, as whisper's AudioEncoder
        and qwen3-asr's tower do.  This cast ALONE measures 1.00x here (see
        .tmp/task4/tower-tight-*.json and the re-measurement in
        .tmp/t7/glm-isolate.json), which is why it was originally skipped -- but
        that A/B was misleading: the tower's partial-RoPE tables were float32,
        so attention re-promoted the stream to float32 regardless.  With the
        tables also following the activation dtype (see ``_Attn._rope``) the
        two casts together give 1.44x on the tower and 28.7% on the pipeline.
        
```

## glm_asr_thinker.py :: _tower_dtype

```
Dtype of the audio tower's checkpoint weights (first conv).
```

## glm_asr_thinker.py :: _decoder_dtype

```
Dtype the decoder activations must follow.

        Read through :func:`activation_dtype`, not ``weight.dtype``: a quantized
        embedding stores packed ``uint32`` codes in ``weight``, and casting the
        spliced audio embeddings to ``uint32`` destroys them.
        
```

## glm_asr_thinker.py :: GlmAsrThinker

```
audio_tower (with projector folded in) + Llama causal LM.
```

## moss_transcribe.py :: forward_with_audio

```
Replace the audio-token run with adaptor output (shared splice).
```

## moss_transcribe.py :: _decoder_dtype

```
Dtype the decoder activations must follow.

        Read through :func:`activation_dtype`, not ``weight.dtype``: a quantized
        embedding stores packed ``uint32`` codes in ``weight``, and casting the
        spliced audio embeddings to ``uint32`` destroys them.
        
```

## moss_transcribe.py :: encode_audio_from_hidden

```
Merged/chunked whisper output (1, T, 1024) -> adaptor (1, T/4, hidden).

        Trims to a multiple of the merge size first (official time_merge).
        
```

## moss_transcribe.py :: encode_audio

```
mel (F, T) -> whisper (1, T/2, 1024) -> adaptor (1, T/8, hidden).
```

## moss_transcribe.py :: MossTranscribeDiarize

```
whisper_encoder + vq_adaptor + qwen3 (audio-token splice).
```

## moss_transcribe.py :: VQAdaptor

```
Linear(input_dim->hidden) + SiLU + Linear(hidden->hidden) + LN(1e-6).
```

## moss_transcribe.py :: HfWhisperEncoder

```
HF ``WhisperEncoder`` (inference): mel (F, T) -> (1, T/2, d).
```

## moss_transcribe.py :: _HfWhisperAttn

```
HF ``WhisperAttention``: q/v/out carry biases, **k does not**.

    ``k_proj`` is bias-free in the reference (``nn.Linear(embed_dim,
    embed_dim, bias=False)`` in transformers' ``modeling_whisper.py``), and the
    checkpoint agrees: ``...self_attn.k_proj.bias`` is absent from all 24
    layers.  Building it with ``bias=True`` therefore left a random,
    never-trained key bias in the audio tower -- measured on the previous
    release: exists=True, absmax 0.0312, absmean 0.0158.  The old permissive
    loaders never noticed, which is why MOSS tower parity sat at 1.3e-3.
    
```

## paraformer.py :: _cif_fire_positions

```
Fallback used when the CIF peaks do not line up with the token count.

    Mirrors ``funasr.models.paraformer.cif_predictor.cif_wo_hidden``: a frame
    "fires" when the running integral crosses an integer boundary.
    
```

## paraformer.py :: ts_prediction_lfr6_standard

```
Port of ``funasr.utils.timestamp_tools.ts_prediction_lfr6_standard``.

    Only the millisecond ``res`` output of the reference is produced here (the
    text dump and the ``<sil>`` bookkeeping are dropped).  ``upsample_rate`` must
    be ``1`` to match Paraformer's inference call site.
    
```

## paraformer.py :: timestamps

```
CIF-peak based character timestamps in milliseconds.

        Returns ``(tokens, timestamps)`` where ``timestamps[i]`` is a list of
        ``[start_ms, end_ms]`` pairs aligned with ``tokens[i]``.
        
```

## paraformer.py :: greedy_decode

```
Greedy token ids per utterance.

        Mirrors ``Paraformer.inference`` without beam search: ``argmax`` over the
        decoder posteriors followed by dropping ``sos``/``eos``/``blank_id``
        (the reference wraps the sequence in sos/eos and then strips it).
        
```

## paraformer.py :: load_checkpoint

```
Strict load; see :func:`funasr_mlx.models.sanm.load_strict`.
```

## paraformer.py :: state_dict

```
``{pytorch_name: mlx_array}`` in MLX layout.
```

## paraformer.py :: Paraformer

```
``funasr.models.paraformer.model.Paraformer`` (inference only).
```

## paraformer.py :: __call__

```
``hs_pad`` is the encoder/acoustic memory ``(B, T1, D)``.

        Following the reference (``funasr/models/paraformer/decoder.py:397-450``)
        the offline path does **not** run ``self.embed``: ``x = tgt`` with
        ``tgt = ys_in_pad``.  Only ``forward_one_step`` (``:604``) embeds.  In
        offline Paraformer the decoder input is therefore the CIF acoustic
        embedding, already ``(B, T2, D)`` -- not token ids.
        
```

## paraformer.py :: ParaformerSANMDecoder

```
``funasr.models.paraformer.decoder.ParaformerSANMDecoder`` (offline path).

    Three stacks: ``decoders`` (``att_layer_num`` layers with SANM self-attention
    *and* cross-attention), optional ``decoders2`` (self-attention only) and a
    single ``decoders3`` feed-forward-only layer.
    
```

## paraformer.py :: __call__

```
Return ``(acoustic_embeds, token_num, alphas, cif_peak)``.
```

## paraformer.py :: CifPredictorV2

```
``funasr.models.paraformer.cif_predictor.CifPredictorV2`` (inference path).

    Parameters: ``cif_conv1d`` (a **dense**, i.e. ``groups=1``, ``Conv1d`` in V2)
    and ``cif_output``.  ``__call__`` returns the 4-tuple
    ``(acoustic_embeds, token_num, alphas, cif_peak)`` of the reference.
    
```

## paraformer.py :: _gather

```
Gather time steps of ``array`` in fire order, independently per row.

        ``mx.take_along_axis`` indexes the time axis directly, so no flatten,
        offset arithmetic or output reshape is involved.  Per the installed stub
        (``mlx/core/__init__.pyi:1901-1909``) ``indices`` must be "broadcastable
        with the input array excluding the ``axis`` dimension", so a trailing
        singleton axis broadcasts one time index across every feature lane and
        the result keeps ``(B, T, D)``.
        
```

## paraformer.py :: SANMEncoder

```
``funasr.models.sanm.encoder.SANMEncoder`` (offline inference path).

    Differences from the reference that do not affect checkpoint parity:
    intermediate CTC / ``interctc_layer_idx`` outputs and streaming
    ``forward_chunk`` are not ported (both are training / streaming features).
    
```

## paraformer.py :: CTC

```
``funasr.models.ctc.ctc.CTC`` inference path (``ctc_lo`` projection).
```

## paraformer.py :: paraformer_config_from_dict

```
Build a :class:`ParaformerConfig` from a FunASR ``model.yaml`` mapping.
```

## paraformer.py :: ParaformerConfig

```
Mirror of FunASR's ``model_conf`` for ``Paraformer``.

    Defaults follow the published ``paraformer-zh`` configuration
    (``input_size=560`` LFR fbank, 50 SANM blocks of width 512, 16 decoder
    blocks, CIF ``tail_threshold=0.45``).
    
```

## paraformer.py :: CTCConfig

```
Mirror of FunASR's ``ctc_conf`` (inference-relevant fields only).
```

## paraformer.py :: CifPredictorConfig

```
Mirror of FunASR's ``predictor_conf`` for ``CifPredictorV2``.
```

## paraformer.py :: ParaformerDecoderConfig

```
Mirror of FunASR's ``decoder_conf`` for ``ParaformerSANMDecoder``.
```

## paraformer.py :: SANMEncoderConfig

```
Mirror of FunASR's ``encoder_conf`` for ``SANMEncoder`` (Paraformer side).
```

## paraformer_streaming.py :: streaming_config_from_dict

```
Build the streaming config from the official online yaml mapping.
```

## paraformer_streaming.py :: forward_chunk

```
One chunk end-to-end. Returns ``(am_scores, token_length)``.

        ``None`` when the chunk fired no tokens (mirrors ``generate_chunk``
        returning ``[]`` when ``max(pre_token_length) < 1``).
        
```

## paraformer_streaming.py :: decoder_chunk

```
``cal_decoder_with_predictor_chunk`` -> decoder ``forward_chunk``.
```

## paraformer_streaming.py :: predictor_chunk

```
``calc_predictor_chunk`` -> ``CifPredictorV2.forward_chunk``.
```

## paraformer_streaming.py :: init_cache

```
``ParaformerStreaming.init_cache`` (model.py:511-552), MLX arrays.
```

## paraformer_streaming.py :: ParaformerStreaming

```
``ParaformerStreaming`` -- chunk-wise streaming Paraformer (MLX).

    Reuses the offline module tree (identical parameter names); adds
    ``init_cache`` / ``forward_chunk`` implementing the reference streaming
    protocol. ``generate`` feeds fixed 600 ms chunks ([0,10,5] x 960 samples).
    
```

## paraformer_streaming.py :: _cross_attn_forward_chunk

```
``MultiHeadedAttentionCrossAtt.forward_chunk`` (attention.py:815+ for cross).

    Cross-attention to the (growing) encoder memory with the same tail-window
    cache discipline as the decoder self-attention.
    
```

## paraformer_streaming.py :: _predictor_forward_chunk

```
``CifPredictorV2.forward_chunk`` (``cif_predictor.py:316-412``).

    Returns ``(acoustic_embeds, token_length, alphas, fires)`` with the carry
    state written back into ``cache``.
    
```

## paraformer_streaming.py :: _attn_forward_chunk

```
``MultiHeadedAttentionSANM.forward_chunk`` on the shared offline class.

    Mirrors ``sanm/attention.py:329-366``: compute qkv for the (stitched)
    chunk, merge cached ``k``/``v`` (look-back), commit each chunk's
    right-context-free stride into the cache, attend, add the FSMN memory.
    
```

## paraformer_streaming.py :: StreamSinusoidalPositionEncoder

```
``transformer/embedding.py::StreamSinusoidalPositionEncoder``.

    Stateful through ``cache['start_idx']``: each chunk's positions continue
    from where the previous chunk ended, which is what makes the streamed
    encoding equal the offline one.
    
```

## qwen3_asr.py :: Qwen3ASRAudioEncoder

```
``Qwen3ASRAudioEncoder`` (inference), single-utterance path.

    Input mel: ``(F, T)`` (n_mels first, like the reference). Output:
    ``(1, T', output_dim)`` -- one embedding row per audio token slot,
    which the thinker splices between ``audio_start`` and ``audio_end``.
    
```

## qwen3_asr.py :: _Conv2dStride2

```
``nn.Conv2d(h, h, 3, stride=2, padding=1)`` in MLX (NHWC).
```

## qwen3_asr_thinker.py :: encode_audio

```
``(F, T)`` mel -> ``(1, T', text.hidden_size)`` tower embeddings.

        The front-end always produces float32 mel, so the mel is cast to the
        tower's **weight** dtype first (same rule as whisper's
        ``AudioEncoder.__call__``); without it an fp16 checkpoint runs the
        conv stack and every tower Linear in mixed precision.  Interleaved
        A/B on M4 (25 rounds, .tmp/task4/tower-tight-*.json): 55.7 -> 44.0 ms
        at the real 420-frame shape and 273.9 -> 214.7 ms at 30 s, i.e. 1.27x
        at both dtypes.  For a float32 checkpoint the cast is a no-op, so
        those numerics stay bit-identical.
        
```

## qwen3_asr_thinker.py :: _tower_dtype

```
Dtype of the audio tower's checkpoint weights (first conv).
```

## qwen3_asr_thinker.py :: _decoder_dtype

```
Dtype the decoder activations must follow.

        The token embedding produces the activations the decoder consumes, so
        its OUTPUT dtype is the one to cast spliced audio to.  Read through
        :func:`activation_dtype`, not ``weight.dtype``: a quantized embedding
        stores packed ``uint32`` codes in ``weight``, and casting the audio
        embeddings to ``uint32`` destroys them (see the helper's docstring).
        
```

## sanm.py :: Conv2dSubsampling2

```
``funasr.models.transformer.utils.subsampling.Conv2dSubsampling2`` (1/2).
```

## sanm.py :: Conv2dSubsampling

```
``funasr.models.transformer.utils.subsampling.Conv2dSubsampling`` (1/4).

    Two strided 3x3 convolutions over the ``(B, T, F, 1)`` view followed by a
    linear projection.  Parameter keys stay ``conv.<i>.weight`` and
    ``out.0.weight`` thanks to :class:`ModuleStack`.

    MLX convolutions are channels-last, so the channel axis comes **last**
    (``(B, T, F, C)``).  The flatten before ``out.0`` is transposed back to
    PyTorch's ``(C, F)`` row-major order, which is what ``out.0`` was trained on.
    
```

## sanm.py :: check_short_utt

```
Port of ``funasr.models.transformer.utils.subsampling.check_short_utt``.
```

## sanm.py :: TooShortUttError

```
Port of ``funasr.models.transformer.utils.subsampling.TooShortUttError``.
```

## sanm.py :: DecoderLayerSANM

```
``funasr.models.paraformer.decoder.DecoderLayerSANM``.

    Note the ordering: the feed-forward runs **first** and the self-attention
    residual is taken against the block *input* (``residual``), not against the
    feed-forward output.  ``src_attn`` may be ``None`` (the ``decoders2`` stack).
    
```

## sanm.py :: EncoderLayerSANM

```
``funasr.models.sanm.encoder.EncoderLayerSANM`` (offline inference path).

    The first layer (``encoders0``) has ``in_size != size``; in that case the
    self-attention residual is dropped exactly like the reference does.
    
```

## sanm.py :: MultiHeadedAttentionCrossAtt

```
``funasr.models.sanm.attention.MultiHeadedAttentionCrossAtt``.
```

## sanm.py :: __call__

```
``inputs`` is ``(B, T, D)`` (MLX NLC); ``cache`` mirrors the reference.
```

## sanm.py :: MultiHeadedAttentionSANMDecoder

```
``funasr.models.sanm.attention.MultiHeadedAttentionSANMDecoder``.

    Pure FSMN memory block over the decoder input.  ``cache`` mirrors the
    reference streaming state; offline inference passes ``None``.

    Deviation: the cache is kept channels-last ``(b, L, d)`` whereas the
    reference keeps ``(b, d, L)``.  See the module docstring's "Known
    deviations" section -- this matters only for the streaming/export contract,
    not for offline inference.
    
```

## sanm.py :: _split_heads

```
``(B, T, n_feat)`` -> ``(B, head, T, d_k)``.
```

## sanm.py :: MultiHeadedAttentionSANM

```
``funasr.models.sanm.attention.MultiHeadedAttentionSANM``.

    Fused QKV projection (``linear_q_k_v``) plus a depth-wise FSMN memory block
    whose output is added to the attention output.  The reference applies
    ``d_k ** -0.5`` to the queries *after* splitting heads (equivalent to
    scaling the scores by ``1/sqrt(d_k)``).
    
```

## sanm.py :: PositionwiseFeedForwardDecoderSANM

```
``funasr.models.sanm.positionwise_feed_forward.PositionwiseFeedForwardDecoderSANM``.

    Two details matter for checkpoint compatibility: ``w_2`` has **no bias** and
    the ``norm`` sits *between* the two linears and uses ``eps=1e-12``.
    
```

## sanm.py :: PositionwiseFeedForward

```
``funasr.models.transformer.positionwise_feed_forward.PositionwiseFeedForward``.
```

## sanm.py :: SinusoidalPositionEncoder

```
``funasr.models.transformer.embedding.SinusoidalPositionEncoder``.

    Note the reference divides ``log(10000)`` by ``depth / 2 - 1`` (not by
    ``depth / 2``), i.e. positions start at ``1`` and the wavelength exponent
    denominator is ``depth / 2 - 1``.  Both quirks are preserved.
    
```

## sanm.py :: Float32LayerNorm

```
``funasr.models.sense_voice.model.LayerNorm`` (eps ``1e-5``, fp32 math).

    SenseVoice keeps its own ``LayerNorm`` that inherits the ``torch`` default
    ``eps=1e-5`` and forces the reduction to float32 before casting the result
    back to the input dtype.  Reproducing the explicit cast keeps the fp16 path
    numerically identical to the reference.
    
```

## sanm.py :: LayerNorm

```
``funasr.models.transformer.layer_norm.LayerNorm`` (eps ``1e-12``).

    Parameter names are ``weight``/``bias`` and the normalisation axis is the
    last one, exactly like the PyTorch original.  MLX performs the reduction in
    float32 internally, which matches ``torch.nn.LayerNorm`` for float32 inputs.
    
```

## sanm.py :: repeat

```
Port of ``funasr.models.transformer.utils.repeat.repeat``.
```

## sanm.py :: __call__

```
Chain blocks with canonical ``nn.Sequential`` semantics.

        The leading argument is threaded through the chain while the remaining
        positional arguments and the keywords stay fixed.  Feeding a block's
        *return value* back as ``*args`` instead would unpack an ``mx.array``
        along axis 0 and hand the next block ``B`` separate arguments, which
        only works at batch size 1 or when the arities happen to line up.
        
```

## sanm.py :: checkpoint_state_dict

```
Return the module's parameters as ``{pytorch_name: mlx_array}``.

    The names are byte-compatible with the FunASR ``.pt`` checkpoints; the
    arrays are already in MLX layout.
    
```

## sanm.py :: flatten_state_dict_with_rules

```
Like :func:`flatten_state_dict`, but also returns the per-key layout rule.
```

## sanm.py :: flatten_state_dict

```
Flatten a module tree into FunASR/PyTorch-compatible parameter names.

    Names are built from the Module's **mapping storage** (see
    :func:`_flatten_into`), which is the only place MLX can hold a tensor.  The
    result matches what ``torch.nn.Module.state_dict()`` produces for the
    corresponding PyTorch modules.
    
```

## sanm.py :: convert_checkpoint_tensor

```
Convert one PyTorch checkpoint tensor into the MLX layout (shared kernel).
```

## sanm.py :: CheckpointKeyError

```
Raised by :func:`load_strict` when a checkpoint does not match a module.

    The message always lists the offending keys so a failed load is never
    silently downgraded to a partial load.
    
```

## sanm.py :: decimate_mask

```
Port of the ``x_mask[:, :, :-2:2][:, :, :-2:2]`` subsampling masks.

    ``torch`` slices with an explicit stop index (``[:-2]`` stops at
    ``length - 2``); MLX negative-stop semantics are easy to get subtly wrong,
    so the equivalent index vector is built explicitly and applied with
    :func:`mx.take`.  ``stride=2`` models ``[:-2:2]`` and ``stride=1`` models
    ``[:-2:1]``; both stop at ``length - end_offset``.
    
```

## sanm.py :: constant_pad_1d

```
``torch.nn.ConstantPad1d`` port.

    MLX convolutions are **channels-last**: ``mx.conv1d`` consumes ``NLC`` and
    ``mx.conv2d`` consumes ``NHWC``.  The reference SANM modules therefore pad
    the *time* axis of an ``NLC`` tensor, which is ``axis=1`` -- not the last
    axis as it is for PyTorch's ``NCL`` tensors.  ``axis`` is explicit so the
    intent is visible at every call site.
    
```

## sanm.py :: make_pad_mask

```
Port of ``funasr.models.transformer.utils.nets_utils.make_pad_mask``.

    Returns a boolean ``(batch, maxlen)`` array that is ``True`` on padded
    positions (i.e. the logical inverse of :func:`sequence_mask`).
    
```

## sanm.py :: sequence_mask

```
Port of ``funasr.models.scama.utils.sequence_mask``.

    Returns a ``(batch, maxlen)`` array holding ``1.0`` on valid positions and
    ``0.0`` on padded ones.  ``lengths`` may be ``(B,)`` or ``(B, 1)``.
    
```

## sensevoice.py :: ctc_greedy_decode

```
``argmax`` -> ``unique_consecutive`` -> drop ``blank_id``.

    Faithful to ``SenseVoiceSmall.inference``: consecutive duplicates collapse
    *before* blanks are removed, so ``[a, blank, a]`` decodes to ``[a, a]``.
    
```

## sensevoice.py :: greedy_decode

```
Greedy CTC token ids per utterance (mirrors ``SenseVoiceSmall.inference``).
```

## sensevoice.py :: load_checkpoint

```
Strict load; see :func:`funasr_mlx.models.sanm.load_strict`.
```

## sensevoice.py :: state_dict

```
``{pytorch_name: mlx_array}`` in MLX layout (no key rewriting).
```

## sensevoice.py :: SenseVoiceSmall

```
``funasr.models.sense_voice.model.SenseVoiceSmall`` (inference only).
```

## sensevoice.py :: _weight_dtype

```
Dtype of this encoder's checkpoint weights (``encoders0`` always exists).

        Read through :func:`~funasr_mlx.models.llm.quant.activation_dtype` rather
        than ``weight.dtype`` directly: the SANM stack is a documented
        quantization target (``audio_encoder.`` is in fun-asr-nano's scope), and
        a quantized Linear's ``weight`` holds packed ``uint32`` codes.  Casting
        the fbank input to ``uint32`` would destroy it exactly as the decoder
        embedding cast did.  The helper returns the side tensor's dtype for a
        quantized module, so this stays correct either way.
        
```

## sensevoice.py :: CTC

```
``funasr.models.ctc.ctc.CTC`` restricted to its inference path.

    The only parameter is ``ctc_lo`` (``Linear(encoder_output_size, odim)``);
    the loss/argmax helpers of the reference are re-implemented on top of it.
    
```

## sensevoice.py :: sensevoice_config_from_dict

```
Build a :class:`SenseVoiceSmallConfig` from a FunASR ``model.yaml`` mapping.

    Accepts either the flat ``model_conf`` block or the whole document with
    ``model_conf``/``encoder_conf``/``ctc_conf`` sections.
    
```

## sensevoice.py :: SenseVoiceSmallConfig

```
Mirror of FunASR's ``model_conf`` for ``SenseVoiceSmall``.

    The defaults are the published ``SenseVoiceSmall`` values
    (``input_size=560``, 50 SANM blocks + 20 tail blocks, ``vocab_size=55465``).
    
```

## sensevoice.py :: CTCConfig

```
Mirror of FunASR's ``ctc_conf`` (only the inference-relevant fields).
```

## sensevoice.py :: SenseVoiceEncoderSmallConfig

```
Mirror of FunASR's ``encoder_conf`` for ``SenseVoiceEncoderSmall``.
```

## whisper.py :: _suppress

```
Set the given vocabulary columns to ``-inf`` (a port of the logit filters).

    ``vocab`` (V,) is broadcast **against** the banned ids (N,) rather than the
    other way round, and the membership test is reduced over the N axis.  Building
    a ``(1, V)`` and a ``(1, N)`` array and comparing them would only broadcast
    when ``N == 1`` or ``N == V``; with two or more suppressed tokens and a
    vocabulary of a different size it raises instead of masking.
    
```

## whisper.py :: GreedyDecodeResult

```
Token-level greedy output (ids only; decoding to text is not provided).
```

## whisper.py :: GreedyDecodeConfig

```
Bounds and logit filters for :func:`greedy_decode`.

    ``eot_token_id`` and ``space_token_id`` **must** be supplied by the caller
    (they depend on the BPE vocabulary of the particular checkpoint) or left as
    ``None`` to disable the corresponding filter.  No default is guessed.
    
```

## whisper.py :: mel_filterbank

```
Slaney-normalised mel filterbank, shape ``(n_mels, n_fft // 2 + 1)``.

    Re-implementation of ``librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)``,
    which is how the reference's ``mel_filters.npz`` asset was produced.  **Not
    numerically verified against the official asset** -- if the official array is
    available, load it and pass it to :func:`log_mel_spectrogram` directly instead
    of relying on this function.
    
```

## whisper.py :: _is_non_persistent_buffer

```
Whether ``owner``'s parameter ``key`` is a derived, non-persistent buffer.

    ``isinstance`` rather than an exact type match, so a subclass still matches.
    The result is independent of any flatten prefix or of whether the owning
    module is reached standalone or as part of a full ``Whisper``.
    
```

## whisper.py :: _hz_to_mel

```
Slaney mel scale (librosa default, ``htk=False``).
```

## whisper.py :: pad_or_trim

```
Port of ``whisper.audio.pad_or_trim`` (zero-pad or truncate).
```

## whisper.py :: reflect_pad_1d

```
``torch.nn.functional.pad(..., mode="reflect")`` along the last axis.

    The edge value is **not** repeated.  With ``x = [1, 2, 3, 4]`` and ``pad=2``
    on both sides the result is ``[3, 2, 1, 2, 3, 4, 3, 2]``: the left filler
    reads left-to-right as ``x[pad] ... x[1]`` and the right filler as
    ``x[n-2] ... x[n-1-pad]`` -- both *descending* index runs.
    
```

## whisper.py :: embed_audio

```
Encode ``(B, 2 * n_audio_ctx, n_mels)`` mel frames.

        The encoder casts the mel to its own weight dtype (see
        :meth:`AudioEncoder.__call__`), so an fp16 checkpoint produces fp16
        audio features even though the front-end mel is float32.
        
```

## whisper.py :: offset

```
Absolute position of the next token, from the self-attention cache.

        The reference reads ``next(iter(kv_cache.values())).shape[1]``, which
        relies on ``dict`` insertion order.  Here it is read deterministically
        from the first block's self-attention key.
        
```

## whisper.py :: _weight_dtype

```
Dtype of this module's checkpoint weights (``ln`` always exists).
```

## whisper.py :: TextDecoder

```
``whisper.model.TextDecoder`` with an explicit MLX KV cache.

    The causal mask is **not** a parameter: the reference registers it with
    ``persistent=False``, so it never appears in a ``state_dict`` and is
    regenerated on demand here.
    
```

## whisper.py :: _weight_dtype

```
Dtype of this module's checkpoint weights (``conv1`` always exists).
```

## whisper.py :: AudioEncoder

```
``whisper.model.AudioEncoder``.

    ``mel`` is channels-last ``(B, T', n_mels)``; the reference's
    ``permute(0, 2, 1)`` after ``conv2`` is unnecessary under MLX's channels-last
    Conv1d.  ``positional_embedding`` is materialised (the reference registers it
    as a persistent buffer) so it round-trips through strict loading; the
    regenerated value is identical to :func:`sinusoids`.
    
```

## whisper.py :: whisper_config_from_dict

```
Build a :class:`WhisperConfig` from a FunASR ``model.yaml``-like mapping.

    ``WhisperParameters`` is also accepted as an alias for the ``parameters``
    section of FunASR's document.
    
```

## whisper.py :: WhisperConfig

```
Mirror of ``funasr/models/whisper/model.py::Whisper``'s inference config.
```

## whisper.py :: ModelDimensions

```
Mirror of ``whisper.model.ModelDimensions``.
```

## whisper.py :: qkv_attention

```
Port of ``MultiHeadAttention.qkv_attention`` (the non-SDPA path).

        Layouts follow the reference exactly: ``q`` is ``(B, H, T, d)`` and
        ``k`` is **transposed** to ``(B, H, d, T)``.  That ``.permute(0, 2, 3, 1)``
        in the reference is what makes ``q @ k`` contract the head dimension and
        produce the ``(B, H, T, T)`` score matrix; leaving ``k`` as
        ``(B, H, T, d)`` would contract ``T`` with ``T`` instead.

        Whisper applies ``(n_state // n_head) ** -0.25`` to *both* ``q`` and
        ``k``, which is exactly a ``d_k ** -0.5`` factor on the scores.
        
```

## whisper.py :: _split

```
``(B, T, C)`` -> ``(B, H, T, d)``.
```

## whisper.py :: cache_prefix

```
Cache key prefix for this attention (e.g. ``"blocks.0.attn"``).
```

## whisper.py :: sinusoids

```
Port of ``whisper.model.sinusoids``.

    Note this is **0-based** and uses ``-log_increment * arange(channels // 2)``
    -- different from FunASR's ``SinusoidalPositionEncoder`` (1-based, positive
    increments), which the SANM blocks use.  Both quirks matter and both are
    preserved exactly.
    
```

## whisper.py :: LayerNorm

```
``whisper.model.LayerNorm`` -- fp32 reduction, torch's default ``eps=1e-5``.

    The reference casts the input to ``float32`` before normalising and casts
    the result back to the input dtype; MLX's ``LayerNorm`` already accumulates
    in float32, so the explicit round-trip is kept for clarity and for the fp16
    path.
    
```

## whisper.py :: Linear

```
``torch.nn.Linear`` alias.

    MLX already uses the ``weight (out, in)`` / ``bias (out,)`` layout and the
    same parameter names, so this is purely a readability marker and the anchor
    for the conv-layout rule detection above.
    
```

## whisper.py :: load_strict

```
Load ``tensors`` into ``module``, refusing any key mismatch.

    Arrays must already be in MLX layout unless ``convert=True`` (raw PyTorch
    ``state_dict``).  Raises :class:`CheckpointKeyError` listing the offending
    keys, and ``ValueError`` on any shape mismatch -- never a partial load.
    
```

## whisper.py :: LoadReport

```
Outcome of a :func:`load_strict` call.
```

## whisper.py :: CheckpointKeyError

```
Raised by :func:`load_strict` when a checkpoint does not match a module.
```

## whisper.py :: flatten_state_dict_with_rules

```
Like :func:`flatten_state_dict`, but also returns the per-key layout rule.

    Returns ``(tensors, rules)`` with the same two dictionaries that the
    :class:`LayoutReport` carries -- nothing is dropped and the rule metadata is
    not recomputed by a second walk.  This is the contract
    :func:`funasr_mlx.models.sanm.flatten_state_dict_with_rules` and
    :func:`funasr_mlx.models.campplus.flatten_state_dict_with_rules` expose, so
    the three model modules agree on one name for the same operation; callers
    that want the richer object should use :func:`flatten_state_dict` instead.
    
```

## whisper.py :: flatten_state_dict

```
Flatten a module tree into FunASR/PyTorch-compatible parameter names.

    The layout rule is derived from the owning module type rather than from a
    name pattern, so it cannot silently mis-convert a renamed submodule.
    Attributes starting with ``_`` are configuration, never weights.
    
```

## whisper.py :: LayoutReport

```
Flat parameter tensors plus the layout rule that applies to each key.

    Behaves like a read-only ``Mapping`` of ``{parameter_name: mlx_array}``, so it
    can be iterated, indexed, ``len()``-ed and fed to ``dict(...)`` exactly like a
    plain state dict, while also carrying the per-key rule used for the
    torch -> MLX layout conversion.
    
```

## whisper.py :: _rule_for

```
Conv1d is the only module whose MLX layout differs from PyTorch's.
```

## whisper.py :: ModuleStack

```
Ordered children registered under their numeric index.

    ``torch.nn.Sequential`` yields ``<parent>.<i>.<leaf>`` parameter names; MLX's
    ``nn.Sequential`` would insert a ``layers`` level, so children are registered
    directly under ``"0"``, ``"1"``, ...
    
```

## awq.py :: apply_scale

```
Fold the chosen scales into the site's modules (bf16 stays bf16).

        Two fold shapes, exactly as mlx-lm: a Linear predecessor divides its
        OUTPUT-channel rows (``weight / scales[:, None]`` -- the scales index
        the successor's input channels, which are the predecessor's outputs),
        while a norm predecessor divides elementwise (its weight is a vector
        over the same channels it normalises).
        
```

## awq.py :: search_scale

```
Grid-search channel scales for one site; returns the best scales.

        The reference output is the block (or sub-block) run on the captured
        input of the site's FIRST layer; each candidate scales that input's
        channels via the successor weights only (equivalent math, see
        upstream), quantize-dequantizes them, and the MSE of the sub-block's
        output picks the winner.
        
```

## awq.py :: _run_block

```
Run one decoder block on ``x`` [1, rows, hidden] in row batches.

        Our ``Qwen3TransformerBlock`` has no ``__call__`` of its own (the
        forward is unrolled in ``Qwen3Model.__call__``), so the qwen3 residual
        structure is applied here: ``h += attn(norm(h)); h += mlp(norm(h))``.
        The site's sub-block (``self_attn`` / ``mlp``) IS callable directly.
        
```

## awq.py :: capture

```
Run ``run()`` with input hooks on this block's Linears.

        ``run`` is a callable that performs the model's real prefill for one
        utterance; the block must be part of the live model.  Each Linear's
        input rows are appended to ``input_feat`` on the module (mlx-lm's
        convention), in float32, and the block's own input is recorded on
        ``self.block_input`` for the never-worse reference.
        
```

## awq.py :: MultiBlockCapture

```
Capture Linear inputs for MANY blocks in ONE pass over the dataset.

    Running the model once per block (28 x 480 prefills) is what made the
    first calibration attempt take ~575s per block; mlx-lm's AWQ catches every
    Linear of every block in the same forward.  Memory stays bounded by
    stride-subsampling each Linear's rows down towards ``row_cap`` after
    every utterance (halving whenever over the cap) and storing float16.
    
```

## awq.py :: _to_mx

```
Captured activations are numpy (host); the search wants mx arrays.

    The conversion happens once per consumer, so the device copy never
    exceeds the capped rows either.
    
```

## awq.py :: ScaleSite

```
One scale search site: a norm (or Linear) feeding several Linears.

    Mirrors mlx-lm's ``ScaleConfig`` for the Qwen3 block layout:

    * ``prev`` -- the module whose OUTPUT is scaled (an RMSNorm for qwen3);
      its weight is divided by the scales.
    * ``layers`` -- the successor Linear weights are multiplied by the scales.
    * ``block`` -- optional sub-block to inspect (``self_attn``), because the
      loss must be measured on the attention output, not one Linear alone.
    
```

## calibrate.py :: resolve

```
Locate the head on ``model`` and describe how it is called.

        ``head_path`` is a dotted path to the head module (e.g.
        ``thinker.model.embed_tokens`` or ``language_model.lm_head``).
        
```

## calibrate.py :: HeadTarget

```
Which module in a live model is the LM head, and how it is called.

    ``kind`` is ``"tied"`` when the head is the token embedding used through
    ``as_linear`` (qwen3 / moss) and ``"linear"`` when the checkpoint has a
    separate ``lm_head`` (glm).  ``hidden_size`` is the head's *input*
    dimension -- the size of the Hessian.
    
```

## gptq.py :: hessian_from_activations

```
Convenience: ``H = 2 X^T X`` over a stream of activation batches.
```

## gptq.py :: language_mass

```
Share of the Hessian's effective rows per language (the guard's check).

        Mirrors :meth:`hessian` exactly, so the reported shares are the weights
        actually applied to the accumulated products.
        
```

## gptq.py :: _row_weights

```
Multiplier applied to each language's accumulated ``X^T X``.

        The stored products are already sums, so an unweighted Hessian needs a
        multiplier of **1**, not the row count.
        
```

## gptq.py :: _effective_rows

```
Per-language row count the Hessian effectively sees.

        Both modes are one family -- a common row threshold ``T`` with
        ``rows_i -> min(rows_i, T)``:

        * ``balance`` picks ``T = min(rows)``, giving every language an equal
          share;
        * ``cap_fraction`` bisects for the *largest* ``T`` whose resulting
          maximum share is still within the cap, so the most calibration data is
          kept subject to the guard.
        
```

## gptq.py :: _weighted_effective_rows

```
Per-language effective rows after weighting, total preserved.

        The renormalisation is done in EFFECTIVE-ROW space, which is the space
        the 30% guard is stated in and the only one where "total mass" is
        meaningful.  An earlier version renormalised the multipliers instead,
        which silently changed the overall scale of the Hessian and therefore the
        damping (defined relative to the mean diagonal).
        
```

## gptq.py :: _pack_triple

```
Re-derive the stored triple from a reconstructed matrix.

    Deriving the triple from the *final* reconstructed values (rather than
    keeping the per-column codes) guarantees the checkpoint is self-consistent:
    ``mx.dequantize(packed, scales, biases)`` reproduces exactly what GPTQ
    decided, with no dependency on how the inner loop was organized.
    
```

## gptq.py :: rtn_quantize

```
Plain round-to-nearest group quantization (``mx.quantize``'s own rule).

    Provided as the baseline GPTQ has to beat, computed with the *same*
    quantizer so the comparison isolates the algorithm and not the rounding.
    Pass ``hessian`` to also get the layer-output error, which is the metric the
    head-to-head claim is made on.
    
```

## gptq.py :: gptq_quantize

```
Quantize ``weight`` ``[out, in]`` with GPTQ using ``hessian`` ``[in, in]``.

    ``hessian`` is the second-order statistic of the layer input,
    ``H = 2 X^T X`` accumulated in float32 (:class:`HessianAccumulator` does
    exactly that).  The returned triple is what a W4A16 checkpoint stores, in
    the standard MLX affine group layout.
    
```

## gptq.py :: _upper_cholesky_inverse

```
``U`` with ``Hinv = U^T U`` and ``U`` upper triangular (float32 out).

    The reference implementation goes ``chol(H) -> cholesky_inverse -> chol(·,
    upper=True)``; ``inv(H)`` then an upper Cholesky is the same matrix with one
    fewer decomposition, and the dampening has already made ``H`` positive
    definite.
    
```

## gptq.py :: GptqConfig

```
GPTQ hyper-parameters, mirroring the reference implementation's knobs.
```

## gptq.py :: cast

```
Round ``values`` to the storage dtype, keeping a float32 carrier.
```

## gptq.py :: storage_dtype

```
NumPy dtype the side tensors are returned in.

        bfloat16 has no NumPy representation, so it is carried as float32 whose
        *values* are already bf16-rounded -- converting to bf16 later is exact.
        
```

## gptq.py :: AffineQuantizer

```
MLX affine group quantization, bit-exact (see module docstring).

    ``bits`` and ``group_size`` must be a combination ``mx.quantize`` accepts:
    bits in (2, 3, 4, 5, 6, 8) and group size in (32, 64, 128).  ``storage``
    names the dtype the side tensors are stored in; it only affects rounding of
    ``scales``/``biases`` and therefore which codes are chosen.
    
```

## gptq.py :: _round_half_away

```
``round()`` with ties away from zero, as the MLX kernel does.

    NumPy's ``np.round`` is half-to-even, which disagrees with the kernel on
    exact .5 ties; they are rare on real weights but not on synthetic ones.
    
```

## kv_cache.py :: reserve

```
Grow the backing store to ``positions`` capacity immediately.

        ``like`` may carry the (B, H, D) shape when this cache is still empty.
        
```

## kv_cache.py :: KVCache

```
mlx-lm ``KVCache``: append-only keys/values with an offset cursor.

    ``reserve`` pre-allocates the buffer once (call it after prefill, when the
    final length is known: audio positions + prompt + max decode steps). Without
    it, the cache grows in ``step``-sized chunks, and every growth allocates a
    fresh buffer and copies the whole history -- measured ~3 ms per event on the
    0.6B, a 2-3x tax on a 7 ms decode step, hitting exactly when a long
    utterance crosses a 256-position boundary mid-decode.
    
```

## quant.py :: apply_quantization

```
Quantize ``model`` in place per the ``config.json`` section ``data``.

    Called by :func:`funasr_mlx.pipeline.load_model` *before* the expected
    parameter set is computed, so a quantized checkpoint's ``.scales`` and
    ``.biases`` entries are part of the strict key/shape check.  Returns
    ``None`` when the configuration carries no quantization section, i.e. the
    checkpoint is full precision.
    
```

## quant.py :: quantizes_module

```
Whether ``module`` (a live ``nn.Module``) should become quantized.

        The token embedding is a separate case from ``nn.Linear``: it is not a
        ``Linear`` subclass, so it can only be selected through an explicit
        ``embed_tokens`` target prefix, and it is the tied LM head as well.
        
```

## quant.py :: quantizes_tensor

```
Whether the converter should quantize checkpoint tensor ``name``.

        Mirrors the model-side predicate exactly: a Linear weight (see
        :meth:`is_linear_weight`) or a selected token embedding (see
        :meth:`quantizes_embedding`), whose input dimension the group size
        divides.
        
```

## quant.py :: is_linear_weight

```
Whether ``name`` is a Linear weight the target prefixes select.

        Rank-2 ``.weight`` under a selected prefix and not one of the
        :data:`NON_LINEAR_WEIGHT_NAMES` exceptions -- i.e. a tensor the model
        side would have turned into a ``QuantizedLinear`` if the group size
        fitted. This is the predicate the converter uses to decide whether a
        tensor it did *not* pack is a legitimate skip or an unnoticed miss.
        
```

## quant.py :: quantizes_embedding

```
Converter predicate for the token-embedding matrix.
```

## quant.py :: selects_embedding

```
Whether a token-embedding module at ``module_path`` is selected.

        The tied ``embed_tokens`` is the one non-Linear weight worth offering:
        when ``tie_word_embeddings`` is set the same matrix is also the LM head,
        so at bs=1 it is streamed once per step and is a large share of the step
        (311 MB bf16 for a 151936-vocab family). It needs its own predicate
        because ``nn.Embedding`` is not an ``nn.Linear`` subclass, so the Linear
        rule can never reach it; a family opts in by naming ``embed_tokens`` in
        ``targets``.
        
```

## quant.py :: QuantizationSpec

```
One family's quantization plan, as declared in ``config.json``.

    ``method`` records *how* the stored ``scales``/``biases``/codes were
    computed: ``rtn`` is ``mx.quantize``'s round-to-nearest, ``gptq`` is
    Hessian-based error compensation (:mod:`funasr_mlx.models.llm.gptq`).  The
    runtime layout is identical either way -- ``nn.QuantizedLinear`` and
    ``mx.quantized_matmul`` cannot tell them apart -- so the loader ignores it
    and it exists purely as provenance in the checkpoint's ``config.json``.
    
```

## qwen3.py :: Qwen3ForCausalLM

```
``Qwen3ForCausalLM``: the model plus the (tied or separate) LM head.
```

## rope.py :: initialize_rope

```
mlx-lm ``models/rope_utils.py::initialize_rope`` (inference subset).

    ``default`` (and absent config) and ``linear`` are supported; anything
    else raises rather than silently degrading position handling.
    
```

## splice.py :: splice_and_forward

```
Replace the ``pad_span``-wide audio-token run with tower output; decode.

    Prefill needs the causal mask even when KVCache objects already exist:
    the generic model call only builds one when ``cache is None`` as a whole.
    ``decoder_dtype``: the tower emits float32 embeds (float32 mel in); the
    cast prevents a mixed-dtype promotion at every Linear of the stack (a
    no-op for float32 checkpoints).
    
```


---

# Second wave: shorter docstrings (moved verbatim)

## api.py :: pseudo_stream

```
Create a PseudoStream bound to a loaded LLM transcriber.

    Must run inside a guard (use with TranscribeWorker / guarded child):

        worker = TranscribeWorker(model)
        stream = worker.pseudo_stream(chunk_seconds=1.0)
        for chunk in mic_chunks:
            ev = stream.push(chunk)
            if ev: render(ev.text)
        final = stream.close()   # == offline output, bit-identical
    
```

## api.py :: download_model

```
Download a published checkpoint (see funasr_mlx.download.download).
```

## checkpoint_stream.py :: write_tensor

```
Write one tensor, converting only bounded slices rather than its full data.

    ``code`` overrides the safetensors type code written for the payload; it is
    how a bfloat16 tensor (carried as a ``uint16`` array, since NumPy has no
    bfloat16) is stored as ``BF16`` rather than ``U16``. The override must agree
    with the array's element width, so no reinterpretation can silently happen.
    
```

## checkpoints.py :: _bf16_carrier_to

```
Round a ``uint16``-carried bfloat16 tensor to the requested precision.

    bfloat16 -> float32 is exact (a 16-bit left shift; the shift must be done
    in 32-bit lanes or the high half is discarded). The further round to
    float16 happens on the host, exactly as an on-device ``astype`` would.
    
```

## checkpoints.py :: _carry

```
Copy one source tensor into a small owned array in its carrier dtype.

    The copy is the point: the source buffer belongs to a lazily-read shard
    handle, and converted arrays that still referenced it would pin the whole
    shard in memory. Every layout transpose therefore happens *after* this
    copy, on the owned array. bfloat16 is viewed as ``uint16`` first because
    ``torch.Tensor.numpy`` refuses the bfloat16 scalar type outright.
    
```

## checkpoints.py :: quantize_tensor

```
Quantize one tensor and return MLX tensors in the usual layout.
```

## checkpoints.py :: GptqWeight

```
Per-tensor GPTQ calibration state, keyed by native tensor name.

    ``hessians`` maps a native ``<module>.weight`` name to that matrix's
    ``H = 2 X^T X`` (float32).  ``configs`` maps the same name to the
    :class:`~funasr_mlx.models.llm.gptq.GptqConfig` to use, so different
    matrices can carry different hyper-parameters.  A name with a Hessian but no
    explicit config uses ``default_config``.

    Kept as a plain data holder (no MLX import at module scope) so the converter
    can accept it without pulling the runtime in.
    
```

## checkpoints.py :: SourceRule

```
How one LLM family's source keys map onto the native parameter names.

    ``prefixes`` maps a source key prefix to the native prefix, longest first.
    ``renames`` are literal substring rewrites applied afterwards (used for the
    Qwen3-ASR ``conv2d`` -> ``conv`` submodule move and for MOSS's
    ``language_model`` -> ``model`` decoder nesting). ``drop`` lists source keys
    that the native model does not carry, because the port *derives* them (the
    Qwen3-ASR sinusoidal position table). A rule is strict: a key that matches
    no prefix and is not in ``drop`` raises rather than being passed through, so
    a renamed upstream checkpoint cannot silently produce a wrong key set.
    
```

## llm_asr.py :: LlmAsrTranscriber

```
Resident transcriber: one model load, many calls.

    Construct inside a guarded child (``memory.run_guarded``); the API layer
    arranges that. KV caches are allocated once and reset (offset=0) between
    utterances -- reuse validated bit-stable in .tmp/opt/LOG.md.
    
```

## pipeline.py :: load_model

```
Prevalidate every key/shape, then load one bounded shard at a time.

    Only native-layout, explicitly tagged checkpoints are admitted. Running in
    a guarded child is mandatory; importing this module does not load MLX.
    
```

## pseudo_streaming.py :: llm_transcriber

```
Adapt an ``LlmAsrTranscriber`` into a ``samples -> text`` callable.
```

## pseudo_streaming.py :: classic_transcriber

```
Build a ``samples -> text`` callable for classic (non-LLM) families.

    Covers sensevoice / paraformer (greedy CTC/CIF). ``cmvn`` is REQUIRED
    for sensevoice-family checkpoints (``am.mvn``; found next to the source
    configuration) -- without it the fbank distribution is shifted and the
    encoder collapses to all-blanks (verified: argmax == 0 everywhere).
    
```

## pseudo_streaming.py :: StreamEvent

```
One pseudo-stream block result.

    ``text`` is the full current best-effort transcription (stable prefix +
    rolling tail). ``is_final`` marks the closing event, whose text equals
    offline transcription exactly.
    
```

## _layout_state.py :: convert_tensor

```
Apply the transpose for ``rule``; ``names`` is the family's identity set.
```

## _strict_load.py :: load_state_dict_strict

```
Strictly load ``state`` into ``model``; raise on any mismatch.
```

## _strict_load.py :: convert_checkpoint_tensors

```
Convert a FunASR/PyTorch ``state_dict`` into MLX layout.

    ``rules`` maps key suffixes to transforms; each family's depthwise-conv
    transpose lives there. Everything else is copied verbatim.
    
```

## _strict_load.py :: flat_parameter_names

```
Flatten ``module.parameters()`` into FunASR-style dotted names.
```

## campplus.py :: forward

```
``x`` is fbank ``(B, T, feat_dim)``.

        Literal port of the reference ``forward``: the batch is assumed to be
        already padded (or to contain a single utterance), because the reference
        pools without any padding mask.  See :meth:`embed` for the length-aware
        variant.
        
```

## campplus.py :: frame_features

```
Run everything up to (but excluding) pooling; returns ``(B, T, C)``.

        The stage order is taken from the ``xvector`` container's own insertion
        order, which mirrors the reference's ``nn.Sequential``: ``tdnn``, then
        ``block1, transit1, block2, transit2, ...``, then ``out_nonlinear``.
        Iterating separate block/transit name lists instead would apply **all**
        blocks before **any** transit, which both misorders the network and
        desynchronises the channel arithmetic (each transit halves the channel
        count feeding the next block).
        
```

## campplus.py :: load_checkpoint

```
Strict load; see :func:`load_strict`.
```

## campplus.py :: state_dict

```
Flat ``{pytorch_name: mlx_array}`` plus per-key layout rules.
```

## campplus.py :: CAMPPlus

```
``funasr.models.campplus.model.CAMPPlus`` (inference only).

    Input is 80-dim Kaldi fbank **without** the LFR stack and **without** CMVN;
    ``funasr/models/campplus/utils.py::extract_feature`` mean-normalises each
    utterance across all frames and zero-pads the batch.  Both steps stay
    outside this module (frontend scope).

    ``output_level="segment"`` returns ``(B, embedding_size)``;
    ``output_level="frame"`` returns ``(B, T, channels)``.
    
```

## campplus.py :: campplus_config_from_dict

```
Build a :class:`CAMPlusConfig` from a FunASR ``model.yaml`` mapping.
```

## campplus.py :: CAMPlusConfig

```
Mirror of FunASR's ``model_conf`` for ``CAMPPlus``.

    Defaults are the published ``speech_campplus_sv_zh-cn_16k-common`` values
    taken from ``funasr/models/campplus/template.yaml``.
    
```

## campplus.py :: DenseLayer

```
``components.DenseLayer``.

    The reference special-cases 2-D input by squeezing through a length-1 axis so
    that a 1x1 ``Conv1d`` can be used as a plain ``Linear``; that branch is
    reproduced (channels-last, so the squeezed axis is the time axis).
    
```

## campplus.py :: TransitLayer

```
``components.TransitLayer`` (nonlinearity **before** the linear map).
```

## campplus.py :: CAMDenseTDNNLayer

```
``components.CAMDenseTDNNLayer``.

    ``memory_efficient`` only gates ``torch.utils.checkpoint`` during *training*
    and is therefore inert here (documented, not silently dropped).
    
```

## campplus.py :: CAMLayer

```
``components.CAMLayer`` (channels attention over a dilated TDNN).

    ``context`` mixes the *global* time mean with a non-overlapping segment mean
    (``seg_len=100``, average pooling with ``ceil_mode=True``), squeezes it
    through ``linear1``/``relu``/``linear2``/``sigmoid``, and rescales the local
    branch.  Because the channel axis is last, the time axis is axis 1.
    
```

## campplus.py :: TDNNLayer

```
``components.TDNNLayer``.

    ``padding < 0`` selects the reference's same-padding rule
    ``padding = (kernel_size - 1) // 2 * dilation`` and requires an odd kernel.
    
```

## campplus.py :: __call__

```
``x`` is ``(B, F, T)`` (mel-major, as in the reference) -> ``(B, T, C*F/8)``.
```

## campplus.py :: BasicResBlock

```
``components.BasicResBlock`` (channels-last NHWC).

    The torch shortcut is an ``nn.Sequential`` (empty => identity), which puts
    its parameters at ``shortcut.0.weight`` / ``shortcut.1.weight``; a
    :class:`ModuleStack` (empty in the identity case) reproduces those names
    exactly.
    
```

## campplus.py :: statistics_pooling

```
Time-axis mean/std pooling of an ``(B, L, C)`` tensor -> ``(B, 1, 2C)``.

    The reference's ``statistics_pooling(x, dim=-1)`` reduces the *time* axis of
    an NCL tensor and concatenates along the channel axis, giving ``(B, 2C)``.
    Reducing axis 1 of the channels-last tensor and concatenating on the last
    axis gives ``(B, 1, 2C)`` with the same numbers in the same order, which is
    what the following ``DenseLayer`` consumes.  ``eps`` is accepted for
    signature parity; the reference never uses it.
    
```

## campplus.py :: get_nonlinear

```
Retain the reference Sequential's named children and per-channel PReLU.
```

## campplus.py :: kaiming_normal_

```
In-place-style ``torch.nn.init.kaiming_normal_`` for MLX conv/linear weights.

    The reference re-initialises every ``Conv1d``/``Linear`` weight with
    ``kaiming_normal_`` and zeroes the biases *after* building ``CAMPPlus``.
    ``fan_in`` is the number of input elements per output element, which is the
    product of every axis except the output axis in both layouts.
    
```

## campplus.py :: LoadReport

```
Outcome of a :func:`load_strict` call.

    ``dropped`` lists the torch-only buffers that were intentionally ignored.
    It is never empty silently: callers can assert on it.
    
```

## campplus.py :: flatten_state_dict_with_rules

```
Like :func:`flatten_state_dict`, but also returns the per-key layout rule.

    Returns ``(tensors, rules)`` with the same two dictionaries that the
    :class:`LayoutReport` carries -- nothing is dropped.  This is the contract
    :func:`funasr_mlx.models.sanm.flatten_state_dict_with_rules` exposes, so the
    three model modules agree on one name for the same operation; callers that
    want the richer object should use :func:`flatten_state_dict` instead.
    
```

## campplus.py :: flatten_state_dict

```
Flatten a module tree into FunASR/PyTorch-compatible parameter names.

    Unlike a name-pattern heuristic, the layout rule is derived from the *owning
    module type*, so it is correct for any naming the architecture happens to
    use.  Attributes starting with ``_`` are configuration, never weights.
    
```

## campplus.py :: LayoutReport

```
Flat parameter tensors plus the layout rule that applies to each key.

    Behaves like a read-only ``Mapping`` of ``{parameter_name: mlx_array}``, so it
    can be iterated, indexed, ``len()``-ed and fed to ``dict(...)`` exactly like a
    plain state dict, while also carrying the per-key rule used for the
    torch -> MLX layout conversion.
    
```

## campplus.py :: NamedLayerList

```
Named children iterated in insertion order.

    Mirrors ``torch.nn.ModuleList`` with explicit names (``CAMDenseTDNNBlock``
    uses ``tdnnd1``, ``tdnnd2``, ...).
    
```

## campplus.py :: NamedModuleDict

```
Named children applied in order; keeps the ``<parent>.<name>.<leaf>`` keys.

    Used for ``CAMPPlus.xvector``, which the reference builds as an
    ``nn.Sequential`` of named sub-modules.
    
```

## campplus.py :: ModuleStack

```
Ordered children registered under their numeric index.

    ``torch.nn.Sequential`` yields ``<parent>.<i>.<leaf>`` parameter names; MLX's
    ``nn.Sequential`` would insert an extra ``layers`` level, so children are
    registered directly under ``"0"``, ``"1"``, ...
    
```

## ct_punc.py :: reset

```
No-op: ``ct-punc`` carries no state between calls.

        Provided for API symmetry with the streaming VAD; the mini-sentence cache
        is local to :meth:`punctuate`.
        
```

## ct_punc.py :: CtPuncPunctuator

```
End-to-end ``ct-punc``: tokens -> model -> punctuated text + ``punc_array``.

    The mini-sentence chunking / look-back cache behaviour of
    ``CTTransformer.inference`` is reproduced exactly, including the case where a
    chunk contains no sentence terminator: nothing is emitted for that chunk, the
    whole chunk is carried into the next one, and once it exceeds
    ``cache_pop_trigger_limit`` tokens the last comma is promoted to a sentence
    end.
    
```

## ct_punc.py :: expand_word_level_punc

```
FunASR's jieba word-level ``punc_array`` expansion.

    Mirrors the reverse walk in ``CTTransformer.inference`` for CJK word tokens
    longer than one character: insert ``len(token) - 1`` entries of ``no_punc_id``
    so the array lines up with the character-level token stream.

    The reference hard-codes the inserted value as the literal ``1``; that is kept
    as the default so behaviour is bit-identical, but note that ``1`` only means
    "no punctuation" when ``punc_list[1] == "_"``.  Pass ``no_punc_id=0`` (or
    whatever the target config uses) when the label table is ordered differently.
    
```

## ct_punc.py :: split_to_mini_sentence

```
FunASR ``split_to_mini_sentence``: fixed-size chunks plus one remainder.
```

## ct_punc.py :: split_words

```
FunASR ``split_words`` without a jieba user dictionary.

    Splits on whitespace, then keeps maximal ASCII runs as one token and emits
    every non-ASCII character as its own token.
    
```

## ct_punc.py :: argmax_punc_ids

```
FunASR's ``y.view(-1, C).topk(1, dim=1)`` followed by ``squeeze(1)``.
```

## ct_punc.py :: CtPuncModel

```
``CTTransformer``: embedding -> SANMEncoder -> linear punc classifier.
```

## ct_punc.py :: SanmEncoder

```
``SANMEncoder`` restricted to ``input_layer="pe"``.

    Parameter paths are ``encoders0.0.…`` for the first block and ``encoders.i.…``
    for the remaining ``num_blocks - 1``, matching FunASR exactly.
    
```

## ct_punc.py :: SanmAttention

```
``MultiHeadedAttentionSANM`` (funasr/models/sanm/attention.py).

    Weight layout conversion: ``self.fsmn_block.weight`` is ``(C, 1, K)`` in
    PyTorch (depthwise ``nn.Conv1d``) and must be transposed to MLX's
    ``(C, K, 1)``.  ``linear_q_k_v`` and ``linear_out`` already use MLX's layout.
    
```

## ct_punc.py :: PositionwiseFeedForward

```
``PositionwiseFeedForward`` with the ``w_1``/``w_2`` names preserved.

    Dropout is omitted: FunASR's ``nn.Dropout`` is an identity at inference and
    this port is inference-only.
    
```

## ct_punc.py :: ModuleLayerList

```
``nn.ModuleList`` variant keyed by *string* index.

    MLX's own ``ModuleList`` nests children under ``layers``, which would shift
    every checkpoint key (``encoders.0.…`` -> ``encoders.layers.0.…``).  Storing
    the children directly under ``"0"``, ``"1"``, … keeps FunASR's
    ``nn.ModuleList`` parameter paths byte-identical.

    Note ``nn.Module.update()`` can only overwrite *existing* keys, so children
    are inserted as dict items (which is what ``Module.__setattr__`` does for
    ``dict``/``list`` values too).
    
```

## ct_punc.py :: sinusoidal_position_encoding

```
FunASR ``SinusoidalPositionEncoder`` (``input_layer="pe"``), parameter-free.

    ``log_timescale_increment = log(10000) / (depth/2 - 1)``; the sin block is
    concatenated *before* the cos block; positions start at 1.
    
```

## ct_punc.py :: make_valid_mask

```
``(~make_pad_mask(lengths)[:, None, :])`` -> ``(B, 1, maxlen)`` of 1/0 floats.
```

## ct_punc.py :: is_first_byte_ascii

```
FunASR's ``len(token[0].encode()) == 1`` test (first character is ASCII).
```

## ct_punc.py :: from_config

```
Build from a FunASR ``tokenizer_conf`` mapping (or an explicit path).
```

## ct_punc.py :: from_file

```
Load a newline-separated vocabulary (``token_list.txt``).
```

## ct_punc.py :: CharVocab

```
Character vocabulary equivalent to FunASR's ``CharTokenizer``.

    ``BaseTokenizer.encode`` simply maps every token through ``token2id`` with an
    ``<unk>`` fallback, so this class exposes exactly that behaviour and nothing
    more (no segmentation: the caller already produced tokens via
    :func:`split_words`).
    
```

## ct_punc.py :: from_dict

```
Build from a full FunASR ``config.yaml`` mapping or from a flat one.

        Accepts ``{"model_conf": {...}, "encoder_conf": {...},
        "tokenizer_conf": {...}}`` (the on-disk layout) as well as a single flat
        ``model_conf`` mapping.
        
```

## ct_punc.py :: CtPuncConfig

```
Validated mirror of ``model_conf`` for FunASR's ``CTTransformer``.
```

## ct_punc.py :: from_dict

```
Build from a FunASR ``encoder_conf`` mapping.
```

## ct_punc.py :: SanmEncoderConfig

```
Validated mirror of ``encoder_conf`` for FunASR's ``SANMEncoder``.

    Only the ``ct-punc`` configuration is supported: ``input_layer="pe"`` (a
    parameter-free sinusoidal position encoder) and
    ``selfattention_layer_type="sanm"``.
    
```

## emotion2vec.py :: load_state_dict_strict

```
Load ``state`` into ``module``; return ``(discarded, unexpected)``.

    Never silently skips a key: anything outside
    :data:`INFERENCE_ONLY_PREFIXES` must match a parameter by name *and* shape.
    
```

## emotion2vec.py :: convert_checkpoint_tensors

```
Convert a published ``state_dict`` and report what was discarded.

    Only ``Conv1d`` weights change layout: ``(out, in//g, K) -> (out, K, in//g)``.
    Linears, LayerNorms, ``extra_tokens`` and ``alibi_scale`` already match MLX.
    
```

## emotion2vec.py :: is_conv_weight

```
True for the depthwise ``Conv1d`` weights listed in :data:`CONV_WEIGHT_PATTERNS`.
```

## emotion2vec.py :: emotion_scores

```
Full FunASR-equivalent result: labels + softmax scores + features.

        Runs the encoder exactly once; ``feats`` is the utterance-level pooled
        embedding the classifier consumed, which is the same
        ``x.mean(dim=1)`` the reference uses, so no second pass is needed.
        
```

## emotion2vec.py :: classify

```
Return ``(features (B, embed_dim), logits (B, vocab_size), frames)``.

        The head is ``nn.Linear(embed_dim, vocab_size)`` applied to the mean over
        time **after** the extra tokens have been removed
        (``Emotion2vec.inference``: ``x = x.mean(dim=1); x = self.proj(x)``).  The
        frame features are returned so a caller that also wants them does not have
        to run the whole encoder a second time.
        
```

## emotion2vec.py :: extract_features

```
Waveform ``(B, samples)`` -> frame features ``(B, T', embed_dim)``.

        ``remove_extra_tokens=True`` strips the ``num_extra_tokens`` cls tokens,
        which is what FunASR does before the utterance-level pooling.  The encoder
        is evaluated exactly **once**.
        
```

## emotion2vec.py :: Emotion2VecModel

```
``Emotion2vec`` (FunASR wrapper) as native MLX modules.

    The checkpoint is a plain ``state_dict`` of the ``Emotion2vec`` module, so the
    parameter prefixes are ``modality_encoders.AUDIO.*``, ``blocks.N.*`` and
    ``proj.*``.  ``self.modality_encoders`` is a plain dict so the keys come out
    as ``modality_encoders.AUDIO.<...>`` exactly like the reference
    ``nn.ModuleDict``.
    
```

## emotion2vec.py :: preprocess_waveform

```
``source = F.layer_norm(source, source.shape)`` then ``view(1, -1)``.

    The reference normalises over the *entire* 1-D waveform (one global
    normalisation), not per frame.
    
```

## emotion2vec.py :: EmotionResult

```
Mirrors the keys FunASR's ``Emotion2vec.inference`` returns.
```

## emotion2vec.py :: contextualized_features

```
``contextualized_features(..., mask=False, remove_masked=False)``.

        Returns ``(x, alibi_bias, alibi_scale_for_model_blocks)``.  When the
        learned scale has a single row it is folded into ``alibi_bias`` here and
        the third element is ``None``; with one row **per layer** the rows stay
        live and the tail (``alibi_scale[prenet_depth:]``) is handed back so the
        shared blocks pick their own row, exactly like the reference.
        
```

## emotion2vec.py :: alibi_bias

```
``get_alibi_bias`` -> ``(B, heads, T, T)``.
```

## emotion2vec.py :: local_features

```
Waveform ``(B, samples)`` -> projected features ``(B, T, embed_dim)``.

        The conv stack returns ``NCT`` and ``project_features`` is the source's
        ``Sequential(TransposeLast, LayerNorm(512), Linear)``.  Its leading
        ``TransposeLast`` is a **real** transpose, so the ``NCT`` tensor is fed
        straight in and the norm correctly reduces the **512 channels**.
        
```

## emotion2vec.py :: _weight_dtype

```
Dtype of this encoder's checkpoint weights (first conv always exists).
```

## emotion2vec.py :: ffn_targets

```
``ffn_targets = not end_of_block_targets`` (``emotion2vec.py``).
```

## emotion2vec.py :: pos_kernel_size

```
``k = max(3, conv_pos_width // conv_pos_depth)`` (``audio.py:87``).
```

## emotion2vec.py :: Emotion2VecConfig

```
Validated mirror of the audio modality config plus the model fields.
```

## emotion2vec.py :: __call__

```
``modules.py:227-244`` with every dropout/drop-path removed.

        Note the post-norm branch feeds the **raw** ``x`` to the attention and
        only then applies ``norm1``; ``t`` is the pre-``norm2`` mlp output when
        ``ffn_targets`` and the final ``x`` otherwise.  ``t`` is training-only
        (it supervises the teacher) and is returned for API parity only.
        
```

## emotion2vec.py :: AltBlock

```
``modules.py:180-245`` ``AltBlock`` with ``layer_norm_first=false``.

    Post-norm ordering::

        x = x + attn(norm1(x))
        x = x + mlp(norm2(x))

    and the second return value is the post-norm ``x`` when ``ffn_targets`` else
    the pre-norm mlp output.  That value is training-only (it supervises the
    teacher), so the port returns it for API parity but never consumes it.
    
```

## emotion2vec.py :: AltAttention

```
``modules.py:255-300`` ``AltAttention`` (cosine attention is not used).
```

## emotion2vec.py :: __call__

```
``NTC`` in, ``NTC`` out -- the untouched source chain.

        The leading ``TransposeLast`` converts ``NTC -> NCT`` for the blocks (each
        ``NCT -> NCT``) and the trailing one restores ``NTC``.
        
```

## emotion2vec.py :: PositionalEncoder

```
``Sequential(TransposeLast, 5 x PositionalBlock, TransposeLast)``.

    With ``conv_pos_pre_ln=false`` (the large config) the conv's
    ``padding = k // 2 = 9`` preserves ``T`` and ``SamePad(19)`` is the identity
    on top of it, so the output length is always exactly ``T``.

    Time axis contract: the encoder takes ``NTC`` and returns ``NTC``; each
    :class:`PositionalBlock` takes ``NTC`` and returns ``NCT``, so the encoder
    restores ``NTC`` for the final output.  Each :class:`PositionalBlock` is
    ``NCT`` in and ``NCT`` out, so nothing is transposed *between* blocks.
    
```

## emotion2vec.py :: ConvFeatureExtraction

```
``ConvFeatureExtractionModel`` with ``mode="layer_norm"``.
```

## emotion2vec.py :: ConvFeatureBlock

```
``wav2vec2.py:839-860`` ``block(...)`` for ``is_layer_norm=True``.

    ``Conv1d -> Dropout -> Sequential(TransposeLast, Fp32LayerNorm(affine=True),
    TransposeLast) -> GELU``.  The double transpose turns the ``NCT`` conv
    output into ``NTC`` so the LayerNorm normalises the **512 channels**, and the
    norm is **affine**, i.e. its ``weight``/``bias`` are real checkpoint tensors.
    
```

## emotion2vec.py :: parse_feature_encoder_spec

```
Parse ``feature_encoder_spec`` with ``ast`` -- ``eval`` is never used.

    Accepts only literal arithmetic over tuples of ints, i.e. ``+``, ``*`` and
    ``[a, b, c]``.  Anything else (names, calls, attributes, comprehensions)
    raises.
    
```

## emotion2vec.py :: Mlp

```
``fairseq`` ``Mlp``: ``fc1 -> act -> fc2`` with dropout around.
```

## emotion2vec.py :: Gelu

```
``nn.GELU()`` with the default (exact, erf-based) approximation.
```

## emotion2vec.py :: Fp32LayerNorm

```
``fairseq.layers.fp32_group_norm.Fp32LayerNorm``.

    A ``nn.LayerNorm`` that upcasts to float32, normalises and casts back, so
    the reduction never happens in half precision.
    
```

## emotion2vec.py :: TransposeLast

```
``fairseq.modules.TransposeLast``: swap the last two axes.
```

## emotion2vec.py :: __getitem__

```
Fetch by position (``self[0]``) or by the mapping key (``self["0"]``).

        ``nn.Module.update`` / ``tree_unflatten`` hand back the *string* keys that
        were registered here, so a positional-only ``index < 0`` comparison raises
        ``TypeError: '<' not supported between instances of 'str' and 'int'``.
        
```

## emotion2vec.py :: ModuleListStack

```
``nn.ModuleList``; children registered under their decimal index.
```

## emotion2vec.py :: __getitem__

```
Fetch by Sequential position (``self[0]``) **or** by raw key (``self["0"]``).

        Both spellings are used: ``__call__`` walks ``_order`` positionally while
        :class:`PositionalBlock` needs named access so it can reorder the
        transposes around the channel-last conv.
        
```

## emotion2vec.py :: OrderedModules

```
An ``nn.Sequential`` whose children keep their integer index in the key.

    ``nn.Module`` is a dict subclass and ``tree_flatten`` walks ``.items()``, so
    registering a child under the *string* ``"2"`` reproduces PyTorch's
    ``...2.1.weight`` naming exactly.  A plain list would also be captured by
    ``Module.__setattr__`` and would pollute the key space, hence the tuple.
    
```

## emotion2vec.py :: get_alibi

```
``get_alibi(max_positions, attention_heads, dims=1)``.

    1-D manhattan ALiBi: ``slopes[h] * -|i - j|`` with shape
    ``(heads, max_positions, max_positions)``.
    
```

## fsmn_vad.py :: detect

```
Convenience wrapper for the offline case (auto-initialised cache).
```

## fsmn_vad.py :: __call__

```
``FsmnVADStreaming.forward``.

        ``feats`` is ``(1, T, input_dim)`` fbank (the frontend is owned by the
        shared runtime); ``waveform`` is the exact ``(1, N)`` (or ``(N,)``)
        sample span those ``T`` frames cover.
        
```

## fsmn_vad.py :: drop_cached_frames

```
``DropCachedFrames``: release decibel/score history before a frame.
```

## fsmn_vad.py :: reset

```
Full reset between utterances (``init_cache``).
```

## fsmn_vad.py :: init_cache

```
``init_cache``: fresh encoder caches, window detector and stats.
```

## fsmn_vad.py :: from_dict

```
Build from a FunASR ``config.yaml`` mapping.
```

## fsmn_vad.py :: FsmnVadModel

```
FSMN-VAD: streaming/offline speech endpointing.

    ``__call__(feats, waveform, cache, is_final=…, is_streaming_input=…)`` mirrors
    ``FsmnVADStreaming.forward`` and returns ``[[[start_ms, end_ms], …], …]``
    (batch x segment; batch size 1 is the only supported case, as in FunASR).
    
```

## fsmn_vad.py :: forward_with_caches

```
ONNX-export style: return ``(scores, [cache_layer_0, …])``.
```

## fsmn_vad.py :: FSMN

```
``FSMN`` encoder (``funasr/models/fsmn_vad_streaming/encoder.py``).

    ``cache=None`` runs the non-streaming path (zero left padding); passing a dict
    runs the streaming path and updates ``cache["cache_layer_<i>"]`` in place,
    exactly like the reference.
    
```

## fsmn_vad.py :: BasicBlock

```
``BasicBlock``: linear -> FSMN -> affine -> relu, with a per-layer cache.
```

## fsmn_vad.py :: __call__

```
Return ``(output, new_cache)`` with ``output`` shaped ``(B, T, D)``.

        Layout: ``mx.nn.Conv1d`` is channels-last ``(batch, length, channels)``.
        The reference works in ``(B, D, T)``; the time axis is moved to the middle
        so the ``(lorder - 1) * lstride`` left padding (or cache concatenation) is
        expressed on the same axis FunASR pads, then moved back for the conv.

        ``mx.core.conv1d`` reads ``in[t * stride + k * dilation]``; with
        ``y[i] = x[i - keep]`` kernel tap ``k`` therefore reads ``x[t - keep + k]``.
        
```

## fsmn_vad.py :: FSMNBlock

```
``FSMNBlock``: left/right dilated depthwise memory block.

    The reference applies ``nn.Conv2d`` to a ``(B, D, T, 1)`` view; here the time
    axis is the sequence axis of ``mx.conv1d`` and the trailing singleton axis is
    dropped (it carries no data).

    Cache shapes: ``cache`` is ``(B, D, (lorder - 1) * lstride)`` or ``None``.
    
```

## fsmn_vad.py :: AffineTransform

```
``AffineTransform``: ``self.linear = nn.Linear(input_dim, output_dim)``.

    Like :class:`LinearTransform`, this retains the named ``linear`` submodule;
    unlike it, the affine transform also has a bias.
    
```

## fsmn_vad.py :: LinearTransform

```
Bias-free wrapper retaining the reference's named ``linear`` child.

    ``BasicBlock.linear`` owns this wrapper, so its child adds the second
    ``linear`` in ``fsmn.<i>.linear.linear.weight``.
    
```

## fsmn_vad.py :: ModuleLayerList

```
``nn.ModuleList`` keyed by *string* index (see ``ct_punc.py`` for why).

    MLX's own ``ModuleList`` nests children under ``layers`` and would shift every
    checkpoint key; FunASR's ``FsmnStack`` (an ``nn.Sequential``) indexes blocks
    as ``fsmn.0.…``, ``fsmn.1.…``, so children must sit directly under ``"0"``,
    ``"1"``, …  ``nn.Module.update()`` can only overwrite existing keys, so
    children are inserted as dict items.
    
```

## fsmn_vad.py :: max_end_sil_frame_cnt_thresh

```
``Stats.max_end_sil_frame_cnt_thresh`` at cache-initialisation time.
```

## fsmn_vad.py :: VadOptions

```
Mutable mirror of FunASR's ``VADXOptions``.

    Mutable because ``ComputeScores`` overwrites ``nn_eval_block_size`` with the
    number of frames in the current block, exactly like the reference.
    
```

## fsmn_vad.py :: cache_frames

```
Left-context cache length ``(lorder - 1) * lstride``.
```

## fsmn_vad.py :: FsmnEncoderConfig

```
Validated mirror of ``encoder_conf`` for FunASR's ``FSMN`` network.
```

## fsmn_vad.py :: convert_checkpoint_tensors

```
Depthwise 4-D kernels transposed; strict elsewhere (see _strict_load).
```

## fun_asr_nano.py :: nano_config_from_nano_yaml

```
Build the config from the official Fun-ASR-Nano ``config.yaml``.

    Reads ``audio_encoder_conf`` / ``audio_adaptor_conf`` and pins the
    Qwen3-0.6B ``llm_conf.args`` (the architecture of the shipped ``llm.*``
    subtree, verified against the checkpoint).
    
```

## fun_asr_nano.py :: embed_with_speech

```
Token embeddings with the speech span spliced in, then one LLM pass.
```

## fun_asr_nano.py :: encode_audio

```
``(B, T, 560)`` LFR fbank -> adaptor embeddings ``(B, T', llm_dim)``.
```

## fun_asr_nano.py :: FunASRNano

```
``FunASRNano`` (inference): encode -> adapt -> LLM decode.
```

## fun_asr_nano.py :: StandardMHA

```
ESPnet ``MultiHeadedAttention`` (transformer/attention.py): separate
    ``linear_q/k/v`` + ``linear_out`` and NO fsmn memory -- the adaptor
    checkpoint keys (``blocks.N.self_attn.linear_q...``) are exactly this
    module, distinct from the SANM attention of the speech encoders.
```

## fun_asr_nano.py :: TransformerAdaptor

```
``funasr.models.llm_asr.adaptor.Transformer`` (inference).

    Stacks ``k`` frames (k=1 here), projects encoder_dim -> llm_dim through
    an FFN, then runs standard encoder layers. The reference layers are
    ESPnet-style ``EncoderLayer`` (post-norm when normalize_before=False,
    which is the adaptor default: EncoderLayer's default normalize_before
    is False) with ``MultiHeadedAttention`` and ``PositionwiseFeedForward``.
    
```

## glm_asr.py :: __call__

```
mel: ``(F, T)``; returns ``(1, T'', output_size)`` embeddings.

        The official projector consumes the encoder output RE-GROUPED into
        ``intermediate_size``-wide rows: every 4 frames of 1280 become one
        5120-wide row (a 4-frame merge), then linear_1 5120->4096 + gelu +
        linear_2 4096->2048.
        
```

## glm_asr.py :: GlmAsrAudioEncoder

```
``glmasr_encoder`` (inference): mel (F, T) -> (1, T', output_size).
```

## glm_asr_thinker.py :: encode_audio

```
mel (F, T) -> (1, N, text_hidden) embeddings for the pad run.

        The mel is cast to the tower's weight dtype, as whisper's AudioEncoder
        and qwen3-asr's tower do.  This cast ALONE measures 1.00x here (see
        .tmp/task4/tower-tight-*.json and the re-measurement in
        .tmp/t7/glm-isolate.json), which is why it was originally skipped -- but
        that A/B was misleading: the tower's partial-RoPE tables were float32,
        so attention re-promoted the stream to float32 regardless.  With the
        tables also following the activation dtype (see ``_Attn._rope``) the
        two casts together give 1.44x on the tower and 28.7% on the pipeline.
        
```

## glm_asr_thinker.py :: _tower_dtype

```
Dtype of the audio tower's checkpoint weights (first conv).
```

## glm_asr_thinker.py :: _decoder_dtype

```
Dtype the decoder activations must follow.

        Read through :func:`activation_dtype`, not ``weight.dtype``: a quantized
        embedding stores packed ``uint32`` codes in ``weight``, and casting the
        spliced audio embeddings to ``uint32`` destroys them.
        
```

## glm_asr_thinker.py :: GlmAsrThinker

```
audio_tower (with projector folded in) + Llama causal LM.
```

## moss_transcribe.py :: forward_with_audio

```
Replace the audio-token run with adaptor output (shared splice).
```

## moss_transcribe.py :: _decoder_dtype

```
Dtype the decoder activations must follow.

        Read through :func:`activation_dtype`, not ``weight.dtype``: a quantized
        embedding stores packed ``uint32`` codes in ``weight``, and casting the
        spliced audio embeddings to ``uint32`` destroys them.
        
```

## moss_transcribe.py :: encode_audio_from_hidden

```
Merged/chunked whisper output (1, T, 1024) -> adaptor (1, T/4, hidden).

        Trims to a multiple of the merge size first (official time_merge).
        
```

## moss_transcribe.py :: encode_audio

```
mel (F, T) -> whisper (1, T/2, 1024) -> adaptor (1, T/8, hidden).
```

## moss_transcribe.py :: MossTranscribeDiarize

```
whisper_encoder + vq_adaptor + qwen3 (audio-token splice).
```

## moss_transcribe.py :: VQAdaptor

```
Linear(input_dim->hidden) + SiLU + Linear(hidden->hidden) + LN(1e-6).
```

## moss_transcribe.py :: HfWhisperEncoder

```
HF ``WhisperEncoder`` (inference): mel (F, T) -> (1, T/2, d).
```

## moss_transcribe.py :: _HfWhisperAttn

```
HF ``WhisperAttention``: q/v/out carry biases, **k does not**.

    ``k_proj`` is bias-free in the reference (``nn.Linear(embed_dim,
    embed_dim, bias=False)`` in transformers' ``modeling_whisper.py``), and the
    checkpoint agrees: ``...self_attn.k_proj.bias`` is absent from all 24
    layers.  Building it with ``bias=True`` therefore left a random,
    never-trained key bias in the audio tower -- measured on the previous
    release: exists=True, absmax 0.0312, absmean 0.0158.  The old permissive
    loaders never noticed, which is why MOSS tower parity sat at 1.3e-3.
    
```

## paraformer.py :: _cif_fire_positions

```
Fallback used when the CIF peaks do not line up with the token count.

    Mirrors ``funasr.models.paraformer.cif_predictor.cif_wo_hidden``: a frame
    "fires" when the running integral crosses an integer boundary.
    
```

## paraformer.py :: ts_prediction_lfr6_standard

```
Port of ``funasr.utils.timestamp_tools.ts_prediction_lfr6_standard``.

    Only the millisecond ``res`` output of the reference is produced here (the
    text dump and the ``<sil>`` bookkeeping are dropped).  ``upsample_rate`` must
    be ``1`` to match Paraformer's inference call site.
    
```

## paraformer.py :: timestamps

```
CIF-peak based character timestamps in milliseconds.

        Returns ``(tokens, timestamps)`` where ``timestamps[i]`` is a list of
        ``[start_ms, end_ms]`` pairs aligned with ``tokens[i]``.
        
```

## paraformer.py :: greedy_decode

```
Greedy token ids per utterance.

        Mirrors ``Paraformer.inference`` without beam search: ``argmax`` over the
        decoder posteriors followed by dropping ``sos``/``eos``/``blank_id``
        (the reference wraps the sequence in sos/eos and then strips it).
        
```

## paraformer.py :: load_checkpoint

```
Strict load; see :func:`funasr_mlx.models.sanm.load_strict`.
```

## paraformer.py :: state_dict

```
``{pytorch_name: mlx_array}`` in MLX layout.
```

## paraformer.py :: Paraformer

```
``funasr.models.paraformer.model.Paraformer`` (inference only).
```

## paraformer.py :: __call__

```
``hs_pad`` is the encoder/acoustic memory ``(B, T1, D)``.

        Following the reference (``funasr/models/paraformer/decoder.py:397-450``)
        the offline path does **not** run ``self.embed``: ``x = tgt`` with
        ``tgt = ys_in_pad``.  Only ``forward_one_step`` (``:604``) embeds.  In
        offline Paraformer the decoder input is therefore the CIF acoustic
        embedding, already ``(B, T2, D)`` -- not token ids.
        
```

## paraformer.py :: ParaformerSANMDecoder

```
``funasr.models.paraformer.decoder.ParaformerSANMDecoder`` (offline path).

    Three stacks: ``decoders`` (``att_layer_num`` layers with SANM self-attention
    *and* cross-attention), optional ``decoders2`` (self-attention only) and a
    single ``decoders3`` feed-forward-only layer.
    
```

## paraformer.py :: __call__

```
Return ``(acoustic_embeds, token_num, alphas, cif_peak)``.
```

## paraformer.py :: CifPredictorV2

```
``funasr.models.paraformer.cif_predictor.CifPredictorV2`` (inference path).

    Parameters: ``cif_conv1d`` (a **dense**, i.e. ``groups=1``, ``Conv1d`` in V2)
    and ``cif_output``.  ``__call__`` returns the 4-tuple
    ``(acoustic_embeds, token_num, alphas, cif_peak)`` of the reference.
    
```

## paraformer.py :: _gather

```
Gather time steps of ``array`` in fire order, independently per row.

        ``mx.take_along_axis`` indexes the time axis directly, so no flatten,
        offset arithmetic or output reshape is involved.  Per the installed stub
        (``mlx/core/__init__.pyi:1901-1909``) ``indices`` must be "broadcastable
        with the input array excluding the ``axis`` dimension", so a trailing
        singleton axis broadcasts one time index across every feature lane and
        the result keeps ``(B, T, D)``.
        
```

## paraformer.py :: SANMEncoder

```
``funasr.models.sanm.encoder.SANMEncoder`` (offline inference path).

    Differences from the reference that do not affect checkpoint parity:
    intermediate CTC / ``interctc_layer_idx`` outputs and streaming
    ``forward_chunk`` are not ported (both are training / streaming features).
    
```

## paraformer.py :: CTC

```
``funasr.models.ctc.ctc.CTC`` inference path (``ctc_lo`` projection).
```

## paraformer.py :: paraformer_config_from_dict

```
Build a :class:`ParaformerConfig` from a FunASR ``model.yaml`` mapping.
```

## paraformer.py :: ParaformerConfig

```
Mirror of FunASR's ``model_conf`` for ``Paraformer``.

    Defaults follow the published ``paraformer-zh`` configuration
    (``input_size=560`` LFR fbank, 50 SANM blocks of width 512, 16 decoder
    blocks, CIF ``tail_threshold=0.45``).
    
```

## paraformer.py :: CTCConfig

```
Mirror of FunASR's ``ctc_conf`` (inference-relevant fields only).
```

## paraformer.py :: CifPredictorConfig

```
Mirror of FunASR's ``predictor_conf`` for ``CifPredictorV2``.
```

## paraformer.py :: ParaformerDecoderConfig

```
Mirror of FunASR's ``decoder_conf`` for ``ParaformerSANMDecoder``.
```

## paraformer.py :: SANMEncoderConfig

```
Mirror of FunASR's ``encoder_conf`` for ``SANMEncoder`` (Paraformer side).
```

## paraformer_streaming.py :: streaming_config_from_dict

```
Build the streaming config from the official online yaml mapping.
```

## paraformer_streaming.py :: forward_chunk

```
One chunk end-to-end. Returns ``(am_scores, token_length)``.

        ``None`` when the chunk fired no tokens (mirrors ``generate_chunk``
        returning ``[]`` when ``max(pre_token_length) < 1``).
        
```

## paraformer_streaming.py :: decoder_chunk

```
``cal_decoder_with_predictor_chunk`` -> decoder ``forward_chunk``.
```

## paraformer_streaming.py :: predictor_chunk

```
``calc_predictor_chunk`` -> ``CifPredictorV2.forward_chunk``.
```

## paraformer_streaming.py :: init_cache

```
``ParaformerStreaming.init_cache`` (model.py:511-552), MLX arrays.
```

## paraformer_streaming.py :: ParaformerStreaming

```
``ParaformerStreaming`` -- chunk-wise streaming Paraformer (MLX).

    Reuses the offline module tree (identical parameter names); adds
    ``init_cache`` / ``forward_chunk`` implementing the reference streaming
    protocol. ``generate`` feeds fixed 600 ms chunks ([0,10,5] x 960 samples).
    
```

## paraformer_streaming.py :: _cross_attn_forward_chunk

```
``MultiHeadedAttentionCrossAtt.forward_chunk`` (attention.py:815+ for cross).

    Cross-attention to the (growing) encoder memory with the same tail-window
    cache discipline as the decoder self-attention.
    
```

## paraformer_streaming.py :: _predictor_forward_chunk

```
``CifPredictorV2.forward_chunk`` (``cif_predictor.py:316-412``).

    Returns ``(acoustic_embeds, token_length, alphas, fires)`` with the carry
    state written back into ``cache``.
    
```

## paraformer_streaming.py :: _attn_forward_chunk

```
``MultiHeadedAttentionSANM.forward_chunk`` on the shared offline class.

    Mirrors ``sanm/attention.py:329-366``: compute qkv for the (stitched)
    chunk, merge cached ``k``/``v`` (look-back), commit each chunk's
    right-context-free stride into the cache, attend, add the FSMN memory.
    
```

## paraformer_streaming.py :: StreamSinusoidalPositionEncoder

```
``transformer/embedding.py::StreamSinusoidalPositionEncoder``.

    Stateful through ``cache['start_idx']``: each chunk's positions continue
    from where the previous chunk ended, which is what makes the streamed
    encoding equal the offline one.
    
```

## qwen3_asr.py :: Qwen3ASRAudioEncoder

```
``Qwen3ASRAudioEncoder`` (inference), single-utterance path.

    Input mel: ``(F, T)`` (n_mels first, like the reference). Output:
    ``(1, T', output_dim)`` -- one embedding row per audio token slot,
    which the thinker splices between ``audio_start`` and ``audio_end``.
    
```

## qwen3_asr.py :: _Conv2dStride2

```
``nn.Conv2d(h, h, 3, stride=2, padding=1)`` in MLX (NHWC).
```

## qwen3_asr_thinker.py :: encode_audio

```
``(F, T)`` mel -> ``(1, T', text.hidden_size)`` tower embeddings.

        The front-end always produces float32 mel, so the mel is cast to the
        tower's **weight** dtype first (same rule as whisper's
        ``AudioEncoder.__call__``); without it an fp16 checkpoint runs the
        conv stack and every tower Linear in mixed precision.  Interleaved
        A/B on M4 (25 rounds, .tmp/task4/tower-tight-*.json): 55.7 -> 44.0 ms
        at the real 420-frame shape and 273.9 -> 214.7 ms at 30 s, i.e. 1.27x
        at both dtypes.  For a float32 checkpoint the cast is a no-op, so
        those numerics stay bit-identical.
        
```

## qwen3_asr_thinker.py :: _tower_dtype

```
Dtype of the audio tower's checkpoint weights (first conv).
```

## qwen3_asr_thinker.py :: _decoder_dtype

```
Dtype the decoder activations must follow.

        The token embedding produces the activations the decoder consumes, so
        its OUTPUT dtype is the one to cast spliced audio to.  Read through
        :func:`activation_dtype`, not ``weight.dtype``: a quantized embedding
        stores packed ``uint32`` codes in ``weight``, and casting the audio
        embeddings to ``uint32`` destroys them (see the helper's docstring).
        
```

## sanm.py :: Conv2dSubsampling2

```
``funasr.models.transformer.utils.subsampling.Conv2dSubsampling2`` (1/2).
```

## sanm.py :: Conv2dSubsampling

```
``funasr.models.transformer.utils.subsampling.Conv2dSubsampling`` (1/4).

    Two strided 3x3 convolutions over the ``(B, T, F, 1)`` view followed by a
    linear projection.  Parameter keys stay ``conv.<i>.weight`` and
    ``out.0.weight`` thanks to :class:`ModuleStack`.

    MLX convolutions are channels-last, so the channel axis comes **last**
    (``(B, T, F, C)``).  The flatten before ``out.0`` is transposed back to
    PyTorch's ``(C, F)`` row-major order, which is what ``out.0`` was trained on.
    
```

## sanm.py :: check_short_utt

```
Port of ``funasr.models.transformer.utils.subsampling.check_short_utt``.
```

## sanm.py :: TooShortUttError

```
Port of ``funasr.models.transformer.utils.subsampling.TooShortUttError``.
```

## sanm.py :: DecoderLayerSANM

```
``funasr.models.paraformer.decoder.DecoderLayerSANM``.

    Note the ordering: the feed-forward runs **first** and the self-attention
    residual is taken against the block *input* (``residual``), not against the
    feed-forward output.  ``src_attn`` may be ``None`` (the ``decoders2`` stack).
    
```

## sanm.py :: EncoderLayerSANM

```
``funasr.models.sanm.encoder.EncoderLayerSANM`` (offline inference path).

    The first layer (``encoders0``) has ``in_size != size``; in that case the
    self-attention residual is dropped exactly like the reference does.
    
```

## sanm.py :: MultiHeadedAttentionCrossAtt

```
``funasr.models.sanm.attention.MultiHeadedAttentionCrossAtt``.
```

## sanm.py :: __call__

```
``inputs`` is ``(B, T, D)`` (MLX NLC); ``cache`` mirrors the reference.
```

## sanm.py :: MultiHeadedAttentionSANMDecoder

```
``funasr.models.sanm.attention.MultiHeadedAttentionSANMDecoder``.

    Pure FSMN memory block over the decoder input.  ``cache`` mirrors the
    reference streaming state; offline inference passes ``None``.

    Deviation: the cache is kept channels-last ``(b, L, d)`` whereas the
    reference keeps ``(b, d, L)``.  See the module docstring's "Known
    deviations" section -- this matters only for the streaming/export contract,
    not for offline inference.
    
```

## sanm.py :: _split_heads

```
``(B, T, n_feat)`` -> ``(B, head, T, d_k)``.
```

## sanm.py :: MultiHeadedAttentionSANM

```
``funasr.models.sanm.attention.MultiHeadedAttentionSANM``.

    Fused QKV projection (``linear_q_k_v``) plus a depth-wise FSMN memory block
    whose output is added to the attention output.  The reference applies
    ``d_k ** -0.5`` to the queries *after* splitting heads (equivalent to
    scaling the scores by ``1/sqrt(d_k)``).
    
```

## sanm.py :: PositionwiseFeedForwardDecoderSANM

```
``funasr.models.sanm.positionwise_feed_forward.PositionwiseFeedForwardDecoderSANM``.

    Two details matter for checkpoint compatibility: ``w_2`` has **no bias** and
    the ``norm`` sits *between* the two linears and uses ``eps=1e-12``.
    
```

## sanm.py :: PositionwiseFeedForward

```
``funasr.models.transformer.positionwise_feed_forward.PositionwiseFeedForward``.
```

## sanm.py :: SinusoidalPositionEncoder

```
``funasr.models.transformer.embedding.SinusoidalPositionEncoder``.

    Note the reference divides ``log(10000)`` by ``depth / 2 - 1`` (not by
    ``depth / 2``), i.e. positions start at ``1`` and the wavelength exponent
    denominator is ``depth / 2 - 1``.  Both quirks are preserved.
    
```

## sanm.py :: Float32LayerNorm

```
``funasr.models.sense_voice.model.LayerNorm`` (eps ``1e-5``, fp32 math).

    SenseVoice keeps its own ``LayerNorm`` that inherits the ``torch`` default
    ``eps=1e-5`` and forces the reduction to float32 before casting the result
    back to the input dtype.  Reproducing the explicit cast keeps the fp16 path
    numerically identical to the reference.
    
```

## sanm.py :: LayerNorm

```
``funasr.models.transformer.layer_norm.LayerNorm`` (eps ``1e-12``).

    Parameter names are ``weight``/``bias`` and the normalisation axis is the
    last one, exactly like the PyTorch original.  MLX performs the reduction in
    float32 internally, which matches ``torch.nn.LayerNorm`` for float32 inputs.
    
```

## sanm.py :: repeat

```
Port of ``funasr.models.transformer.utils.repeat.repeat``.
```

## sanm.py :: __call__

```
Chain blocks with canonical ``nn.Sequential`` semantics.

        The leading argument is threaded through the chain while the remaining
        positional arguments and the keywords stay fixed.  Feeding a block's
        *return value* back as ``*args`` instead would unpack an ``mx.array``
        along axis 0 and hand the next block ``B`` separate arguments, which
        only works at batch size 1 or when the arities happen to line up.
        
```

## sanm.py :: checkpoint_state_dict

```
Return the module's parameters as ``{pytorch_name: mlx_array}``.

    The names are byte-compatible with the FunASR ``.pt`` checkpoints; the
    arrays are already in MLX layout.
    
```

## sanm.py :: flatten_state_dict_with_rules

```
Like :func:`flatten_state_dict`, but also returns the per-key layout rule.
```

## sanm.py :: flatten_state_dict

```
Flatten a module tree into FunASR/PyTorch-compatible parameter names.

    Names are built from the Module's **mapping storage** (see
    :func:`_flatten_into`), which is the only place MLX can hold a tensor.  The
    result matches what ``torch.nn.Module.state_dict()`` produces for the
    corresponding PyTorch modules.
    
```

## sanm.py :: convert_checkpoint_tensor

```
Convert one PyTorch checkpoint tensor into the MLX layout (shared kernel).
```

## sanm.py :: decimate_mask

```
Port of the ``x_mask[:, :, :-2:2][:, :, :-2:2]`` subsampling masks.

    ``torch`` slices with an explicit stop index (``[:-2]`` stops at
    ``length - 2``); MLX negative-stop semantics are easy to get subtly wrong,
    so the equivalent index vector is built explicitly and applied with
    :func:`mx.take`.  ``stride=2`` models ``[:-2:2]`` and ``stride=1`` models
    ``[:-2:1]``; both stop at ``length - end_offset``.
    
```

## sanm.py :: constant_pad_1d

```
``torch.nn.ConstantPad1d`` port.

    MLX convolutions are **channels-last**: ``mx.conv1d`` consumes ``NLC`` and
    ``mx.conv2d`` consumes ``NHWC``.  The reference SANM modules therefore pad
    the *time* axis of an ``NLC`` tensor, which is ``axis=1`` -- not the last
    axis as it is for PyTorch's ``NCL`` tensors.  ``axis`` is explicit so the
    intent is visible at every call site.
    
```

## sanm.py :: make_pad_mask

```
Port of ``funasr.models.transformer.utils.nets_utils.make_pad_mask``.

    Returns a boolean ``(batch, maxlen)`` array that is ``True`` on padded
    positions (i.e. the logical inverse of :func:`sequence_mask`).
    
```

## sanm.py :: sequence_mask

```
Port of ``funasr.models.scama.utils.sequence_mask``.

    Returns a ``(batch, maxlen)`` array holding ``1.0`` on valid positions and
    ``0.0`` on padded ones.  ``lengths`` may be ``(B,)`` or ``(B, 1)``.
    
```

## sensevoice.py :: ctc_greedy_decode

```
``argmax`` -> ``unique_consecutive`` -> drop ``blank_id``.

    Faithful to ``SenseVoiceSmall.inference``: consecutive duplicates collapse
    *before* blanks are removed, so ``[a, blank, a]`` decodes to ``[a, a]``.
    
```

## sensevoice.py :: greedy_decode

```
Greedy CTC token ids per utterance (mirrors ``SenseVoiceSmall.inference``).
```

## sensevoice.py :: load_checkpoint

```
Strict load; see :func:`funasr_mlx.models.sanm.load_strict`.
```

## sensevoice.py :: state_dict

```
``{pytorch_name: mlx_array}`` in MLX layout (no key rewriting).
```

## sensevoice.py :: SenseVoiceSmall

```
``funasr.models.sense_voice.model.SenseVoiceSmall`` (inference only).
```

## sensevoice.py :: _weight_dtype

```
Dtype of this encoder's checkpoint weights (``encoders0`` always exists).

        Read through :func:`~funasr_mlx.models.llm.quant.activation_dtype` rather
        than ``weight.dtype`` directly: the SANM stack is a documented
        quantization target (``audio_encoder.`` is in fun-asr-nano's scope), and
        a quantized Linear's ``weight`` holds packed ``uint32`` codes.  Casting
        the fbank input to ``uint32`` would destroy it exactly as the decoder
        embedding cast did.  The helper returns the side tensor's dtype for a
        quantized module, so this stays correct either way.
        
```

## sensevoice.py :: CTC

```
``funasr.models.ctc.ctc.CTC`` restricted to its inference path.

    The only parameter is ``ctc_lo`` (``Linear(encoder_output_size, odim)``);
    the loss/argmax helpers of the reference are re-implemented on top of it.
    
```

## sensevoice.py :: sensevoice_config_from_dict

```
Build a :class:`SenseVoiceSmallConfig` from a FunASR ``model.yaml`` mapping.

    Accepts either the flat ``model_conf`` block or the whole document with
    ``model_conf``/``encoder_conf``/``ctc_conf`` sections.
    
```

## sensevoice.py :: SenseVoiceSmallConfig

```
Mirror of FunASR's ``model_conf`` for ``SenseVoiceSmall``.

    The defaults are the published ``SenseVoiceSmall`` values
    (``input_size=560``, 50 SANM blocks + 20 tail blocks, ``vocab_size=55465``).
    
```

## sensevoice.py :: CTCConfig

```
Mirror of FunASR's ``ctc_conf`` (only the inference-relevant fields).
```

## sensevoice.py :: SenseVoiceEncoderSmallConfig

```
Mirror of FunASR's ``encoder_conf`` for ``SenseVoiceEncoderSmall``.
```

## whisper.py :: _suppress

```
Set the given vocabulary columns to ``-inf`` (a port of the logit filters).

    ``vocab`` (V,) is broadcast **against** the banned ids (N,) rather than the
    other way round, and the membership test is reduced over the N axis.  Building
    a ``(1, V)`` and a ``(1, N)`` array and comparing them would only broadcast
    when ``N == 1`` or ``N == V``; with two or more suppressed tokens and a
    vocabulary of a different size it raises instead of masking.
    
```

## whisper.py :: GreedyDecodeResult

```
Token-level greedy output (ids only; decoding to text is not provided).
```

## whisper.py :: GreedyDecodeConfig

```
Bounds and logit filters for :func:`greedy_decode`.

    ``eot_token_id`` and ``space_token_id`` **must** be supplied by the caller
    (they depend on the BPE vocabulary of the particular checkpoint) or left as
    ``None`` to disable the corresponding filter.  No default is guessed.
    
```

## whisper.py :: mel_filterbank

```
Slaney-normalised mel filterbank, shape ``(n_mels, n_fft // 2 + 1)``.

    Re-implementation of ``librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)``,
    which is how the reference's ``mel_filters.npz`` asset was produced.  **Not
    numerically verified against the official asset** -- if the official array is
    available, load it and pass it to :func:`log_mel_spectrogram` directly instead
    of relying on this function.
    
```

## whisper.py :: _is_non_persistent_buffer

```
Whether ``owner``'s parameter ``key`` is a derived, non-persistent buffer.

    ``isinstance`` rather than an exact type match, so a subclass still matches.
    The result is independent of any flatten prefix or of whether the owning
    module is reached standalone or as part of a full ``Whisper``.
    
```

## whisper.py :: _hz_to_mel

```
Slaney mel scale (librosa default, ``htk=False``).
```

## whisper.py :: pad_or_trim

```
Port of ``whisper.audio.pad_or_trim`` (zero-pad or truncate).
```

## whisper.py :: reflect_pad_1d

```
``torch.nn.functional.pad(..., mode="reflect")`` along the last axis.

    The edge value is **not** repeated.  With ``x = [1, 2, 3, 4]`` and ``pad=2``
    on both sides the result is ``[3, 2, 1, 2, 3, 4, 3, 2]``: the left filler
    reads left-to-right as ``x[pad] ... x[1]`` and the right filler as
    ``x[n-2] ... x[n-1-pad]`` -- both *descending* index runs.
    
```

## whisper.py :: embed_audio

```
Encode ``(B, 2 * n_audio_ctx, n_mels)`` mel frames.

        The encoder casts the mel to its own weight dtype (see
        :meth:`AudioEncoder.__call__`), so an fp16 checkpoint produces fp16
        audio features even though the front-end mel is float32.
        
```

## whisper.py :: offset

```
Absolute position of the next token, from the self-attention cache.

        The reference reads ``next(iter(kv_cache.values())).shape[1]``, which
        relies on ``dict`` insertion order.  Here it is read deterministically
        from the first block's self-attention key.
        
```

## whisper.py :: _weight_dtype

```
Dtype of this module's checkpoint weights (``ln`` always exists).
```

## whisper.py :: TextDecoder

```
``whisper.model.TextDecoder`` with an explicit MLX KV cache.

    The causal mask is **not** a parameter: the reference registers it with
    ``persistent=False``, so it never appears in a ``state_dict`` and is
    regenerated on demand here.
    
```

## whisper.py :: _weight_dtype

```
Dtype of this module's checkpoint weights (``conv1`` always exists).
```

## whisper.py :: AudioEncoder

```
``whisper.model.AudioEncoder``.

    ``mel`` is channels-last ``(B, T', n_mels)``; the reference's
    ``permute(0, 2, 1)`` after ``conv2`` is unnecessary under MLX's channels-last
    Conv1d.  ``positional_embedding`` is materialised (the reference registers it
    as a persistent buffer) so it round-trips through strict loading; the
    regenerated value is identical to :func:`sinusoids`.
    
```

## whisper.py :: whisper_config_from_dict

```
Build a :class:`WhisperConfig` from a FunASR ``model.yaml``-like mapping.

    ``WhisperParameters`` is also accepted as an alias for the ``parameters``
    section of FunASR's document.
    
```

## whisper.py :: WhisperConfig

```
Mirror of ``funasr/models/whisper/model.py::Whisper``'s inference config.
```

## whisper.py :: ModelDimensions

```
Mirror of ``whisper.model.ModelDimensions``.
```

## whisper.py :: qkv_attention

```
Port of ``MultiHeadAttention.qkv_attention`` (the non-SDPA path).

        Layouts follow the reference exactly: ``q`` is ``(B, H, T, d)`` and
        ``k`` is **transposed** to ``(B, H, d, T)``.  That ``.permute(0, 2, 3, 1)``
        in the reference is what makes ``q @ k`` contract the head dimension and
        produce the ``(B, H, T, T)`` score matrix; leaving ``k`` as
        ``(B, H, T, d)`` would contract ``T`` with ``T`` instead.

        Whisper applies ``(n_state // n_head) ** -0.25`` to *both* ``q`` and
        ``k``, which is exactly a ``d_k ** -0.5`` factor on the scores.
        
```

## whisper.py :: _split

```
``(B, T, C)`` -> ``(B, H, T, d)``.
```

## whisper.py :: cache_prefix

```
Cache key prefix for this attention (e.g. ``"blocks.0.attn"``).
```

## whisper.py :: sinusoids

```
Port of ``whisper.model.sinusoids``.

    Note this is **0-based** and uses ``-log_increment * arange(channels // 2)``
    -- different from FunASR's ``SinusoidalPositionEncoder`` (1-based, positive
    increments), which the SANM blocks use.  Both quirks matter and both are
    preserved exactly.
    
```

## whisper.py :: LayerNorm

```
``whisper.model.LayerNorm`` -- fp32 reduction, torch's default ``eps=1e-5``.

    The reference casts the input to ``float32`` before normalising and casts
    the result back to the input dtype; MLX's ``LayerNorm`` already accumulates
    in float32, so the explicit round-trip is kept for clarity and for the fp16
    path.
    
```

## whisper.py :: load_strict

```
Load ``tensors`` into ``module``, refusing any key mismatch.

    Arrays must already be in MLX layout unless ``convert=True`` (raw PyTorch
    ``state_dict``).  Raises :class:`CheckpointKeyError` listing the offending
    keys, and ``ValueError`` on any shape mismatch -- never a partial load.
    
```

## whisper.py :: LoadReport

```
Outcome of a :func:`load_strict` call.
```

## whisper.py :: flatten_state_dict_with_rules

```
Like :func:`flatten_state_dict`, but also returns the per-key layout rule.

    Returns ``(tensors, rules)`` with the same two dictionaries that the
    :class:`LayoutReport` carries -- nothing is dropped and the rule metadata is
    not recomputed by a second walk.  This is the contract
    :func:`funasr_mlx.models.sanm.flatten_state_dict_with_rules` and
    :func:`funasr_mlx.models.campplus.flatten_state_dict_with_rules` expose, so
    the three model modules agree on one name for the same operation; callers
    that want the richer object should use :func:`flatten_state_dict` instead.
    
```

## whisper.py :: flatten_state_dict

```
Flatten a module tree into FunASR/PyTorch-compatible parameter names.

    The layout rule is derived from the owning module type rather than from a
    name pattern, so it cannot silently mis-convert a renamed submodule.
    Attributes starting with ``_`` are configuration, never weights.
    
```

## whisper.py :: LayoutReport

```
Flat parameter tensors plus the layout rule that applies to each key.

    Behaves like a read-only ``Mapping`` of ``{parameter_name: mlx_array}``, so it
    can be iterated, indexed, ``len()``-ed and fed to ``dict(...)`` exactly like a
    plain state dict, while also carrying the per-key rule used for the
    torch -> MLX layout conversion.
    
```

## whisper.py :: _rule_for

```
Conv1d is the only module whose MLX layout differs from PyTorch's.
```

## whisper.py :: ModuleStack

```
Ordered children registered under their numeric index.

    ``torch.nn.Sequential`` yields ``<parent>.<i>.<leaf>`` parameter names; MLX's
    ``nn.Sequential`` would insert a ``layers`` level, so children are registered
    directly under ``"0"``, ``"1"``, ...
    
```

## awq.py :: apply_scale

```
Fold the chosen scales into the site's modules (bf16 stays bf16).

        Two fold shapes, exactly as mlx-lm: a Linear predecessor divides its
        OUTPUT-channel rows (``weight / scales[:, None]`` -- the scales index
        the successor's input channels, which are the predecessor's outputs),
        while a norm predecessor divides elementwise (its weight is a vector
        over the same channels it normalises).
        
```

## awq.py :: search_scale

```
Grid-search channel scales for one site; returns the best scales.

        The reference output is the block (or sub-block) run on the captured
        input of the site's FIRST layer; each candidate scales that input's
        channels via the successor weights only (equivalent math, see
        upstream), quantize-dequantizes them, and the MSE of the sub-block's
        output picks the winner.
        
```

## awq.py :: _run_block

```
Run one decoder block on ``x`` [1, rows, hidden] in row batches.

        Our ``Qwen3TransformerBlock`` has no ``__call__`` of its own (the
        forward is unrolled in ``Qwen3Model.__call__``), so the qwen3 residual
        structure is applied here: ``h += attn(norm(h)); h += mlp(norm(h))``.
        The site's sub-block (``self_attn`` / ``mlp``) IS callable directly.
        
```

## awq.py :: capture

```
Run ``run()`` with input hooks on this block's Linears.

        ``run`` is a callable that performs the model's real prefill for one
        utterance; the block must be part of the live model.  Each Linear's
        input rows are appended to ``input_feat`` on the module (mlx-lm's
        convention), in float32, and the block's own input is recorded on
        ``self.block_input`` for the never-worse reference.
        
```

## awq.py :: MultiBlockCapture

```
Capture Linear inputs for MANY blocks in ONE pass over the dataset.

    Running the model once per block (28 x 480 prefills) is what made the
    first calibration attempt take ~575s per block; mlx-lm's AWQ catches every
    Linear of every block in the same forward.  Memory stays bounded by
    stride-subsampling each Linear's rows down towards ``row_cap`` after
    every utterance (halving whenever over the cap) and storing float16.
    
```

## awq.py :: _to_mx

```
Captured activations are numpy (host); the search wants mx arrays.

    The conversion happens once per consumer, so the device copy never
    exceeds the capped rows either.
    
```

## awq.py :: ScaleSite

```
One scale search site: a norm (or Linear) feeding several Linears.

    Mirrors mlx-lm's ``ScaleConfig`` for the Qwen3 block layout:

    * ``prev`` -- the module whose OUTPUT is scaled (an RMSNorm for qwen3);
      its weight is divided by the scales.
    * ``layers`` -- the successor Linear weights are multiplied by the scales.
    * ``block`` -- optional sub-block to inspect (``self_attn``), because the
      loss must be measured on the attention output, not one Linear alone.
    
```

## calibrate.py :: resolve

```
Locate the head on ``model`` and describe how it is called.

        ``head_path`` is a dotted path to the head module (e.g.
        ``thinker.model.embed_tokens`` or ``language_model.lm_head``).
        
```

## calibrate.py :: HeadTarget

```
Which module in a live model is the LM head, and how it is called.

    ``kind`` is ``"tied"`` when the head is the token embedding used through
    ``as_linear`` (qwen3 / moss) and ``"linear"`` when the checkpoint has a
    separate ``lm_head`` (glm).  ``hidden_size`` is the head's *input*
    dimension -- the size of the Hessian.
    
```

## gptq.py :: hessian_from_activations

```
Convenience: ``H = 2 X^T X`` over a stream of activation batches.
```

## gptq.py :: language_mass

```
Share of the Hessian's effective rows per language (the guard's check).

        Mirrors :meth:`hessian` exactly, so the reported shares are the weights
        actually applied to the accumulated products.
        
```

## gptq.py :: _row_weights

```
Multiplier applied to each language's accumulated ``X^T X``.

        The stored products are already sums, so an unweighted Hessian needs a
        multiplier of **1**, not the row count.
        
```

## gptq.py :: _effective_rows

```
Per-language row count the Hessian effectively sees.

        Both modes are one family -- a common row threshold ``T`` with
        ``rows_i -> min(rows_i, T)``:

        * ``balance`` picks ``T = min(rows)``, giving every language an equal
          share;
        * ``cap_fraction`` bisects for the *largest* ``T`` whose resulting
          maximum share is still within the cap, so the most calibration data is
          kept subject to the guard.
        
```

## gptq.py :: _weighted_effective_rows

```
Per-language effective rows after weighting, total preserved.

        The renormalisation is done in EFFECTIVE-ROW space, which is the space
        the 30% guard is stated in and the only one where "total mass" is
        meaningful.  An earlier version renormalised the multipliers instead,
        which silently changed the overall scale of the Hessian and therefore the
        damping (defined relative to the mean diagonal).
        
```

## gptq.py :: _pack_triple

```
Re-derive the stored triple from a reconstructed matrix.

    Deriving the triple from the *final* reconstructed values (rather than
    keeping the per-column codes) guarantees the checkpoint is self-consistent:
    ``mx.dequantize(packed, scales, biases)`` reproduces exactly what GPTQ
    decided, with no dependency on how the inner loop was organized.
    
```

## gptq.py :: rtn_quantize

```
Plain round-to-nearest group quantization (``mx.quantize``'s own rule).

    Provided as the baseline GPTQ has to beat, computed with the *same*
    quantizer so the comparison isolates the algorithm and not the rounding.
    Pass ``hessian`` to also get the layer-output error, which is the metric the
    head-to-head claim is made on.
    
```

## gptq.py :: gptq_quantize

```
Quantize ``weight`` ``[out, in]`` with GPTQ using ``hessian`` ``[in, in]``.

    ``hessian`` is the second-order statistic of the layer input,
    ``H = 2 X^T X`` accumulated in float32 (:class:`HessianAccumulator` does
    exactly that).  The returned triple is what a W4A16 checkpoint stores, in
    the standard MLX affine group layout.
    
```

## gptq.py :: _upper_cholesky_inverse

```
``U`` with ``Hinv = U^T U`` and ``U`` upper triangular (float32 out).

    The reference implementation goes ``chol(H) -> cholesky_inverse -> chol(·,
    upper=True)``; ``inv(H)`` then an upper Cholesky is the same matrix with one
    fewer decomposition, and the dampening has already made ``H`` positive
    definite.
    
```

## gptq.py :: GptqConfig

```
GPTQ hyper-parameters, mirroring the reference implementation's knobs.
```

## gptq.py :: cast

```
Round ``values`` to the storage dtype, keeping a float32 carrier.
```

## gptq.py :: storage_dtype

```
NumPy dtype the side tensors are returned in.

        bfloat16 has no NumPy representation, so it is carried as float32 whose
        *values* are already bf16-rounded -- converting to bf16 later is exact.
        
```

## gptq.py :: AffineQuantizer

```
MLX affine group quantization, bit-exact (see module docstring).

    ``bits`` and ``group_size`` must be a combination ``mx.quantize`` accepts:
    bits in (2, 3, 4, 5, 6, 8) and group size in (32, 64, 128).  ``storage``
    names the dtype the side tensors are stored in; it only affects rounding of
    ``scales``/``biases`` and therefore which codes are chosen.
    
```

## gptq.py :: _round_half_away

```
``round()`` with ties away from zero, as the MLX kernel does.

    NumPy's ``np.round`` is half-to-even, which disagrees with the kernel on
    exact .5 ties; they are rare on real weights but not on synthetic ones.
    
```

## kv_cache.py :: reserve

```
Grow the backing store to ``positions`` capacity immediately.

        ``like`` may carry the (B, H, D) shape when this cache is still empty.
        
```

## kv_cache.py :: KVCache

```
mlx-lm ``KVCache``: append-only keys/values with an offset cursor.

    ``reserve`` pre-allocates the buffer once (call it after prefill, when the
    final length is known: audio positions + prompt + max decode steps). Without
    it, the cache grows in ``step``-sized chunks, and every growth allocates a
    fresh buffer and copies the whole history -- measured ~3 ms per event on the
    0.6B, a 2-3x tax on a 7 ms decode step, hitting exactly when a long
    utterance crosses a 256-position boundary mid-decode.
    
```

## quant.py :: apply_quantization

```
Quantize ``model`` in place per the ``config.json`` section ``data``.

    Called by :func:`funasr_mlx.pipeline.load_model` *before* the expected
    parameter set is computed, so a quantized checkpoint's ``.scales`` and
    ``.biases`` entries are part of the strict key/shape check.  Returns
    ``None`` when the configuration carries no quantization section, i.e. the
    checkpoint is full precision.
    
```

## quant.py :: quantizes_module

```
Whether ``module`` (a live ``nn.Module``) should become quantized.

        The token embedding is a separate case from ``nn.Linear``: it is not a
        ``Linear`` subclass, so it can only be selected through an explicit
        ``embed_tokens`` target prefix, and it is the tied LM head as well.
        
```

## quant.py :: quantizes_tensor

```
Whether the converter should quantize checkpoint tensor ``name``.

        Mirrors the model-side predicate exactly: a Linear weight (see
        :meth:`is_linear_weight`) or a selected token embedding (see
        :meth:`quantizes_embedding`), whose input dimension the group size
        divides.
        
```

## quant.py :: is_linear_weight

```
Whether ``name`` is a Linear weight the target prefixes select.

        Rank-2 ``.weight`` under a selected prefix and not one of the
        :data:`NON_LINEAR_WEIGHT_NAMES` exceptions -- i.e. a tensor the model
        side would have turned into a ``QuantizedLinear`` if the group size
        fitted. This is the predicate the converter uses to decide whether a
        tensor it did *not* pack is a legitimate skip or an unnoticed miss.
        
```

## quant.py :: quantizes_embedding

```
Converter predicate for the token-embedding matrix.
```

## quant.py :: selects_embedding

```
Whether a token-embedding module at ``module_path`` is selected.

        The tied ``embed_tokens`` is the one non-Linear weight worth offering:
        when ``tie_word_embeddings`` is set the same matrix is also the LM head,
        so at bs=1 it is streamed once per step and is a large share of the step
        (311 MB bf16 for a 151936-vocab family). It needs its own predicate
        because ``nn.Embedding`` is not an ``nn.Linear`` subclass, so the Linear
        rule can never reach it; a family opts in by naming ``embed_tokens`` in
        ``targets``.
        
```

## quant.py :: QuantizationSpec

```
One family's quantization plan, as declared in ``config.json``.

    ``method`` records *how* the stored ``scales``/``biases``/codes were
    computed: ``rtn`` is ``mx.quantize``'s round-to-nearest, ``gptq`` is
    Hessian-based error compensation (:mod:`funasr_mlx.models.llm.gptq`).  The
    runtime layout is identical either way -- ``nn.QuantizedLinear`` and
    ``mx.quantized_matmul`` cannot tell them apart -- so the loader ignores it
    and it exists purely as provenance in the checkpoint's ``config.json``.
    
```

## qwen3.py :: Qwen3ForCausalLM

```
``Qwen3ForCausalLM``: the model plus the (tied or separate) LM head.
```

## rope.py :: initialize_rope

```
mlx-lm ``models/rope_utils.py::initialize_rope`` (inference subset).

    ``default`` (and absent config) and ``linear`` are supported; anything
    else raises rather than silently degrading position handling.
    
```

## splice.py :: splice_and_forward

```
Replace the ``pad_span``-wide audio-token run with tower output; decode.

    Prefill needs the causal mask even when KVCache objects already exist:
    the generic model call only builds one when ``cache is None`` as a whole.
    ``decoder_dtype``: the tower emits float32 embeds (float32 mel in); the
    cast prevents a mixed-dtype promotion at every Linear of the stack (a
    no-op for float32 checkpoints).
    
```


---

# Second wave: shorter docstrings (moved verbatim)

## api.py :: download_model

```
Download a published checkpoint (see funasr_mlx.download.download).
```

## api.py :: pseudo_stream

```
Create a PseudoStream bound to a loaded LLM transcriber.

    Must run inside a guard (use with TranscribeWorker / guarded child):

        worker = TranscribeWorker(model)
        stream = worker.pseudo_stream(chunk_seconds=1.0)
        for chunk in mic_chunks:
            ev = stream.push(chunk)
            if ev: render(ev.text)
        final = stream.close()   # == offline output, bit-identical
    
```

## checkpoint_stream.py :: write_tensor

```
Write one tensor, converting only bounded slices rather than its full data.

    ``code`` overrides the safetensors type code written for the payload; it is
    how a bfloat16 tensor (carried as a ``uint16`` array, since NumPy has no
    bfloat16) is stored as ``BF16`` rather than ``U16``. The override must agree
    with the array's element width, so no reinterpretation can silently happen.
    
```

## checkpoints.py :: SourceRule

```
How one LLM family's source keys map onto the native parameter names.

    ``prefixes`` maps a source key prefix to the native prefix, longest first.
    ``renames`` are literal substring rewrites applied afterwards (used for the
    Qwen3-ASR ``conv2d`` -> ``conv`` submodule move and for MOSS's
    ``language_model`` -> ``model`` decoder nesting). ``drop`` lists source keys
    that the native model does not carry, because the port *derives* them (the
    Qwen3-ASR sinusoidal position table). A rule is strict: a key that matches
    no prefix and is not in ``drop`` raises rather than being passed through, so
    a renamed upstream checkpoint cannot silently produce a wrong key set.
    
```

## checkpoints.py :: GptqWeight

```
Per-tensor GPTQ calibration state, keyed by native tensor name.

    ``hessians`` maps a native ``<module>.weight`` name to that matrix's
    ``H = 2 X^T X`` (float32).  ``configs`` maps the same name to the
    :class:`~funasr_mlx.models.llm.gptq.GptqConfig` to use, so different
    matrices can carry different hyper-parameters.  A name with a Hessian but no
    explicit config uses ``default_config``.

    Kept as a plain data holder (no MLX import at module scope) so the converter
    can accept it without pulling the runtime in.
    
```

## checkpoints.py :: quantize_tensor

```
Quantize one tensor and return MLX tensors in the usual layout.
```

## checkpoints.py :: _carry

```
Copy one source tensor into a small owned array in its carrier dtype.

    The copy is the point: the source buffer belongs to a lazily-read shard
    handle, and converted arrays that still referenced it would pin the whole
    shard in memory. Every layout transpose therefore happens *after* this
    copy, on the owned array. bfloat16 is viewed as ``uint16`` first because
    ``torch.Tensor.numpy`` refuses the bfloat16 scalar type outright.
    
```

## checkpoints.py :: _bf16_carrier_to

```
Round a ``uint16``-carried bfloat16 tensor to the requested precision.

    bfloat16 -> float32 is exact (a 16-bit left shift; the shift must be done
    in 32-bit lanes or the high half is discarded). The further round to
    float16 happens on the host, exactly as an on-device ``astype`` would.
    
```

## llm_asr.py :: LlmAsrTranscriber

```
Resident transcriber: one model load, many calls.

    Construct inside a guarded child (``memory.run_guarded``); the API layer
    arranges that. KV caches are allocated once and reset (offset=0) between
    utterances -- reuse validated bit-stable in .tmp/opt/LOG.md.
    
```

## pipeline.py :: load_model

```
Prevalidate every key/shape, then load one bounded shard at a time.

    Only native-layout, explicitly tagged checkpoints are admitted. Running in
    a guarded child is mandatory; importing this module does not load MLX.
    
```

## pseudo_streaming.py :: StreamEvent

```
One pseudo-stream block result.

    ``text`` is the full current best-effort transcription (stable prefix +
    rolling tail). ``is_final`` marks the closing event, whose text equals
    offline transcription exactly.
    
```

## pseudo_streaming.py :: classic_transcriber

```
Build a ``samples -> text`` callable for classic (non-LLM) families.

    Covers sensevoice / paraformer (greedy CTC/CIF). ``cmvn`` is REQUIRED
    for sensevoice-family checkpoints (``am.mvn``; found next to the source
    configuration) -- without it the fbank distribution is shifted and the
    encoder collapses to all-blanks (verified: argmax == 0 everywhere).
    
```

## pseudo_streaming.py :: llm_transcriber

```
Adapt an ``LlmAsrTranscriber`` into a ``samples -> text`` callable.
```

## _layout_state.py :: convert_tensor

```
Apply the transpose for ``rule``; ``names`` is the family's identity set.
```

## _strict_load.py :: flat_parameter_names

```
Flatten ``module.parameters()`` into FunASR-style dotted names.
```

## _strict_load.py :: convert_checkpoint_tensors

```
Convert a FunASR/PyTorch ``state_dict`` into MLX layout.

    ``rules`` maps key suffixes to transforms; each family's depthwise-conv
    transpose lives there. Everything else is copied verbatim.
    
```

## _strict_load.py :: load_state_dict_strict

```
Strictly load ``state`` into ``model``; raise on any mismatch.
```

## campplus.py :: ModuleStack

```
Ordered children registered under their numeric index.

    ``torch.nn.Sequential`` yields ``<parent>.<i>.<leaf>`` parameter names; MLX's
    ``nn.Sequential`` would insert an extra ``layers`` level, so children are
    registered directly under ``"0"``, ``"1"``, ...
    
```

## campplus.py :: NamedModuleDict

```
Named children applied in order; keeps the ``<parent>.<name>.<leaf>`` keys.

    Used for ``CAMPPlus.xvector``, which the reference builds as an
    ``nn.Sequential`` of named sub-modules.
    
```

## campplus.py :: NamedLayerList

```
Named children iterated in insertion order.

    Mirrors ``torch.nn.ModuleList`` with explicit names (``CAMDenseTDNNBlock``
    uses ``tdnnd1``, ``tdnnd2``, ...).
    
```

## campplus.py :: LayoutReport

```
Flat parameter tensors plus the layout rule that applies to each key.

    Behaves like a read-only ``Mapping`` of ``{parameter_name: mlx_array}``, so it
    can be iterated, indexed, ``len()``-ed and fed to ``dict(...)`` exactly like a
    plain state dict, while also carrying the per-key rule used for the
    torch -> MLX layout conversion.
    
```

## campplus.py :: flatten_state_dict

```
Flatten a module tree into FunASR/PyTorch-compatible parameter names.

    Unlike a name-pattern heuristic, the layout rule is derived from the *owning
    module type*, so it is correct for any naming the architecture happens to
    use.  Attributes starting with ``_`` are configuration, never weights.
    
```

## campplus.py :: flatten_state_dict_with_rules

```
Like :func:`flatten_state_dict`, but also returns the per-key layout rule.

    Returns ``(tensors, rules)`` with the same two dictionaries that the
    :class:`LayoutReport` carries -- nothing is dropped.  This is the contract
    :func:`funasr_mlx.models.sanm.flatten_state_dict_with_rules` exposes, so the
    three model modules agree on one name for the same operation; callers that
    want the richer object should use :func:`flatten_state_dict` instead.
    
```

## campplus.py :: LoadReport

```
Outcome of a :func:`load_strict` call.

    ``dropped`` lists the torch-only buffers that were intentionally ignored.
    It is never empty silently: callers can assert on it.
    
```

## campplus.py :: kaiming_normal_

```
In-place-style ``torch.nn.init.kaiming_normal_`` for MLX conv/linear weights.

    The reference re-initialises every ``Conv1d``/``Linear`` weight with
    ``kaiming_normal_`` and zeroes the biases *after* building ``CAMPPlus``.
    ``fan_in`` is the number of input elements per output element, which is the
    product of every axis except the output axis in both layouts.
    
```

## campplus.py :: get_nonlinear

```
Retain the reference Sequential's named children and per-channel PReLU.
```

## campplus.py :: statistics_pooling

```
Time-axis mean/std pooling of an ``(B, L, C)`` tensor -> ``(B, 1, 2C)``.

    The reference's ``statistics_pooling(x, dim=-1)`` reduces the *time* axis of
    an NCL tensor and concatenates along the channel axis, giving ``(B, 2C)``.
    Reducing axis 1 of the channels-last tensor and concatenating on the last
    axis gives ``(B, 1, 2C)`` with the same numbers in the same order, which is
    what the following ``DenseLayer`` consumes.  ``eps`` is accepted for
    signature parity; the reference never uses it.
    
```

## campplus.py :: BasicResBlock

```
``components.BasicResBlock`` (channels-last NHWC).

    The torch shortcut is an ``nn.Sequential`` (empty => identity), which puts
    its parameters at ``shortcut.0.weight`` / ``shortcut.1.weight``; a
    :class:`ModuleStack` (empty in the identity case) reproduces those names
    exactly.
    
```

## campplus.py :: __call__

```
``x`` is ``(B, F, T)`` (mel-major, as in the reference) -> ``(B, T, C*F/8)``.
```

## campplus.py :: TDNNLayer

```
``components.TDNNLayer``.

    ``padding < 0`` selects the reference's same-padding rule
    ``padding = (kernel_size - 1) // 2 * dilation`` and requires an odd kernel.
    
```

## campplus.py :: CAMLayer

```
``components.CAMLayer`` (channels attention over a dilated TDNN).

    ``context`` mixes the *global* time mean with a non-overlapping segment mean
    (``seg_len=100``, average pooling with ``ceil_mode=True``), squeezes it
    through ``linear1``/``relu``/``linear2``/``sigmoid``, and rescales the local
    branch.  Because the channel axis is last, the time axis is axis 1.
    
```

## campplus.py :: CAMDenseTDNNLayer

```
``components.CAMDenseTDNNLayer``.

    ``memory_efficient`` only gates ``torch.utils.checkpoint`` during *training*
    and is therefore inert here (documented, not silently dropped).
    
```

## campplus.py :: TransitLayer

```
``components.TransitLayer`` (nonlinearity **before** the linear map).
```

## campplus.py :: DenseLayer

```
``components.DenseLayer``.

    The reference special-cases 2-D input by squeezing through a length-1 axis so
    that a 1x1 ``Conv1d`` can be used as a plain ``Linear``; that branch is
    reproduced (channels-last, so the squeezed axis is the time axis).
    
```

## campplus.py :: CAMPlusConfig

```
Mirror of FunASR's ``model_conf`` for ``CAMPPlus``.

    Defaults are the published ``speech_campplus_sv_zh-cn_16k-common`` values
    taken from ``funasr/models/campplus/template.yaml``.
    
```

## campplus.py :: campplus_config_from_dict

```
Build a :class:`CAMPlusConfig` from a FunASR ``model.yaml`` mapping.
```

## campplus.py :: CAMPPlus

```
``funasr.models.campplus.model.CAMPPlus`` (inference only).

    Input is 80-dim Kaldi fbank **without** the LFR stack and **without** CMVN;
    ``funasr/models/campplus/utils.py::extract_feature`` mean-normalises each
    utterance across all frames and zero-pads the batch.  Both steps stay
    outside this module (frontend scope).

    ``output_level="segment"`` returns ``(B, embedding_size)``;
    ``output_level="frame"`` returns ``(B, T, channels)``.
    
```

## campplus.py :: state_dict

```
Flat ``{pytorch_name: mlx_array}`` plus per-key layout rules.
```

## campplus.py :: load_checkpoint

```
Strict load; see :func:`load_strict`.
```

## campplus.py :: frame_features

```
Run everything up to (but excluding) pooling; returns ``(B, T, C)``.

        The stage order is taken from the ``xvector`` container's own insertion
        order, which mirrors the reference's ``nn.Sequential``: ``tdnn``, then
        ``block1, transit1, block2, transit2, ...``, then ``out_nonlinear``.
        Iterating separate block/transit name lists instead would apply **all**
        blocks before **any** transit, which both misorders the network and
        desynchronises the channel arithmetic (each transit halves the channel
        count feeding the next block).
        
```

## campplus.py :: forward

```
``x`` is fbank ``(B, T, feat_dim)``.

        Literal port of the reference ``forward``: the batch is assumed to be
        already padded (or to contain a single utterance), because the reference
        pools without any padding mask.  See :meth:`embed` for the length-aware
        variant.
        
```

## ct_punc.py :: SanmEncoderConfig

```
Validated mirror of ``encoder_conf`` for FunASR's ``SANMEncoder``.

    Only the ``ct-punc`` configuration is supported: ``input_layer="pe"`` (a
    parameter-free sinusoidal position encoder) and
    ``selfattention_layer_type="sanm"``.
    
```

## ct_punc.py :: from_dict

```
Build from a FunASR ``encoder_conf`` mapping.
```

## ct_punc.py :: CtPuncConfig

```
Validated mirror of ``model_conf`` for FunASR's ``CTTransformer``.
```

## ct_punc.py :: from_dict

```
Build from a full FunASR ``config.yaml`` mapping or from a flat one.

        Accepts ``{"model_conf": {...}, "encoder_conf": {...},
        "tokenizer_conf": {...}}`` (the on-disk layout) as well as a single flat
        ``model_conf`` mapping.
        
```

## ct_punc.py :: CharVocab

```
Character vocabulary equivalent to FunASR's ``CharTokenizer``.

    ``BaseTokenizer.encode`` simply maps every token through ``token2id`` with an
    ``<unk>`` fallback, so this class exposes exactly that behaviour and nothing
    more (no segmentation: the caller already produced tokens via
    :func:`split_words`).
    
```

## ct_punc.py :: from_file

```
Load a newline-separated vocabulary (``token_list.txt``).
```

## ct_punc.py :: from_config

```
Build from a FunASR ``tokenizer_conf`` mapping (or an explicit path).
```

## ct_punc.py :: is_first_byte_ascii

```
FunASR's ``len(token[0].encode()) == 1`` test (first character is ASCII).
```

## ct_punc.py :: make_valid_mask

```
``(~make_pad_mask(lengths)[:, None, :])`` -> ``(B, 1, maxlen)`` of 1/0 floats.
```

## ct_punc.py :: sinusoidal_position_encoding

```
FunASR ``SinusoidalPositionEncoder`` (``input_layer="pe"``), parameter-free.

    ``log_timescale_increment = log(10000) / (depth/2 - 1)``; the sin block is
    concatenated *before* the cos block; positions start at 1.
    
```

## ct_punc.py :: ModuleLayerList

```
``nn.ModuleList`` variant keyed by *string* index.

    MLX's own ``ModuleList`` nests children under ``layers``, which would shift
    every checkpoint key (``encoders.0.…`` -> ``encoders.layers.0.…``).  Storing
    the children directly under ``"0"``, ``"1"``, … keeps FunASR's
    ``nn.ModuleList`` parameter paths byte-identical.

    Note ``nn.Module.update()`` can only overwrite *existing* keys, so children
    are inserted as dict items (which is what ``Module.__setattr__`` does for
    ``dict``/``list`` values too).
    
```

## ct_punc.py :: PositionwiseFeedForward

```
``PositionwiseFeedForward`` with the ``w_1``/``w_2`` names preserved.

    Dropout is omitted: FunASR's ``nn.Dropout`` is an identity at inference and
    this port is inference-only.
    
```

## ct_punc.py :: SanmAttention

```
``MultiHeadedAttentionSANM`` (funasr/models/sanm/attention.py).

    Weight layout conversion: ``self.fsmn_block.weight`` is ``(C, 1, K)`` in
    PyTorch (depthwise ``nn.Conv1d``) and must be transposed to MLX's
    ``(C, K, 1)``.  ``linear_q_k_v`` and ``linear_out`` already use MLX's layout.
    
```

## ct_punc.py :: SanmEncoder

```
``SANMEncoder`` restricted to ``input_layer="pe"``.

    Parameter paths are ``encoders0.0.…`` for the first block and ``encoders.i.…``
    for the remaining ``num_blocks - 1``, matching FunASR exactly.
    
```

## ct_punc.py :: CtPuncModel

```
``CTTransformer``: embedding -> SANMEncoder -> linear punc classifier.
```

## ct_punc.py :: argmax_punc_ids

```
FunASR's ``y.view(-1, C).topk(1, dim=1)`` followed by ``squeeze(1)``.
```

## ct_punc.py :: split_words

```
FunASR ``split_words`` without a jieba user dictionary.

    Splits on whitespace, then keeps maximal ASCII runs as one token and emits
    every non-ASCII character as its own token.
    
```

## ct_punc.py :: split_to_mini_sentence

```
FunASR ``split_to_mini_sentence``: fixed-size chunks plus one remainder.
```

## ct_punc.py :: expand_word_level_punc

```
FunASR's jieba word-level ``punc_array`` expansion.

    Mirrors the reverse walk in ``CTTransformer.inference`` for CJK word tokens
    longer than one character: insert ``len(token) - 1`` entries of ``no_punc_id``
    so the array lines up with the character-level token stream.

    The reference hard-codes the inserted value as the literal ``1``; that is kept
    as the default so behaviour is bit-identical, but note that ``1`` only means
    "no punctuation" when ``punc_list[1] == "_"``.  Pass ``no_punc_id=0`` (or
    whatever the target config uses) when the label table is ordered differently.
    
```

## ct_punc.py :: CtPuncPunctuator

```
End-to-end ``ct-punc``: tokens -> model -> punctuated text + ``punc_array``.

    The mini-sentence chunking / look-back cache behaviour of
    ``CTTransformer.inference`` is reproduced exactly, including the case where a
    chunk contains no sentence terminator: nothing is emitted for that chunk, the
    whole chunk is carried into the next one, and once it exceeds
    ``cache_pop_trigger_limit`` tokens the last comma is promoted to a sentence
    end.
    
```

## ct_punc.py :: reset

```
No-op: ``ct-punc`` carries no state between calls.

        Provided for API symmetry with the streaming VAD; the mini-sentence cache
        is local to :meth:`punctuate`.
        
```

## emotion2vec.py :: get_alibi

```
``get_alibi(max_positions, attention_heads, dims=1)``.

    1-D manhattan ALiBi: ``slopes[h] * -|i - j|`` with shape
    ``(heads, max_positions, max_positions)``.
    
```

## emotion2vec.py :: OrderedModules

```
An ``nn.Sequential`` whose children keep their integer index in the key.

    ``nn.Module`` is a dict subclass and ``tree_flatten`` walks ``.items()``, so
    registering a child under the *string* ``"2"`` reproduces PyTorch's
    ``...2.1.weight`` naming exactly.  A plain list would also be captured by
    ``Module.__setattr__`` and would pollute the key space, hence the tuple.
    
```

## emotion2vec.py :: __getitem__

```
Fetch by Sequential position (``self[0]``) **or** by raw key (``self["0"]``).

        Both spellings are used: ``__call__`` walks ``_order`` positionally while
        :class:`PositionalBlock` needs named access so it can reorder the
        transposes around the channel-last conv.
        
```

## emotion2vec.py :: ModuleListStack

```
``nn.ModuleList``; children registered under their decimal index.
```

## emotion2vec.py :: __getitem__

```
Fetch by position (``self[0]``) or by the mapping key (``self["0"]``).

        ``nn.Module.update`` / ``tree_unflatten`` hand back the *string* keys that
        were registered here, so a positional-only ``index < 0`` comparison raises
        ``TypeError: '<' not supported between instances of 'str' and 'int'``.
        
```

## emotion2vec.py :: TransposeLast

```
``fairseq.modules.TransposeLast``: swap the last two axes.
```

## emotion2vec.py :: Fp32LayerNorm

```
``fairseq.layers.fp32_group_norm.Fp32LayerNorm``.

    A ``nn.LayerNorm`` that upcasts to float32, normalises and casts back, so
    the reduction never happens in half precision.
    
```

## emotion2vec.py :: Gelu

```
``nn.GELU()`` with the default (exact, erf-based) approximation.
```

## emotion2vec.py :: Mlp

```
``fairseq`` ``Mlp``: ``fc1 -> act -> fc2`` with dropout around.
```

## emotion2vec.py :: parse_feature_encoder_spec

```
Parse ``feature_encoder_spec`` with ``ast`` -- ``eval`` is never used.

    Accepts only literal arithmetic over tuples of ints, i.e. ``+``, ``*`` and
    ``[a, b, c]``.  Anything else (names, calls, attributes, comprehensions)
    raises.
    
```

## emotion2vec.py :: ConvFeatureBlock

```
``wav2vec2.py:839-860`` ``block(...)`` for ``is_layer_norm=True``.

    ``Conv1d -> Dropout -> Sequential(TransposeLast, Fp32LayerNorm(affine=True),
    TransposeLast) -> GELU``.  The double transpose turns the ``NCT`` conv
    output into ``NTC`` so the LayerNorm normalises the **512 channels**, and the
    norm is **affine**, i.e. its ``weight``/``bias`` are real checkpoint tensors.
    
```

## emotion2vec.py :: ConvFeatureExtraction

```
``ConvFeatureExtractionModel`` with ``mode="layer_norm"``.
```

## emotion2vec.py :: PositionalEncoder

```
``Sequential(TransposeLast, 5 x PositionalBlock, TransposeLast)``.

    With ``conv_pos_pre_ln=false`` (the large config) the conv's
    ``padding = k // 2 = 9`` preserves ``T`` and ``SamePad(19)`` is the identity
    on top of it, so the output length is always exactly ``T``.

    Time axis contract: the encoder takes ``NTC`` and returns ``NTC``; each
    :class:`PositionalBlock` takes ``NTC`` and returns ``NCT``, so the encoder
    restores ``NTC`` for the final output.  Each :class:`PositionalBlock` is
    ``NCT`` in and ``NCT`` out, so nothing is transposed *between* blocks.
    
```

## emotion2vec.py :: __call__

```
``NTC`` in, ``NTC`` out -- the untouched source chain.

        The leading ``TransposeLast`` converts ``NTC -> NCT`` for the blocks (each
        ``NCT -> NCT``) and the trailing one restores ``NTC``.
        
```

## emotion2vec.py :: AltAttention

```
``modules.py:255-300`` ``AltAttention`` (cosine attention is not used).
```

## emotion2vec.py :: AltBlock

```
``modules.py:180-245`` ``AltBlock`` with ``layer_norm_first=false``.

    Post-norm ordering::

        x = x + attn(norm1(x))
        x = x + mlp(norm2(x))

    and the second return value is the post-norm ``x`` when ``ffn_targets`` else
    the pre-norm mlp output.  That value is training-only (it supervises the
    teacher), so the port returns it for API parity but never consumes it.
    
```

## emotion2vec.py :: __call__

```
``modules.py:227-244`` with every dropout/drop-path removed.

        Note the post-norm branch feeds the **raw** ``x`` to the attention and
        only then applies ``norm1``; ``t`` is the pre-``norm2`` mlp output when
        ``ffn_targets`` and the final ``x`` otherwise.  ``t`` is training-only
        (it supervises the teacher) and is returned for API parity only.
        
```

## emotion2vec.py :: Emotion2VecConfig

```
Validated mirror of the audio modality config plus the model fields.
```

## emotion2vec.py :: pos_kernel_size

```
``k = max(3, conv_pos_width // conv_pos_depth)`` (``audio.py:87``).
```

## emotion2vec.py :: ffn_targets

```
``ffn_targets = not end_of_block_targets`` (``emotion2vec.py``).
```

## emotion2vec.py :: _weight_dtype

```
Dtype of this encoder's checkpoint weights (first conv always exists).
```

## emotion2vec.py :: local_features

```
Waveform ``(B, samples)`` -> projected features ``(B, T, embed_dim)``.

        The conv stack returns ``NCT`` and ``project_features`` is the source's
        ``Sequential(TransposeLast, LayerNorm(512), Linear)``.  Its leading
        ``TransposeLast`` is a **real** transpose, so the ``NCT`` tensor is fed
        straight in and the norm correctly reduces the **512 channels**.
        
```

## emotion2vec.py :: alibi_bias

```
``get_alibi_bias`` -> ``(B, heads, T, T)``.
```

## emotion2vec.py :: contextualized_features

```
``contextualized_features(..., mask=False, remove_masked=False)``.

        Returns ``(x, alibi_bias, alibi_scale_for_model_blocks)``.  When the
        learned scale has a single row it is folded into ``alibi_bias`` here and
        the third element is ``None``; with one row **per layer** the rows stay
        live and the tail (``alibi_scale[prenet_depth:]``) is handed back so the
        shared blocks pick their own row, exactly like the reference.
        
```

## emotion2vec.py :: EmotionResult

```
Mirrors the keys FunASR's ``Emotion2vec.inference`` returns.
```

## emotion2vec.py :: preprocess_waveform

```
``source = F.layer_norm(source, source.shape)`` then ``view(1, -1)``.

    The reference normalises over the *entire* 1-D waveform (one global
    normalisation), not per frame.
    
```

## emotion2vec.py :: Emotion2VecModel

```
``Emotion2vec`` (FunASR wrapper) as native MLX modules.

    The checkpoint is a plain ``state_dict`` of the ``Emotion2vec`` module, so the
    parameter prefixes are ``modality_encoders.AUDIO.*``, ``blocks.N.*`` and
    ``proj.*``.  ``self.modality_encoders`` is a plain dict so the keys come out
    as ``modality_encoders.AUDIO.<...>`` exactly like the reference
    ``nn.ModuleDict``.
    
```

## emotion2vec.py :: extract_features

```
Waveform ``(B, samples)`` -> frame features ``(B, T', embed_dim)``.

        ``remove_extra_tokens=True`` strips the ``num_extra_tokens`` cls tokens,
        which is what FunASR does before the utterance-level pooling.  The encoder
        is evaluated exactly **once**.
        
```

## emotion2vec.py :: classify

```
Return ``(features (B, embed_dim), logits (B, vocab_size), frames)``.

        The head is ``nn.Linear(embed_dim, vocab_size)`` applied to the mean over
        time **after** the extra tokens have been removed
        (``Emotion2vec.inference``: ``x = x.mean(dim=1); x = self.proj(x)``).  The
        frame features are returned so a caller that also wants them does not have
        to run the whole encoder a second time.
        
```

## emotion2vec.py :: emotion_scores

```
Full FunASR-equivalent result: labels + softmax scores + features.

        Runs the encoder exactly once; ``feats`` is the utterance-level pooled
        embedding the classifier consumed, which is the same
        ``x.mean(dim=1)`` the reference uses, so no second pass is needed.
        
```

## emotion2vec.py :: is_conv_weight

```
True for the depthwise ``Conv1d`` weights listed in :data:`CONV_WEIGHT_PATTERNS`.
```

## emotion2vec.py :: convert_checkpoint_tensors

```
Convert a published ``state_dict`` and report what was discarded.

    Only ``Conv1d`` weights change layout: ``(out, in//g, K) -> (out, K, in//g)``.
    Linears, LayerNorms, ``extra_tokens`` and ``alibi_scale`` already match MLX.
    
```

## emotion2vec.py :: load_state_dict_strict

```
Load ``state`` into ``module``; return ``(discarded, unexpected)``.

    Never silently skips a key: anything outside
    :data:`INFERENCE_ONLY_PREFIXES` must match a parameter by name *and* shape.
    
```

## fsmn_vad.py :: convert_checkpoint_tensors

```
Depthwise 4-D kernels transposed; strict elsewhere (see _strict_load).
```

## fsmn_vad.py :: FsmnEncoderConfig

```
Validated mirror of ``encoder_conf`` for FunASR's ``FSMN`` network.
```

## fsmn_vad.py :: cache_frames

```
Left-context cache length ``(lorder - 1) * lstride``.
```

## fsmn_vad.py :: VadOptions

```
Mutable mirror of FunASR's ``VADXOptions``.

    Mutable because ``ComputeScores`` overwrites ``nn_eval_block_size`` with the
    number of frames in the current block, exactly like the reference.
    
```

## fsmn_vad.py :: max_end_sil_frame_cnt_thresh

```
``Stats.max_end_sil_frame_cnt_thresh`` at cache-initialisation time.
```

## fsmn_vad.py :: ModuleLayerList

```
``nn.ModuleList`` keyed by *string* index (see ``ct_punc.py`` for why).

    MLX's own ``ModuleList`` nests children under ``layers`` and would shift every
    checkpoint key; FunASR's ``FsmnStack`` (an ``nn.Sequential``) indexes blocks
    as ``fsmn.0.…``, ``fsmn.1.…``, so children must sit directly under ``"0"``,
    ``"1"``, …  ``nn.Module.update()`` can only overwrite existing keys, so
    children are inserted as dict items.
    
```

## fsmn_vad.py :: LinearTransform

```
Bias-free wrapper retaining the reference's named ``linear`` child.

    ``BasicBlock.linear`` owns this wrapper, so its child adds the second
    ``linear`` in ``fsmn.<i>.linear.linear.weight``.
    
```

## fsmn_vad.py :: AffineTransform

```
``AffineTransform``: ``self.linear = nn.Linear(input_dim, output_dim)``.

    Like :class:`LinearTransform`, this retains the named ``linear`` submodule;
    unlike it, the affine transform also has a bias.
    
```

## fsmn_vad.py :: FSMNBlock

```
``FSMNBlock``: left/right dilated depthwise memory block.

    The reference applies ``nn.Conv2d`` to a ``(B, D, T, 1)`` view; here the time
    axis is the sequence axis of ``mx.conv1d`` and the trailing singleton axis is
    dropped (it carries no data).

    Cache shapes: ``cache`` is ``(B, D, (lorder - 1) * lstride)`` or ``None``.
    
```

## fsmn_vad.py :: __call__

```
Return ``(output, new_cache)`` with ``output`` shaped ``(B, T, D)``.

        Layout: ``mx.nn.Conv1d`` is channels-last ``(batch, length, channels)``.
        The reference works in ``(B, D, T)``; the time axis is moved to the middle
        so the ``(lorder - 1) * lstride`` left padding (or cache concatenation) is
        expressed on the same axis FunASR pads, then moved back for the conv.

        ``mx.core.conv1d`` reads ``in[t * stride + k * dilation]``; with
        ``y[i] = x[i - keep]`` kernel tap ``k`` therefore reads ``x[t - keep + k]``.
        
```

## fsmn_vad.py :: BasicBlock

```
``BasicBlock``: linear -> FSMN -> affine -> relu, with a per-layer cache.
```

## fsmn_vad.py :: FSMN

```
``FSMN`` encoder (``funasr/models/fsmn_vad_streaming/encoder.py``).

    ``cache=None`` runs the non-streaming path (zero left padding); passing a dict
    runs the streaming path and updates ``cache["cache_layer_<i>"]`` in place,
    exactly like the reference.
    
```

## fsmn_vad.py :: forward_with_caches

```
ONNX-export style: return ``(scores, [cache_layer_0, …])``.
```

## fsmn_vad.py :: FsmnVadModel

```
FSMN-VAD: streaming/offline speech endpointing.

    ``__call__(feats, waveform, cache, is_final=…, is_streaming_input=…)`` mirrors
    ``FsmnVADStreaming.forward`` and returns ``[[[start_ms, end_ms], …], …]``
    (batch x segment; batch size 1 is the only supported case, as in FunASR).
    
```

## fsmn_vad.py :: from_dict

```
Build from a FunASR ``config.yaml`` mapping.
```

## fsmn_vad.py :: init_cache

```
``init_cache``: fresh encoder caches, window detector and stats.
```

## fsmn_vad.py :: reset

```
Full reset between utterances (``init_cache``).
```

## fsmn_vad.py :: drop_cached_frames

```
``DropCachedFrames``: release decibel/score history before a frame.
```

## fsmn_vad.py :: __call__

```
``FsmnVADStreaming.forward``.

        ``feats`` is ``(1, T, input_dim)`` fbank (the frontend is owned by the
        shared runtime); ``waveform`` is the exact ``(1, N)`` (or ``(N,)``)
        sample span those ``T`` frames cover.
        
```

## fsmn_vad.py :: detect

```
Convenience wrapper for the offline case (auto-initialised cache).
```

## fun_asr_nano.py :: TransformerAdaptor

```
``funasr.models.llm_asr.adaptor.Transformer`` (inference).

    Stacks ``k`` frames (k=1 here), projects encoder_dim -> llm_dim through
    an FFN, then runs standard encoder layers. The reference layers are
    ESPnet-style ``EncoderLayer`` (post-norm when normalize_before=False,
    which is the adaptor default: EncoderLayer's default normalize_before
    is False) with ``MultiHeadedAttention`` and ``PositionwiseFeedForward``.
    
```

## fun_asr_nano.py :: StandardMHA

```
ESPnet ``MultiHeadedAttention`` (transformer/attention.py): separate
    ``linear_q/k/v`` + ``linear_out`` and NO fsmn memory -- the adaptor
    checkpoint keys (``blocks.N.self_attn.linear_q...``) are exactly this
    module, distinct from the SANM attention of the speech encoders.
```

## fun_asr_nano.py :: FunASRNano

```
``FunASRNano`` (inference): encode -> adapt -> LLM decode.
```

## fun_asr_nano.py :: encode_audio

```
``(B, T, 560)`` LFR fbank -> adaptor embeddings ``(B, T', llm_dim)``.
```

## fun_asr_nano.py :: embed_with_speech

```
Token embeddings with the speech span spliced in, then one LLM pass.
```

## fun_asr_nano.py :: nano_config_from_nano_yaml

```
Build the config from the official Fun-ASR-Nano ``config.yaml``.

    Reads ``audio_encoder_conf`` / ``audio_adaptor_conf`` and pins the
    Qwen3-0.6B ``llm_conf.args`` (the architecture of the shipped ``llm.*``
    subtree, verified against the checkpoint).
    
```

## glm_asr.py :: GlmAsrAudioEncoder

```
``glmasr_encoder`` (inference): mel (F, T) -> (1, T', output_size).
```

## glm_asr.py :: __call__

```
mel: ``(F, T)``; returns ``(1, T'', output_size)`` embeddings.

        The official projector consumes the encoder output RE-GROUPED into
        ``intermediate_size``-wide rows: every 4 frames of 1280 become one
        5120-wide row (a 4-frame merge), then linear_1 5120->4096 + gelu +
        linear_2 4096->2048.
        
```

## glm_asr_thinker.py :: GlmAsrThinker

```
audio_tower (with projector folded in) + Llama causal LM.
```

## glm_asr_thinker.py :: _decoder_dtype

```
Dtype the decoder activations must follow.

        Read through :func:`activation_dtype`, not ``weight.dtype``: a quantized
        embedding stores packed ``uint32`` codes in ``weight``, and casting the
        spliced audio embeddings to ``uint32`` destroys them.
        
```

## glm_asr_thinker.py :: _tower_dtype

```
Dtype of the audio tower's checkpoint weights (first conv).
```

## glm_asr_thinker.py :: encode_audio

```
mel (F, T) -> (1, N, text_hidden) embeddings for the pad run.

        The mel is cast to the tower's weight dtype, as whisper's AudioEncoder
        and qwen3-asr's tower do.  This cast ALONE measures 1.00x here (see
        .tmp/task4/tower-tight-*.json and the re-measurement in
        .tmp/t7/glm-isolate.json), which is why it was originally skipped -- but
        that A/B was misleading: the tower's partial-RoPE tables were float32,
        so attention re-promoted the stream to float32 regardless.  With the
        tables also following the activation dtype (see ``_Attn._rope``) the
        two casts together give 1.44x on the tower and 28.7% on the pipeline.
        
```

## moss_transcribe.py :: _HfWhisperAttn

```
HF ``WhisperAttention``: q/v/out carry biases, **k does not**.

    ``k_proj`` is bias-free in the reference (``nn.Linear(embed_dim,
    embed_dim, bias=False)`` in transformers' ``modeling_whisper.py``), and the
    checkpoint agrees: ``...self_attn.k_proj.bias`` is absent from all 24
    layers.  Building it with ``bias=True`` therefore left a random,
    never-trained key bias in the audio tower -- measured on the previous
    release: exists=True, absmax 0.0312, absmean 0.0158.  The old permissive
    loaders never noticed, which is why MOSS tower parity sat at 1.3e-3.
    
```

## moss_transcribe.py :: HfWhisperEncoder

```
HF ``WhisperEncoder`` (inference): mel (F, T) -> (1, T/2, d).
```

## moss_transcribe.py :: VQAdaptor

```
Linear(input_dim->hidden) + SiLU + Linear(hidden->hidden) + LN(1e-6).
```

## moss_transcribe.py :: MossTranscribeDiarize

```
whisper_encoder + vq_adaptor + qwen3 (audio-token splice).
```

## moss_transcribe.py :: encode_audio

```
mel (F, T) -> whisper (1, T/2, 1024) -> adaptor (1, T/8, hidden).
```

## moss_transcribe.py :: encode_audio_from_hidden

```
Merged/chunked whisper output (1, T, 1024) -> adaptor (1, T/4, hidden).

        Trims to a multiple of the merge size first (official time_merge).
        
```

## moss_transcribe.py :: _decoder_dtype

```
Dtype the decoder activations must follow.

        Read through :func:`activation_dtype`, not ``weight.dtype``: a quantized
        embedding stores packed ``uint32`` codes in ``weight``, and casting the
        spliced audio embeddings to ``uint32`` destroys them.
        
```

## moss_transcribe.py :: forward_with_audio

```
Replace the audio-token run with adaptor output (shared splice).
```

## paraformer.py :: SANMEncoderConfig

```
Mirror of FunASR's ``encoder_conf`` for ``SANMEncoder`` (Paraformer side).
```

## paraformer.py :: ParaformerDecoderConfig

```
Mirror of FunASR's ``decoder_conf`` for ``ParaformerSANMDecoder``.
```

## paraformer.py :: CifPredictorConfig

```
Mirror of FunASR's ``predictor_conf`` for ``CifPredictorV2``.
```

## paraformer.py :: CTCConfig

```
Mirror of FunASR's ``ctc_conf`` (inference-relevant fields only).
```

## paraformer.py :: ParaformerConfig

```
Mirror of FunASR's ``model_conf`` for ``Paraformer``.

    Defaults follow the published ``paraformer-zh`` configuration
    (``input_size=560`` LFR fbank, 50 SANM blocks of width 512, 16 decoder
    blocks, CIF ``tail_threshold=0.45``).
    
```

## paraformer.py :: paraformer_config_from_dict

```
Build a :class:`ParaformerConfig` from a FunASR ``model.yaml`` mapping.
```

## paraformer.py :: CTC

```
``funasr.models.ctc.ctc.CTC`` inference path (``ctc_lo`` projection).
```

## paraformer.py :: SANMEncoder

```
``funasr.models.sanm.encoder.SANMEncoder`` (offline inference path).

    Differences from the reference that do not affect checkpoint parity:
    intermediate CTC / ``interctc_layer_idx`` outputs and streaming
    ``forward_chunk`` are not ported (both are training / streaming features).
    
```

## paraformer.py :: _gather

```
Gather time steps of ``array`` in fire order, independently per row.

        ``mx.take_along_axis`` indexes the time axis directly, so no flatten,
        offset arithmetic or output reshape is involved.  Per the installed stub
        (``mlx/core/__init__.pyi:1901-1909``) ``indices`` must be "broadcastable
        with the input array excluding the ``axis`` dimension", so a trailing
        singleton axis broadcasts one time index across every feature lane and
        the result keeps ``(B, T, D)``.
        
```

## paraformer.py :: CifPredictorV2

```
``funasr.models.paraformer.cif_predictor.CifPredictorV2`` (inference path).

    Parameters: ``cif_conv1d`` (a **dense**, i.e. ``groups=1``, ``Conv1d`` in V2)
    and ``cif_output``.  ``__call__`` returns the 4-tuple
    ``(acoustic_embeds, token_num, alphas, cif_peak)`` of the reference.
    
```

## paraformer.py :: __call__

```
Return ``(acoustic_embeds, token_num, alphas, cif_peak)``.
```

## paraformer.py :: ParaformerSANMDecoder

```
``funasr.models.paraformer.decoder.ParaformerSANMDecoder`` (offline path).

    Three stacks: ``decoders`` (``att_layer_num`` layers with SANM self-attention
    *and* cross-attention), optional ``decoders2`` (self-attention only) and a
    single ``decoders3`` feed-forward-only layer.
    
```

## paraformer.py :: __call__

```
``hs_pad`` is the encoder/acoustic memory ``(B, T1, D)``.

        Following the reference (``funasr/models/paraformer/decoder.py:397-450``)
        the offline path does **not** run ``self.embed``: ``x = tgt`` with
        ``tgt = ys_in_pad``.  Only ``forward_one_step`` (``:604``) embeds.  In
        offline Paraformer the decoder input is therefore the CIF acoustic
        embedding, already ``(B, T2, D)`` -- not token ids.
        
```

## paraformer.py :: Paraformer

```
``funasr.models.paraformer.model.Paraformer`` (inference only).
```

## paraformer.py :: state_dict

```
``{pytorch_name: mlx_array}`` in MLX layout.
```

## paraformer.py :: load_checkpoint

```
Strict load; see :func:`funasr_mlx.models.sanm.load_strict`.
```

## paraformer.py :: greedy_decode

```
Greedy token ids per utterance.

        Mirrors ``Paraformer.inference`` without beam search: ``argmax`` over the
        decoder posteriors followed by dropping ``sos``/``eos``/``blank_id``
        (the reference wraps the sequence in sos/eos and then strips it).
        
```

## paraformer.py :: timestamps

```
CIF-peak based character timestamps in milliseconds.

        Returns ``(tokens, timestamps)`` where ``timestamps[i]`` is a list of
        ``[start_ms, end_ms]`` pairs aligned with ``tokens[i]``.
        
```

## paraformer.py :: ts_prediction_lfr6_standard

```
Port of ``funasr.utils.timestamp_tools.ts_prediction_lfr6_standard``.

    Only the millisecond ``res`` output of the reference is produced here (the
    text dump and the ``<sil>`` bookkeeping are dropped).  ``upsample_rate`` must
    be ``1`` to match Paraformer's inference call site.
    
```

## paraformer.py :: _cif_fire_positions

```
Fallback used when the CIF peaks do not line up with the token count.

    Mirrors ``funasr.models.paraformer.cif_predictor.cif_wo_hidden``: a frame
    "fires" when the running integral crosses an integer boundary.
    
```

## paraformer_streaming.py :: StreamSinusoidalPositionEncoder

```
``transformer/embedding.py::StreamSinusoidalPositionEncoder``.

    Stateful through ``cache['start_idx']``: each chunk's positions continue
    from where the previous chunk ended, which is what makes the streamed
    encoding equal the offline one.
    
```

## paraformer_streaming.py :: _attn_forward_chunk

```
``MultiHeadedAttentionSANM.forward_chunk`` on the shared offline class.

    Mirrors ``sanm/attention.py:329-366``: compute qkv for the (stitched)
    chunk, merge cached ``k``/``v`` (look-back), commit each chunk's
    right-context-free stride into the cache, attend, add the FSMN memory.
    
```

## paraformer_streaming.py :: _predictor_forward_chunk

```
``CifPredictorV2.forward_chunk`` (``cif_predictor.py:316-412``).

    Returns ``(acoustic_embeds, token_length, alphas, fires)`` with the carry
    state written back into ``cache``.
    
```

## paraformer_streaming.py :: _cross_attn_forward_chunk

```
``MultiHeadedAttentionCrossAtt.forward_chunk`` (attention.py:815+ for cross).

    Cross-attention to the (growing) encoder memory with the same tail-window
    cache discipline as the decoder self-attention.
    
```

## paraformer_streaming.py :: ParaformerStreaming

```
``ParaformerStreaming`` -- chunk-wise streaming Paraformer (MLX).

    Reuses the offline module tree (identical parameter names); adds
    ``init_cache`` / ``forward_chunk`` implementing the reference streaming
    protocol. ``generate`` feeds fixed 600 ms chunks ([0,10,5] x 960 samples).
    
```

## paraformer_streaming.py :: init_cache

```
``ParaformerStreaming.init_cache`` (model.py:511-552), MLX arrays.
```

## paraformer_streaming.py :: predictor_chunk

```
``calc_predictor_chunk`` -> ``CifPredictorV2.forward_chunk``.
```

## paraformer_streaming.py :: decoder_chunk

```
``cal_decoder_with_predictor_chunk`` -> decoder ``forward_chunk``.
```

## paraformer_streaming.py :: forward_chunk

```
One chunk end-to-end. Returns ``(am_scores, token_length)``.

        ``None`` when the chunk fired no tokens (mirrors ``generate_chunk``
        returning ``[]`` when ``max(pre_token_length) < 1``).
        
```

## paraformer_streaming.py :: streaming_config_from_dict

```
Build the streaming config from the official online yaml mapping.
```

## qwen3_asr.py :: _Conv2dStride2

```
``nn.Conv2d(h, h, 3, stride=2, padding=1)`` in MLX (NHWC).
```

## qwen3_asr.py :: Qwen3ASRAudioEncoder

```
``Qwen3ASRAudioEncoder`` (inference), single-utterance path.

    Input mel: ``(F, T)`` (n_mels first, like the reference). Output:
    ``(1, T', output_dim)`` -- one embedding row per audio token slot,
    which the thinker splices between ``audio_start`` and ``audio_end``.
    
```

## qwen3_asr_thinker.py :: _decoder_dtype

```
Dtype the decoder activations must follow.

        The token embedding produces the activations the decoder consumes, so
        its OUTPUT dtype is the one to cast spliced audio to.  Read through
        :func:`activation_dtype`, not ``weight.dtype``: a quantized embedding
        stores packed ``uint32`` codes in ``weight``, and casting the audio
        embeddings to ``uint32`` destroys them (see the helper's docstring).
        
```

## qwen3_asr_thinker.py :: _tower_dtype

```
Dtype of the audio tower's checkpoint weights (first conv).
```

## qwen3_asr_thinker.py :: encode_audio

```
``(F, T)`` mel -> ``(1, T', text.hidden_size)`` tower embeddings.

        The front-end always produces float32 mel, so the mel is cast to the
        tower's **weight** dtype first (same rule as whisper's
        ``AudioEncoder.__call__``); without it an fp16 checkpoint runs the
        conv stack and every tower Linear in mixed precision.  Interleaved
        A/B on M4 (25 rounds, .tmp/task4/tower-tight-*.json): 55.7 -> 44.0 ms
        at the real 420-frame shape and 273.9 -> 214.7 ms at 30 s, i.e. 1.27x
        at both dtypes.  For a float32 checkpoint the cast is a no-op, so
        those numerics stay bit-identical.
        
```

## sanm.py :: sequence_mask

```
Port of ``funasr.models.scama.utils.sequence_mask``.

    Returns a ``(batch, maxlen)`` array holding ``1.0`` on valid positions and
    ``0.0`` on padded ones.  ``lengths`` may be ``(B,)`` or ``(B, 1)``.
    
```

## sanm.py :: make_pad_mask

```
Port of ``funasr.models.transformer.utils.nets_utils.make_pad_mask``.

    Returns a boolean ``(batch, maxlen)`` array that is ``True`` on padded
    positions (i.e. the logical inverse of :func:`sequence_mask`).
    
```

## sanm.py :: constant_pad_1d

```
``torch.nn.ConstantPad1d`` port.

    MLX convolutions are **channels-last**: ``mx.conv1d`` consumes ``NLC`` and
    ``mx.conv2d`` consumes ``NHWC``.  The reference SANM modules therefore pad
    the *time* axis of an ``NLC`` tensor, which is ``axis=1`` -- not the last
    axis as it is for PyTorch's ``NCL`` tensors.  ``axis`` is explicit so the
    intent is visible at every call site.
    
```

## sanm.py :: decimate_mask

```
Port of the ``x_mask[:, :, :-2:2][:, :, :-2:2]`` subsampling masks.

    ``torch`` slices with an explicit stop index (``[:-2]`` stops at
    ``length - 2``); MLX negative-stop semantics are easy to get subtly wrong,
    so the equivalent index vector is built explicitly and applied with
    :func:`mx.take`.  ``stride=2`` models ``[:-2:2]`` and ``stride=1`` models
    ``[:-2:1]``; both stop at ``length - end_offset``.
    
```

## sanm.py :: convert_checkpoint_tensor

```
Convert one PyTorch checkpoint tensor into the MLX layout (shared kernel).
```

## sanm.py :: flatten_state_dict

```
Flatten a module tree into FunASR/PyTorch-compatible parameter names.

    Names are built from the Module's **mapping storage** (see
    :func:`_flatten_into`), which is the only place MLX can hold a tensor.  The
    result matches what ``torch.nn.Module.state_dict()`` produces for the
    corresponding PyTorch modules.
    
```

## sanm.py :: flatten_state_dict_with_rules

```
Like :func:`flatten_state_dict`, but also returns the per-key layout rule.
```

## sanm.py :: checkpoint_state_dict

```
Return the module's parameters as ``{pytorch_name: mlx_array}``.

    The names are byte-compatible with the FunASR ``.pt`` checkpoints; the
    arrays are already in MLX layout.
    
```

## sanm.py :: __call__

```
Chain blocks with canonical ``nn.Sequential`` semantics.

        The leading argument is threaded through the chain while the remaining
        positional arguments and the keywords stay fixed.  Feeding a block's
        *return value* back as ``*args`` instead would unpack an ``mx.array``
        along axis 0 and hand the next block ``B`` separate arguments, which
        only works at batch size 1 or when the arities happen to line up.
        
```

## sanm.py :: repeat

```
Port of ``funasr.models.transformer.utils.repeat.repeat``.
```

## sanm.py :: LayerNorm

```
``funasr.models.transformer.layer_norm.LayerNorm`` (eps ``1e-12``).

    Parameter names are ``weight``/``bias`` and the normalisation axis is the
    last one, exactly like the PyTorch original.  MLX performs the reduction in
    float32 internally, which matches ``torch.nn.LayerNorm`` for float32 inputs.
    
```

## sanm.py :: Float32LayerNorm

```
``funasr.models.sense_voice.model.LayerNorm`` (eps ``1e-5``, fp32 math).

    SenseVoice keeps its own ``LayerNorm`` that inherits the ``torch`` default
    ``eps=1e-5`` and forces the reduction to float32 before casting the result
    back to the input dtype.  Reproducing the explicit cast keeps the fp16 path
    numerically identical to the reference.
    
```

## sanm.py :: SinusoidalPositionEncoder

```
``funasr.models.transformer.embedding.SinusoidalPositionEncoder``.

    Note the reference divides ``log(10000)`` by ``depth / 2 - 1`` (not by
    ``depth / 2``), i.e. positions start at ``1`` and the wavelength exponent
    denominator is ``depth / 2 - 1``.  Both quirks are preserved.
    
```

## sanm.py :: PositionwiseFeedForward

```
``funasr.models.transformer.positionwise_feed_forward.PositionwiseFeedForward``.
```

## sanm.py :: PositionwiseFeedForwardDecoderSANM

```
``funasr.models.sanm.positionwise_feed_forward.PositionwiseFeedForwardDecoderSANM``.

    Two details matter for checkpoint compatibility: ``w_2`` has **no bias** and
    the ``norm`` sits *between* the two linears and uses ``eps=1e-12``.
    
```

## sanm.py :: MultiHeadedAttentionSANM

```
``funasr.models.sanm.attention.MultiHeadedAttentionSANM``.

    Fused QKV projection (``linear_q_k_v``) plus a depth-wise FSMN memory block
    whose output is added to the attention output.  The reference applies
    ``d_k ** -0.5`` to the queries *after* splitting heads (equivalent to
    scaling the scores by ``1/sqrt(d_k)``).
    
```

## sanm.py :: _split_heads

```
``(B, T, n_feat)`` -> ``(B, head, T, d_k)``.
```

## sanm.py :: MultiHeadedAttentionSANMDecoder

```
``funasr.models.sanm.attention.MultiHeadedAttentionSANMDecoder``.

    Pure FSMN memory block over the decoder input.  ``cache`` mirrors the
    reference streaming state; offline inference passes ``None``.

    Deviation: the cache is kept channels-last ``(b, L, d)`` whereas the
    reference keeps ``(b, d, L)``.  See the module docstring's "Known
    deviations" section -- this matters only for the streaming/export contract,
    not for offline inference.
    
```

## sanm.py :: __call__

```
``inputs`` is ``(B, T, D)`` (MLX NLC); ``cache`` mirrors the reference.
```

## sanm.py :: MultiHeadedAttentionCrossAtt

```
``funasr.models.sanm.attention.MultiHeadedAttentionCrossAtt``.
```

## sanm.py :: EncoderLayerSANM

```
``funasr.models.sanm.encoder.EncoderLayerSANM`` (offline inference path).

    The first layer (``encoders0``) has ``in_size != size``; in that case the
    self-attention residual is dropped exactly like the reference does.
    
```

## sanm.py :: DecoderLayerSANM

```
``funasr.models.paraformer.decoder.DecoderLayerSANM``.

    Note the ordering: the feed-forward runs **first** and the self-attention
    residual is taken against the block *input* (``residual``), not against the
    feed-forward output.  ``src_attn`` may be ``None`` (the ``decoders2`` stack).
    
```

## sanm.py :: TooShortUttError

```
Port of ``funasr.models.transformer.utils.subsampling.TooShortUttError``.
```

## sanm.py :: check_short_utt

```
Port of ``funasr.models.transformer.utils.subsampling.check_short_utt``.
```

## sanm.py :: Conv2dSubsampling

```
``funasr.models.transformer.utils.subsampling.Conv2dSubsampling`` (1/4).

    Two strided 3x3 convolutions over the ``(B, T, F, 1)`` view followed by a
    linear projection.  Parameter keys stay ``conv.<i>.weight`` and
    ``out.0.weight`` thanks to :class:`ModuleStack`.

    MLX convolutions are channels-last, so the channel axis comes **last**
    (``(B, T, F, C)``).  The flatten before ``out.0`` is transposed back to
    PyTorch's ``(C, F)`` row-major order, which is what ``out.0`` was trained on.
    
```

## sanm.py :: Conv2dSubsampling2

```
``funasr.models.transformer.utils.subsampling.Conv2dSubsampling2`` (1/2).
```

## sensevoice.py :: SenseVoiceEncoderSmallConfig

```
Mirror of FunASR's ``encoder_conf`` for ``SenseVoiceEncoderSmall``.
```

## sensevoice.py :: CTCConfig

```
Mirror of FunASR's ``ctc_conf`` (only the inference-relevant fields).
```

## sensevoice.py :: SenseVoiceSmallConfig

```
Mirror of FunASR's ``model_conf`` for ``SenseVoiceSmall``.

    The defaults are the published ``SenseVoiceSmall`` values
    (``input_size=560``, 50 SANM blocks + 20 tail blocks, ``vocab_size=55465``).
    
```

## sensevoice.py :: sensevoice_config_from_dict

```
Build a :class:`SenseVoiceSmallConfig` from a FunASR ``model.yaml`` mapping.

    Accepts either the flat ``model_conf`` block or the whole document with
    ``model_conf``/``encoder_conf``/``ctc_conf`` sections.
    
```

## sensevoice.py :: CTC

```
``funasr.models.ctc.ctc.CTC`` restricted to its inference path.

    The only parameter is ``ctc_lo`` (``Linear(encoder_output_size, odim)``);
    the loss/argmax helpers of the reference are re-implemented on top of it.
    
```

## sensevoice.py :: _weight_dtype

```
Dtype of this encoder's checkpoint weights (``encoders0`` always exists).

        Read through :func:`~funasr_mlx.models.llm.quant.activation_dtype` rather
        than ``weight.dtype`` directly: the SANM stack is a documented
        quantization target (``audio_encoder.`` is in fun-asr-nano's scope), and
        a quantized Linear's ``weight`` holds packed ``uint32`` codes.  Casting
        the fbank input to ``uint32`` would destroy it exactly as the decoder
        embedding cast did.  The helper returns the side tensor's dtype for a
        quantized module, so this stays correct either way.
        
```

## sensevoice.py :: SenseVoiceSmall

```
``funasr.models.sense_voice.model.SenseVoiceSmall`` (inference only).
```

## sensevoice.py :: state_dict

```
``{pytorch_name: mlx_array}`` in MLX layout (no key rewriting).
```

## sensevoice.py :: load_checkpoint

```
Strict load; see :func:`funasr_mlx.models.sanm.load_strict`.
```

## sensevoice.py :: greedy_decode

```
Greedy CTC token ids per utterance (mirrors ``SenseVoiceSmall.inference``).
```

## sensevoice.py :: ctc_greedy_decode

```
``argmax`` -> ``unique_consecutive`` -> drop ``blank_id``.

    Faithful to ``SenseVoiceSmall.inference``: consecutive duplicates collapse
    *before* blanks are removed, so ``[a, blank, a]`` decodes to ``[a, a]``.
    
```

## whisper.py :: ModuleStack

```
Ordered children registered under their numeric index.

    ``torch.nn.Sequential`` yields ``<parent>.<i>.<leaf>`` parameter names; MLX's
    ``nn.Sequential`` would insert a ``layers`` level, so children are registered
    directly under ``"0"``, ``"1"``, ...
    
```

## whisper.py :: _rule_for

```
Conv1d is the only module whose MLX layout differs from PyTorch's.
```

## whisper.py :: LayoutReport

```
Flat parameter tensors plus the layout rule that applies to each key.

    Behaves like a read-only ``Mapping`` of ``{parameter_name: mlx_array}``, so it
    can be iterated, indexed, ``len()``-ed and fed to ``dict(...)`` exactly like a
    plain state dict, while also carrying the per-key rule used for the
    torch -> MLX layout conversion.
    
```

## whisper.py :: flatten_state_dict

```
Flatten a module tree into FunASR/PyTorch-compatible parameter names.

    The layout rule is derived from the owning module type rather than from a
    name pattern, so it cannot silently mis-convert a renamed submodule.
    Attributes starting with ``_`` are configuration, never weights.
    
```

## whisper.py :: flatten_state_dict_with_rules

```
Like :func:`flatten_state_dict`, but also returns the per-key layout rule.

    Returns ``(tensors, rules)`` with the same two dictionaries that the
    :class:`LayoutReport` carries -- nothing is dropped and the rule metadata is
    not recomputed by a second walk.  This is the contract
    :func:`funasr_mlx.models.sanm.flatten_state_dict_with_rules` and
    :func:`funasr_mlx.models.campplus.flatten_state_dict_with_rules` expose, so
    the three model modules agree on one name for the same operation; callers
    that want the richer object should use :func:`flatten_state_dict` instead.
    
```

## whisper.py :: LoadReport

```
Outcome of a :func:`load_strict` call.
```

## whisper.py :: load_strict

```
Load ``tensors`` into ``module``, refusing any key mismatch.

    Arrays must already be in MLX layout unless ``convert=True`` (raw PyTorch
    ``state_dict``).  Raises :class:`CheckpointKeyError` listing the offending
    keys, and ``ValueError`` on any shape mismatch -- never a partial load.
    
```

## whisper.py :: LayerNorm

```
``whisper.model.LayerNorm`` -- fp32 reduction, torch's default ``eps=1e-5``.

    The reference casts the input to ``float32`` before normalising and casts
    the result back to the input dtype; MLX's ``LayerNorm`` already accumulates
    in float32, so the explicit round-trip is kept for clarity and for the fp16
    path.
    
```

## whisper.py :: sinusoids

```
Port of ``whisper.model.sinusoids``.

    Note this is **0-based** and uses ``-log_increment * arange(channels // 2)``
    -- different from FunASR's ``SinusoidalPositionEncoder`` (1-based, positive
    increments), which the SANM blocks use.  Both quirks matter and both are
    preserved exactly.
    
```

## whisper.py :: cache_prefix

```
Cache key prefix for this attention (e.g. ``"blocks.0.attn"``).
```

## whisper.py :: _split

```
``(B, T, C)`` -> ``(B, H, T, d)``.
```

## whisper.py :: qkv_attention

```
Port of ``MultiHeadAttention.qkv_attention`` (the non-SDPA path).

        Layouts follow the reference exactly: ``q`` is ``(B, H, T, d)`` and
        ``k`` is **transposed** to ``(B, H, d, T)``.  That ``.permute(0, 2, 3, 1)``
        in the reference is what makes ``q @ k`` contract the head dimension and
        produce the ``(B, H, T, T)`` score matrix; leaving ``k`` as
        ``(B, H, T, d)`` would contract ``T`` with ``T`` instead.

        Whisper applies ``(n_state // n_head) ** -0.25`` to *both* ``q`` and
        ``k``, which is exactly a ``d_k ** -0.5`` factor on the scores.
        
```

## whisper.py :: ModelDimensions

```
Mirror of ``whisper.model.ModelDimensions``.
```

## whisper.py :: WhisperConfig

```
Mirror of ``funasr/models/whisper/model.py::Whisper``'s inference config.
```

## whisper.py :: whisper_config_from_dict

```
Build a :class:`WhisperConfig` from a FunASR ``model.yaml``-like mapping.

    ``WhisperParameters`` is also accepted as an alias for the ``parameters``
    section of FunASR's document.
    
```

## whisper.py :: AudioEncoder

```
``whisper.model.AudioEncoder``.

    ``mel`` is channels-last ``(B, T', n_mels)``; the reference's
    ``permute(0, 2, 1)`` after ``conv2`` is unnecessary under MLX's channels-last
    Conv1d.  ``positional_embedding`` is materialised (the reference registers it
    as a persistent buffer) so it round-trips through strict loading; the
    regenerated value is identical to :func:`sinusoids`.
    
```

## whisper.py :: _weight_dtype

```
Dtype of this module's checkpoint weights (``conv1`` always exists).
```

## whisper.py :: TextDecoder

```
``whisper.model.TextDecoder`` with an explicit MLX KV cache.

    The causal mask is **not** a parameter: the reference registers it with
    ``persistent=False``, so it never appears in a ``state_dict`` and is
    regenerated on demand here.
    
```

## whisper.py :: _weight_dtype

```
Dtype of this module's checkpoint weights (``ln`` always exists).
```

## whisper.py :: offset

```
Absolute position of the next token, from the self-attention cache.

        The reference reads ``next(iter(kv_cache.values())).shape[1]``, which
        relies on ``dict`` insertion order.  Here it is read deterministically
        from the first block's self-attention key.
        
```

## whisper.py :: embed_audio

```
Encode ``(B, 2 * n_audio_ctx, n_mels)`` mel frames.

        The encoder casts the mel to its own weight dtype (see
        :meth:`AudioEncoder.__call__`), so an fp16 checkpoint produces fp16
        audio features even though the front-end mel is float32.
        
```

## whisper.py :: reflect_pad_1d

```
``torch.nn.functional.pad(..., mode="reflect")`` along the last axis.

    The edge value is **not** repeated.  With ``x = [1, 2, 3, 4]`` and ``pad=2``
    on both sides the result is ``[3, 2, 1, 2, 3, 4, 3, 2]``: the left filler
    reads left-to-right as ``x[pad] ... x[1]`` and the right filler as
    ``x[n-2] ... x[n-1-pad]`` -- both *descending* index runs.
    
```

## whisper.py :: pad_or_trim

```
Port of ``whisper.audio.pad_or_trim`` (zero-pad or truncate).
```

## whisper.py :: _hz_to_mel

```
Slaney mel scale (librosa default, ``htk=False``).
```

## whisper.py :: _is_non_persistent_buffer

```
Whether ``owner``'s parameter ``key`` is a derived, non-persistent buffer.

    ``isinstance`` rather than an exact type match, so a subclass still matches.
    The result is independent of any flatten prefix or of whether the owning
    module is reached standalone or as part of a full ``Whisper``.
    
```

## whisper.py :: mel_filterbank

```
Slaney-normalised mel filterbank, shape ``(n_mels, n_fft // 2 + 1)``.

    Re-implementation of ``librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)``,
    which is how the reference's ``mel_filters.npz`` asset was produced.  **Not
    numerically verified against the official asset** -- if the official array is
    available, load it and pass it to :func:`log_mel_spectrogram` directly instead
    of relying on this function.
    
```

## whisper.py :: GreedyDecodeConfig

```
Bounds and logit filters for :func:`greedy_decode`.

    ``eot_token_id`` and ``space_token_id`` **must** be supplied by the caller
    (they depend on the BPE vocabulary of the particular checkpoint) or left as
    ``None`` to disable the corresponding filter.  No default is guessed.
    
```

## whisper.py :: GreedyDecodeResult

```
Token-level greedy output (ids only; decoding to text is not provided).
```

## whisper.py :: _suppress

```
Set the given vocabulary columns to ``-inf`` (a port of the logit filters).

    ``vocab`` (V,) is broadcast **against** the banned ids (N,) rather than the
    other way round, and the membership test is reduced over the N axis.  Building
    a ``(1, V)`` and a ``(1, N)`` array and comparing them would only broadcast
    when ``N == 1`` or ``N == V``; with two or more suppressed tokens and a
    vocabulary of a different size it raises instead of masking.
    
```

## awq.py :: ScaleSite

```
One scale search site: a norm (or Linear) feeding several Linears.

    Mirrors mlx-lm's ``ScaleConfig`` for the Qwen3 block layout:

    * ``prev`` -- the module whose OUTPUT is scaled (an RMSNorm for qwen3);
      its weight is divided by the scales.
    * ``layers`` -- the successor Linear weights are multiplied by the scales.
    * ``block`` -- optional sub-block to inspect (``self_attn``), because the
      loss must be measured on the attention output, not one Linear alone.
    
```

## awq.py :: _to_mx

```
Captured activations are numpy (host); the search wants mx arrays.

    The conversion happens once per consumer, so the device copy never
    exceeds the capped rows either.
    
```

## awq.py :: MultiBlockCapture

```
Capture Linear inputs for MANY blocks in ONE pass over the dataset.

    Running the model once per block (28 x 480 prefills) is what made the
    first calibration attempt take ~575s per block; mlx-lm's AWQ catches every
    Linear of every block in the same forward.  Memory stays bounded by
    stride-subsampling each Linear's rows down towards ``row_cap`` after
    every utterance (halving whenever over the cap) and storing float16.
    
```

## awq.py :: capture

```
Run ``run()`` with input hooks on this block's Linears.

        ``run`` is a callable that performs the model's real prefill for one
        utterance; the block must be part of the live model.  Each Linear's
        input rows are appended to ``input_feat`` on the module (mlx-lm's
        convention), in float32, and the block's own input is recorded on
        ``self.block_input`` for the never-worse reference.
        
```

## awq.py :: _run_block

```
Run one decoder block on ``x`` [1, rows, hidden] in row batches.

        Our ``Qwen3TransformerBlock`` has no ``__call__`` of its own (the
        forward is unrolled in ``Qwen3Model.__call__``), so the qwen3 residual
        structure is applied here: ``h += attn(norm(h)); h += mlp(norm(h))``.
        The site's sub-block (``self_attn`` / ``mlp``) IS callable directly.
        
```

## awq.py :: search_scale

```
Grid-search channel scales for one site; returns the best scales.

        The reference output is the block (or sub-block) run on the captured
        input of the site's FIRST layer; each candidate scales that input's
        channels via the successor weights only (equivalent math, see
        upstream), quantize-dequantizes them, and the MSE of the sub-block's
        output picks the winner.
        
```

## awq.py :: apply_scale

```
Fold the chosen scales into the site's modules (bf16 stays bf16).

        Two fold shapes, exactly as mlx-lm: a Linear predecessor divides its
        OUTPUT-channel rows (``weight / scales[:, None]`` -- the scales index
        the successor's input channels, which are the predecessor's outputs),
        while a norm predecessor divides elementwise (its weight is a vector
        over the same channels it normalises).
        
```

## calibrate.py :: HeadTarget

```
Which module in a live model is the LM head, and how it is called.

    ``kind`` is ``"tied"`` when the head is the token embedding used through
    ``as_linear`` (qwen3 / moss) and ``"linear"`` when the checkpoint has a
    separate ``lm_head`` (glm).  ``hidden_size`` is the head's *input*
    dimension -- the size of the Hessian.
    
```

## calibrate.py :: resolve

```
Locate the head on ``model`` and describe how it is called.

        ``head_path`` is a dotted path to the head module (e.g.
        ``thinker.model.embed_tokens`` or ``language_model.lm_head``).
        
```

## gptq.py :: _round_half_away

```
``round()`` with ties away from zero, as the MLX kernel does.

    NumPy's ``np.round`` is half-to-even, which disagrees with the kernel on
    exact .5 ties; they are rare on real weights but not on synthetic ones.
    
```

## gptq.py :: AffineQuantizer

```
MLX affine group quantization, bit-exact (see module docstring).

    ``bits`` and ``group_size`` must be a combination ``mx.quantize`` accepts:
    bits in (2, 3, 4, 5, 6, 8) and group size in (32, 64, 128).  ``storage``
    names the dtype the side tensors are stored in; it only affects rounding of
    ``scales``/``biases`` and therefore which codes are chosen.
    
```

## gptq.py :: storage_dtype

```
NumPy dtype the side tensors are returned in.

        bfloat16 has no NumPy representation, so it is carried as float32 whose
        *values* are already bf16-rounded -- converting to bf16 later is exact.
        
```

## gptq.py :: cast

```
Round ``values`` to the storage dtype, keeping a float32 carrier.
```

## gptq.py :: GptqConfig

```
GPTQ hyper-parameters, mirroring the reference implementation's knobs.
```

## gptq.py :: _upper_cholesky_inverse

```
``U`` with ``Hinv = U^T U`` and ``U`` upper triangular (float32 out).

    The reference implementation goes ``chol(H) -> cholesky_inverse -> chol(·,
    upper=True)``; ``inv(H)`` then an upper Cholesky is the same matrix with one
    fewer decomposition, and the dampening has already made ``H`` positive
    definite.
    
```

## gptq.py :: gptq_quantize

```
Quantize ``weight`` ``[out, in]`` with GPTQ using ``hessian`` ``[in, in]``.

    ``hessian`` is the second-order statistic of the layer input,
    ``H = 2 X^T X`` accumulated in float32 (:class:`HessianAccumulator` does
    exactly that).  The returned triple is what a W4A16 checkpoint stores, in
    the standard MLX affine group layout.
    
```

## gptq.py :: rtn_quantize

```
Plain round-to-nearest group quantization (``mx.quantize``'s own rule).

    Provided as the baseline GPTQ has to beat, computed with the *same*
    quantizer so the comparison isolates the algorithm and not the rounding.
    Pass ``hessian`` to also get the layer-output error, which is the metric the
    head-to-head claim is made on.
    
```

## gptq.py :: _pack_triple

```
Re-derive the stored triple from a reconstructed matrix.

    Deriving the triple from the *final* reconstructed values (rather than
    keeping the per-column codes) guarantees the checkpoint is self-consistent:
    ``mx.dequantize(packed, scales, biases)`` reproduces exactly what GPTQ
    decided, with no dependency on how the inner loop was organized.
    
```

## gptq.py :: _weighted_effective_rows

```
Per-language effective rows after weighting, total preserved.

        The renormalisation is done in EFFECTIVE-ROW space, which is the space
        the 30% guard is stated in and the only one where "total mass" is
        meaningful.  An earlier version renormalised the multipliers instead,
        which silently changed the overall scale of the Hessian and therefore the
        damping (defined relative to the mean diagonal).
        
```

## gptq.py :: _effective_rows

```
Per-language row count the Hessian effectively sees.

        Both modes are one family -- a common row threshold ``T`` with
        ``rows_i -> min(rows_i, T)``:

        * ``balance`` picks ``T = min(rows)``, giving every language an equal
          share;
        * ``cap_fraction`` bisects for the *largest* ``T`` whose resulting
          maximum share is still within the cap, so the most calibration data is
          kept subject to the guard.
        
```

## gptq.py :: _row_weights

```
Multiplier applied to each language's accumulated ``X^T X``.

        The stored products are already sums, so an unweighted Hessian needs a
        multiplier of **1**, not the row count.
        
```

## gptq.py :: language_mass

```
Share of the Hessian's effective rows per language (the guard's check).

        Mirrors :meth:`hessian` exactly, so the reported shares are the weights
        actually applied to the accumulated products.
        
```

## gptq.py :: hessian_from_activations

```
Convenience: ``H = 2 X^T X`` over a stream of activation batches.
```

## kv_cache.py :: KVCache

```
mlx-lm ``KVCache``: append-only keys/values with an offset cursor.

    ``reserve`` pre-allocates the buffer once (call it after prefill, when the
    final length is known: audio positions + prompt + max decode steps). Without
    it, the cache grows in ``step``-sized chunks, and every growth allocates a
    fresh buffer and copies the whole history -- measured ~3 ms per event on the
    0.6B, a 2-3x tax on a 7 ms decode step, hitting exactly when a long
    utterance crosses a 256-position boundary mid-decode.
    
```

## kv_cache.py :: reserve

```
Grow the backing store to ``positions`` capacity immediately.

        ``like`` may carry the (B, H, D) shape when this cache is still empty.
        
```

## quant.py :: QuantizationSpec

```
One family's quantization plan, as declared in ``config.json``.

    ``method`` records *how* the stored ``scales``/``biases``/codes were
    computed: ``rtn`` is ``mx.quantize``'s round-to-nearest, ``gptq`` is
    Hessian-based error compensation (:mod:`funasr_mlx.models.llm.gptq`).  The
    runtime layout is identical either way -- ``nn.QuantizedLinear`` and
    ``mx.quantized_matmul`` cannot tell them apart -- so the loader ignores it
    and it exists purely as provenance in the checkpoint's ``config.json``.
    
```

## quant.py :: selects_embedding

```
Whether a token-embedding module at ``module_path`` is selected.

        The tied ``embed_tokens`` is the one non-Linear weight worth offering:
        when ``tie_word_embeddings`` is set the same matrix is also the LM head,
        so at bs=1 it is streamed once per step and is a large share of the step
        (311 MB bf16 for a 151936-vocab family). It needs its own predicate
        because ``nn.Embedding`` is not an ``nn.Linear`` subclass, so the Linear
        rule can never reach it; a family opts in by naming ``embed_tokens`` in
        ``targets``.
        
```

## quant.py :: quantizes_embedding

```
Converter predicate for the token-embedding matrix.
```

## quant.py :: is_linear_weight

```
Whether ``name`` is a Linear weight the target prefixes select.

        Rank-2 ``.weight`` under a selected prefix and not one of the
        :data:`NON_LINEAR_WEIGHT_NAMES` exceptions -- i.e. a tensor the model
        side would have turned into a ``QuantizedLinear`` if the group size
        fitted. This is the predicate the converter uses to decide whether a
        tensor it did *not* pack is a legitimate skip or an unnoticed miss.
        
```

## quant.py :: quantizes_tensor

```
Whether the converter should quantize checkpoint tensor ``name``.

        Mirrors the model-side predicate exactly: a Linear weight (see
        :meth:`is_linear_weight`) or a selected token embedding (see
        :meth:`quantizes_embedding`), whose input dimension the group size
        divides.
        
```

## quant.py :: quantizes_module

```
Whether ``module`` (a live ``nn.Module``) should become quantized.

        The token embedding is a separate case from ``nn.Linear``: it is not a
        ``Linear`` subclass, so it can only be selected through an explicit
        ``embed_tokens`` target prefix, and it is the tied LM head as well.
        
```

## quant.py :: apply_quantization

```
Quantize ``model`` in place per the ``config.json`` section ``data``.

    Called by :func:`funasr_mlx.pipeline.load_model` *before* the expected
    parameter set is computed, so a quantized checkpoint's ``.scales`` and
    ``.biases`` entries are part of the strict key/shape check.  Returns
    ``None`` when the configuration carries no quantization section, i.e. the
    checkpoint is full precision.
    
```

## qwen3.py :: Qwen3ForCausalLM

```
``Qwen3ForCausalLM``: the model plus the (tied or separate) LM head.
```

## rope.py :: initialize_rope

```
mlx-lm ``models/rope_utils.py::initialize_rope`` (inference subset).

    ``default`` (and absent config) and ``linear`` are supported; anything
    else raises rather than silently degrading position handling.
    
```

## splice.py :: splice_and_forward

```
Replace the ``pad_span``-wide audio-token run with tower output; decode.

    Prefill needs the causal mask even when KVCache objects already exist:
    the generic model call only builds one when ``cache is None`` as a whole.
    ``decoder_dtype``: the tower emits float32 embeds (float32 mel in); the
    cast prevents a mixed-dtype promotion at every Linear of the stack (a
    no-op for float32 checkpoints).
    
```

