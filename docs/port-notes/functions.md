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

