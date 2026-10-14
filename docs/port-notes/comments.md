# Inline comment blocks extracted from the code

(Moved verbatim; find by file:line.)

## api.py

### api.py:71

```
    # Library callers load in-process by default (the guard is a host choice,
    # not a project requirement); the CLI subprocess stays as the fallback so
    # the process boundary still applies where a caller wants it.
```

## checkpoint_stream.py

### checkpoint_stream.py:25

```
#: safetensors type codes the bounded writer can emit directly, with their
#: bytes per element. Includes ``BF16``, which NumPy cannot name as a dtype:
#: a bfloat16 payload is carried as raw ``uint16`` words and written under the
#: ``BF16`` code, so the bits reach disk unchanged. ``U32`` is the packed-code
#: width of group quantization, which is why a W4A16 shard mixes it with the
#: 2-byte ``BF16`` scales and needs a per-tensor code map.
```

## checkpoints.py

### checkpoints.py:23

```
    # LLM-based ASR families (safetensors sources; see SOURCE_RULES).
```

### checkpoints.py:29

```
#: Output precisions ``convert_checkpoint`` may emit. ``bfloat16`` is only
#: reachable through the streaming writer (NumPy has no bfloat16), so it is
#: admitted for families whose canonical dtype is bf16.
```

### checkpoints.py:55

```
#: The five LLM-ASR families. Fun-ASR-Nano's ``llm.*`` subtree is an
#: unmodified ``Qwen3ForCausalLM`` state dict (PROGRESS.md), so it is bound
#: under the model's ``llm.`` attribute with the prefix stripped. That family's
#: source is a torch ZIP and keeps the identity rule; the rest are safetensors.
#:
#: ``llm.lm_head.weight`` is absent from the native fun-asr-nano checkpoint
#: because the Qwen3 backbone is built with ``tie_word_embeddings=True`` and the
#: port reaches the head through ``embed_tokens.as_linear``; the source ships a
#: separate (numerically identical) copy that the strict loader would reject.
```

### checkpoints.py:66

```
        # The audio tower is a top-level attribute of the native model, but the
        # decoder stays under ``thinker.`` (``Qwen3ASR.thinker`` wraps
        # ``Qwen3ForCausalLM``, whose transformer is its own ``model``).
```

### checkpoints.py:72

```
        # ``lm_head`` is dropped because the decoder is tied
        # (``tie_word_embeddings=True``) and the port reaches the head through
        # ``embed_tokens.as_linear``. The audio tower's sinusoidal table is not
        # in the checkpoint at all: the port rebuilds it from the config, so it
        # is declared in :data:`DERIVED_NATIVE_PARAMETERS` instead.
```

### checkpoints.py:78

```
            # The port folds the projector into the encoder, so its two linears
            # sit under ``audio_tower.`` rather than beside it.
```

### checkpoints.py:85

```
        # ``Qwen3ForCausalLM`` nests the transformer under ``model``, so the
        # source's ``language_model.<...>`` becomes ``language_model.model.<...>``.
```

### checkpoints.py:89

```
#: Sentinel ``source_rule`` for a source that is already in the native layout
#: (a converted bf16 checkpoint being re-emitted as W4A16, for instance). Keys
#: pass through unchanged; the strict loader still checks every name.
```

### checkpoints.py:91

```
#: Families whose source tensors are bfloat16 and whose canonical native
#: dtype is therefore bfloat16 (no lossy cast on conversion).
```

### checkpoints.py:93

```
#: Families whose source is a torch ZIP rather than safetensors.
```

### checkpoints.py:95

```
#: Native parameters a port *derives* at construction rather than loading from a
#: checkpoint, so the strict key diff must not expect them on disk. Listed here
#: as data beside the layout rules and consumed by ``pipeline.load_model``,
#: which validates that each declared name really is a model parameter (a stale
#: declaration is an error, not a silent skip).
#:
#: ``qwen3-asr`` builds its sinusoidal position table from the config in
#: ``_Sinusoids``; the source checkpoint stores no equivalent tensor.
```

### checkpoints.py:114

```
#: Suffix added to a quantized ``<module>.weight`` for MLX's group-quantization
#: side tensors. ``nn.QuantizedLinear`` owns exactly these three parameters, so
#: the emitted names are what the strict loader expects to find.
```

### checkpoints.py:158

```
    # NumPy cannot name bfloat16, so the 16-bit side tensors go back out as raw
    # uint16 carriers under the same BF16 code the full-precision path uses.
```

### checkpoints.py:179

```
    #: ``gptq`` runs Hessian-based error compensation; ``rtn`` runs plain
    #: round-to-nearest **through this same code path**, using the identical
    #: packing, side-tensor layout and dtype handling.  That is what makes an
    #: rtn-vs-gptq comparison honest: both arms differ only in how the stored
    #: values were computed, so any measured difference is the algorithm's.
    #: ``rtn`` still requires a Hessian entry (the matrix is what marks a tensor
    #: as GPTQ-path); pass an identity if none was collected.
```

### checkpoints.py:181

```
    #: Per-tensor reported outcome, filled in as tensors are quantized.
```

### checkpoints.py:215

```
        # Convert to float32 in row blocks.  ``weight.astype(mx.float32)`` on a
        # 151936x1024 head would materialise a 622 MB float32 array on the device
        # and ``np.array`` a second 622 MB one on the host, on top of the
        # bfloat16 original -- enough to push the conversion past the guard's
        # budget on its own.  Blocked conversion keeps a few tens of MB live at a
        # time, and the host array it fills is the one GPTQ consumes.
```

### checkpoints.py:222

```
        # ``np.asarray`` is a no-op when the Hessian is already float32, which is
        # how the calibrator writes it; a copy here would double it for nothing.
```

### checkpoints.py:229

```
        # Side tensors must land in the checkpoint dtype (bfloat16/float16), the
        # same as mx.quantize would produce, or the runtime would promote.
```

### checkpoints.py:230

```
        # ``mx.array`` copies straight into the device buffer, so the host arrays
        # are dropped as soon as the tensors are evaluated.
```

### checkpoints.py:363

```
        # SANM fsmn_block: torch (D, 1, K) -> MLX Conv1d (D, K, 1).
```

### checkpoints.py:368

```
        # Qwen3-ASR audio tower: torch Conv2d (O, I, KH, KW) -> MLX (O, KH, KW, I).
```

### checkpoints.py:371

```
        # torch Conv1d (O, I, K) -> MLX Conv1d (O, K, I).
```

### checkpoints.py:402

```
    # guard optional (host choice); policies only bound when present
```

### checkpoints.py:415

```
    # A safetensors source of a non-torch family must declare its layout rule
    # (otherwise nothing renames its keys onto the native names). A torch
    # source's layout lives in convert_tensor, but a rule may still be passed
    # for its `drop` list (fun-asr-nano's tied lm_head).
```

### checkpoints.py:471

```
    # A bfloat16 source kept at bfloat16 stays a raw uint16 carrier (NumPy has
    # no bfloat16) and is written under the BF16 code, exactly as the
    # safetensors path does. Anything else is rounded on the host, in place.
```

### checkpoints.py:477

```
        # Exactly the two keys every already-published native shard carries;
        # the output dtype belongs in the index metadata, not here.
```

### checkpoints.py:533

```
        # ascontiguousarray promotes a 0-d array to shape (1,), which would
        # corrupt the rank-0 BatchNorm training counters; keep those as-is.
```

### checkpoints.py:581

```
#: safetensors type code -> NumPy dtype used to carry the payload in memory.
#: ``BF16`` maps to ``uint16``: NumPy has no bfloat16, so a bfloat16 tensor is
#: moved around as its raw 16-bit words and only ever written under the
#: ``BF16`` code. Every other code maps to its exact NumPy equivalent.
```

### checkpoints.py:660

```
    # Bound here rather than at module scope: ``checkpoints`` is imported by the
    # package root, and the runtime requires MLX imports to stay lazy.
```

### checkpoints.py:683

```
    # Resolve every source key to its native name up front, so a key outside
    # the declared layout fails before anything is written to disk.
```

### checkpoints.py:688

```
    # One output code per shard, so a bf16 checkpoint stays bf16 on disk. A
    # quantized run mixes widths -- 4-byte packed codes beside 2-byte group
    # scales -- so those shards carry a per-tensor code map instead.
```

### checkpoints.py:702

```
        # Two keys only, matching every already-published native shard; the
        # output dtype is recorded in the index metadata instead.
```

### checkpoints.py:752

```
                # A bfloat16 source kept at bfloat16 stays a raw uint16 carrier
                # and is written under the BF16 code; anything else is rounded
                # to the requested precision here, on the host.
```

### checkpoints.py:754

```
                    # A quantized Linear must land in ONE shard together with its
                    # scales/biases: `pipeline.load_model` binds the group as a
                    # unit, and splitting it across shards is not supported. So
                    # flush whatever is buffered, then check the group itself
                    # against the loader's per-shard bound -- `max(256MB,
                    # largest_tensor) + 32MB` -- and fail loudly rather than
                    # emit a checkpoint that cannot be loaded.  The bound is
                    # sized by the LARGEST group the conversion itself packs
                    # (a 1.7B tied embedding at w4/g64 packs to ~175MB), so a
                    # legal conversion can always be loaded on the machine
                    # class the memory policy targets: one mmap's wired swing
                    # is bounded by one shard, and the measured conversion
                    # peak (~4.4GB on 16GB hosts) leaves 2x headroom to the
                    # per-process wired limit.
```

### checkpoints.py:756

```
                    # Per-target bit width: a w8 head beside a w4 stack (see
                    # QuantizationSpec.bits_for).  This must match what
                    # apply_quantization builds at load time or the strict
                    # loader rejects the shapes.
```

### checkpoints.py:769

```
                    # A GPTQ pass on a large head allocates several hundred MB of
                    # host scratch.  MLX's allocator is capped by the policy but
                    # keeps freed blocks in its cache, so hand them back before
                    # the next tensor rather than letting the cache grow to its
                    # limit on top of the Python-side peak.
```

### checkpoints.py:800

```
                    # A Linear the group size cannot divide (e.g. fun-asr-nano's
                    # (1536, 560) q_k_v). The model predicate skips it too, so
                    # the loader stays consistent -- but it is recorded, because
                    # an unnoticed skip would silently leave the model partly
                    # full precision.
```

## cli.py

### cli.py:169

```
    # Snipped fbank frames do not consume a trailing incomplete frame.
```

## download.py

### download.py:11

```
#: The 14 published repos (``baicai1145/*`` on both platforms).
```

### download.py:101

```
        # ModelScope primary (measured 20.4 MB/s vs HF 5.6 MB/s on the same
        # shard set from this network); HF is the per-file fallback.
```

## frontend.py

### frontend.py:64

```
    # Reference (wav_frontend.apply_lfr): prepend (m-1)//2 COPIES of the first
    # row, then take strided windows of m rows. Clipping instead would repeat
    # the first frame one extra time at the left edge (4 copies for m=7), which
    # shifts every fused row's content by one frame.
```

## llm_asr.py

### llm_asr.py:10

```
#: family -> (tokenizer source default subdir name, eos token ids)
```

### llm_asr.py:56

```
            # tokenizer files may live in the checkpoint dir or in a sibling
            # fp16 source dir (quant dirs don't always carry them)
```

### llm_asr.py:116

```
        # load_model does not retain the path; the API layer passes the
        # checkpoint directory and tokenizer files live alongside.
```

### llm_asr.py:299

```
    # infer from directory name
```

## memory.py

### memory.py:37

```
    #: macOS frees purgeable pages on demand, so "available" snapshots deny
    #: jobs the compressor would satisfy. The RSS ceilings stay binding;
    #: enable only for jobs that cannot tolerate paging at all.
```

### memory.py:134

```
                    # Herdr stores worktrees under the lead repository.
```

### memory.py:161

```
    # The macOS purgeable/paging path frees memory on demand; an "available"
    # snapshot would wrongly deny jobs the compressor can satisfy. The
    # external RSS watchdog (project ceiling + per-process kill) stays the
    # binding constraint, so min_available is advisory only by default.
```

### memory.py:173

```
    # Configure before constructing any model or evaluating any array.
```

## pipeline.py

### pipeline.py:29

```
    # YAML 1.1 reads bare exponent forms like 1e-05 as strings, silently
    # corrupting numeric config fields. JSON is the artifact format, so parse
    # it as JSON first and only fall back to YAML for hand-written configs.
```

### pipeline.py:39

```
    # guard is a host choice, not a project requirement
```

### pipeline.py:92

```
    # The guard is an OPTIONAL host-protection wrapper (CLI default for this
    # 16GB dev box), not a project requirement: library callers load directly.
```

### pipeline.py:113

```
    # Same bound the converter enforces when packing quantized groups (see
    # checkpoints.py): one shard bounds one mmap's wired swing.  256MB+32MB
    # admits the largest single group a legal conversion can emit (a 1.7B
    # tied embedding at w4/g64 is ~175MB) while keeping the per-file
    # residency predictable on 16GB hosts.
```

### pipeline.py:132

```
    # Weight quantization is applied BEFORE the expected parameter set is
    # computed, so a quantized checkpoint's `.scales`/`.biases` tensors are
    # part of the strict key/shape check like every other parameter. The
    # converter uses the same declaration to decide which tensors to pack, so
    # a mismatch between the two rules fails here instead of mis-binding.
```

### pipeline.py:139

```
    # A port may derive a parameter at construction instead of loading it (the
    # Qwen3-ASR sinusoidal position table is computed from the config). Those
    # names are declared as data beside the layout rules; each declaration is
    # checked against the built model here, so a stale entry fails loudly
    # instead of quietly excusing a missing tensor.
```

## pseudo_streaming.py

### pseudo_streaming.py:73

```
# Adapters: offline callables per family
```

## _layout_state.py

### _layout_state.py:25

```
        # torch (out, in/groups, k) -> mlx (out, k, in/groups)
```

### _layout_state.py:27

```
        # torch (out, in, kh, kw) -> mlx (out, kh, kw, in)
```

## campplus.py

### campplus.py:51

```
#: Buffer names that exist in a PyTorch BatchNorm but have no MLX counterpart.
```

### campplus.py:55

```
# containers that reproduce PyTorch parameter names
```

### campplus.py:65

```
        # Keep dict semantics: MLX may call len() on this object.
```

### campplus.py:158

```
# checkpoint plumbing
#: Layout rule names produced by :func:`flatten_state_dict`.
```

### campplus.py:174

```
        # torch (out, in/groups, k)  ->  MLX (out, k, in/groups)
```

### campplus.py:176

```
        # torch (out, in, kh, kw)  ->  MLX (out, kh, kw, in)
```

### campplus.py:334

```
# initialisation (matches the reference's post-build kaiming pass)
```

### campplus.py:357

```
# normalisation
```

### campplus.py:438

```
# statistics pooling
```

### campplus.py:464

```
# FCM front-end
```

### campplus.py:530

```
        # Reference quirk preserved: components.py:72-73 passes ``num_blocks[0]``
        # to BOTH ``_make_layer`` calls, so ``num_blocks[1]`` is never consulted.
        # Irrelevant for the default (2, 2), but kept faithful rather than
        # "corrected", so a checkpoint built with the reference cannot diverge.
```

### campplus.py:549

```
        # torch: (B, C, H, W) -> (B, C*H, W);  here: (B, H, W, C) -> (B, W, C*H)
```

### campplus.py:558

```
# TDNN building blocks (all channels-last NLC, matching MLX Conv1d)
```

### campplus.py:638

```
            # torch's avg_pool with ceil_mode=True excludes padded positions from
            # the mean (it pads with -inf internally).
```

### campplus.py:764

```
# configuration
```

### campplus.py:810

```
# model
```

### campplus.py:831

```
        # The reference keeps everything after ``head`` inside ``self.xvector``,
        # an ``nn.Sequential`` built by ``add_module``; keeping that wrapper
        # preserves the ``xvector.`` prefix in the checkpoint keys.
```

### campplus.py:870

```
        # Introspection only.  These two lists must NEVER be used for execution
        # order: the reference interleaves block_i / transit_i, so
        # ``xvector.names()`` is the single source of truth (see
        # ``CAMPPlus.frame_features``).
```

## ct_punc.py

### ct_punc.py:52

```
# FunASR maps the full-width marks to ASCII when the neighbouring token is ASCII.
```

### ct_punc.py:54

```
# ``forward_attention`` fills masked scores with this value before the softmax.
```

### ct_punc.py:56

```
# Depthwise SANM convolution weights are the only tensors needing a layout change.
```

### ct_punc.py:60

```
    # (C, 1, K) -> (C, K, 1); SANM depthwise memory kernels only
```

### ct_punc.py:77

```
# FunASR's ``LayerNorm`` (funasr/models/transformer/layer_norm.py) overrides the
# PyTorch default of 1e-5.  Getting this wrong silently changes every output.
```

### ct_punc.py:79

```
# FunASR's jieba word-level expansion only considers CJK Unified Ideographs
# (``"\u4e00" <= token[0] <= "\u9fa5"`` in the reference).
```

### ct_punc.py:82

```
# validation helpers
```

### ct_punc.py:84

```
# config
```

### ct_punc.py:162

```
        # The official config passes pos_enc_class (a class object the SANM
        # encoder only reads for other input_layers; input_layer="pe" never
        # touches it). It selects no weights here, so accept-and-ignore keeps
        # the config loadable while adding no silent behaviour.
```

### ct_punc.py:181

```
    #: Marks that terminate a sentence for the mini-sentence cache search.
```

### ct_punc.py:182

```
    #: Mark considered as the fallback cut point when a chunk grows past the limit.
```

### ct_punc.py:183

```
    #: ``CTTransformer.inference``'s ``split_size`` kwarg.
```

### ct_punc.py:184

```
    #: ``CTTransformer.inference``'s ``cache_pop_trigger_limit``.
```

### ct_punc.py:185

```
    #: Hard bound on tokens fed to the encoder in one forward (attention is O(T^2);
    #: the mini-sentence cache keeps T far below this).
```

### ct_punc.py:264

```
# vocabulary
```

### ct_punc.py:314

```
# tensor helpers
```

### ct_punc.py:351

```
# modules
```

### ct_punc.py:496

```
        # FunASR only adds the residual when in_size == size (encoders0).
```

### ct_punc.py:592

```
# strict checkpoint loading
```

### ct_punc.py:594

```
# text post-processing (pure Python, mirrors funasr/models/ct_transformer)
```

### ct_punc.py:728

```
                # FunASR scans [len - 2 .. 2]; index 0/1 and the final token are
                # never eligible to close a sentence.
```

### ct_punc.py:768

```
            # FunASR keeps `new_mini_sentence_punc` as the running list and returns
            # it verbatim as `punc_array`; the trailing sentence-end rewrite below
            # is applied to both, so `punc_array` is accumulated directly here.
```

## emotion2vec.py

### emotion2vec.py:48

```
#: Prefixes that exist in the published checkpoint but carry no inference
#: parameters.  They are dropped by :func:`convert_checkpoint_tensors` only when
#: this exact prefix list matches, and the dropped names are returned so the
#: caller can log them.  Any *other* unexpected key is an error.
```

### emotion2vec.py:52

```
# small helpers
```

### emotion2vec.py:90

```
# containers that preserve PyTorch Sequential / ModuleList key indices
```

### emotion2vec.py:161

```
# norms and activations
```

### emotion2vec.py:189

```
# ConvFeatureExtractionModel (fairseq v0.12.2 wav2vec2.py:819-895)
```

### emotion2vec.py:291

```
        # self["2"] is Sequential(TransposeLast, Fp32LayerNorm, TransposeLast):
        # NCT -> NTC puts the 512 channels on the norm's last axis -> NCT.
```

### emotion2vec.py:320

```
# relative positional encoder (audio.py:83-106)
```

### emotion2vec.py:333

```
            # reference: x[:, :, : -remove] -- the final slot on the last axis
```

### emotion2vec.py:392

```
# AltBlock / AltAttention (modules.py)
```

### emotion2vec.py:411

```
        # The entire score pipeline (scale + QK^T + bias + softmax + @V) is one
        # fused SDPA kernel; the ALiBi bias goes in as the additive ``mask``
        # argument and SDPA's internal softmax runs in float32. Equivalent to
        # the manual path (covered by the alignment suite) minus four kernel
        # launches per layer.
```

### emotion2vec.py:490

```
# config
```

### emotion2vec.py:494

```
    # model
```

### emotion2vec.py:503

```
    # feature encoder
```

### emotion2vec.py:508

```
    # relative positional encoder
```

### emotion2vec.py:513

```
    # context encoder
```

### emotion2vec.py:515

```
    # extra tokens / alibi
```

### emotion2vec.py:526

```
    # classifier
```

### emotion2vec.py:529

```
    # memory bounds (attention is O(T^2) in 16 * heads * frames)
```

### emotion2vec.py:628

```
# modality encoder
```

### emotion2vec.py:706

```
        # Bound the O(T^2) attention here too, not only via Emotion2VecModel.
```

### emotion2vec.py:708

```
        # Follow the module's own weight dtype, exactly as whisper's
        # ``AudioEncoder.__call__`` does.  The waveform is normalised in float32
        # (``preprocess_waveform``), so without this cast an fp16 checkpoint
        # runs the conv stack and all 12 blocks with float32 activations against
        # fp16 weights and pays a mixed-dtype promotion at every op (measured on
        # M4, 3.3 s window: 51.7 ms vs 37.2 ms, 1.39x).
```

### emotion2vec.py:741

```
# top level model
```

### emotion2vec.py:802

```
        # Attention is O(T^2) * num_heads, so the bound is enforced here rather
        # than only by callers who pre-flight with config.frames_for.
```

### emotion2vec.py:861

```
# checkpoint loading
```

### emotion2vec.py:870

```
#: Exact checkpoint keys whose layout differs between PyTorch and MLX.  These
#: are written out in full rather than derived from a positional guess, because
#: the sequence index sits at ``parts[-3]`` and a ``parts[-3] == "conv_layers"``
#: check silently stops matching.
#:
#: * ``conv_layers.<i>.0.weight`` -- ``nn.Conv1d`` weight ``(out, in//g, K)``
#: * ``relative_positional_encoder.<i>.0.weight`` -- same ``nn.Conv1d``
```

## fsmn_vad.py

### fsmn_vad.py:53

```
# Depthwise FSMN convolutions: ``(D, 1, K, 1)`` -> ``(D, K, 1)``.
```

### fsmn_vad.py:57

```
    # (D, 1, K, 1) -> (D, K, 1); only depthwise trailing-width-1 kernels exist here
```

### fsmn_vad.py:79

```
# FunASR's ``ComputeDecibel`` adds this before the log.
```

### fsmn_vad.py:82

```
# enums (values mirror funasr/models/fsmn_vad_streaming/model.py)
```

### fsmn_vad.py:108

```
# validation helpers
```

### fsmn_vad.py:110

```
# config
```

### fsmn_vad.py:261

```
# modules
```

### fsmn_vad.py:354

```
            # FunASR right-pads by rorder*rstride then drops rstride frames, which
            # is exactly "slice the first rstride frames, then right-pad by the
            # same amount".
```

### fsmn_vad.py:439

```
# decision logic (pure Python; mirrors funasr/models/fsmn_vad_streaming/model.py)
```

### fsmn_vad.py:564

```
    #: ``sum(scores[0][t][sil_pdf_id] for sil_pdf_id in sil_pdf_ids)`` per frame,
    #: kept host-side so the decision loop stays pure Python.  Trimmed in lockstep
    #: with ``decibel`` by :meth:`FsmnVadModel.drop_cached_frames`.
```

### fsmn_vad.py:567

```
    #: Sample counts replacing FunASR's ``data_buf`` / ``data_buf_all`` tensors.
```

### fsmn_vad.py:569

```
    #: ``True`` once ``data_buf_all_len`` holds real samples (FunASR tests
    #: ``data_buf_all is None``; a 0-length buffer is not the same as "unset").
```

### fsmn_vad.py:572

```
# model + driver
```

### fsmn_vad.py:753

```
        # FunASR: ``sum(scores[0][t][sil_pdf_id] for sil_pdf_id in sil_pdf_ids)``.
        # ``mx.take`` with an explicit axis is used instead of ``scores[0][:, ids]``
        # so the semantics do not depend on list-index broadcasting.
```

### fsmn_vad.py:771

```
                # FunASR's loop here spins forever; break out instead of hanging.
```

### fsmn_vad.py:850

```
            # FunASR only prints here and keeps the stale value; mirror that.
```

### fsmn_vad.py:871

```
            # FunASR only prints here and keeps the stale value; mirror that.
```

### fsmn_vad.py:1178

```
# strict checkpoint loading
```

## fun_asr_nano.py

### fun_asr_nano.py:107

```
        # ESPnet EncoderLayer default normalize_before=True: pre-norm.
```

## glm_asr.py

### glm_asr.py:28

```
    # intermediate_size (5120) == hidden (1280) * groups (4): the projector
    # merges 4 encoder frames into one row
```

### glm_asr.py:72

```
        # The table is built in float32 for accuracy but is NOT stored as the
        # activation dtype on purpose: it is cast per call in ``_rope``.  A
        # float32 table multiplied into a bf16 stream promotes the whole
        # attention path (and the whole tower) to float32 against bf16 weights;
        # see the note in :meth:`_rope`.
```

### glm_asr.py:84

```
        # Follow the activation dtype.  ``x * cos`` with a float32 ``cos``
        # promotes q/k (and therefore the scores, the softmax, v, o_proj and
        # every downstream layer) to float32 against bf16 weights -- a
        # mixed-dtype promotion at every Linear, exactly the disease
        # 57cec39/57bf9cb fixed for the decoder.  Casting ONLY the mel is not
        # enough: this line re-promotes the stream inside attention, which is
        # why the earlier mel-only A/B measured 1.00x and the fix was skipped.
        # With both cast, the tower is 1.44x (interleaved A/B, M4).
```

### glm_asr.py:133

```
        # positions are handled by the per-block partial RoPE ONLY; the
        # official GlmAsrEncoder has no additive sinusoid table
```

### glm_asr.py:138

```
        # the reference pads the mel to 3000 frames so L % 4 == 0; on real
        # length pad the conv output to a multiple of g, reshape, then keep
        # the first (t - g) // g + 1 rows (the padding rows are dropped)
```

## paraformer.py

### paraformer.py:62

```
# configuration
```

### paraformer.py:194

```
            # ``idim`` must follow the encoder width unless given explicitly.
```

### paraformer.py:229

```
# CTC head (shared shape with the SenseVoice port, kept local for independence)
```

### paraformer.py:255

```
# encoder
```

### paraformer.py:324

```
            # FunASR's ``nn.Embedding(..., padding_idx=...)`` only affects weight
            # *initialisation*; MLX's ``Embedding`` has no such argument, and the
            # forward pass is identical, so it is intentionally dropped here.
```

### paraformer.py:337

```
            # The streaming position encoder; parameters-free like the offline
            # one, so the checkpoint tree is unchanged. ``forward_chunk`` in
            # ``paraformer_streaming`` drives it through ``cache['start_idx']``.
```

### paraformer.py:350

```
        # When every sample fills the window the mask is uniformly True and all
        # its uses are identity; fold it to None so each layer takes the fused
        # SDPA path (numerically identical, covered by the alignment suite).
```

### paraformer.py:384

```
# CIF predictor
```

### paraformer.py:413

```
    # The reference accumulates in float64 and only then narrows:
    #   torch.cumsum(prefix_sum, dim=1, dtype=torch.float64).to(torch.float32)
    # ``mx.cumsum(..., dtype=...)`` casts the input to that type before summing,
    # so this is the same operation, not an approximation of it.  The narrowing
    # cast matters: a running sum that lands just below an integer in float64
    # can round up to exactly that integer in float32 and fire a token the
    # reference would not.
```

### paraformer.py:458

```
        # Very short / silent segment: the reference returns zero frames here
        # (the upstream fix that avoids an IndexError in the export variant).
```

### paraformer.py:466

```
    # ``time_of_rank[b, j]`` = time of the j-th fire of utterance b.
    # The sort key is **tie-free**: fires carry 0..batch_len-1 and non-fires carry
    # t..2t-1, so the resulting order never depends on sort stability.  Keying all
    # non-fires with a single sentinel value left the tail of the permutation
    # backend-dependent, which is a needless nondeterminism source.
```

### paraformer.py:471

```
    # Gather along the **time** axis natively rather than flattening the batch.
    # The reference's ``prefix_sum_hidden[fire_idxs]`` over a (B, T, D) tensor is a
    # batched gather on axis 1; the flattened form instead reshapes (B, T, D) to
    # (B*T, D), adds a per-row ``b * T`` offset and takes on axis 0, then reshapes
    # back.  That round-trip through a flat view is the part worth falsifying on
    # GPU, and it is also why a missing row offset was silent at B=1.
```

### paraformer.py:475

```
            # The fires signal carries no feature axis, so index it directly.
```

### paraformer.py:481

```
    # One boundary per fire, then subtract the previous one.
```

### paraformer.py:482

```
    #   boundary[j] = P[t_j] - rem[t_j] * h[t_j]
    #   frames[j]   = boundary[j] - boundary[j-1]          (boundary[-1] = 0)
```

### paraformer.py:483

```
    # Expanding the second line gives exactly the reference expression
    #   P[t_j] - P[t_{j-1}] + rem[t_{j-1}] * h[t_{j-1}] - rem[t_j] * h[t_j]
    # with ``rem[t] = fires[t] - floor(fires[t])`` and every previous term zero at
    # j = 0.  That is the same algebra the reference reaches by rolling the
    # gathered frames by one and zeroing the first entry of each batch
    # (``cif_predictor.py:881-895``); this form needs a single gather per operand
    # instead of two, and builds no separate "previous" tensors and no
    # first-position ``where`` masks.
```

### paraformer.py:563

```
        # MLX conv1d is channels-last (NLC), so ``hidden`` (B, T, D) feeds the
        # Conv1d directly and both reference transposes disappear.  Padding is
        # applied on the time axis, i.e. axis=1.
```

### paraformer.py:581

```
            # Reference ``cif_predictor.py:293`` is a plain ``alphas.sum(-1)`` with
            # **no floor**; ``hidden`` is likewise left unextended because the
            # reference has no ``else`` branch at all when tail_threshold <= 0.
            # Flooring here would change ``round(token_num)`` downstream, which
            # the reference's ``Paraformer.forward`` applies to the raw sum.
```

### paraformer.py:591

```
# decoder
```

### paraformer.py:609

```
            # The reference intentionally drops the positional encoder here.
```

### paraformer.py:716

```
# model
```

### paraformer.py:867

```
        # Degenerate case (alphas all zero): no integration boundary at all.
```

## paraformer_streaming.py

### paraformer_streaming.py:94

```
    # MLX conv1d is NLC: hidden (B, T, D) feeds directly; the reference's two
    # transposes around the pad/conv collapse into padding on axis=1.
```

### paraformer_streaming.py:220

```
# model
```

### paraformer_streaming.py:280

```
            # ``_add_overlap_chunk`` (scama/encoder.py:480-494): stitch the
            # retained look-back window with the new chunk, keep a tail window.
```

## qwen3_asr.py

### qwen3_asr.py:58

```
        # x and output are NHWC (B, F, T, C) -- MLX Conv2d's native layout
```

### qwen3_asr.py:110

```
        # reference: inv_timescales[i] = exp(-log(1e4)/(C/2-1) * i) -- the
        # exponent multiplies arange BEFORE exp; my old form computed
        # exp(-log(1e4)*i)/(C/2-1) which is a different frequency ladder.
```

### qwen3_asr.py:131

```
        # conv_out: Linear over (h * F_after_cnn)
```

### qwen3_asr.py:143

```
        # The reference time-splits into 2*n_window-mel chunks, pads every
        # chunk to the full window (pad_sequence), convolves per chunk, adds
        # positional embeddings computed for the PADDED length, then keeps
        # only each chunk's real frames (padded_mask_after_cnn). Verified
        # per-block against the reference hooks: padding the tail chunk
        # before the conv matters (4.2 abs otherwise), and the pos table is
        # applied on the padded 13 frames before slicing.
```

### qwen3_asr.py:166

```
        # Reference windowing: after concatenating the chunk frames, the
        # layers attend within windows of window_aftercnn = padded_chunk_frames
        # * (n_window_infer / (n_window*2)) AFTER-CNN frames (modeling file's
        # cu_chunk_lens over aftercnn_lens). For the default n_window_infer=
        # 800 and n_window=50 that is 104 frames; utterances shorter than one
        # window attend globally (mask None).
```

## sanm.py

### sanm.py:55

```
# masking / padding helpers
```

### sanm.py:93

```
# checkpoint plumbing
```

### sanm.py:106

```
#: Public mapping ``"<key suffix regex>" -> layout rule name`` for consumers that
#: need to build a conversion table themselves.
```

### sanm.py:184

```
            # Only the kernel needs a transpose.  The rule follows the OWNING
            # module, so a conv's 1-D ``bias`` would inherit it and then be
            # transposed with three axes; the bias is IDENTITY in both layouts.
```

### sanm.py:186

```
            # A plain dict of tensors stored directly on a Module.
```

### sanm.py:209

```
    # ``mx.array`` values go into the mapping storage via MLX's ``__setattr__``.
```

### sanm.py:242

```
            # The rule comes from the owning module type, so a renamed
            # parameter still gets the right layout conversion.
```

### sanm.py:256

```
# containers
```

### sanm.py:269

```
        # Keep dict semantics: MLX may call len() on this object.
```

### sanm.py:299

```
# normalisation
```

### sanm.py:319

```
# positional encodings
```

### sanm.py:339

```
        # ``reverse`` flips the position index here; the reference instead
        # reverses an already-built table of length ``max_len``.  Identical for
        # ``reverse=False``, which is the only configured path.
```

### sanm.py:343

```
        # reference: torch.arange(0, d_model, 2) * -(log(10000) / d_model)
```

### sanm.py:347

```
        # stack(...).reshape(...) yields [sin0, cos0, sin1, cos1, ...], which is
        # exactly pe[:, 0::2] = sin / pe[:, 1::2] = cos.
```

### sanm.py:382

```
# feed forward
```

### sanm.py:413

```
# attention
```

### sanm.py:467

```
            # (B, 1, T) -> (B, 1, 1, T) boolean "is padding" mask.
```

### sanm.py:484

```
        # The unmasked branch uses the fused SDPA kernel (one kernel for
        # scale+BMM+softmax+BMM, softmax in float32 internally). Numerically
        # equivalent to the manual path -- asserted by tests -- but without
        # four separate kernel launches per layer, which dominates latency on
        # short sequences.
```

### sanm.py:609

```
# encoder / decoder layers
```

### sanm.py:708

```
# subsampling front-ends
```

### sanm.py:752

```
        # (B, T, F, C) -> (B, T, C, F) -> flatten in PyTorch's (C, F) order.
```

## sensevoice.py

### sensevoice.py:46

```
# metadata dictionaries (verbatim from sense_voice/model.py)
```

### sensevoice.py:73

```
#: ``self.embed = nn.Embedding(7 + len(lid_dict) + len(textnorm_dict), input_size)``
#: i.e. rows ``0..6`` are reserved for the rich-transcription/event slots.
```

### sensevoice.py:75

```
#: Index pairs used verbatim by ``SenseVoiceSmall.encode`` / ``.inference``.
```

### sensevoice.py:80

```
#: language + event + emotion + textnorm
```

### sensevoice.py:83

```
# configuration
```

### sensevoice.py:182

```
# modules
```

### sensevoice.py:204

```
        # ``mx.logsoftmax`` does not exist in MLX; the max-shifted functional
        # ``mlx.nn.log_softmax`` is the API.  Qualified as ``mxnn.`` because this
        # class also defines a *method* named ``log_softmax``.
```

### sensevoice.py:236

```
        # The reference hard-codes SinusoidalPositionEncoder regardless of
        # ``input_layer`` (SenseVoice has no Conv2dSubsampling front-end).
```

### sensevoice.py:273

```
        # The front-end always produces float32 fbank.  Without this cast an
        # fp16 checkpoint runs the whole 70-layer SANM stack with float32
        # activations against fp16 weights and pays a mixed-dtype promotion at
        # every Linear (measured on M4, 3.275 s window: 37.5 ms vs 19.2 ms,
        # 1.95x).  For a float32 checkpoint the cast is a no-op, so those
        # numerics stay bit-identical.
```

### sensevoice.py:277

```
        # When every sample fills the whole window the mask is uniformly 1.0:
        # every ``where(blocked, ...)`` is a no-op and the mask products in the
        # FSMN branch are identity. Fold it to None so each layer can take the
        # fused SDPA path (numerically identical; covered by the alignment
        # suite). ``olens`` is then just ``ilens``.
```

### sensevoice.py:290

```
        # ``olens`` is derived here, before the tail blocks (faithful to the
        # reference; the mask is unchanged by the tail blocks anyway).
```

### sensevoice.py:352

```
        # One row of ``NUM_PREFIX_FRAMES`` token ids, tiled across the batch.
        # Note the broadcast target is ``(B, NUM_PREFIX_FRAMES)``: ``ids`` is
        # token *indices*, so its width is the number of prefix frames and has
        # nothing to do with ``input_size``.
```

## whisper.py

### whisper.py:56

```
#: OpenAI Whisper defaults (see ``whisper/__init__.py`` of the reference).
```

### whisper.py:67

```
# containers / checkpoint plumbing (self-contained by design)
```

### whisper.py:77

```
        # Keep dict semantics: MLX may call len() on this object.
```

### whisper.py:108

```
        # torch (out, in, k) -> MLX (out, k, in)
```

### whisper.py:119

```
#: Buffers canonical registers with ``persistent=False``, which are therefore
#: **absent from every published checkpoint**: its derived causal ``mask``
#: (OpenAI release v20250625 ``model.py:225``) and ``alignment_heads``
#: (``:276``, ``:285``).
#:
#: The actual table is :data:`NON_PERSISTENT_BUFFERS`, defined **after** the
#: class definitions it names and reached through
#: :func:`_is_non_persistent_buffer`, which resolves the global at call time.
#: It pairs each buffer with the module type that owns it, rather than matching
#: a leaf name, so an unrelated parameter called ``_mask`` on some other module
#: is still collected.
#:
#: Either way this is an explicit list of exactly those buffers.  A blanket
#: "skip every private array" rule would swallow real parameters -- both
#: ``positional_embedding`` buffers are *persistent* (``:181``) and must stay.
```

### whisper.py:132

```
            # Owner + native key, so the full model, a standalone submodule and
            # any caller prefix are all covered by one entry.
```

### whisper.py:255

```
# building blocks
```

### whisper.py:287

```
#: ``torch.nn.GELU()``'s default is the *exact* erf formulation, and MLX's
#: ``nn.GELU`` defaults to ``approx="none"`` (exact), so the two agree.  If MLX's
#: default ever changed, this constant would silently switch to the tanh
#: approximation -- hence the explicit note rather than an inline call.
```

### whisper.py:333

```
            # The reference slices the same ``n_ctx`` from both axes; that only
            # broadcasts for a full-length pass (n_ctx == key_len) or for the
            # single-token incremental steps this decoder performs.
```

### whisper.py:340

```
        # The reference computes the scores in fp32, takes the softmax there and
        # casts the weights back to the query dtype.
```

### whisper.py:360

```
        # ``xa is None`` forces the recompute branch for self-attention, which is
        # what makes the cache append and cover the current step.
```

### whisper.py:378

```
        # Fused path: with no mask (cross-attention or any unmasked pass) the
        # whole scale+QK^T+softmax+@V chain is one SDPA kernel -- at T=1500 the
        # manual path materializes a (B,H,T,T) score matrix whose traffic
        # dominates the encoder. Whisper's dual ``d^-0.25`` scaling is exactly
        # SDPA's ``scale=d_k**-0.5``, and SDPA's internal softmax runs in
        # float32 like the reference. The causal-masked self-attention path
        # keeps the manual implementation (mask semantics + qk return value
        # are covered by tests against the reference).
```

### whisper.py:442

```
# dimensions / configuration
```

### whisper.py:526

```
# encoder / decoder / model
```

### whisper.py:557

```
        # The canonical encoder *asserts* an exact length rather than adapting to
        # whatever it produced: ``positional_embedding`` has ``n_ctx`` rows.  The
        # previous version sliced the table down to whatever length it got, which
        # silently returned a short encoder output and is precisely what hid the
        # ``conv1`` stride bug from the test suite.
```

### whisper.py:566

```
        # Evaluate block by block instead of letting the whole 32-block graph
        # accumulate.  MLX frees a block's intermediates only once its output is
        # materialised; left lazy, every block's activation stays live until the
        # single `mx.eval` at the call site.  On the 30 s window (1500 frames)
        # that transient is the largest single allocation in the encoder path.
        # Measured on M4 (whisper-large-v3, 30 s, peak `Physical footprint` from
        # vmmap, 6 alternating rounds x 3 calls): 3600 MB -> 3000 MB, i.e.
        # -600 MB (-16.7%), reproducible to +-100 MB.  Speed is unchanged:
        # paired in-process A/B over 30 pairs gives -0.04%, CI95 [-0.36, +0.14]%
        # (median 973.6 vs 976.6 ms).  The arithmetic is untouched -- the same
        # `block(x)` calls in the same order, only the evaluation boundary moves
        # -- so results are bit-identical.
```

### whisper.py:594

```
        # torch: empty(n_ctx, n_ctx).fill_(-inf).triu_(1)
```

### whisper.py:632

```
        # Follow the **decoder's own weight dtype**, not ``xa``/the mel dtype.
        # The front-end mel is float32, so casting to ``xa.dtype`` used to lock
        # the entire decoder into float32 activations against fp16 weights: a
        # mixed-dtype promotion at every Linear, every step (measured on M4:
        # 9.12 ms/step vs 2.02 ms/step for a 24-step decode).  A float32
        # checkpoint is unaffected -- the cast is a no-op there.
```

### whisper.py:637

```
        # Tied projection (reference: x @ token_embedding.weight.T), with the weight
        # cast to the activation dtype first.  The logits are always float32.
```

### whisper.py:703

```
# front-end helpers (raw mel only -- no tokenizer, no downloads)
```

### whisper.py:798

```
    # Slaney normalisation
```

### whisper.py:834

```
    # torch slices ``[..., :-1]`` along the *frame* axis.
```

### whisper.py:838

```
    # MLX's matmul does not accept a 2-D right-hand side against a 3-D left-hand
    # side (it treats the pair as batched and then requires the feature axis to
    # match), so the transposed filterbank gets an explicit leading batch axis.
```

### whisper.py:839

```
    # (B, frames-1, freq) @ (1, freq, n_mels) -> (B, frames-1, n_mels)
```

### whisper.py:845

```
# raw decoding helpers (token ids only -- no text, no tokenizer)
```

### whisper.py:937

```
    # Only the SELF-attention cache grows, one step at a time.  Cross-attention is
    # keyed and stored once at ``n_audio_ctx`` and never appended, so including
    # it would compare an audio length against a text bound -- and since
    # ``n_audio_ctx`` is larger than ``n_text_ctx`` for every published Whisper
    # (1500 vs 448), that trips the check on the very first decode.
```

## awq.py

### awq.py:14

```
    # never differentiated, so the context is a no-op guard upstream added for
    # its training runtime.  Define a local fallback rather than import it.
```

### awq.py:27

```
    #: Subsampled rows for the clip search (upstream ``n_sample_token``).
```

### awq.py:28

```
    #: Batch width for the row dimension during the clip search.
```

### awq.py:29

```
    #: Batch width over output channels during the clip search.
```

### awq.py:30

```
    #: Scale grid over ``x_max ** ratio``; upstream range.
```

### awq.py:42

```
#: Qwen3 decoder block: identical topology to mlx-lm's llama config.
```

### awq.py:50

```
#: q/k projections are excluded from clipping (attention couples their error);
#: o_proj/v_proj remain clip-eligible per upstream's no_clip=["q_proj","k_proj"].
```

### awq.py:98

```
                    # ONLY the block's first norm records (its input IS the
                    # block input); the mid-block norm's input is not used.
```

### awq.py:108

```
                        # HOST-side storage with a hard stop: once ``row_cap``
                        # rows are held, further rows are DROPPED (the cap is
                        # a statistical sample, not an exhaustive log).  This
                        # avoids both the wired-GW trap of MLX-side buffers and
                        # the O(n^2) re-concat allocation spikes.
```

### awq.py:124

```
                        # else: at cap, drop
```

### awq.py:177

```
                    # HOST-side numpy storage (see MultiBlockCapture for why:
                    # MLX-side buffers are wired GPU memory).
```

### awq.py:209

```
                # a whole transformer block: unroll the qwen3 residual pattern
```

### awq.py:227

```
        # mlx-lm's ``module2inspect``: the sub-block when the site names one
        # (attention), else the FIRST layer itself -- never the whole block,
        # because the whole block's input is the residual stream, not the
        # successor Linear's input.
```

### awq.py:250

```
                # restore
```

### awq.py:314

```
        # keep the captured input consistent with the folded weights (host
        # numpy divided by the mx scales, back through numpy for the next site)
```

### awq.py:326

```
        # reference loss: plain RTN quantization of the untouched block
```

### awq.py:327

```
        # reference output for the never-worse check uses the block's OWN
        # input: what enters its first norm (the residual stream).
```

### awq.py:361

```
            # NEVER-WORSE: if the AWQ-adjusted block quantizes WORSE than the
            # untouched one, the adjustment is discarded and the block keeps
            # its original weights. (The non-fallback path is the difference
            # from the original port: it used to restore here unconditionally,
            # which silently discarded every scale/clip the search had folded
            # in -- the saved "AWQ" checkpoint was the raw weights in f16 and
            # every downstream "AWQ vs RTN" comparison was two RTN runs.)
```

### awq.py:364

```
        # The adjustment helps: re-fold it. ``weights`` was captured before the
        # search started, so the adjusted parameters must be re-applied --
        # cheapest correct path is to re-run the same deterministic search on
        # the pristine block (the captures are unchanged, so the search
        # reproduces the same scales/clips) rather than stash a second tree.
```

## calibrate.py

### calibrate.py:70

```
    #: Rows to accumulate per observation before folding into the Hessian. The
    #: hidden states of one utterance are a few hundred rows, so this is a
    #: memory guard rather than a performance knob.
```

### calibrate.py:72

```
    #: Per-language row counts, mirrored from the accumulator for reporting.
```

### calibrate.py:86

```
            # nn.Embedding.as_linear is looked up on the *instance* dict first
            # (verified against mlx 0.32), so an instance attribute is enough and
            # is restored by deleting it.
```

### calibrate.py:94

```
            # An nn.Linear head is called through nn.Linear.__call__, which is a
            # *class* attribute: patch the class and restore it in uninstall.
```

### calibrate.py:145

```
        # Zero rows (a batch that happened to be empty) carry no information.
```

## gptq.py

### gptq.py:25

```
#: Epsilon from MLX's ``affine_quantize`` kernel: the smallest scale it will use.
```

### gptq.py:27

```
#: Storage dtypes a quantized checkpoint can carry its ``scales``/``biases`` in.
```

### gptq.py:30

```
# the quantizer
```

### gptq.py:49

```
        #: Number of codes packed into one uint32 word (8 for 4 bits, 4 for 8).
```

### gptq.py:112

```
        # A negative scale is legal in this scheme (bias is the max end), so the
        # low clamp is applied to the code, not to x.
```

### gptq.py:171

```
# configuration
```

### gptq.py:177

```
    #: Hessian dampening, as a fraction of ``mean(diag(H))``.
```

### gptq.py:178

```
    #: Columns per lazy batch update (Frantar's act-order note).
```

### gptq.py:179

```
    #: Quantize in descending-``diag(H)`` order (recommended for embeddings).
    #: This permutes the columns, so it is paired with **static groups**: the
    #: group parameters are taken once, from the original matrix, over each
    #: group's own columns in storage order.  (Frantar's act-order note makes the
    #: same pairing; with dynamic groups the stored codes and the stored side
    #: tensors would describe different groups.)
```

### gptq.py:181

```
    #: Dtype the side tensors are stored in (must match the checkpoint dtype).
```

### gptq.py:182

```
    #: Rows processed per pass.  GPTQ is row-independent -- every row of ``W`` is
    #: quantized against the same ``H`` with no coupling between rows -- so this
    #: is exact, not an approximation.  It exists because a tied LM head is large
    #: (151936x1024 is 622 MB as a single float32 array, and the inner loop wants
    #: several such arrays live at once) while the memory guard's MLX limit is far
    #: below that.  0 means "all rows in one pass".
```

### gptq.py:219

```
    #: ``uint32`` packed codes, ``[out, in * bits / 32]``.
```

### gptq.py:220

```
    #: Per-group scale, ``[out, in / group_size]``, in the storage dtype.
```

### gptq.py:221

```
    #: Per-group bias, same shape and dtype as ``scales``.
```

### gptq.py:222

```
    #: Mean squared Frobenius error of the reconstructed weight.
```

### gptq.py:223

```
    #: Total squared layer-output error ``trace(D (H/2) D^T)``.
```

### gptq.py:224

```
    #: Mean squared error of plain round-to-nearest on the same matrix.
```

### gptq.py:225

```
    #: Output error of plain round-to-nearest on the same matrix.
```

### gptq.py:226

```
    #: Fraction of *output* error removed relative to round-to-nearest.
```

### gptq.py:227

```
    #: Fraction of Frobenius error removed relative to round-to-nearest (may be
    #: negative: see the class docstring).
```

### gptq.py:228

```
    #: Column permutation applied by act-order (``None`` when disabled).
```

### gptq.py:246

```
# GPTQ core
```

### gptq.py:266

```
    # Symmetrize: the inverse of a symmetric matrix is symmetric only up to
    # rounding, and Cholesky reads the lower triangle.
```

### gptq.py:298

```
    # Keep the caller's basis: act-order permutes ``weight``/``hessian`` below,
    # and the reported error must be measured in the original coordinates.
```

### gptq.py:315

```
    # A column the calibration never exercised has a zero Hessian diagonal; it
    # carries no second-order information, so it must not be allowed to blow up
    # the inverse.  Zero the column and give the Hessian a unit diagonal there,
    # exactly as the reference does.
```

### gptq.py:336

```
    # Group parameters, keyed by the group index in the *stored* layout.
```

### gptq.py:337

```
    # Without act-order the columns are visited in storage order, so a group's
    # parameters are simply taken when the loop reaches its first column, from
    # the running (already error-compensated) matrix -- Frantar's "dynamic
    # groups".  With act-order the columns arrive in permuted order, so the
    # columns of one stored group are no longer contiguous, and taking the
    # parameters from the permuted matrix would leave the stored codes and the
    # stored side tensors describing different groups (the checkpoint would
    # dequantize to something the metrics never described).  act-order is
    # therefore paired with **static groups**, as in the reference
    # implementation: the parameters are chosen once, from the original matrix,
    # over the group's own columns in storage order.
```

### gptq.py:344

```
    # GPTQ is row-independent: each row is quantized against the same ``H`` with
    # no coupling between rows, so processing row chunks is exact.  It is what
    # keeps a 151936x1024 tied head inside the guard's MLX budget -- one full
    # float32 copy of that matrix is already 622 MB, and the inner loop wants
    # several live at once.
```

### gptq.py:350

```
        # Group parameters are keyed by the group index in the *STORED* layout,
        # because that is what the checkpoint carries.
```

### gptq.py:351

```
        # Without act-order the columns are visited in storage order, so a
        # group's parameters are simply taken when the loop reaches its first
        # column, read off the running (already error-compensated) matrix --
        # Frantar's "dynamic groups".  With act-order the columns arrive in
        # permuted order, so a stored group's columns are no longer contiguous
        # and taking the parameters from the permuted matrix would leave the
        # stored codes and the stored side tensors describing different groups
        # (the checkpoint would dequantize to something the metrics never
        # described).  act-order is therefore paired with **static groups**, as
        # in the reference implementation: the parameters are chosen once, from
        # the original matrix, over the group's own columns in storage order.
```

### gptq.py:353

```
        # Either way the parameters are per-chunk: they are read off this chunk's
        # rows, so they are recomputed here rather than shared across chunks.
```

### gptq.py:380

```
            # ``block`` is this block's slice of the running (error-compensated)
            # matrix.  The compensation must be applied to it as the loop
            # proceeds, so it is kept separate from the values chosen.
```

### gptq.py:399

```
                # Lazy batch update (Frantar's act-order note): every column in
                # the block contributes ``error_i * Hinv[i, stop:]`` to the
                # remaining columns, so the whole block's compensation is applied
                # at once.  ``error_i`` is the *scaled* error
                # ``(w - q) / Hinv[i, i]`` -- the reference's ``Err1`` -- and NOT
                # ``block - block_quantized``, which differs by a factor of
                # ``Hinv[i, i]`` per column because the running block has already
                # absorbed the earlier columns' updates.
```

### gptq.py:416

```
    # The metrics are computed from the STORED triple, not accumulated from the
    # inner loop's own arrays.  Those are not the same object -- the loop's
    # per-chunk reconstruction is built from the running (error-compensated)
    # matrix, while the checkpoint stores codes and side tensors that are read
    # back through a different path -- and accumulating chunk-local traces of it
    # did not reproduce the stored triple's metric (measured 1112899.59 against
    # the true 392.65 on the module's own test problem).  Rebuilding the
    # reconstruction from the stored triple, in row blocks so no full float32
    # copy is materialised, makes the reported numbers describe exactly what a
    # reader of the checkpoint would get.
```

### gptq.py:418

```
    # ``trace(D (H/2) D^T)`` does decompose over the rows of ``D`` (row ``i``
    # contributes ``D[i] (H/2) D[i]^T``), so blocking it is exact, not an
    # approximation.
```

### gptq.py:521

```
# Hessian collection
```

### gptq.py:529

```
    #: Per-language ``X^T X`` (float32) and row count.
```

### gptq.py:539

```
            # [batch, sequence, hidden] -> rows
```

## kv_cache.py

### kv_cache.py:46

```
                # The backing store may be LARGER than offset (capacity is
                # step-quantized); copy the whole store, not just [:offset] --
                # a prefill of 253 into a 256 buffer followed by a 5-token
                # speculative block would otherwise try to squeeze 256 columns
                # into a 253-column slice.
```

## quant.py

### quant.py:26

```
#: How the stored quantization values were computed. ``rtn`` is ``mx.quantize``'s
#: own round-to-nearest; ``gptq`` is Hessian-based error compensation. Both
#: produce the identical runtime layout, so this is provenance, not behaviour.
```

### quant.py:28

```
#: Substrings that mark a rank-2 ``.weight`` tensor which must NOT be quantized
#: through the ``nn.Linear`` path, even when it sits under a quantized prefix:
#: the untied ``lm_head``, a learned position table, or the sinusoidal table the
#: qwen3-asr tower derives. ``embed_tokens`` is handled separately (it is an
#: ``nn.Embedding``, not an ``nn.Linear``, and can only be selected by an
#: explicit ``embed_tokens`` target prefix); the marker is kept here so a Linear
#: named ``...embed_tokens`` is still never quantized by the Linear rule.
```

### quant.py:46

```
    #: Optional per-target bit widths, e.g. ``{"thinker.model.embed_tokens": 8}``.
    #: The LONGEST matching target prefix wins, so a specific module can differ
    #: from the general rule.  This exists because the tied embedding and the
    #: decoder Linears do not have the same sensitivity: measured on
    #: qwen3-asr-0.6b, a w4 head costs ~10% mean span token edit rate while the
    #: decoder Linears at w4 are fine, so spending 8 bits on the head and 4 on
    #: the stack is a better use of the budget than 4 bits on both.
```

### quant.py:48

```
    #: Optional per-target group sizes, e.g. ``{"thinker.model.embed_tokens": 32}``.
    #: Same longest-prefix rule as ``target_bits``.  This is mlx-lm's AWQ
    #: default for the tied embedding (``embed_group_size=32`` beside a g64
    #: stack): the embedding's side tensors are what its w4 accuracy needs,
    #: not the stack's.
```

### quant.py:109

```
        # Sorted longest-first so the tie-break is deterministic regardless of
        # dict order; bits_for() also compares lengths, this is belt-and-braces.
```

### quant.py:173

```
        # An EXPLICIT target (the module path itself, not a prefix it sits
        # under) overrides the non-Linear markers: glm's untied ``lm_head`` IS
        # a real nn.Linear, and the marker exists for the tied families where
        # the head is an Embedding handled by the embedding rule. Naming a
        # module directly is the operator saying "quantize THIS", which must
        # not be silently vetoed by a table written for another topology.
```

### quant.py:187

```
        # The only divisibility MLX requires is that the GROUP SIZE divides the
        # input dimension.  The pack factor (codes per uint32 word) is MLX's own
        # business and is not a divisibility requirement on the columns: verified
        # against mx.quantize for bits in {2,3,4,5,6,8} x cols in {64..256}, every
        # width packs any column count that the group size divides.  An earlier
        # version of this predicate ALSO required ``columns % (32 // bits) == 0``,
        # which is wrong in both directions -- it rejected w3/w5/w6 at 64 columns
        # (which MLX accepts) and accepted w2 at 96 columns (which it rejects, for
        # the group-size reason).  Over-strict here silently skips tensors that
        # should be quantized.
```

### quant.py:196

```
        # An EXPLICITLY named target overrides the non-Linear markers (glm's
        # untied lm_head is a real Linear; the marker exists for tied
        # topologies where the head is the Embedding -- see is_linear_weight).
```

### quant.py:219

```
    #: Modules quantized at a width other than ``bits`` (see
    #: :meth:`QuantizationSpec.bits_for`), recorded so a report shows the plan
    #: that was actually applied rather than only the default.
```

### quant.py:221

```
    #: Modules quantized at a group size other than ``group_size`` (see
    #: :meth:`QuantizationSpec.group_for`).
```

### quant.py:271

```
        # Only the two weight-bearing module kinds the scheme supports are
        # considered; everything else (norms, convs, position tables) is left
        # at the checkpoint dtype and is not reported as a skip.
```

### quant.py:283

```
                    # Returning a dict is nn.quantize's per-module escape hatch.
                    # The override must be honoured here, at load time, or the
                    # module is built at the wrong width/group and the strict
                    # loader rejects the checkpoint's shapes -- so this is not
                    # cosmetic.
```

### quant.py:296

```
    # Every parameter of a quantized module is frozen by nn.quantize; inference
    # never needs gradients, and an unfrozen parameter would be re-quantized by
    # a later nn.quantize call.
```
