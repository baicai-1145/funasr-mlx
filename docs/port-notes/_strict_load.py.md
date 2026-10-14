# _strict_load.py — port notes

## module

```
Shared strict-loading kernel for the flat-checkpoint model families.

Extracted from the near-identical copies in ct_punc.py, emotion2vec.py and
fsmn_vad.py (duplication-audit cluster B). Differences that existed between
the copies are parameters, not branches:

* ``convert_checkpoint_tensors(state, rules)``: each family passes its own
  key-suffix -> transform rules (fsmn_vad: 4-D depthwise (D,1,K,1)->(D,K,1);
  ct_punc: 3-D depthwise (C,1,K)->(C,K,1); emotion2vec: none).
* ``load_state_dict_strict(model, state, ignore_prefixes=...)``: the
  fsmn_vad allow-list behavior is kept as the general signature.

All transposes were validated against exact-source golden tests by the lead;
behavior is byte-identical to the previous per-family copies.

```

## module

```
Shared strict-loading kernel for the flat-checkpoint model families.

Extracted from the near-identical copies in ct_punc.py, emotion2vec.py and
fsmn_vad.py (duplication-audit cluster B). Differences that existed between
the copies are parameters, not branches:

* ``convert_checkpoint_tensors(state, rules)``: each family passes its own
  key-suffix -> transform rules (fsmn_vad: 4-D depthwise (D,1,K,1)->(D,K,1);
  ct_punc: 3-D depthwise (C,1,K)->(C,K,1); emotion2vec: none).
* ``load_state_dict_strict(model, state, ignore_prefixes=...)``: the
  fsmn_vad allow-list behavior is kept as the general signature.

All transposes were validated against exact-source golden tests by the lead;
behavior is byte-identical to the previous per-family copies.

```

## module

```
Shared strict-loading kernel for the flat-checkpoint model families.

Extracted from the near-identical copies in ct_punc.py, emotion2vec.py and
fsmn_vad.py (duplication-audit cluster B). Differences that existed between
the copies are parameters, not branches:

* ``convert_checkpoint_tensors(state, rules)``: each family passes its own
  key-suffix -> transform rules (fsmn_vad: 4-D depthwise (D,1,K,1)->(D,K,1);
  ct_punc: 3-D depthwise (C,1,K)->(C,K,1); emotion2vec: none).
* ``load_state_dict_strict(model, state, ignore_prefixes=...)``: the
  fsmn_vad allow-list behavior is kept as the general signature.

All transposes were validated against exact-source golden tests by the lead;
behavior is byte-identical to the previous per-family copies.

```
