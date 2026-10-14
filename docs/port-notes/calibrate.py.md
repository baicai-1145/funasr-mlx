# calibrate.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
Collect GPTQ calibration activations for the LM head through the REAL model.

The Hessian for a tied LM head must come from the activations the head actually
sees: the post-final-norm hidden states of the real decode forward, over real
audio in all ten languages.  Random or synthetic activations give a Hessian that
describes a distribution the model never visits, and GPTQ would then compensate
error in the wrong directions.

This module is the model-side half of the GPTQ pipeline:

* :class:`HeadActivationCollector` patches the head call (``nn.Embedding.
  as_linear`` for a tied head, ``nn.Linear.__call__`` for an untied one) on the
  live model, runs the real prefill forward per utterance, and feeds the hidden
  states to a :class:`~funasr_mlx.models.llm.gptq.HessianAccumulator` tagged by
  language.  Patching is restored in a ``finally`` block, so a failure cannot
  leave a hooked model behind.
* Only the **prefill** pass is collected.  Decode steps consume one token at a
  time with the same distribution of hidden states as prefill (same final norm,
  same layer stack), and prefill is where the head sees long sequences; adding
  decode steps would multiply the cost for no extra information.

The collector never modifies weights, never runs gradients, and evaluates every
array it touches (``mx.eval`` outside any timed region), so nothing is left
lazily pending when the Hessian is read out.

```
