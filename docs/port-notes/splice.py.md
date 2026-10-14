# splice.py — port notes

## module

```
Shared audio-splice prefill for the LLM-ASR thinkers (cluster D).

The glm/qwen3/moss thinkers each reimplemented the same forward:
find the audio-token run, splice tower embeds over it (cast to the
decoder's dtype first -- see the dtype note in each family), run one
decoder pass with an explicit causal mask for prefill, and pick the
output head by tie_word_embeddings. The only real differences are where
each family keeps its attributes:

* embedder/decoder call: ``self.thinker(...)`` vs ``self.language_model(...)``
* audio token id: ``cfg.audio_token_id`` vs ``self.audio_token_id``
* head selection: tie flag location differs; glm probes lm_head presence.

``splice_and_forward`` takes those as explicit parameters; each thinker
keeps its public ``forward_with_audio`` as a one-line adapter.

```

## module

```
Shared audio-splice prefill for the LLM-ASR thinkers (cluster D).

The glm/qwen3/moss thinkers each reimplemented the same forward:
find the audio-token run, splice tower embeds over it (cast to the
decoder's dtype first -- see the dtype note in each family), run one
decoder pass with an explicit causal mask for prefill, and pick the
output head by tie_word_embeddings. The only real differences are where
each family keeps its attributes:

* embedder/decoder call: ``self.thinker(...)`` vs ``self.language_model(...)``
* audio token id: ``cfg.audio_token_id`` vs ``self.audio_token_id``
* head selection: tie flag location differs; glm probes lm_head presence.

``splice_and_forward`` takes those as explicit parameters; each thinker
keeps its public ``forward_with_audio`` as a one-line adapter.

```

## module

```
Shared audio-splice prefill for the LLM-ASR thinkers (cluster D).

The glm/qwen3/moss thinkers each reimplemented the same forward:
find the audio-token run, splice tower embeds over it (cast to the
decoder's dtype first -- see the dtype note in each family), run one
decoder pass with an explicit causal mask for prefill, and pick the
output head by tie_word_embeddings. The only real differences are where
each family keeps its attributes:

* embedder/decoder call: ``self.thinker(...)`` vs ``self.language_model(...)``
* audio token id: ``cfg.audio_token_id`` vs ``self.audio_token_id``
* head selection: tie flag location differs; glm probes lm_head presence.

``splice_and_forward`` takes those as explicit parameters; each thinker
keeps its public ``forward_with_audio`` as a one-line adapter.

```
