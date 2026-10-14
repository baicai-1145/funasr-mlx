# glm_asr_thinker.py — port notes

## module

```
GLM-ASR-Nano thinker: audio tower + projector + Llama decoder assembly.

Prompt (from the checkpoint's chat_template.jinja, verified against the
processor): ``<|user|>\n<|begin_of_audio|><|pad|>*N<|end_of_audio|>\n<|assistant|>\n``
where N = number of projector output frames. The ``<|pad|>`` run's
embeddings are replaced with the tower+projector output.

N for raw audio of T mel frames (no 30s padding -- the reference pads the
mel to 3000 but the encoder is position-local so running real length only
is equivalent):
  conv1 s1: T -> T; conv2 s2: T -> (T+1)//2; projector 4-merge: -> (L-4)//4+1
where L = (T+1)//2.

```

## module

```
GLM-ASR-Nano thinker: audio tower + projector + Llama decoder assembly.

Prompt (from the checkpoint's chat_template.jinja, verified against the
processor): ``<|user|>\n<|begin_of_audio|><|pad|>*N<|end_of_audio|>\n<|assistant|>\n``
where N = number of projector output frames. The ``<|pad|>`` run's
embeddings are replaced with the tower+projector output.

N for raw audio of T mel frames (no 30s padding -- the reference pads the
mel to 3000 but the encoder is position-local so running real length only
is equivalent):
  conv1 s1: T -> T; conv2 s2: T -> (T+1)//2; projector 4-merge: -> (L-4)//4+1
where L = (T+1)//2.

```

## module

```
GLM-ASR-Nano thinker: audio tower + projector + Llama decoder assembly.

Prompt (from the checkpoint's chat_template.jinja, verified against the
processor): ``<|user|>\n<|begin_of_audio|><|pad|>*N<|end_of_audio|>\n<|assistant|>\n``
where N = number of projector output frames. The ``<|pad|>`` run's
embeddings are replaced with the tower+projector output.

N for raw audio of T mel frames (no 30s padding -- the reference pads the
mel to 3000 but the encoder is position-local so running real length only
is equivalent):
  conv1 s1: T -> T; conv2 s2: T -> (T+1)//2; projector 4-merge: -> (L-4)//4+1
where L = (T+1)//2.

```
