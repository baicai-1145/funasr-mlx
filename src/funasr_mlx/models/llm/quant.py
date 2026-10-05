"""W4A16 weight quantization for the LLM-ASR decoder stacks.

Scheme: MLX **affine group quantization** (``mx.quantize``), the same layout
``mlx-lm`` uses -- per-group ``scales``/``biases`` carried in the checkpoint
dtype and the weights packed into ``uint32`` words.  ``nn.quantize`` turns each
selected ``nn.Linear`` into ``nn.QuantizedLinear``, whose ``weight`` /
``scales`` / ``biases`` parameters are exactly the three tensors the converter
writes, so a quantized checkpoint strict-loads through the ordinary
``pipeline.load_model`` path with no special-casing anywhere else.

The whole configuration lives in the native ``config.json``::

    "quantization": {"bits": 4, "group_size": 64, "mode": "affine",
                     "targets": ["thinker.model.layers."]}

``targets`` are module-path prefixes.  Only ``nn.Linear`` modules under one of
them are quantized, so the token embeddings, the LM head, every convolution and
every norm keep the checkpoint dtype.  The converter in
:mod:`funasr_mlx.checkpoints` applies the *same* rule to tensor names
(``<module path>.weight``); because the two rules must agree for the strict
loader to find ``.scales``/``.biases``, any divergence fails loudly at load
time instead of silently mis-binding weights.

Why decoder-only by default: decode at bs=1 streams the decoder weights once
per step, so 4-bit decoder weights cut the step's dominant traffic ~3.5x.  The
audio tower runs once per utterance (prefill side), so quantizing it buys
memory but not decode time; it is offered as an extra target prefix for
families whose memory budget needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "DEFAULT_BITS",
    "DEFAULT_GROUP_SIZE",
    "NON_LINEAR_WEIGHT_NAMES",
    "QuantizationSpec",
    "apply_quantization",
]

DEFAULT_BITS = 4
DEFAULT_GROUP_SIZE = 64

#: Substrings that mark a rank-2 ``.weight`` tensor which must NOT be quantized
#: even when it sits under a quantized prefix: the tied/untied token embedding,
#: a learned position table, or the sinusoidal table the qwen3-asr tower derives.
#: The model-side rule is "only ``nn.Linear``", so these are exactly the
#: exceptions to the tensor-name rule and they must stay in sync with it.
NON_LINEAR_WEIGHT_NAMES = (
    "embed_tokens",
    "embed_positions",
    "lm_head",
    "positional_embedding",
)


@dataclass(frozen=True)
class QuantizationSpec:
    """One family's quantization plan, as declared in ``config.json``."""

    bits: int = DEFAULT_BITS
    group_size: int = DEFAULT_GROUP_SIZE
    mode: str = "affine"
    targets: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Any) -> "QuantizationSpec":
        if not isinstance(data, dict):
            raise ValueError("Quantization configuration must be a mapping")
        bits = data.get("bits", DEFAULT_BITS)
        group_size = data.get("group_size", DEFAULT_GROUP_SIZE)
        mode = data.get("mode", "affine")
        targets = data.get("targets", ())
        if bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"Unsupported quantization bit width: {bits}")
        if group_size not in (32, 64, 128):
            raise ValueError(f"Unsupported quantization group size: {group_size}")
        if mode not in ("affine", "mxfp4", "nvfp4"):
            raise ValueError(f"Unsupported quantization mode: {mode}")
        if isinstance(targets, str):
            targets = (targets,)
        if (
            not isinstance(targets, (list, tuple))
            or not targets
            or not all(isinstance(entry, str) and entry for entry in targets)
        ):
            raise ValueError("Quantization targets must be a nonempty list of module paths")
        return cls(bits=bits, group_size=group_size, mode=mode, targets=tuple(targets))

    def to_dict(self) -> dict:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "targets": list(self.targets),
        }

    def selects(self, module_path: str) -> bool:
        """Whether ``module_path`` sits under one of the declared prefixes."""
        return any(module_path.startswith(target) for target in self.targets)

    def is_linear_weight(self, name: str, shape: Iterable[int]) -> bool:
        """Whether ``name`` is a Linear weight the target prefixes select.

        Rank-2 ``.weight`` under a selected prefix and not one of the
        :data:`NON_LINEAR_WEIGHT_NAMES` exceptions -- i.e. a tensor the model
        side would have turned into a ``QuantizedLinear`` if the group size
        fitted. This is the predicate the converter uses to decide whether a
        tensor it did *not* pack is a legitimate skip or an unnoticed miss.
        """
        if not name.endswith(".weight"):
            return False
        module_path = name[: -len(".weight")]
        if not self.selects(module_path):
            return False
        if any(marker in module_path for marker in NON_LINEAR_WEIGHT_NAMES):
            return False
        return len(tuple(shape)) == 2

    def quantizes_tensor(self, name: str, shape: Iterable[int]) -> bool:
        """Whether the converter should quantize checkpoint tensor ``name``.

        Mirrors the model-side predicate exactly: a Linear weight (see
        :meth:`is_linear_weight`) whose input dimension the group size divides.
        """
        if not self.is_linear_weight(name, shape):
            return False
        return tuple(shape)[1] % self.group_size == 0

    def quantizes_module(self, module_path: str, module: nn.Module) -> bool:
        """Whether ``module`` (a live ``nn.Module``) should become quantized."""
        if not isinstance(module, nn.Linear):
            return False
        if any(marker in module_path for marker in NON_LINEAR_WEIGHT_NAMES):
            return False
        if not self.selects(module_path):
            return False
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            return False
        return weight.shape[1] % self.group_size == 0


@dataclass
class QuantizationResult:
    """What :func:`apply_quantization` actually did to a model."""

    bits: int
    group_size: int
    mode: str
    targets: tuple[str, ...]
    quantized: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "targets": list(self.targets),
            "linear_layers_quantized": len(self.quantized),
            "linear_layers_skipped": len(self.skipped),
        }


def apply_quantization(model: nn.Module, data: Any) -> QuantizationResult | None:
    """Quantize ``model`` in place per the ``config.json`` section ``data``.

    Called by :func:`funasr_mlx.pipeline.load_model` *before* the expected
    parameter set is computed, so a quantized checkpoint's ``.scales`` and
    ``.biases`` entries are part of the strict key/shape check.  Returns
    ``None`` when the configuration carries no quantization section, i.e. the
    checkpoint is full precision.
    """
    if data is None:
        return None
    spec = QuantizationSpec.from_dict(data)
    if not spec.targets:
        raise ValueError("A quantization section must declare at least one target prefix")
    result = QuantizationResult(
        bits=spec.bits, group_size=spec.group_size, mode=spec.mode, targets=spec.targets
    )

    def predicate(path: str, module: nn.Module) -> bool:
        if isinstance(module, nn.Linear) and spec.selects(path):
            if spec.quantizes_module(path, module):
                result.quantized.append(path)
                return True
            result.skipped.append(path)
        return False

    nn.quantize(
        model,
        group_size=spec.group_size,
        bits=spec.bits,
        mode=spec.mode,
        class_predicate=predicate,
    )
    # Every parameter of a quantized module is frozen by nn.quantize; inference
    # never needs gradients, and an unfrozen parameter would be re-quantized by
    # a later nn.quantize call.
    mx.eval(model.parameters())
    return result
