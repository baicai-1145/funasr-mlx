"""W4A16 weight quantization for the LLM-ASR decoder stacks.

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/quant.py.md
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
    "activation_dtype",
    "apply_quantization",
]

DEFAULT_BITS = 4
DEFAULT_GROUP_SIZE = 64

#: How the stored quantization values were computed. ``rtn`` is ``mx.quantize``'s
#: own round-to-nearest; ``gptq`` is Hessian-based error compensation. Both
#: produce the identical runtime layout, so this is provenance, not behaviour.
DEFAULT_METHOD = "rtn"

#: Substrings that mark a rank-2 ``.weight`` tensor which must NOT be quantized
#: through the ``nn.Linear`` path, even when it sits under a quantized prefix:
#: the untied ``lm_head``, a learned position table, or the sinusoidal table the
#: qwen3-asr tower derives. ``embed_tokens`` is handled separately (it is an
#: ``nn.Embedding``, not an ``nn.Linear``, and can only be selected by an
#: explicit ``embed_tokens`` target prefix); the marker is kept here so a Linear
#: named ``...embed_tokens`` is still never quantized by the Linear rule.
NON_LINEAR_WEIGHT_NAMES = (
    "embed_tokens",
    "embed_positions",
    "lm_head",
    "positional_embedding",
)


@dataclass(frozen=True)
class QuantizationSpec:
    # see docs/port-notes/functions.md :: QuantizationSpec

    bits: int = DEFAULT_BITS
    group_size: int = DEFAULT_GROUP_SIZE
    mode: str = "affine"
    targets: tuple[str, ...] = ()
    method: str = DEFAULT_METHOD
    #: Optional per-target bit widths, e.g. ``{"thinker.model.embed_tokens": 8}``.
    #: The LONGEST matching target prefix wins, so a specific module can differ
    #: from the general rule.  This exists because the tied embedding and the
    #: decoder Linears do not have the same sensitivity: measured on
    #: qwen3-asr-0.6b, a w4 head costs ~10% mean span token edit rate while the
    #: decoder Linears at w4 are fine, so spending 8 bits on the head and 4 on
    #: the stack is a better use of the budget than 4 bits on both.
    target_bits: tuple[tuple[str, int], ...] = ()
    #: Optional per-target group sizes, e.g. ``{"thinker.model.embed_tokens": 32}``.
    #: Same longest-prefix rule as ``target_bits``.  This is mlx-lm's AWQ
    #: default for the tied embedding (``embed_group_size=32`` beside a g64
    #: stack): the embedding's side tensors are what its w4 accuracy needs,
    #: not the stack's.
    target_groups: tuple[tuple[str, int], ...] = ()

    def bits_for(self, module_path: str) -> int:
        best_prefix = ""
        best_bits = self.bits
        for prefix, value in self.target_bits:
            if module_path.startswith(prefix) and len(prefix) > len(best_prefix):
                best_prefix, best_bits = prefix, value
        return best_bits

    def group_for(self, module_path: str) -> int:
        best_prefix = ""
        best_group = self.group_size
        for prefix, value in self.target_groups:
            if module_path.startswith(prefix) and len(prefix) > len(best_prefix):
                best_prefix, best_group = prefix, value
        return best_group

    def bits_for_weight(self, name: str) -> int:
        module = name[: -len(".weight")] if name.endswith(".weight") else name
        return self.bits_for(module)

    @classmethod
    def from_dict(cls, data: Any) -> "QuantizationSpec":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise ValueError("Quantization configuration must be a mapping")
        bits = data.get("bits", DEFAULT_BITS)
        group_size = data.get("group_size", DEFAULT_GROUP_SIZE)
        mode = data.get("mode", "affine")
        targets = data.get("targets", ())
        method = data.get("method", DEFAULT_METHOD)
        raw_target_bits = data.get("target_bits", {}) or {}
        raw_target_groups = data.get("target_groups", {}) or {}
        if bits not in (2, 3, 4, 5, 6, 8):
            raise ValueError(f"Unsupported quantization bit width: {bits}")
        if group_size not in (32, 64, 128):
            raise ValueError(f"Unsupported quantization group size: {group_size}")
        if mode not in ("affine", "mxfp4", "nvfp4"):
            raise ValueError(f"Unsupported quantization mode: {mode}")
        if method not in ("rtn", "gptq"):
            raise ValueError(f"Unsupported quantization method: {method}")
        if isinstance(targets, str):
            targets = (targets,)
        if (
            not isinstance(targets, (list, tuple))
            or not targets
            or not all(isinstance(entry, str) and entry for entry in targets)
        ):
            raise ValueError("Quantization targets must be a nonempty list of module paths")
        if not isinstance(raw_target_bits, dict):
            raise ValueError("target_bits must be a mapping of prefix -> bit width")
        target_bits = []
        for prefix, value in raw_target_bits.items():
            if not isinstance(prefix, str) or not prefix:
                raise ValueError("target_bits keys must be nonempty module paths")
            if value not in (2, 3, 4, 5, 6, 8):
                raise ValueError(f"Unsupported bit width for {prefix!r}: {value}")
            target_bits.append((prefix, int(value)))
        # Sorted longest-first so the tie-break is deterministic regardless of
        # dict order; bits_for() also compares lengths, this is belt-and-braces.
        target_bits.sort(key=lambda entry: (-len(entry[0]), entry[0]))
        target_groups = []
        for prefix, value in raw_target_groups.items():
            if not isinstance(prefix, str) or not prefix:
                raise ValueError("target_groups keys must be nonempty module paths")
            if value not in (16, 32, 64, 128):
                raise ValueError(f"Unsupported group size for {prefix!r}: {value}")
            target_groups.append((prefix, int(value)))
        target_groups.sort(key=lambda entry: (-len(entry[0]), entry[0]))
        return cls(
            bits=bits,
            group_size=group_size,
            mode=mode,
            targets=tuple(targets),
            method=method,
            target_bits=tuple(target_bits),
            target_groups=tuple(target_groups),
        )

    def to_dict(self) -> dict:
        payload = {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "targets": list(self.targets),
            "method": self.method,
        }
        if self.target_bits:
            payload["target_bits"] = dict(self.target_bits)
        if self.target_groups:
            payload["target_groups"] = dict(self.target_groups)
        return payload

    def selects(self, module_path: str) -> bool:
        return any(module_path.startswith(target) for target in self.targets)

    def selects_embedding(self, module_path: str) -> bool:
        # see docs/port-notes/functions.md :: selects_embedding
        if not module_path.endswith("embed_tokens"):
            return False
        return any(
            target.startswith(module_path) or module_path.startswith(target)
            for target in self.targets
            if target.rstrip(".").endswith("embed_tokens")
        )

    def quantizes_embedding(self, name: str, shape: Iterable[int]) -> bool:
        # see docs/port-notes/functions.md :: quantizes_embedding
        if not name.endswith("embed_tokens.weight"):
            return False
        dims = tuple(shape)
        if len(dims) != 2:
            return False
        if not self.selects_embedding(name[: -len(".weight")]):
            return False
        return dims[1] % self.group_size == 0

    def is_linear_weight(self, name: str, shape: Iterable[int]) -> bool:
        # see docs/port-notes/functions.md :: is_linear_weight
        if not name.endswith(".weight"):
            return False
        module_path = name[: -len(".weight")]
        if not self.selects(module_path):
            return False
        # An EXPLICIT target (the module path itself, not a prefix it sits
        # under) overrides the non-Linear markers: glm's untied ``lm_head`` IS
        # a real nn.Linear, and the marker exists for the tied families where
        # the head is an Embedding handled by the embedding rule. Naming a
        # module directly is the operator saying "quantize THIS", which must
        # not be silently vetoed by a table written for another topology.
        explicitly_named = module_path in self.targets
        if not explicitly_named and any(
            marker in module_path for marker in NON_LINEAR_WEIGHT_NAMES
        ):
            return False
        return len(tuple(shape)) == 2

    def quantizes_tensor(self, name: str, shape: Iterable[int]) -> bool:
        # see docs/port-notes/functions.md :: quantizes_tensor
        if self.quantizes_embedding(name, shape):
            return True
        if not self.is_linear_weight(name, shape):
            return False
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
        return tuple(shape)[1] % self.group_size == 0

    def quantizes_module(self, module_path: str, module: nn.Module) -> bool:
        # see docs/port-notes/functions.md :: quantizes_module
        if isinstance(module, nn.Embedding):
            return self.selects_embedding(module_path) and self._fits(module)
        if not isinstance(module, nn.Linear):
            return False
        # An EXPLICITLY named target overrides the non-Linear markers (glm's
        # untied lm_head is a real Linear; the marker exists for tied
        # topologies where the head is the Embedding -- see is_linear_weight).
        explicitly_named = module_path in self.targets
        if not explicitly_named and any(
            marker in module_path for marker in NON_LINEAR_WEIGHT_NAMES
        ):
            return False
        if not self.selects(module_path):
            return False
        return self._fits(module)

    def _fits(self, module: nn.Module) -> bool:
        weight = getattr(module, "weight", None)
        return weight is not None and weight.ndim == 2 and weight.shape[1] % self.group_size == 0


@dataclass
class QuantizationResult:

    bits: int
    group_size: int
    mode: str
    targets: tuple[str, ...]
    quantized: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    #: Modules quantized at a width other than ``bits`` (see
    #: :meth:`QuantizationSpec.bits_for`), recorded so a report shows the plan
    #: that was actually applied rather than only the default.
    overrides: dict[str, int] = field(default_factory=dict)
    target_bits: dict[str, int] = field(default_factory=dict)
    #: Modules quantized at a group size other than ``group_size`` (see
    #: :meth:`QuantizationSpec.group_for`).
    group_overrides: dict[str, int] = field(default_factory=dict)
    target_groups: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        payload = {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "targets": list(self.targets),
            "linear_layers_quantized": len(self.quantized),
            "linear_layers_skipped": len(self.skipped),
        }
        if self.overrides:
            payload["bit_width_overrides"] = dict(self.overrides)
        if self.group_overrides:
            payload["group_size_overrides"] = dict(self.group_overrides)
        return payload


def activation_dtype(module: nn.Module) -> mx.Dtype:
    """The dtype a weight-bearing module emits, for activations to follow.

    Full notes: docs/port-notes/functions.md :: activation_dtype
    """
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ValueError("Module has no weight to read an activation dtype from")
    if isinstance(module, (nn.QuantizedEmbedding, nn.QuantizedLinear)):
        scales = getattr(module, "scales", None)
        if scales is not None:
            return scales.dtype
    return weight.dtype


def apply_quantization(model: nn.Module, data: Any) -> QuantizationResult | None:
    # see docs/port-notes/functions.md :: apply_quantization
    if data is None:
        return None
    spec = QuantizationSpec.from_dict(data)
    if not spec.targets:
        raise ValueError("A quantization section must declare at least one target prefix")
    result = QuantizationResult(
        bits=spec.bits, group_size=spec.group_size, mode=spec.mode, targets=spec.targets
    )
    if spec.target_bits:
        result.target_bits = dict(spec.target_bits)
    if spec.target_groups:
        result.target_groups = dict(spec.target_groups)

    def predicate(path: str, module: nn.Module):
        # Only the two weight-bearing module kinds the scheme supports are
        # considered; everything else (norms, convs, position tables) is left
        # at the checkpoint dtype and is not reported as a skip.
        if isinstance(module, (nn.Linear, nn.Embedding)) and (
            spec.selects(path) or spec.selects_embedding(path)
        ):
            if spec.quantizes_module(path, module):
                result.quantized.append(path)
                bits = spec.bits_for(path)
                group = spec.group_for(path)
                if bits != spec.bits:
                    result.overrides[path] = bits
                if group != spec.group_size:
                    result.group_overrides[path] = group
                if bits != spec.bits or group != spec.group_size:
                    # Returning a dict is nn.quantize's per-module escape hatch.
                    # The override must be honoured here, at load time, or the
                    # module is built at the wrong width/group and the strict
                    # loader rejects the checkpoint's shapes -- so this is not
                    # cosmetic.
                    return {"group_size": group, "bits": bits, "mode": spec.mode}
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
