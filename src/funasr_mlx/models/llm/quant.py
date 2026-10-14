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

DEFAULT_METHOD = "rtn"

# see docs/port-notes/comments.md :: quant.py:28
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
    # see docs/port-notes/comments.md :: quant.py:46
    target_bits: tuple[tuple[str, int], ...] = ()
    # see docs/port-notes/comments.md :: quant.py:48
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
        # see docs/port-notes/comments.md :: quant.py:173
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
        # see docs/port-notes/comments.md :: quant.py:187
        return tuple(shape)[1] % self.group_size == 0

    def quantizes_module(self, module_path: str, module: nn.Module) -> bool:
        # see docs/port-notes/functions.md :: quantizes_module
        if isinstance(module, nn.Embedding):
            return self.selects_embedding(module_path) and self._fits(module)
        if not isinstance(module, nn.Linear):
            return False
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
    overrides: dict[str, int] = field(default_factory=dict)
    target_bits: dict[str, int] = field(default_factory=dict)
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
                    # see docs/port-notes/comments.md :: quant.py:283
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
    mx.eval(model.parameters())
    return result
