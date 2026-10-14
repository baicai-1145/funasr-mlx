"""Port notes: docs/port-notes/_strict_load.py.md"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields as dataclass_fields
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "_filter",
    "_require",
    "_as_float",
    "_as_int",
    "_known_fields",
    "_nest",
    "convert_checkpoint_tensors",
    "flat_parameter_names",
    "load_state_dict_strict",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    return float(value)


def _filter(cls: type, data: Mapping[str, Any] | None) -> dict[str, Any]:
    known = {f.name for f in dataclass_fields(cls)}
    return {k: v for k, v in (data or {}).items() if k in known}


def _known_fields(cls: type, mapping: Mapping[str, Any], context: str) -> dict[str, Any]:
    allowed = {f.name for f in dataclass_fields(cls)}
    unknown = sorted(set(mapping) - allowed)
    _require(not unknown, f"unknown {context} keys: {unknown}; allowed: {sorted(allowed)}")
    return {k: mapping[k] for k in mapping}


def flat_parameter_names(module: nn.Module) -> dict[str, mx.array]:
    # see docs/port-notes/functions.md :: flat_parameter_names
    flat: dict[str, mx.array] = {}

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, mx.array):
            flat[prefix] = obj
        elif isinstance(obj, dict):
            for key, value in obj.items():
                walk(f"{prefix}.{key}" if prefix else str(key), value)
        elif isinstance(obj, (list, tuple)):
            for index, value in enumerate(obj):
                walk(f"{prefix}.{index}" if prefix else str(index), value)
        elif obj is not None:
            raise TypeError(f"unexpected parameter container {type(obj).__name__}")

    walk("", module.parameters())
    return flat


def convert_checkpoint_tensors(
    state: Mapping[str, mx.array | Any],
    rules: Mapping[str, Callable[[str, mx.array], mx.array]] | None = None,
) -> dict[str, mx.array]:
    # see docs/port-notes/functions.md :: convert_checkpoint_tensors
    converted: dict[str, mx.array] = {}
    for name, value in state.items():
        if not isinstance(value, mx.array):
            value = mx.array(value)
        if rules:
            for suffix, transform in rules.items():
                if name.endswith(suffix):
                    value = transform(name, value)
                    break
        converted[name] = value
    return converted


def _nest(flat: Mapping[str, mx.array]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for name, array in flat.items():
        parts = name.split(".")
        node = root
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = array
    return root


def load_state_dict_strict(
    model: nn.Module,
    state: Mapping[str, mx.array | Any],
    *,
    ignore_prefixes: Sequence[str] = (),
    rules: Mapping[str, Callable[[str, mx.array], mx.array]] | None = None,
) -> dict[str, mx.array]:
    # see docs/port-notes/functions.md :: load_state_dict_strict
    expected = flat_parameter_names(model)
    provided = convert_checkpoint_tensors(state, rules)

    missing = sorted(set(expected) - set(provided))
    if ignore_prefixes:
        missing = [
            name
            for name in missing
            if not any(name.startswith(prefix) for prefix in ignore_prefixes)
        ]
    unexpected = sorted(set(provided) - set(expected))
    if missing or unexpected:
        raise KeyError(
            "checkpoint does not match the model exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )

    mismatched = [
        (name, tuple(expected[name].shape), tuple(provided[name].shape))
        for name in sorted(expected)
        if tuple(expected[name].shape) != tuple(provided[name].shape)
    ]
    if mismatched:
        details = ", ".join(f"{n}: expected {e}, got {g}" for n, e, g in mismatched[:8])
        raise ValueError(
            f"checkpoint tensor shapes do not match ({len(mismatched)} tensor(s)): {details}"
        )

    applied = {name: provided[name] for name in expected}
    model.update(_nest(applied))
    return applied
