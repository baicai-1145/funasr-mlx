"""Shared owner-type layout kernel (duplication-audit cluster A).

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/_layout_state.py.md
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as mxnn

def layout_rule(module: mxnn.Module, names: Mapping[str, str]) -> str:
    if isinstance(module, mxnn.Conv1d):
        return names["conv1d"]
    if isinstance(module, mxnn.Conv2d):
        return names["conv2d"]
    return names["identity"]


def convert_tensor(name: str, array: mx.array, rule: str, names: Sequence[str]) -> mx.array:
    """Apply the transpose for ``rule``; ``names`` is the family's identity set."""
    conv1d, conv2d = names[0], names[1]
    if rule == conv1d:
        # torch (out, in/groups, k) -> mlx (out, k, in/groups)
        return mx.transpose(array, (0, 2, 1))
    if rule == conv2d:
        # torch (out, in, kh, kw) -> mlx (out, kh, kw, in)
        return mx.transpose(array, (0, 2, 3, 1))
    return array


def resolve_child(module: Any, key: str) -> Any:
    child = module.get(key)
    if child is None:
        child = getattr(module, key, None)
    return child


def assign(module: mxnn.Module, parts: Sequence[str], array: mx.array) -> None:
    node = module
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = getattr(node, part, None)
        node = child
    setattr(node, parts[-1], array)
