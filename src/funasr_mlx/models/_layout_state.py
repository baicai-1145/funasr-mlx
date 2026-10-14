"""Shared owner-type layout kernel (duplication-audit cluster A).

The campplus/sanm/whisper families each reimplemented the same mechanism:
walk a module tree, derive each parameter's PyTorch->MLX layout rule from
the OWNING module type (never from the name -- the same attribute name can
be a Linear in one block and a Conv1d in a sibling), and convert checkpoint
tensors accordingly.

The three copies differed in: rule coverage (campplus/sanm know Conv1d +
Conv2d; whisper only Conv1d), docstring wording, and small extras kept in
the families (campplus's LayoutReport/LoadReport objects, whisper's
non-persistent-buffer filtering). This kernel keeps the superset behavior:

* ``layout_rule(module)``: Conv1d/Conv2d/identity -- usable by every family.
* ``convert_tensor(name, array, rule)``: the transpose math, one copy.
* ``resolve_child(module, key)``: mapping-storage child access.
* ``assign(module, parts, array)``: dotted-path setter used by loaders.

Family-specific report types, name-based fallbacks and buffer filters stay
in their modules -- only the mechanism is shared here.
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
