# _layout_state.py — port notes

(Extracted from the module docstring; the code keeps only the summary. Reference repo: /Users/baicai1145/repos/funasr, read-only.)

```
Shared owner-type layout kernel (duplication-audit cluster A).

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

```
