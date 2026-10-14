"""Collect GPTQ calibration activations for the LM head through the REAL model.

Full port notes (upstream file map, design rules, known deviations): docs/port-notes/calibrate.py.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .gptq import HessianAccumulator

__all__ = ["HeadActivationCollector", "HeadTarget"]


@dataclass
class HeadTarget:
    """Which module in a live model is the LM head, and how it is called.

    ``kind`` is ``"tied"`` when the head is the token embedding used through
    ``as_linear`` (qwen3 / moss) and ``"linear"`` when the checkpoint has a
    separate ``lm_head`` (glm).  ``hidden_size`` is the head's *input*
    dimension -- the size of the Hessian.
    """

    module: Any
    kind: str
    name: str
    hidden_size: int

    def __post_init__(self) -> None:
        if self.kind not in ("tied", "linear"):
            raise ValueError(f"Unknown head kind: {self.kind}")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

    @classmethod
    def resolve(cls, model: Any, *, head_path: str) -> "HeadTarget":
        """Locate the head on ``model`` and describe how it is called.

        ``head_path`` is a dotted path to the head module (e.g.
        ``thinker.model.embed_tokens`` or ``language_model.lm_head``).
        """
        import mlx.nn as nn

        module: Any = model
        for part in head_path.split("."):
            if not hasattr(module, part):
                raise ValueError(f"No attribute {part!r} on the path to {head_path!r}")
            module = getattr(module, part)
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            raise ValueError(f"{head_path} has no 2-D weight to quantize")
        if isinstance(module, nn.Embedding):
            kind = "tied"
        elif isinstance(module, nn.Linear):
            kind = "linear"
        else:
            raise ValueError(f"{head_path} is neither an nn.Embedding nor an nn.Linear")
        return cls(module=module, kind=kind, name=head_path, hidden_size=int(weight.shape[1]))

    def weight(self) -> np.ndarray:
        import mlx.core as mx

        weight = self.module.weight
        mx.eval(weight)
        return np.array(weight.astype(mx.float32))


@dataclass
class HeadActivationCollector:
    """Accumulates ``H = 2 X^T X`` for one head, one utterance at a time.

    Full notes: docs/port-notes/functions.md :: HeadActivationCollector
    """

    target: HeadTarget
    #: Rows to accumulate per observation before folding into the Hessian. The
    #: hidden states of one utterance are a few hundred rows, so this is a
    #: memory guard rather than a performance knob.
    max_rows_per_call: int = 8192
    accumulator: HessianAccumulator = field(init=False)
    #: Per-language row counts, mirrored from the accumulator for reporting.
    observations: dict[str, int] = field(default_factory=dict, init=False)
    _original: Callable | None = field(default=None, init=False, repr=False)
    _patched_owner: Any = field(default=None, init=False, repr=False)
    _patched_attribute: str = field(default="", init=False, repr=False)
    _pending: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.accumulator = HessianAccumulator(columns=self.target.hidden_size)

    # -- hook management ---------------------------------------------------- #
    def install(self) -> None:
        if self._original is not None:
            raise RuntimeError("The collector is already installed")
        owner = self.target.module
        if self.target.kind == "tied":
            # nn.Embedding.as_linear is looked up on the *instance* dict first
            # (verified against mlx 0.32), so an instance attribute is enough and
            # is restored by deleting it.
            original = owner.as_linear
            self._patched_attribute = "as_linear"

            def hooked(x, *args, **kwargs):
                self._capture(x)
                return original(x, *args, **kwargs)

        else:
            # An nn.Linear head is called through nn.Linear.__call__, which is a
            # *class* attribute: patch the class and restore it in uninstall.
            import mlx.nn as nn

            original = nn.Linear.__call__
            self._patched_owner = nn.Linear
            self._patched_attribute = "__call__"
            module = owner

            def hooked(self, x, *args, **kwargs):  # noqa: ANN001 - mlx signature
                if self is module:
                    collector._capture(x)
                return original(self, x, *args, **kwargs)

            collector = self
            nn.Linear.__call__ = hooked
            self._original = original
            return
        setattr(owner, self._patched_attribute, hooked)
        self._original = original

    def uninstall(self) -> None:
        if self._original is None:
            return
        if self._patched_owner is not None:
            self._patched_owner.__call__ = self._original
        elif hasattr(self.target.module, self._patched_attribute):
            delattr(self.target.module, self._patched_attribute)
        self._original = None
        self._patched_owner = None

    def __enter__(self) -> "HeadActivationCollector":
        self.install()
        return self

    def __exit__(self, *exc_info) -> None:
        self.uninstall()

    # -- capture ------------------------------------------------------------ #
    def _capture(self, x: Any) -> None:
        import mlx.core as mx

        hidden = x.astype(mx.float32)
        mx.eval(hidden)
        array = np.array(hidden)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim > 2:
            array = array.reshape(-1, array.shape[-1])
        if array.shape[-1] != self.target.hidden_size:
            raise ValueError(
                f"Head input has width {array.shape[-1]}, expected "
                f"{self.target.hidden_size}"
            )
        # Zero rows (a batch that happened to be empty) carry no information.
        if array.shape[0]:
            self._pending.append(array)

    def flush(self, language: str) -> int:
        if not self._pending:
            return 0
        rows = 0
        for batch in self._pending:
            for start in range(0, batch.shape[0], self.max_rows_per_call):
                chunk = batch[start : start + self.max_rows_per_call]
                rows += self.accumulator.add(chunk, language)
        self._pending.clear()
        self.observations[language] = self.observations.get(language, 0) + rows
        return rows

    def observe(self, *, language: str, run: Callable[[], Any]) -> int:
        self._pending.clear()
        with self:
            run()
        return self.flush(language)

    # -- results ------------------------------------------------------------ #
    def hessian(self, *, balance: bool = True, cap_fraction: float | None = None) -> np.ndarray:
        return self.accumulator.hessian(balance=balance, cap_fraction=cap_fraction)

    def language_mass(self, **kwargs) -> dict[str, float]:
        return self.accumulator.language_mass(**kwargs)

    def report(self) -> dict:
        return {
            "head": self.target.name,
            "kind": self.target.kind,
            "hidden_size": self.target.hidden_size,
            "rows": self.accumulator.row_counts(),
            "language_mass_balanced": self.language_mass(balance=True),
        }
