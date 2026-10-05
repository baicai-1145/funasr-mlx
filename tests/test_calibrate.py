"""Tests for the LM-head activation collector.

The collector is the model-side half of GPTQ: it patches the head call on a live
MLX model, runs the real forward, and folds the hidden states into
``H = 2 X^T X`` per language.  The hook mechanism is the risky part -- an
``nn.Embedding`` head is called through ``as_linear`` (looked up on the instance)
while an ``nn.Linear`` head goes through ``nn.Linear.__call__`` (a *class*
attribute, shared by every Linear in the model) -- so it is tested on real
``mlx.nn`` modules here rather than trusted.

Everything is tiny: no checkpoint, no audio, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from funasr_mlx.models.llm.calibrate import HeadActivationCollector, HeadTarget  # noqa: E402


class _TiedModel(nn.Module):
    """A model whose head is a tied ``nn.Embedding`` reached through ``as_linear``."""

    def __init__(self, vocab: int = 32, hidden: int = 8) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.other = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x):
        return self.embed_tokens.as_linear(self.other(x))


class _UntiedModel(nn.Module):
    """A model with a separate ``lm_head``, plus another Linear to prove isolation."""

    def __init__(self, vocab: int = 32, hidden: int = 8) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.mlp = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x):
        return self.lm_head(self.mlp(x))


# --------------------------------------------------------------------------- #
# HeadTarget.resolve
# --------------------------------------------------------------------------- #
def test_resolve_finds_a_tied_embedding_head():
    model = _TiedModel()
    target = HeadTarget.resolve(model, head_path="embed_tokens")
    assert target.kind == "tied"
    assert target.hidden_size == 8
    assert target.name == "embed_tokens"


def test_resolve_finds_an_untied_linear_head():
    model = _UntiedModel()
    target = HeadTarget.resolve(model, head_path="lm_head")
    assert target.kind == "linear"
    assert target.hidden_size == 8


def test_resolve_walks_a_dotted_path():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.thinker = _TiedModel()

    target = HeadTarget.resolve(Wrapper(), head_path="thinker.embed_tokens")
    assert target.kind == "tied"


def test_resolve_rejects_a_missing_path():
    with pytest.raises(ValueError, match="No attribute"):
        HeadTarget.resolve(_TiedModel(), head_path="thinker.model.embed_tokens")


def test_resolve_rejects_a_module_without_a_2d_weight():
    class NoWeight(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.RMSNorm(8)

    with pytest.raises(ValueError, match="no 2-D weight"):
        HeadTarget.resolve(NoWeight(), head_path="norm")


def test_head_target_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="Unknown head kind"):
        HeadTarget(module=_TiedModel(), kind="magic", name="x", hidden_size=8)


def test_weight_returns_float32_on_the_host():
    model = _TiedModel()
    target = HeadTarget.resolve(model, head_path="embed_tokens")
    weight = target.weight()
    assert weight.dtype == np.float32
    assert weight.shape == (32, 8)


# --------------------------------------------------------------------------- #
# capture: tied head
# --------------------------------------------------------------------------- #
def test_tied_head_capture_matches_a_manual_forward():
    """The Hessian must equal ``2 X^T X`` over the real head inputs."""
    model = _TiedModel()
    target = HeadTarget.resolve(model, head_path="embed_tokens")
    collector = HeadActivationCollector(target)
    x = mx.array(np.random.RandomState(0).randn(4, 8).astype(np.float32))
    expected = np.array(model.other(x))
    # The head input is `other(x)`; capture it through the real forward.
    rows = collector.observe(language="zh", run=lambda: model(x))
    assert rows == 4
    hessian = collector.hessian(balance=False)
    assert np.allclose(hessian, 2.0 * (expected.T @ expected), atol=1e-4)


def test_tied_head_hook_is_removed_after_the_observation():
    model = _TiedModel()
    target = HeadTarget.resolve(model, head_path="embed_tokens")
    collector = HeadActivationCollector(target)
    x = mx.array(np.zeros((2, 8), dtype=np.float32))
    with collector:
        model(x)
    assert "as_linear" not in model.embed_tokens.__dict__
    # The model still works normally after the hook is gone.
    model(x)


def test_tied_head_hook_is_removed_when_the_body_raises():
    model = _TiedModel()
    target = HeadTarget.resolve(model, head_path="embed_tokens")
    collector = HeadActivationCollector(target)

    def boom():
        model(mx.array(np.zeros((2, 8), dtype=np.float32)))
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        collector.observe(language="zh", run=boom)
    assert "as_linear" not in model.embed_tokens.__dict__


# --------------------------------------------------------------------------- #
# capture: untied head, and isolation from other Linears
# --------------------------------------------------------------------------- #
def test_untied_head_capture_ignores_other_linears():
    """Patching ``nn.Linear.__call__`` must only record the head's own inputs."""
    model = _UntiedModel()
    target = HeadTarget.resolve(model, head_path="lm_head")
    collector = HeadActivationCollector(target)
    x = mx.array(np.random.RandomState(1).randn(3, 8).astype(np.float32))
    expected = np.array(model.mlp(x))
    rows = collector.observe(language="en", run=lambda: model(x))
    assert rows == 3
    hessian = collector.hessian(balance=False)
    assert np.allclose(hessian, 2.0 * (expected.T @ expected), atol=1e-4)


def test_untied_head_class_hook_is_restored():
    original = nn.Linear.__call__
    model = _UntiedModel()
    target = HeadTarget.resolve(model, head_path="lm_head")
    with HeadActivationCollector(target):
        model(mx.array(np.zeros((2, 8), dtype=np.float32)))
    assert nn.Linear.__call__ is original


def test_untied_head_class_hook_is_restored_when_the_body_raises():
    original = nn.Linear.__call__
    model = _UntiedModel()
    target = HeadTarget.resolve(model, head_path="lm_head")

    def boom():
        model(mx.array(np.zeros((2, 8), dtype=np.float32)))
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError):
        HeadActivationCollector(target).observe(language="en", run=boom)
    assert nn.Linear.__call__ is original


def test_installing_twice_is_refused():
    model = _TiedModel()
    collector = HeadActivationCollector(HeadTarget.resolve(model, head_path="embed_tokens"))
    collector.install()
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            collector.install()
    finally:
        collector.uninstall()
    # uninstall is idempotent
    collector.uninstall()


# --------------------------------------------------------------------------- #
# language tagging and the balance guard
# --------------------------------------------------------------------------- #
def test_rows_are_tagged_by_language():
    model = _TiedModel()
    collector = HeadActivationCollector(HeadTarget.resolve(model, head_path="embed_tokens"))
    for language, rows in (("zh", 6), ("th", 4), ("vi", 4)):
        x = mx.array(np.random.RandomState(rows).randn(rows, 8).astype(np.float32))
        assert collector.observe(language=language, run=lambda: model(x)) == rows
    assert collector.accumulator.row_counts() == {"zh": 6, "th": 4, "vi": 4}


def test_language_mass_is_balanced_by_default():
    model = _TiedModel()
    collector = HeadActivationCollector(HeadTarget.resolve(model, head_path="embed_tokens"))
    for language, rows in (("zh", 500), ("en", 500), ("th", 50)):
        x = mx.array(np.random.RandomState(rows).randn(rows, 8).astype(np.float32))
        collector.observe(language=language, run=lambda: model(x))
    unbalanced = collector.language_mass(balance=False)
    assert unbalanced["zh"] > 0.4
    balanced = collector.language_mass(balance=True)
    assert max(balanced.values()) <= 0.34
    assert abs(sum(balanced.values()) - 1.0) < 1e-9


def test_report_summarises_the_head_and_rows():
    model = _UntiedModel()
    collector = HeadActivationCollector(HeadTarget.resolve(model, head_path="lm_head"))
    collector.observe(
        language="ja", run=lambda: model(mx.array(np.zeros((2, 8), dtype=np.float32)))
    )
    report = collector.report()
    assert report["head"] == "lm_head"
    assert report["kind"] == "linear"
    assert report["hidden_size"] == 8
    assert report["rows"] == {"ja": 2}
    assert set(report["language_mass_balanced"]) == {"ja"}


def test_a_forward_that_never_calls_the_head_records_nothing():
    """A silent zero would corrupt the Hessian, so it must be visible as 0."""
    model = _TiedModel()
    collector = HeadActivationCollector(HeadTarget.resolve(model, head_path="embed_tokens"))
    assert collector.observe(language="zh", run=lambda: None) == 0
    assert collector.accumulator.row_counts() == {}


def test_wrong_hidden_width_is_rejected():
    model = _TiedModel(hidden=8)
    collector = HeadActivationCollector(HeadTarget.resolve(model, head_path="embed_tokens"))
    with pytest.raises(ValueError, match="expected 8"):
        collector._capture(mx.array(np.zeros((2, 5), dtype=np.float32)))


def test_two_dimensional_and_ragged_inputs_are_flattened():
    model = _TiedModel()
    collector = HeadActivationCollector(HeadTarget.resolve(model, head_path="embed_tokens"))
    collector._capture(mx.array(np.zeros((2, 3, 8), dtype=np.float32)))
    collector._capture(mx.array(np.zeros(8, dtype=np.float32)))
    assert collector.flush("zh") == 7


def test_large_observations_are_chunked_but_counted_once():
    model = _TiedModel()
    collector = HeadActivationCollector(
        HeadTarget.resolve(model, head_path="embed_tokens"), max_rows_per_call=8
    )
    collector._capture(mx.array(np.zeros((20, 8), dtype=np.float32)))
    assert collector.flush("zh") == 20
    assert collector.accumulator.row_counts() == {"zh": 20}
