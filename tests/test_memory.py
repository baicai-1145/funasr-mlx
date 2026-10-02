"""Admission checks never allocate large tensors or contact the network."""

from types import SimpleNamespace

import pytest

from funasr_mlx.memory import GB, HARD_CEILING, MemoryBudgetExceeded, MemoryPolicy, preflight


def test_default_budget_leaves_room_below_ceiling():
    policy = MemoryPolicy()
    assert policy.project_limit + policy.headroom < HARD_CEILING
    assert policy.cache_limit < policy.mlx_limit < policy.process_limit


@pytest.mark.parametrize(
    "overrides",
    [
        {"project_limit": 10 * GB},
        {"process_limit": 9 * GB},
        {"mlx_limit": 3 * GB},
        {"cache_limit": -1},
        {"poll_seconds": float("nan")},
        {"poll_seconds": 1.0},
        {"process_limit": -1},
        {"headroom": 0},
    ],
)
def test_unsafe_policies_rejected(overrides):
    with pytest.raises(ValueError):
        MemoryPolicy(**overrides)


def test_low_host_memory_refuses_job(monkeypatch, tmp_path):
    from funasr_mlx import memory

    monkeypatch.setattr(
        memory,
        "_psutil",
        lambda: SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=4 * GB)),
    )
    monkeypatch.setattr(memory, "project_rss", lambda root: 0)
    with pytest.raises(MemoryBudgetExceeded, match="available"):
        preflight(tmp_path, MemoryPolicy())


def test_high_project_memory_refuses_job(monkeypatch, tmp_path):
    from funasr_mlx import memory

    monkeypatch.setattr(
        memory,
        "_psutil",
        lambda: SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=12 * GB)),
    )
    monkeypatch.setattr(memory, "project_rss", lambda root: 7 * GB)
    with pytest.raises(MemoryBudgetExceeded, match="Project RSS"):
        preflight(tmp_path, MemoryPolicy())


def test_import_is_lazy():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys,funasr_mlx; assert not any(k.startswith(("mlx","torch")) for k in sys.modules)',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
