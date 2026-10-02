"""Supervisor tests use simulated RSS, never a large real allocation."""

import json
from pathlib import Path
import signal
from types import SimpleNamespace
import sys

import pytest

from funasr_mlx import memory, runtime


def test_guarded_child_is_authenticated():
    assert memory.guarded_child_active(), "Run these tests using the guarded launcher"


def test_policy_environment_alone_is_not_authentication(monkeypatch):
    monkeypatch.delenv("FUNASR_MLX_GUARD_FD", raising=False)
    assert not memory.guarded_child_active()


def test_nonce_tampering_is_rejected(monkeypatch):
    monkeypatch.setenv("FUNASR_MLX_GUARD_NONCE", "0" * 64)
    assert not memory.guarded_child_active()


@pytest.mark.parametrize(
    "command", [[], ["bash", "-c", "false"], ["python3", "-c", "pass"], ["python3", "-m"]]
)
def test_command_validation_happens_before_launch(command, tmp_path):
    with pytest.raises(ValueError):
        memory.run_guarded(command, root=tmp_path, configure_device=False)


def test_lock_contention_covers_current_checkout(tmp_path):
    # The outer supervisor owns the canonical checkout lock already.
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(memory.MemoryBudgetExceeded, match="Another guarded job"):
        memory.run_guarded(
            [sys.executable, "-m", "never_execute"], root=root, configure_device=False
        )


class FakeProcess:
    pid = 123
    returncode = 0
    ended = False

    def poll(self):
        return 0 if self.ended else None

    def wait(self, timeout=None):
        self.ended = True
        return 0


def fake_supervisor(monkeypatch, rss):
    process = FakeProcess()
    killed, captured = [], {}
    monkeypatch.setattr(memory, "_canonical_lock_root", lambda root: root)

    def popen(*args, **kwargs):
        captured.update(kwargs)
        return process

    monkeypatch.setattr(memory.subprocess, "Popen", popen)
    monkeypatch.setattr(
        memory,
        "_psutil",
        lambda: SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=12 * memory.GB)),
    )
    monkeypatch.setattr(memory, "project_rss", lambda *args: 0)
    monkeypatch.setattr(memory, "process_tree_rss", lambda pid: (rss, {pid}))
    monkeypatch.setattr(memory.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    return process, killed, captured


def test_watchdog_stops_simulated_over_budget_job_and_writes_report(monkeypatch, tmp_path):
    process, killed, captured = fake_supervisor(monkeypatch, 3 * memory.GB)
    output = tmp_path / "watchdog.json"
    with pytest.raises(memory.MemoryBudgetExceeded, match="process RSS limit"):
        memory.run_guarded(
            [sys.executable, "-m", "fake"],
            root=tmp_path,
            configure_device=False,
            report_path=output,
        )
    assert killed == [(process.pid, signal.SIGTERM)]
    assert captured["start_new_session"] is True
    assert int(captured["env"]["FUNASR_MLX_SUPERVISOR_FD"]) in captured["pass_fds"]
    assert int(captured["env"]["FUNASR_MLX_GUARD_FD"]) in captured["pass_fds"]
    report = json.loads(output.read_text())
    assert report["violation"] == "process RSS limit"
    assert report["peak_process_rss"] == 3 * memory.GB


def test_failed_launch_closes_both_supervisor_pipe_descriptors(monkeypatch, tmp_path):
    fake_supervisor(monkeypatch, 0)
    descriptors = []
    real_pipe = memory.os.pipe

    def pipe():
        pair = real_pipe()
        descriptors.extend(pair)
        return pair

    def failed_launch(*args, **kwargs):
        raise RuntimeError("simulated launch failure")

    monkeypatch.setattr(memory.os, "pipe", pipe)
    monkeypatch.setattr(memory.subprocess, "Popen", failed_launch)
    with pytest.raises(RuntimeError, match="simulated launch failure"):
        memory.run_guarded([sys.executable, "-m", "fake"], root=tmp_path, configure_device=False)
    assert len(descriptors) == 2
    for descriptor in descriptors:
        with pytest.raises(OSError):
            memory.os.fstat(descriptor)


def test_successful_job_closes_both_supervisor_pipe_descriptors(monkeypatch, tmp_path):
    process, _, _ = fake_supervisor(monkeypatch, 0)
    process.ended = True
    descriptors = []
    real_pipe = memory.os.pipe

    def pipe():
        pair = real_pipe()
        descriptors.extend(pair)
        return pair

    monkeypatch.setattr(memory.os, "pipe", pipe)
    report = memory.run_guarded(
        [sys.executable, "-m", "fake"], root=tmp_path, configure_device=False
    )
    assert report.returncode == 0
    assert len(descriptors) == 2
    for descriptor in descriptors:
        with pytest.raises(OSError):
            memory.os.fstat(descriptor)


def test_supervisor_eof_stops_only_the_owned_process_group(monkeypatch):
    killed = []
    monkeypatch.setenv("FUNASR_MLX_SUPERVISOR_FD", "9")
    monkeypatch.setattr(runtime.os, "getpid", lambda: 4321)
    monkeypatch.setattr(runtime.os, "getpgrp", lambda: 4321)
    monkeypatch.setattr(runtime.os, "read", lambda descriptor, count: b"")
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    class ImmediateThread:
        def __init__(self, target, **kwargs):
            self.target = target
            assert kwargs["daemon"] is True

        def start(self):
            self.target()

    monkeypatch.setattr(runtime.threading, "Thread", ImmediateThread)
    runtime._watch_supervisor()
    assert killed == [(4321, signal.SIGKILL)]


def test_supervisor_watch_rejects_a_foreign_process_group(monkeypatch):
    monkeypatch.setattr(runtime.os, "getpid", lambda: 4321)
    monkeypatch.setattr(runtime.os, "getpgrp", lambda: 1)
    with pytest.raises(RuntimeError, match="own its process group"):
        runtime._watch_supervisor()


def test_timeout_stops_simulated_job(monkeypatch, tmp_path):
    process, killed, _ = fake_supervisor(monkeypatch, 0)
    times = iter([0.0, 10.0, 10.0])
    monkeypatch.setattr(memory.time, "monotonic", lambda: next(times))
    with pytest.raises(memory.MemoryBudgetExceeded, match="timeout"):
        memory.run_guarded(
            [sys.executable, "-m", "fake"], root=tmp_path, configure_device=False, timeout_seconds=5
        )
    assert killed == [(process.pid, signal.SIGTERM)]
