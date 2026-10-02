"""Conservative memory admission and serialized process supervision.

The 10 GB ceiling applies to this project, not unrelated applications.
RSS includes resident unified memory; MLX limits add an allocator-side bound.
A watchdog is a secondary defense, not a substitute for input shape limits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import signal
import stat
import secrets
import subprocess
import sys
import time
from typing import Sequence

HARD_CEILING = 10_000_000_000
MB = 1_000_000
GB = 1_000_000_000


class MemoryBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryPolicy:
    process_limit: int = 2 * GB
    project_limit: int = 8 * GB
    mlx_limit: int = 1_250 * MB
    cache_limit: int = 32 * MB
    min_available: int = 3 * GB
    headroom: int = 500 * MB
    poll_seconds: float = 0.05

    def __post_init__(self):
        for name in ("process_limit", "project_limit", "mlx_limit", "min_available", "headroom"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive byte count")
        if self.project_limit + self.headroom >= HARD_CEILING:
            raise ValueError("Project budget plus headroom must remain below 10 GB")
        if self.process_limit + self.headroom > self.project_limit:
            raise ValueError("Process budget leaves insufficient project headroom")
        if not 0 <= self.cache_limit < self.mlx_limit < self.process_limit:
            raise ValueError("Require cache_limit < mlx_limit < process_limit")
        if not math.isfinite(self.poll_seconds) or not 0.01 <= self.poll_seconds <= 0.25:
            raise ValueError("Watchdog interval must be between 0.01 and 0.25 seconds")


@dataclass
class MemoryReport:
    command: list[str]
    returncode: int = 0
    peak_process_rss: int = 0
    peak_project_rss: int = 0
    lowest_available: int = 0
    elapsed_seconds: float = 0.0
    violation: str | None = None


def guarded_child_active() -> bool:
    """Validate an inherited supervisor lock, nonce and parent identity."""
    try:
        descriptor = int(os.environ["FUNASR_MLX_GUARD_FD"])
        metadata = json.loads(os.pread(descriptor, 4096, 0))
        opened = os.fstat(descriptor)
        expected = os.stat(os.environ["FUNASR_MLX_GUARD_LOCK"])
        policy = json.loads(os.environ["FUNASR_MLX_MEMORY_POLICY"])
        alive = os.fstat(int(os.environ["FUNASR_MLX_SUPERVISOR_FD"]))
        return (
            stat.S_ISFIFO(alive.st_mode)
            and opened.st_dev == expected.st_dev
            and opened.st_ino == expected.st_ino
            and metadata["pid"] == os.getppid()
            and secrets.compare_digest(metadata["nonce"], os.environ["FUNASR_MLX_GUARD_NONCE"])
            and metadata["policy"] == policy
        )
    except (KeyError, ValueError, OSError, TypeError):
        return False


def _psutil():
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required for guarded execution") from exc
    return psutil


def _is_within(path: str, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root)
        return True
    except (ValueError, OSError):
        return False


def process_tree_rss(pid: int) -> tuple[int, set[int]]:
    psutil = _psutil()
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.Error:
        return 0, set()
    rss, pids = 0, set()
    for process in processes:
        try:
            rss += process.memory_info().rss
            pids.add(process.pid)
        except psutil.Error:
            continue
    return rss, pids


@lru_cache(maxsize=4)
def _project_roots(root: Path) -> tuple[Path, ...]:
    roots = [root]
    result = subprocess.run(
        ["git", "-C", str(root), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                checkout = Path(line[len("worktree ") :]).resolve()
                roots.append(checkout)
                if ".pi" in checkout.parts:
                    # Herdr stores worktrees under the lead repository.
                    index = checkout.parts.index(".pi")
                    roots.append(Path(*checkout.parts[:index]))
    return tuple(dict.fromkeys(roots))


def project_rss(root: Path, extra_pids: set[int] | None = None) -> int:
    psutil = _psutil()
    extra_pids = extra_pids or set()
    roots = _project_roots(root)
    total = 0
    for process in psutil.process_iter():
        try:
            cwd = process.cwd()
            if process.pid in extra_pids or any(_is_within(cwd, path) for path in roots):
                total += process.memory_info().rss
        except (psutil.Error, OSError):
            continue
    return total


def preflight(root: Path, policy: MemoryPolicy) -> dict[str, int]:
    psutil = _psutil()
    available = psutil.virtual_memory().available
    _project_roots.cache_clear()
    current = project_rss(root)
    if current + policy.process_limit + policy.headroom > policy.project_limit:
        raise MemoryBudgetExceeded("Project RSS plus job reservation exceeds budget")
    if available < policy.process_limit + policy.min_available:
        raise MemoryBudgetExceeded(
            f"Only {available / GB:.2f} GB available; need "
            f"{(policy.process_limit + policy.min_available) / GB:.2f} GB"
        )
    return {"available": available, "project_rss": current}


def configure_mlx(policy: MemoryPolicy) -> dict[str, int]:
    import mlx.core as mx

    # Configure before constructing any model or evaluating any array.
    mx.set_memory_limit(policy.mlx_limit)
    mx.set_cache_limit(policy.cache_limit)
    if hasattr(mx, "set_wired_limit"):
        mx.set_wired_limit(0)
    return {"mlx_limit": policy.mlx_limit, "cache_limit": policy.cache_limit}


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _canonical_lock_root(root: Path) -> Path:
    try:
        common = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MemoryBudgetExceeded("Cannot resolve the canonical project lock") from exc
    if common.returncode == 0:
        return Path(common.stdout.strip()).resolve().parent
    if any((parent / ".git").exists() for parent in (root, *root.parents)):
        raise MemoryBudgetExceeded("Git failed to resolve the canonical project lock")
    return root


def run_guarded(
    command: Sequence[str],
    *,
    root: str | Path,
    policy: MemoryPolicy | None = None,
    report_path: str | Path | None = None,
    timeout_seconds: float = 300,
    configure_device: bool = True,
) -> MemoryReport:
    """Run Python -m module or Python script.py under one lock and watchdog."""
    import fcntl

    policy = policy or MemoryPolicy()
    root = Path(root).resolve()
    command = list(command)
    if not command or Path(command[0]).name not in {"python", "python3", Path(sys.executable).name}:
        raise ValueError("Guarded jobs must use the current Python executable")
    if len(command) < 3 and (len(command) != 2 or command[1].startswith("-")):
        raise ValueError("Expected python -m module or python script.py")
    if command[1].startswith("-") and command[1] != "-m":
        raise ValueError("Only -m module or script.py execution is supported")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    lock_root = _canonical_lock_root(root)
    temporary = lock_root / ".tmp"
    temporary.mkdir(exist_ok=True)
    with (temporary / "memory.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MemoryBudgetExceeded("Another guarded job owns the project memory lock") from exc
        snapshot = preflight(root, policy)
        report = MemoryReport(command, lowest_available=snapshot["available"])
        nonce = secrets.token_hex(32)
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({"pid": os.getpid(), "nonce": nonce, "policy": asdict(policy)}))
        lock.flush()
        env = os.environ.copy()
        env["FUNASR_MLX_MEMORY_POLICY"] = json.dumps(asdict(policy))
        env["FUNASR_MLX_GUARD_FD"] = str(lock.fileno())
        env["FUNASR_MLX_GUARD_LOCK"] = str(temporary / "memory.lock")
        env["FUNASR_MLX_GUARD_NONCE"] = nonce
        env["FUNASR_MLX_CONFIGURE_DEVICE"] = "1" if configure_device else "0"
        paths = [str(root / "src"), *env.get("PYTHONPATH", "").split(os.pathsep)]
        env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        psutil = _psutil()
        alive_read, alive_write = os.pipe()
        env["FUNASR_MLX_SUPERVISOR_FD"] = str(alive_read)
        started = time.monotonic()
        refresh_roots_at = started + 1.0
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "funasr_mlx.runtime", "_child", *command[1:]],
                cwd=root,
                env=env,
                start_new_session=True,
                pass_fds=(lock.fileno(), alive_read),
            )
        except BaseException:
            os.close(alive_read)
            os.close(alive_write)
            raise
        os.close(alive_read)
        try:
            while process.poll() is None:
                now = time.monotonic()
                if now >= refresh_roots_at:
                    _project_roots.cache_clear()
                    refresh_roots_at = now + 1.0
                rss, pids = process_tree_rss(process.pid)
                aggregate = project_rss(root, pids | {os.getpid()})
                available = psutil.virtual_memory().available
                report.peak_process_rss = max(report.peak_process_rss, rss)
                report.peak_project_rss = max(report.peak_project_rss, aggregate)
                report.lowest_available = min(report.lowest_available, available)
                elapsed = now - started
                if rss > policy.process_limit:
                    report.violation = "process RSS limit"
                elif aggregate > policy.project_limit:
                    report.violation = "project RSS limit"
                elif available < policy.min_available:
                    report.violation = "host available-memory reserve"
                elif elapsed > timeout_seconds:
                    report.violation = "job timeout"
                if report.violation:
                    _stop(process)
                    break
                time.sleep(policy.poll_seconds)
            report.returncode = process.wait()
        except BaseException:
            _stop(process)
            raise
        finally:
            os.close(alive_write)
            report.elapsed_seconds = time.monotonic() - started
            if report_path:
                target = Path(report_path)
                if not target.is_absolute():
                    target = root / target
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(asdict(report), indent=2) + "\n")
        if report.violation:
            raise MemoryBudgetExceeded(
                f"Stopped job: {report.violation}; peak RSS {report.peak_process_rss / MB:.1f} MB"
            )
        return report
