"""Guarded Python launcher; the package root remains dependency-lazy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import threading

from .memory import MemoryBudgetExceeded, MemoryPolicy, guarded_child_active, run_guarded


def _watch_supervisor() -> None:
    descriptor = int(os.environ["FUNASR_MLX_SUPERVISOR_FD"])
    if os.getpgrp() != os.getpid():
        raise RuntimeError("Guarded child must own its process group")

    def stop_on_parent_exit():
        try:
            while os.read(descriptor, 1):
                pass
        finally:
            os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=stop_on_parent_exit, daemon=True, name="memory-parent-watch").start()


def _child(arguments: list[str]) -> None:
    if not guarded_child_active():
        raise RuntimeError("Invalid or missing memory supervisor")
    _watch_supervisor()
    policy = MemoryPolicy(**json.loads(os.environ["FUNASR_MLX_MEMORY_POLICY"]))
    if os.environ.get("FUNASR_MLX_CONFIGURE_DEVICE") == "1":
        from .memory import configure_mlx

        configure_mlx(policy)
    if arguments[0] == "-m":
        if len(arguments) < 2:
            raise ValueError("Missing Python module")
        sys.argv = arguments[1:]
        runpy.run_module(arguments[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = arguments
        runpy.run_path(arguments[0], run_name="__main__")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "_child":
        _child(arguments[1:])
        return 0
    parser = argparse.ArgumentParser(description="Serialized memory-bounded Python job")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, default=Path(".tmp/memory-report.json"))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    try:
        report = run_guarded(
            command,
            root=args.root,
            report_path=args.report,
            timeout_seconds=args.timeout,
            configure_device=not args.no_device,
        )
    except (MemoryBudgetExceeded, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 125
    print(json.dumps(report.__dict__, indent=2))
    return report.returncode


if __name__ == "__main__":
    raise SystemExit(main())
