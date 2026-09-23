#!/usr/bin/env python3
"""Safely replace the fixed Transformer coordinator after GNN work drains.

The child trainers are deliberately left alive.  The updated coordinator
adopts them by output directory, then gives GNN-free GPUs extra Transformer
slots.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/zxy/project/TAPD/baselinemodels/pub0818/transfomer")
PYTHON = Path("/home/zxy/project/TAPD/.venv/bin/python")
POLL_SECONDS = 60


def processes_containing(needle: str) -> list[tuple[int, str]]:
    rows = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    result = []
    for row in rows:
        if needle not in row or "switch_to_dynamic_after_gnn.py" in row:
            continue
        pid_text, args = row.strip().split(None, 1)
        # Match an actual Python script argument, not a monitoring shell/awk
        # command whose quoted source merely mentions the filename.
        try:
            argv = shlex.split(args)
        except ValueError:
            continue
        if not argv or "python" not in Path(argv[0]).name.lower():
            continue
        if not any(Path(arg).name == needle for arg in argv[1:]):
            continue
        result.append((int(pid_text), args))
    return result


def main() -> None:
    zero_checks = 0
    print("dynamic-switch supervisor started", flush=True)
    while True:
        gnn_workers = processes_containing("gnn_validation_worker.py")
        zero_checks = zero_checks + 1 if not gnn_workers else 0
        print(f"gnn_workers={len(gnn_workers)} zero_checks={zero_checks}", flush=True)
        if zero_checks >= 2:
            break
        time.sleep(POLL_SECONDS)

    old = [(pid, args) for pid, args in processes_containing("transfomer/run_pipeline.py")]
    if len(old) != 1:
        raise RuntimeError(f"Expected exactly one fixed Transformer coordinator, found: {old}")
    old_pid, _ = old[0]
    print(f"stopping fixed coordinator pid={old_pid}; child trainers are retained", flush=True)
    os.kill(old_pid, signal.SIGTERM)
    for _ in range(30):
        if not any(pid == old_pid for pid, _ in processes_containing("transfomer/run_pipeline.py")):
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"Fixed coordinator {old_pid} did not stop")

    log_path = ROOT / "dynamic_pipeline.log"
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen([str(PYTHON), "-u", str(ROOT / "run_pipeline.py")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    print(f"dynamic coordinator started pid={proc.pid} log={log_path}", flush=True)


if __name__ == "__main__":
    main()
