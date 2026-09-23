#!/usr/bin/env python3
"""Wait for the existing pub0818 MLP/GNN GPU studies, then run Transformer OFAT."""

import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/zxy/project/TAPD")
PYTHON = ROOT / ".venv/bin/python"
PIPELINE = ROOT / "baselinemodels/pub0818/transfomer/run_pipeline.py"
BLOCKERS = (
    "baselinemodels/pub0818/mlp/hyperparameter_selection_80k/run_hyperparameter_search.py",
    "baselinemodels/pub0818/mlp/hyperparameter_selection_80k/mlp_validation_worker.py",
    "baselinemodels/pub0818/gnn/hyperparameter_selection_80k/run_hyperparameter_search.py",
    "baselinemodels/pub0818/gnn/hyperparameter_selection_80k/gnn_validation_worker.py",
)


def blocking_processes():
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(blocker in command for blocker in BLOCKERS):
            found.append((int(entry.name), command))
    return found


def main():
    while True:
        blockers = blocking_processes()
        if not blockers:
            break
        summary = ", ".join(str(pid) for pid, _ in blockers)
        print(f"waiting for existing MLP/GNN GPU studies: pids={summary}", flush=True)
        time.sleep(60)
    print("GPU-study blockers cleared; launching complete Transformer pipeline", flush=True)
    subprocess.run(
        [str(PYTHON), str(PIPELINE)],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
