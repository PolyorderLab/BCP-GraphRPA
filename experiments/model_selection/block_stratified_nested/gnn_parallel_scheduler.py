from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


def run(
    records: list[dict],
    output: Path,
    gpus: str,
    slots_per_gpu: int,
    retries: int,
    *,
    worker: Path,
    cwd: Path,
) -> None:
    pending = deque(
        cfg for cfg in records
        if not (output / "runs" / cfg["run_id"] / "metrics.json").is_file()
    )
    slots = [gpu.strip() for gpu in gpus.split(",") for _ in range(slots_per_gpu)]
    active: list[dict] = []
    attempts: dict[str, int] = {}
    while pending or active:
        free = list(slots)
        for item in active:
            free.remove(item["gpu"])
        while pending and free:
            cfg, gpu = pending.popleft(), free.pop(0)
            run_dir = output / "runs" / cfg["run_id"]
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = output / "logs" / f"{cfg['run_id']}.log"
            log = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [
                sys.executable,
                str(worker),
                "--config-json", str(output / "configs" / f"{cfg['run_id']}.json"),
                "--train-indices", cfg["train_indices_path"],
                "--output-dir", str(run_dir),
                "--device", "cuda:0",
            ]
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active.append({
                "cfg": cfg,
                "gpu": gpu,
                "proc": process,
                "log": log,
                "log_path": log_path,
            })
            print(f"started gpu={gpu} {cfg['run_id']}", flush=True)
        time.sleep(5)
        remaining = []
        for item in active:
            code = item["proc"].poll()
            if code is None:
                remaining.append(item)
                continue
            item["log"].close()
            cfg = item["cfg"]
            metric = output / "runs" / cfg["run_id"] / "metrics.json"
            if code == 0 and metric.is_file():
                print(f"finished {cfg['run_id']}", flush=True)
            elif attempts.get(cfg["run_id"], 0) < retries:
                attempts[cfg["run_id"]] = attempts.get(cfg["run_id"], 0) + 1
                pending.append(cfg)
                print(f"retrying {cfg['run_id']}", flush=True)
            else:
                raise RuntimeError(f"Failed: {cfg['run_id']}; see {item['log_path']}")
        active = remaining
        completed = len(records) - len(pending) - len(active)
        print(f"completed={completed}/{len(records)} active={len(active)}", flush=True)
