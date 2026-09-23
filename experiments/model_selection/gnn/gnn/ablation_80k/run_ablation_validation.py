#!/usr/bin/env python3
"""Run all frozen A/B/C GNN ablations on validation data only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
GNN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mainmodel.scripts import train_single_chin_supplementary_confirmation as source

SEEDS = (202601, 202602, 202603, 202604, 202605)
HYPER_ROOT = GNN_ROOT / "hyperparameter_selection_80k/runs_100k_fixed80_20_validation_only"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs_100k_fixed80_20_validation_only"


def frozen_hyperparameters() -> dict:
    selection_path = HYPER_ROOT / "summary/selected_by_validation_mape.json"
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    raw = payload["selected_values"]
    if set(raw) != {"batch_size", "learning_rate", "hidden_dim", "num_layers", "dec_layers"}:
        raise RuntimeError(f"Incomplete frozen hyperparameters: {raw}")
    return {
        "batch_size": int(raw["batch_size"]),
        "learning_rate": float(raw["learning_rate"]),
        "hidden_dim": int(raw["hidden_dim"]),
        "num_layers": int(raw["num_layers"]),
        "dec_layers": int(raw["dec_layers"]),
    }


def prepare_split(output_dir: Path) -> dict:
    source_path = HYPER_ROOT / "fixed_80k_20k_split_indices.npz"
    source_split = np.load(source_path)
    train = source_split["train_index"]
    val = source_split["val_index"]
    target = output_dir / "fixed_80k_20k_split_indices.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        old = np.load(target)
        if not (np.array_equal(old["train_index"], train) and np.array_equal(old["val_index"], val)):
            raise RuntimeError("Ablation split differs from frozen hyperparameter split.")
    else:
        np.savez(target, train_index=train, val_index=val)
    return {
        "random_state": 42,
        "train_size": int(len(train)),
        "validation_size": int(len(val)),
        "copied_from": str(source_path),
    }


def build_tasks(frozen: dict) -> list[dict]:
    tasks = []
    for condition_key in source.CONDITIONS:
        condition = source.condition_config(condition_key)
        for seed in SEEDS:
            run_id = f"{condition_key}__seed{seed}"
            cfg = dict(condition)
            cfg.update(frozen)
            cfg.update({
                "run_id": run_id,
                "seed": seed,
                "train_data_dir": str(ROOT / "TrainData/data/chiNlimited/n_train_100k"),
                "dataset_cache_dir": str(HYPER_ROOT / "dataset_cache"),
                "test_data_dir": "SEALED: not available to validation worker",
                "epochs": 200, "minimum_epochs": 120,
                "early_stopping_patience": 30,
                "val_fraction": 0.2,
                "split_random_state": 42,
                "optimizer": "Adam",
                "loss": "MSE on log10(chiN*)",
                "frozen_hyperparameters": True,
            })
            tasks.append(cfg)
    return tasks


def summarize(run_root: Path, summary_dir: Path) -> None:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(run_root.glob("*/metrics.json"))]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["condition_key"], []).append(row)
    summary = []
    for condition_key, items in grouped.items():
        if len(items) != len(SEEDS):
            continue
        scores = [float(item["validation_mape_percent"]) for item in items]
        first = items[0]
        summary.append({
            "condition_key": condition_key,
            "aliases": ";".join(first["aliases"]),
            "condition_label": first["condition_label"],
            "optional": bool(first["optional"]),
            "n_seeds": len(items),
            "validation_mape_mean_percent": mean(scores),
            "validation_mape_sd_percent": stdev(scores),
            "validation_mape_per_seed_percent": json.dumps(scores),
        })
    summary.sort(key=lambda row: row["condition_key"])
    summary_dir.mkdir(parents=True, exist_ok=True)
    columns = list(summary[0]) if summary else ["condition_key"]
    with (summary_dir / "ablation_validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)
    (summary_dir / "ablation_validation_summary.json").write_text(
        json.dumps({"selection_status": "hyperparameters frozen", "final_test_status": "sealed", "conditions": summary}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    frozen = frozen_hyperparameters()
    split = prepare_split(args.output_dir)
    configs = args.output_dir / "configs"
    runs = args.output_dir / "runs"
    logs = args.output_dir / "logs"
    for directory in (configs, runs, logs):
        directory.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(frozen)
    (args.output_dir / "manifest.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "phase": "validation-only A/B/C ablation after hyperparameter freeze",
        "development_data": str(ROOT / "TrainData/data/chiNlimited/n_train_100k"),
        "split": split,
        "seeds": SEEDS,
        "frozen_hyperparameters": frozen,
        "frozen_from": str(HYPER_ROOT / "summary/selected_by_validation_mape.json"),
        "development_cache_reused_from": str(HYPER_ROOT / "dataset_cache"),
        "condition_source": str(ROOT / "mainmodel/scripts/train_single_chin_supplementary_confirmation.py"),
        "final_test": str(ROOT / "TrainData/data/chiNlimited/n_test_2k_leakage_free_seed42"),
        "final_test_policy": "not read in this phase",
    }, indent=2), encoding="utf-8")
    for task in tasks:
        (configs / f"{task['run_id']}.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(f"Prepared {len(tasks)} runs across {len(source.CONDITIONS)} conditions.")
        return

    pending = deque(task for task in tasks if not (runs / task["run_id"] / "metrics.json").exists())
    active: list[dict] = []
    retries: dict[str, int] = {}
    slots = [gpu for gpu in args.gpus.split(",") for _ in range(args.slots_per_gpu)]
    worker = GNN_ROOT / "gnn_validation_worker.py"
    while pending or active:
        used = {gpu: sum(item["gpu"] == gpu for item in active) for gpu in set(slots)}
        free = []
        for gpu in slots:
            if used.get(gpu, 0):
                used[gpu] -= 1
            else:
                free.append(gpu)
        while pending and free:
            task = pending.popleft()
            gpu = free.pop(0)
            run_dir = runs / task["run_id"]
            run_dir.mkdir(exist_ok=True)
            log_path = logs / f"{task['run_id']}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            command = [
                sys.executable, str(worker),
                "--config-json", str(configs / f"{task['run_id']}.json"),
                "--split-indices", str(args.output_dir / "fixed_80k_20k_split_indices.npz"),
                "--output-dir", str(run_dir), "--device", "cuda:0",
            ]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            active.append({"task": task, "gpu": gpu, "process": process, "log": log_handle, "log_path": log_path})
        time.sleep(3)
        remaining = []
        for item in active:
            code = item["process"].poll()
            if code is None:
                remaining.append(item)
                continue
            item["log"].close()
            run_id = item["task"]["run_id"]
            if code and retries.get(run_id, 0) < args.max_retries:
                retries[run_id] = retries.get(run_id, 0) + 1
                pending.append(item["task"])
            elif code:
                raise SystemExit(f"Failed: {run_id}; see {item['log_path']}")
        active = remaining
        completed = len(tasks) - len(pending) - len(active)
        print(f"completed={completed}/{len(tasks)} active={len(active)}", flush=True)
    summarize(runs, args.output_dir / "summary")


if __name__ == "__main__":
    main()
