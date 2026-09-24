#!/usr/bin/env python3
"""Run the five-seed MLP OFAT search on the fixed 80k/20k development split."""

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
from sklearn.model_selection import train_test_split


SEEDS = (202601, 202602, 202603, 202604, 202605)
FEATURES = ("SLE", "r_deg", "CV_arm", "n_leaves", "M_AB", "IR", "D_f", "ISC", "CAF", "tau_mean", "f_A")
REFERENCE = {"hidden_dim": 256, "depth": 4, "dropout": 0.1, "lr": 1e-2, "batch_size": 32, "weight_decay": 1e-5}
GRIDS = {
    "batch_size": (16, 32, 64, 128, 256),
    "lr": (1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2),
    "hidden_dim": (16, 32, 64, 128, 256),
    "depth": (1, 2, 3, 4, 5),
    "dropout": (0.0, 0.05, 0.1, 0.2, 0.3),
}


def build_manifest() -> list[dict]:
    tasks = []
    for dimension, values in GRIDS.items():
        for value in values:
            for seed in SEEDS:
                config = dict(REFERENCE)
                config[dimension] = value
                config.update({"dimension": dimension, "value_label": str(value), "seed": seed,
                               "experiment_id": f"hyper__{dimension}__{value}__seed{seed}"})
                tasks.append(config)
    return tasks


def prepare_split(path: Path) -> dict:
    train_index, val_index = train_test_split(np.arange(100_000), test_size=0.2, random_state=42)
    if path.exists():
        saved = np.load(path)
        if not (np.array_equal(saved["train_index"], train_index) and np.array_equal(saved["val_index"], val_index)):
            raise RuntimeError(f"Existing split at {path} does not match random_state=42.")
    else:
        np.savez(path, train_index=train_index, val_index=val_index)
    return {"random_state": 42, "train_size": len(train_index), "validation_size": len(val_index)}


def write_status(path: Path, pending: deque, active: list[dict], completed: list[str], failed: list[dict]) -> None:
    path.write_text(json.dumps({
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "total": len(pending) + len(active) + len(completed) + len(failed),
        "pending": len(pending), "completed": len(completed), "failed": failed,
        "active": [{"experiment_id": item["task"]["experiment_id"], "gpu": item["gpu"], "pid": item["process"].pid} for item in active],
    }, indent=2), encoding="utf-8")


def summarize(runs: Path, output: Path) -> None:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(runs.glob("*.json"))]
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["dimension"], row["value_label"]), []).append(row)
    summary = []
    for (dimension, value), items in groups.items():
        if len(items) != len(SEEDS):
            raise RuntimeError(f"Incomplete seed set for {dimension}={value}")
        values = [float(item["validation_mape_percent"]) for item in items]
        summary.append({"dimension": dimension, "value_label": value, "n_seeds": len(items),
                        "validation_mape_mean_percent": mean(values),
                        "validation_mape_sd_percent": stdev(values),
                        "validation_mape_per_seed_percent": json.dumps(values),
                        "validation_best_mse_mean": mean(float(item["best_val_mse_log10"]) for item in items)})
    summary.sort(key=lambda row: (row["dimension"], row["validation_mape_mean_percent"], row["validation_mape_sd_percent"]))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "hyperparameter_validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    selected = {dimension: min((row for row in summary if row["dimension"] == dimension),
                               key=lambda row: (row["validation_mape_mean_percent"], row["validation_mape_sd_percent"]))
                for dimension in GRIDS}
    (output / "selected_by_validation_mape.json").write_text(json.dumps({
        "selection_metric": "five-seed mean original-scale MAPE on the fixed 20k validation fold",
        "selected_values": {key: value["value_label"] for key, value in selected.items()},
        "details": selected,
        "final_test_status": "not accessed; evaluation is permitted only after all values are frozen",
    }, indent=2), encoding="utf-8")


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here / "runs_100k_fixed80_20_validation_only")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=here / "data" / "n_train_100k_descriptor_matrix.csv",
        help="Deterministic descriptor matrix for all 100,000 development samples.",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()
    if args.slots_per_gpu < 1:
        parser.error("--slots-per-gpu must be positive")
    runs, logs, checkpoints, predictions, summary_dir = (args.output_dir / name for name in ("runs", "logs", "checkpoints", "validation_predictions", "summary"))
    for directory in (runs, logs, checkpoints, predictions, summary_dir): directory.mkdir(parents=True, exist_ok=True)
    split = prepare_split(args.output_dir / "fixed_80k_20k_split_indices.npz")
    manifest = build_manifest()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "development_data": str(root / "data/datasets/development_100k"), "development_count": 100000,
        "train_csv": str(args.train_csv.resolve()), "split": split, "features": list(FEATURES), "seeds": list(SEEDS),
        "checkpoint_selection": "validation MSE in log10 chiN*", "hyperparameter_selection": "five-seed mean validation MAPE on original chiN* scale",
        "final_test": str(root / "data/datasets/validation_seed42"),
        "final_test_policy": "declared but never read or passed to this search; use only after selected values are frozen",
        "expected_runs": len(manifest), "slots_per_gpu": args.slots_per_gpu,
    }, indent=2), encoding="utf-8")
    pending = deque(task for task in manifest if not (runs / f"{task['experiment_id']}.json").exists())
    completed = [task["experiment_id"] for task in manifest if (runs / f"{task['experiment_id']}.json").exists()]
    slots = [int(gpu) for gpu in args.gpus.split(",") for _ in range(args.slots_per_gpu)]
    active: list[dict] = []; retries: dict[str, int] = {}; failed: list[dict] = []
    worker = root / "src/models/mlp_validation_worker.py"
    while pending or active:
        available = list(slots)
        for item in active: available.remove(item["gpu"])
        while pending and available:
            task, gpu = pending.popleft(), available.pop(0)
            eid = task["experiment_id"]
            command = [sys.executable, str(worker), "--train-csv", str(args.train_csv), "--split-indices", str(args.output_dir / "fixed_80k_20k_split_indices.npz"),
                       "--output", str(runs / f"{eid}.json"), "--checkpoint-output", str(checkpoints / f"{eid}.pt"),
                       "--validation-predictions-output", str(predictions / f"{eid}.npz"), "--experiment-id", eid,
                       "--dimension", task["dimension"], "--value-label", task["value_label"], "--features", ",".join(FEATURES),
                       "--seed", str(task["seed"]), "--hidden-dim", str(task["hidden_dim"]), "--depth", str(task["depth"]),
                       "--dropout", str(task["dropout"]), "--lr", str(task["lr"]), "--batch-size", str(task["batch_size"]),
                       "--weight-decay", str(task["weight_decay"]), "--epochs", str(args.epochs), "--patience", str(args.patience), "--device", "cuda:0"]
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            handle = (logs / f"{eid}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=root)
            active.append({"task": task, "gpu": gpu, "process": process, "handle": handle})
        time.sleep(2)
        remaining = []
        for item in active:
            code = item["process"].poll(); eid = item["task"]["experiment_id"]
            if code is None:
                remaining.append(item); continue
            item["handle"].close()
            if code == 0 and (runs / f"{eid}.json").exists(): completed.append(eid)
            else:
                retries[eid] = retries.get(eid, 0) + 1
                if retries[eid] <= args.max_retries: pending.append(item["task"])
                else: failed.append({"experiment_id": eid, "exit_code": code})
        active = remaining
        write_status(args.output_dir / "status.json", pending, active, completed, failed)
        print(f"completed={len(completed)}/{len(manifest)} active={len(active)} pending={len(pending)} failed={len(failed)}", flush=True)
    if failed: raise SystemExit(f"Failures: {failed}")
    summarize(runs, summary_dir)


if __name__ == "__main__":
    main()
