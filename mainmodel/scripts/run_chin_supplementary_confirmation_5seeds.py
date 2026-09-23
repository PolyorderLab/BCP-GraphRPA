#!/usr/bin/env python3
"""Orchestrate 5-seed supplementary chiN confirmation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("TAPD_PROJECT_ROOT", PACKAGE_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_SCRIPT = SCRIPT_DIR / "train_single_chin_supplementary_confirmation.py"
DEFAULT_BATCH_DIR = (
    PACKAGE_ROOT
    / "results"
    / "training_runs"
    / f"chiN_final_confirmation_5runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

sys.path.append(str(PROJECT_ROOT))
from mainmodel.scripts.train_single_chin_supplementary_confirmation import CONDITIONS, SEEDS, condition_config


def visible_gpus(jobs_per_gpu=1):
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_value:
        gpus = [gpu.strip() for gpu in env_value.split(",") if gpu.strip()]
    elif torch.cuda.is_available():
        gpus = [str(i) for i in range(torch.cuda.device_count())]
    else:
        gpus = [""]
    return [gpu for gpu in gpus for _ in range(jobs_per_gpu)]


def unique_conditions():
    return list(CONDITIONS.keys())


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plan_files(batch_dir: Path):
    rows = []
    alias_rows = []
    for condition_key in unique_conditions():
        cfg = condition_config(condition_key)
        rows.append(
            {
                "condition_key": condition_key,
                "aliases": ";".join(cfg["aliases"]),
                "groups": ";".join(cfg["group_labels"]),
                "optional": bool(cfg["optional"]),
                "condition_label": cfg["condition_label"],
                "graph": cfg["graph_label"],
                "node_features": cfg["node_features_label"],
                "edge_features": cfg["edge_features_label"],
                "hidden_dim": cfg["hidden_dim"],
                "learning_rate": cfg["learning_rate"],
                "batch_size": cfg["batch_size"],
                "num_layers": cfg["num_layers"],
                "dec_layers": cfg["dec_layers"],
                "epochs": cfg["epochs"],
                "train_data_dir": cfg["train_data_dir"],
                "test_data_dir": cfg["test_data_dir"],
            }
        )
        for alias in cfg["aliases"]:
            alias_rows.append(
                {
                    "alias": alias,
                    "condition_key": condition_key,
                    "condition_label": cfg["condition_label"],
                    "groups": ";".join(cfg["group_labels"]),
                    "optional": bool(cfg["optional"]),
                }
            )
    write_csv(batch_dir / "planned_conditions.csv", rows)
    write_json(batch_dir / "planned_conditions.json", rows)
    write_csv(batch_dir / "alias_map.csv", alias_rows)
    write_json(batch_dir / "alias_map.json", alias_rows)


def metrics_path_for(batch_dir: Path, condition_key: str, run_id: int, seed: int) -> Path:
    run_dir = (
        batch_dir
        / "runs"
        / condition_key
        / f"{condition_key}_run{run_id:02d}_seed{seed}"
    )
    return run_dir / "metrics.json"


def run_dir_for(batch_dir: Path, condition_key: str, run_id: int, seed: int) -> Path:
    return metrics_path_for(batch_dir, condition_key, run_id, seed).parent


def task_is_running(condition_key: str, run_id: int, seed: int) -> bool:
    cmd = [
        "ps",
        "-eo",
        "args",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    needle = (
        f"train_single_chin_supplementary_confirmation.py --condition-key {condition_key} "
        f"--batch-dir "
    )
    run_needle = f"--seed {seed} --run-id {run_id}"
    for line in result.stdout.splitlines():
        if needle in line and run_needle in line:
            return True
    return False


def run_orchestrator(batch_dir: Path, jobs_per_gpu: int):
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "logs").mkdir(exist_ok=True)
    write_plan_files(batch_dir)

    tasks = []
    for condition_key in unique_conditions():
        for run_id, seed in enumerate(SEEDS, start=1):
            if metrics_path_for(batch_dir, condition_key, run_id, seed).exists():
                continue
            if task_is_running(condition_key, run_id, seed):
                continue
            tasks.append((condition_key, run_id, seed))

    free_gpus = visible_gpus(jobs_per_gpu=jobs_per_gpu)
    running = []
    task_index = 0

    print(f"Batch dir: {batch_dir}")
    print(f"Single-run script: {SINGLE_SCRIPT}")
    print(f"Unique conditions: {len(unique_conditions())}")
    print(f"Pending runs: {len(tasks)}")
    print(f"Jobs per GPU: {jobs_per_gpu}")
    print(f"GPU slots: {free_gpus if free_gpus != [''] else ['CPU']}")

    if not tasks:
        aggregate(batch_dir)
        print(f"No pending runs. Summary written to {batch_dir}")
        return

    while task_index < len(tasks) or running:
        while task_index < len(tasks) and free_gpus:
            condition_key, run_id, seed = tasks[task_index]
            gpu = free_gpus.pop(0)
            run_dir = run_dir_for(batch_dir, condition_key, run_id, seed)
            if run_dir.exists():
                shutil.rmtree(run_dir)
            log_path = batch_dir / "logs" / f"{condition_key}_run{run_id:02d}_seed{seed}.log"
            env = os.environ.copy()
            if gpu:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            cmd = [
                sys.executable,
                str(SINGLE_SCRIPT),
                "--condition-key",
                condition_key,
                "--batch-dir",
                str(batch_dir),
                "--seed",
                str(seed),
                "--run-id",
                str(run_id),
            ]
            log_file = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
            running.append((proc, log_file, log_path, condition_key, run_id, seed, gpu))
            print(f"Started {condition_key} run {run_id} seed {seed} on {gpu or 'CPU'}")
            task_index += 1

        if not running:
            continue

        while True:
            completed_index = None
            for idx, (proc, *_rest) in enumerate(running):
                if proc.poll() is not None:
                    completed_index = idx
                    break
            if completed_index is not None:
                proc, log_file, log_path, condition_key, run_id, seed, gpu = running.pop(completed_index)
                return_code = proc.returncode
                log_file.close()
                free_gpus.append(gpu)
                if return_code != 0:
                    raise RuntimeError(f"{condition_key} run {run_id} seed {seed} failed; see {log_path}")
                print(f"Finished {condition_key} run {run_id} seed {seed} on {gpu or 'CPU'}")
                break
            time.sleep(5)

    aggregate(batch_dir)
    print(f"Done. Summary written to {batch_dir}")


def load_rows(batch_dir: Path):
    rows = []
    runs_root = batch_dir / "runs"
    if not runs_root.is_dir():
        return rows
    for condition_dir in sorted(runs_root.iterdir()):
        if not condition_dir.is_dir():
            continue
        for run_dir in sorted(condition_dir.iterdir()):
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    rows.append(json.load(f))
    return sorted(rows, key=lambda row: (row["condition_key"], row["run_id"], row["seed"]))


def summarize_rows(rows):
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition_key"], []).append(row)

    summary_rows = []
    for condition_key, items in sorted(by_condition.items()):
        cfg = condition_config(condition_key)
        summary = {
            "condition_key": condition_key,
            "aliases": ";".join(cfg["aliases"]),
            "groups": ";".join(cfg["group_labels"]),
            "optional": bool(cfg["optional"]),
            "condition_label": cfg["condition_label"],
            "graph": cfg["graph_label"],
            "node_features": cfg["node_features_label"],
            "edge_features": cfg["edge_features_label"],
            "n": len(items),
            "seeds": ";".join(str(item["seed"]) for item in items),
        }
        for metric in ("MAPE", "RMSE", "MAE", "R2", "MaxAE"):
            values = np.array([float(item[metric]) for item in items], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=0))
            summary[f"{metric}_min"] = float(values.min())
            summary[f"{metric}_max"] = float(values.max())
        summary_rows.append(summary)
    return summary_rows


def write_group_summaries(batch_dir: Path, summary_rows):
    grouped = {}
    for row in summary_rows:
        for group in str(row["groups"]).split(";"):
            grouped.setdefault(group, []).append(row)
    for group, rows in grouped.items():
        stem = group.lower().replace(":", "").replace(" ", "_")
        write_csv(batch_dir / f"{stem}_summary.csv", rows)


def aggregate(batch_dir: Path):
    rows = load_rows(batch_dir)
    if not rows:
        raise RuntimeError(f"No metrics.json files found under {batch_dir}")

    metrics_fieldnames = [
        "condition_key",
        "aliases",
        "condition_label",
        "group_labels",
        "optional",
        "run_id",
        "seed",
        "MAPE",
        "RMSE",
        "MAE",
        "R2",
        "MaxAE",
        "result_dir",
    ]
    write_csv(batch_dir / "supplementary_confirmation_5runs_metrics.csv", rows, fieldnames=metrics_fieldnames)

    summary_rows = summarize_rows(rows)
    write_csv(batch_dir / "supplementary_confirmation_5runs_summary.csv", summary_rows)
    write_json(batch_dir / "supplementary_confirmation_5runs_summary.json", summary_rows)
    write_group_summaries(batch_dir, summary_rows)

    with open(batch_dir / "supplementary_confirmation_5runs_stats.txt", "w", encoding="utf-8") as f:
        f.write("chiN supplementary confirmation experiments\n")
        f.write("=" * 72 + "\n")
        for row in summary_rows:
            f.write(f"{row['condition_key']} ({row['aliases']})\n")
            f.write(f"  condition_label: {row['condition_label']}\n")
            f.write(f"  groups: {row['groups']}\n")
            f.write(f"  graph: {row['graph']}\n")
            f.write(f"  node_features: {row['node_features']}\n")
            f.write(f"  edge_features: {row['edge_features']}\n")
            f.write(
                f"  MAPE={row['MAPE_mean']:.4f} +/- {row['MAPE_std']:.4f}, "
                f"RMSE={row['RMSE_mean']:.4f} +/- {row['RMSE_std']:.4f}, "
                f"MAE={row['MAE_mean']:.4f} +/- {row['MAE_std']:.4f}, "
                f"R2={row['R2_mean']:.4f} +/- {row['R2_std']:.4f}, "
                f"MaxAE={row['MaxAE_mean']:.4f} +/- {row['MaxAE_std']:.4f}\n"
            )
            f.write("-" * 72 + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.jobs_per_gpu < 1:
        raise ValueError("--jobs-per-gpu must be >= 1")
    run_orchestrator(Path(args.batch_dir), jobs_per_gpu=args.jobs_per_gpu)


if __name__ == "__main__":
    main()
