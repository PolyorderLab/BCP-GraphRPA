#!/usr/bin/env python3
"""Corrected copy of the MLP nested training-size runner and 2k evaluator."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from stratified_nested_common import block_labels, build_stratified_nested_indices, load_sample_ids, write_indices

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MLP_ROOT = ROOT / "baselinemodels/pub0818/mlp"
SEARCH_ROOT = MLP_ROOT / "hyperparameter_selection_80k"
TRAIN_CSV = SEARCH_ROOT / "data/n_train_100k_descriptor_matrix.csv"
SPLIT_FILE = SEARCH_ROOT / "runs_100k_fixed80_20_validation_only/fixed_80k_20k_split_indices.npz"
WORKER = SEARCH_ROOT / "mlp_validation_worker.py"
OUTPUT = HERE / "results/mlp"
SEEDS = (202601, 202602, 202603, 202604, 202605)
SIZES = tuple(range(8_000, 80_001, 8_000))
FEATURES = "SLE,r_deg,CV_arm,n_leaves,M_AB,IR,D_f,ISC,CAF,tau_mean,f_A"
CONFIG = {"batch_size": 256, "lr": 0.0005, "hidden_dim": 256, "depth": 3, "dropout": 0.0, "weight_decay": 1e-5, "epochs": 200, "patience": 30}


def prepare() -> list[dict]:
    split = np.load(SPLIT_FILE)
    pool = np.asarray(split["train_index"], dtype=np.int64)
    labels = block_labels(load_sample_ids(TRAIN_CSV))
    subsets, audit = build_stratified_nested_indices(pool, labels, SIZES)
    write_indices(OUTPUT / "nested_indices", subsets, audit)
    tasks = [{
        "id": f"nested_train__{size // 1000}k__seed{seed}",
        "train_size": size,
        "seed": seed,
        "train_index_file": str(OUTPUT / "nested_indices" / f"nested_train_{size // 1000}k.npz"),
    } for size in SIZES for seed in SEEDS]
    tasks.sort(key=lambda item: (-int(item["train_size"]), int(item["seed"])))
    (OUTPUT / "manifest.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    protocol = {
        "runner": str(Path(__file__).resolve()),
        "source_runner": str(MLP_ROOT / "training_size_nested_80k_common_val_seed42/run_nested_training_size.py"),
        "nested_subset_rule": "within each 2/3/4/5-block stratum: one random_state=42 permutation; monotone largest-remainder quotas proportional to the frozen 80k pool",
        "train_sizes": list(SIZES), "seeds": list(SEEDS), "features": FEATURES.split(","),
        "configuration": CONFIG, "fixed_split": str(SPLIT_FILE),
        "stratification_audit": str(OUTPUT / "nested_indices/stratification_audit.json"),
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return tasks


def run(tasks: list[dict], gpus: str, slots_per_gpu: int) -> None:
    for name in ("runs", "checkpoints", "predictions", "logs"):
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)
    pending = deque(task for task in tasks if not (OUTPUT / "runs" / f"{task['id']}.json").exists())
    slots = [gpu.strip() for gpu in gpus.split(",") for _ in range(slots_per_gpu)]
    active, failed = [], []
    while pending or active:
        available = list(slots)
        for item in active:
            if item["gpu"] in available:
                available.remove(item["gpu"])
        while pending and available:
            task, gpu = pending.popleft(), available.pop(0)
            log = (OUTPUT / "logs" / f"{task['id']}.log").open("a", encoding="utf-8")
            command = [sys.executable, str(WORKER), "--train-csv", str(TRAIN_CSV), "--split-indices", str(SPLIT_FILE),
                "--train-index-file", task["train_index_file"], "--output", str(OUTPUT / "runs" / f"{task['id']}.json"),
                "--checkpoint-output", str(OUTPUT / "checkpoints" / f"{task['id']}.pt"),
                "--validation-predictions-output", str(OUTPUT / "predictions" / f"{task['id']}.npz"),
                "--experiment-id", task["id"], "--dimension", "block_stratified_nested_training_size",
                "--value-label", f"{task['train_size'] // 1000}k", "--features", FEATURES, "--seed", str(task["seed"]),
                "--hidden-dim", str(CONFIG["hidden_dim"]), "--depth", str(CONFIG["depth"]), "--dropout", str(CONFIG["dropout"]),
                "--lr", str(CONFIG["lr"]), "--batch-size", str(CONFIG["batch_size"]), "--weight-decay", str(CONFIG["weight_decay"]),
                "--epochs", str(CONFIG["epochs"]), "--patience", str(CONFIG["patience"]), "--device", "cuda:0"]
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append({"task": task, "gpu": gpu, "proc": proc, "log": log})
            print(f"started gpu={gpu} {task['id']}", flush=True)
        time.sleep(3)
        remaining = []
        for item in active:
            code = item["proc"].poll()
            if code is None:
                remaining.append(item); continue
            item["log"].close()
            if code != 0 or not (OUTPUT / "runs" / f"{item['task']['id']}.json").exists():
                failed.append({"id": item["task"]["id"], "exit_code": code})
            else:
                print(f"finished {item['task']['id']}", flush=True)
        active = remaining
        status = {"total": len(tasks), "completed": len(tasks) - len(pending) - len(active) - len(failed),
                  "pending": len(pending), "active": len(active), "failed": failed, "updated_at": time.strftime("%F %T")}
        (OUTPUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status), flush=True)
    if failed:
        raise RuntimeError(f"MLP failures: {failed}")


def evaluate_common_2k() -> None:
    import pandas as pd
    import torch
    sys.path.insert(0, str(SEARCH_ROOT))
    from mlp_validation_worker import MLP
    source = MLP_ROOT / "final_test_evaluation/test_descriptor_matrix.csv"
    test = pd.read_csv(source)
    rows = []
    pred_dir = OUTPUT / "development_evaluation_seed42/predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for size in SIZES:
        for seed in SEEDS:
            experiment_id = f"nested_train__{size // 1000}k__seed{seed}"
            payload = torch.load(OUTPUT / "checkpoints" / f"{experiment_id}.pt", map_location="cpu", weights_only=False)
            features = list(payload["features"])
            x = test[features].to_numpy(np.float32)
            x = (x - np.asarray(payload["scaler_mean"], np.float32)) / np.asarray(payload["scaler_scale"], np.float32)
            true_log = test["log10_chiN_star"].to_numpy(np.float32)
            model = MLP(len(features), int(payload["hidden_dim"]), int(payload["depth"]), float(payload["dropout"]))
            model.load_state_dict(payload["model_state_dict"]); model.to(device).eval()
            with torch.no_grad(): pred_log = model(torch.from_numpy(x).to(device)).cpu().numpy().ravel()
            true, pred = 10.0 ** true_log, 10.0 ** pred_log
            mape = float(np.mean(np.abs(pred - true) / np.clip(np.abs(true), 1e-12, None)) * 100.0)
            rows.append({"experiment_id": experiment_id, "train_size_k": size // 1000, "seed": seed,
                         "evaluation_size": len(test), "evaluation_mape_percent": mape})
            np.savez_compressed(pred_dir / f"{experiment_id}.npz", sample_id=test["sample_id"].astype(str).to_numpy(),
                                y_true_log=true_log, y_pred_log=pred_log)
    out = OUTPUT / "development_evaluation_seed42"; out.mkdir(exist_ok=True)
    with (out / "nested_training_size_seed42_per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    for size in (value // 1000 for value in SIZES):
        values = np.asarray([row["evaluation_mape_percent"] for row in rows if row["train_size_k"] == size])
        summary.append({"train_size_k": size, "n_seeds": len(values), "evaluation_mape_mean_percent": values.mean(),
                        "evaluation_mape_sd_percent": values.std(ddof=1)})
    with (out / "nested_training_size_seed42_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(); tasks = prepare()
    if args.prepare_only:
        print(json.dumps({"prepared": len(tasks), "output": str(OUTPUT)}, indent=2)); return
    run(tasks, args.gpus, args.slots_per_gpu); evaluate_common_2k()


if __name__ == "__main__":
    main()
