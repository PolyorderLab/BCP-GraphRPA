#!/usr/bin/env python3
"""Prepare, run, and aggregate the frozen 100k validation-only study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import time
from copy import deepcopy
from pathlib import Path


ROOT = Path("/home/zxy/project/TAPD")
CODE_ROOT = ROOT / "baselinemodels/pub0818/transfomer"
# The default preserves the published 100k study.  Dedicated reruns can set
# these paths through the environment so their manifests, checkpoints, and
# summaries remain completely separate.
EXPERIMENT_ROOT = Path(os.environ.get("TRANSFORMER_EXPERIMENT_ROOT", str(CODE_ROOT)))
PYTHON = ROOT / ".venv/bin/python"
TRAINER = CODE_ROOT / "train_validation_only.py"
DATA_DIR = Path(os.environ.get(
    "TRANSFORMER_DATA_DIR", str(ROOT / "TrainData/data/chiNlimited/n_train_100k")
))
RECORDS_CACHE = Path(os.environ.get(
    "TRANSFORMER_RECORDS_CACHE", str(CODE_ROOT / "data_cache/n_train_100k_records.pkl")
))
FINAL_TEST_DIR = Path(os.environ.get(
    "TRANSFORMER_FINAL_TEST_DIR", str(ROOT / "TrainData/data/chiNlimited/n_test_2k_leakage_free_seed42")
))
SEEDS = [202601, 202602, 202603, 202604, 202605]

ANCHOR = {
    "batch_size": 40,
    "lr": 2e-4,
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 2,
    "ff_mult": 4,
    "dropout": 0.10,
    "readout": "cls",
    "position_encoding": "learnable",
    "weight_decay": 1e-5,
    "representation": "full_topology",
    "architecture": "transformer",
}

FAMILIES = {
    "batch_size": [16, 32, 40, 64, 128],
    "lr": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    "model_size": [(64, 2), (128, 4), (192, 6), (256, 8)],
    "num_layers": [1, 2, 3, 4],
    "ff_mult": [2, 4, 6],
    "dropout": [0.0, 0.05, 0.10, 0.20],
    "readout": ["cls", "mean", "cls_mean", "attention"],
    "position_encoding": ["learnable", "sinusoidal", "none"],
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
}


def slug(value):
    if isinstance(value, tuple):
        return f"d{value[0]}h{value[1]}"
    if isinstance(value, float):
        return f"{value:.0e}".replace("+", "").replace("-0", "-")
    return str(value).replace("_", "-")


def anchor_value(family):
    return (ANCHOR["d_model"], ANCHOR["num_heads"]) if family == "model_size" else ANCHOR[family]


def apply_value(config, family, value):
    result = deepcopy(config)
    if family == "model_size":
        result["d_model"], result["num_heads"] = value
    else:
        result[family] = value
    return result


def build_ofat_configs():
    configs = [{
        "config_id": "anchor",
        "phase": "ofat",
        "families": list(FAMILIES),
        "varied_family": "anchor",
        "varied_value": "anchor",
        **deepcopy(ANCHOR),
    }]
    for family, values in FAMILIES.items():
        for value in values:
            if value == anchor_value(family):
                continue
            configs.append({
                "config_id": f"{family}__{slug(value)}",
                "phase": "ofat",
                "families": [family],
                "varied_family": family,
                "varied_value": list(value) if isinstance(value, tuple) else value,
                **apply_value(ANCHOR, family, value),
            })
    if len(configs) != 28:
        raise AssertionError(f"Expected 28 OFAT configs, found {len(configs)}")
    return configs


def manifest_path(phase):
    return EXPERIMENT_ROOT / phase / "experiment_manifest.json"


def write_manifest(phase, configs):
    phase_root = EXPERIMENT_ROOT / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "development_directory": str(DATA_DIR),
        "fixed_split": {"train_fraction": 0.8, "validation_fraction": 0.2, "random_state": 42},
        "final_test_directory": str(FINAL_TEST_DIR),
        "final_test_access_during_phase": "forbidden",
        "seeds": SEEDS,
        "primary_selection_metric": "five-seed mean validation_MAPE",
        "checkpoint_metric": "validation log-MSE",
        "configs": configs,
    }
    path = manifest_path(phase)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_configs(phase):
    return json.loads(manifest_path(phase).read_text(encoding="utf-8"))["configs"]


def command_for(config, seed, output_dir, smoke=False):
    command = [
        str(PYTHON), str(TRAINER),
        "--data-dir", str(DATA_DIR),
        "--output-dir", str(output_dir),
        "--run-name", f"{config['config_id']}__seed{seed}",
        "--seed", str(seed),
        "--device", "cuda:0" if not smoke else "cpu",
        "--epochs", "120" if not smoke else "2",
        "--early-stopping-patience", "25",
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--weight-decay", str(config["weight_decay"]),
        "--d-model", str(config["d_model"]),
        "--num-heads", str(config["num_heads"]),
        "--num-layers", str(config["num_layers"]),
        "--ff-mult", str(config["ff_mult"]),
        "--dropout", str(config["dropout"]),
        "--readout", str(config["readout"]),
        "--position-encoding", str(config["position_encoding"]),
        "--representation", str(config.get("representation", "full_topology")),
        "--architecture", str(config.get("architecture", "transformer")),
    ]
    if smoke:
        command.extend(["--max-samples", "200"])
    else:
        command.extend(["--records-cache", str(RECORDS_CACHE)])
    return command


def _gpu_index_by_pid():
    """Return the physical GPU index for visible CUDA process IDs, if available."""
    try:
        gpu_rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
        ).splitlines()
        uuid_to_index = {row.split(", ")[1]: int(row.split(", ")[0]) for row in gpu_rows}
        app_rows = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        return {int(row.split(", ")[1]): uuid_to_index[row.split(", ")[0]] for row in app_rows}
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError, IndexError):
        return {}


def _gnn_gpu_counts():
    """Count active GNN validation workers on each physical GPU."""
    try:
        ps_rows = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    gnn_pids = {int(row.strip().split(None, 1)[0]) for row in ps_rows if "gnn_validation_worker.py" in row}
    pid_to_gpu = _gpu_index_by_pid()
    if gnn_pids and not pid_to_gpu:
        # CUDA visibility could not be inspected: keep the conservative
        # one-worker-per-GPU policy rather than accidentally over-subscribing.
        return None
    counts = [0, 0, 0, 0]
    for pid, gpu in pid_to_gpu.items():
        if pid in gnn_pids and 0 <= gpu < len(counts):
            counts[gpu] += 1
    return counts


def _adopt_active_workers(runs_root, waiting, logs_root):
    """Adopt Transformer children left running by a replaced coordinator.

    Each adopted child is matched by its --output-dir.  This permits a new
    coordinator to resume safely without launching a duplicate run.
    """
    try:
        ps_rows = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []
    pending = {str(output_dir.resolve()): (config, seed, output_dir) for config, seed, output_dir in waiting}
    pid_to_gpu = _gpu_index_by_pid()
    adopted = []
    for row in ps_rows:
        if "train_validation_only.py" not in row or "--output-dir" not in row:
            continue
        pid_text, args = row.strip().split(None, 1)
        match = re.search(r"--output-dir\s+(\S+)", args)
        if not match:
            continue
        output_key = str(Path(match.group(1)).resolve())
        task = pending.pop(output_key, None)
        if task is None:
            continue
        config, seed, output_dir = task
        run_name = f"{config['config_id']}__seed{seed}"
        adopted.append({
            "process": None, "pid": int(pid_text), "gpu": pid_to_gpu.get(int(pid_text), 0),
            "config": config, "seed": seed, "output_dir": output_dir,
            "log_path": logs_root / f"{run_name}.log", "handle": None, "started": time.time(),
        })
    waiting[:] = list(pending.values())
    return adopted


def _adopted_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def run_phase(phase, max_per_gpu, idle_gpu_slots=None):
    configs = load_configs(phase)
    runs_root = EXPERIMENT_ROOT / phase / "runs"
    logs_root = EXPERIMENT_ROOT / phase / "logs"
    failures_root = EXPERIMENT_ROOT / phase / "failures"
    for path in (runs_root, logs_root, failures_root):
        path.mkdir(parents=True, exist_ok=True)
    waiting = []
    for config in configs:
        for seed in SEEDS:
            output_dir = runs_root / config["config_id"] / f"seed{seed}"
            if (output_dir / "metrics.json").exists():
                continue
            waiting.append((config, seed, output_dir))

    active = _adopt_active_workers(runs_root, waiting, logs_root)
    gpu_counts = [0, 0, 0, 0]
    for item in active:
        if 0 <= item["gpu"] < len(gpu_counts):
            gpu_counts[item["gpu"]] += 1
    if active:
        print(f"adopted {len(active)} active Transformer worker(s)", flush=True)
    while waiting or active:
        gnn_counts = _gnn_gpu_counts()
        slot_limits = (
            [max_per_gpu] * 4 if gnn_counts is None
            else [max_per_gpu if count else (idle_gpu_slots or max_per_gpu) for count in gnn_counts]
        )
        for gpu in range(4):
            while waiting and gpu_counts[gpu] < slot_limits[gpu]:
                config, seed, output_dir = waiting.pop(0)
                output_dir.mkdir(parents=True, exist_ok=True)
                run_name = f"{config['config_id']}__seed{seed}"
                log_path = logs_root / f"{run_name}.log"
                handle = log_path.open("w", encoding="utf-8")
                env = os.environ.copy()
                env.update({
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "MPLCONFIGDIR": "/tmp/matplotlib",
                })
                process = subprocess.Popen(
                    command_for(config, seed, output_dir),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                active.append({
                    "process": process, "pid": process.pid, "gpu": gpu, "config": config, "seed": seed,
                    "output_dir": output_dir, "log_path": log_path, "handle": handle, "started": time.time(),
                })
                gpu_counts[gpu] += 1
                print(f"launch {run_name} gpu={gpu} pid={process.pid}", flush=True)
        if active:
            time.sleep(10)
        remaining = []
        for item in active:
            process, pid, gpu = item["process"], item["pid"], item["gpu"]
            config, seed, output_dir = item["config"], item["seed"], item["output_dir"]
            log_path, handle, started = item["log_path"], item["handle"], item["started"]
            if process is None:
                code = None if _adopted_running(pid) else (0 if (output_dir / "metrics.json").exists() else 1)
            else:
                code = process.poll()
            if code is None:
                remaining.append(item)
                continue
            if handle is not None:
                handle.close()
            gpu_counts[gpu] -= 1
            run_name = f"{config['config_id']}__seed{seed}"
            if code != 0:
                failure = {
                    "run_name": run_name,
                    "return_code": code,
                    "log": str(log_path),
                    "elapsed_seconds": time.time() - started,
                }
                (failures_root / f"{run_name}.json").write_text(
                    json.dumps(failure, indent=2), encoding="utf-8"
                )
                for other in remaining:
                    if other["process"] is not None:
                        other["process"].terminate()
                raise RuntimeError(f"Run failed: {run_name}; inspect {log_path}")
            print(f"complete {run_name} elapsed={time.time() - started:.1f}s", flush=True)
        active = remaining


def aggregate(phase):
    configs = load_configs(phase)
    rows = []
    missing = []
    for config in configs:
        for seed in SEEDS:
            run_dir = EXPERIMENT_ROOT / phase / "runs" / config["config_id"] / f"seed{seed}"
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                missing.append(str(metrics_path))
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("test_loaded") is not False:
                raise RuntimeError(f"Test-isolation audit failed: {metrics_path}")
            rows.append({
                "phase": phase,
                "config_id": config["config_id"],
                "seed": seed,
                "validation_MAPE": metrics["validation_MAPE"],
                "validation_RMSE": metrics["validation_RMSE"],
                "validation_MAE": metrics["validation_MAE"],
                "validation_R2": metrics["validation_R2"],
                "best_val_loss": metrics["best_val_loss"],
                "best_epoch": metrics["best_epoch"],
                "trainable_parameters": metrics["trainable_parameters"],
                "run_dir": str(run_dir),
            })
    if missing:
        raise RuntimeError(f"Cannot aggregate incomplete phase; first missing: {missing[0]}")
    summary = []
    for config in configs:
        config_rows = [row for row in rows if row["config_id"] == config["config_id"]]
        mapes = [float(row["validation_MAPE"]) for row in config_rows]
        summary.append({
            "phase": phase,
            "config_id": config["config_id"],
            "n": len(mapes),
            "validation_MAPE_mean": statistics.mean(mapes),
            "validation_MAPE_sd": statistics.stdev(mapes),
            "validation_MAPE_sem": statistics.stdev(mapes) / math.sqrt(len(mapes)),
            "trainable_parameters": int(config_rows[0]["trainable_parameters"]),
            "config_json": json.dumps(config, sort_keys=True),
        })
    summary.sort(key=lambda row: (row["validation_MAPE_mean"], row["validation_MAPE_sd"]))
    summary_root = EXPERIMENT_ROOT / phase / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    for filename, payload in (("per_run.csv", rows), ("config_summary.csv", summary)):
        with (summary_root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
            writer.writeheader()
            writer.writerows(payload)
    return summary


def smoke_test():
    output_dir = EXPERIMENT_ROOT / "smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = command_for(build_ofat_configs()[0], SEEDS[0], output_dir, smoke=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare-ofat", "smoke-test", "run-ofat", "aggregate-ofat"])
    parser.add_argument("--max-per-gpu", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare-ofat":
        print(write_manifest("ofat", build_ofat_configs()))
    elif args.action == "smoke-test":
        smoke_test()
    elif args.action == "run-ofat":
        if not manifest_path("ofat").exists():
            write_manifest("ofat", build_ofat_configs())
        run_phase("ofat", args.max_per_gpu)
    elif args.action == "aggregate-ofat":
        print(json.dumps(aggregate("ofat")[:5], indent=2))


if __name__ == "__main__":
    main()
