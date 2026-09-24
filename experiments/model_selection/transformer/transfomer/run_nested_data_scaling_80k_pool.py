#!/usr/bin/env python3
"""Run the fixed-common-validation nested Transformer data-scaling study.

The 100k development corpus is split once into an 80k training pool and a
fixed 20k validation set with random_state=42. Training subsets are prefixes
of one frozen pool ordering, hence 8k is a strict subset of 16k, and so on.
The held-out seed42 2k test is inaccessible until every training run and
the validation aggregation have completed successfully.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parent
STUDY_ROOT = CODE_ROOT / "nested_data_scaling_80k_pool_seed42"
DATA_DIR = ROOT / "data/datasets/development_100k"
TEST_DIR = ROOT / "data/datasets/validation_seed42"
CACHE = CODE_ROOT / "data_cache/n_train_100k_records.pkl"
TRAINER = CODE_ROOT / "train_validation_only.py"
EVALUATOR = CODE_ROOT / "evaluate_final_test.py"
PYTHON = Path(sys.executable)

SIZES_K = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
SEEDS = [202601, 202602, 202603, 202604, 202605]
SPLIT_SEED = 42
MIN_EPOCHS = 120
MAX_EPOCHS = 120
FROZEN_CONFIG = {
    "batch_size": 40,
    "lr": 2e-4,
    "weight_decay": 0.0,
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 3,
    "ff_mult": 4,
    "dropout": 0.0,
    "readout": "cls",
    "position_encoding": "learnable",
    "representation": "full_topology",
    "architecture": "transformer",
}


def load_verified_records() -> list[dict]:
    if not CACHE.exists():
        raise FileNotFoundError(f"Missing verified 100k cache: {CACHE}")
    with CACHE.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("data_directory") != str(DATA_DIR.resolve()):
        raise RuntimeError("100k cache data-directory mismatch")
    if payload.get("record_count") != 100_000 or len(payload.get("records", [])) != 100_000:
        raise RuntimeError("100k cache record-count mismatch")
    records = payload["records"]
    names = [str(record["base_name"]) for record in records]
    if len(set(names)) != len(names):
        raise RuntimeError("100k cache contains duplicate base names")
    return records


def verify_final_test_isolation(records: list[dict]) -> list[str]:
    test_names = sorted(path.stem for path in TEST_DIR.glob("*.yml"))
    if len(test_names) != 2_000 or len(set(test_names)) != 2_000:
        raise RuntimeError("Expected exactly 2,000 unique leakage-free test records")
    development_names = {str(record["base_name"]) for record in records}
    overlap = development_names.intersection(test_names)
    if overlap:
        raise RuntimeError(f"Development/test leakage detected for {len(overlap)} records")
    return test_names


def prepare_protocol() -> None:
    records = load_verified_records()
    test_names = verify_final_test_isolation(records)
    all_indices = np.arange(len(records), dtype=np.int64)
    pool_indices, validation_indices = train_test_split(
        all_indices, test_size=0.2, random_state=SPLIT_SEED
    )
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if len(pool_indices) != 80_000 or len(validation_indices) != 20_000:
        raise RuntimeError("Expected an 80k/20k development split")
    if np.intersect1d(pool_indices, validation_indices).size:
        raise RuntimeError("Training pool overlaps fixed validation")

    indices_root = STUDY_ROOT / "nested_indices"
    indices_root.mkdir(parents=True, exist_ok=True)
    previous: set[int] = set()
    for size_k in SIZES_K:
        subset = pool_indices[: size_k * 1000]
        current = set(int(index) for index in subset)
        if not previous.issubset(current):
            raise RuntimeError(f"Nested-set audit failed at {size_k}k")
        previous = current
        path = indices_root / f"nested_train_{size_k}k_common_val20k_seed42.npz"
        np.savez_compressed(
            path,
            train_indices=subset,
            validation_indices=validation_indices,
        )

    manifest = {
        "study": "nested_training_size_ablation_with_common_validation",
        "development_directory": str(DATA_DIR),
        "development_count": 100_000,
        "fixed_split": {
            "random_state": SPLIT_SEED,
            "training_pool_count": 80_000,
            "validation_count": 20_000,
        },
        "sizes_k": SIZES_K,
        "nested_rule": "Each training subset is a prefix of one frozen 80k pool ordering",
        "seeds": SEEDS,
        "checkpoint_and_early_stopping_metric": "fixed 20k validation log-MSE",
        "minimum_epochs_per_seed": MIN_EPOCHS,
        "maximum_epochs_per_seed": MAX_EPOCHS,
        "reported_development_metric": "fixed 20k validation MAPE",
        "frozen_hyperparameters": FROZEN_CONFIG,
        "final_test_directory": str(TEST_DIR),
        "final_test_count": len(test_names),
        "final_test_access_during_training": "forbidden",
        "final_test_rule": "Evaluate all frozen size-by-seed checkpoints only after 50/50 training runs complete",
        "interpretation_rule": (
            "Use the common 20k validation curve for model-size selection; "
            "the seed42 2k curve is a held-out report, not a tuning signal"
        ),
    }
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    (STUDY_ROOT / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def command_for(size_k: int, seed: int, output_dir: Path) -> list[str]:
    command = [
        str(PYTHON), str(TRAINER),
        "--data-dir", str(DATA_DIR),
        "--records-cache", str(CACHE),
        "--split-indices", str(
            STUDY_ROOT / "nested_indices" / f"nested_train_{size_k}k_common_val20k_seed42.npz"
        ),
        "--output-dir", str(output_dir),
        "--run-name", f"{size_k}k__seed{seed}",
        "--seed", str(seed),
        "--device", "cuda:0",
        "--epochs", str(MAX_EPOCHS),
        "--min-epochs", str(MIN_EPOCHS),
        "--early-stopping-patience", "25",
    ]
    for key, flag in (
        ("batch_size", "--batch-size"), ("lr", "--lr"),
        ("weight_decay", "--weight-decay"), ("d_model", "--d-model"),
        ("num_heads", "--num-heads"), ("num_layers", "--num-layers"),
        ("ff_mult", "--ff-mult"), ("dropout", "--dropout"),
        ("readout", "--readout"), ("position_encoding", "--position-encoding"),
        ("representation", "--representation"), ("architecture", "--architecture"),
    ):
        command.extend([flag, str(FROZEN_CONFIG[key])])
    return command


def aggregate_validation() -> None:
    rows = []
    for size_k in SIZES_K:
        for seed in SEEDS:
            run_dir = STUDY_ROOT / "training/runs" / f"{size_k}k" / f"seed{seed}"
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                raise RuntimeError(f"Missing {metrics_path}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("test_loaded") is not False:
                raise RuntimeError(f"Test-isolation audit failed: {metrics_path}")
            if metrics.get("train_count") != size_k * 1000 or metrics.get("validation_count") != 20_000:
                raise RuntimeError(f"Count audit failed: {metrics_path}")
            if int(metrics.get("epochs_completed", -1)) < MIN_EPOCHS:
                raise RuntimeError(f"Minimum-epoch audit failed: {metrics_path}")
            rows.append({
                "size_k": size_k,
                "seed": seed,
                "train_count": metrics["train_count"],
                "validation_count": metrics["validation_count"],
                "validation_MAPE": metrics["validation_MAPE"],
                "validation_RMSE": metrics["validation_RMSE"],
                "best_epoch": metrics["best_epoch"],
                "best_val_loss": metrics["best_val_loss"],
                "run_dir": str(run_dir),
            })
    summary = []
    for size_k in SIZES_K:
        values = [float(row["validation_MAPE"]) for row in rows if row["size_k"] == size_k]
        summary.append({
            "size_k": size_k,
            "n": len(values),
            "validation_MAPE_mean": statistics.mean(values),
            "validation_MAPE_sd": statistics.stdev(values),
            "validation_MAPE_sem": statistics.stdev(values) / len(values) ** 0.5,
        })
    write_csv(STUDY_ROOT / "summary/validation_per_run.csv", rows)
    write_csv(STUDY_ROOT / "summary/validation_summary.csv", summary)


def aggregate_test() -> None:
    source = STUDY_ROOT / "final_test/per_run.csv"
    with source.open(encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        size_name = Path(raw["run_dir"]).parent.name
        size_k = int(size_name.removesuffix("k"))
        rows.append({
            "size_k": size_k,
            "seed": int(raw["seed"]),
            "test_count": int(raw["test_count"]),
            "test_MAPE": float(raw["MAPE"]),
            "test_RMSE": float(raw["RMSE"]),
            "test_MAE": float(raw["MAE"]),
            "test_R2": float(raw["R2"]),
            "test_MaxAE": float(raw["MaxAE"]),
            "run_dir": raw["run_dir"],
        })
    summary = []
    for size_k in SIZES_K:
        values = [row["test_MAPE"] for row in rows if row["size_k"] == size_k]
        if len(values) != len(SEEDS):
            raise RuntimeError(f"Expected five test results for {size_k}k, found {len(values)}")
        summary.append({
            "size_k": size_k,
            "n": len(values),
            "test_MAPE_mean": statistics.mean(values),
            "test_MAPE_sd": statistics.stdev(values),
            "test_MAPE_sem": statistics.stdev(values) / len(values) ** 0.5,
        })
    write_csv(STUDY_ROOT / "summary/test_per_run.csv", rows)
    write_csv(STUDY_ROOT / "summary/test_summary.csv", summary)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_training(max_per_gpu: int) -> None:
    runs_root = STUDY_ROOT / "training/runs"
    logs_root = STUDY_ROOT / "training/logs"
    failures_root = STUDY_ROOT / "training/failures"
    for path in (runs_root, logs_root, failures_root):
        path.mkdir(parents=True, exist_ok=True)
    waiting = []
    for size_k in SIZES_K:
        for seed in SEEDS:
            output_dir = runs_root / f"{size_k}k" / f"seed{seed}"
            metrics_path = output_dir / "metrics.json"
            checkpoint_path = output_dir / "best_model.pth"
            reusable = False
            if metrics_path.exists() and checkpoint_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    reusable = (
                        metrics.get("train_count") == size_k * 1000
                        and metrics.get("validation_count") == 20_000
                        and metrics.get("test_loaded") is False
                        and int(metrics.get("epochs_completed", -1)) >= MIN_EPOCHS
                    )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    reusable = False
            if not reusable:
                waiting.append((size_k, seed, output_dir))
    active = []
    gpu_counts = [0, 0, 0, 0]
    while waiting or active:
        for gpu in range(4):
            while waiting and gpu_counts[gpu] < max_per_gpu:
                size_k, seed, output_dir = waiting.pop(0)
                output_dir.mkdir(parents=True, exist_ok=True)
                log_path = logs_root / f"{size_k}k__seed{seed}.log"
                handle = log_path.open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment.update({
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "MPLCONFIGDIR": "/tmp/matplotlib-tapd-nested-transformer-scaling",
                })
                process = subprocess.Popen(
                    command_for(size_k, seed, output_dir),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    env=environment,
                )
                active.append((process, gpu, size_k, seed, output_dir, log_path, handle, time.time()))
                gpu_counts[gpu] += 1
                print(f"launch size={size_k}k seed={seed} gpu={gpu} pid={process.pid}", flush=True)
        time.sleep(10)
        remaining = []
        for process, gpu, size_k, seed, output_dir, log_path, handle, started in active:
            code = process.poll()
            if code is None:
                remaining.append((process, gpu, size_k, seed, output_dir, log_path, handle, started))
                continue
            handle.close()
            gpu_counts[gpu] -= 1
            if code != 0 or not (output_dir / "metrics.json").exists():
                failure = {
                    "size_k": size_k,
                    "seed": seed,
                    "return_code": code,
                    "log": str(log_path),
                }
                (failures_root / f"{size_k}k__seed{seed}.json").write_text(
                    json.dumps(failure, indent=2), encoding="utf-8"
                )
                for other, *_ in remaining:
                    other.terminate()
                raise RuntimeError(f"Failed {size_k}k seed={seed}; see {log_path}")
            print(
                f"complete size={size_k}k seed={seed} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
        active = remaining


def run_final_test() -> None:
    environment = os.environ.copy()
    environment["TRANSFORMER_EXPERIMENT_ROOT"] = str(STUDY_ROOT)
    subprocess.run(
        [
            str(PYTHON), str(EVALUATOR),
            "--phase", "training",
            "--test-dir", str(TEST_DIR),
            "--output-dir", str(STUDY_ROOT / "final_test"),
            "--device", "cuda:0",
            "--save-predictions",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def main() -> None:
    max_per_gpu = int(os.environ.get("TRANSFORMER_NESTED_MAX_PER_GPU", "2"))
    prepare_protocol()
    run_training(max_per_gpu=max_per_gpu)
    aggregate_validation()
    if not all((STUDY_ROOT / "training/runs" / f"{size}k" / f"seed{seed}" / "metrics.json").is_file() for size in SIZES_K for seed in SEEDS):
        raise RuntimeError("Refusing test access before all planned training runs complete")
    run_final_test()
    aggregate_test()
    completion = {
        "status": "complete",
        "training_runs": len(SIZES_K) * len(SEEDS),
        "test_runs": len(SIZES_K) * len(SEEDS),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (STUDY_ROOT / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
