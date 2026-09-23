#!/usr/bin/env python3
"""Reproduce the two unstable 40k/48k Transformer runs with identical seeds.

This post-hoc diagnostic is isolated from the formal nested-size study.  It
uses the original fixed split-index files and frozen 120-epoch OFAT protocol,
then evaluates the two completed checkpoints on the leakage-free seed42 test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/zxy/project/TAPD")
CODE_ROOT = ROOT / "baselinemodels/pub0818/transfomer"
FORMAL_ROOT = CODE_ROOT / "nested_data_scaling_main_ofat_80k_pool_seed42"
STUDY_ROOT = FORMAL_ROOT / "reproducibility_rerun_same_seed_120epoch_20260827"
PYTHON = ROOT / ".venv/bin/python"
TRAINER = CODE_ROOT / "train_validation_only.py"
EVALUATOR = CODE_ROOT / "evaluate_final_test.py"
DATA_DIR = ROOT / "TrainData/data/chiNlimited/n_train_100k"
TEST_DIR = ROOT / "TrainData/data/chiNlimited/n_test_2k_leakage_free_seed42"
CACHE = CODE_ROOT / "data_cache/n_train_100k_records.pkl"

RUNS = ((40, 202603, 0), (48, 202604, 1))
FROZEN = {
    "batch_size": 32,
    "lr": 5e-4,
    "weight_decay": 0.0,
    "d_model": 192,
    "num_heads": 6,
    "num_layers": 3,
    "ff_mult": 6,
    "dropout": 0.0,
    "readout": "attention",
    "position_encoding": "sinusoidal",
    "representation": "full_topology",
    "architecture": "transformer",
}


def training_command(size_k: int, seed: int, output_dir: Path) -> list[str]:
    command = [
        str(PYTHON), str(TRAINER),
        "--data-dir", str(DATA_DIR),
        "--records-cache", str(CACHE),
        "--split-indices", str(
            FORMAL_ROOT / "nested_indices" / f"nested_train_{size_k}k_common_val20k_seed42.npz"
        ),
        "--output-dir", str(output_dir),
        "--run-name", f"{size_k}k__seed{seed}__exact_reproduction",
        "--seed", str(seed),
        "--device", "cuda:0",
        "--epochs", "120",
        "--min-epochs", "120",
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
        command.extend([flag, str(FROZEN[key])])
    return command


def write_manifest() -> None:
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    original_snapshot = (
        FORMAL_ROOT / "training/runs/40k/seed202603/source_snapshot/train_validation_only.py"
    )
    if TRAINER.read_bytes() != original_snapshot.read_bytes():
        raise RuntimeError("Current trainer differs from the original source snapshot")
    manifest = {
        "study": "same_seed_exact_protocol_reproducibility_diagnostic",
        "diagnostic_only": True,
        "formal_study": str(FORMAL_ROOT),
        "runs": [
            {"size_k": size_k, "seed": seed, "physical_gpu": gpu}
            for size_k, seed, gpu in RUNS
        ],
        "epochs": 120,
        "minimum_epochs": 120,
        "early_stopping_patience": 25,
        "frozen_hyperparameters": FROZEN,
        "trainer": str(TRAINER),
        "fixed_split_source": str(FORMAL_ROOT / "nested_indices"),
        "test_directory": str(TEST_DIR),
        "test_access_during_training": "forbidden",
        "test_rule": "Evaluate both checkpoints only after both training processes complete",
        "note": (
            "Seeds are identical to the formal anomalous runs. PyTorch seeds and the "
            "DataLoader generator are fixed by the unchanged trainer, but deterministic "
            "algorithms are not forced because the formal protocol did not force them."
        ),
    }
    (STUDY_ROOT / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def train_both() -> None:
    active = []
    for size_k, seed, gpu in RUNS:
        output_dir = STUDY_ROOT / "training/runs" / f"{size_k}k" / f"seed{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = STUDY_ROOT / "training/logs" / f"{size_k}k__seed{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        })
        process = subprocess.Popen(
            training_command(size_k, seed, output_dir), cwd=ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT,
        )
        active.append((size_k, seed, gpu, process, handle, log_path, output_dir))
        print(f"launched {size_k}k seed={seed} physical_gpu={gpu} pid={process.pid}", flush=True)

    failures = []
    for size_k, seed, gpu, process, handle, log_path, output_dir in active:
        return_code = process.wait()
        handle.close()
        complete = (
            return_code == 0
            and (output_dir / "metrics.json").exists()
            and (output_dir / "best_model.pth").exists()
        )
        print(f"finished {size_k}k seed={seed} return_code={return_code}", flush=True)
        if not complete:
            failures.append({"size_k": size_k, "seed": seed, "gpu": gpu, "log": str(log_path)})
    if failures:
        raise RuntimeError(f"Reproduction training failed: {failures}")


def evaluate_test() -> None:
    environment = os.environ.copy()
    environment.update({
        "TRANSFORMER_EXPERIMENT_ROOT": str(STUDY_ROOT),
        "CUDA_VISIBLE_DEVICES": "2",
    })
    subprocess.run(
        [
            str(PYTHON), str(EVALUATOR), "--phase", "training",
            "--test-dir", str(TEST_DIR),
            "--output-dir", str(STUDY_ROOT / "final_test"),
            "--device", "cuda:0", "--save-predictions",
        ],
        cwd=ROOT, env=environment, check=True,
    )


def write_comparison() -> None:
    original_validation = {}
    with (FORMAL_ROOT / "summary/validation_per_run.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            original_validation[(int(row["size_k"]), int(row["seed"]))] = row
    original_test = {}
    with (FORMAL_ROOT / "summary/test_per_run.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            original_test[(int(row["size_k"]), int(row["seed"]))] = row

    rerun_test = {}
    with (STUDY_ROOT / "final_test/per_run.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["run_dir"])
            rerun_test[(int(path.parent.name.removesuffix("k")), int(row["seed"]))] = row

    fields = [
        "size_k", "seed", "original_validation_MAPE", "rerun_validation_MAPE",
        "validation_MAPE_delta", "original_test_MAPE", "rerun_test_MAPE",
        "test_MAPE_delta", "original_best_epoch", "rerun_best_epoch",
        "original_best_val_loss", "rerun_best_val_loss",
    ]
    rows = []
    for size_k, seed, _ in RUNS:
        original_v = original_validation[(size_k, seed)]
        original_t = original_test[(size_k, seed)]
        rerun_metrics = json.loads(
            (STUDY_ROOT / "training/runs" / f"{size_k}k" / f"seed{seed}" / "metrics.json").read_text()
        )
        rerun_t = rerun_test[(size_k, seed)]
        original_vm = float(original_v["validation_MAPE"])
        rerun_vm = float(rerun_metrics["validation_MAPE"])
        original_tm = float(original_t["test_MAPE"])
        rerun_tm = float(rerun_t["MAPE"])
        rows.append({
            "size_k": size_k,
            "seed": seed,
            "original_validation_MAPE": original_vm,
            "rerun_validation_MAPE": rerun_vm,
            "validation_MAPE_delta": rerun_vm - original_vm,
            "original_test_MAPE": original_tm,
            "rerun_test_MAPE": rerun_tm,
            "test_MAPE_delta": rerun_tm - original_tm,
            "original_best_epoch": int(original_v["best_epoch"]),
            "rerun_best_epoch": int(rerun_metrics["best_epoch"]),
            "original_best_val_loss": float(original_v["best_val_loss"]),
            "rerun_best_val_loss": float(rerun_metrics["best_val_loss"]),
        })
    summary = STUDY_ROOT / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    with (summary / "same_seed_reproducibility_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (summary / "same_seed_reproducibility_comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    started = time.time()
    write_manifest()
    train_both()
    evaluate_test()
    write_comparison()
    completion = {
        "status": "complete",
        "training_runs": 2,
        "test_runs": 2,
        "elapsed_seconds": time.time() - started,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (STUDY_ROOT / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
