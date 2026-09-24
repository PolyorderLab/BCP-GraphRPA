#!/usr/bin/env python3
"""Run five-method, five-seed k* label-budget experiments on available GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
TRANSFER_ROOT = Path(__file__).resolve().parent.parent
GRAPH_TRAINER = TRANSFER_ROOT / "scripts/train_c2_kstar_transfer.py"
MLP_TRAINER = TRANSFER_ROOT / "scripts/train_kstar_label_curve_embedding_mlp.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/runs/kstar_label_curve"
TRAIN_CACHE = PROJECT_ROOT / "results/cache/kstar_train_100k.pt"
TEST_CACHE = PROJECT_ROOT / "results/cache/kstar_seed43_independent_2k.pt"
SEEDS = (202601, 202602, 202603, 202604, 202605)
SIZES = (100, 250, 500, 1000, 2000)
METHODS = (
    "frozen_encoder",
    "frozen_embedding_mlp",
    "finetune_encoder",
    "two_stage",
    "scratch",
)


def output_dir(root: Path, method: str, size: int, seed: int) -> Path:
    return root / method / f"n{size}" / f"seed{seed}"


def command(args, method: str, size: int, seed: int) -> list[str]:
    target = output_dir(args.output_root, method, size, seed)
    common = [
        args.python,
        str(MLP_TRAINER if method == "frozen_embedding_mlp" else GRAPH_TRAINER),
        "--seed", str(seed),
        "--train-size", str(size),
        "--validation-size", str(args.validation_size),
        "--output-dir", str(target),
        "--device", "cuda:0",
        "--stratify-block-count",
    ]
    if method == "frozen_embedding_mlp":
        return common
    return common + [
        "--transfer-mode", method,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--encoder-lr", str(args.encoder_lr),
        "--decoder-lr", str(args.decoder_lr),
        "--frozen-epochs", str(args.frozen_epochs),
        "--num-workers", str(args.num_workers),
        "--train-cache", str(TRAIN_CACHE),
        "--test-cache", str(TEST_CACHE),
    ]


def complete(root: Path, method: str, size: int, seed: int) -> bool:
    target = output_dir(root, method, size, seed)
    return (target / "metrics.json").is_file() and (target / "seed43_predictions.npz").is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-lr", type=float, default=5e-4)
    parser.add_argument("--decoder-lr", type=float, default=5e-3)
    parser.add_argument("--frozen-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs_per_gpu < 1:
        raise ValueError("--jobs-per-gpu must be at least 1")

    unknown_sizes = sorted(set(args.sizes) - set(SIZES))
    if unknown_sizes:
        raise ValueError(f"Unsupported sizes: {unknown_sizes}; expected a subset of {SIZES}")
    jobs = [
        (method, size, seed)
        for size in args.sizes
        for seed in SEEDS
        for method in args.methods
        if not complete(args.output_root, method, size, seed)
    ]
    if args.dry_run:
        for job in jobs:
            print(" ".join(command(args, *job)))
        print(f"Pending jobs: {len(jobs)}")
        return
    if not TRAIN_CACHE.is_file() or not TEST_CACHE.is_file():
        raise FileNotFoundError("The shared train/test graph caches must exist before launch")

    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "kstar_label_curve_five_methods_five_seeds",
        "methods": args.methods,
        "sizes": args.sizes,
        "seeds": list(SEEDS),
        "validation_size": args.validation_size,
        "test_set": "seed43-independent leakage-free 2k",
        "paired_nested_subsets": True,
        "stratified_by_block_count": True,
        "block_balance": (
            "equal counts when divisible by four; otherwise counts differ by at most one, "
            "with remainder groups rotated across seeds"
        ),
        "epochs": args.epochs,
    }
    (args.output_root / "orchestration_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    slots = [
        (gpu, slot_id)
        for gpu in args.gpus
        for slot_id in range(1, args.jobs_per_gpu + 1)
    ]
    active = {}
    pending = list(jobs)
    failures = []
    while pending or active:
        for slot in slots:
            if slot in active or not pending:
                continue
            gpu, slot_id = slot
            method, size, seed = pending.pop(0)
            target = output_dir(args.output_root, method, size, seed)
            if target.exists() and any(target.iterdir()):
                raise FileExistsError(f"Refusing to overwrite incomplete directory: {target}")
            log_path = log_root / f"{method}_n{size}_seed{seed}.log"
            handle = log_path.open("w", encoding="utf-8")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command(args, method, size, seed),
                cwd=PROJECT_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[slot] = (process, handle, method, size, seed, log_path)
            print(
                f"GPU {gpu} slot {slot_id}: started {method} n={size} "
                f"seed={seed} pid={process.pid}",
                flush=True,
            )
        time.sleep(args.poll_seconds)
        for slot, record in list(active.items()):
            gpu, slot_id = slot
            process, handle, method, size, seed, log_path = record
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            del active[slot]
            print(
                f"GPU {gpu} slot {slot_id}: finished {method} n={size} "
                f"seed={seed} rc={return_code}",
                flush=True,
            )
            if return_code:
                failures.append((method, size, seed, log_path, return_code))
    if failures:
        for method, size, seed, log_path, return_code in failures:
            print(
                f"FAILED {method} n={size} seed={seed} rc={return_code}: {log_path}",
                file=sys.stderr,
            )
        raise SystemExit(1)
    print("All requested k* label-curve jobs completed.", flush=True)


if __name__ == "__main__":
    main()
