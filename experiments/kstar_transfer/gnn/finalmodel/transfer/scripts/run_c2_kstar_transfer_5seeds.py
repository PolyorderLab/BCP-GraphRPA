#!/usr/bin/env python3
"""Print or launch the 15 C2-to-k* transfer jobs.

The scheduler assigns one job per visible GPU and starts a replacement as soon
as a GPU becomes free.  Use ``--dry-run`` to print commands without launching.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[6]
TRANSFER_ROOT = PROJECT_ROOT / "experiments/kstar_transfer/gnn/finalmodel/transfer"
TRAINER = TRANSFER_ROOT / "scripts/train_c2_kstar_transfer.py"
SEEDS = (202601, 202602, 202603, 202604, 202605)
MODES = ("scratch", "frozen_encoder", "finetune_encoder", "two_stage")


def output_dir(mode: str, run_id: int, seed: int) -> Path:
    return TRANSFER_ROOT / mode / f"{mode}_run{run_id:02d}_seed{seed}"


def command(args, mode, run_id, seed):
    cmd = [
        args.python,
        str(TRAINER),
        "--transfer-mode", mode,
        "--seed", str(seed),
        "--output-dir", str(output_dir(mode, run_id, seed)),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--encoder-lr", str(args.encoder_lr),
        "--decoder-lr", str(args.decoder_lr),
        "--frozen-epochs", str(args.frozen_epochs),
        "--num-workers", str(args.num_workers),
        "--device", "cuda:0",
    ]
    if args.train_cache:
        cmd.extend(("--train-cache", str(args.train_cache)))
    if args.test_cache:
        cmd.extend(("--test-cache", str(args.test_cache)))
    return cmd


def run(args):
    jobs = [
        (mode, run_id, seed)
        for mode in MODES
        for run_id, seed in enumerate(SEEDS, start=1)
        if not (output_dir(mode, run_id, seed) / "best_model.pth").exists()
    ]
    if args.dry_run:
        for job in jobs:
            print(" ".join(command(args, *job)))
        return
    if not args.gpus:
        raise ValueError("At least one GPU id is required")
    if args.jobs_per_gpu < 1:
        raise ValueError("--jobs-per-gpu must be at least 1")
    if args.prepare_caches and (
        not args.train_cache.exists() or not args.test_cache.exists()
    ):
        prepare = [
            args.python,
            str(TRAINER),
            "--transfer-mode", "frozen_encoder",
            "--seed", str(SEEDS[0]),
            "--output-dir", "/tmp/unused_c2_kstar_cache_prepare",
            "--device", "cpu",
            "--train-cache", str(args.train_cache),
            "--test-cache", str(args.test_cache),
            "--prepare-data-cache-only",
        ]
        print("Preparing shared parsed datasets before launching GPU jobs...")
        subprocess.run(prepare, cwd=PROJECT_ROOT, check=True)
    logs = TRANSFER_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    slots = [
        (gpu, slot_id)
        for gpu in args.gpus
        for slot_id in range(1, args.jobs_per_gpu + 1)
    ]
    active = {}
    pending = list(jobs)
    failed = []
    while pending or active:
        for slot in slots:
            if slot in active or not pending:
                continue
            gpu, slot_id = slot
            mode, run_id, seed = pending.pop(0)
            log_path = logs / f"{mode}_run{run_id:02d}_seed{seed}.log"
            handle = log_path.open("w", encoding="utf-8")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command(args, mode, run_id, seed),
                cwd=PROJECT_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[slot] = (process, handle, mode, run_id, seed, log_path)
            print(
                f"GPU {gpu} slot {slot_id}: started {mode} seed {seed} "
                f"(pid {process.pid})"
            )
        time.sleep(args.poll_seconds)
        for slot, record in list(active.items()):
            gpu, slot_id = slot
            process, handle, mode, run_id, seed, log_path = record
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            del active[slot]
            print(
                f"GPU {gpu} slot {slot_id}: finished {mode} seed {seed}, "
                f"rc={return_code}"
            )
            if return_code:
                failed.append((mode, seed, str(log_path)))
    if failed:
        for mode, seed, log_path in failed:
            print(f"FAILED: {mode} seed {seed}: {log_path}", file=sys.stderr)
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="*", default=[0])
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="Concurrent training processes placed on each listed GPU.",
    )
    parser.add_argument("--python", default=str(PROJECT_ROOT / ".venv/bin/python"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-lr", type=float, default=5e-4)
    parser.add_argument("--decoder-lr", type=float, default=5e-3)
    parser.add_argument("--frozen-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=TRANSFER_ROOT / "data_cache/kstar_train_100k.pt",
    )
    parser.add_argument(
        "--test-cache",
        type=Path,
        default=TRANSFER_ROOT / "data_cache/kstar_seed43_independent_2k.pt",
    )
    parser.add_argument(
        "--prepare-caches",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
