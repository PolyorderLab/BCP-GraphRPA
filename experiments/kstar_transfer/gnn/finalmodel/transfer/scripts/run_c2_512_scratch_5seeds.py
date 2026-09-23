#!/usr/bin/env python3
"""Run the five-seed randomly initialized C2-512 k* scratch baseline.

This scheduler uses one process per visible GPU and queues the fifth seed until
one of the first four jobs finishes.  The result tree is intentionally separate
from the legacy 256-dimensional ``transfer/scratch`` copy so that the two
baselines cannot be confused during aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path("/home/zxy/project/TAPD")
TRANSFER_ROOT = PROJECT_ROOT / "baselinemodels/pub0818/gnn/finalmodel/transfer"
TRAINER = TRANSFER_ROOT / "scripts/train_c2_kstar_transfer.py"
SCRATCH_ROOT = TRANSFER_ROOT / "scratch_c2_512"
LOG_ROOT = TRANSFER_ROOT / "logs/c2_scratch_512"
TRAIN_CACHE = TRANSFER_ROOT / "data_cache/kstar_train_100k.pt"
TEST_CACHE = TRANSFER_ROOT / "data_cache/kstar_seed43_independent_2k.pt"
TEST_DATA_DIR = PROJECT_ROOT / "TrainData/data/chiNlimited/n_test_2k_leakage_free_seed43_independent"
SEEDS = (202601, 202602, 202603, 202604, 202605)


def output_dir(run_id: int, seed: int) -> Path:
    return SCRATCH_ROOT / f"scratch_c2_512_run{run_id:02d}_seed{seed}"


def run_name(run_id: int, seed: int) -> str:
    return f"scratch_c2_512_run{run_id:02d}_seed{seed}"


def command(args, run_id: int, seed: int) -> list[str]:
    return [
        args.python,
        "-u",
        str(TRAINER),
        "--transfer-mode",
        "scratch",
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir(run_id, seed)),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--encoder-lr",
        str(args.encoder_lr),
        "--decoder-lr",
        str(args.decoder_lr),
        "--num-workers",
        str(args.num_workers),
        "--train-cache",
        str(args.train_cache),
        "--test-cache",
        str(args.test_cache),
        "--device",
        "cuda:0",
    ]


def write_manifest(args) -> None:
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "C2_512_random_initialization_kstar_scratch",
        "transfer_mode": "scratch",
        "seeds": list(SEEDS),
        "result_root": str(SCRATCH_ROOT.resolve()),
        "trainer": str(TRAINER.resolve()),
        "architecture": {
            "encoder": "LearnableNodeGINEEncoder",
            "hidden_dim": 512,
            "num_layers": 5,
            "max_nodes": 80,
            "decoder_layers": 1,
            "node_features": "learnable positional node embedding; data.x zeroed",
            "virtual_node": True,
            "initialization": "random seeded initialization; no pretrained encoder",
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "encoder_lr": args.encoder_lr,
            "decoder_lr": args.decoder_lr,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=10)",
        },
        "data": {
            "task_type": "k_wavevector",
            "train_data_dir": str((PROJECT_ROOT / "TrainData/data/chiNlimited/n_train_100k").resolve()),
            "test_data_dir": str(TEST_DATA_DIR.resolve()),
            "test_cache": str(args.test_cache.resolve()),
            "test_role": "independent final inference only; never used for selection",
        },
        "scheduler": {
            "gpus": args.gpus,
            "jobs_per_gpu": 1,
            "poll_seconds": args.poll_seconds,
        },
    }
    (SCRATCH_ROOT / "orchestration_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run(args) -> int:
    if not args.gpus:
        raise ValueError("At least one GPU id is required")
    if not TRAINER.is_file():
        raise FileNotFoundError(TRAINER)
    if not args.train_cache.is_file():
        raise FileNotFoundError(args.train_cache)
    if not args.test_cache.is_file():
        raise FileNotFoundError(args.test_cache)

    tasks = [(run_id, seed) for run_id, seed in enumerate(SEEDS, start=1)]
    for run_id, seed in tasks:
        target = output_dir(run_id, seed)
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty scratch result directory: {target}"
            )

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    write_manifest(args)
    slots = [(gpu, slot_id) for slot_id, gpu in enumerate(args.gpus, start=1)]
    pending = list(tasks)
    active: dict[tuple[int, int], tuple[subprocess.Popen, object, int, int, Path]] = {}
    failed: list[tuple[int, int, Path, int]] = []

    print(f"Scratch result root: {SCRATCH_ROOT}", flush=True)
    print(f"Test data directory: {TEST_DATA_DIR}", flush=True)
    print(f"Seeds queued: {list(SEEDS)}", flush=True)
    print(f"GPU slots: {slots}", flush=True)

    while pending or active:
        for slot in slots:
            if slot in active or not pending:
                continue
            gpu, _slot_id = slot
            run_id, seed = pending.pop(0)
            log_path = LOG_ROOT / f"{run_name(run_id, seed)}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.Popen(
                command(args, run_id, seed),
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[slot] = (proc, log_handle, run_id, seed, log_path)
            print(
                f"GPU {gpu} slot {_slot_id}: started {run_name(run_id, seed)} "
                f"(pid {proc.pid}); remaining_queue={len(pending)}",
                flush=True,
            )

        time.sleep(args.poll_seconds)
        for slot, record in list(active.items()):
            proc, log_handle, run_id, seed, log_path = record
            return_code = proc.poll()
            if return_code is None:
                continue
            log_handle.close()
            del active[slot]
            gpu, slot_id = slot
            print(
                f"GPU {gpu} slot {slot_id}: finished {run_name(run_id, seed)} "
                f"rc={return_code}; active={len(active)}",
                flush=True,
            )
            if return_code:
                failed.append((run_id, seed, log_path, return_code))

    if failed:
        for run_id, seed, log_path, return_code in failed:
            print(
                f"FAILED run{run_id:02d} seed{seed}: rc={return_code}; log={log_path}",
                file=sys.stderr,
            )
        return 1
    print("All five C2-512 scratch runs completed successfully.", flush=True)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--python", default=str(PROJECT_ROOT / ".venv/bin/python"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-lr", type=float, default=5e-4)
    parser.add_argument("--decoder-lr", type=float, default=5e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-cache", type=Path, default=TRAIN_CACHE)
    parser.add_argument("--test-cache", type=Path, default=TEST_CACHE)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
