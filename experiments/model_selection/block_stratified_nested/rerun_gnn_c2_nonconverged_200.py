#!/usr/bin/env python3
"""Force selected non-converged C2 GNN runs to complete all 200 epochs."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import gnn_parallel_scheduler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CANONICAL = HERE / "results/gnn"
OUTPUT = CANONICAL / "diagnostics/nonconverged_epoch200"
SOURCE = ROOT / "sourcecode/pub/gnn/training_size_c2_nested_common_val_seed42/run_nested_training_size.py"
CASES = (
    (8, 202602), (8, 202603),
    (16, 202604),
    (24, 202602), (24, 202604), (24, 202605),
    (32, 202601), (32, 202604),
    (40, 202602),
    (48, 202602), (48, 202604), (48, 202605),
    (64, 202601), (64, 202605),
    (72, 202605),
)


def load_source():
    spec = importlib.util.spec_from_file_location("c2_training_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=3)
    args = parser.parse_args()
    for name in ("configs", "runs", "logs"):
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)
    records = []
    for size_k, seed in CASES:
        source_path = CANONICAL / "configs" / f"C2_nested_train__{size_k}k__seed{seed}.json"
        cfg = json.loads(source_path.read_text(encoding="utf-8"))
        cfg["run_id"] = f"C2_nested_train__{size_k}k__seed{seed}__epoch200_confirmation"
        cfg["epochs"] = 200
        cfg["minimum_epochs"] = 200
        cfg["early_stopping_patience"] = 30
        cfg["protocol_version"] = "c2_block_stratified_nonconverged_epoch200_confirmation_20260922"
        (OUTPUT / "configs" / f"{cfg['run_id']}.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8"
        )
        records.append(cfg)
    records.sort(key=lambda item: (-int(item["train_size"]), int(item["seed"])))
    source = load_source()
    gnn_parallel_scheduler.run(
        records,
        OUTPUT,
        args.gpus,
        args.slots_per_gpu,
        retries=2,
        worker=source.WORKER,
        cwd=source.ROOT,
    )
    completed = sum(1 for _ in (OUTPUT / "runs").glob("*/metrics.json"))
    if completed != len(records):
        raise RuntimeError(f"Expected {len(records)} confirmations, found {completed}")
    (OUTPUT / "COMPLETED.json").write_text(
        json.dumps({"status": "complete", "training_runs": completed}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"training_complete": completed, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
