#!/usr/bin/env python3
"""Run the final C2 GNN on block-stratified strict nested training subsets."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

import gnn_parallel_scheduler

from stratified_nested_common import (
    block_labels,
    build_stratified_nested_indices,
    load_sample_ids,
    write_indices,
)

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
GNN_ROOT = ROOT / "sourcecode/pub/gnn"
SOURCE = GNN_ROOT / "training_size_c2_nested_common_val_seed42/run_nested_training_size.py"
DESCRIPTORS = ROOT / "sourcecode/pub/mlp/hyperparameter_selection_80k/data/n_train_100k_descriptor_matrix.csv"
OUTPUT = HERE / "results/gnn"
SIZES = tuple(range(8_000, 80_001, 8_000))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    source = load_module(SOURCE, "c2_nested_source")
    for name in ("configs", "runs", "logs", "summary", "figures", "nested_indices"):
        (OUTPUT / name).mkdir(parents=True, exist_ok=True)

    fixed = np.load(source.BASE_SPLIT)
    pool = np.asarray(fixed["train_index"], dtype=np.int64)
    labels = block_labels(load_sample_ids(DESCRIPTORS))
    subsets, audit = build_stratified_nested_indices(pool, labels, SIZES)
    write_indices(OUTPUT / "nested_indices", subsets, audit, train_key="train_index")
    paths = {size: OUTPUT / "nested_indices" / f"nested_train_{size // 1000}k.npz" for size in SIZES}
    records = source.build_configs(OUTPUT, paths)
    records.sort(key=lambda item: (-int(item["train_size"]), int(item["seed"])))
    for cfg in records:
        cfg["sampling_rule"] = "block-count-stratified strict nested prefixes"
        cfg["sampling_random_state"] = 42
        cfg["protocol_version"] = "c2_block_stratified_nested_common_seed42_20260921"
        (OUTPUT / "configs" / f"{cfg['run_id']}.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8"
        )
    nested = {
        "paths": paths,
        "records": [
            {
                "train_size": size,
                "path": str(paths[size]),
            }
            for size in SIZES
        ],
    }
    source.write_protocol(OUTPUT, nested, records)
    protocol_path = OUTPUT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update({
        "runner": str(Path(__file__).resolve()),
        "source_runner": str(SOURCE),
        "nested_rule": "within each 2/3/4/5-block stratum: one random_state=42 permutation; every requested size uses monotone largest-remainder quotas proportional to the frozen 80k pool",
        "stratification_audit": str(OUTPUT / "nested_indices/stratification_audit.json"),
        "protocol_version": "c2_block_stratified_nested_common_seed42_20260921",
    })
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps({"prepared": len(records), "output": str(OUTPUT)}, indent=2))
        return
    gnn_parallel_scheduler.run(
        records,
        OUTPUT,
        args.gpus,
        args.slots_per_gpu,
        retries=2,
        worker=source.WORKER,
        cwd=source.ROOT,
    )
    source.summarize(OUTPUT, records)
    completed = sum(1 for _ in (OUTPUT / "runs").glob("C2_nested_train__*/metrics.json"))
    if completed != len(records):
        raise RuntimeError(f"Expected {len(records)} C2 runs, found {completed}")
    (OUTPUT / "COMPLETED.json").write_text(
        json.dumps({"status": "complete", "training_runs": completed}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"training_complete": completed, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
