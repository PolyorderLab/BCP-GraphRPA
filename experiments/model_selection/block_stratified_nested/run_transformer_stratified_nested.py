#!/usr/bin/env python3
"""Corrected CLS/full-topology Transformer nested training-size runner."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from stratified_nested_common import block_labels, build_stratified_nested_indices, load_sample_ids, write_indices

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CODE_ROOT = ROOT / "baselinemodels/pub0818/transfomer"
SOURCE = CODE_ROOT / "run_nested_data_scaling_80k_pool.py"
DESCRIPTORS = ROOT / "baselinemodels/pub0818/mlp/hyperparameter_selection_80k/data/n_train_100k_descriptor_matrix.csv"
SPLIT_FILE = ROOT / "baselinemodels/pub0818/gnn/hyperparameter_selection_80k/runs_100k_fixed80_20_validation_only/fixed_80k_20k_split_indices.npz"
OUTPUT = HERE / "results/transformer_cls_correct"
SIZES_K = [8, 16, 24, 32, 40, 48, 56, 64, 70, 72, 80]


def load_source():
    spec = importlib.util.spec_from_file_location("transformer_nested_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.STUDY_ROOT = OUTPUT
    return module


def prepare(source, base) -> None:
    split = np.load(SPLIT_FILE)
    pool = np.asarray(split["train_index"], dtype=np.int64)
    validation = np.asarray(split["val_index"], dtype=np.int64)
    labels = block_labels(load_sample_ids(DESCRIPTORS))
    subsets, audit = build_stratified_nested_indices(pool, labels, [size * 1000 for size in SIZES_K])
    write_indices(OUTPUT / "nested_indices", subsets, audit, validation_indices=validation, train_key="train_indices")
    records = base.load_verified_records()
    # The trainer expects the historical common-val filename convention.
    for size_k in SIZES_K:
        src = OUTPUT / "nested_indices" / f"nested_train_{size_k}k.npz"
        dst = OUTPUT / "nested_indices" / f"nested_train_{size_k}k_common_val20k_seed42.npz"
        data = np.load(src)
        np.savez_compressed(dst, train_indices=data["train_indices"], validation_indices=data["validation_indices"])
    manifest = {
        "study": "block_count_stratified_nested_training_size_ablation_with_common_validation",
        "runner": str(Path(__file__).resolve()), "source_runner": str(SOURCE),
        "sizes_k": SIZES_K, "seeds": source.SEEDS,
        "nested_rule": "within each 2/3/4/5-block stratum: one random_state=42 permutation; monotone largest-remainder quotas proportional to the frozen 80k pool",
        "fixed_split_source": str(SPLIT_FILE), "training_pool_count": len(pool), "validation_count": len(validation),
        "stratification_audit": str(OUTPUT / "nested_indices/stratification_audit.json"),
        "frozen_hyperparameters": source.FROZEN_CONFIG,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    source = load_source(); source.SIZES_K = list(reversed(SIZES_K)); base = source; prepare(source, base)
    if args.prepare_only:
        print(json.dumps({"prepared": 55, "output": str(OUTPUT)}, indent=2)); return
    source.run_training(args.slots_per_gpu)
    base.SIZES_K = SIZES_K
    base.aggregate_validation()
    completed = sum(1 for _ in (OUTPUT / "training/runs").glob("*/seed*/metrics.json"))
    if completed != 55:
        raise RuntimeError(f"Expected 55 completed Transformer runs, found {completed}")
    (OUTPUT / "COMPLETED.json").write_text(json.dumps({"status": "complete", "training_runs": completed}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
