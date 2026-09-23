#!/usr/bin/env python3
"""Evaluate the copied legacy-S01 scratch k* models on seed43.

The independent seed43 dataset is inference-only.  This script consumes the shared
raw PyG cache prepared with ``task_type=k_wavevector``, reconstructs the exact node
features and virtual-node graph wrapper used by the original scratch trainer, and
strictly loads each complete checkpoint before evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader


os.environ.setdefault("MPLCONFIGDIR", "/tmp/tapd_mplconfig")


PROJECT_ROOT = Path("/home/zxy/project/TAPD")
TRANSFER_ROOT = (
    PROJECT_ROOT / "baselinemodels/pub0818/gnn/finalmodel/transfer"
)
SCRATCH_ROOT = TRANSFER_ROOT / "scratch"
DEFAULT_CACHE = TRANSFER_ROOT / "data_cache/kstar_seed43_independent_2k.pt"
TEST_DATA_DIR = (
    PROJECT_ROOT
    / "TrainData/data/chiNlimited/n_test_2k_leakage_free_seed43_independent"
)
LEGACY_SCRIPT_DIR = PROJECT_ROOT / "mainmodel/scripts"
SEEDS = (202601, 202602, 202603, 202604, 202605)
TASK_TYPE = "k_wavevector"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LEGACY_SCRIPT_DIR))

import train_single_virtualnode_undirected_freeend_branch_degree_kstar as legacy  # noqa: E402
from src.ira_graph_common import extract_predictions  # noqa: E402
from src.plotting import compute_metrics  # noqa: E402


def run_dir(seed: int) -> Path:
    run_id = SEEDS.index(seed) + 1
    return SCRATCH_ROOT / f"scratch_run{run_id:02d}_seed{seed}"


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def load_cached_test_dataset(cache: Path):
    if not cache.is_file():
        raise FileNotFoundError(
            f"Shared seed43 k* cache is not ready: {cache}. "
            "Prepare it with task_type=k_wavevector before running inference."
        )
    raw = torch.load(cache, map_location="cpu", weights_only=False)
    if len(raw) != 2000:
        raise RuntimeError(f"Expected 2,000 seed43 samples, found {len(raw)} in {cache}")
    # Reproduce load_wrapped_splits() from the original S01 k* trainer exactly.
    featured = legacy.FreeendBranchDegreeNodeFeatureWrapper(raw)
    return legacy.VirtualNodeWrapper(featured, node_mode="original")


def evaluate_one(seed: int, dataset, args: argparse.Namespace) -> dict[str, object]:
    output_dir = run_dir(seed)
    checkpoint = output_dir / "best_model.pth"
    config_path = output_dir / "run_config.json"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Incomplete scratch run directory: {output_dir}")

    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if int(config["seed"]) != seed or config["task_type"] != TASK_TYPE:
        raise RuntimeError(f"Run metadata mismatch in {config_path}")
    hyper = config["hyperparameters"]
    expected = {
        "hidden_dim": legacy.HIDDEN_DIM,
        "num_layers": legacy.NUM_LAYERS,
        "dec_layers": legacy.DEC_LAYERS,
        "in_feats": legacy.IN_FEATS,
        "edge_dim": legacy.EDGE_DIM,
    }
    mismatched = {
        key: (hyper.get(key), value)
        for key, value in expected.items()
        if hyper.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"Scratch architecture metadata mismatch: {mismatched}")

    device = torch.device(args.device)
    model = legacy.build_model(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed for {checkpoint}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    y_pred, y_true = extract_predictions(model, loader, device)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size != 2000 or not np.isfinite(y_pred).all():
        raise RuntimeError(f"Invalid predictions for seed {seed}: n={y_true.size}")

    absolute_error = np.abs(y_pred - y_true)
    ape_percent = absolute_error / np.clip(np.abs(y_true), 1e-12, None) * 100.0
    metric_values = compute_metrics(y_true, y_pred)
    metrics = {
        "experiment": "legacy_S01_scratch_kstar_seed43_inference",
        "transfer_mode": "scratch",
        "seed": seed,
        "task_type": TASK_TYPE,
        "test_role": "independent leakage-free inference only; not used for training or selection",
        "test_data_dir": str(TEST_DATA_DIR),
        "test_cache": str(args.cache.resolve()),
        "test_size": int(y_true.size),
        "source_checkpoint": str(checkpoint.resolve()),
        "strict_full_checkpoint_load": True,
        "architecture": {
            "encoder": "legacy GINEEncoder",
            "hidden_dim": legacy.HIDDEN_DIM,
            "num_layers": legacy.NUM_LAYERS,
            "decoder_layers": legacy.DEC_LAYERS,
            "node_features": ["is_free_end", "is_branch_point", "degree"],
            "edge_features": ["f", "A/B type"],
            "virtual_node": True,
            "graph_topology": "original_bidirectional",
        },
        "MAPE": float(metric_values["MAPE"]),
        "RMSE": float(metric_values["RMSE"]),
        "MAE": float(metric_values["MAE"]),
        "R2": float(metric_values["R²"]),
        "MaxAE": float(metric_values["MaxAE"]),
        "runtime": {
            "device": str(device),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    atomic_npz(
        output_dir / "seed43_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        abs_error=absolute_error,
        ape_percent=ape_percent,
    )
    atomic_json(output_dir / "seed43_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=legacy.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    unknown = sorted(set(args.seeds) - set(SEEDS))
    if unknown:
        raise ValueError(f"Unknown scratch seeds: {unknown}")
    dataset = load_cached_test_dataset(args.cache)
    for seed in args.seeds:
        result = evaluate_one(seed, dataset, args)
        print(
            f"scratch seed {seed}: MAPE={result['MAPE']:.6f}% "
            f"R2={result['R2']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
