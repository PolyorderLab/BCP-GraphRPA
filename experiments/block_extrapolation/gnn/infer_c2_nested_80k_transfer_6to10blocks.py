#!/usr/bin/env python3
"""Evaluate each frozen C2 nested-80k checkpoint on the 6--10-block OOD set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.models.gnn_validation_worker import evaluate
from mainmodel.scripts import train_single_chin_supplementary_confirmation as source
from src.data_utils import load_and_process_dataset


MODEL_ROOT = ROOT / "checkpoints/gnn"
OUTPUT_ROOT = ROOT / "results/block_extrapolation_rerun/gnn"
TEST_DIR = ROOT / "data/datasets/6to10blocks_500_each"
CACHE = None
SEEDS = tuple(range(202601, 202606))


def checkpoint_config(seed: int) -> tuple[dict, Path, Path]:
    checkpoint_dir = MODEL_ROOT / f"seed{seed}"
    checkpoint = checkpoint_dir / "best_model.pth"
    config_path = checkpoint_dir / "metrics.json"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Incomplete C2 checkpoint directory: {checkpoint_dir}")
    return json.loads(config_path.read_text(encoding="utf-8")), checkpoint, checkpoint_dir


def test_source() -> list:
    data = load_and_process_dataset(str(TEST_DIR), task_type="chiN")
    counts = [int(item.num_blocks.reshape(-1)[0]) for item in data]
    if len(data) != 2500 or any(counts.count(n) != 500 for n in range(6, 11)):
        raise ValueError("Expected 500 valid samples for each block count 6--10")
    return data


def sample_names() -> list[str]:
    return [path.stem for path in sorted(TEST_DIR.glob("*.mat"))]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    relative = np.abs(residual) / np.clip(np.abs(y_true), 1e-12, None) * 100
    ss_total = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAPE": float(relative.mean()),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "R2": float(1.0 - np.sum(residual**2) / ss_total),
        "MaxAE": float(np.max(np.abs(residual))),
    }


def save_seed_result(seed: int, config: dict, checkpoint: Path, checkpoint_dir: Path, data: list) -> dict:
    wrapped = source.build_wrapped_dataset(data, config)
    loader = DataLoader(wrapped, batch_size=512, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = source.build_model(config, wrapped, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    pred_log, true_log, _ = evaluate(model, loader, device)
    y_pred, y_true = 10**np.asarray(pred_log), 10**np.asarray(true_log)
    names, block_count = sample_names(), np.asarray(
        [int(item.num_blocks.reshape(-1)[0].item()) for item in data], dtype=np.int64
    )
    if len(names) != len(y_true) or len(block_count) != len(y_true):
        raise RuntimeError("Transfer sample metadata does not align with loaded data")

    output_dir = OUTPUT_ROOT / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    abs_error = np.abs(y_pred - y_true)
    ape_percent = abs_error / np.clip(np.abs(y_true), 1e-12, None) * 100
    np.savez_compressed(
        output_dir / "transfer_predictions.npz", sample_name=np.asarray(names), block_count=block_count,
        y_true=y_true, y_pred=y_pred, y_true_log=true_log, y_pred_log=pred_log,
        abs_error=abs_error, ape_percent=ape_percent,
    )
    with (output_dir / "transfer_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_name", "block_count", "y_true", "y_pred", "abs_error", "ape_percent"])
        writer.writeheader()
        for row in zip(names, block_count, y_true, y_pred, abs_error, ape_percent, strict=True):
            writer.writerow(dict(zip(writer.fieldnames, row, strict=True)))

    per_block = []
    for count in sorted(set(block_count.tolist())):
        mask = block_count == count
        per_block.append({"block_count": int(count), "num_samples": int(mask.sum()), **metrics(y_true[mask], y_pred[mask])})
    with (output_dir / "per_block_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_block[0]))
        writer.writeheader(); writer.writerows(per_block)
    (output_dir / "per_block_metrics.json").write_text(json.dumps(per_block, indent=2) + "\n", encoding="utf-8")

    overall = metrics(y_true, y_pred)
    result = {
        "experiment": "C2 nested-80k 6--10-block transfer generalization",
        "run_id": config["run_id"], "seed": seed, "checkpoint": str(checkpoint),
        "test_dir": str(TEST_DIR), "num_samples": int(len(y_true)),
        "block_count_distribution": {str(count): int((block_count == count).sum()) for count in sorted(set(block_count.tolist()))},
        "device": str(device), **overall, "per_block_metrics": per_block,
    }
    (output_dir / "transfer_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"seed {seed}: MAPE={overall['MAPE']:.6f}%", flush=True)
    return {"seed": seed, **overall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    global TEST_DIR, OUTPUT_ROOT
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    TEST_DIR, OUTPUT_ROOT = args.test_dir, args.output_dir
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        rows = []
        for seed in SEEDS:
            result_path = OUTPUT_ROOT / f"seed{seed}" / "transfer_metrics.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append({"seed": seed, "MAPE": result["MAPE"], "RMSE": result["RMSE"], "MAE": result["MAE"], "R2": result["R2"], "MaxAE": result["MaxAE"]})
        output = OUTPUT_ROOT / "transfer_6to10blocks_summary.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        summary = {"test_dir": str(TEST_DIR), "num_samples": 2500, "n_checkpoints": len(rows),
                   "mape_mean_percent": mean(row["MAPE"] for row in rows),
                   "mape_sd_percent": stdev(row["MAPE"] for row in rows), "per_seed": rows}
        (OUTPUT_ROOT / "transfer_6to10blocks_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    data = test_source()
    if args.cache_only:
        print(f"cached_samples={len(data)}")
        return
    selected = (args.seed,) if args.seed else SEEDS
    rows = []
    for seed in selected:
        config, checkpoint, checkpoint_dir = checkpoint_config(seed)
        rows.append(save_seed_result(seed, config, checkpoint, checkpoint_dir, data))
if __name__ == "__main__":
    main()
