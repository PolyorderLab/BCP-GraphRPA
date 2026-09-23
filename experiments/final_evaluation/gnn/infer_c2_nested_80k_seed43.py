#!/usr/bin/env python3
"""Inference-only seed43 evaluation for the five frozen C2 nested-80k models."""

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

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from baselinemodels.pub0818.gnn.gnn_validation_worker import evaluate
from mainmodel.scripts import train_single_chin_supplementary_confirmation as source
from src.data_utils import load_and_process_dataset


MODEL_ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "TrainData/data/chiNlimited/n_test_2k_leakage_free_seed43_independent"
OUTPUT_ROOT = MODEL_ROOT / "seed43_independent_2k_inference"
CACHE = Path("/tmp/tapd_c2_seed43_test_data_chiN.pt")
SEEDS = tuple(range(202601, 202606))


def records() -> list[tuple[dict, Path]]:
    items = []
    for seed in SEEDS:
        checkpoint_dir = MODEL_ROOT / f"C2_nested_train__80k__seed{seed}"
        checkpoint, metrics_path = checkpoint_dir / "best_model.pth", checkpoint_dir / "metrics.json"
        if not checkpoint.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Incomplete C2 checkpoint directory: {checkpoint_dir}")
        items.append((json.loads(metrics_path.read_text(encoding="utf-8")), checkpoint))
    return items


def test_source() -> list:
    if CACHE.is_file():
        return torch.load(CACHE, map_location="cpu", weights_only=False)
    data = load_and_process_dataset(str(TEST_DIR), task_type="chiN")
    torch.save(data, CACHE)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    items = records()
    if args.seed is not None:
        items = [(cfg, ckpt) for cfg, ckpt in items if int(cfg["seed"]) == args.seed]
    if args.cache_only:
        print(f"cached_samples={len(test_source())}")
        return
    if args.summarize_only:
        rows = [json.loads((OUTPUT_ROOT / f"C2_nested_train__80k__seed{seed}.json").read_text()) for seed in SEEDS]
        scores = [row["mape_percent"] for row in rows]
        summary = {"model_family": "C2_nested_train_80k", "test_dir": str(TEST_DIR),
                   "n_test_samples": 2000, "n_checkpoints": len(rows),
                   "mape_mean_percent": mean(scores), "mape_sd_percent": stdev(scores),
                   "mape_per_seed_percent": scores}
        (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        with (OUTPUT_ROOT / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(json.dumps(summary, indent=2)); return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = source.build_wrapped_dataset(test_source(), items[0][0])
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    rows = []
    for config, checkpoint in items:
        model = source.build_model(config, data, device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        pred_log, true_log, metrics = evaluate(model, loader, device)
        stem = config["run_id"]
        np.savez_compressed(OUTPUT_ROOT / f"{stem}_predictions.npz", y_pred_log=pred_log, y_true_log=true_log)
        row = {
            "run_id": stem, "seed": int(config["seed"]), "checkpoint": str(checkpoint),
            "test_dir": str(TEST_DIR), "mape_percent": float(metrics["MAPE"]),
            "rmse": float(metrics["RMSE"]), "mae": float(metrics["MAE"]), "r2": float(metrics["R²"]),
        }
        (OUTPUT_ROOT / f"{stem}.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        print(f"{stem}: MAPE={row['mape_percent']:.6f}%", flush=True)

    if args.seed is not None:
        return
    scores = [row["mape_percent"] for row in rows]
    summary = {"model_family": "C2_nested_train_80k", "test_dir": str(TEST_DIR),
               "n_test_samples": len(data), "n_checkpoints": len(rows), "device": str(device),
               "mape_mean_percent": mean(scores), "mape_sd_percent": stdev(scores),
               "mape_per_seed_percent": scores}
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT_ROOT / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
