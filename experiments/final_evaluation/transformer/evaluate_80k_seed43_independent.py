#!/usr/bin/env python3
"""Evaluate the five frozen finalmode-80k checkpoints on the seed43 2k test set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / "checkpoints/transformer"
MODEL_DIR = ROOT / "experiments/model_selection/transformer/transfomer/dependencies_handoff/models"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(ROOT))

from baseline_common import load_yml_records, predict  # noqa: E402
from train_transformer_polymer_type import (  # noqa: E402
    PolymerTypeDataset,
    PolymerTypeTransformerRegressor,
    forward_batch,
    polymer_type_collate,
)
from src.plotting import compute_metrics  # noqa: E402


def build_model(config: dict, vocab: dict) -> torch.nn.Module:
    return PolymerTypeTransformerRegressor(
        vocab_size=len(vocab), d_model=int(config["d_model"]), num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]), dim_feedforward=int(config["d_model"]) * int(config["ff_mult"]),
        dropout=float(config["dropout"]), max_len=128, readout=config["readout"],
        position_encoding=config["position_encoding"], numeric_mode="side_channel",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    records = load_yml_records(str(args.test_dir), "chiN")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(MODEL_ROOT.glob("seed20*")):
        checkpoint = run_dir / "best_model.pth"
        if not checkpoint.is_file():
            continue
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        vocab = json.loads((run_dir / "vocab.json").read_text(encoding="utf-8"))
        dataset = PolymerTypeDataset(
            records, vocab=vocab, order_variant="canonical", seed=int(config["seed"]),
            max_len=128, numeric_mode="side_channel",
            representation=config.get("representation", "full_topology"),
            canonicalization="strict",
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=polymer_type_collate)
        model = build_model(config, vocab).to(device)
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
        model.eval()
        y_pred, y_true, names = predict(model, loader, device, forward_batch)
        metrics = compute_metrics(y_true, y_pred)
        stem = f"{run_dir.name}_seed43_independent"
        with (args.output_dir / f"{stem}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["base_name", "true", "pred"])
            writer.writeheader()
            writer.writerows({"base_name": name, "true": float(true), "pred": float(pred)}
                             for name, true, pred in zip(names, y_true, y_pred, strict=True))
        row = {
            "run_id": run_dir.name, "seed": int(config["seed"]), "checkpoint": str(checkpoint),
            "test_dir": str(args.test_dir.resolve()), "test_count": len(dataset),
            "mape_percent": float(metrics["MAPE"]), "rmse": float(metrics["RMSE"]),
            "mae": float(metrics["MAE"]), "r2": float(metrics["R²"]),
        }
        (args.output_dir / f"{stem}.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        print(f"{run_dir.name}: MAPE={row['mape_percent']:.6f}%", flush=True)
    if len(rows) != 5:
        raise RuntimeError(f"Expected five checkpoints, evaluated {len(rows)}")
    with (args.output_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
