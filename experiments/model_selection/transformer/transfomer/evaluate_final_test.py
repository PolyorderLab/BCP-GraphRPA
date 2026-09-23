#!/usr/bin/env python3
"""Evaluate frozen checkpoints on the 2k final test only after tuning is complete."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/zxy/project/TAPD")
CODE_ROOT = ROOT / "baselinemodels/pub0818/transfomer"
EXPERIMENT_ROOT = Path(os.environ.get("TRANSFORMER_EXPERIMENT_ROOT", str(CODE_ROOT)))
MODEL_DIR = ROOT / "baselinemodels/models"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="final")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save-predictions", action="store_true",
                        help="Write per-sample frozen-test predictions for post-freeze figures.")
    args = parser.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    phase_root = EXPERIMENT_ROOT / args.phase / "runs"
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    records = load_yml_records(args.test_dir, "chiN")
    rows = []
    for run_dir in sorted(path for path in phase_root.glob("*/seed*")):
        if args.phase == "training" and run_dir.parent.name not in {f"{size}k" for size in [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]}:
            continue
        config_path = run_dir / "run_config.json"
        metrics_path = run_dir / "metrics.json"
        if not config_path.exists() or not (run_dir / "best_model.pth").exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        vocab = json.loads((run_dir / "vocab.json").read_text(encoding="utf-8"))
        representation = config.get("representation", "full_topology")
        dataset = PolymerTypeDataset(
            records,
            vocab=vocab,
            order_variant="canonical",
            seed=int(config["seed"]),
            max_len=128,
            numeric_mode="side_channel",
            representation=representation,
            canonicalization="strict",
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=polymer_type_collate)
        model = PolymerTypeTransformerRegressor(
                        vocab_size=len(vocab), d_model=int(config["d_model"]),
                        num_heads=int(config["num_heads"]), num_layers=int(config["num_layers"]),
                        dim_feedforward=int(config["d_model"]) * int(config["ff_mult"]),
                        dropout=float(config["dropout"]), max_len=128,
                        readout=config["readout"], position_encoding=config["position_encoding"],
                        numeric_mode="side_channel",
                    )
        model.load_state_dict(torch.load(run_dir / "best_model.pth", weights_only=True, map_location=device))
        model.to(device).eval()
        y_pred, y_true, names = predict(model, loader, device, forward_batch)
        metrics = compute_metrics(y_true, y_pred)
        result = {
            "run_dir": str(run_dir), "seed": config["seed"], "representation": representation,
            "architecture": config.get("architecture", "transformer"), "test_count": len(records),
            "MAPE": float(metrics["MAPE"]), "RMSE": float(metrics["RMSE"]),
            "MAE": float(metrics["MAE"]), "R2": float(metrics["R²"]), "MaxAE": float(metrics["MaxAE"]),
        }
        (output_root / f"{run_dir.parent.name}__{run_dir.name}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if args.save_predictions:
            prediction_path = output_root / f"{run_dir.parent.name}__{run_dir.name}_predictions.csv"
            with prediction_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["base_name", "true", "pred"])
                writer.writeheader()
                writer.writerows(
                    {"base_name": name, "true": float(true), "pred": float(pred)}
                    for name, true, pred in zip(names, y_true, y_pred)
                )
        rows.append(result)
    if not rows:
        raise RuntimeError(f"No frozen checkpoints found under {phase_root}")
    with (output_root / "per_run.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for key in sorted({(r["representation"], r["architecture"]) for r in rows}):
        subset = [r for r in rows if (r["representation"], r["architecture"]) == key]
        mapes = [r["MAPE"] for r in subset]
        summary.append({"representation": key[0], "architecture": key[1], "n": len(mapes),
                        "MAPE_mean": sum(mapes) / len(mapes),
                        "MAPE_sd": __import__("statistics").stdev(mapes) if len(mapes) > 1 else 0.0})
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
