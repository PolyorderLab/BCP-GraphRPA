#!/usr/bin/env python3
"""Evaluate frozen finalmode-80k checkpoints on the 6--10-block OOD set."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / "checkpoints/transformer"
MODEL_DIR = ROOT / "experiments/model_selection/transformer/transfomer/dependencies_handoff/models"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(ROOT))

from baseline_common import predict  # noqa: E402
from train_transformer_polymer_type import (  # noqa: E402
    PolymerTypeDataset, PolymerTypeTransformerRegressor,
    forward_batch, polymer_type_collate,
)
from src.plotting import compute_metrics  # noqa: E402


def build_model(config: dict, vocab: dict) -> torch.nn.Module:
    return PolymerTypeTransformerRegressor(
        vocab_size=len(vocab), d_model=int(config["d_model"]), num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]), dim_feedforward=int(config["d_model"]) * int(config["ff_mult"]),
        dropout=float(config["dropout"]), max_len=128, readout=config["readout"],
        position_encoding=config["position_encoding"], numeric_mode="side_channel",
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    result = compute_metrics(y_true, y_pred)
    return {"MAPE": float(result["MAPE"]), "RMSE": float(result["RMSE"]),
            "MAE": float(result["MAE"]), "R2": float(result["R²"]),
            "MaxAE": float(result["MaxAE"])}


def block_count(name: str) -> int:
    match = re.match(r"(\d+)blocks_sample_\d+$", name)
    if not match:
        raise ValueError(f"Cannot determine block count from {name!r}")
    return int(match.group(1))


def load_ood_records(test_dir: Path) -> list[dict]:
    """Load only sample YAMLs, excluding generation_summary.yml metadata."""
    records = []
    for path in sorted(test_dir.glob("[6-9]blocks_sample_*.yml")) + sorted(test_dir.glob("10blocks_sample_*.yml")):
        with path.open(encoding="utf-8") as handle:
            system = yaml.safe_load(handle)["system"]
        target = float(system["χN_star"])
        records.append({"base_name": path.stem, "system": system,
                        "target_log": float(np.log10(max(target, 1e-6))), "target_raw": target})
    if len(records) != 2500 or {block_count(record["base_name"]) for record in records} != {6, 7, 8, 9, 10}:
        raise RuntimeError(f"Expected 2,500 OOD samples spanning 6--10 blocks, got {len(records)}")
    return records


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
    records = load_ood_records(args.test_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(MODEL_ROOT.glob("seed20*")):
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        vocab = json.loads((run_dir / "vocab.json").read_text(encoding="utf-8"))
        dataset = PolymerTypeDataset(records, vocab=vocab, order_variant="canonical", seed=int(config["seed"]),
                                     max_len=128, numeric_mode="side_channel",
                                     representation=config.get("representation", "full_topology"), canonicalization="strict")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=polymer_type_collate)
        model = build_model(config, vocab).to(device)
        checkpoint = run_dir / "best_model.pth"
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
        model.eval()
        y_pred, y_true, names = predict(model, loader, device, forward_batch)
        blocks = np.asarray([block_count(name) for name in names], dtype=np.int64)
        absolute = np.abs(y_pred - y_true)
        ape = absolute / np.clip(np.abs(y_true), 1e-12, None) * 100
        stem = f"{run_dir.name}_transfer_6to10blocks"
        with (args.output_dir / f"{stem}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["sample_name", "block_count", "y_true", "y_pred", "abs_error", "ape_percent"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            writer.writerows({"sample_name": name, "block_count": int(count), "y_true": float(true),
                              "y_pred": float(pred), "abs_error": float(error), "ape_percent": float(percent)}
                             for name, count, true, pred, error, percent in zip(names, blocks, y_true, y_pred, absolute, ape, strict=True))
        per_block = [{"block_count": int(count), "num_samples": int((blocks == count).sum()),
                      **metrics(y_true[blocks == count], y_pred[blocks == count])}
                     for count in sorted(set(blocks.tolist()))]
        with (args.output_dir / f"{stem}_per_block_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_block[0])); writer.writeheader(); writer.writerows(per_block)
        overall = metrics(y_true, y_pred)
        row = {"run_id": run_dir.name, "seed": int(config["seed"]), "checkpoint": str(checkpoint),
               "test_dir": str(args.test_dir.resolve()), "test_count": len(dataset), **overall}
        (args.output_dir / f"{stem}.json").write_text(json.dumps({**row, "per_block_metrics": per_block}, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        print(f"{run_dir.name}: MAPE={overall['MAPE']:.6f}%", flush=True)
    with (args.output_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
