#!/usr/bin/env python3
"""Run seed-43 inference using the bundled final C2 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from mainmodel.scripts import train_single_chin_supplementary_confirmation as model_source
from src.data_utils import load_and_process_dataset
from src.ira_graph_common import extract_predictions
from src.plotting import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "results" / "seed43_inference")
    parser.add_argument("--seed", type=int, choices=range(202601, 202606))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    raw_data = load_and_process_dataset(str(args.data_dir), task_type="chiN")
    checkpoint_root = PACKAGE_ROOT / "checkpoints" / "gnn"
    seeds = [args.seed] if args.seed else list(range(202601, 202606))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        run_dir = checkpoint_root / f"seed{seed}"
        config = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        wrapped = model_source.build_wrapped_dataset(raw_data, config)
        loader = DataLoader(wrapped, batch_size=args.batch_size, shuffle=False)
        model = model_source.build_model(config, wrapped, device)
        model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
        pred, true = extract_predictions(model, loader, device)
        metrics = compute_metrics(true, pred)
        np.savez_compressed(args.output_dir / f"seed{seed}_predictions.npz", y_pred=pred, y_true=true)
        row = {"seed": seed, **{key: float(value) for key, value in metrics.items()}}
        rows.append(row)
        print(json.dumps(row))
    (args.output_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
