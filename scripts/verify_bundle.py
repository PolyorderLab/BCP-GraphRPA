#!/usr/bin/env python3
"""Verify checkpoint loading, a mini-batch forward pass, and one short epoch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from mainmodel.scripts import train_single_chin_supplementary_confirmation as model_source
from src.data_utils import load_and_process_dataset
from src.train_utils import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202602, choices=range(202601, 202606))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "verification" / "verification.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_dir = PACKAGE_ROOT / "checkpoints" / "gnn" / f"seed{args.seed}"
    config = json.loads((checkpoint_dir / "metrics.json").read_text(encoding="utf-8"))
    raw = load_and_process_dataset(str(args.data_dir), task_type="chiN")
    count = min(args.samples, len(raw))
    dataset = model_source.build_wrapped_dataset(Subset(raw, range(count)), config)
    loader = DataLoader(dataset, batch_size=min(args.batch_size, count), shuffle=False)

    model = model_source.build_model(config, dataset, device)
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pth", map_location=device, weights_only=True))
    model.eval()
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        output = model(batch)
        output = output[0] if isinstance(output, (tuple, list)) else output

    # A one-epoch update uses a fresh model so that the bundled checkpoint stays untouched.
    short_model = model_source.build_model(config, dataset, device)
    optimizer = torch.optim.Adam(short_model.parameters(), lr=float(config["learning_rate"]))
    scratch_checkpoint = args.output.parent / "short_training_model.pth"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(short_model, optimizer, torch.nn.MSELoss(), device, save_path=str(scratch_checkpoint))
    train_history, validation_history = trainer.fit(loader, loader, epochs=1)
    result = {
        "checkpoint": str(checkpoint_dir / "best_model.pth"),
        "device": str(device),
        "samples": count,
        "batch_output_shape": list(output.shape),
        "one_epoch_train_loss": float(train_history[-1]),
        "one_epoch_validation_loss": float(validation_history[-1]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
