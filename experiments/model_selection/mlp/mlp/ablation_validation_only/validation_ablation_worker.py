#!/usr/bin/env python3
"""Train one MLP candidate using only the fixed 80k/20k development split.

No final-test argument exists by design.  The 20k validation fold selects both
the checkpoint and the hyperparameter candidate; its original-scale MAPE is
the only model-selection metric recorded by this worker.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true, y_pred = 10.0 ** y_true_log, 10.0 ** y_pred_log
    relative = np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1e-12, None) * 100.0
    return {
        "validation_mape_percent": float(relative.mean()),
        "validation_relative_error_p95_percent": float(np.percentile(relative, 95)),
        "validation_rmse_log10": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "validation_mae_log10": float(mean_absolute_error(y_true_log, y_pred_log)),
        "validation_r2_log10": float(r2_score(y_true_log, y_pred_log)),
        "validation_rmse_raw": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "validation_mae_raw": float(mean_absolute_error(y_true, y_pred)),
        "validation_r2_raw": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--split-indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--validation-predictions-output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--value-label", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-train", type=int, default=None,
                        help="Optional fixed-size subset of the pool's train fold; sampled with random_state=42.")
    args = parser.parse_args()

    started = time.time()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This publication search must run on CUDA.")

    features = [item for item in args.features.split(",") if item]
    frame = pd.read_csv(args.train_csv, usecols=[*features, "log10_chiN_star"])
    split = np.load(args.split_indices)
    train_index, val_index = split["train_index"], split["val_index"]
    if len(train_index) + len(val_index) != len(frame):
        raise ValueError("Split indices do not cover the descriptor matrix.")
    if np.intersect1d(train_index, val_index).size:
        raise ValueError("Training and validation indices overlap.")
    if args.max_train is not None and args.max_train < len(train_index):
        train_index = np.sort(np.random.default_rng(42).choice(train_index, size=args.max_train, replace=False))

    x_all = frame[features].to_numpy(np.float32)
    y_all = frame["log10_chiN_star"].to_numpy(np.float32)
    scaler = StandardScaler().fit(x_all[train_index])
    x_train = torch.from_numpy(scaler.transform(x_all[train_index]).astype(np.float32)).to(device)
    y_train = torch.from_numpy(y_all[train_index, None]).to(device)
    x_val = torch.from_numpy(scaler.transform(x_all[val_index]).astype(np.float32)).to(device)
    y_val = torch.from_numpy(y_all[val_index, None]).to(device)

    model = MLP(len(features), args.hidden_dim, args.depth, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    loss_fn = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss, best_epoch, stagnant = float("inf"), 0, 0
    train_history, val_history = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), device=device)
        total = 0.0
        for start in range(0, len(x_train), args.batch_size):
            batch = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
        train_history.append(total / len(x_train))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).item())
        val_history.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val_loss - 1e-7:
            best_val_loss, best_epoch, stagnant = val_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stagnant += 1
        if stagnant >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(x_val).cpu().numpy().ravel()
    y_val_np = y_all[val_index]
    result = {
        "experiment_id": args.experiment_id,
        "selection_role": "fixed_validation_20k_only",
        "dimension": args.dimension,
        "value_label": args.value_label,
        "features": features,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "train_pool_size": len(frame),
        "fit_size": len(train_index),
        "validation_size": len(val_index),
        "split_random_state": 42,
        "max_train": args.max_train,
        "epochs_ran": len(train_history),
        "best_epoch": best_epoch,
        "best_val_mse_log10": best_val_loss,
        "train_history": train_history,
        "val_history": val_history,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        **metrics(y_val_np, prediction),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.validation_predictions_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.validation_predictions_output, y_true_log=y_val_np, y_pred_log=prediction, validation_index=val_index)
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state, "features": features, "hidden_dim": args.hidden_dim,
        "depth": args.depth, "dropout": args.dropout, "seed": args.seed, "best_epoch": best_epoch,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "validation_index": val_index,
        "split_random_state": 42, "selection_role": "fixed_validation_20k_only",
    }, args.checkpoint_output)
    print(json.dumps({key: result[key] for key in ("experiment_id", "best_epoch", "validation_mape_percent", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
