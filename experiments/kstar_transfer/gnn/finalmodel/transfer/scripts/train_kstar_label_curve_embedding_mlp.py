#!/usr/bin/env python3
"""Train the frozen C2-512 embedding MLP for one k* label budget."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting import compute_metrics  # noqa: E402

TRANSFER_ROOT = Path(__file__).resolve().parent.parent
FULL_EMBEDDING_ROOT = PROJECT_ROOT / "results/cache/kstar_transfer/c2_512_embedding_mlp_probe"
SEEDS = (202601, 202602, 202603, 202604, 202605)


class ProbeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, features):
        return self.net(features)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def source_cache(seed: int) -> Path:
    run_id = SEEDS.index(seed) + 1
    return (
        FULL_EMBEDDING_ROOT
        / f"run{run_id:02d}_seed{seed}"
        / "c2_512_embeddings.npz"
    )


def recover_unscaled(cache: np.lib.npyio.NpzFile):
    mean = cache["feature_mean"]
    std = cache["feature_std"]
    train_x = cache["train_x"] * std + mean
    val_x = cache["val_x"] * std + mean
    test_x = cache["test_x"] * std + mean
    return train_x, cache["train_y_log"], val_x, cache["val_y_log"], test_x, cache["test_y_log"]


def standardize(train_x, val_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std


def train_model(train_x, train_y, val_x, val_y, seed, device):
    model = ProbeMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()),
        batch_size=min(512, len(train_x)),
        shuffle=True,
        generator=generator,
    )
    val_features = torch.from_numpy(val_x).float().to(device)
    val_targets = torch.from_numpy(val_y).float().to(device)
    best = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(1, 501):
        model.train()
        total = 0.0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(features)
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_features), val_targets).item())
        history.append(
            {"epoch": epoch, "train_loss": total / len(train_x), "val_loss": val_loss}
        )
        if val_loss < best:
            best = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= 40:
            break
    if best_state is None:
        raise RuntimeError("No MLP checkpoint was selected")
    model.load_state_dict(best_state)
    return model, history, best


def predict(model, features, device):
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(features).float()), batch_size=512)
    with torch.no_grad():
        for (batch,) in loader:
            outputs.append(model(batch.to(device)).cpu())
    return torch.cat(outputs).numpy().reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--stratify-block-count", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device(args.device)
    cache_path = source_cache(args.seed)
    with np.load(cache_path) as cache:
        train_x, train_y, val_x, val_y, test_x, test_y = recover_unscaled(cache)
    if not 1 <= args.train_size <= len(train_x):
        raise ValueError(f"Invalid train size {args.train_size}; maximum is {len(train_x)}")
    if not 1 <= args.validation_size <= len(val_x):
        raise ValueError(
            f"Invalid validation size {args.validation_size}; maximum is {len(val_x)}"
        )
    # Reset the generator for each canonical pool, matching the graph trainer's
    # paired subset protocol exactly for every method, size, and seed.
    def select_positions(pool_name, pool_length, size):
        if not args.stratify_block_count:
            return np.random.default_rng(args.seed).permutation(pool_length)[:size]
        split = np.load(cache_path.parent / "split_indices.npz")
        source_indices = split[f"{pool_name}_idx"]
        mat_paths = sorted((PROJECT_ROOT / "data/datasets/development_100k").glob("*.mat"))
        if len(source_indices) != pool_length:
            raise RuntimeError(f"Embedding rows do not match {pool_name} split indices")
        labels = np.asarray(
            [int(re.match(r"(\d+)blocks_", path.name).group(1)) for path in mat_paths]
        )[source_indices]
        selected = []
        per_group = {blocks: size // 4 for blocks in (2, 3, 4, 5)}
        remainder_start = SEEDS.index(args.seed) % 4
        for offset in range(size % 4):
            per_group[(2, 3, 4, 5)[(remainder_start + offset) % 4]] += 1
        positions = np.arange(pool_length)
        for blocks in (2, 3, 4, 5):
            group = positions[labels == blocks]
            selected.append(
                np.random.default_rng(args.seed + blocks).permutation(group)[:per_group[blocks]]
            )
        return np.concatenate(selected)

    train_positions = select_positions("train", len(train_x), args.train_size)
    val_positions = select_positions("val", len(val_x), args.validation_size)
    train_x, train_y = train_x[train_positions], train_y[train_positions]
    val_x, val_y = val_x[val_positions], val_y[val_positions]
    train_x, val_x, test_x = standardize(train_x, val_x, test_x)
    np.savez(
        args.output_dir / "subset_positions.npz",
        train_positions=train_positions,
        val_positions=val_positions,
    )

    model, history, best = train_model(
        train_x, train_y, val_x, val_y, args.seed, device
    )
    torch.save(model.state_dict(), args.output_dir / "best_probe_mlp.pth")
    with (args.output_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    prediction = 10 ** predict(model, test_x, device)
    truth = 10 ** test_y.reshape(-1)
    metrics = compute_metrics(truth, prediction)
    ape = np.abs(prediction - truth) / np.clip(np.abs(truth), 1e-12, None) * 100
    np.savez(
        args.output_dir / "seed43_predictions.npz",
        y_true=truth,
        y_pred=prediction,
        abs_error=np.abs(prediction - truth),
        ape_percent=ape,
    )
    result = {
        "experiment": "C2_512_fixed_embedding_MLP_kstar_label_curve",
        "method": "frozen_embedding_mlp",
        "seed": args.seed,
        "train_size": args.train_size,
        "validation_size": args.validation_size,
        "embedding_cache": str(cache_path),
        "subset_protocol": (
            "seed-paired nested, exactly balanced 2/3/4/5-block subsets of the fixed split"
            if args.stratify_block_count
            else "seed-paired nested prefix of the fixed split's embedding rows"
        ),
        "stratified_by_block_count": args.stratify_block_count,
        "best_val_loss_log_space": best,
        "epochs_ran": len(history),
        "test_size": len(test_y),
        "MAPE": float(metrics["MAPE"]),
        "RMSE": float(metrics["RMSE"]),
        "MAE": float(metrics["MAE"]),
        "R2": float(metrics["R虏"]),
        "MaxAE": float(metrics["MaxAE"]),
    }
    text = json.dumps(result, indent=2) + "\n"
    (args.output_dir / "metrics.json").write_text(text, encoding="utf-8")
    (args.output_dir / "run_config.json").write_text(text, encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
